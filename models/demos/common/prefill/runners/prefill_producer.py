#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


import argparse
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotate_chunk_tokens
from models.demos.common.prefill.prefill_chunk import (
    SHUTDOWN_WORD,
    PrefillChunk,
    check_packed_forward_support,
    metadata_size_bytes,
    pack_forward_metadata,
    pad_chunks,
    sentinel_metadata,
)
from models.demos.common.prefill.runners.layer_completion_ring_reader import (
    connect_layer_completion_channel,
    start_layer_completion_drain,
    wait_for_layer_completions,
)
from models.demos.common.prefill.runners.migration import (
    is_per_host_storage,
    migration_table_path,
    rank_scoped_device_map_path,
)
from models.demos.common.prefill.runners.runner_utils import (
    MTP_PAD_TOKEN_ID,
    h2d_row_len,
    load_trace_token_ids,
    num_mtp_tokens,
    resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import mtp_lookahead_positions


def _apply_manifest_env(manifest_path: str) -> dict:
    import yaml

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    def sd_bool(key, val):
        if val is not None:
            os.environ.setdefault(key, "1" if val else "0")

    for key, val in (manifest.get("env") or {}).items():
        sd(key, val)

    model = manifest.get("model") or {}
    sd("PREFILL_MODEL", model.get("variant"))
    sd("PREFILL_NUM_LAYERS", model.get("num_layers"))
    sd("PREFILL_MAX_SEQ_LEN", model.get("max_seq_len"))
    sd("PREFILL_CHUNK_SIZE", model.get("chunk_size"))
    sd("PREFILL_FORWARD_TOKENS_BUDGET", model.get("forward_tokens_budget"))

    transport = manifest.get("transport") or {}
    sd("PREFILL_SP", transport.get("sp"))
    sd("PREFILL_TP", transport.get("tp"))
    sd("PREFILL_H2D_SERVICE_ID", transport.get("h2d_service_id"))
    sd("PREFILL_H2D_CONNECT_TIMEOUT", transport.get("connect_timeout_s"))

    workload = manifest.get("workload") or {}
    sd("PREFILL_NUM_USERS", workload.get("num_users"))
    sd("PREFILL_PRODUCER_CHUNKS", workload.get("chunks"))
    sd("PREFILL_PRODUCER_MAX_REQUESTS", workload.get("max_requests"))
    sd("PREFILL_PRODUCER_DURATION_S", workload.get("duration_s"))
    sd("PREFILL_PRODUCER_INTERLEAVE", workload.get("interleave"))
    sd("PREFILL_PRODUCER_P_GAP", workload.get("p_gap"))
    sd("PREFILL_PRODUCER_P_BURST", workload.get("p_burst"))
    sd("PREFILL_PRODUCER_MID_END_PROB", workload.get("mid_end_prob"))
    sd("PREFILL_PRODUCER_MULTI_TURN_PROB", workload.get("multi_turn_prob"))
    sd("PREFILL_PRODUCER_SEED", workload.get("seed"))
    sd_bool("PREFILL_PRODUCER_CHECK_PCC", workload.get("check_pcc"))
    sd("PREFILL_TRACE_DIR", workload.get("trace_dir"))
    slot_prompts = workload.get("slot_prompts")
    if slot_prompts is not None:
        sd("PREFILL_PRODUCER_SLOT_TRACES", slot_prompts if isinstance(slot_prompts, str) else ",".join(slot_prompts))
    gap_ms = workload.get("gap_ms")
    if gap_ms is not None:
        sd("PREFILL_PRODUCER_GAP_MS", gap_ms if isinstance(gap_ms, str) else ",".join(str(x) for x in gap_ms))

    logger.info(f"[producer] applied manifest {manifest_path}")
    return manifest


def _load_env_config(check_budget: bool = True) -> None:
    global SP_AXIS, TP_AXIS, GLOBAL_MESH_SHAPE, CHUNK_SIZE, MAX_SEQ_LEN, NUM_LAYERS, ADAPTER, NUM_ACK_LAYERS, MTP_LEVELS
    global MTP_DECODE_LEVELS, FORWARD_CHUNKS, CHUNK_METADATA_SIZE_BYTES
    SP_AXIS = int(os.environ.get("PREFILL_SP", 8))
    TP_AXIS = int(os.environ.get("PREFILL_TP", 4))
    GLOBAL_MESH_SHAPE = (SP_AXIS, TP_AXIS)
    CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
    # Tokens per forward, a multiple of the chunk size; must match the runner's.
    forward_tokens_budget = int(os.environ.get("PREFILL_FORWARD_TOKENS_BUDGET", CHUNK_SIZE))
    assert not check_budget or (
        forward_tokens_budget >= CHUNK_SIZE and forward_tokens_budget % CHUNK_SIZE == 0
    ), f"PREFILL_FORWARD_TOKENS_BUDGET={forward_tokens_budget} must be a multiple of PREFILL_CHUNK_SIZE={CHUNK_SIZE}"
    FORWARD_CHUNKS = max(forward_tokens_budget // CHUNK_SIZE, 1)
    CHUNK_METADATA_SIZE_BYTES = metadata_size_bytes(FORWARD_CHUNKS)
    MTP_LEVELS = int(os.environ.get("PREFILL_MTP_LEVELS", 0))
    # Decode's level count: the levels past MTP_LEVELS are table aliases of level 0's rows, acked with it.
    MTP_DECODE_LEVELS = int(os.environ.get("PREFILL_MTP_DECODE_LEVELS", MTP_LEVELS))
    MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 16))
    ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
    # Default from the adapter's own depth (see prefill_runner.py's MODEL_CFG.NUM_LAYERS
    # pattern) rather than a literal -- a hardcoded default silently under-covers any model
    # deeper than that literal when a manifest omits PREFILL_NUM_LAYERS.
    NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", ADAPTER.model_config.NUM_LAYERS))
    NUM_ACK_LAYERS = NUM_LAYERS


# At import the manifest may still set the chunk size; the entry points reload and check after applying it.
_load_env_config(check_budget=False)


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    return struct.pack("<3I", slot_id, actual_start, actual_end)


def _push(service, expect_bytes: int, tokens, mtp_tokens, metadata: bytes) -> None:
    payload = tokens if mtp_tokens is None else np.concatenate([tokens, mtp_tokens], axis=-1)
    assert payload.nbytes == expect_bytes, f"payload {payload.nbytes}B != service-expected {expect_bytes}B"
    service.forward_to_tensor_bytes(payload, metadata=metadata)


def _to_host_array(rows) -> "np.ndarray":
    return rows.to(torch.uint32).contiguous().numpy()


def _pool_slice(pool, start: int, count: int, actual_isl=None):
    ids = list(pool[start : start + count])
    ids = ids + [1] * (count - len(ids))
    if actual_isl is not None:
        ids = [MTP_PAD_TOKEN_ID if start + i >= actual_isl else v for i, v in enumerate(ids)]
    return ids


def _chunk_slice(pool, actual_start: int, actual_isl=None):
    return _pool_slice(pool, actual_start, CHUNK_SIZE, actual_isl)


def _h2d_rows(tokens, actual_start: int = 0):
    # Same host-side reshuffle as the engine's H2D prefill connector: a chunk starting mid-slab (multi-turn
    # resume) is rotated so each SP chip receives the tokens the KV writer places on it (identity when the
    # start is chunk-aligned).
    sp = GLOBAL_MESH_SHAPE[0]
    stride = h2d_row_len(CHUNK_SIZE, sp)
    assert len(tokens) == CHUNK_SIZE, f"expected {CHUNK_SIZE} tokens, got {len(tokens)}"
    tokens = rotate_chunk_tokens(list(tokens), actual_start, sp)
    return _to_host_array(torch.tensor(tokens, dtype=torch.int64).view(sp, 1, stride))


def _h2d_forward_rows(chunks, starts):
    """``[sp, 1, FORWARD_CHUNKS * CHUNK_SIZE / sp]`` rows of one packed forward: chip c gets slice c of every
    chunk's rotated tokens, in chunk order (pack_forward_tokens). One chunk gives _h2d_rows."""
    sp = GLOBAL_MESH_SHAPE[0]
    assert all(len(t) == CHUNK_SIZE for t in chunks), f"every chunk is {CHUNK_SIZE} tokens"
    tokens = pack_forward_tokens(chunks, starts, sp)
    return _to_host_array(torch.tensor(tokens, dtype=torch.int64).view(sp, 1, len(tokens) // sp))


class ChunkPush(NamedTuple):
    slot_id: int
    chunk_idx: int
    actual_start: int
    actual_end: int
    actual_isl: int


def _forward_payload(pushes: list, pool_of) -> tuple:
    """(token rows, MTP rows, metadata) of one forward of ``pushes`` (at most FORWARD_CHUNKS chunks; the rest are
    empty chunks). ``pool_of(slot_id)`` is that slot's token pool."""
    if FORWARD_CHUNKS == 1:
        (p,) = pushes
        pool = pool_of(p.slot_id)
        return (
            _h2d_rows(_chunk_slice(pool, p.actual_start, p.actual_isl), p.actual_start),
            _mtp_rows(pool, p.actual_start, p.actual_isl, actual_end=p.actual_end),
            _pack_metadata(p.slot_id, p.actual_start, p.actual_end),
        )
    chunks = pad_chunks([PrefillChunk(p.slot_id, p.actual_start, p.actual_end) for p in pushes], FORWARD_CHUNKS)
    tokens = [_chunk_slice(pool_of(p.slot_id), p.actual_start, p.actual_isl) for p in pushes]
    tokens += [[1] * CHUNK_SIZE] * (FORWARD_CHUNKS - len(pushes))
    rows = _h2d_forward_rows(tokens, [c.actual_start for c in chunks])
    return rows, None, pack_forward_metadata(chunks, FORWARD_CHUNKS)


def _shutdown_payload() -> tuple:
    """(token rows, MTP rows, metadata) of the SHUTDOWN sentinel: every metadata word is SHUTDOWN_WORD."""
    rows = _h2d_forward_rows([[1] * CHUNK_SIZE] * FORWARD_CHUNKS, [0] * FORWARD_CHUNKS)
    return rows, _mtp_rows([], 0), sentinel_metadata(SHUTDOWN_WORD, FORWARD_CHUNKS)


def _mtp_rows(pool, actual_start: int, actual_isl=None, actual_end=None):
    """``[sp, 1, num_mtp_tokens]`` lookahead ids sent after each chip's chunk ids: the ``MTP_LEVELS`` ids after the
    chip's last position, then pad; a split chip whose second run lies at or past ``actual_end`` takes the next
    chip's first ids."""
    n_mtp = num_mtp_tokens(MTP_LEVELS)
    if not n_mtp:
        return None
    sp = GLOBAL_MESH_SHAPE[0]
    stride = h2d_row_len(CHUNK_SIZE, sp)
    actual_end = actual_start + CHUNK_SIZE if actual_end is None else actual_end
    slots = mtp_lookahead_positions(actual_start, sp, stride, actual_end, MTP_LEVELS)
    align_pad = [MTP_PAD_TOKEN_ID] * (n_mtp - MTP_LEVELS)
    rows = [_pool_slice(pool, chip_slots[0], MTP_LEVELS, actual_isl) + align_pad for chip_slots in slots]
    return _to_host_array(torch.tensor(rows, dtype=torch.int64).unsqueeze(1))


def _require_shared_table_path(world_size: int) -> None:
    if world_size <= 1:
        return
    table_path = os.path.abspath(migration_table_path())
    if is_per_host_storage(table_path):
        logger.error(
            f"[producer] KV chunk table {table_path!r} is on per-host storage; multi-rank "
            f"(world_size={world_size}) validators on other hosts cannot read rank 0's table. Point it at "
            "shared/NFS storage (e.g. /data/...)."
        )
        sys.exit(1)


def _read_kv_chunk_table(timeout_s: int):
    table_path = migration_table_path()
    deadline = time.perf_counter() + timeout_s
    while not os.path.exists(table_path):
        if time.perf_counter() > deadline:
            logger.warning(f"[producer] KV chunk table {table_path} not found after {timeout_s}s; skipping table read.")
            return None
        time.sleep(0.1)

    import_from_protobuf_file = getattr(ttnn.experimental.disaggregation, "import_from_protobuf_file", None)
    if import_from_protobuf_file is None:
        logger.error("[producer] ttnn import_from_protobuf_file missing — rebuild ttnn; skipping table read.")
        return None

    table = import_from_protobuf_file(table_path)
    table_cfg = table.config()
    logger.info(
        f"[producer] read KV chunk table {table_path}: entries={table.total_entries()} "
        f"num_layers={table_cfg.num_layers} num_slots={table_cfg.num_slots} "
        f"max_seq_len={table_cfg.max_sequence_length} chunk_n_tokens={table_cfg.chunk_n_tokens}"
    )
    return table


def _read_device_map(timeout_s: int, rank: int | None = None, num_ranks: int = 1) -> dict:
    import glob as _glob
    import json

    def _parse(raw: str) -> dict:
        try:
            return {tuple(int(x) for x in key.split(":")): int(unique_id) for key, unique_id in json.loads(raw).items()}
        except json.JSONDecodeError:
            parsed = {}
            for line in raw.splitlines():
                if not line.strip():
                    continue
                mesh_id, chip_id, unique_id = line.split()
                parsed[(int(mesh_id), int(chip_id))] = int(unique_id)
            return parsed

    path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
    stem, ext = os.path.splitext(path)

    # A KV read is UMD-local: read_dram_umd only reaches chips visible to THIS process. Under
    # pipeline parallelism every rank exports its own map, and merging them all makes the caller
    # believe it can resolve another galaxy's chips -- such a layer then clears the visibility guard
    # and dies inside the read with "no visible chip with ASIC unique_id".
    def _own():
        # This rank's own map: the only chips this process can actually reach. Resolve the name
        # with the same helper the runner writes it with -- at num_ranks <= 1 that is the
        # unsuffixed base path, and hardcoding the _r<rank> form here made the single-rank SC1
        # leg wait out its timeout for a file nothing ever writes.
        own = rank_scoped_device_map_path(path, rank, num_ranks)
        return [own] if os.path.exists(own) else []

    def _any():
        return ([path] if os.path.exists(path) else []) + sorted(_glob.glob(f"{stem}_r*{ext}"))

    # Waiting on _any() under pipeline parallelism is a race, not a fallback: the ranks write to a
    # shared path at different times, so a rank whose own map is not out yet would see a PEER's and
    # stop waiting immediately, picking up exactly the cross-galaxy map the rank-scoping exists to
    # avoid. Wait for our own, or for nothing.
    _wanted = _own if rank is not None else _any

    deadline = time.perf_counter() + timeout_s
    files = _wanted()
    while not files:
        if time.perf_counter() > deadline:
            if rank is not None and _any():
                logger.warning(
                    f"[producer] rank {rank}'s own device map {stem}_r{rank}{ext} never appeared, though "
                    f"other ranks' maps are present. Not merging them -- they describe chips this "
                    f"process cannot reach. Skipping KV read."
                )
            else:
                logger.warning(
                    f"[producer] device map {path} (or {stem}_r*{ext}) not found after {timeout_s}s; skipping KV read."
                )
            return {}
        time.sleep(0.1)
        files = _wanted()

    device_map = {}
    for f_path in files:
        with open(f_path) as f:
            parsed = _parse(f.read())
        device_map.update(parsed)
        logger.info(f"[producer] read device map {f_path}: {len(parsed)} chips")
    if len(files) > 1:
        logger.info(f"[producer] merged {len(files)} device maps: {len(device_map)} chips total")
    return device_map


def _completion_layers_per_chunk(kv_table) -> int:
    """Layer completions one chunk produces across all ranks.

    NUM_ACK_LAYERS counts only what the model's own layers emit; DFlash acks the drafter's context K/V
    past the verifier's last layer, so the runner emits more. The drafter config's rows continue the model's
    layer axis, so its depth is the config's extent past NUM_LAYERS -- read off the published table, the only
    thing this device-less process and the runner both see.
    """
    base = NUM_ACK_LAYERS + MTP_DECODE_LEVELS
    if kv_table is None:
        return base
    dflash = [name for name in _config_names(kv_table) if name.startswith("dflash_")]
    if not dflash:
        return base
    return base + kv_table.config(kv_table.config_id_of(dflash[0])).num_layers - NUM_LAYERS


def _completion_layer_bound(chunk_layers: int) -> int:
    """Global ack-layer axis per chunk: the model's layers plus the rows acked past them (MTP, DFlash)."""
    return NUM_LAYERS + (chunk_layers - NUM_ACK_LAYERS)


def _decode_bfp8_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    TILE = 32
    n_tiles = head_dim // TILE
    raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(n_tiles, 1088)

    exponents = raw_u8[:, :64].astype(np.int32).reshape(n_tiles, 4, 16)
    mantissas = raw_u8[:, 64:].reshape(n_tiles, 4, 16, 16)
    signs = (mantissas >> 7).astype(np.int32)
    magnitude = (mantissas & 0x7F).astype(np.float32)
    scale = np.exp2((exponents - 133).astype(np.float32))[..., None]
    values = np.where(signs > 0, -(magnitude * scale), magnitude * scale)

    by_face = values.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _decode_bf16_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    TILE = 32
    n_tiles = head_dim // TILE
    u16 = np.frombuffer(raw, dtype="<u2").reshape(n_tiles, 4, 16, 16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    by_face = f32.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _decode_row_major_chunk(raw: bytes, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    if len(raw) % _KV_CHUNK_TOKENS != 0:
        raise ValueError(f"row-major KV chunk has {len(raw)} bytes, not a multiple of {_KV_CHUNK_TOKENS} rows")
    row_size_bytes = len(raw) // _KV_CHUNK_TOKENS
    logical_row_size_bytes = head_dim * element_size
    if row_size_bytes < logical_row_size_bytes:
        raise ValueError(f"row-major KV row has {row_size_bytes} bytes, smaller than {head_dim} x {element_size} bytes")

    rows = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(_KV_CHUNK_TOKENS, row_size_bytes)
    logical = rows[:, :logical_row_size_bytes].contiguous()
    return logical.view(dtype).reshape(_KV_CHUNK_TOKENS, head_dim).float()


_KV_CHUNK_TOKENS = 32
_BFP8_TILE_BYTES = 1088
_SCALED_FP8_LATENT_DIM = 512
_SCALED_FP8_SCALE_BLOCK = 128
_SCALED_FP8_NUM_SCALES = _SCALED_FP8_LATENT_DIM // _SCALED_FP8_SCALE_BLOCK
_SCALED_FP8_ROPE_DIM = 64
_SCALED_FP8_SCALE_OFFSET = _SCALED_FP8_LATENT_DIM
_SCALED_FP8_ROPE_OFFSET = _SCALED_FP8_SCALE_OFFSET + _SCALED_FP8_NUM_SCALES * 4
_SCALED_FP8_ROW_BYTES = _SCALED_FP8_ROPE_OFFSET + _SCALED_FP8_ROPE_DIM * 2


def _decode_scaled_fp8_kv_rows(rows: torch.Tensor, head_dim: int) -> torch.Tensor:
    if head_dim != _SCALED_FP8_LATENT_DIM + _SCALED_FP8_ROPE_DIM:
        raise ValueError(f"packed scaled-FP8 KV requires head_dim 576, got {head_dim}")

    latent = rows[:, :_SCALED_FP8_SCALE_OFFSET].contiguous().view(torch.float8_e4m3fn).float()
    scales = (
        rows[:, _SCALED_FP8_SCALE_OFFSET:_SCALED_FP8_ROPE_OFFSET]
        .contiguous()
        .view(torch.float32)
        .reshape(_KV_CHUNK_TOKENS, _SCALED_FP8_NUM_SCALES)
    )
    rope = (
        rows[:, _SCALED_FP8_ROPE_OFFSET:_SCALED_FP8_ROW_BYTES]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(_KV_CHUNK_TOKENS, _SCALED_FP8_ROPE_DIM)
        .float()
    )
    latent = latent * scales.repeat_interleave(_SCALED_FP8_SCALE_BLOCK, dim=-1)
    return torch.cat((latent, rope), dim=-1)


def _decode_kv_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    if head_dim % 32 == 0 and len(raw) == (head_dim // 32) * _BFP8_TILE_BYTES:
        return _decode_bfp8_chunk(raw, head_dim)

    if len(raw) % _KV_CHUNK_TOKENS != 0:
        raise ValueError(f"unsupported KV chunk size {len(raw)} for head_dim {head_dim}")
    row_size_bytes = len(raw) // _KV_CHUNK_TOKENS
    rows = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(_KV_CHUNK_TOKENS, row_size_bytes)
    if head_dim == 576 and _SCALED_FP8_ROW_BYTES <= row_size_bytes < 2 * head_dim:
        return _decode_scaled_fp8_kv_rows(rows, head_dim)
    fp8_fits = row_size_bytes >= head_dim
    bf16_fits = row_size_bytes >= 2 * head_dim
    if fp8_fits and not bf16_fits:
        return _decode_row_major_chunk(raw, head_dim, torch.float8_e4m3fn)
    if bf16_fits and row_size_bytes == 2 * head_dim:
        return _decode_row_major_chunk(raw, head_dim, torch.bfloat16)
    raise ValueError(f"ambiguous or unsupported row-major KV row size {row_size_bytes} bytes for head_dim {head_dim}")


def _resolve_unique_id(fabric_node_ids, device_map: dict) -> int:
    for node in fabric_node_ids:
        key = (int(node.mesh_id), int(node.chip_id))
        if key in device_map:
            return device_map[key]
    tried = [(int(n.mesh_id), int(n.chip_id)) for n in fabric_node_ids]
    raise KeyError(f"no fabric node {tried} in device map ({len(device_map)} chips; single-rank/one-galaxy only)")


@dataclass
class ProducerConfig:
    num_users: int
    chunks_min: int
    chunks_max: int
    max_requests: int
    duration_s: float
    p_gap: float
    p_burst: float
    gap_ms: tuple
    mid_chunk_end_prob: float
    seed: int
    verify: bool
    pcc_threshold: float
    interleave: str = "random"
    slot_lengths: dict = None
    multi_turn_prob: float = 0.0
    forward_chunks: int = 1


def _config_from_env() -> ProducerConfig:
    max_chunks = MAX_SEQ_LEN // CHUNK_SIZE
    chunk_bounds = [int(x) for x in os.environ.get("PREFILL_PRODUCER_CHUNKS", "11").split(",")]
    chunks_max = min(chunk_bounds[-1], max_chunks)
    chunks_min = min(chunk_bounds[0], chunks_max)
    gap_lo, gap_hi = (float(x) for x in os.environ.get("PREFILL_PRODUCER_GAP_MS", "200,2000").split(","))

    interleave = os.environ.get("PREFILL_PRODUCER_INTERLEAVE", "random")
    if interleave not in ("random", "round_robin"):
        raise ValueError(f"PREFILL_PRODUCER_INTERLEAVE must be 'random' or 'round_robin', got {interleave!r}")
    p_burst = float(os.environ.get("PREFILL_PRODUCER_P_BURST", "0.0"))
    check_packed_forward_support(
        FORWARD_CHUNKS, supports_packed_forward=ADAPTER.supports_packed_forward, mtp_levels=MTP_LEVELS
    )
    if FORWARD_CHUNKS > 1 and p_burst > 0:
        raise ValueError(
            "PREFILL_PRODUCER_P_BURST must be 0 with more than one chunk per forward (packing already bursts)"
        )

    return ProducerConfig(
        num_users=int(os.environ.get("PREFILL_NUM_USERS", "1")),
        chunks_min=chunks_min,
        chunks_max=chunks_max,
        max_requests=int(os.environ.get("PREFILL_PRODUCER_MAX_REQUESTS", "1")),
        duration_s=float(os.environ.get("PREFILL_PRODUCER_DURATION_S", "inf")),
        p_gap=float(os.environ.get("PREFILL_PRODUCER_P_GAP", "0.0")),
        p_burst=p_burst,
        gap_ms=(gap_lo, gap_hi),
        mid_chunk_end_prob=float(os.environ.get("PREFILL_PRODUCER_MID_END_PROB", "0.0")),
        seed=int(os.environ.get("PREFILL_PRODUCER_SEED", "1234")),
        verify=os.environ.get("PREFILL_PRODUCER_CHECK_PCC", "0") == "1",
        pcc_threshold=float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.93")),
        interleave=interleave,
        multi_turn_prob=float(os.environ.get("PREFILL_PRODUCER_MULTI_TURN_PROB", "0.0")),
        forward_chunks=FORWARD_CHUNKS,
    )


class _Slot:
    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.req_id = -1
        self.target_chunks = 0
        self.next_chunk = 0
        self.actual_isl = 0
        self.prefix_len = 0
        self.turn_idx = 0

    @property
    def done(self) -> bool:
        return self.next_chunk >= self.target_chunks


class _SlotFill(NamedTuple):
    real_len: int


def _new_request(
    slot: _Slot, req_id: int, cfg: ProducerConfig, rng: random.Random, *, prefix_len: int = 0, turn_idx: int = 0
) -> None:
    slot.req_id = req_id
    slot.next_chunk = 0
    slot.prefix_len = prefix_len
    slot.turn_idx = turn_idx
    if cfg.slot_lengths is not None and slot.slot_id in cfg.slot_lengths:
        real_len = cfg.slot_lengths[slot.slot_id]
        if prefix_len + real_len > MAX_SEQ_LEN:
            prefix_len = slot.prefix_len = 0
            slot.turn_idx = 0
        slot.target_chunks = (real_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        slot.actual_isl = prefix_len + real_len
        return
    remaining_chunks = (MAX_SEQ_LEN - prefix_len) // CHUNK_SIZE
    chunks_max = min(cfg.chunks_max, remaining_chunks)
    chunks_min = min(cfg.chunks_min, chunks_max)
    slot.target_chunks = rng.randint(chunks_min, chunks_max)
    full_tokens = prefix_len + slot.target_chunks * CHUNK_SIZE
    if cfg.mid_chunk_end_prob > 0 and rng.random() < cfg.mid_chunk_end_prob and slot.target_chunks >= 1:
        slot.actual_isl = full_tokens - rng.randint(1, CHUNK_SIZE - 1)
    else:
        slot.actual_isl = full_tokens


@dataclass
class RunStats:
    resident: dict
    total_pushes: int
    push_ms: list
    completed: int
    wall_s: float
    total_chunks: int = 0


def run_schedule(cfg: ProducerConfig, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    """Push cfg's requests as forwards: ``push_fn(pushes) -> ms`` gets one forward's ChunkPush list, a single chunk
    at forward_chunks == 1. Above that, each forward takes one run of consecutive chunks per slot (round-robin or
    shuffled slot order) until it is full or no slot has chunks left; runs ending in a full chunk come first."""
    rng = rng if rng is not None else random.Random(cfg.seed)
    forward_chunks = cfg.forward_chunks
    assert forward_chunks >= 1, f"forward_chunks must be >= 1, got {forward_chunks}"
    assert forward_chunks == 1 or cfg.p_burst == 0, "p_burst must be 0 with forward_chunks > 1"
    slots = [_Slot(i) for i in range(cfg.num_users)]
    resident: dict = {}

    next_req_id = 0
    for slot in slots:
        _new_request(slot, next_req_id, cfg, rng)
        next_req_id += 1

    push_ms: list = []
    total_pushes = 0
    total_chunks = 0
    completed = 0
    round_robin_cursor = -1
    start = now_fn()

    def chunk_of(slot: _Slot, ahead: int = 0) -> ChunkPush:
        chunk_idx = slot.next_chunk + ahead
        actual_start = slot.prefix_len + chunk_idx * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        return ChunkPush(slot.slot_id, chunk_idx, actual_start, actual_end, slot.actual_isl)

    def advance(slot: _Slot, push: ChunkPush) -> None:
        nonlocal completed, next_req_id
        slot.next_chunk += 1
        resident[slot.slot_id] = _SlotFill(real_len=push.actual_end)
        if slot.done:
            completed += 1
            if next_req_id < cfg.max_requests:
                prefix = (slot.actual_isl // _KV_CHUNK_TOKENS) * _KV_CHUNK_TOKENS
                if (
                    cfg.multi_turn_prob > 0
                    and prefix + CHUNK_SIZE <= MAX_SEQ_LEN
                    and rng.random() < cfg.multi_turn_prob
                ):
                    _new_request(slot, next_req_id, cfg, rng, prefix_len=prefix, turn_idx=slot.turn_idx + 1)
                else:
                    _new_request(slot, next_req_id, cfg, rng)
                next_req_id += 1

    def send(runs: list) -> None:
        nonlocal total_pushes, total_chunks
        pushes = [push for _, run in runs for push in run]
        push_ms.append(push_fn(pushes))
        total_pushes += 1
        total_chunks += len(pushes)
        for slot, run in runs:
            for push in run:
                advance(slot, push)

    def send_chunk(slot: _Slot) -> None:
        send([(slot, [chunk_of(slot)])])

    def packed_runs(active_slots: list) -> list:
        nonlocal round_robin_cursor
        if cfg.interleave == "round_robin":
            order = [slots[(round_robin_cursor + 1 + i) % len(slots)] for i in range(len(slots))]
            order = [s for s in order if not s.done]
        else:
            order = list(active_slots)
            rng.shuffle(order)
        # A forward can finish several requests; past max_requests the slots' initial requests stay unsent.
        order = [s for s in order if s.req_id < cfg.max_requests]
        runs = []
        filled = 0
        for slot in order:
            if filled == forward_chunks:
                break
            take = min(slot.target_chunks - slot.next_chunk, forward_chunks - filled)
            runs.append((slot, [chunk_of(slot, k) for k in range(take)]))
            filled += take
        if cfg.interleave == "round_robin":
            round_robin_cursor = runs[-1][0].slot_id
        runs.sort(key=lambda run: run[1][-1].actual_end < run[1][-1].actual_start + CHUNK_SIZE)
        return runs

    while (now_fn() - start) < cfg.duration_s and completed < cfg.max_requests:
        active_slots = [s for s in slots if not s.done]
        if not active_slots:
            break

        roll = rng.random()
        if roll < cfg.p_gap:
            sleep_fn(rng.uniform(*cfg.gap_ms) / 1000.0)
            continue

        if forward_chunks > 1:
            send(packed_runs(active_slots))
            continue

        if cfg.interleave == "round_robin":
            for _ in range(len(slots)):
                round_robin_cursor = (round_robin_cursor + 1) % len(slots)
                if not slots[round_robin_cursor].done:
                    break
            slot = slots[round_robin_cursor]
        else:
            slot = rng.choice(active_slots)

        if roll < cfg.p_gap + cfg.p_burst:
            for _ in range(rng.randint(2, 3)):
                if slot.done:
                    break
                send_chunk(slot)
        else:
            send_chunk(slot)

    return RunStats(
        resident=resident,
        total_pushes=total_pushes,
        push_ms=push_ms,
        completed=completed,
        wall_s=now_fn() - start,
        total_chunks=total_chunks,
    )


def _read_slot_kv_and_check_pcc(
    table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_ranks: int | None = None
):
    golden_cap = int(os.environ.get("PREFILL_PCC_GOLDEN_LEN", "0"))
    if golden_cap:
        real_len = min(real_len, golden_cap)
    if ADAPTER.name == "gemma4_d_p":
        from models.demos.gemma4_d_p.tt.runners.kv_validation import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir)
    if ADAPTER.name == "minimax_m3":
        return _read_slot_kv_and_check_pcc_m3(table, device_map, slot_id, real_len, trace_dir)
    if ADAPTER.name == "llama_3p1_8b":
        return _read_slot_kv_and_check_pcc_llama(table, device_map, slot_id, real_len, trace_dir)
    if ADAPTER.name == "gpt_oss_d_p":
        return _read_slot_kv_and_check_pcc_gpt_oss(table, device_map, slot_id, real_len, trace_dir, num_ranks)
    return _read_slot_kv_and_check_pcc_mla(table, device_map, slot_id, real_len, trace_dir)


def _config_names(table) -> list:
    return [table.config_name(i) for i in range(table.num_configs())]


def _num_model_configs(table) -> int:
    return sum(1 for name in _config_names(table) if name.isdigit())


def _has_index_config(table) -> bool:
    """Whether config 1 is a DSA indexer key cache. Asked of the adapter, not inferred from the count:
    Kimi-K3 publishes three decimal configs (kvpe + two KDA state configs) and none is an index."""
    return _num_model_configs(table) > 1 and ADAPTER.cache_kind(1) == "index"


def _read_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim, decode, *, start=0):
    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    rows = []
    for pos in range(start, read_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        loc = table.lookup(layer, pos, slot_id, config_id)
        unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
        rows.append(decode(raw, head_dim))
    return torch.cat(rows, dim=0)[: read_len - start]


def _golden_pe_as_device(golden_pe, layout: str):
    if layout == "interleaved":
        return golden_pe
    if layout != "half_split":
        raise ValueError(f"unknown golden pe layout {layout!r} (expected 'half_split' or 'interleaved')")
    pe_dim = golden_pe.shape[-1]
    return torch.stack([golden_pe[:, : pe_dim // 2], golden_pe[:, pe_dim // 2 :]], dim=-1).reshape(-1, pe_dim)


def _kvpe_pcc_vs_golden(golden, device_kv, kv_lora: int, golden_pe_layout: str = "half_split"):
    from tests.ttnn.utils_for_testing import comp_pcc

    _, pcc_nope = comp_pcc(golden[:, :kv_lora], device_kv[:, :kv_lora])
    _, pcc_pe = comp_pcc(_golden_pe_as_device(golden[:, kv_lora:], golden_pe_layout), device_kv[:, kv_lora:])
    return pcc_nope, pcc_pe


def _mtp_golden_source(trunk_trace_dir):
    spec = os.environ.get("PREFILL_MTP_TRACE_DIR", "").strip()
    mtp_dir = resolve_trace_dir(spec) if spec else trunk_trace_dir
    return (
        mtp_dir,
        os.environ.get("PREFILL_MTP_GOLDEN_PE_LAYOUT", "").strip() or "half_split",
        os.environ.get("PREFILL_MTP_GOLDEN_INDEX_ROPE_LAYOUT", "").strip() or "half_split",
    )


def _check_mtp_kv_slots(
    table, device_map: dict, slot_id: int, window, read_end: int, head_dim: int, kv_lora: int, trace_dir
):
    first_pos, skip_rows, cmp_len, golden_offset = window
    span = f"[{first_pos + skip_rows},{first_pos + skip_rows + cmp_len})"
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post, kvpe_golden_present

    declared = table.config(0).num_layers
    assert declared == NUM_LAYERS + MTP_DECODE_LEVELS, (
        f"PREFILL_MTP_DECODE_LEVELS={MTP_DECODE_LEVELS} but config 0 declares {declared} layers, not "
        f"{NUM_LAYERS}+{MTP_DECODE_LEVELS}. The runner sized its KV cache / migration stage without the MTP "
        f"tail, so every slot but 0 addresses the wrong region."
    )
    # Decode levels past the computed ones share level 0's cache, so their rows must be its addresses.
    for k in range(MTP_LEVELS, MTP_DECODE_LEVELS):
        for pos in range(first_pos, read_end, table.config(0).chunk_n_tokens):
            alias, source = table.lookup(NUM_LAYERS + k, pos, slot_id), table.lookup(NUM_LAYERS, pos, slot_id)
            assert (alias.noc_addr, alias.size_bytes, alias.device_group_index) == (
                source.noc_addr,
                source.size_bytes,
                source.device_group_index,
            ), f"MTP decode level {k} (layer {NUM_LAYERS + k}) is not an alias of level 0's row at position {pos}"
    if MTP_DECODE_LEVELS > MTP_LEVELS:
        logger.info(
            f"[producer] slot {slot_id} MTP: decode levels {MTP_LEVELS}..{MTP_DECODE_LEVELS - 1} alias level 0's rows"
        )
    per_level = []
    for k in range(MTP_LEVELS):
        layer = NUM_LAYERS + k
        loc0 = table.lookup(layer, 0, slot_id)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue
        kv = _read_kv_slice(
            table, device_map, 0, layer, slot_id, read_end, head_dim, _decode_kv_chunk, start=first_pos
        )[skip_rows : skip_rows + cmp_len]
        assert torch.isfinite(kv).all(), f"MTP level {k} KV (layer {layer}) has non-finite entries"
        assert kv.abs().max() > 0, (
            f"MTP level {k} KV slot (layer {layer}) is all zeros over {span}: the level never "
            f"wrote it (cache slot never reached, or the level did not run)."
        )
        logger.info(f"[producer] slot {slot_id} MTP level {k} (layer {layer:>2}) KV max|.|={kv.abs().max():.4f}")
        per_level.append((k, kv))
    for i in range(len(per_level)):
        for j in range(i + 1, len(per_level)):
            (a, ka), (b, kb) = per_level[i], per_level[j]
            assert not torch.equal(ka, kb), (
                f"MTP levels {a} and {b} wrote IDENTICAL KV over {span}: they are sharing one "
                f"cache slot (level k must write slot NUM_LAYERS+k)."
            )
    logger.info(
        f"[producer] slot {slot_id} MTP: {len(per_level)}/{MTP_LEVELS} local level slots written and "
        f"pairwise distinct"
    )
    # Decode reads its later levels through their own rows, so read them that way too: they must return
    # level 0's KV bit for bit, which is what the golden scores below.
    level0_kv = dict(per_level).get(0)
    if level0_kv is not None and MTP_DECODE_LEVELS > MTP_LEVELS:
        for k in range(MTP_LEVELS, MTP_DECODE_LEVELS):
            layer = NUM_LAYERS + k
            alias_kv = _read_kv_slice(
                table, device_map, 0, layer, slot_id, read_end, head_dim, _decode_kv_chunk, start=first_pos
            )[skip_rows : skip_rows + cmp_len]
            assert torch.equal(alias_kv, level0_kv), (
                f"MTP decode level {k} (layer {layer}) reads back different KV than level 0's row over {span}, "
                f"though the table gives both the same addresses: something wrote that memory between the reads."
            )
        logger.info(
            f"[producer] slot {slot_id} MTP: decode levels {MTP_LEVELS}..{MTP_DECODE_LEVELS - 1} read back "
            f"through their own rows bit-identical to level 0's KV over {span}"
        )

    if not per_level:
        return None

    golden_dir, pe_layout, _ = _mtp_golden_source(trace_dir)
    missing = [NUM_LAYERS + k for k, _ in per_level if not kvpe_golden_present(golden_dir, NUM_LAYERS + k)]
    if missing:
        logger.info(
            f"[producer] slot {slot_id} MTP: golden KV comparison SKIPPED -- {golden_dir} carries no "
            f"kv_post_transform for layer(s) {missing}. The plumbing checks above are the only MTP gate; "
            f"MTP numerics are covered by tests/mtp_prefill/."
        )
        return None
    logger.info(f"[producer] slot {slot_id} MTP: golden KV from {golden_dir} (pe columns: {pe_layout})")

    from tests.ttnn.utils_for_testing import comp_pcc

    rejected = "half_split" if pe_layout == "interleaved" else "interleaved"
    min_pcc = 1.0
    for k, kv in per_level:
        layer = NUM_LAYERS + k
        golden = _load_golden_kv_post(golden_dir, layer, cmp_len, start=golden_offset)
        pcc_nope, pcc_pe = _kvpe_pcc_vs_golden(golden, kv, kv_lora, pe_layout)
        min_pcc = min(min_pcc, pcc_nope, pcc_pe)
        _, pe_rejected = comp_pcc(_golden_pe_as_device(golden[:, kv_lora:], rejected), kv[:, kv_lora:])
        logger.info(
            f"[producer] slot {slot_id} MTP level {k} (layer {layer:>2}) KV PCC: "
            f"nope={pcc_nope:.5f} pe={pcc_pe:.5f} (pe as {rejected}: {pe_rejected:.5f})"
        )
    logger.info(
        f"[producer] slot {slot_id} MTP KV PCC over {span} across {len(per_level)}/{MTP_LEVELS} "
        f"levels -> {min_pcc:.6f}"
    )
    return min_pcc


def _read_slot_kv_and_check_pcc_gpt_oss(
    table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_ranks: int | None = None
):
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.gpt_oss_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from models.demos.gpt_oss_d_p.tt.runners.acceptance import validate_kv_for_pcc
    from tests.ttnn.utils_for_testing import comp_pcc

    mc = ADAPTER.model_config
    n_kv, head_dim = mc.NUM_KEY_VALUE_HEADS, mc.HEAD_DIM
    rotary_dim = getattr(mc, "ROTARY_DIM", head_dim)
    half = rotary_dim // 2
    perm = list(range(head_dim))
    for m in range(rotary_dim):
        perm[m] = half * (m % 2) + (m // 2)
    perm = torch.tensor(perm, dtype=torch.long)

    read_len = ((real_len + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1) // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * (
        NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    kv_dir = Path(trace_dir) / "kv_cache"
    mins = {"k": 1.0, "v": 1.0}
    checked = 0
    for layer in range(NUM_LAYERS):
        loc0 = table.lookup(layer, 0, slot_id, 0)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue
        checked += 1
        dev_k = torch.stack(
            [
                _read_kv_slice(table, device_map, h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]
        dev_v = torch.stack(
            [
                _read_kv_slice(table, device_map, n_kv + h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as h:
            g_k = h.get_tensor(f"key_cache_layer_{layer}").float()[0, :, :real_len, :][..., perm]
            g_v = h.get_tensor(f"value_cache_layer_{layer}").float()[0, :, :real_len, :]

        validate_kv_for_pcc(g_k, dev_k)
        validate_kv_for_pcc(g_v, dev_v)
        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        logger.info(f"  layer {layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")

    min_pcc = min(mins.values())
    logger.info(
        f"[producer] slot {slot_id} GPT-OSS KV PCC over [0,{real_len}) across {checked}/{NUM_LAYERS} local layers -> "
        f"K={mins['k']:.5f} V={mins['v']:.5f} (min {min_pcc:.6f})"
    )
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")
    # A single rank holds every layer, so a layer missing from its device map is a broken map, not another rank's.
    if num_ranks == 1 and checked != NUM_LAYERS:
        raise RuntimeError(f"slot {slot_id}: only {checked}/{NUM_LAYERS} layers resolved against the device map")
    return mins


def _read_llama_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim):
    rows = []
    expected_bytes = (head_dim // 32) * _BFP8_TILE_BYTES
    for position in range(0, read_len, _KV_CHUNK_TOKENS):
        location = table.lookup(layer, position, slot_id, config_id)
        nodes = table.get_device_group(location.device_group_index).fabric_node_ids
        if len(nodes) != 1 or location.size_bytes != expected_bytes:
            raise ValueError(
                f"invalid Llama GQA page ownership or size: config={config_id}, " f"layer={layer}, position={position}"
            )
        unique_id = _resolve_unique_id(nodes, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, location.noc_addr, location.size_bytes)
        if len(raw) != location.size_bytes:
            raise ValueError(f"incomplete KV page read: expected {location.size_bytes} bytes, got {len(raw)}")
        rows.append(_decode_bfp8_chunk(raw, head_dim))
    return torch.cat(rows, dim=0)[:read_len]


def _read_slot_kv_and_check_pcc_llama(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """Read every Llama layer/head from the published table and compare against its HF trace."""
    from pathlib import Path

    from safetensors import safe_open

    from tests.ttnn.utils_for_testing import comp_pcc

    mc = ADAPTER.model_config
    n_kv, head_dim = mc.NUM_KEY_VALUE_HEADS, mc.HEAD_DIM
    if real_len <= 0 or (NUM_LAYERS, n_kv, head_dim) != (32, 8, 128):
        raise ValueError(
            "Llama GQA verification requires a nonempty prefix, 32 layers, 8 KV heads and head dimension 128"
        )

    expected_names = [f"{kind}_h{head}" for kind in ("k", "v") for head in range(n_kv)]
    if _config_names(table) != expected_names:
        raise ValueError(f"Llama GQA table requires config order {expected_names}")
    expected_bytes = (head_dim // 32) * _BFP8_TILE_BYTES
    for config_id in range(2 * n_kv):
        config = table.config(config_id)
        geometry = (config.num_layers, config.num_slots, config.chunk_n_tokens, config.chunk_size_bytes)
        if geometry != (NUM_LAYERS, 2, _KV_CHUNK_TOKENS, expected_bytes):
            raise ValueError(f"Llama GQA config {config_id} has unsupported layer/slot/page geometry")
        if not 0 <= slot_id < config.num_slots or real_len > config.max_sequence_length:
            raise ValueError(f"Llama GQA request exceeds config {config_id} slot or sequence capacity")

    from models.demos.common.prefill.runners.trace_utils import golden_key_frame

    # HF uses a half split; Meta goldens already have the device's adjacent-pair order.
    key_frame = golden_key_frame(trace_dir)
    half = head_dim // 2
    perm = torch.tensor([half * (index % 2) + index // 2 for index in range(head_dim)], dtype=torch.long)
    read_len = math.ceil(real_len / _KV_CHUNK_TOKENS) * _KV_CHUNK_TOKENS
    kv_dir = Path(trace_dir) / "kv_cache"
    mins = {"k": 1.0, "v": 1.0}
    for layer in range(NUM_LAYERS):
        dev_k = torch.stack(
            [_read_llama_kv_slice(table, device_map, head, layer, slot_id, read_len, head_dim) for head in range(n_kv)],
            dim=0,
        )[:, :real_len]
        dev_v = torch.stack(
            [
                _read_llama_kv_slice(table, device_map, n_kv + head, layer, slot_id, read_len, head_dim)
                for head in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as handle:
            golden = [
                handle.get_slice(f"{kind}_cache_layer_{layer}")[:, :, :real_len, :].float() for kind in ("key", "value")
            ]
        for tensor in golden:
            if (
                tensor.ndim != 4
                or tuple(tensor.shape[:2]) != (1, n_kv)
                or tensor.shape[2] < real_len
                or tensor.shape[3] != head_dim
            ):
                raise ValueError(f"layer {layer}: golden GQA cache must have shape [1,{n_kv},>={real_len},{head_dim}]")
        g_k = golden[0][0, :, :real_len, :]
        if key_frame == "hf":
            g_k = g_k[..., perm]
        g_v = golden[1][0, :, :real_len, :]
        if any(not torch.isfinite(tensor).all() for tensor in (g_k, g_v, dev_k, dev_v)):
            raise ValueError(f"layer {layer}: GQA cache comparison contains nonfinite values")
        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        if not math.isfinite(pcc_k) or not math.isfinite(pcc_v):
            raise ValueError(f"layer {layer}: GQA cache comparison produced nonfinite PCC")
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        logger.info(f"  layer {layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")

    logger.info(
        f"[producer] slot {slot_id} Llama KV PCC over [0,{real_len}) across {NUM_LAYERS}/{NUM_LAYERS} layers -> "
        f"K={mins['k']:.5f} V={mins['v']:.5f} (min {min(mins.values()):.6f})"
    )
    return mins


def _read_slot_kv_and_check_pcc_m3(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import _hf_to_meta_rotary_perm
    from tests.ttnn.utils_for_testing import comp_pcc

    mc = ADAPTER.model_config
    n_kv, head_dim, rotary_dim = mc.NUM_KEY_VALUE_HEADS, mc.HEAD_DIM, mc.ROTARY_DIM
    perm = _hf_to_meta_rotary_perm(head_dim, rotary_dim)
    ik_cfg = 2 * n_kv
    ik_bf16 = table.config(ik_cfg).chunk_size_bytes == (head_dim // 32) * 2048
    ik_decode = _decode_bf16_chunk if ik_bf16 else _decode_bfp8_chunk

    read_len = ((real_len + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1) // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * (
        NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    kv_dir = Path(trace_dir) / "kv_cache"
    mins = {"k": 1.0, "v": 1.0, "index_k": 1.0}
    checked = 0
    ik_checked = 0
    for layer in range(NUM_LAYERS):
        loc0 = table.lookup(layer, 0, slot_id, 0)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue
        checked += 1
        dev_k = torch.stack(
            [
                _read_kv_slice(table, device_map, h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]
        dev_v = torch.stack(
            [
                _read_kv_slice(table, device_map, n_kv + h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{layer}").float()[0, :, :real_len, :][..., perm]
            g_v = h.get_tensor(f"value_cache_layer_{layer}").float()[0, :, :real_len, :]
            has_ik = f"index_k_cache_layer_{layer}" in keys
            g_ik = (
                h.get_tensor(f"index_k_cache_layer_{layer}").float()[0, 0, :real_len, :][..., perm] if has_ik else None
            )

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        line = f"  layer {layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}"
        if has_ik:
            dev_ik = _read_kv_slice(table, device_map, ik_cfg, layer, slot_id, read_len, head_dim, ik_decode)[:real_len]
            pcc_ik = float(comp_pcc(g_ik, dev_ik, 0.0)[1])
            mins["index_k"] = min(mins["index_k"], pcc_ik)
            ik_checked += 1
            line += f" index_k={pcc_ik:.5f}"
        logger.info(line)

    min_pcc = min(mins.values())
    logger.info(
        f"[producer] slot {slot_id} M3 KV PCC over [0,{real_len}) across {checked}/{NUM_LAYERS} local layers -> "
        f"K={mins['k']:.5f} V={mins['v']:.5f} index_k={mins['index_k']:.5f} (min {min_pcc:.6f})"
    )
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")
    if not ik_checked:
        del mins["index_k"]
    return mins


def _full_indexer_layer_indices(num_layers: int):
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused

    hf_config = ADAPTER.load_hf_config()
    if not getattr(hf_config, "indexer_types", None):
        return None
    return [layer for layer in range(num_layers) if not indexer_layer_is_reused(hf_config, layer)]


def _resolve_pcc_window(trace_dir, real_len: int, tokens_per_block: int) -> tuple:
    from models.demos.common.prefill.runners.runner_utils import load_trace_golden_span

    tail_window = int(os.environ.get("PREFILL_PCC_TAIL_WINDOW", "0"))
    win_start = int(os.environ.get("PREFILL_PCC_WINDOW_START", "0"))
    win_end = int(os.environ.get("PREFILL_PCC_WINDOW_END", "0"))
    if (win_end or tail_window) and "PREFILL_PCC_GOLDEN_OFFSET" not in os.environ:
        raise ValueError(
            "a PCC window is set (PREFILL_PCC_WINDOW_END or PREFILL_PCC_TAIL_WINDOW) but "
            "PREFILL_PCC_GOLDEN_OFFSET is not; set it to the golden row the window starts at "
            "(0 is valid and must be passed explicitly)."
        )
    golden_offset = int(os.environ.get("PREFILL_PCC_GOLDEN_OFFSET", "0"))
    if win_end:
        for name, value in (("PREFILL_PCC_WINDOW_START", win_start), ("PREFILL_PCC_WINDOW_END", win_end)):
            if value % tokens_per_block:
                raise ValueError(f"{name}={value} must be a multiple of {tokens_per_block} (the DRAM block)")
        return win_start, 0, min(win_end, real_len) - win_start, golden_offset
    if tail_window:
        cmp_len = min(tail_window, real_len)
        first_pos = ((real_len - cmp_len) // tokens_per_block) * tokens_per_block
        return first_pos, (real_len - cmp_len) - first_pos, cmp_len, golden_offset

    gold_start, gold_end = load_trace_golden_span(trace_dir)
    if not gold_start:
        return 0, 0, min(gold_end, real_len), golden_offset
    if os.environ.get("PREFILL_PCC_GOLDEN_LEN"):
        raise ValueError(
            f"PREFILL_PCC_GOLDEN_LEN caps the compare at a length, but {trace_dir} is a windowed "
            f"golden covering positions [{gold_start},{gold_end}); a length means nothing against it. "
            f"Drop the cap, or state the window with PREFILL_PCC_WINDOW_START/END + "
            f"PREFILL_PCC_GOLDEN_OFFSET."
        )
    if real_len <= gold_start:
        raise ValueError(
            f"{trace_dir} covers positions [{gold_start},{gold_end}) but this slot holds only "
            f"{real_len}; the prompt never reaches the captured window. Push the full prompt the "
            f"trace was captured from, or point PREFILL_TRACE_DIR at a golden that starts at 0."
        )
    first_pos = (gold_start // tokens_per_block) * tokens_per_block
    return first_pos, gold_start - first_pos, min(gold_end, real_len) - gold_start, 0


def _read_slot_kv_and_check_pcc_mla(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
        _load_golden_index_k,
        _load_golden_kv_post,
        index_golden_present,
        kvpe_golden_present,
    )
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs
    from tests.ttnn.utils_for_testing import comp_pcc

    # The pe half needs re-basing only if the model rotates it. The adapter's own declaration wins:
    # `cache_half_pccs` names a second case (a natively interleaved indexer RoPE) where the answer
    # is False with no USE_NOPE flag to derive it from, and the chunked test already reads the
    # attribute, so deriving it a second way here is how the two drift. Absent both, the usual
    # rotated convention.
    _KVPE_INTERLEAVE = getattr(ADAPTER, "kv_pe_interleave", not bool(getattr(ADAPTER.model_config, "USE_NOPE", False)))

    KV_LORA = ADAPTER.model_config.KV_LORA_RANK
    HEAD_DIM = KV_LORA + ADAPTER.model_config.QK_ROPE_HEAD_DIM
    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    first_pos, skip_rows, cmp_len, golden_offset = _resolve_pcc_window(trace_dir, real_len, tokens_per_block)
    read_end = ((first_pos + skip_rows + cmp_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block

    # Which layers own a KV slab according to the MODEL, not according to what happens to be on
    # disk. A hybrid stack legitimately has goldens for only some layers, but a mispointed or partial
    # PREFILL_TRACE_DIR looks exactly the same from a file-presence check, and skipping both leaves a
    # gate that passes on whatever subset survived. None (every dense model) means every layer.
    slab_layers = getattr(ADAPTER, "kv_slot_layer_ids", lambda n: None)(NUM_LAYERS)
    expected_slabs = set(range(NUM_LAYERS)) if slab_layers is None else set(slab_layers)

    min_pcc = 1.0
    checked = 0
    unreferenced = []
    missing_golden = []
    for layer in range(NUM_LAYERS):
        # A hybrid attention stack writes a KV slab on only some layers, so the table publishes rows
        # for only those. Decide from the MODEL, not from what is on disk: a shared or dense-variant
        # trace dir can carry a golden for a slab-less layer, and `table.lookup` on its unpublished
        # row would score whatever sits at the default address and fold that into the gate.
        if layer not in expected_slabs:
            unreferenced.append(layer)
            continue
        if not kvpe_golden_present(trace_dir, layer):
            missing_golden.append(layer)
            continue
        loc0 = table.lookup(layer, 0, slot_id)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue

        decoded_rows = []
        for pos in range(first_pos, read_end, tokens_per_block):
            loc = table.lookup(layer, pos, slot_id)
            unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
            decoded_rows.append(_decode_kv_chunk(raw, HEAD_DIM))
        device_kv = torch.cat(decoded_rows, dim=0)[skip_rows : skip_rows + cmp_len]

        # A single PCC scalar per half per layer cannot separate a static per-column error from one
        # that accumulates per token, which is the question the depth curve actually poses. The
        # tensor is already in host memory here, so dumping it costs no device time. bfloat16 is
        # lossless for a bfp8_b-sourced value (7 mantissa bits into 8) and halves the file.
        dump_dir = os.environ.get("PREFILL_DUMP_KV_DIR")
        if dump_dir:
            try:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(
                    device_kv.to(torch.bfloat16),
                    os.path.join(dump_dir, f"device_kv_layer{layer}_slot{slot_id}.pt"),
                )
            except Exception as exc:  # diagnostics must never fail the gate
                logger.warning(f"[producer] KV dump for layer {layer} failed: {exc}")

        golden = _load_golden_kv_post(trace_dir, layer, cmp_len, start=golden_offset)
        # Re-base the pe half only if the model rotates it. Kimi-K3 is NoPE (`mla_use_nope`): its 64
        # rope dims pass through unrotated, so applying the half-split re-interleave scores the
        # transform instead of the cache -- 0.02 against a nope half of 0.998.
        pcc_nope, pcc_pe = cache_half_pccs(golden, device_kv, KV_LORA, pe_interleave=_KVPE_INTERLEAVE)

        min_pcc = min(min_pcc, pcc_nope, pcc_pe)
        checked += 1
        logger.info(f"[producer] slot {slot_id} layer {layer:>2} KV PCC: nope={pcc_nope:.5f} pe={pcc_pe:.5f}")

    logger.info(
        f"[producer] slot {slot_id} KV PCC over "
        f"[{first_pos + skip_rows},{first_pos + skip_rows + cmp_len}) vs golden "
        f"[{golden_offset},{golden_offset + cmp_len}) "
        f"across {checked}/{len(expected_slabs)} "
        f"slab-owning layers -> {min_pcc:.6f}"
        + (f"; {len(unreferenced)} layers own no KV slab (hybrid stack): {unreferenced}" if unreferenced else "")
    )
    if missing_golden:
        # These layers DO own a slab, so the golden must have covered them. Warning and scoring the
        # rest is exactly the gate quietly narrowing itself: a partial or mispointed trace then
        # reports a passing min PCC over whichever layers happened to be present, and the layers
        # that go missing are the deep ones that carry the minimum. Fail instead.
        raise RuntimeError(
            f"slot {slot_id}: {len(missing_golden)} slab-owning layer(s) have no KV golden under "
            f"{trace_dir} and cannot be scored: {missing_golden}. The trace is partial or "
            f"PREFILL_TRACE_DIR is wrong."
        )
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")

    mtp_min = (
        _check_mtp_kv_slots(
            table,
            device_map,
            slot_id,
            (first_pos, skip_rows, cmp_len, golden_offset),
            read_end,
            HEAD_DIM,
            KV_LORA,
            trace_dir,
        )
        if MTP_LEVELS
        else None
    )

    mins = {"kvpe": min_pcc}
    if mtp_min is not None:
        mins["mtp"] = mtp_min
    if _has_index_config(table):
        index_head_dim = ADAPTER.model_config.INDEX_HEAD_DIM
        index_hadamard = normalized_hadamard_matrix(index_head_dim).float()
        n_index_layers = table.config(1).num_layers
        full_layers = _full_indexer_layer_indices(NUM_LAYERS)
        index_rows = []
        if index_golden_present(trace_dir):
            index_rows = [
                (layer, trace_dir) for layer in (range(n_index_layers) if full_layers is None else full_layers)
            ]
            assert full_layers is None or max(full_layers) < n_index_layers, (
                f"config 1 has {n_index_layers} rows but layers [0,{NUM_LAYERS}) put a full indexer at layer "
                f"{max(full_layers)}. On the layer axis the extent must cover the deepest full-indexer "
                f"layer; a compacted (rank-axis) table reaching here would read another layer's keys."
            )
        else:
            logger.warning(
                f"[producer] slot {slot_id}: table has an index config but {trace_dir} carries no "
                f"indexer-key golden; the trunk index cache is NOT checked."
            )
        n_trunk_rows = len(index_rows)
        mtp_index_layer = None
        mtp_index_layout = "interleaved"
        if MTP_LEVELS and full_layers is not None:
            mtp_dir, _, mtp_index_layout = _mtp_golden_source(trace_dir)
            if index_golden_present(mtp_dir, NUM_LAYERS):
                assert NUM_LAYERS < n_index_layers, (
                    f"config 1 has {n_index_layers} rows, so the MTP indexer slot at layer {NUM_LAYERS} "
                    f"is off its layer axis: the table was built compacted and this lookup would read "
                    f"another layer's keys."
                )
                mtp_index_layer = NUM_LAYERS
                index_rows.append((mtp_index_layer, mtp_dir))
            else:
                logger.warning(
                    f"[producer] slot {slot_id}: {mtp_dir} carries no indexer-key golden for layer "
                    f"{NUM_LAYERS}; the MTP index cache is NOT checked."
                )

        min_index = 1.0
        checked_index = 0
        mtp_index_pcc = None
        for layer, golden_dir in index_rows:
            loc0 = table.lookup(layer, 0, slot_id, 1)
            try:
                _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
            except KeyError:
                continue

            decoded_rows = []
            for pos in range(first_pos, read_end, tokens_per_block):
                loc = table.lookup(layer, pos, slot_id, 1)
                unique_id = _resolve_unique_id(
                    table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                )
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                decoded_rows.append(_decode_kv_chunk(raw, index_head_dim))
            dev_ik = torch.cat(decoded_rows, dim=0)[skip_rows : skip_rows + cmp_len]

            is_mtp_row = layer == mtp_index_layer
            golden_ik = _load_golden_index_k(
                golden_dir,
                layer,
                cmp_len,
                start=golden_offset,
                rope_layout=mtp_index_layout if is_mtp_row else "interleaved",
            )
            dev_ik = (dev_ik.float() @ index_hadamard).to(torch.bfloat16)
            _, pcc_index = comp_pcc(golden_ik, dev_ik)
            if is_mtp_row:
                mtp_index_pcc = pcc_index
                if mtp_index_layout != "interleaved":
                    _, pcc_stored = comp_pcc(
                        _load_golden_index_k(golden_dir, layer, cmp_len, start=golden_offset), dev_ik
                    )
                    logger.info(
                        f"[producer]   mtp_index golden as-stored ({mtp_index_layout}) {pcc_stored:.6f} -> "
                        f"re-based onto the device's rope pairing {pcc_index:.6f}"
                    )
            else:
                min_index = min(min_index, pcc_index)
                checked_index += 1
            logger.info(f"[producer] slot {slot_id} layer {layer:>2} index PCC: {pcc_index:.5f}")

        if checked_index:
            logger.info(
                f"[producer] slot {slot_id} index PCC over "
                f"[{first_pos + skip_rows},{first_pos + skip_rows + cmp_len}) vs golden "
                f"[{golden_offset},{golden_offset + cmp_len}) across "
                f"{checked_index}/{n_trunk_rows} local layers -> {min_index:.6f}"
            )
            mins["index"] = min_index
        if mtp_index_pcc is not None:
            logger.info(f"[producer] slot {slot_id} MTP layer {mtp_index_layer} index PCC -> {mtp_index_pcc:.6f}")
            mins["mtp_index"] = mtp_index_pcc

    return mins


def _write_pcc_verdict(
    rank: int, ok: bool, min_pcc: float, checked: int, threshold: float, per_cache: dict | None = None
) -> None:
    summary_dir = os.environ.get("PREFILL_PCC_SUMMARY_DIR")
    if not summary_dir:
        return
    os.makedirs(summary_dir, exist_ok=True)
    verdict = {
        "rank": rank,
        "ok": bool(ok),
        "min_pcc": min_pcc,
        "per_cache": per_cache or {},
        "slots_checked": checked,
        "threshold": threshold,
    }
    with open(os.path.join(summary_dir, f"rank{rank}.json"), "w") as f:
        json.dump(verdict, f)


def _dflash_caches_are_local(kv_table, device_map: dict, *, slot_id: int) -> bool:
    """Whether this host owns the DFlash drafter caches the table describes.

    The drafter is built on ONE rank (the KV tail), so its caches live on ONE host, but every rank sees
    its configs because the table is shared. ``read_dram_umd`` is local-PCIe only, so a rank elsewhere
    would resolve a chip it cannot read and abort inside umd_dram_reader
    (``TT_FATAL: it != uid_to_chip.end()``) rather than skip. Probing one config is enough: all of them
    describe the same two caches on the same host.

    The KVPE reader has the equivalent behaviour inline (``except KeyError: continue`` per layer); the
    drafter needs it hoisted because its reader checks every (layer, head) as one unit.
    """
    from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import dflash_config_name

    try:
        config_id = kv_table.config_id_of(dflash_config_name("k", 0))
        # Last row, not layer 0: the drafter's configs span the merged table's global layer axis and only
        # its tail is populated. An unpopulated row reads back zeroed, and device_group_index 0 is a real
        # group, so probing layer 0 would resolve some other cache's host and answer True anywhere.
        loc = kv_table.lookup(kv_table.config(config_id).num_layers - 1, 0, slot_id, config_id)
        _resolve_unique_id(kv_table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
    except KeyError:
        return False
    return True


def _verify_resident_slots(
    kv_table, stats: RunStats, threshold: float, slot_traces: dict, rank: int = 0, num_ranks: int = 1
) -> bool:
    device_map = _read_device_map(
        int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")), rank=rank, num_ranks=num_ranks
    )
    if not device_map:
        logger.error("[producer] no device map available; skipping KV read/PCC.")
        _write_pcc_verdict(rank, ok=False, min_pcc=0.0, checked=0, threshold=threshold, per_cache={})
        return False

    dflash_threshold = float(os.environ.get("PREFILL_DFLASH_PCC", "0.88"))
    # MTP-level caches (mtp, mtp_index) get their own floor, defaulting to the trunk one so no other
    # model changes behaviour. An MTP level is a decoder block chained on the trunk output, so its KV
    # inherits the trunk error and adds one block's worth per level -- GLM-5.3 measures ~0.03/level
    # against its own CPU reference. Relaxing the shared threshold instead would weaken the trunk gate.
    mtp_threshold = float(os.environ.get("PREFILL_MTP_PCC", threshold))
    check_dflash = any(name.startswith("dflash_") for name in _config_names(kv_table))
    if check_dflash and not _dflash_caches_are_local(kv_table, device_map, slot_id=min(stats.resident, default=0)):
        logger.info(
            "[producer] drafter caches are not on this host; leaving the drafter PCC to the rank that "
            "owns them (the KV-tail rank)."
        )
        check_dflash = False
    if check_dflash:
        from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import dflash_kv_table_pcc_check

        def read_dflash_slice(config_id, layer, slot_id, read_len, head_dim):
            return _read_kv_slice(
                kv_table, device_map, config_id, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk
            )

    min_pcc_overall = 1.0
    per_cache = {}
    min_dflash_overall = None
    checked = 0
    failures = []
    dflash_failures = []
    for slot_id, res in sorted(stats.resident.items()):
        real_len = res.real_len
        if real_len <= 0:
            continue
        slot_mins = _read_slot_kv_and_check_pcc(
            kv_table, device_map, slot_id, real_len, slot_traces[slot_id], num_ranks=num_ranks
        )
        for cache, value in slot_mins.items():
            per_cache[cache] = value if cache not in per_cache else min(per_cache[cache], value)
        pcc = min(slot_mins.values())
        min_pcc_overall = min(min_pcc_overall, pcc)
        checked += 1
        below = {
            cache: value
            for cache, value in slot_mins.items()
            if value < (mtp_threshold if cache.startswith("mtp") else threshold)
        }
        if below:
            failures.append((slot_id, real_len, min(below.values())))
        if check_dflash:
            dflash_pcc = dflash_kv_table_pcc_check(
                kv_table,
                slot_id,
                real_len,
                read_config_slice=read_dflash_slice,
                golden_dir=os.environ.get("PREFILL_DFLASH_GOLDEN_KV_DIR") or ADAPTER.dflash_golden_default,
                threshold=dflash_threshold,
                rope_convention=ADAPTER.dflash_rope_convention,
            )
            if dflash_pcc is not None:
                min_dflash_overall = dflash_pcc if min_dflash_overall is None else min(min_dflash_overall, dflash_pcc)
                if dflash_pcc < dflash_threshold:
                    dflash_failures.append((slot_id, real_len, dflash_pcc))

    dflash_field = "" if min_dflash_overall is None else f" min_dflash_pcc={min_dflash_overall:.6f}"
    per_cache_field = "".join(f" {cache}_pcc={value:.6f}" for cache, value in per_cache.items())
    print(
        f"[producer] kv_cache_pcc_complete slots_checked={checked} min_pcc={min_pcc_overall:.6f}"
        f"{dflash_field}{per_cache_field}"
    )
    ok = bool(checked) and not failures and not dflash_failures
    _write_pcc_verdict(rank, ok=ok, min_pcc=min_pcc_overall, checked=checked, threshold=threshold, per_cache=per_cache)
    if failures:
        logger.error(
            f"[producer] KV cache PCC below {threshold} (MTP caches: {mtp_threshold}) "
            f"for (slot, real_len, pcc): {failures}"
        )
    if dflash_failures:
        logger.error(f"[producer] drafter KV PCC below {dflash_threshold} for (slot, real_len, pcc): {dflash_failures}")
    if failures or dflash_failures:
        return False
    if not checked:
        logger.error("[producer] verify requested but no resident slots had data to check.")
        return False
    per_cache_note = ", ".join(f"{cache}={value:.6f}" for cache, value in per_cache.items())
    logger.success(
        f"[producer] KV cache PCC PASSED (min {min_pcc_overall:.6f} >= {threshold} across {checked} slots"
        f"{f'; per cache: {per_cache_note}' if per_cache_note else ''})"
    )
    if min_dflash_overall is not None:
        logger.success(
            f"[producer] drafter KV PCC PASSED (min {min_dflash_overall:.6f} >= {dflash_threshold} "
            f"across {checked} slots)"
        )
    return True


def _percentile(sorted_values: list, p: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def _load_token_pool(trace_dir, num_tokens: int) -> list:
    pool = load_trace_token_ids(trace_dir, num_tokens)
    if not pool:
        raise ValueError(f"trace {trace_dir} carries no token_ids; cannot build a token pool")
    if len(pool) < num_tokens:
        pool = pool * -(-num_tokens // len(pool))
    return pool[:num_tokens]


def _resolve_slot_prompts(cfg: ProducerConfig):
    default = os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)
    spec = os.environ.get("PREFILL_PRODUCER_SLOT_TRACES", "").strip()
    if spec and cfg.multi_turn_prob > 0:
        raise ValueError(
            "PREFILL_PRODUCER_SLOT_TRACES is incompatible with multi-turn "
            "(PREFILL_PRODUCER_MULTI_TURN_PROB>0): each slot's per-trace pool is loaded only one turn deep "
            "and its golden defines a single turn, so a resumed turn would read past its pool and mismatch "
            "the golden. Use one shared trace (unset SLOT_TRACES) for multi-turn runs."
        )
    max_chunks = MAX_SEQ_LEN // CHUNK_SIZE

    if not spec:
        trace = resolve_trace_dir(default)
        slot_traces = {s: trace for s in range(cfg.num_users)}
        pool_tokens = MAX_SEQ_LEN if cfg.multi_turn_prob > 0 else cfg.chunks_max * CHUNK_SIZE
        return slot_traces, None, {trace: _load_token_pool(trace, pool_tokens)}

    entries = [e.strip() for e in spec.split(",") if e.strip()]
    resolved = [resolve_trace_dir(e) for e in entries]
    if len(entries) not in (1, cfg.num_users):
        logger.warning(
            f"[producer] {len(entries)} slot prompt(s) for {cfg.num_users} user(s); assigning by slot % {len(entries)}"
        )
    slot_traces = {s: resolved[s % len(resolved)] for s in range(cfg.num_users)}

    len_by_trace = {}
    pools_by_trace = {}
    for trace in set(slot_traces.values()):
        real_len = len(load_trace_token_ids(trace))
        chunks = (real_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        if chunks > max_chunks:
            logger.warning(
                f"[producer] prompt {trace} is {real_len} tok ({chunks} chunks) > per-user cache "
                f"{max_chunks} chunks (MAX_SEQ_LEN={MAX_SEQ_LEN}); clamping to {max_chunks} chunks."
            )
            chunks = max_chunks
            real_len = min(real_len, max_chunks * CHUNK_SIZE)
        len_by_trace[trace] = real_len
        pools_by_trace[trace] = _load_token_pool(trace, chunks * CHUNK_SIZE)

    slot_lengths = {s: len_by_trace[t] for s, t in slot_traces.items()}
    logger.info(
        "[producer] per-slot prompts: "
        + ", ".join(f"slot {s}<-{slot_traces[s].name} ({slot_lengths[s]} tok)" for s in sorted(slot_traces))
    )
    return slot_traces, slot_lengths, pools_by_trace


def _mr_config():
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) <= 1:
        return (0, 1)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    size = int(ttnn.distributed_context_get_size())
    return (rank, size)


def _mr_bcast_resident(rank: int, resident: dict) -> dict:
    items = sorted(resident.items()) if rank == 0 else []
    n = ttnn.distributed_context_allgather_int(len(items) if rank == 0 else 0)[0]
    out: dict = {}
    for k in range(n):
        slot_id, real_len = (items[k][0], items[k][1].real_len) if rank == 0 else (0, 0)
        slot_id = ttnn.distributed_context_allgather_int(slot_id)[0]
        real_len = ttnn.distributed_context_allgather_int(real_len)[0]
        out[slot_id] = _SlotFill(real_len=real_len)
    return out


def _mr_allgather_verdict(ok: bool) -> list:
    return [bool(v) for v in ttnn.distributed_context_allgather_int(1 if ok else 0)]


def _run_validator(rank: int, world_size: int) -> None:
    cfg = _config_from_env()
    if not cfg.verify:
        logger.error("[producer] multi-rank requires PREFILL_PRODUCER_CHECK_PCC=1 (validators only verify).")
        sys.exit(1)
    _require_shared_table_path(world_size)
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(f"[producer] validator rank={rank}/{world_size}: read-back only (no H2D feed)")

    resident = _mr_bcast_resident(rank, {})
    logger.info(f"[producer] validator rank={rank}: GO received, {len(resident)} resident slots")

    try:
        kv_table = _read_kv_chunk_table(timeout_s)
    except Exception as e:
        logger.error(f"[producer] validator: KV table read raised: {type(e).__name__}: {e}")
        kv_table = None

    stats = RunStats(resident=resident, total_pushes=0, push_ms=[], completed=0, wall_s=0.0)
    ok = True
    if kv_table is None:
        logger.error("[producer] validator: no KV table available; cannot validate.")
        ok = False
    else:
        try:
            slot_traces, _slot_lengths, _pools = _resolve_slot_prompts(cfg)
            ok = _verify_resident_slots(
                kv_table, stats, cfg.pcc_threshold, slot_traces, rank=rank, num_ranks=world_size
            )
        except Exception as e:
            logger.error(f"[producer] validator KV read/PCC failed: {type(e).__name__}: {e}")
            ok = False

    _mr_allgather_verdict(ok)
    logger.info(f"[producer] validator rank={rank}: DONE ok={ok}")
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prefill_producer",
        description="H2D producer for the prefill runner. Config comes from a YAML manifest "
        "(--manifest / PREFILL_PRODUCER_MANIFEST) mapped to PREFILL_* env vars, with any explicitly "
        "exported PREFILL_* env var overriding the manifest.",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        default=os.environ.get("PREFILL_PRODUCER_MANIFEST"),
        help="Path to the producer YAML manifest (applied at startup; exported env vars override it).",
    )
    args = parser.parse_args()

    if args.manifest:
        _apply_manifest_env(args.manifest)
    _load_env_config()

    mr_rank, world_size = _mr_config()
    if mr_rank != 0:
        _run_validator(mr_rank, world_size)
        return

    cfg = _config_from_env()
    if world_size > 1 and not cfg.verify:
        logger.error("[producer] multi-rank requires PREFILL_PRODUCER_CHECK_PCC=1 (all ranks verify).")
        sys.exit(1)
    _require_shared_table_path(world_size)
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(
        f"[producer] service_id={service_id!r} users={cfg.num_users} chunks=[{cfg.chunks_min},{cfg.chunks_max}] "
        f"max_requests={cfg.max_requests} duration={cfg.duration_s}s p_gap={cfg.p_gap} p_burst={cfg.p_burst} "
        f"mid_end={cfg.mid_chunk_end_prob} interleave={cfg.interleave} verify={cfg.verify} seed={cfg.seed}"
    )

    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    payload_bytes = service.payload_size_bytes()
    logger.info(f"[producer] attached; payload={payload_bytes}B")

    kv_table = _read_kv_chunk_table(timeout_s) if cfg.verify else None

    # Without verification the completions are left to whatever consumer owns the channel.
    completion_channel = connect_layer_completion_channel(timeout_s) if cfg.verify else None
    if cfg.verify and completion_channel is None:
        logger.error(
            "[producer] CHECK_PCC=1 but LayerAck channel missing — UMD read would race the runner's "
            "prefill (H2D push return ≠ layers done). The master rank always owns this channel, so "
            "either the runner is not up yet or it died before wiring acks."
        )
        sys.exit(1)
    if not cfg.verify:
        logger.info(
            "[producer] CHECK_PCC off — skipping the KV table read and not consuming the completion "
            "channel (pure token feeder; the real consumer owns it). Some consumer must be attached, or "
            "the runner waits on its full completion ring."
        )

    chunk_layers = _completion_layers_per_chunk(kv_table)
    layer_bound = _completion_layer_bound(chunk_layers)
    slot_traces, slot_lengths, pools_by_trace = _resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths

    def push_forward(pushes: list) -> float:
        for p in pushes:
            logger.info(
                f"[producer] push slot={p.slot_id} cidx={p.chunk_idx} start={p.actual_start} end={p.actual_end}"
            )
        push_start = time.perf_counter()
        _push(service, payload_bytes, *_forward_payload(pushes, lambda slot_id: pools_by_trace[slot_traces[slot_id]]))
        return (time.perf_counter() - push_start) * 1000.0

    # Drains from the first push to the last barrier: the completion rings are bounded, so a drain that waited
    # for the barrier would leave the runner blocked on a full ring and the barrier waiting on the runner.
    drain = start_layer_completion_drain(completion_channel, num_layers=layer_bound, chunk_layers=chunk_layers)
    try:
        warmup_chunks = int(os.environ.get("PREFILL_PRODUCER_WARMUP_CHUNKS", "0"))
        if warmup_chunks > 0:
            logger.info(f"[producer] warmup: {warmup_chunks} throwaway chunk(s) on slot 0 (not timed, not verified)")
            warmup = [
                ChunkPush(0, cidx, cidx * CHUNK_SIZE, (cidx + 1) * CHUNK_SIZE, warmup_chunks * CHUNK_SIZE)
                for cidx in range(warmup_chunks)
            ]
            for i in range(0, warmup_chunks, FORWARD_CHUNKS):
                push_forward(warmup[i : i + FORWARD_CHUNKS])
            service.barrier()
            wait_for_layer_completions(drain, chunk_layers * warmup_chunks)
            logger.info("[producer] warmup complete; starting the measured request")

        stats = run_schedule(cfg, push_fn=push_forward)
        service.barrier()
        # One completion per layer per chunk; chunk_layers is what the runner emits (KV-writing layers plus
        # MTP/DFlash extras), not NUM_LAYERS. The count is cumulative, so it includes the warm-up.
        expected_completions = chunk_layers * (max(warmup_chunks, 0) + stats.total_chunks)
        received_completions = wait_for_layer_completions(drain, expected_completions)
    finally:
        if drain is not None:
            drain.stop()

    sorted_ms = sorted(stats.push_ms)
    total_tokens = stats.total_chunks * CHUNK_SIZE
    packing = ""
    if FORWARD_CHUNKS > 1:
        fill = stats.total_chunks / (stats.total_pushes * FORWARD_CHUNKS) if stats.total_pushes else 0.0
        packing = f"chunks={stats.total_chunks} forward_chunks={FORWARD_CHUNKS} fill={fill:.2f} "
    logger.info(
        f"[producer] DONE wall={stats.wall_s:.1f}s pushes={stats.total_pushes} requests={stats.completed} "
        f"{packing}tokens={total_tokens} throughput={total_tokens / stats.wall_s if stats.wall_s else 0:.0f} tok/s "
        f"push_ms p50={_percentile(sorted_ms, 0.5):.1f} p90={_percentile(sorted_ms, 0.9):.1f} "
        f"p99={_percentile(sorted_ms, 0.99):.1f}"
    )

    if world_size > 1:
        _mr_bcast_resident(mr_rank, stats.resident)

    verify_ok = True
    if cfg.verify and kv_table is not None:
        try:
            verify_ok = _verify_resident_slots(
                kv_table, stats, cfg.pcc_threshold, slot_traces, rank=mr_rank, num_ranks=world_size
            )
        except Exception as e:
            logger.error(f"[producer] KV read/PCC failed: {type(e).__name__}: {e}")
            verify_ok = False
    elif cfg.verify:
        logger.error("[producer] PREFILL_PRODUCER_CHECK_PCC=1 but no KV chunk table available; skipping PCC.")
        verify_ok = False
    if cfg.verify and ADAPTER.verify_layer_completions and received_completions != expected_completions:
        logger.error(f"[producer] layer completions: received {received_completions}/{expected_completions}")
        verify_ok = False

    if world_size > 1:
        verdicts = _mr_allgather_verdict(verify_ok)
        for r, v in enumerate(verdicts):
            logger.info(f"[producer] rank={r}: ok={v}")
        verify_ok = all(verdicts)

    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        rows, mtp_rows, sentinel = _shutdown_payload()
        assert len(sentinel) == CHUNK_METADATA_SIZE_BYTES
        logger.info("[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        _push(service, payload_bytes, rows, mtp_rows, sentinel)
        service.barrier()
        logger.info("[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.")
    else:
        logger.info("[producer] exiting (the runner keeps its sync-op loop running).")

    if cfg.verify and not verify_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
