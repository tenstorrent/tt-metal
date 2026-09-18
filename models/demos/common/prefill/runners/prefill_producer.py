#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


import argparse
import json
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
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir


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


METADATA_SIZE_BYTES = 12


def _load_env_config() -> None:
    global SP_AXIS, TP_AXIS, GLOBAL_MESH_SHAPE, CHUNK_SIZE, MAX_SEQ_LEN, NUM_LAYERS, ADAPTER
    SP_AXIS = int(os.environ.get("PREFILL_SP", 8))
    TP_AXIS = int(os.environ.get("PREFILL_TP", 4))
    GLOBAL_MESH_SHAPE = (SP_AXIS, TP_AXIS)
    CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
    MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 11))
    NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
    ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))


_load_env_config()


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    return struct.pack("<III", slot_id, actual_start, actual_end)


def _chunk_to_host_array(chunk_token_ids):
    sp = GLOBAL_MESH_SHAPE[0]
    chunk_local = CHUNK_SIZE // sp
    return (
        torch.tensor(chunk_token_ids, dtype=torch.int64)
        .reshape(sp, 1, chunk_local)
        .to(torch.uint32)
        .contiguous()
        .numpy()
    )


_PER_HOST_FS_PREFIXES = ("/tmp", "/dev/shm", "/run", "/var/tmp")


def _require_shared_table_path(world_size: int) -> None:
    if world_size <= 1:
        return
    table_path = os.path.abspath(os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"))
    if any(table_path == p or table_path.startswith(p + "/") for p in _PER_HOST_FS_PREFIXES):
        logger.error(
            f"[producer] PREFILL_MIGRATION_TABLE_PATH={table_path!r} is on per-host storage; multi-rank "
            f"(world_size={world_size}) validators on other hosts cannot read rank 0's table. Point it at "
            "shared/NFS storage (e.g. /data/...)."
        )
        sys.exit(1)


def _read_kv_chunk_table(timeout_s: int):
    table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
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


def _read_device_map(timeout_s: int) -> dict:
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

    def _matches():
        return ([path] if os.path.exists(path) else []) + sorted(_glob.glob(f"{stem}_r*{ext}"))

    deadline = time.perf_counter() + timeout_s
    files = _matches()
    while not files:
        if time.perf_counter() > deadline:
            logger.warning(
                f"[producer] device map {path} (or {stem}_r*{ext}) not found after {timeout_s}s; skipping KV read."
            )
            return {}
        time.sleep(0.1)
        files = _matches()

    device_map = {}
    for f_path in files:
        with open(f_path) as f:
            parsed = _parse(f.read())
        device_map.update(parsed)
        logger.info(f"[producer] read device map {f_path}: {len(parsed)} chips")
    if len(files) > 1:
        logger.info(f"[producer] merged {len(files)} device maps: {len(device_map)} chips total")
    return device_map


def _connect_layer_ack_channel(timeout_s: int):
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = f"/tt_prefill_layer_acks_{service_id}"
    try:
        channel = ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[producer] could not connect LayerAck channel {shm_name}: {e}; skipping ack wait.")
        return None
    logger.info(f"[producer] connected LayerAck channel {shm_name}")
    return channel


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
    if ack_channel is None:
        return 0
    drained = 0
    last_logged = -1
    start = time.perf_counter()
    while drained < expected:
        drained += ack_channel.try_consume_all()
        if drained != last_logged:
            logger.info(f"[producer] layer acks {drained}/{expected}")
            last_logged = drained
        if drained >= expected:
            break
        if time.perf_counter() - start > timeout_s:
            logger.warning(f"[producer] timed out at {drained}/{expected} acks after {timeout_s}s")
            break
        time.sleep(0.01)
    logger.info(f"[producer] drained {drained}/{expected} layer acks in {(time.perf_counter() - start):.2f}s")
    return drained


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


def _config_from_env() -> ProducerConfig:
    max_chunks = MAX_SEQ_LEN // CHUNK_SIZE
    chunk_bounds = [int(x) for x in os.environ.get("PREFILL_PRODUCER_CHUNKS", "11").split(",")]
    chunks_max = min(chunk_bounds[-1], max_chunks)
    chunks_min = min(chunk_bounds[0], chunks_max)
    gap_lo, gap_hi = (float(x) for x in os.environ.get("PREFILL_PRODUCER_GAP_MS", "200,2000").split(","))

    interleave = os.environ.get("PREFILL_PRODUCER_INTERLEAVE", "random")
    if interleave not in ("random", "round_robin"):
        raise ValueError(f"PREFILL_PRODUCER_INTERLEAVE must be 'random' or 'round_robin', got {interleave!r}")

    return ProducerConfig(
        num_users=int(os.environ.get("PREFILL_NUM_USERS", "1")),
        chunks_min=chunks_min,
        chunks_max=chunks_max,
        max_requests=int(os.environ.get("PREFILL_PRODUCER_MAX_REQUESTS", "1")),
        duration_s=float(os.environ.get("PREFILL_PRODUCER_DURATION_S", "inf")),
        p_gap=float(os.environ.get("PREFILL_PRODUCER_P_GAP", "0.0")),
        p_burst=float(os.environ.get("PREFILL_PRODUCER_P_BURST", "0.0")),
        gap_ms=(gap_lo, gap_hi),
        mid_chunk_end_prob=float(os.environ.get("PREFILL_PRODUCER_MID_END_PROB", "0.0")),
        seed=int(os.environ.get("PREFILL_PRODUCER_SEED", "1234")),
        verify=os.environ.get("PREFILL_PRODUCER_CHECK_PCC", "0") == "1",
        pcc_threshold=float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.93")),
        interleave=interleave,
        multi_turn_prob=float(os.environ.get("PREFILL_PRODUCER_MULTI_TURN_PROB", "0.0")),
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


def run_schedule(cfg: ProducerConfig, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    rng = rng if rng is not None else random.Random(cfg.seed)
    slots = [_Slot(i) for i in range(cfg.num_users)]
    resident: dict = {}

    next_req_id = 0
    for slot in slots:
        _new_request(slot, next_req_id, cfg, rng)
        next_req_id += 1

    push_ms: list = []
    total_pushes = 0
    completed = 0
    round_robin_cursor = -1
    start = now_fn()

    def send_chunk(slot: _Slot) -> None:
        nonlocal total_pushes, completed, next_req_id
        chunk_idx = slot.next_chunk
        actual_start = slot.prefix_len + chunk_idx * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        push_ms.append(push_fn(slot.slot_id, chunk_idx, actual_start, actual_end))
        total_pushes += 1
        slot.next_chunk += 1
        resident[slot.slot_id] = _SlotFill(real_len=actual_end)
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

    while (now_fn() - start) < cfg.duration_s and completed < cfg.max_requests:
        active_slots = [s for s in slots if not s.done]
        if not active_slots:
            break

        roll = rng.random()
        if roll < cfg.p_gap:
            sleep_fn(rng.uniform(*cfg.gap_ms) / 1000.0)
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
        resident=resident, total_pushes=total_pushes, push_ms=push_ms, completed=completed, wall_s=now_fn() - start
    )


def _read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    golden_cap = int(os.environ.get("PREFILL_PCC_GOLDEN_LEN", "0"))
    if golden_cap:
        real_len = min(real_len, golden_cap)
    if ADAPTER.name == "minimax_m3":
        return _read_slot_kv_and_check_pcc_m3(table, device_map, slot_id, real_len, trace_dir)
    if ADAPTER.name == "gpt_oss_d_p":
        return _read_slot_kv_and_check_pcc_gpt_oss(table, device_map, slot_id, real_len, trace_dir)
    return _read_slot_kv_and_check_pcc_mla(table, device_map, slot_id, real_len, trace_dir)


def _config_names(table) -> list:
    return [table.config_name(i) for i in range(table.num_configs())]


def _num_model_configs(table) -> int:
    return sum(1 for name in _config_names(table) if name.isdigit())


def _read_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim, decode):
    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    rows = []
    for pos in range(0, read_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        loc = table.lookup(layer, pos, slot_id, config_id)
        unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
        rows.append(decode(raw, head_dim))
    return torch.cat(rows, dim=0)[:read_len]


def _read_slot_kv_and_check_pcc_gpt_oss(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.gpt_oss_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
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


def _read_slot_kv_and_check_pcc_mla(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
        _load_golden_index_k,
        _load_golden_kv_post,
        index_golden_present,
    )
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    KV_LORA = ADAPTER.model_config.KV_LORA_RANK
    HEAD_DIM = KV_LORA + ADAPTER.model_config.QK_ROPE_HEAD_DIM
    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    read_len = ((real_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block

    min_pcc = 1.0
    checked = 0
    for layer in range(NUM_LAYERS):
        loc0 = table.lookup(layer, 0, slot_id)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue

        decoded_rows = []
        for pos in range(0, read_len, tokens_per_block):
            loc = table.lookup(layer, pos, slot_id)
            unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
            decoded_rows.append(_decode_kv_chunk(raw, HEAD_DIM))
        device_kv = torch.cat(decoded_rows, dim=0)[:real_len]

        golden = _load_golden_kv_post(trace_dir, layer, real_len)
        _, pcc_nope = comp_pcc(golden[:, :KV_LORA], device_kv[:, :KV_LORA])

        golden_pe = golden[:, KV_LORA:]
        pe_dim = golden_pe.shape[-1]
        golden_pe = torch.stack([golden_pe[:, : pe_dim // 2], golden_pe[:, pe_dim // 2 :]], dim=-1).reshape(-1, pe_dim)
        _, pcc_pe = comp_pcc(golden_pe, device_kv[:, KV_LORA:])

        min_pcc = min(min_pcc, pcc_nope, pcc_pe)
        checked += 1
        logger.info(f"[producer] slot {slot_id} layer {layer:>2} KV PCC: nope={pcc_nope:.5f} pe={pcc_pe:.5f}")

    logger.info(
        f"[producer] slot {slot_id} KV PCC over [0,{real_len}) across {checked}/{NUM_LAYERS} local layers -> "
        f"{min_pcc:.6f}"
    )
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")

    mins = {"kvpe": min_pcc}
    if _num_model_configs(table) > 1:
        if not index_golden_present(trace_dir):
            logger.warning(
                f"[producer] slot {slot_id}: table has an index config but {trace_dir} carries no "
                f"indexer-key golden; validating the KVPE cache only (device index cache NOT checked)."
            )
            return mins

        index_head_dim = ADAPTER.model_config.INDEX_HEAD_DIM
        index_hadamard = normalized_hadamard_matrix(index_head_dim).float()
        n_index_layers = table.config(1).num_layers
        full_layers = _full_indexer_layer_indices(NUM_LAYERS)
        index_rows = list(range(n_index_layers)) if full_layers is None else full_layers
        assert full_layers is None or max(full_layers) < n_index_layers, (
            f"config 1 has {n_index_layers} rows but layers [0,{NUM_LAYERS}) put a full indexer at layer "
            f"{max(full_layers)}. On the layer axis the extent must cover the deepest full-indexer "
            f"layer; a compacted (rank-axis) table reaching here would read another layer's keys."
        )
        min_index = 1.0
        checked_index = 0
        for layer in index_rows:
            loc0 = table.lookup(layer, 0, slot_id, 1)
            try:
                _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
            except KeyError:
                continue

            decoded_rows = []
            for pos in range(0, read_len, tokens_per_block):
                loc = table.lookup(layer, pos, slot_id, 1)
                unique_id = _resolve_unique_id(
                    table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                )
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                decoded_rows.append(_decode_kv_chunk(raw, index_head_dim))
            dev_ik = torch.cat(decoded_rows, dim=0)[:real_len]

            golden_ik = _load_golden_index_k(trace_dir, layer, real_len)
            dev_ik = (dev_ik.float() @ index_hadamard).to(torch.bfloat16)
            _, pcc_index = comp_pcc(golden_ik, dev_ik)
            min_index = min(min_index, pcc_index)
            checked_index += 1
            logger.info(f"[producer] slot {slot_id} layer {layer:>2} index PCC: {pcc_index:.5f}")

        logger.info(
            f"[producer] slot {slot_id} index PCC over [0,{real_len}) across "
            f"{checked_index}/{len(index_rows)} local layers -> {min_index:.6f}"
        )
        mins["index"] = min_index

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


def _verify_resident_slots(kv_table, stats: RunStats, threshold: float, slot_traces: dict, rank: int = 0) -> bool:
    device_map = _read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error("[producer] no device map available; skipping KV read/PCC.")
        _write_pcc_verdict(rank, ok=False, min_pcc=0.0, checked=0, threshold=threshold, per_cache={})
        return False

    dflash_threshold = float(os.environ.get("PREFILL_DFLASH_PCC", "0.88"))
    check_dflash = any(name.startswith("dflash_") for name in _config_names(kv_table))
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
        slot_mins = _read_slot_kv_and_check_pcc(kv_table, device_map, slot_id, real_len, slot_traces[slot_id])
        for cache, value in slot_mins.items():
            per_cache[cache] = value if cache not in per_cache else min(per_cache[cache], value)
        pcc = min(slot_mins.values())
        min_pcc_overall = min(min_pcc_overall, pcc)
        checked += 1
        if pcc < threshold:
            failures.append((slot_id, real_len, pcc))
        if check_dflash:
            dflash_pcc = dflash_kv_table_pcc_check(
                kv_table,
                slot_id,
                real_len,
                read_config_slice=read_dflash_slice,
                threshold=dflash_threshold,
                rope_convention="interleaved",
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
        logger.error(f"[producer] KV cache PCC below {threshold} for (slot, real_len, pcc): {failures}")
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
    if len(pool) < num_tokens:
        pool = pool + [1] * (num_tokens - len(pool))
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
            ok = _verify_resident_slots(kv_table, stats, cfg.pcc_threshold, slot_traces, rank=rank)
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

    ack_channel = _connect_layer_ack_channel(timeout_s) if cfg.verify else None
    if cfg.verify and ack_channel is None:
        logger.error(
            "[producer] CHECK_PCC=1 but LayerAck channel missing — UMD read would race the runner's "
            "prefill (H2D push return ≠ layers done). Set PREFILL_ENABLE_LAYER_ACK=1 on the runner "
            "(Gate 1 mock defaults this on via run_prefill_migration_gate.sh)."
        )
        sys.exit(1)
    if not cfg.verify:
        logger.info(
            "[producer] CHECK_PCC off — skipping the KV table read and not consuming the LayerAck "
            "channel (pure token feeder; the runner's migration self-test owns it)"
        )

    slot_traces, slot_lengths, pools_by_trace = _resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths

    def push_chunk(slot_id: int, chunk_idx: int, actual_start: int, actual_end: int) -> float:
        pool = pools_by_trace[slot_traces[slot_id]]
        chunk_bytes = _chunk_to_host_array(pool[actual_start : actual_start + CHUNK_SIZE])
        assert (
            chunk_bytes.nbytes == payload_bytes
        ), f"payload {chunk_bytes.nbytes}B != service-expected {payload_bytes}B"
        logger.info(f"[producer] push slot={slot_id} cidx={chunk_idx} start={actual_start} end={actual_end}")
        push_start = time.perf_counter()
        service.forward_to_tensor_bytes(chunk_bytes, metadata=_pack_metadata(slot_id, actual_start, actual_end))
        return (time.perf_counter() - push_start) * 1000.0

    warmup_chunks = int(os.environ.get("PREFILL_PRODUCER_WARMUP_CHUNKS", "0"))
    if warmup_chunks > 0:
        logger.info(f"[producer] warmup: {warmup_chunks} throwaway chunk(s) on slot 0 (not timed, not verified)")
        for cidx in range(warmup_chunks):
            push_chunk(0, cidx, cidx * CHUNK_SIZE, (cidx + 1) * CHUNK_SIZE)
        service.barrier()
        if ack_channel is not None:
            _drain_layer_acks(ack_channel, NUM_LAYERS * warmup_chunks)
        logger.info("[producer] warmup complete; starting the measured request")

    stats = run_schedule(cfg, push_fn=push_chunk)
    service.barrier()

    sorted_ms = sorted(stats.push_ms)
    total_tokens = stats.total_pushes * CHUNK_SIZE
    logger.info(
        f"[producer] DONE wall={stats.wall_s:.1f}s pushes={stats.total_pushes} requests={stats.completed} "
        f"tokens={total_tokens} throughput={total_tokens / stats.wall_s if stats.wall_s else 0:.0f} tok/s "
        f"push_ms p50={_percentile(sorted_ms, 0.5):.1f} p90={_percentile(sorted_ms, 0.9):.1f} "
        f"p99={_percentile(sorted_ms, 0.99):.1f}"
    )

    _drain_layer_acks(ack_channel, NUM_LAYERS * stats.total_pushes)

    if world_size > 1:
        _mr_bcast_resident(mr_rank, stats.resident)

    verify_ok = True
    if cfg.verify and kv_table is not None:
        try:
            verify_ok = _verify_resident_slots(kv_table, stats, cfg.pcc_threshold, slot_traces, rank=mr_rank)
        except Exception as e:
            logger.error(f"[producer] KV read/PCC failed: {type(e).__name__}: {e}")
            verify_ok = False
    elif cfg.verify:
        logger.error("[producer] PREFILL_PRODUCER_CHECK_PCC=1 but no KV chunk table available; skipping PCC.")
        verify_ok = False

    if world_size > 1:
        verdicts = _mr_allgather_verdict(verify_ok)
        for r, v in enumerate(verdicts):
            logger.info(f"[producer] rank={r}: ok={v}")
        verify_ok = all(verdicts)

    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        assert len(sentinel) == METADATA_SIZE_BYTES
        sentinel_payload = _chunk_to_host_array([1] * CHUNK_SIZE)
        assert sentinel_payload.nbytes == payload_bytes
        logger.info("[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(sentinel_payload, metadata=sentinel)
        service.barrier()
        logger.info("[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.")
    else:
        logger.info("[producer] exiting (the runner keeps its sync-op loop running).")

    if cfg.verify and not verify_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
