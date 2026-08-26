#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parametrized H2D producer for the prefill runner.

It behaves like a small inference scheduler: it drives N concurrent user "slots", each running a
request made of token chunks, and pushes those chunks to the runner over the H2D socket. The order
and timing of the pushes is described entirely by a flat `ProducerConfig` — the producer has no notion
of named "scenarios" or "modes". Scenarios (single-user, round-robin, random stress, ...) live in the
test suite as ProducerConfig data (see tests/test_producer_runner_e2e.py); this engine just runs one.

After pushing, it drains the runner's per-layer LayerAcks and, if asked, reads the resulting KV cache
back and PCC-checks it against the golden trace. The KV read (read_dram_umd over a bare UMD cluster)
and raw cache decode are BOTH device-less on purpose: touching a real ttnn device here would take the
CHIP_IN_USE lock the runner already holds, and deadlock.

`run_schedule()` takes injectable seams (push_fn / now_fn / sleep_fn / rng) so the scheduling logic
can be unit-tested with no device and reproduced deterministically.

Config — a YAML manifest (like the runner's PREFILL_MANIFEST) or PREFILL_* env vars. Point at a
  manifest with ``--manifest <path>`` (or PREFILL_PRODUCER_MANIFEST); main() applies it via setdefault
  (importing this module applies nothing), so any exported PREFILL_* env var still wins. Typed blocks map to the
  env vars documented below; a verbatim ``env:`` block passes any raw PREFILL_* key through (and wins
  over the typed blocks). See producer_manifests/prefill_producer_manifest.example.yaml. Schema:
    model:     {variant, num_layers, max_seq_len, chunk_size}
    transport: {sp, tp, h2d_service_id, connect_timeout_s}
    workload:  {num_users, chunks, max_requests, duration_s, interleave, p_gap, p_burst, gap_ms,
                mid_end_prob, seed, check_pcc, trace_dir, slot_prompts}
    env:       {ANY_PREFILL_KEY: value}   # escape hatch for anything unmodeled
  Any other top-level block is ignored here and returned by _apply_manifest_env() for another entry point
  to apply — that is how runners/migration_driver.py picks up ``migration:``.

Multi-rank: launched under an MPI launcher (OMPI_COMM_WORLD_SIZE > 1, one process per pipeline host)
the same entry point splits by rank — rank 0 is the master (feeds H2D, owns the LayerAck channel) and
every other rank is a device-less validator that only reads its own host's KV back and PCCs it. The
roles coordinate over MPI collectives, not files (see the "Multi-rank coordination" section below).
Standalone (no launcher) it is just the master. Multi-rank requires PREFILL_PRODUCER_CHECK_PCC=1.

Env — schedule knobs (flat; the defaults describe a 1-user, 11-chunk, in-order run):
  PREFILL_NUM_USERS              concurrent cache slots (default 1)
  PREFILL_PRODUCER_CHUNKS        chunks per request: "N" fixed, or "min,max" random (default "11")
  PREFILL_PRODUCER_MAX_REQUESTS  total requests across all slots before stopping (default 1)
  PREFILL_PRODUCER_DURATION_S    wall-clock bound (default "inf" = stop on request count)
  PREFILL_PRODUCER_P_GAP         per-step probability of an idle gap (default 0.0)
  PREFILL_PRODUCER_P_BURST       per-step probability of a 2-3 chunk burst (default 0.0)
  PREFILL_PRODUCER_GAP_MS        "min,max" idle-gap milliseconds (default "200,2000")
  PREFILL_PRODUCER_MID_END_PROB  probability a request ends mid-chunk (default 0.0)
  PREFILL_PRODUCER_INTERLEAVE    slot order: "random" (default) | "round_robin"
  PREFILL_PRODUCER_SEED          RNG seed (default 1234)
  PREFILL_PRODUCER_CHECK_PCC     "1" to read KV back and PCC vs golden per slot (default 0)
  PREFILL_PRODUCER_SLOT_TRACES   per-slot prompts: "dirA,dirB,..." assigns trace i to slot i (cycling by
                                 slot % count if fewer than num_users). Each slot pushes tokens from — and
                                 is PCC'd against — its OWN trace, and its depth is derived from that
                                 prompt's real length (so PREFILL_PRODUCER_CHUNKS/MID_END are ignored per
                                 slot). Unset (default) => one shared PREFILL_TRACE_DIR + the synthetic
                                 schedule for all slots.
  PREFILL_DFLASH_GOLDEN_KV_DIR   dir with the DFlash drafter's golden context K/V (k_cache/v_cache
                                 .safetensors); set it to ALSO PCC the drafter caches the table carries.
  PREFILL_DFLASH_PCC             per-(layer, head) bar for that drafter check (default 0.88). Both feed
                                 dflash_kv_table_pcc_check in deepseek_v3_d_p/tt/dflash_prefill/
                                 dflash_kv_validation.py; unset golden => the drafter half is skipped.
  PREFILL_SEND_SHUTDOWN          "1" to close the stream with an all -1 sentinel so the runner exits
                                 gracefully after the run (sent after the KV read; default 0). PR #48718.
Scope — this module drives the RUNNER and nothing else: push, ack-drain, optional golden PCC. It issues no
  KV migration and imports nothing that does. To prefill AND migrate, run runners/migration_driver.py: it
  reuses the helpers here and adds the migration steps. The dependency runs that way only, never back into
  this module, so a runner-only run can never pull migration in.
Env — transport (must match the runner): PREFILL_SP / PREFILL_TP / PREFILL_CHUNK_SIZE /
  PREFILL_MAX_SEQ_LEN / PREFILL_NUM_LAYERS / PREFILL_H2D_SERVICE_ID / PREFILL_H2D_CONNECT_TIMEOUT.

Usage:
    # From a manifest (env still overrides individual knobs):
    python -m models.demos.common.prefill.runners.prefill_producer \
      --manifest models/demos/common/prefill/runners/producer_manifests/prefill_producer_manifest.example.yaml
    # 1 user, full depth, with PCC:
    PREFILL_PRODUCER_CHUNKS=11 PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_producer
    # 8-user random stress + PCC:
    PREFILL_NUM_USERS=8 PREFILL_PRODUCER_CHUNKS=1,4 PREFILL_PRODUCER_MAX_REQUESTS=200 \
      PREFILL_PRODUCER_P_GAP=0.2 PREFILL_PRODUCER_P_BURST=0.3 PREFILL_PRODUCER_MID_END_PROB=0.33 \
      PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_producer
"""

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
    """Populate the PREFILL_* env from a YAML producer manifest and RETURN the parsed manifest (setdefault
    => an explicitly exported env var still wins). Mirrors the runner's _apply_manifest_env: a verbatim
    ``env:`` passthrough (applied FIRST, so a raw PREFILL_* key wins over the typed mapping) plus typed
    ``model`` / ``transport`` / ``workload`` blocks mapped to the same env vars _load_env_config() and
    _config_from_env() read.

    Blocks this module does not model are left untouched and reachable via the return value, so another
    entry point can apply its own. That is how the migration driver picks up ``migration:`` — this module
    stays unaware of it.

    Called ONLY from main(), and necessarily before those two reads — see the call site. Deliberately not
    invoked at import: this is the one function here that mutates os.environ, and a plain
    ``import prefill_producer`` must not silently apply a manifest just because the env names one."""
    import yaml

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    def sd(key, val):  # setdefault, stringified; skips None so an absent field leaves the default
        if val is not None:
            os.environ.setdefault(key, str(val))

    def sd_bool(key, val):  # YAML true/false -> the "1"/"0" the env parsing expects
        if val is not None:
            os.environ.setdefault(key, "1" if val else "0")

    # 1) verbatim escape hatch first — a raw PREFILL_* key wins over the typed blocks (setdefault:
    #    first write wins; a shell-exported env var pre-empts both).
    for key, val in (manifest.get("env") or {}).items():
        sd(key, val)

    # 2) typed blocks -> PREFILL_* env.
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
    slot_prompts = workload.get("slot_prompts")  # per-slot prompt trace dirs; index = slot_id
    if slot_prompts is not None:
        sd("PREFILL_PRODUCER_SLOT_TRACES", slot_prompts if isinstance(slot_prompts, str) else ",".join(slot_prompts))
    gap_ms = workload.get("gap_ms")  # accept a "lo,hi" string or a [lo, hi] list
    if gap_ms is not None:
        sd("PREFILL_PRODUCER_GAP_MS", gap_ms if isinstance(gap_ms, str) else ",".join(str(x) for x in gap_ms))

    logger.info(f"[producer] applied manifest {manifest_path}")
    return manifest


# PrefillMetadata on the wire: 3 x uint32 = [slot_id, actual_start, actual_end].
METADATA_SIZE_BYTES = 12


def _load_env_config() -> None:
    """(Re)bind the transport/model constants below from the CURRENT environment.

    Called once at import so a plain ``import prefill_producer`` still yields usable constants, and again
    from ``main()`` right after the manifest is applied — the manifest only reaches the env at that point,
    so the import-time values would otherwise be pre-manifest defaults. Importing this module never
    MUTATES the environment; only ``main()`` does, via _apply_manifest_env."""
    global SP_AXIS, TP_AXIS, GLOBAL_MESH_SHAPE, CHUNK_SIZE, MAX_SEQ_LEN, NUM_LAYERS, ADAPTER
    SP_AXIS = int(os.environ.get("PREFILL_SP", 8))
    TP_AXIS = int(os.environ.get("PREFILL_TP", 4))
    GLOBAL_MESH_SHAPE = (SP_AXIS, TP_AXIS)
    CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
    # Same 11-chunk default as the runner: a larger default here would clamp requests to a depth the
    # runner's cache can't hold, and the runner asserts on the overrunning chunk.
    MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 11))
    NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
    ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))


_load_env_config()


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    """Pack one chunk's PrefillMetadata (3 little-endian uint32s)."""
    return struct.pack("<III", slot_id, actual_start, actual_end)


def _chunk_to_host_array(chunk_token_ids):
    """One chunk's tokens as the un-sharded [SP, 1, chunk_local] uint32 buffer the H2D service expects.
    Block-cyclic / chip-major layout, matching the runner's prepare_prefill_input_tensor; the connected
    service resplits it across SP coordinates, so this process needs no MeshDevice."""
    sp = GLOBAL_MESH_SHAPE[0]
    chunk_local = CHUNK_SIZE // sp
    return (
        torch.tensor(chunk_token_ids, dtype=torch.int64)
        .reshape(sp, 1, chunk_local)
        .to(torch.uint32)
        .contiguous()
        .numpy()
    )


# ---------------------------------------------------------------------------
# Device-less helpers: read the KV table / device map, attach the LayerAck channel, decode cache data.
# All device-less on purpose — none may touch a real ttnn device (that would take the CHIP_IN_USE
# lock the runner holds and deadlock).
# ---------------------------------------------------------------------------


# Per-host filesystems: a table written under one of these by rank 0 is invisible to validators on
# other hosts, which resolve the same path against their OWN local mount.
_PER_HOST_FS_PREFIXES = ("/tmp", "/dev/shm", "/run", "/var/tmp")


def _require_shared_table_path(world_size: int) -> None:
    """Every validator reads the same serialized table rank 0 writes, so multi-rank requires it on
    shared storage. Reject the per-host default early and symmetrically — same env + world_size on
    every rank means all ranks exit together, never half-opening a coordination barrier — instead of
    letting a remote validator silently read a missing/stale file."""
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
    """Poll for and deserialize the KV chunk address table the runner published to
    PREFILL_MIGRATION_TABLE_PATH. Fully device-less (import_from_protobuf_file rebuilds it from the
    protobuf alone). Returns the table, or None if it never appears (the producer can still push)."""
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
    """Poll for and read the runner's fabric_node -> ASIC-unique_id sidecar, so read_dram_umd can pick
    chips by unique_id without touching the ControlPlane. Returns {(mesh_id, chip_id): unique_id}.

    The runner's two publishers do not share an encoding -- the shmem path writes JSON keyed
    "<mesh>:<chip>", the file-export path writes "<mesh> <chip> <umd>" lines -- so accept either.

    A multi-rank runner writes one rank-scoped file per co-located rank (``<stem>_r<rank>.json``), so
    all matches on this host are merged. Clear stale ``_r*`` siblings between runs whose topology
    changed -- a leftover file would merge in."""
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
    """Attach (consumer side) to the runner's per-layer LayerAck channel
    (/tt_prefill_layer_acks_<service_id>). Returns the channel, or None if it isn't available (only the
    single-rank runner creates it)."""
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
    """Block until `expected` per-layer acks (NUM_LAYERS per chunk) have been drained, or timeout.
    Returns the count actually drained."""
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
    """Decode a [32, head_dim] bfp8_b tile chunk (raw device bytes) to a float32 [32, head_dim] tensor,
    in pure numpy (no ttnn tensor ops, so it never inits the device context / takes the CHIP_IN_USE
    lock). Validated bit-exact against ttnn._ttnn.bfp_utils.unpack_bfp8.

    Layout per 1088-byte tile: 64 exponent bytes (one per (face, row)) then 1024 mantissa bytes
    ((face, row, col)); value = (-1)^sign * (mantissa & 0x7F) * 2^(exponent - 133). Face f = fr*2 + fc
    maps to tile rows fr*16 + r and cols fc*16 + c; tiles lie along the head_dim (column) axis.
    """
    TILE = 32
    n_tiles = head_dim // TILE
    raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(n_tiles, 1088)

    exponents = raw_u8[:, :64].astype(np.int32).reshape(n_tiles, 4, 16)  # (tile, face, row)
    mantissas = raw_u8[:, 64:].reshape(n_tiles, 4, 16, 16)  # (tile, face, row, col)
    signs = (mantissas >> 7).astype(np.int32)
    magnitude = (mantissas & 0x7F).astype(np.float32)
    scale = np.exp2((exponents - 133).astype(np.float32))[..., None]
    values = np.where(signs > 0, -(magnitude * scale), magnitude * scale)  # (tile, face, row, col)

    # face (fr, fc) -> tile rows fr*16+r, cols fc*16+c
    by_face = values.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)  # tile t -> columns [t*32 : (t+1)*32]
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _decode_bf16_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    """Decode a ``[32, head_dim]`` bf16 TILE chunk (raw device bytes) to float32, device-less. Same face/
    tile de-swizzle as ``_decode_bfp8_chunk`` but bf16 has no exponent block: each 2048-byte tile is 1024
    bf16 values (uint16 -> float32 via a 16-bit left shift, which is exact). NOTE: not validated against a
    ttnn unpack the way the bf8 path is."""
    TILE = 32
    n_tiles = head_dim // TILE
    u16 = np.frombuffer(raw, dtype="<u2").reshape(n_tiles, 4, 16, 16)  # (tile, face, row, col)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    by_face = f32.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _decode_row_major_chunk(raw: bytes, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Decode 32 native row pages, dropping any physical row padding."""
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
    """Reconstruct ``[scaled latent | RoPE]`` from packed mixed-format row bytes."""
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
    """Decode one raw 32-token MLA KV chunk by its table-reported physical byte size.

    The deployed 576-wide formats have distinct physical row sizes: tiled bfp8_b, raw row-major
    fp8_e4m3, packed scaled FP8, and row-major bfloat16. The packed format is reconstructed from its
    512 E4M3 latent bytes, four FP32 scales, and 64 BF16 RoPE values. Row padding is discarded.
    """
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
    """ASIC unique_id for any replica fabric node present in the device map (replicas hold identical KV,
    and add_device_group sorts the ids so index 0 is not a fixed chip). Raises if none are mapped."""
    for node in fabric_node_ids:
        key = (int(node.mesh_id), int(node.chip_id))
        if key in device_map:
            return device_map[key]
    tried = [(int(n.mesh_id), int(n.chip_id)) for n in fabric_node_ids]
    raise KeyError(f"no fabric node {tried} in device map ({len(device_map)} chips; single-rank/one-galaxy only)")


# ---------------------------------------------------------------------------
# Config + scheduler engine (mode-unaware; run_schedule touches no device/ttnn, so it is unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class ProducerConfig:
    """A flat description of a push schedule. A "scenario" is just a set of these values."""

    num_users: int  # number of concurrent cache slots (users)
    chunks_min: int  # per-request chunk count is rng.randint(chunks_min, chunks_max)
    chunks_max: int
    max_requests: int  # total requests across all slots before stopping
    duration_s: float  # wall-clock bound (inf => stop on request count only)
    p_gap: float  # per-step probability of an idle gap
    p_burst: float  # per-step probability of a 2-3 chunk burst to one slot
    gap_ms: tuple  # (min, max) idle-gap milliseconds
    mid_chunk_end_prob: float  # probability a request ends mid-chunk (exercises the actual_end clamp)
    seed: int
    verify: bool  # read KV back and PCC each resident slot vs golden
    pcc_threshold: float
    interleave: str = "random"  # slot order: "random" | "round_robin" (fair alternation)
    slot_lengths: dict = None  # per-slot real token count (from per-slot prompts); overrides the random
    # chunk draw + mid_chunk_end when set (each slot pushes exactly its prompt). None => synthetic depth.
    multi_turn_prob: float = 0.0  # P(a recycled slot resumes its conversation at the 32-aligned cached
    # prefix instead of restarting at 0). 0.0 => every recycle is a fresh request (pre-multi-turn behaviour).


def _config_from_env() -> ProducerConfig:
    """Build a ProducerConfig from the flat PREFILL_PRODUCER_* env vars (every knob independent)."""
    max_chunks = MAX_SEQ_LEN // CHUNK_SIZE  # a request can't exceed the per-user cache
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
    """One cache-user slot holding one in-flight request. `next_chunk` advances per push; `actual_isl` is
    the conversation's real (non-pad) token count. `prefix_len` is the 32-aligned absolute cache offset
    this turn resumes at (0 for a fresh conversation); every length here is absolute, not turn-relative."""

    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.req_id = -1
        self.target_chunks = 0
        self.next_chunk = 0
        self.actual_isl = 0
        self.prefix_len = 0  # absolute, 32-aligned; where this turn starts writing
        self.turn_idx = 0  # 0 = first turn of the conversation resident in this slot

    @property
    def done(self) -> bool:
        return self.next_chunk >= self.target_chunks


class _SlotFill(NamedTuple):
    """How much of one slot's KV cache the prefill filled: `real_len`, the absolute non-pad extent
    [0, real_len) the runner actually wrote (the last `actual_end` pushed). Recorded, not re-derived from
    chunk counts -- a multi-turn slot resumes at a non-zero prefix, so chunks_pushed * CHUNK_SIZE would
    describe only its latest turn. Shared producer/migration_driver contract."""

    real_len: int


def _new_request(
    slot: _Slot, req_id: int, cfg: ProducerConfig, rng: random.Random, *, prefix_len: int = 0, turn_idx: int = 0
) -> None:
    """(Re)assign a request to `slot`, starting at chunk 0 of a turn that writes from `prefix_len`.

    Per-slot-prompt mode (cfg.slot_lengths set): the slot pushes exactly its assigned prompt (depth
    ceil(real_len / CHUNK_SIZE)); the random chunk draw + mid_chunk_end are bypassed. Otherwise a random
    chunk count, optionally ending mid-chunk. A resumed turn (prefix_len > 0) clamps its depth draw to the
    capacity still left under the per-user cache; at prefix_len == 0 that clamp is already chunks_max, so
    the rng sequence is unchanged."""
    slot.req_id = req_id
    slot.next_chunk = 0
    slot.prefix_len = prefix_len
    slot.turn_idx = turn_idx
    if cfg.slot_lengths is not None and slot.slot_id in cfg.slot_lengths:
        real_len = cfg.slot_lengths[slot.slot_id]
        # real_len is relative to the turn; actual_isl is absolute, so a resumed turn ends at
        # prefix_len + real_len. If that no longer fits the per-user cache, restart the conversation.
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
    resident: dict  # slot_id -> _SlotFill (what is physically in that slot's KV cache)
    total_pushes: int
    push_ms: list
    completed: int
    wall_s: float


def run_schedule(cfg: ProducerConfig, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    """Execute the push schedule described by `cfg`.

    Device-free: `push_fn(slot_id, chunk_idx, actual_start, actual_end) -> elapsed_ms` does and times the
    push; now_fn/sleep_fn/rng are injectable so tests run instantly and deterministically. Records each
    slot's resident request as a `_SlotFill` at push time, and returns RunStats. A recycled slot restarts
    at chunk 0 unless `cfg.multi_turn_prob` continues the conversation from its aligned-down length; either
    way `_SlotFill.real_len` is the slot's absolute non-pad extent.
    """
    rng = rng if rng is not None else random.Random(cfg.seed)
    slots = [_Slot(i) for i in range(cfg.num_users)]
    resident: dict = {}

    # Give every slot an initial request; `next_req_id` counts total requests (initial + recycled).
    next_req_id = 0
    for slot in slots:
        _new_request(slot, next_req_id, cfg, rng)
        next_req_id += 1

    push_ms: list = []
    total_pushes = 0
    completed = 0
    round_robin_cursor = -1  # only used when cfg.interleave == "round_robin"
    start = now_fn()

    def send_chunk(slot: _Slot) -> None:
        nonlocal total_pushes, completed, next_req_id
        chunk_idx = slot.next_chunk
        actual_start = slot.prefix_len + chunk_idx * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        push_ms.append(push_fn(slot.slot_id, chunk_idx, actual_start, actual_end))
        total_pushes += 1
        slot.next_chunk += 1
        resident[slot.slot_id] = _SlotFill(real_len=actual_end)  # what's now resident in this slot
        if slot.done:
            completed += 1
            if next_req_id < cfg.max_requests:  # recycle the slot
                # Multi-turn: resume at the aligned prefix already cached. Align DOWN to 32
                # (update_padded_kv_cache asserts kv_actual_global % 32 == 0); the <=31 tail tokens are
                # replayed by this turn's first chunk (PCC-idempotent). Guarded so the default path draws
                # no extra rng value and the schedule stays bit-identical.
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

        # One random draw classifies the step: [0, p_gap) gap, [p_gap, p_gap+p_burst) burst, else single.
        roll = rng.random()
        if roll < cfg.p_gap:
            sleep_fn(rng.uniform(*cfg.gap_ms) / 1000.0)
            continue

        if cfg.interleave == "round_robin":
            # Advance the cursor to the next non-done slot (active_slots is non-empty, so this finds one).
            for _ in range(len(slots)):
                round_robin_cursor = (round_robin_cursor + 1) % len(slots)
                if not slots[round_robin_cursor].done:
                    break
            slot = slots[round_robin_cursor]
        else:
            slot = rng.choice(active_slots)

        if roll < cfg.p_gap + cfg.p_burst:  # burst: 2-3 chunks to this slot
            for _ in range(rng.randint(2, 3)):
                if slot.done:
                    break
                send_chunk(slot)
        else:
            send_chunk(slot)

    return RunStats(
        resident=resident, total_pushes=total_pushes, push_ms=push_ms, completed=completed, wall_s=now_fn() - start
    )


# ---------------------------------------------------------------------------
# Per-slot KV read-back + PCC (device-less: read_dram_umd over UMD + host cache decode)
# ---------------------------------------------------------------------------


def _read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """Read slot `slot_id`'s KV over [0, real_len) via the published table and PCC-check it against the
    golden trace. Dispatches on the model: MLA (single merged kvpe config), M3 (multi-config triple
    cache), or GPT-OSS (multi-config K/V heads, no index_k). Returns ``{cache name: min PCC across that
    cache's layers}`` — one entry per MODEL cache actually validated, so the caller can gate on the min
    while still reporting each cache separately. A cache with no golden to compare against is ABSENT from
    the mapping rather than present at 1.0.

    The reader is NOT adapter-pluggable — a new model whose cache is neither of those layouts needs
    a branch here (and its own decode), not just an adapter.

    The MODEL's own caches only. Under DFlash the drafter's context caches ride in the same table under
    further `dflash_*` configs, checked as a sibling gate from _verify_resident_slots (see
    dflash_kv_table_pcc_check in the deepseek dflash_prefill module)."""
    if ADAPTER.name == "minimax_m3":
        return _read_slot_kv_and_check_pcc_m3(table, device_map, slot_id, real_len, trace_dir)
    if ADAPTER.name == "gpt_oss_d_p":
        return _read_slot_kv_and_check_pcc_gpt_oss(table, device_map, slot_id, real_len, trace_dir)
    return _read_slot_kv_and_check_pcc_mla(table, device_map, slot_id, real_len, trace_dir)


def _config_names(table) -> list:
    """Every config's name, indexed by config id."""
    return [table.config_name(i) for i in range(table.num_configs())]


def _num_model_configs(table) -> int:
    """Configs describing the MODEL's own caches (KVPE + any index cache), i.e. the decimal-named ones. The
    drafter's `dflash_*` are excluded, so `num_configs() > 1` can't read a dense+DFlash table as sparse."""
    return sum(1 for name in _config_names(table) if name.isdigit())


def _read_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim, decode):
    """Read one config's KV chunks over [0, read_len) for (layer, slot) via the address table and return
    the decoded ``[read_len, head_dim]`` tensor in natural token order."""
    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    rows = []
    for pos in range(0, read_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        loc = table.lookup(layer, pos, slot_id, config_id)
        unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
        rows.append(decode(raw, head_dim))
    return torch.cat(rows, dim=0)[:read_len]


def _read_slot_kv_and_check_pcc_gpt_oss(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """GPT-OSS multi-config read-back: reconstruct per-head K/V from the 2N-config table and PCC vs the
    GQA golden (no index_k). Config layout: k_h0..N-1 = 0..N-1, v_h0..N-1 = N..2N-1. Returns the per-cache
    minima keyed by cache name."""
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
        # Skip layers owned by another host's stage (their chips are not in this host's device map).
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
    # Zero resolved layers would return the 1.0 inits as a perfect pass — fail loudly instead.
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")
    return mins


def _read_slot_kv_and_check_pcc_m3(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """M3 multi-config read-back: reconstruct per-head K/V + index_k from the 9-config table and PCC vs the
    separate_k_v golden. Config layout matches the builder: k_h0..N-1 = 0..N-1, v_h0..N-1 = N..2N-1,
    index_k = 2N. Returns the per-cache minima keyed by cache name."""
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import _hf_to_meta_rotary_perm
    from tests.ttnn.utils_for_testing import comp_pcc

    mc = ADAPTER.model_config
    n_kv, head_dim, rotary_dim = mc.NUM_KEY_VALUE_HEADS, mc.HEAD_DIM, mc.ROTARY_DIM
    perm = _hf_to_meta_rotary_perm(head_dim, rotary_dim)  # golden HF -> device Meta rotary swizzle
    # index_k dtype from its config's chunk size (bf8 vs bf16) -> the right decoder.
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
        # Skip layers owned by another host's stage (their chips are not in this host's device map).
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
        )[
            :, :real_len
        ]  # [n_kv, real_len, head_dim]
        dev_v = torch.stack(
            [
                _read_kv_slice(table, device_map, n_kv + h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{layer}").float()[0, :, :real_len, :][..., perm]  # HF -> Meta
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
    # Zero resolved layers would return the 1.0 inits as a perfect pass — fail loudly instead.
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")
    # Only some traces carry an index_k golden. Its min is still the 1.0 init when none did, so drop the
    # key rather than reporting an unmeasured cache as perfect.
    if not ik_checked:
        del mins["index_k"]
    return mins


def _full_indexer_layer_indices(num_layers: int):
    """Global layer indices that own a lightning indexer, over layers [0, num_layers).

    GLM-5.2 reuses one ``full`` layer's top-k across the ``shared`` layers that follow, so only full
    layers write the index cache and config 1 of the merged table is COMPACTED to that count: its layer
    axis is the full-indexer RANK, not the global layer index. The golden trace's
    ``dsa/indexer_k_layer_N`` is numbered by GLOBAL layer, so rank -> global has to be mapped before
    loading it (GLM-5.2 full layers are {0, 1, 2, 6, 10, ... 74}, so rank 3 is global layer 6).

    Returns None when the model has no ``indexer_types`` map (GLM-5.1 / v3.2: every layer is a full
    indexer owner, so rank == global index and no mapping is needed).
    """
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused

    hf_config = ADAPTER.load_hf_config()  # host-only attribute config; no device, no weights
    if not getattr(hf_config, "indexer_types", None):
        return None
    return [layer for layer in range(num_layers) if not indexer_layer_is_reused(hf_config, layer)]


def _read_slot_kv_and_check_pcc_mla(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """Read slot `slot_id`'s KV over [0, real_len) via the table and validate it. Config 0 (the KVPE
    cache) is PCC'd vs the golden trace. For a sparse/DSA model the merged table also carries config 1
    (the index-key cache), which is PCC'd vs the golden indexer key. Returns ``{"kvpe": min}`` plus
    ``"index": min`` when the index cache was validated — omitted, not defaulted to 1.0, when the model is
    dense or the trace carries no indexer-key golden (warned, not fatal — see below). Raises on an
    index-cache READ failure."""
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
        _load_golden_index_k,
        _load_golden_kv_post,
        index_golden_present,
    )
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    KV_LORA = ADAPTER.model_config.KV_LORA_RANK  # "nope" part: device_kv[:, :KV_LORA]
    HEAD_DIM = KV_LORA + ADAPTER.model_config.QK_ROPE_HEAD_DIM  # + rope "pe" part: device_kv[:, KV_LORA:]
    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    read_len = ((real_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block  # round up to a block

    min_pcc = 1.0
    checked = 0
    for layer in range(NUM_LAYERS):
        # A merged multi-rank table spans every host's layers, but this producer's device map holds only
        # its co-located host's chips, so a layer owned by another rank resolves to no local unique_id.
        # Skip it: each rank validates exactly the layers physically resident on its own machine.
        loc0 = table.lookup(layer, 0, slot_id)
        try:
            _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
        except KeyError:
            continue

        # Read this layer's KV block by block over UMD, decode its physical cache format, and concat.
        decoded_rows = []
        for pos in range(0, read_len, tokens_per_block):
            loc = table.lookup(layer, pos, slot_id)
            unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
            decoded_rows.append(_decode_kv_chunk(raw, HEAD_DIM))
        device_kv = torch.cat(decoded_rows, dim=0)[:real_len]  # natural order (table un-rotates block-cyclic)

        golden = _load_golden_kv_post(trace_dir, layer, real_len)
        _, pcc_nope = comp_pcc(golden[:, :KV_LORA], device_kv[:, :KV_LORA])

        # The rope "pe" columns are stored interleaved in HF layout; re-interleave the golden to Meta layout.
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
    # No local layer resolved to this rank's device map — min_pcc is still its 1.0 init, which would
    # masquerade as a perfect pass. A rank that was asked to verify but owns no layers of this slot is a
    # misconfiguration (wrong device map / stage split), so fail loudly instead of returning 1.0.
    if checked == 0:
        raise RuntimeError(f"slot {slot_id}: no local layers resolved against the device map (nothing verified)")

    # config 1: index cache (sparse/DSA only). Validated the SAME way as config 0 — read block-by-block
    # via the table, decode, and PCC vs the golden indexer key. Config 1 holds all layers on GLM-5.1 and
    # only the full-indexer layers on GLM-5.2, so iterate its OWN layer count, not NUM_LAYERS. The index
    # cache is bf8 TILE, and the golden is already in the device rope frame (no re-interleave, unlike pe).
    #
    # A trace can carry the KVPE golden but no indexer-key golden (some vllm dumps store only
    # dsa/dsa_topk_indices_layer_*). There is then nothing to PCC config 1 against, so warn and return the
    # config-0 PCC rather than failing the slot: the KVPE result is still a valid check.
    # `_num_model_configs`, not `num_configs()`: under DFlash the same table also carries the drafter's
    # `dflash_*` configs, and config 1 is only the index cache when the MODEL published two caches.
    mins = {"kvpe": min_pcc}
    if _num_model_configs(table) > 1:
        if not index_golden_present(trace_dir):
            logger.warning(
                f"[producer] slot {slot_id}: table has an index config but {trace_dir} carries no "
                f"indexer-key golden; validating the KVPE cache only (device index cache NOT checked)."
            )
            return mins

        index_head_dim = ADAPTER.model_config.INDEX_HEAD_DIM
        n_index_layers = table.config(1).num_layers
        # Config 1 is published on the GLOBAL LAYER axis, same numbering as the golden: rows == full-indexer layer ids.
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
            # Same host-local filter as config 0: skip an index layer owned by another rank.
            loc0 = table.lookup(layer, 0, slot_id, 1)  # config 1 = index cache
            try:
                _resolve_unique_id(table.get_device_group(loc0.device_group_index).fabric_node_ids, device_map)
            except KeyError:
                continue

            decoded_rows = []
            for pos in range(0, read_len, tokens_per_block):
                loc = table.lookup(layer, pos, slot_id, 1)  # config 1 = index cache, keyed by global layer
                unique_id = _resolve_unique_id(
                    table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                )
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                # Same byte-size-driven dispatch as config 0; a 128-wide bfp8 index chunk is
                # (128/32) x 1088 = 4352 B, which lands on the bfp8 tile branch.
                decoded_rows.append(_decode_kv_chunk(raw, index_head_dim))
            dev_ik = torch.cat(decoded_rows, dim=0)[:real_len]

            golden_ik = _load_golden_index_k(trace_dir, layer, real_len)
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
    """Persist this rank's PCC verdict to PREFILL_PCC_SUMMARY_DIR (opt-in). The verdict is logged to
    stdout just before the shutdown sentinel, and mpirun drops its buffered output forwarding when the
    runner tears down in response — so under a launcher the numbers never reach the captured log. A file
    on shared storage is read back by the harness independently of that teardown.

    `min_pcc` is the gated number: the min over every validated cache. `per_cache` breaks it down by cache
    (a sparse model's KVPE vs index cache), so a durable verdict distinguishes which cache set the bar and
    shows that the others were checked at all — a cache missing from it was NOT validated."""
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
    """PCC-check every slot that holds resident trace-derived KV, each against ITS OWN golden trace
    (slot_traces[slot_id]). Returns True only if at least one slot was checked and all met the threshold.

    Two independent gates per slot, both of which must pass: the model's own caches vs the trace, and --
    when the table carries the DFlash drafter's configs -- the drafter's context K/V at its own bar (see
    dflash_kv_table_pcc_check). The drafter half is silent when not measured (no marker field, no success
    line), so a non-DFlash run's output is unchanged."""
    device_map = _read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error("[producer] no device map available; skipping KV read/PCC.")
        _write_pcc_verdict(rank, ok=False, min_pcc=0.0, checked=0, threshold=threshold, per_cache={})
        return False

    dflash_threshold = float(os.environ.get("PREFILL_DFLASH_PCC", "0.88"))
    # Under DFlash the table also carries the drafter's context caches (extra dflash_* configs); PCC them
    # via the deepseek gate. The import + read closure are built only when those configs are present, so a
    # non-DFlash run neither imports the deepseek module nor measures anything.
    check_dflash = any(name.startswith("dflash_") for name in _config_names(kv_table))
    if check_dflash:
        from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import dflash_kv_table_pcc_check

        def read_dflash_slice(config_id, layer, slot_id, read_len, head_dim):
            return _read_kv_slice(
                kv_table, device_map, config_id, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk
            )

    min_pcc_overall = 1.0
    per_cache = {}  # cache name -> min over slots; a cache never validated stays absent
    min_dflash_overall = None  # stays None unless a drafter slice is actually measured
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
        pcc = min(slot_mins.values())  # the gate is the weakest cache
        min_pcc_overall = min(min_pcc_overall, pcc)
        checked += 1
        if pcc < threshold:
            failures.append((slot_id, real_len, pcc))
        if check_dflash:
            # None when no golden is configured (the drafter golden is prompt-specific, so opt-in).
            dflash_pcc = dflash_kv_table_pcc_check(
                kv_table,
                slot_id,
                real_len,
                read_config_slice=read_dflash_slice,
                threshold=dflash_threshold,
                rope_convention="interleaved",  # the only convention this branch's drafter implements
            )
            if dflash_pcc is not None:
                min_dflash_overall = dflash_pcc if min_dflash_overall is None else min(min_dflash_overall, dflash_pcc)
                if dflash_pcc < dflash_threshold:
                    dflash_failures.append((slot_id, real_len, dflash_pcc))

    # The drafter field appears ONLY when it was measured: on every non-DFlash run the marker line is exactly
    # what it was before this gate existed. Keep `min_pcc=` last-but-one so a `min_pcc=([\d.]+)` reader is
    # unaffected either way.
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list, p: float) -> float:
    """p-th percentile (p in 0..1) of an already-sorted list; 0.0 if empty."""
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def _load_token_pool(trace_dir, num_tokens: int) -> list:
    """A token pool a request replays from chunk 0, padded up to `num_tokens` if the trace is shorter."""
    pool = load_trace_token_ids(trace_dir, num_tokens)
    if len(pool) < num_tokens:
        pool = pool + [1] * (num_tokens - len(pool))
    return pool[:num_tokens]


def _resolve_slot_prompts(cfg: ProducerConfig):
    """Resolve each slot's prompt (tokens + golden trace) and load the token pool(s).

    Returns ``(slot_traces, slot_lengths, pools_by_trace)``:
      * slot_traces: {slot_id -> resolved trace Path}. Both the tokens pushed AND the golden PCC'd for a
        slot come from its trace, so per-slot traces == per-slot prompts + per-slot goldens.
      * slot_lengths: {slot_id -> real token count} in per-slot-prompt mode, else None (see _new_request).
      * pools_by_trace: {trace Path -> token pool}, deduped so a trace shared by N slots loads once.

    Single-prompt (default): no PREFILL_PRODUCER_SLOT_TRACES => every slot uses PREFILL_TRACE_DIR (or the
    adapter default), the synthetic schedule drives depth (slot_lengths=None), one pool sized to chunks_max.

    Multi-prompt: PREFILL_PRODUCER_SLOT_TRACES="dirA,dirB,..." assigns trace i to slot i (cycling by
    ``slot % len`` if fewer entries than users, so "dirA,dirB" alternates across 8 users). Each slot then
    pushes exactly its prompt: depth = ceil(real_len/CHUNK_SIZE), clamped to the per-user cache."""
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
        # A resumed turn slices the pool from its prefix onward, so multi-turn needs a pool covering the
        # whole per-user cache, not just one request's depth (else push_chunk's payload-size assert fires).
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


# Multi-rank coordination (device-less; GO/DONE over MPI collectives, not sync files)
#
# Only the barriers are MPI — the merged KV table and the device map are still delivered as files the
# producer polls; the collectives just replace the old NFS GO/DONE sentinels.
#
# One producer runs per host under an MPI launcher (mpirun-ulfm), mirroring the pipeline runner's
# ranks. rank 0 is the master (co-located with the runner's first rank): it alone feeds tokens over
# H2D and owns the aggregated LayerAck channel. Every rank reads its OWN host's KV back and PCCs only
# the layers resident on its machine (the merged table + a host-local device map filter to the local
# layers automatically). Coordination is two collectives over the distributed context (host-side MPI,
# NO mesh device): the master broadcasts the resident-slot map once every layer of every chunk has
# acked — this releases the validators (GO) — then an allgather of each rank's PCC ok-flag both waits
# for every validator's read-back to finish (DONE) and folds the verdicts, so the master holds the
# runner's shutdown sentinel until the mesh/DRAM is safe to tear down.
# ---------------------------------------------------------------------------


def _mr_config():
    """(rank, world_size). Under an MPI launcher (OMPI_COMM_WORLD_SIZE > 1) initialize the distributed
    context and take rank/size from it. Standalone (the single-rank de-risk, no mpirun) skips MPI
    entirely: 0 / 1, no coordination. rank 0 is the master; every other rank is a validator."""
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) <= 1:
        return (0, 1)
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    size = int(ttnn.distributed_context_get_size())
    return (rank, size)


def _mr_bcast_resident(rank: int, resident: dict) -> dict:
    """Broadcast the master's resident-slot map (slot_id -> _SlotFill) to every rank via allgather_int:
    element [0] of each allgather is rank 0's contribution, giving a broadcast from the only value-carrying
    collective ttnn exposes (no native broadcast/scatter). Doubles as the GO barrier: a validator blocks in
    the first allgather until the master arrives (only after it has drained every LayerAck). Non-master ranks
    pass {} and receive the map. All ranks must issue the same number of allgathers in the same order, so the
    slot count is broadcast first, then each slot's (slot_id, real_len)."""
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
    """Collective: every rank contributes its PCC ok-flag and receives all of them. Doubles as the DONE
    barrier — the master cannot proceed to the shutdown sentinel until every validator has reached here
    (i.e. finished its read-back)."""
    return [bool(v) for v in ttnn.distributed_context_allgather_int(1 if ok else 0)]


def _run_validator(rank: int, world_size: int) -> None:
    """Non-master path: no H2D feed. Read the merged table, wait for the master's GO (the resident-map
    broadcast), PCC this host's local layers, then join the verdict allgather (the master reads the
    result). Exits non-zero on PCC failure."""
    cfg = _config_from_env()
    # A validator's whole job is read-back PCC, and the GO barrier's "every layer is written" guarantee
    # comes from the master draining LayerAcks — which it only does when verify is on. Both ranks see the
    # same env, so this exits on every rank symmetrically (before any collective) — no half-open barrier.
    if not cfg.verify:
        logger.error("[producer] multi-rank requires PREFILL_PRODUCER_CHECK_PCC=1 (validators only verify).")
        sys.exit(1)
    _require_shared_table_path(world_size)
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(f"[producer] validator rank={rank}/{world_size}: read-back only (no H2D feed)")

    # GO before the table read: the master broadcasts the resident map only after draining every
    # LayerAck, which also means rank 0 finished publishing this run's table. Reading after GO can't
    # observe a stale prior-run table, and a read failure here still reaches the verdict allgather
    # (ok=False) — the master already cleared GO, so it can't hang.
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
            # Rebuild the SAME config-derived slot->trace map the master builds (both ranks see identical
            # env + the shared trace dir), so the validator PCCs its host's layers against the same goldens.
            # Purely config-derived (no push/device state), so it's safe to resolve here on the read-back path.
            slot_traces, _slot_lengths, _pools = _resolve_slot_prompts(cfg)
            ok = _verify_resident_slots(kv_table, stats, cfg.pcc_threshold, slot_traces, rank=rank)
        except Exception as e:
            logger.error(f"[producer] validator KV read/PCC failed: {type(e).__name__}: {e}")
            ok = False

    _mr_allgather_verdict(ok)  # DONE barrier + verdict fold
    logger.info(f"[producer] validator rank={rank}: DONE ok={ok}")
    if not ok:
        sys.exit(1)


def main() -> None:
    # argparse is the SINGLE place argv is read: --manifest defaults to PREFILL_PRODUCER_MANIFEST, so one
    # parse covers both sources and unknown args still error out.
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

    # Apply the manifest HERE, not at import: importing this module must never mutate os.environ. Order
    # matters — the manifest lands in the env first, then _load_env_config() re-reads the module constants
    # (bound to pre-manifest defaults at import) and _config_from_env() reads the schedule knobs.
    if args.manifest:
        _apply_manifest_env(args.manifest)
    _load_env_config()

    mr_rank, world_size = _mr_config()
    if mr_rank != 0:
        _run_validator(mr_rank, world_size)
        return

    cfg = _config_from_env()
    # See _run_validator: multi-rank coordination only holds together when every rank verifies (the GO
    # barrier depends on the master draining LayerAcks, which is gated on verify). Assert it on the
    # master too — same env on every rank means this exits symmetrically, never half-opening a barrier.
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

    # Read the KV table BEFORE pushing (the runner publishes it at setup). Only the read-back needs it,
    # and a runner that publishes no table (no migration, no PREFILL_MOCK_MIGRATION) would otherwise cost
    # a full connect timeout of dead air before the first push.
    kv_table = _read_kv_chunk_table(timeout_s) if cfg.verify else None
    # The LayerAck channel is a shared counter and try_consume_all() REMOVES completions. Draining it
    # serves ONLY the golden-trace KV read-back below

    # If we're not performing golden trace PCC-validation, then don't consume these and allow loopback
    # migration test in prefill_runner.py to consume acks and perform the testing of loopback migration
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

    # Per-slot prompts: each slot pushes tokens from (and is PCC'd against) its own trace. With no
    # PREFILL_PRODUCER_SLOT_TRACES every slot shares one trace (PREFILL_TRACE_DIR / the adapter default).
    slot_traces, slot_lengths, pools_by_trace = _resolve_slot_prompts(cfg)
    cfg.slot_lengths = slot_lengths  # None => synthetic schedule depth; else depth per prompt length

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

    # Wait for the runner's per-layer LayerAcks: NUM_LAYERS per chunk, for every chunk pushed. With a
    # pipeline runner (num_ranks>1) the branch's LayerCompletionRouter funnels every rank's completions
    # into this master channel, so this waits for ALL ranks' layers, not just the first stage's.
    _drain_layer_acks(ack_channel, NUM_LAYERS * stats.total_pushes)

    # Multi-rank: all layers of all chunks are now written across every stage's DRAM. Release the
    # validators (they PCC their own host's layers) by broadcasting the resident-slot map. Do it BEFORE
    # the master's own read-back so the reads overlap across hosts; the broadcast is the GO barrier.
    if world_size > 1:
        _mr_bcast_resident(mr_rank, stats.resident)

    # Opt-in: read the generated KV back per resident slot and PCC-check vs the golden trace.
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

    # Multi-rank: the verdict allgather is the DONE barrier — it can't return until every validator has
    # finished its read-back, so the shutdown sentinel below won't tear the mesh/DRAM down while one is
    # still reading. Fold every rank's verdict (including this master's own, contributed as element [0]).
    if world_size > 1:
        verdicts = _mr_allgather_verdict(verify_ok)
        for r, v in enumerate(verdicts):
            logger.info(f"[producer] rank={r}: ok={v}")
        verify_ok = all(verdicts)

    # Optional graceful shutdown (PR #48718): close the stream with an all -1 PrefillMetadata sentinel so
    # the runner breaks its request loop and tears down cleanly instead of blocking to SIGKILL. Sent LAST,
    # after the KV read, because read_dram_umd needs the mesh/DRAM alive (the runner is idle until now).
    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        assert len(sentinel) == METADATA_SIZE_BYTES
        sentinel_payload = _chunk_to_host_array([1] * CHUNK_SIZE)  # contents ignored by the runner; size must match
        assert sentinel_payload.nbytes == payload_bytes
        logger.info("[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(sentinel_payload, metadata=sentinel)
        service.barrier()  # drain the sentinel before releasing the descriptor
        logger.info("[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.")
    else:
        logger.info("[producer] exiting (the runner keeps its sync-op loop running).")

    # Non-zero exit on PCC failure so a CI / scripted run can gate on the exit code (after the sentinel,
    # so the runner is still told to drain even when verification failed).
    if cfg.verify and not verify_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
