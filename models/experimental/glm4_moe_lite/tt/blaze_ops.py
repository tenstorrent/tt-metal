# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optional tt-blaze acceleration for GLM-4.7-Flash decode clusters.

Import-guarded end to end: tt-blaze is only importable inside its own tt-metal checkout (it
needs the -ftt-nttp / -ftt-constinit / -ftt-consteval / -ftt-no-dyninit SFPI flags), so on our
tree `blaze_available()` is False and every entry point returns None. Callers fall back to the
ttnn path, and nothing about the shipping model changes.

Two paths are available:

* ``GLM4_MOE_LITE_BLAZE_Q_STAGE=1`` replaces the real decode chain
  ``q_a projection -> q_a RMSNorm -> q_b projection`` with one Blaze program.  The q_a
  intermediate is gathered and multicast entirely in L1.
* ``GLM4_MOE_LITE_BLAZE_QKV_A=1`` retains the older single-projection experiment.

WHAT THIS RUNS. One `DRAMStreamingMatmul` over the model's own concatenated 2048x1344 q_kv_a
weight -- the same single matmul the ttnn path does -- bracketed by two blaze micro-ops that
convert at the model's tensor boundary:

    TileRowReplicate  ->  DRAMStreamingMatmul(w_q_kv_a)  ->  GatherRowToDRAM
    (row 0 of each 32x32          32 DRAM-bank workers        row 0 of each 32x32
     TILE page -> 1x32 pages,      at W=4                      DRAM output tile)
     replicated per worker)

It does NOT use `GLMQKVAProjection`. That op fuses two separate weights around one shared
activation, which is not what the model carries: the default decode path holds a single
concatenated weight. A concatenated weight shares the activation by construction, in one matmul
instead of two that serialise on the same cores, and N=1344 pads to 2048 (a multiple of
32*banks*W = 1024) instead of 768->1024 plus 576->1024 separately. Fusing shared-input
projections was measured at only ~4% of the win; all of it is the streaming matmul plus the
three boundary fixes, none of which need the two-weight structure.

MEASURED IN ISOLATION: 36.8 us against ttnn's 47.4 us, PCC 0.999947 / 0.999957, at W=4.
See blaze_eval/RESUME_HERE.md for the full arc.

BATCH=1 ONLY. `GatherRowToDRAM` writes one row per destination tile because that is all bs=1
decode needs. A batched destination needs its per-row loop restored, guarded on batch. This
module refuses to run at batch > 1 rather than return a wrong answer quietly.
"""

from __future__ import annotations

import os
import time
from typing import Any

import ttnn
from loguru import logger

_BLAZE: Any = None
_BLAZE_IMPORT_TRIED = False

# Keyed by (id(weight_object), activation buffer address). The activation address is part of the
# key because TileRowReplicate bakes `src.buffer_address()` into a COMPILE-time arg -- a program
# built against one activation buffer silently reads the wrong memory if the allocator hands out
# a different address later. Under trace the address is stable, so this is one entry per layer.
_PROGRAM_CACHE: dict[tuple[int, int], Any] = {}
_Q_STAGE_PROGRAM_CACHE: dict[tuple[int, int], Any] = {}

# Shared across every layer's program on purpose. Layer programs execute sequentially, so they
# can reuse the same named semaphores; a fresh dict per program allocates one mesh-global L1
# semaphore per layer and eventually collides with SDPA's static CB region.
_PROGRAM_SEMAPHORES: dict[str, Any] = {}

# Shared for the same reason, and far more load-bearing: see `_init_shared_scratch`.
_SCRATCH_TENSORS: dict[str, Any] = {}
_SCRATCH_MAPPING: dict[str, dict] = {}
_Q_STAGE_SCRATCH_TENSORS: dict[str, Any] = {}
_Q_STAGE_SCRATCH_MAPPING: dict[str, dict] = {}

# Proof-of-execution counter. A previous session's headline number came from a run where the
# flag was never set and the op never executed; `run_count()` makes that checkable from inside
# the run rather than inferred from the environment.
_RUN_COUNT = 0
_Q_STAGE_RUN_COUNT = 0
_LOGGED_SPEC = False
_Q_STAGE_LOGGED_SPEC = False
_GATE_PCCS: list[tuple[float, float]] = []
_Q_STAGE_GATE_PCCS: list[float] = []


def _try_import_blaze() -> Any:
    global _BLAZE, _BLAZE_IMPORT_TRIED
    if _BLAZE_IMPORT_TRIED:
        return _BLAZE
    _BLAZE_IMPORT_TRIED = True
    try:
        from blaze.fused_program import FusedProgram
        from blaze.blaze_op import Risc
        from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul
        from blaze.ops.dram_streaming_matmul.common import dram_bank_worker_cores
        from blaze.ops.gather_row_to_dram import GatherRowToDRAM
        from blaze.ops.glm_qa_norm_qb_projection import GLMQANormQBProjection
        from blaze.ops.mcast import Mcast, McastGridConfig
        from blaze.ops.tile_row_replicate import TileRowReplicate

        _BLAZE = {
            "FusedProgram": FusedProgram,
            "Risc": Risc,
            "DRAMStreamingMatmul": DRAMStreamingMatmul,
            "GatherRowToDRAM": GatherRowToDRAM,
            "GLMQANormQBProjection": GLMQANormQBProjection,
            "Mcast": Mcast,
            "McastGridConfig": McastGridConfig,
            "TileRowReplicate": TileRowReplicate,
            "bank_worker_cores": dram_bank_worker_cores,
        }
        logger.info("tt-blaze available; GLM blaze ops can be enabled")
    except Exception as exc:  # pragma: no cover - the common case on our tree
        _BLAZE = None
        logger.debug("tt-blaze not available ({}); ttnn paths will be used", type(exc).__name__)
    return _BLAZE


def blaze_available() -> bool:
    return _try_import_blaze() is not None


def blaze_qkv_a_enabled() -> bool:
    """True when the caller should try the blaze q_kv_a path."""
    return os.environ.get("GLM4_MOE_LITE_BLAZE_QKV_A", "").strip() == "1" and blaze_available()


def blaze_q_stage_enabled() -> bool:
    """True when the two-projection Blaze Q stage should replace the ttnn chain."""
    return os.environ.get("GLM4_MOE_LITE_BLAZE_Q_STAGE", "").strip() == "1" and blaze_available()


def workers_per_bank() -> int:
    return max(1, int(os.environ.get("BLAZE_DSM_WORKERS_PER_BANK", "1")))


def run_count() -> int:
    """How many optional Blaze programs have actually been dispatched."""
    return _RUN_COUNT + _Q_STAGE_RUN_COUNT


def qkv_a_gate_enabled() -> bool:
    """GLM4_MOE_LITE_BLAZE_QKV_A_GATE=1: also run the ttnn path and PCC every layer.

    Costs an extra matmul per layer, so this is a correctness run, not a timing run.
    """
    return os.environ.get("GLM4_MOE_LITE_BLAZE_QKV_A_GATE", "").strip() == "1"


def q_stage_gate_enabled() -> bool:
    return os.environ.get("GLM4_MOE_LITE_BLAZE_Q_STAGE_GATE", "").strip() == "1"


def q_stage_shared_scratch_enabled() -> bool:
    """GLM4_MOE_LITE_BLAZE_Q_STAGE_SHARED_SCRATCH=1: remap the stage's CBs into one arena.

    Off by default: binding all 12 CBs into a single 120-core arena wedges the second
    `_build_q_stage_program` call, while the same op chain with ordinary per-CB allocation
    builds and runs (bench_e_fused_stage.py). The arena only exists to bound cross-layer L1
    fragmentation, which a single-stage program does not need.
    """
    return os.environ.get("GLM4_MOE_LITE_BLAZE_Q_STAGE_SHARED_SCRATCH", "").strip() == "1"


def _first_device_torch(t):
    """Torch view of device 0's shard, which is all we need: these tensors are replicated."""
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0] if shards else t).float()


def report_gate(ref, q_a, kv, q_lora_rank: int, kvpe_dim: int, batch: int) -> None:
    """PCC blaze q_a/kv against the ttnn concatenated matmul, on the same activation.

    This is the gate the brief demands: not a torch golden, but the exact shipping op the blaze
    path replaces, on real weights, inside the real model, layer by layer.
    """
    r = _first_device_torch(ref)[..., :batch, :]
    got_q = _first_device_torch(q_a)[..., :batch, :]
    got_kv = _first_device_torch(kv)[..., :batch, :]

    def _pcc(a, b):
        a, b = a.flatten(), b.flatten()
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (a.norm() * b.norm() + 1e-12))

    q_pcc = _pcc(r[..., :q_lora_rank], got_q[..., :q_lora_rank])
    kv_pcc = _pcc(r[..., q_lora_rank : q_lora_rank + kvpe_dim], got_kv[..., :kvpe_dim])
    _GATE_PCCS.append((q_pcc, kv_pcc))
    print(f"BLAZE_GATE layer#{len(_GATE_PCCS) - 1:02d} q_a_pcc={q_pcc:.6f} kv_pcc={kv_pcc:.6f}", flush=True)


def gate_summary() -> str:
    if _Q_STAGE_GATE_PCCS:
        return (
            f"BLAZE_Q_STAGE_GATE SUMMARY layers={len(_Q_STAGE_GATE_PCCS)} "
            f"q min={min(_Q_STAGE_GATE_PCCS):.6f} "
            f"mean={sum(_Q_STAGE_GATE_PCCS)/len(_Q_STAGE_GATE_PCCS):.6f}"
        )
    if not _GATE_PCCS:
        return "BLAZE_GATE: no layers gated"
    q = [p[0] for p in _GATE_PCCS]
    k = [p[1] for p in _GATE_PCCS]
    return (
        f"BLAZE_GATE SUMMARY layers={len(_GATE_PCCS)} "
        f"q_a min={min(q):.6f} mean={sum(q)/len(q):.6f} | kv min={min(k):.6f} mean={sum(k)/len(k):.6f}"
    )


def report_q_stage_gate(ref, got, batch: int) -> None:
    r = _first_device_torch(ref)[..., :batch, :]
    q = _first_device_torch(got)[..., :batch, :]
    r, q = r.flatten(), q.flatten()
    r, q = r - r.mean(), q - q.mean()
    pcc = float((r @ q) / (r.norm() * q.norm() + 1e-12))
    _Q_STAGE_GATE_PCCS.append(pcc)
    print(f"BLAZE_Q_STAGE_GATE layer#{len(_Q_STAGE_GATE_PCCS) - 1:02d} q_pcc={pcc:.6f}", flush=True)


def _pad_up(num: int, multiple: int) -> int:
    rem = num % multiple
    return num if rem == 0 else num + (multiple - rem)


def _shuffle_tensor_tiles(tensor, tile_size: int, num_banks: int):
    """Row-major -> column-major tile order within each DRAM bank shard.

    DRAMStreamingMatmul's reader walks K tiles contiguously per N column. Mirrors
    `_shuffle_tensor_tiles` in blaze's own dram_streaming_matmul test, kept here so weight prep
    does not depend on blaze's test namespace being registered.
    """
    import torch

    orig_shape = tensor.shape
    K, N = orig_shape[-2], orig_shape[-1]
    lcm = tile_size * num_banks
    if N % lcm:
        raise ValueError(f"N={N} must already be padded to a multiple of {lcm}")

    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]
    K_tiles = K // tile_size
    per_N = N // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    t = tensor.reshape(batch_size, K, num_banks, per_N).permute(0, 2, 1, 3).contiguous()
    tiles = t.reshape(-1, K, per_N).reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous().reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    i = torch.arange(num_tiles_per_shard)
    tiles = tiles[:, (i % K_tiles) * per_N_tiles + (i // K_tiles), :, :]

    tiles = tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size).permute(0, 1, 3, 2, 4).contiguous()
    out = tiles.reshape(-1, K, per_N).reshape(batch_size, num_banks, K, per_N)
    return out.permute(0, 2, 1, 3).contiguous().reshape(batch_size, K, N).reshape(*orig_shape)


def _replicate_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None


def prepare_qkv_a_weights(
    device,
    w_q_kv_a_torch_k_n,
    *,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> dict | None:
    """One-time, LOAD-TIME prep of the concatenated q_kv_a weight for the blaze path.

    `w_q_kv_a_torch_k_n` is [K, N] = [hidden, q_lora_rank + kvpe_dim], i.e. TT orientation, not
    HF's [out, in].

    `weight_dtype` must match what the ttnn path holds (the model's `dense_dtype`, bf16 by
    default) or this is not a substitute for it: a bf8_b copy would be a different numerical
    path AND half the DRAM traffic, which would flatter the timing. The standalone 1.29x was
    measured with both sides in bf8_b; at the model's bf16 both sides carry twice the bytes.

    DRAMStreamingMatmul needs its weight DRAM-width-sharded across the banks AND column-major
    tile-shuffled, and `GatherRowToDRAM` needs a persistent destination whose address it can bake
    into a compile-time arg. Both are load-time transforms, so neither costs anything per token.
    Returns None when blaze is unavailable or the flag is off, so callers stay on ttnn.
    """
    if not blaze_qkv_a_enabled():
        return None
    import torch

    b = _try_import_blaze()
    if w_q_kv_a_torch_k_n.ndim != 2:
        raise ValueError(f"expected 2D [K, N] weight, got {tuple(w_q_kv_a_torch_k_n.shape)}")
    k, n = int(w_q_kv_a_torch_k_n.shape[0]), int(w_q_kv_a_torch_k_n.shape[1])

    banks = int(device.dram_grid_size().x)
    w_per_bank = workers_per_bank()
    # A worker's shard must be a whole number of 32-wide tiles, so N pads to 32*banks*W.
    # The concatenated 1344 pads to 2048 at W=4; the two separate projections would have paid
    # 768->1024 and 576->1024 instead.
    n_pad = _pad_up(n, 32 * banks * w_per_bank)

    w = w_q_kv_a_torch_k_n.float()
    if n_pad != n:
        w = torch.nn.functional.pad(w, (0, n_pad - n))
    w = _shuffle_tensor_tiles(w.reshape(1, 1, k, n_pad), 32, banks)

    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, device.dram_grid_size().y - 1))}
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(shard_grid, [k, n_pad // banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mapper = _replicate_mapper(device)
    weights = ttnn.from_torch(
        w,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )
    # Persistent destination: GatherRowToDRAM bakes dst.buffer_address() into a compile-time arg,
    # so this must outlive every program built against it. Zero-filled because the gather writes
    # only row 0 of each tile -- rows 1..31 are never touched again.
    out = ttnn.from_torch(
        torch.zeros(1, 1, ttnn.TILE_SIZE, n_pad),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )
    _ = b  # import side effect only; the ops are resolved at emit time
    print(
        f"BLAZE_QKV_A prep: GLM4_MOE_LITE_BLAZE_QKV_A="
        f"{os.environ.get('GLM4_MOE_LITE_BLAZE_QKV_A', '<unset>')!r} "
        f"BLAZE_DSM_WORKERS_PER_BANK={os.environ.get('BLAZE_DSM_WORKERS_PER_BANK', '<unset>')!r} "
        f"K={k} N={n} N_pad={n_pad} banks={banks} W={w_per_bank} dtype={weight_dtype}",
        flush=True,
    )
    return {"weights": weights, "out": out, "k": k, "n": n, "n_pad": n_pad, "banks": banks}


def _prepare_dsm_weight(device, weight_k_n, *, n_pad: int, weight_dtype):
    """Upload one tile-shuffled, DRAM-width-sharded DSM weight."""
    import torch

    k, n = map(int, weight_k_n.shape)
    if n_pad < n:
        raise ValueError(f"n_pad={n_pad} is smaller than weight N={n}")
    w = weight_k_n.float()
    if n_pad != n:
        w = torch.nn.functional.pad(w, (0, n_pad - n))
    banks = int(device.dram_grid_size().x)
    w = _shuffle_tensor_tiles(w.reshape(1, 1, k, n_pad), 32, banks)
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, device.dram_grid_size().y - 1))}
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(shard_grid, [k, n_pad // banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mapper = _replicate_mapper(device)
    return ttnn.from_torch(
        w,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )


def _persistent_tile_output(device, width: int):
    import torch

    mapper = _replicate_mapper(device)
    return ttnn.from_torch(
        torch.zeros(1, 1, ttnn.TILE_SIZE, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )


def prepare_q_stage_weights(
    device,
    w_q_a_torch_k_n,
    w_q_b_torch_k_n,
    q_a_gamma,
    *,
    weight_dtype: ttnn.DataType,
) -> dict | None:
    """Prepare the two real Q-stage projections once at model load time."""
    if not blaze_q_stage_enabled():
        return None
    import torch

    if w_q_a_torch_k_n.ndim != 2 or w_q_b_torch_k_n.ndim != 2:
        raise ValueError(
            f"Q-stage weights must be 2-D [K,N], got {tuple(w_q_a_torch_k_n.shape)} "
            f"and {tuple(w_q_b_torch_k_n.shape)}"
        )
    hidden, q_rank = map(int, w_q_a_torch_k_n.shape)
    qb_k, qb_n = map(int, w_q_b_torch_k_n.shape)
    if qb_k != q_rank:
        raise ValueError(f"Q-stage chain mismatch: q_a N={q_rank}, q_b K={qb_k}")
    gamma = q_a_gamma.reshape(-1).float()
    if gamma.numel() != q_rank:
        raise ValueError(f"q_a gamma has {gamma.numel()} values, expected {q_rank}")

    banks = int(device.dram_grid_size().x)
    w_per_bank = workers_per_bank()
    q_rank_pad = _pad_up(q_rank, 32 * banks * w_per_bank)
    # Deferred RMSNorm: fold gamma into q_b, and zero-pad the K dimension to the q_a
    # projection's padded output. The intermediate therefore needs no materialized norm.
    qb_folded = gamma[:, None] * w_q_b_torch_k_n.float()
    qb_folded = torch.nn.functional.pad(qb_folded, (0, 0, 0, q_rank_pad - q_rank))
    qb_n_pad = _pad_up(qb_n, 32 * banks * w_per_bank)

    profile_startup = os.environ.get("GLM4_MOE_LITE_BLAZE_STARTUP_PROFILE", "").strip() == "1"

    def prepare_step(name, fn):
        start = time.monotonic()
        if profile_startup:
            print(f"BLAZE_Q_STAGE startup begin {name}", flush=True)
        value = fn()
        if profile_startup:
            print(f"BLAZE_Q_STAGE startup end {name} elapsed_s={time.monotonic() - start:.3f}", flush=True)
        return value

    w_q_a = prepare_step(
        "w_q_a",
        lambda: _prepare_dsm_weight(device, w_q_a_torch_k_n, n_pad=q_rank_pad, weight_dtype=weight_dtype),
    )
    w_q_b = prepare_step(
        "w_q_b",
        lambda: _prepare_dsm_weight(device, qb_folded, n_pad=qb_n_pad, weight_dtype=weight_dtype),
    )
    q_a_staging_shape = prepare_step("q_a_staging_shape", lambda: _persistent_tile_output(device, q_rank_pad))
    out = prepare_step("out", lambda: _persistent_tile_output(device, qb_n_pad))

    prepared = {
        "w_q_a": w_q_a,
        "w_q_b": w_q_b,
        # The first tensor is metadata-only for GatherRowToDRAM(write_to_dram=False);
        # its bytes are never written. The final tensor is the model-visible Q output.
        "q_a_staging_shape": q_a_staging_shape,
        "out": out,
        "hidden": hidden,
        "q_rank": q_rank,
        "q_rank_pad": q_rank_pad,
        "q_width": qb_n,
        "q_width_pad": qb_n_pad,
        "banks": banks,
        "workers_per_bank": w_per_bank,
    }
    print(
        f"BLAZE_Q_STAGE prep: K={hidden} q_rank={q_rank}->{q_rank_pad} "
        f"q_width={qb_n}->{qb_n_pad} banks={banks} W={w_per_bank} dtype={weight_dtype}",
        flush=True,
    )
    return prepared


def _full_grid(device) -> ttnn.CoreRangeSet:
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])


def _probe_scratch_cbs(device, x, prepared) -> list[dict]:
    """Record what `cb_scratch` the program asks for, by building one and throwing it away.

    The sizes are derived inside blaze (read chunk depth, subblock_k, per_core_N, page counts),
    so recomputing them here would be a second source of truth that silently drifts. Building a
    throwaway program and watching the calls cannot drift.
    """
    b = _try_import_blaze()
    fp = b["FusedProgram"]
    recorded: list[dict] = []
    original = fp.cb_scratch

    def spy(self, name, **kwargs):
        recorded.append({"name": name, "size": int(kwargs["num_pages"]) * int(kwargs["page_size"])})
        return original(self, name, **kwargs)

    fp.cb_scratch = spy
    try:
        _build_program(device, x, prepared)
    finally:
        fp.cb_scratch = original
    return recorded


def _init_shared_scratch(device, x, prepared) -> None:
    """One L1 arena for every layer's scratch CBs, replacing one arena per layer.

    THIS IS WHAT MAKES THE INTEGRATION POSSIBLE AT ALL. Each `FusedProgram` otherwise backs its
    own scratch in L1, and 47 live programs do not fit: the 9th layer died with "statically
    allocated circular buffers ... clash with L1 buffers". Layer programs execute strictly
    sequentially, so one arena aliased by all of them is safe -- the same reasoning that lets
    them share `_sem_dict`. Slots are packed without overlap, so no assumption about which CBs
    coexist is needed.
    """
    global _SCRATCH_TENSORS, _SCRATCH_MAPPING
    import torch

    offset = 0
    mapping: dict[str, dict] = {}
    for cb in _probe_scratch_cbs(device, x, prepared):
        mapping[cb["name"]] = {"tensor_name": "scratch", "offset_address": offset, "size_bytes": cb["size"]}
        offset += _pad_up(cb["size"], 64)  # 64 B is Blackhole's NOC/DRAM alignment

    grid = _full_grid(device)
    shard_u32 = (_pad_up(offset, 4096) + 3) // 4
    mapper = _replicate_mapper(device)
    scratch = ttnn.from_torch(
        torch.zeros((grid.num_cores(), shard_u32), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (1, shard_u32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )
    _SCRATCH_TENSORS = {"scratch": scratch}
    _SCRATCH_MAPPING = mapping
    print(
        f"BLAZE_QKV_A shared scratch arena: {offset} B/core over {grid.num_cores()} cores, "
        f"{len(mapping)} CBs -> {sorted(mapping)}",
        flush=True,
    )


def _receiver_core(device, worker_grid) -> ttnn.CoreCoord:
    """A compute core that is not a DRAM-bank worker, for the gather receiver."""
    taken = {(c.x, c.y) for c in ttnn.corerange_to_cores(worker_grid, row_wise=True)}
    grid = device.compute_with_storage_grid_size()
    # Scan from the FAR CORNER, not the origin. `taken` only excludes DRAM-bank workers, so a
    # forward scan returns a low core like (0,0) -- which the compute grid is also using. Triage of
    # the Q-stage deadlock found (0,0) parked in DRAMStreamingMatmul's cb_wait_front while it was
    # also meant to be the mcast sender: a core cannot send the tiles it is itself blocked waiting
    # for, so every consumer starves. bench_e_fused_stage picks (11,9), the far corner, and does not
    # hang. Reverse iteration reproduces that choice.
    for y in range(grid.y - 1, -1, -1):
        for x in range(grid.x - 1, -1, -1):
            if (x, y) not in taken:
                return ttnn.CoreCoord(x, y)
    raise ValueError("no spare core for the GatherRowToDRAM receiver")


def _build_program(device, x, prepared: dict):
    b = _try_import_blaze()
    _, worker_grid = b["bank_worker_cores"](device)
    k, n_pad = prepared["k"], prepared["n_pad"]

    f = b["FusedProgram"](
        kernel=None,
        device=device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="glm_qkv_a",
        _sem_dict=_PROGRAM_SEMAPHORES,
        scratch_tensors=_SCRATCH_TENSORS,
        scratch_mapping=_SCRATCH_MAPPING,
    )
    act = b["TileRowReplicate"].emit(
        f,
        x,
        prefix="glm_qkv_a__input_row",
        cores=worker_grid,
        num_tile_cols=k // 32,
        row=0,
    )
    mm = b["DRAMStreamingMatmul"].emit(
        f,
        act,
        prepared["weights"],
        index=None,
        bias=None,
        out=None,
        prefix="glm_qkv_a__matmul",
        fp32_dest_acc_en=True,
        # subblock_k=1 deadlocks the weight triple-buffer on a mesh at this bf16 K=2048 geometry;
        # 2 is the smallest mesh-safe value and keeps L1 free for the SDPA that follows.
        subblock_k=2,
        fused_activation=None,
        index_offset=0,
        wait_for_out=False,
        pop_index=False,
        pop_act=True,
    )
    b["GatherRowToDRAM"].emit(
        f,
        mm,
        prepared["out"],
        prefix="glm_qkv_a__output",
        receiver=_receiver_core(device, worker_grid),
    )
    logger.info(
        "blaze q_kv_a program built: K={} N_pad={} workers={} act_addr=0x{:x} out_addr=0x{:x}",
        k,
        n_pad,
        len(ttnn.corerange_to_cores(worker_grid, row_wise=True)),
        x.buffer_address(),
        prepared["out"].buffer_address(),
    )
    return f


def qkv_a(device, x, w, q_lora_rank: int, kvpe_dim: int, batch: int):
    """Blaze concatenated q_kv_a for the decode path, or None to fall back to ttnn.

    Returns (q_a, kv) with the same logical widths and shapes the ttnn path produces, or None
    when disabled / unavailable / not prepared, so callers keep the ttnn path unchanged.
    """
    global _RUN_COUNT, _LOGGED_SPEC
    if not blaze_qkv_a_enabled():
        return None
    prepared = getattr(w, "blaze_qkv_a", None)
    if prepared is None:
        return None
    if batch != 1:
        raise ValueError(
            f"blaze q_kv_a is batch=1 only (GatherRowToDRAM writes one row per destination tile); got batch={batch}"
        )
    if x.layout != ttnn.TILE_LAYOUT or tuple(x.get_tile().tile_shape) != (32, 32):
        raise ValueError(f"blaze q_kv_a needs 32x32 TILE activations, got layout={x.layout} tile={x.get_tile()}")
    if x.dtype != ttnn.bfloat16:
        raise ValueError(f"blaze q_kv_a needs a bfloat16 activation, got {x.dtype}")
    if int(x.padded_shape[-1]) != prepared["k"]:
        raise ValueError(f"activation K={int(x.padded_shape[-1])} does not match weight K={prepared['k']}")

    if not _LOGGED_SPEC:
        _LOGGED_SPEC = True
        print(
            f"BLAZE_QKV_A first call site hit: x shape={tuple(x.shape)} padded={tuple(x.padded_shape)} "
            f"dtype={x.dtype} mem={x.memory_config().buffer_type}",
            flush=True,
        )

    if not _SCRATCH_TENSORS:
        _init_shared_scratch(device, x, prepared)

    # Attribution knob, not a shipping mode: pay every allocation this path costs (the load-time
    # weight copy and the 47.6 KB/core L1 arena) but run the ttnn matmul anyway. The difference
    # between this and a plain baseline is the price of the memory, isolated from the op.
    if os.environ.get("GLM4_MOE_LITE_BLAZE_QKV_A_ARENA_ONLY", "").strip() == "1":
        return None

    key = (id(w), int(x.buffer_address()))
    program = _PROGRAM_CACHE.get(key)
    if program is None:
        program = _build_program(device, x, prepared)
        _PROGRAM_CACHE[key] = program

    program.run()
    _RUN_COUNT += 1

    out = prepared["out"]
    q_a = ttnn.slice(out, [0, 0, 0, 0], [1, 1, batch, q_lora_rank])
    kv = ttnn.slice(out, [0, 0, 0, q_lora_rank], [1, 1, batch, q_lora_rank + kvpe_dim])
    return q_a, kv


def _build_q_stage_program(device, x, prepared: dict, *, epsilon: float):
    """Build x -> q_a -> deferred RMSNorm -> q_b as one Blaze program."""

    # CONFIG DIFF vs bench_e_fused_stage_mesh.py: the bench builds the same op chain and does NOT
    # hang, so the difference is in what each hands to blaze. Print the same fields the bench does.
    if not globals().get("_Q_STAGE_CFG_LOGGED"):
        globals()["_Q_STAGE_CFG_LOGGED"] = True
        try:
            _b = _try_import_blaze()
            _wl, _wg = _b["bank_worker_cores"](device)
            _rc = _receiver_core(device, _wg)
            _g = device.compute_with_storage_grid_size()
            print(
                f"QSTAGE_CFG grid={_g.x}x{_g.y} banks={device.dram_grid_size().x} "
                f"W={workers_per_bank()} cores={len(_wl)} workers={[(c.x, c.y) for c in _wl]} "
                f"receiver=({_rc.x},{_rc.y}) k={prepared.get('k')} n_pad={prepared.get('n_pad')} "
                f"k2={prepared.get('k2')} n_pad2={prepared.get('n_pad2')} "
                f"x_shape={tuple(x.padded_shape)} x_mem={x.memory_config().buffer_type}",
                flush=True,
            )
        except Exception as _e:
            print(f"QSTAGE_CFG log failed: {_e}", flush=True)
    b = _try_import_blaze()
    w_per_bank = int(prepared["workers_per_bank"])
    _, worker_grid = b["bank_worker_cores"](device, w_per_bank)
    f = b["FusedProgram"](
        kernel=None,
        device=device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        name="glm_q_stage",
        # DIAGNOSTIC (GLM4_MOE_LITE_BLAZE_Q_STAGE_OWN_SEM=1): give this stage's program its own
        # semaphore dict instead of sharing one across all 47 layers. _alloc_mesh_semaphore applies
        # initial_value on FIRST allocation only, so a shared dict means 47 programs share one
        # semaphore object; captured into a trace, a residual count leaves the Mcast receiver
        # waiting forever -- which is exactly where triage found q_b_proj stalled.
        _sem_dict=({} if os.environ.get("GLM4_MOE_LITE_BLAZE_Q_STAGE_OWN_SEM") == "1" else _PROGRAM_SEMAPHORES),
        scratch_tensors=_Q_STAGE_SCRATCH_TENSORS,
        scratch_mapping=_Q_STAGE_SCRATCH_MAPPING,
    )
    if worker_grid.contains(f.sender):
        raise ValueError("Blaze Q-stage sender overlaps a DRAM-streaming worker")

    act = b["TileRowReplicate"].emit(
        f,
        x,
        prefix="glm_q_stage__input_row",
        cores=worker_grid,
        num_tile_cols=int(prepared["hidden"]) // 32,
        row=0,
    )
    q_a = b["DRAMStreamingMatmul"].emit(
        f,
        act,
        prepared["w_q_a"],
        index=None,
        bias=None,
        out=None,
        prefix="glm_q_stage__q_a",
        fp32_dest_acc_en=True,
        subblock_k=2,
        fused_activation=None,
        index_offset=0,
        wait_for_out=False,
        pop_index=False,
        pop_act=True,
        workers_per_bank=w_per_bank,
    )
    staged = b["GatherRowToDRAM"].emit(
        f,
        q_a,
        prepared["q_a_staging_shape"],
        prefix="glm_q_stage__q_a_l1",
        receiver=f.sender,
        write_to_dram=False,
    )
    replicated_q_a = b["Mcast"].emit(
        f,
        staged,
        prefix="glm_q_stage__q_a_mcast",
        receiver_risc=b["Risc"].DM0,
        mcast_grid_config=b["McastGridConfig"](
            receiving_core_range_set=worker_grid,
            acknowledging_core_range_set=worker_grid,
        ),
    )
    q = b["GLMQANormQBProjection"].emit(
        f,
        replicated_q_a,
        prepared["w_q_b"],
        q_out=None,
        logical_k=int(prepared["q_rank"]),
        epsilon=epsilon,
        prefix="glm_q_stage__norm_q_b",
        fp32_dest_acc_en=True,
        subblock_k=2,
    )
    b["GatherRowToDRAM"].emit(
        f,
        q,
        prepared["out"],
        prefix="glm_q_stage__output",
        receiver=f.sender,
    )
    return f


def _init_q_stage_scratch(device, x, prepared: dict, *, epsilon: float) -> None:
    global _Q_STAGE_SCRATCH_TENSORS, _Q_STAGE_SCRATCH_MAPPING
    import torch

    b = _try_import_blaze()
    fp = b["FusedProgram"]
    recorded: list[dict] = []
    original = fp.cb_scratch

    def spy(self, name, **kwargs):
        recorded.append({"name": name, "size": int(kwargs["num_pages"]) * int(kwargs["page_size"])})
        return original(self, name, **kwargs)

    fp.cb_scratch = spy
    try:
        _build_q_stage_program(device, x, prepared, epsilon=epsilon)
    finally:
        fp.cb_scratch = original

    offset = 0
    mapping: dict[str, dict] = {}
    for cb in recorded:
        mapping[cb["name"]] = {"tensor_name": "scratch", "offset_address": offset, "size_bytes": cb["size"]}
        offset += _pad_up(cb["size"], 64)
    grid = _full_grid(device)
    shard_u32 = (_pad_up(offset, 4096) + 3) // 4
    mapper = _replicate_mapper(device)
    scratch = ttnn.from_torch(
        torch.zeros((grid.num_cores(), shard_u32), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (1, shard_u32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )
    _Q_STAGE_SCRATCH_TENSORS = {"scratch": scratch}
    _Q_STAGE_SCRATCH_MAPPING = mapping
    print(
        f"BLAZE_Q_STAGE shared scratch arena: {offset} B/core over {grid.num_cores()} cores, " f"{len(mapping)} CBs",
        flush=True,
    )


def q_stage(device, x, w, *, batch: int, epsilon: float):
    """Run the fused two-projection Q stage, or return None for the ttnn fallback."""
    global _Q_STAGE_RUN_COUNT, _Q_STAGE_LOGGED_SPEC
    if not blaze_q_stage_enabled():
        return None
    prepared = getattr(w, "blaze_q_stage", None)
    if prepared is None:
        return None
    if batch != 1:
        raise ValueError(f"Blaze Q stage is batch=1 only; got batch={batch}")
    if x.layout != ttnn.TILE_LAYOUT or tuple(x.get_tile().tile_shape) != (32, 32):
        raise ValueError(f"Blaze Q stage needs a 32x32 TILE activation, got {x.layout} / {x.get_tile()}")
    if x.dtype != ttnn.bfloat16 or int(x.padded_shape[-1]) != int(prepared["hidden"]):
        raise ValueError(
            f"Blaze Q stage expected bf16 K={prepared['hidden']}, got {x.dtype} K={int(x.padded_shape[-1])}"
        )
    if not _Q_STAGE_LOGGED_SPEC:
        _Q_STAGE_LOGGED_SPEC = True
        print(
            f"BLAZE_Q_STAGE first call: shape={tuple(x.shape)} padded={tuple(x.padded_shape)} "
            f"W={prepared['workers_per_bank']}",
            flush=True,
        )
    if q_stage_shared_scratch_enabled() and not _Q_STAGE_SCRATCH_TENSORS:
        _init_q_stage_scratch(device, x, prepared, epsilon=epsilon)
    key = (id(w), int(x.buffer_address()))
    program = _Q_STAGE_PROGRAM_CACHE.get(key)
    if program is None:
        program = _build_q_stage_program(device, x, prepared, epsilon=epsilon)
        _Q_STAGE_PROGRAM_CACHE[key] = program
    program.run()
    _Q_STAGE_RUN_COUNT += 1
    return ttnn.slice(prepared["out"], [0, 0, 0, 0], [1, 1, batch, int(prepared["q_width"])])
