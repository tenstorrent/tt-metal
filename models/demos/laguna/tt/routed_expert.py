# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna's routed-expert prefill FFN over the stacked expert weights, as one generic_op (kernels/routed_expert/).

out[start[g] + r, :] = down_e(silu(x[start[g] + r] @ gate_e) * (x[start[g] + r] @ up_e))   for r < counts[g]

for every local expert e (global id g = global_expert_idx[e]) of the chip, where gate_up [1, E, K, 2N] holds expert
e's gate in columns 0..N-1 and its up in N..2N-1, and down is [1, E, N, K]. x is the row-major bf16 dispatch buffer
[R, K]; the result is a fresh TILE bf8 [R, K] buffer whose expert rows the combine reads (other rows stay unwritten).

This is a Python port, for the one configuration Laguna uses, of
ttnn.experimental.deepseek_prefill.unified_routed_expert_moe_stacked as built on jerrywangTT/laguna-s-merge-main
(host wrapper unified_routed_expert_ffn.cpp, device op validate unified_routed_expert_ffn_device_operation.cpp,
program factory device/unified_routed_expert_ffn_program_factory.cpp `create_descriptor`). The kernels under
kernels/routed_expert/ are that branch's kernels byte for byte, except that compute/fused_swiglu.cpp includes
"../adaptive_chunk.hpp" instead of the ttnn copy. `_plan` mirrors create_descriptor statement by statement (same
names, same order; comments say which C++ lines a block mirrors), so the program -- kernels, compile-time args,
defines, CBs, semaphores, per-core and common runtime args, kernel configs, core ranges -- is the one the C++
factory builds for the same call.

The one configuration (anything else asserts): ROW_MAJOR bf16 x (x_is_row_major), stacked packed weights,
SiLU activation, no biases, the full active-token band [0, UINT32_MAX], weights DRAM-interleaved or Laguna's
column-sharded layout (colpage.py), and a stacked-mode row-group plan (always the case on a grid of >= 8 rows).

Dropped from the C++ factory: the TT_ROUTED_EXPERT_ROW_GRID_Y override (nothing in Laguna sets it; the grid rows are
always the device's compute grid rows). Kept: DS_NO_WRITER_MCAST (the factory's documented fallback for the NoC-1
weight multicast; read on every call and part of the cache key, like the C++ getenv).

compute_kernel_config is accepted and ignored, exactly as in the C++ op: its program factory never reads
op.compute_kernel_config and hard-codes the compute config (LoFi, bf16 dst, no dst full sync, no approx mode).

Host cost: everything except buffer addresses is built once per (device, m_tiles, l1_end, tensor specs) and cached
in _PROGRAMS; a call allocates the output, writes the two common-runtime-arg lists (the only address-carrying args)
and runs generic_op.
"""
from __future__ import annotations

import os
from pathlib import Path

import ttnn

from .l1_budget import l1_allocator_base

_KDIR = Path(__file__).resolve().parent / "kernels" / "routed_expert"
READER_KERNEL = _KDIR / "dataflow" / "unified_routed_expert_ffn_reader.cpp"
WRITER_KERNEL = _KDIR / "dataflow" / "unified_routed_expert_ffn_writer.cpp"
COMPUTE_KERNEL = _KDIR / "compute" / "fused_swiglu.cpp"

TILE = 32  # tt::constants::TILE_HEIGHT
# unified_routed_expert_ffn_types.hpp
kCoreGridX = 11
kCoreGridY = 8
MAX_GLOBAL_EXPERTS = 1024  # tt::constants::TILE_HW
U32_MAX = 0xFFFFFFFF  # std::numeric_limits<uint32_t>::max()
# The C++ wrapper's fixed op attributes for unified_routed_expert_moe_stacked
MIN_ACTIVE_TOKENS = 0
MAX_ACTIVE_TOKENS = U32_MAX

# CB index allocation (program factory, anonymous namespace)
CB_IN0_X = 0
CB_IN1_GATE = 1
CB_IN1_UP = 2
CB_IN1_DOWN = 3
CB_GATE_INT = 4
CB_UP_INT = 5
CB_ACTIVATED = 6
CB_PARTIALS_GU = 7
CB_PARTIALS_D = 8
CB_OUT = 9
CB_COUNTS_SCRATCH = 10
CB_IDX_SCRATCH = 11
CB_IN0_DOWN_FULL = 12
CB_PARTIALS_UP = 13
CB_START_SCRATCH = 14
CB_START_SCRATCH_READER = 15
CB_X_RM = 16

# tt::tile_size(datatype_to_dataformat_converter(dtype)) for the formats this program uses
_TILE_BYTES = {
    ttnn.bfloat16: 2048,  # Float16_b
    ttnn.bfloat8_b: 1088,  # Bfp8_b
    ttnn.bfloat4_b: 576,  # Bfp4_b
    ttnn.float32: 4096,
}
_FLOAT16_B_DATUM_BYTES = 2  # tt::datum_size(tt::DataFormat::Float16_b)

# Program-factory constants
_L1_SCRATCH_MARGIN = 48 * 1024
_K_MIN_CHUNK_M_TILES = 16  # adaptive_chunk.hpp kMinChunkMTiles
_ROW_CHUNK = 8  # tile-rows per row-group chunk
_FP32_DEST_ACC_EN = False  # kFp32DestAccEn
_DST_CAPACITY = 4 if _FP32_DEST_ACC_EN else 8

_DEVICE_INFO: dict = {}  # id(device) -> (grid_x, grid_y, l1_size_per_core, l1_base, dram_banks)
_PROGRAMS: dict = {}  # program cache key -> _Program


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _device_info(device):
    """The device constants the factory reads: compute grid, l1_size_per_core(), the L1 allocator base and the
    DRAM bank count (allocator()->get_num_banks(DRAM)). Read once per device object."""
    key = id(device)
    info = _DEVICE_INFO.get(key)
    if info is None:
        g = device.compute_with_storage_grid_size()
        # IDevice::l1_size_per_core() is the SOC descriptor's worker_l1_size, which is also the allocator's
        # get_worker_l1_size() that reports.get_device_info returns (the Python MeshDevice has no l1_size_per_core).
        l1_size = int(ttnn._ttnn.reports.get_device_info(device).worker_l1_size)
        banks = int(ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks)
        info = (int(g.x), int(g.y), l1_size, l1_allocator_base(device), banks)
        _DEVICE_INFO[key] = info
    return info


def _is_dram_interleaved(t) -> bool:
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _is_dram_column_sharded(t) -> bool:
    """device op is_dram_column_sharded: one [1, 1, K, 32] DRAM ND shard per (expert, tile column)."""
    mc = t.memory_config()
    if (
        mc.buffer_type != ttnn.BufferType.DRAM
        or mc.memory_layout != ttnn.TensorMemoryLayout.ND_SHARDED
        or mc.nd_shard_spec is None
    ):
        return False
    shard = list(mc.nd_shard_spec.shard_shape)
    shp = list(t.padded_shape)
    return len(shard) == 4 and shard[0] == 1 and shard[1] == 1 and shard[2] == shp[-2] and shard[3] == TILE


def _validate(x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles):
    """The host wrapper's and UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss's checks that
    apply to Laguna's configuration, plus asserts on the modes this port does not carry."""
    # host wrapper (unified_routed_expert_moe_stacked)
    gu_shape, d_shape = list(gate_up.padded_shape), list(down.padded_shape)
    assert len(gu_shape) == 4 and len(d_shape) == 4, f"stacked gate_up/down must be rank 4 (got {gu_shape}, {d_shape})"
    experts_per_chip = gu_shape[-3]
    assert experts_per_chip > 0, "Need at least one stacked expert per chip"
    # Laguna port: only the row-major (tilize in-op) x path
    assert x.layout == ttnn.ROW_MAJOR_LAYOUT, (
        "routed_expert_stacked supports only Laguna's ROW_MAJOR bf16 dispatch buffer (the C++ op's TILE bf8 in-place "
        f"mode is not ported), got layout {x.layout}"
    )

    # validate_on_program_cache_miss
    assert x.storage_type() == ttnn.StorageType.DEVICE, "x must be on device"
    assert x.dtype == ttnn.bfloat16, f"x must be BFLOAT16 when x_is_row_major, got {x.dtype}"
    assert _is_dram_interleaved(x), "x must be DRAM-interleaved"
    x_logical = list(x.shape)
    assert len(x_logical) >= 2, f"x must have rank >= 2, got rank {len(x_logical)}"
    for i, d in enumerate(x_logical[:-2]):
        assert d == 1, f"x leading dim {i} must be 1, got {d}"
    x_shape = list(x.padded_shape)
    assert (
        x_shape[-1] == gu_shape[-2]
    ), f"x's last dim {x_shape[-1]} must match gate/up's K dim ({gu_shape[-2]}, {gu_shape[-2]})"
    gu_n = gu_shape[-1] // 2  # stacked: gate_up packs gate|up along N
    assert gu_n == d_shape[-2], f"gate/up N ({gu_n}) must equal down K ({d_shape[-2]})"
    assert (
        gu_shape[-3] == experts_per_chip and d_shape[-3] == experts_per_chip
    ), f"stacked gate_up/down must be [1, {experts_per_chip}, K, N] (got {gu_shape}, {d_shape})"
    assert d_shape[-1] == x_shape[-1], f"down N ({d_shape[-1]}) must equal x K ({x_shape[-1]})"
    assert x_shape[-2] % TILE == 0, f"x M ({x_shape[-2]}) must be tile-aligned"
    assert m_tiles > 0, "m_tiles must be > 0"
    assert m_tiles <= x_shape[-2] // TILE, f"m_tiles ({m_tiles}) must be <= x M in tiles ({x_shape[-2] // TILE})"
    for name, w in (("gate_up", gate_up), ("down", down)):
        assert w.storage_type() == ttnn.StorageType.DEVICE, f"{name} must be on device"
        assert w.layout == ttnn.TILE_LAYOUT, f"{name} must be TILE layout"
        assert _is_dram_interleaved(w) or _is_dram_column_sharded(w), (
            f"{name} must be DRAM-interleaved or DRAM ND-sharded with one [1, 1, K, 32] shard per expert column, "
            f"got {w.memory_config()}"
        )
        assert w.dtype in _TILE_BYTES, f"{name} dtype {w.dtype} has no tile size here"
    for name, a in (("counts", counts), ("global_expert_idx_table", global_expert_idx)):
        assert a.storage_type() == ttnn.StorageType.DEVICE, f"{name} must be on device"
        assert a.dtype == ttnn.uint32, f"{name} must be UINT32"
        assert _is_dram_interleaved(a), f"{name} must be DRAM-interleaved"
        n = list(a.shape)[-1]
        assert n <= MAX_GLOBAL_EXPERTS, (
            f"{name} length ({n}) exceeds the maximum supported number of experts ({MAX_GLOBAL_EXPERTS}) -- "
            "the reader fetches only page 0 of this tensor into a fixed-size L1 scratch"
        )
    assert (
        experts_per_chip <= list(global_expert_idx.shape)[-1]
    ), f"experts_per_chip ({experts_per_chip}) must be <= idx_table size ({list(global_expert_idx.shape)[-1]})"
    start = region_offsets
    assert start.storage_type() == ttnn.StorageType.DEVICE, "expert_region_offsets must be on device"
    assert start.dtype == ttnn.uint32, "expert_region_offsets must be UINT32"
    assert start.layout == ttnn.ROW_MAJOR_LAYOUT, f"expert_region_offsets must be ROW_MAJOR layout, got {start.layout}"
    assert _is_dram_interleaved(start), "expert_region_offsets must be DRAM-interleaved"
    s_shape = list(start.shape)
    assert len(s_shape) == 1 or (
        len(s_shape) == 2 and s_shape[0] == 1
    ), f"expert_region_offsets must be 1D or 2D with first dimension == 1, got shape {s_shape}"
    assert (
        s_shape[-1] <= MAX_GLOBAL_EXPERTS
    ), f"expert_region_offsets length ({s_shape[-1]}) exceeds the maximum supported number of experts"
    assert (
        s_shape[-1] == list(counts.shape)[-1]
    ), f"expert_region_offsets length ({s_shape[-1]}) must equal counts length ({list(counts.shape)[-1]})"
    # The output checks (TILE, DRAM-interleaved, bf8 allowed in row-major mode, same rank / last dim / leading dims
    # as x, M tile-aligned and >= x M) hold by construction: the output is allocated here exactly as the C++ wrapper
    # allocates it. The bias checks do not apply (no biases).


def _stacked_expert_bytes(shape, aligned_page_size, is_sharded, experts_per_chip, banks):
    """program factory stacked_expert_bytes: expert e's slice of a stacked [1, E, K, N] weight starts
    e * (K/32 * N/32 / banks) pages past the tensor's address on every bank."""
    tiles = (shape[-2] // TILE) * (shape[-1] // TILE)
    if experts_per_chip > 1:
        assert tiles % banks == 0, f"stacked expert slice ({tiles} tiles) must be a multiple of the {banks} DRAM banks"
        if is_sharded:
            assert (shape[-1] // TILE) % banks == 0, (
                f"stacked column-sharded expert slice ({shape[-1] // TILE} shards) must be a multiple of the "
                f"{banks} DRAM banks"
            )
    return (tiles // banks) * aligned_page_size


def _plan(
    *,
    m_tiles,
    x_shape,
    gu_shape,
    d_shape,
    gu_dtype,
    d_dtype,
    out_dtype,
    out_m,
    counts_entries,
    counts_page,
    idx_entries,
    idx_page,
    start_page,
    gu_page,
    d_page,
    gu_sharded,
    d_sharded,
    grid_x,
    grid_y,
    l1_size,
    l1_reserved,
    l1_end,
    dram_banks,
    writer_mcast_disabled,
):
    """UnifiedRoutedExpertFfnProgramFactory::create_descriptor for op.stacked_packed_weights, op.x_is_row_major,
    Silu, no bias: every number of the program except buffer addresses, NoC coordinates and accessor args. Pure
    Python on ints (no device), so it can be checked on the host."""
    p = {}
    x_is_row_major = True
    experts_per_chip = gu_shape[-3]

    # This expert's M (not x's allocated M); K from x's last dim.
    M_tiles_full = m_tiles
    K_gate_tiles = x_shape[-1] // TILE
    # stacked gate_up packs gate | up along N, so its last dim is twice the hidden size
    N_gate_tiles_full = gu_shape[-1] // TILE // 2
    K_down_tiles = d_shape[-2] // TILE
    N_down_tiles_full = d_shape[-1] // TILE

    # Row groups (stacked weights): GRID_Y = the device's grid rows (TT_ROUTED_EXPERT_ROW_GRID_Y override dropped).
    GRID_X = kCoreGridX
    GRID_Y = grid_y
    big = 4 if GRID_Y % 4 == 0 else (5 if GRID_Y % 5 == 0 else (3 if GRID_Y % 3 == 0 else 2))
    row_group = 1 if m_tiles <= 64 else (2 if m_tiles <= 128 else big)
    assert GRID_Y % row_group == 0, f"row group {row_group} must divide the {GRID_Y} grid rows"
    row_experts = row_group < GRID_Y
    assert row_experts, (
        f"stacked plan with one row group spanning all {GRID_Y} grid rows (row_group {row_group}) is the C++ "
        "factory's non-row-group mode, which this port does not carry"
    )
    M_ROWS = row_group
    chunk_M_tiles = _ROW_CHUNK * row_group  # row_experts ? row_chunk * row_group : kMaxChunkMTiles
    in0_block_w_gu = 16
    assert (
        grid_x >= kCoreGridX and grid_y >= kCoreGridY
    ), f"unified_routed_expert_ffn: expected at least {kCoreGridX}x{kCoreGridY} compute grid, got {grid_x}x{grid_y}"
    per_core_M_max = chunk_M_tiles // M_ROWS
    assert (
        per_core_M_max * M_ROWS == chunk_M_tiles and per_core_M_max >= 1
    ), f"chunk_M_tiles ({chunk_M_tiles}) must be a positive multiple of GRID_Y ({GRID_Y})"
    per_core_M = per_core_M_max

    bias_ts = 0  # no FUSE_BIAS

    while in0_block_w_gu > 1 and K_gate_tiles % in0_block_w_gu != 0:
        in0_block_w_gu -= 1

    per_core_N_gu = _ceil_div(N_gate_tiles_full, GRID_X)
    per_core_N_d = _ceil_div(N_down_tiles_full, GRID_X)
    N_gate_tiles_padded = per_core_N_gu * GRID_X
    K_down_tiles_padded = N_gate_tiles_padded  # down K = gate N

    # Stacked weights read per tile: ND shard width 0 (gu_shard_w == up_shard_w, both checks then trivially hold)
    gu_shard_w = 0
    d_shard_w = 0

    in0_block_w_d = per_core_N_gu
    assert (
        K_down_tiles_padded % in0_block_w_d == 0
    ), f"K_down_tiles_padded ({K_down_tiles_padded}) must be divisible by in0_block_w_d ({in0_block_w_d})"

    # Subblock dims (bf16 dst: 8 tiles)
    gu_out_subblock_h = 1
    gu_sub_w = 1
    for cand in range(_DST_CAPACITY, 0, -1):
        if per_core_N_gu % cand == 0:
            gu_sub_w = cand
            break
    gu_out_subblock_w = gu_sub_w
    assert gu_out_subblock_h * gu_out_subblock_w <= _DST_CAPACITY
    d_out_subblock_h = 1
    d_sub_w = 1
    for cand in range(_DST_CAPACITY, 0, -1):
        if per_core_N_d % cand == 0:
            d_sub_w = cand
            break
    if d_sub_w < 3:  # ragged last subblock below the unpack-bound cliff
        d_sub_w = min(_DST_CAPACITY, per_core_N_d)
    d_out_subblock_w = d_sub_w
    d_in1_num_subblocks = _ceil_div(per_core_N_d, d_out_subblock_w)
    d_out_subblock_w_tail = per_core_N_d - (d_in1_num_subblocks - 1) * d_out_subblock_w
    d_subblocks_exact = d_out_subblock_w_tail == d_out_subblock_w

    cb_out_stage_count = 2
    cb_out_tiles = d_out_subblock_h * (d_out_subblock_w if d_subblocks_exact else per_core_N_d) * cb_out_stage_count

    # data formats / tile sizes
    intermed_dtype = ttnn.bfloat8_b
    partials_dtype = ttnn.bfloat16
    in0_x_dtype = ttnn.bfloat8_b  # x_is_row_major: cb_in0_x holds the tilize output
    in0_x_tile_size = _TILE_BYTES[in0_x_dtype]
    gate_tile_size = _TILE_BYTES[gu_dtype]
    up_tile_size = _TILE_BYTES[gu_dtype]
    down_tile_size = _TILE_BYTES[d_dtype]
    out_tile_size = _TILE_BYTES[out_dtype]
    intermed_tile_size = _TILE_BYTES[intermed_dtype]
    partials_gu_tile_size = _TILE_BYTES[partials_dtype]
    partials_d_tile_size = _TILE_BYTES[partials_dtype]

    def cb_footprint_bytes(M, w_gu):
        total = M * w_gu * (1 if x_is_row_major else 2) * in0_x_tile_size  # cb_in0_x (RM: single-buf)
        if x_is_row_major:
            total += M * w_gu * 2 * partials_gu_tile_size  # cb_x_rm (bf16 staging)
        total += w_gu * per_core_N_gu * 2 * gate_tile_size  # cb_in1_gate
        total += w_gu * per_core_N_gu * 2 * up_tile_size  # cb_in1_up
        total += in0_block_w_d * per_core_N_d * 2 * down_tile_size  # cb_in1_down
        total += M * per_core_N_gu * intermed_tile_size  # cb_gate_intermed
        total += M * per_core_N_gu * intermed_tile_size  # cb_activated
        total += M * per_core_N_gu * partials_gu_tile_size  # cb_mm_partials_gu
        total += M * per_core_N_gu * partials_gu_tile_size  # cb_mm_partials_up
        total += M * per_core_N_d * partials_d_tile_size  # cb_mm_partials_d
        total += cb_out_tiles * out_tile_size  # cb_out
        total += M * in0_block_w_d * 2 * intermed_tile_size  # cb_in0_down_full
        total += (2 * per_core_N_gu + per_core_N_d) * bias_ts
        return total

    # Real per-core L1 for CBs
    assert l1_size > l1_reserved + _L1_SCRATCH_MARGIN, (
        f"unexpected L1 geometry: l1_size_per_core ({l1_size}) <= reserved base ({l1_reserved}) + margin "
        f"({_L1_SCRATCH_MARGIN})"
    )
    l1_end_eff = min(l1_end, l1_size) if l1_end != 0 else l1_size
    assert l1_end_eff > l1_reserved + _L1_SCRATCH_MARGIN, (
        f"unified_routed_expert_ffn: l1_end ({l1_end_eff}) leaves no L1 above the reserved base ({l1_reserved}) + "
        f"margin ({_L1_SCRATCH_MARGIN})"
    )
    l1_budget = l1_end_eff - l1_reserved - _L1_SCRATCH_MARGIN

    def fit_config(requested_M):
        for M in range(requested_M, 0, -1):
            if requested_M % M != 0:
                continue
            for w in range(min(in0_block_w_gu, K_gate_tiles), 0, -1):
                if K_gate_tiles % w == 0 and cb_footprint_bytes(M, w) <= l1_budget:
                    return M, w
        return 0, 0

    fit_M, fit_w = fit_config(per_core_M)
    assert fit_M != 0, (
        "unified_routed_expert_ffn: per-core CBs do not fit in L1 even at the smallest config "
        f"(per_core_M=1, in0_block_w_gu=1): need {cb_footprint_bytes(1, 1)} B but only {l1_budget} B available "
        f"(emb={N_down_tiles_full * TILE}, hidden={N_gate_tiles_full * TILE}, grid {GRID_X}x{GRID_Y})"
    )
    # (The per_core_M doubling, kWidenMMinTilesPerBlock, runs only when !row_experts.)
    per_core_M = fit_M
    in0_block_w_gu = fit_w
    chunk_M_tiles = per_core_M * M_ROWS
    assert (
        K_gate_tiles % in0_block_w_gu == 0
    ), f"K_gate_tiles ({K_gate_tiles}) must be divisible by in0_block_w_gu ({in0_block_w_gu})"

    min_chunk = chunk_M_tiles if chunk_M_tiles < _K_MIN_CHUNK_M_TILES else _K_MIN_CHUNK_M_TILES
    num_chunks = _ceil_div(M_tiles_full, min_chunk)

    # Phase-level numbers
    gu_in0_num_subblocks = per_core_M // gu_out_subblock_h
    gu_in1_num_subblocks = per_core_N_gu // gu_out_subblock_w
    gu_in0_block_num_tiles = per_core_M * in0_block_w_gu
    gu_in0_subblock_num_tiles = gu_out_subblock_h * in0_block_w_gu
    gu_in1_block_num_tiles = in0_block_w_gu * per_core_N_gu
    gu_in1_block_w = per_core_N_gu
    gu_num_blocks = K_gate_tiles // in0_block_w_gu
    gu_out_block_num_tiles = per_core_M * per_core_N_gu

    d_in0_num_subblocks = per_core_M // d_out_subblock_h
    d_in0_block_num_tiles = per_core_M * in0_block_w_d
    d_in0_subblock_num_tiles = d_out_subblock_h * in0_block_w_d
    d_in1_block_num_tiles = in0_block_w_d * per_core_N_d
    d_in1_block_w = per_core_N_d
    d_num_blocks = K_down_tiles_padded // in0_block_w_d
    d_out_block_num_tiles = per_core_M * per_core_N_d

    dst_M_tiles = out_m // TILE

    # -------------------------- semaphores (ids in creation order) --------------------------
    n_sem = 0

    def add_semaphore():
        nonlocal n_sem
        n_sem += 1
        return n_sem - 1

    in1_ready_sem_id = add_semaphore()
    in1_valid_sem_id = add_semaphore()
    in0_ready_sem_id = add_semaphore()
    in0_valid_sem_id = add_semaphore()
    act_ready_sem_id = add_semaphore()
    act_valid_sem_id = add_semaphore()
    up_mode = 2  # kEnableSplitUp: UP_SPLIT
    reader_reads_up = up_mode == 0
    reader_mcasts_up = up_mode in (0, 2)
    counts_valid_sem_id = add_semaphore()
    kWriterMcastsIn1 = (not writer_mcast_disabled) and row_group > 1
    mcast_go_sem_id = add_semaphore() if kWriterMcastsIn1 else 0
    mcast_done_sem_id = add_semaphore() if kWriterMcastsIn1 else 0
    kEnableSplitDown = in0_block_w_d >= 2
    down_split_k = (in0_block_w_d // 2) if kEnableSplitDown else in0_block_w_d
    down_go_sem_id = add_semaphore() if kEnableSplitDown else 0
    down_done_sem_id = add_semaphore() if kEnableSplitDown else 0
    up_go_sem_id = add_semaphore() if up_mode == 2 else 0
    up_done_sem_id = add_semaphore() if up_mode == 2 else 0

    # -------------------------- circular buffers (index, dtype, tiles, page bytes), C++ order --------------------
    cbs = [(CB_IN0_X, in0_x_dtype, gu_in0_block_num_tiles * (1 if x_is_row_major else 2), in0_x_tile_size)]
    if x_is_row_major:
        cbs.append((CB_X_RM, ttnn.bfloat16, gu_in0_block_num_tiles * 2, _TILE_BYTES[ttnn.bfloat16]))
    cbs += [
        (CB_IN1_GATE, gu_dtype, gu_in1_block_num_tiles * 2, gate_tile_size),
        (CB_IN1_UP, gu_dtype, gu_in1_block_num_tiles * 2, up_tile_size),
        (CB_IN1_DOWN, d_dtype, d_in1_block_num_tiles * 2, down_tile_size),
        (CB_GATE_INT, intermed_dtype, gu_out_block_num_tiles, intermed_tile_size),
        (CB_ACTIVATED, intermed_dtype, gu_out_block_num_tiles, intermed_tile_size),
        (CB_PARTIALS_GU, partials_dtype, gu_out_block_num_tiles, partials_gu_tile_size),
        (CB_PARTIALS_UP, partials_dtype, gu_out_block_num_tiles, partials_gu_tile_size),
        (CB_PARTIALS_D, partials_dtype, d_out_block_num_tiles, partials_d_tile_size),
        (CB_OUT, out_dtype, cb_out_tiles, out_tile_size),
        (CB_IN0_DOWN_FULL, intermed_dtype, d_in0_block_num_tiles * 2, intermed_tile_size),
    ]

    def check_ring(cb_name, ring_tiles, granule_tiles):
        assert granule_tiles > 0 and ring_tiles % granule_tiles == 0, (
            f"unified_routed_expert_ffn: {cb_name} ring ({ring_tiles} tiles) is not a whole number of the "
            f"{granule_tiles}-tile granule its kernels push; the FIFO pointer would never land on fifo_limit"
        )

    check_ring("cb_in0_x", gu_in0_block_num_tiles * (1 if x_is_row_major else 2), in0_block_w_gu)
    if x_is_row_major:
        check_ring("cb_x_rm", gu_in0_block_num_tiles * 2, in0_block_w_gu)
    check_ring("cb_gate_intermed", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w)
    check_ring("cb_partials_gu", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w)
    check_ring("cb_partials_up", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w)
    check_ring("cb_activated", gu_out_block_num_tiles, per_core_M * in0_block_w_d)
    d_row_granule = d_out_subblock_h * (d_out_subblock_w if d_subblocks_exact else per_core_N_d)
    check_ring("cb_mm_partials_d", d_out_block_num_tiles, d_row_granule)
    check_ring("cb_out", cb_out_tiles, d_row_granule)
    check_ring("cb_in0_down_full", d_in0_block_num_tiles * 2, d_in0_block_num_tiles)
    check_ring("cb_in1_gate", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles)
    check_ring("cb_in1_up", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles)
    check_ring("cb_in1_down", d_in1_block_num_tiles * 2, d_in1_block_num_tiles)

    # Scratch CBs: one page each, sized to the real per-call requirement
    counts_scratch_bytes = max(counts_page, counts_entries * 4)
    idx_scratch_bytes = max(idx_page, idx_entries * 4)
    cbs.append((CB_COUNTS_SCRATCH, ttnn.uint32, 1, counts_scratch_bytes))
    cbs.append((CB_IDX_SCRATCH, ttnn.uint32, 1, idx_scratch_bytes))
    start_scratch_bytes = max(start_page, counts_entries * 4)
    cbs.append((CB_START_SCRATCH, ttnn.uint32, 1, start_scratch_bytes))
    cbs.append((CB_START_SCRATCH_READER, ttnn.uint32, 1, start_scratch_bytes))

    # -------------------------- compile-time args (before the TensorAccessorArgs) --------------------------
    reader_ct = [
        CB_IN0_X,
        CB_IN1_GATE,
        CB_IN1_UP,
        CB_IN1_DOWN,
        CB_IN0_DOWN_FULL,
        CB_COUNTS_SCRATCH,
        CB_IDX_SCRATCH,
        experts_per_chip,
        per_core_M,
        per_core_N_gu,
        per_core_N_d,
        K_gate_tiles,
        K_down_tiles,
        in0_block_w_gu,
        in0_block_w_d,
        N_gate_tiles_full,
        N_down_tiles_full,
        M_tiles_full,
        num_chunks,
        chunk_M_tiles,
        CB_ACTIVATED,  # 20
        GRID_X,  # 21
        K_down_tiles_padded,  # 22
        int(reader_reads_up),  # 23
        int(reader_mcasts_up),  # 24
        CB_START_SCRATCH_READER,  # 25
        int(x_is_row_major),  # 26
        CB_X_RM,  # 27
        TILE,  # 28 TILE_HEIGHT
        _FLOAT16_B_DATUM_BYTES,  # 29 X_RM_ELEM_BYTES
        down_split_k,  # 30
        int(kWriterMcastsIn1),  # 31
        MIN_ACTIVE_TOKENS,  # 32
        MAX_ACTIVE_TOKENS,  # 33
        gu_shard_w,  # 34
        d_shard_w,  # 35
    ]
    assert len(reader_ct) == 36

    # Defines: STACKED_* (stacked weights), ROW_EXPERTS / ROW_GROUP (row groups)
    gu_expert_bytes = _stacked_expert_bytes(gu_shape, gu_page, gu_sharded, experts_per_chip, dram_banks)
    d_expert_bytes = _stacked_expert_bytes(d_shape, d_page, d_sharded, experts_per_chip, dram_banks)
    assert gu_expert_bytes < 2**32 and d_expert_bytes < 2**32
    stacked_row_defines = [
        ("STACKED_GATE_UP", "1"),
        ("STACKED_GU_EXPERT_BYTES", str(gu_expert_bytes)),
        ("STACKED_D_EXPERT_BYTES", str(d_expert_bytes)),
    ]
    if row_experts:
        stacked_row_defines += [("ROW_EXPERTS", str(GRID_Y)), ("ROW_GROUP", str(row_group))]
    reader_defines = list(stacked_row_defines)  # no FUSE_BIAS

    writer_ct = [
        CB_OUT,  # 0
        per_core_M,  # 1
        per_core_N_gu,  # 2
        per_core_N_d,  # 3
        d_out_subblock_h,  # 4
        d_out_subblock_w,  # 5
        N_gate_tiles_full,  # 6
        N_down_tiles_full,  # 7
        num_chunks,  # 8
        chunk_M_tiles,  # 9
        CB_COUNTS_SCRATCH,  # 10
        CB_IDX_SCRATCH,  # 11
        experts_per_chip,  # 12
        M_tiles_full,  # 13
        dst_M_tiles,  # 14
        CB_START_SCRATCH,  # 15
        CB_IN1_UP,  # 16
        in0_block_w_gu,  # 17
        K_gate_tiles,  # 18
        int(up_mode == 2),  # 19 writer_split_up
        CB_IN1_DOWN,  # 20
        in0_block_w_d,  # 21
        K_down_tiles,  # 22
        d_num_blocks,  # 23
        d_in1_block_num_tiles,  # 24
        down_split_k,  # 25
        CB_IN1_GATE,  # 26
        int(kWriterMcastsIn1),  # 27
        MIN_ACTIVE_TOKENS,  # 28
        MAX_ACTIVE_TOKENS,  # 29
        gu_shard_w,  # 30
        d_shard_w,  # 31
        d_out_subblock_w_tail,  # 32
    ]
    assert len(writer_ct) == 33
    writer_defines = list(stacked_row_defines)

    compute_ct = [
        # gate
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        # up
        in0_block_w_gu,
        gu_in0_num_subblocks,
        gu_in0_block_num_tiles,
        gu_in0_subblock_num_tiles,
        gu_in1_num_subblocks,
        gu_in1_block_num_tiles,
        gu_in1_block_w,
        gu_num_blocks,
        # down
        in0_block_w_d,
        d_in0_num_subblocks,
        d_in0_block_num_tiles,
        d_in0_subblock_num_tiles,
        d_in1_num_subblocks,
        d_in1_block_num_tiles,
        d_in1_block_w,
        d_num_blocks,
        # gate/up out subblock
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_block_num_tiles,
        # down out subblock
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_subblock_w_tail,
        d_out_block_num_tiles,
        # chunk loop control
        num_chunks,
        experts_per_chip,
        chunk_M_tiles,
        int(x_is_row_major),
        K_down_tiles,
        MIN_ACTIVE_TOKENS,
        MAX_ACTIVE_TOKENS,
    ]
    assert len(compute_ct) == 38
    compute_named = [
        ("cb_x_rm", CB_X_RM),
        ("cb_in0_x", CB_IN0_X),
        ("cb_in1_gate", CB_IN1_GATE),
        ("cb_in1_up", CB_IN1_UP),
        ("cb_in1_down", CB_IN1_DOWN),
        ("cb_gate_intermed", CB_GATE_INT),
        ("cb_up_intermed", CB_UP_INT),
        ("cb_activated", CB_ACTIVATED),
        ("cb_in0_down_full", CB_IN0_DOWN_FULL),
        ("cb_mm_partials_gu", CB_PARTIALS_GU),
        ("cb_mm_partials_up", CB_PARTIALS_UP),
        ("cb_mm_partials_d", CB_PARTIALS_D),
        ("cb_out", CB_OUT),
        ("cb_counts_scratch", CB_COUNTS_SCRATCH),
        ("cb_idx_scratch", CB_IDX_SCRATCH),
        ("m_tiles_full", M_tiles_full),
    ]
    compute_defines = [("PACKER_L1_ACC", "1")]
    if row_experts:
        compute_defines += [("ROW_EXPERTS", str(GRID_Y)), ("ROW_GROUP", str(row_group))]
    compute_defines.append(("FP32_DEST_ACC_EN", "1" if _FP32_DEST_ACC_EN else "0"))
    # Silu: no SWIGLU_OAI / SITU_GLU / CLAMPED_SILU_GLU define; no FUSE_BIAS

    p.update(
        experts_per_chip=experts_per_chip,
        GRID_X=GRID_X,
        GRID_Y=GRID_Y,
        row_group=row_group,
        per_core_M=per_core_M,
        in0_block_w_gu=in0_block_w_gu,
        chunk_M_tiles=chunk_M_tiles,
        num_chunks=num_chunks,
        per_core_N_gu=per_core_N_gu,
        per_core_N_d=per_core_N_d,
        N_gate_tiles_full=N_gate_tiles_full,
        N_down_tiles_full=N_down_tiles_full,
        gu_out_subblock_w=gu_out_subblock_w,
        d_out_subblock_w=d_out_subblock_w,
        writer_mcasts_in1=kWriterMcastsIn1,
        dst_M_tiles=dst_M_tiles,
        l1_budget=l1_budget,
        cb_bytes=cb_footprint_bytes(per_core_M, in0_block_w_gu),
        gu_expert_bytes=gu_expert_bytes,
        d_expert_bytes=d_expert_bytes,
        n_semaphores=n_sem,
        sem=dict(
            in1_ready=in1_ready_sem_id,
            in1_valid=in1_valid_sem_id,
            in0_ready=in0_ready_sem_id,
            in0_valid=in0_valid_sem_id,
            act_ready=act_ready_sem_id,
            act_valid=act_valid_sem_id,
            counts_valid=counts_valid_sem_id,
            mcast_go=mcast_go_sem_id,
            mcast_done=mcast_done_sem_id,
            down_go=down_go_sem_id,
            down_done=down_done_sem_id,
            up_go=up_go_sem_id,
            up_done=up_done_sem_id,
        ),
        cbs=cbs,
        reader_ct=reader_ct,
        writer_ct=writer_ct,
        compute_ct=compute_ct,
        compute_named=compute_named,
        reader_defines=reader_defines,
        writer_defines=writer_defines,
        compute_defines=compute_defines,
    )
    return p


def _runtime_args(p, noc):
    """Per-core runtime args of the three kernels (create_descriptor's per-core loop). noc(x, y) -> (nx, ny) is
    worker_core_from_logical_core. Returns three lists of ((gx, gy), args) in the C++ core order (gy-major)."""
    GRID_X, GRID_Y, row_group = p["GRID_X"], p["GRID_Y"], p["row_group"]
    per_core_N_gu, per_core_N_d = p["per_core_N_gu"], p["per_core_N_d"]
    N_gate_tiles_full, N_down_tiles_full = p["N_gate_tiles_full"], p["N_down_tiles_full"]
    gu_w, d_w = p["gu_out_subblock_w"], p["d_out_subblock_w"]
    s = p["sem"]
    row_experts = row_group < GRID_Y
    grid_first, grid_last = noc(0, 0), noc(GRID_X - 1, GRID_Y - 1)
    reader_rt, writer_rt, compute_rt = [], [], []
    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            my_mt = gy % row_group
            group_y0 = gy - my_mt  # first grid row of this core's expert group
            my_nt_gu = gx
            my_nt_d = gx

            # DRAM-read sender placement: weight sender of column gx -> row group_y0 + gx % row_group; x sender of
            # row gy -> column (gy + 1) % GRID_X.
            in1_sender_row = group_y0 + gx % row_group
            in0_sender_col = (gy + 1) % GRID_X
            assert row_experts or not (
                gy == in1_sender_row and gx == in0_sender_col
            ), f"sender placement collision at ({gx}, {gy}): a core must not own both the weight and x DRAM read"
            is_in1_sender = gy == in1_sender_row
            sender_noc = noc(gx, in1_sender_row)
            first_recv_noc = noc(gx, group_y0)
            last_recv_noc = noc(gx, group_y0 + row_group - 1)
            in1_num_receivers = row_group - 1

            is_in0_sender = gx == in0_sender_col
            in0_sender_noc = noc(in0_sender_col, gy)
            in0_first_recv_noc = noc(0, gy)
            in0_last_recv_noc = noc(GRID_X - 1, gy)
            in0_num_receivers = GRID_X - 1

            reader_args = [
                my_mt,
                my_nt_gu,
                my_nt_d,
                int(is_in1_sender),
                s["in1_ready"],
                s["in1_valid"],
                in1_num_receivers,
                first_recv_noc[0],
                first_recv_noc[1],
                last_recv_noc[0],
                last_recv_noc[1],
                sender_noc[0],
                sender_noc[1],
                int(is_in0_sender),
                s["in0_ready"],
                s["in0_valid"],
                in0_num_receivers,
                in0_first_recv_noc[0],
                in0_first_recv_noc[1],
                in0_last_recv_noc[0],
                in0_last_recv_noc[1],
                in0_sender_noc[0],
                in0_sender_noc[1],
                s["act_ready"],
                s["act_valid"],
                s["up_go"],
                s["up_done"],
                # COUNTS_BCAST (7)
                int(gx == 0 and gy == 0),
                grid_first[0],
                grid_first[1],
                grid_last[0],
                grid_last[1],
                s["counts_valid"],
                GRID_X * GRID_Y - 1,
                # DOWN_SPLIT go/done, IN1_WRITER_MCAST go/done
                s["down_go"],
                s["down_done"],
                s["mcast_go"],
                s["mcast_done"],
            ]
            # M-row NoC coord table
            for gxi in range(GRID_X):
                reader_args += list(noc(gxi, gy))
            reader_rt.append(((gx, gy), reader_args))

            writer_rt.append(
                (
                    (gx, gy),
                    [
                        my_mt,
                        my_nt_d,
                        my_nt_gu,
                        int(is_in1_sender),
                        s["up_go"],
                        s["up_done"],
                        s["down_go"],
                        s["down_done"],
                        first_recv_noc[0],
                        first_recv_noc[1],
                        last_recv_noc[0],
                        last_recv_noc[1],
                        in1_num_receivers,
                        s["in1_ready"],
                        s["in1_valid"],
                        s["mcast_go"],
                        s["mcast_done"],
                    ],
                )
            )

            valid_n_gu = min(
                per_core_N_gu,
                N_gate_tiles_full - my_nt_gu * per_core_N_gu if N_gate_tiles_full > my_nt_gu * per_core_N_gu else 0,
            )
            valid_n_d = min(
                per_core_N_d,
                N_down_tiles_full - my_nt_d * per_core_N_d if N_down_tiles_full > my_nt_d * per_core_N_d else 0,
            )
            compute_rt.append(((gx, gy), [_ceil_div(valid_n_gu, gu_w), _ceil_div(valid_n_d, d_w)]))
    return reader_rt, writer_rt, compute_rt


class _Program:
    """The cached descriptor objects of one plan. A call only sets the reader's and writer's common runtime args."""

    def __init__(self, p, device, x, counts, idx, start, gate_up, down, out):
        self.plan = p
        self.experts_per_chip = p["experts_per_chip"]
        core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(p["GRID_X"] - 1, p["GRID_Y"] - 1))]
        )
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=core_range_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx_, data_format=dtype, page_size=page)],
            )
            for idx_, dtype, tiles, page in p["cbs"]
        ]
        self.semaphores = [
            ttnn.SemaphoreDescriptor(id=i, core_type=ttnn.CoreType.WORKER, core_ranges=core_range_set, initial_value=0)
            for i in range(p["n_semaphores"])
        ]

        noc_cache = {}

        def noc(gx, gy):
            c = noc_cache.get((gx, gy))
            if c is None:
                w = device.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy))
                c = noc_cache[(gx, gy)] = (int(w.x), int(w.y))
            return c

        reader_rt, writer_rt, compute_rt = _runtime_args(p, noc)

        def acc(t):
            return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        # reader accessors: x, gate, up, down, counts, idx, start (gate and up are both the stacked gate_up)
        reader_ct = list(p["reader_ct"]) + acc(x) + acc(gate_up) + acc(gate_up) + acc(down)
        reader_ct += acc(counts) + acc(idx) + acc(start)
        # writer accessors: out, start, up, down
        writer_ct = list(p["writer_ct"]) + acc(out) + acc(start) + acc(gate_up) + acc(down)

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=_FP32_DEST_ACC_EN,
            dst_full_sync_en=False,
            bfp8_pack_precise=False,
        )
        file_path = ttnn.KernelDescriptor.SourceType.FILE_PATH
        # Kernel order is the factory's: reader, writer, compute.
        self.reader = ttnn.KernelDescriptor(
            kernel_source=str(READER_KERNEL),
            source_type=file_path,
            core_ranges=core_range_set,
            compile_time_args=reader_ct,
            defines=p["reader_defines"],
            runtime_args=reader_rt,
            common_runtime_args=[0] * (4 + 3 * self.experts_per_chip),
            config=ttnn.ReaderConfigDescriptor(),
        )
        self.writer = ttnn.KernelDescriptor(
            kernel_source=str(WRITER_KERNEL),
            source_type=file_path,
            core_ranges=core_range_set,
            compile_time_args=writer_ct,
            defines=p["writer_defines"],
            runtime_args=writer_rt,
            common_runtime_args=[0] * (2 + 2 * self.experts_per_chip),
            config=ttnn.WriterConfigDescriptor(),
        )
        self.compute = ttnn.KernelDescriptor(
            kernel_source=str(COMPUTE_KERNEL),
            source_type=file_path,
            core_ranges=core_range_set,
            compile_time_args=list(p["compute_ct"]),
            named_compile_time_args=list(p["compute_named"]),
            defines=p["compute_defines"],
            runtime_args=compute_rt,
            config=compute_cfg,
        )

    def descriptor(self, x, counts, idx, start, gate_up, down, out):
        E = self.experts_per_chip
        gu_a, d_a = gate_up.buffer_address(), down.buffer_address()
        start_a = start.buffer_address()
        # Reader: x, counts, index table, region offsets, then gate[E], up[E], down[E]. Every stacked expert's entry
        # is the tensor's own address; the kernels add e * STACKED_*_EXPERT_BYTES (the virtual base of expert e).
        self.reader.common_runtime_args = (
            [
                x.buffer_address(),
                counts.buffer_address(),
                idx.buffer_address(),
                start_a,
            ]
            + [gu_a] * (2 * E)
            + [d_a] * E
        )
        # Writer: output, region offsets, then up[E], down[E].
        self.writer.common_runtime_args = [out.buffer_address(), start_a] + [gu_a] * E + [d_a] * E
        # ProgramDescriptor copies the kernel descriptors, so the cached ones can be re-pointed on the next call.
        return ttnn.ProgramDescriptor(
            kernels=[self.reader, self.writer, self.compute], semaphores=self.semaphores, cbs=self.cbs
        )


def _cache_key(device, x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles, l1_end):
    def spec(t):
        return (tuple(t.padded_shape), tuple(t.shape), t.dtype, t.layout, t.memory_config())

    return (
        id(device),
        m_tiles,
        l1_end,
        spec(x),
        spec(region_offsets),
        spec(counts),
        spec(global_expert_idx),
        spec(gate_up),
        spec(down),
        "DS_NO_WRITER_MCAST" in os.environ,
    )


def _plan_for(device, x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles, l1_end):
    """_plan on these tensors and this device (the output is the TILE bf8 copy of x's shape the wrapper allocates)."""
    grid_x, grid_y, l1_size, l1_base, banks = _device_info(device)
    return _plan(
        m_tiles=m_tiles,
        x_shape=list(x.padded_shape),
        gu_shape=list(gate_up.padded_shape),
        d_shape=list(down.padded_shape),
        gu_dtype=gate_up.dtype,
        d_dtype=down.dtype,
        out_dtype=ttnn.bfloat8_b,
        out_m=_ceil_div(list(x.padded_shape)[-2], TILE) * TILE,  # the TILE output's padded M
        counts_entries=list(counts.shape)[-1],
        counts_page=counts.buffer_aligned_page_size(),
        idx_entries=list(global_expert_idx.shape)[-1],
        idx_page=global_expert_idx.buffer_aligned_page_size(),
        start_page=region_offsets.buffer_aligned_page_size(),
        gu_page=gate_up.buffer_aligned_page_size(),
        d_page=down.buffer_aligned_page_size(),
        gu_sharded=gate_up.memory_config().is_sharded(),
        d_sharded=down.memory_config().is_sharded(),
        grid_x=grid_x,
        grid_y=grid_y,
        l1_size=l1_size,
        l1_reserved=l1_base,
        l1_end=l1_end,
        dram_banks=banks,
        writer_mcast_disabled="DS_NO_WRITER_MCAST" in os.environ,
    )


def routed_expert_stacked(
    x,
    region_offsets,
    counts,
    global_expert_idx,
    gate_up,
    down,
    max_dispatched_tokens_per_expert,
    compute_kernel_config=None,
    l1_end=0,
):
    """Drop-in for ttnn.experimental.deepseek_prefill.unified_routed_expert_moe_stacked(x, region_offsets, counts,
    global_expert_idx, gate_up, down, max_dispatched_tokens_per_expert, compute_kernel_config=..., l1_end=...) in
    Laguna's configuration (see the module docstring). Returns the fresh TILE bf8 DRAM output [R, K]."""
    del compute_kernel_config  # unused, as in the C++ op (its factory hard-codes the compute config)
    device = x.device()
    m_tiles = (int(max_dispatched_tokens_per_expert) + 31) // 32
    l1_end = int(l1_end)
    # host wrapper: a ROW_MAJOR x gets a fresh TILE bf8 DRAM-interleaved output of x's logical shape
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    key = _cache_key(device, x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles, l1_end)
    prog = _PROGRAMS.get(key)
    if prog is None:
        _validate(x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles)
        p = _plan_for(device, x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles, l1_end)
        assert out.dtype == ttnn.bfloat8_b and list(out.padded_shape)[-2] == p["dst_M_tiles"] * TILE
        prog = _Program(p, device, x, counts, global_expert_idx, region_offsets, gate_up, down, out)
        _PROGRAMS[key] = prog
    desc = prog.descriptor(x, counts, global_expert_idx, region_offsets, gate_up, down, out)
    ttnn.generic_op([x, region_offsets, counts, global_expert_idx, gate_up, down, out], desc)
    return out


def routed_expert_plan(
    x, region_offsets, counts, global_expert_idx, gate_up, down, max_dispatched_tokens_per_expert, l1_end=0
):
    """The host-side plan (block sizes, CB list, compile-time args, defines, ...) that routed_expert_stacked runs for
    these tensors; for tests."""
    m_tiles = (int(max_dispatched_tokens_per_expert) + 31) // 32
    _validate(x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles)
    return _plan_for(x.device(), x, region_offsets, counts, global_expert_idx, gate_up, down, m_tiles, int(l1_end))
