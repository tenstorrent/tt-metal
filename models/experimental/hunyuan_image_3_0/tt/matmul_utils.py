# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# L1 matmul helpers for the HunyuanImage-3.0 TT port.
#
# Hunyuan weights are loaded as DRAM interleaved TILE tensors. For large-M
# prefill (backbone) we use 2D multicast + L1 interleaved output. For small-M
# ops (patch_embed emb_layers) we use WIDTH_SHARDED DRAM weights, WIDTH_SHARDED
# L1 activations, and MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig so
# Tracy shows in0:width_sharded instead of dram_interleaved.
# Set HY_L1_SHARDED_MATMUL=0 to fall back to plain DRAM linear.

import math
import os

import ttnn
from models.common.utility_functions import is_blackhole, nearest_32
from models.demos.blackhole.qwen36.tt.tp_common import _find_grid, _find_largest_divisor

TILE_SIZE = 32


def _l1_sharded_matmul_enabled() -> bool:
    return os.environ.get("HY_L1_SHARDED_MATMUL", "1") != "0"


def _roundup(a: int, b: int) -> int:
    return b * math.ceil(a / b)


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int) -> int:
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _prefill_grid_default() -> tuple[int, int]:
    return (8, 10) if is_blackhole() else (8, 8)


def infer_matmul_mkn(x: ttnn.Tensor, weight: ttnn.Tensor) -> tuple[int, int, int]:
    """Infer (M, K, N) for x @ weight with x [*, K], weight [K, N]."""
    w_shape = list(weight.shape)
    k = int(w_shape[-2])
    n = int(w_shape[-1])
    x_shape = list(x.shape)
    m = 1
    for d in x_shape[:-1]:
        m *= int(d)
    return m, k, n


def dram_width_sharded_weight_mem_config(device, k: int, n: int) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED DRAM config for a [k, n] linear weight (device dram grid)."""
    dram_gs = device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_gs.x - 1, dram_gs.y - 1))})
    padded_n = _roundup(n, TILE_SIZE * dram_gs.x)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, padded_n // dram_gs.x), ttnn.ShardOrientation.ROW_MAJOR),
    )


def width_sharded_act_mem_config(k: int) -> tuple[ttnn.MemoryConfig, ttnn.CoreGrid, int]:
    """WIDTH_SHARDED L1 activation config for a [*, k] tensor (decode-style)."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    grid = ttnn.CoreGrid(x=cols, y=rows)
    num_cores = grid.num_cores
    mem_config = ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, k // num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return mem_config, grid, num_cores


def decode_width_sharded_matmul_program_config(m: int, k: int, n: int, num_cores: int):
    k_tiles_per_core = max(1, (k // TILE_SIZE) // num_cores)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k_tiles_per_core),
        per_core_M=max(1, math.ceil(m / TILE_SIZE)),
        per_core_N=max(1, math.ceil(n / (TILE_SIZE * num_cores))),
        fused_activation=None,
    )


def prefill_matmul_program_config(
    m: int, k: int, n: int, grid_size: tuple[int, int] | None = None
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    if grid_size is None:
        grid_size = _prefill_grid_default()
    per_core_m = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_n = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_n, out_subblock_h)

    k_tiles = math.ceil(k / TILE_SIZE)
    in0_block_w = min(4, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# Final-layer ResBlock AdaGN emb linear (M=32, K=4096, N=2048=2×1024).
# Filled by tests/perf/test_final_layer_emb_matmul_sweep.py → apply winner here.
_FINAL_LAYER_EMB_MATMUL_GRID: tuple[int, int] | None = (8, 2)  # from summary best_grid


def final_layer_emb_matmul_grid() -> tuple[int, int] | None:
    """Optional grid override for HunyuanTtUNetUp ResBlock emb linear."""
    raw = os.environ.get("HY_FINAL_LAYER_EMB_GRID")
    if raw:
        a, b = raw.lower().split("x")
        return (int(a), int(b))
    return _FINAL_LAYER_EMB_MATMUL_GRID


def _fit_1d_matmul_grid(device, k: int, n: int) -> tuple[tuple[int, int], int, int, int]:
    """Pick a rectangular grid for 1D-mcast matmul (resnet50 fc pattern).

    Returns (grid_size, num_cores, per_core_N, in0_block_w). Both K and N must
    divide evenly across cores in tile units.
    """
    dev_grid = device.compute_with_storage_grid_size()
    k_tiles = max(1, k // TILE_SIZE)
    n_tiles = max(1, n // TILE_SIZE)
    best_gx, best_gy, best_nc = 1, 1, 1
    for gy in range(1, dev_grid.y + 1):
        for gx in range(1, dev_grid.x + 1):
            nc = gx * gy
            if k_tiles % nc == 0 and n_tiles % nc == 0 and nc > best_nc:
                best_gx, best_gy, best_nc = gx, gy, nc
    per_core_N = n_tiles // best_nc
    kt_per_core = k_tiles // best_nc
    in0_block_w = _find_largest_divisor(kt_per_core) if kt_per_core > 0 else 1
    return (best_gx, best_gy), best_nc, per_core_N, in0_block_w


def _width_sharded_output_mem_config(m: int, n: int, grid_size: tuple[int, int]) -> ttnn.MemoryConfig:
    """Shape-specific L1 WIDTH_SHARDED output for 1D-mcast matmul."""
    num_cores = grid_size[0] * grid_size[1]
    return ttnn.create_sharded_memory_config_(
        [nearest_32(m), n // num_cores],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_act_mem_config_from_tensor(x: ttnn.Tensor, grid_size: tuple[int, int]) -> ttnn.MemoryConfig:
    """Build L1 WIDTH_SHARDED input memcfg (resnet50-linear / create_sharded_memory_config_)."""
    shape = list(x.shape)
    m = int(shape[-2])
    k = int(shape[-1])
    num_cores = grid_size[0] * grid_size[1]
    return ttnn.create_sharded_memory_config_(
        [nearest_32(m), k // num_cores],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def matmul_1d_program_config(
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    *,
    per_core_N: int | None = None,
    in0_block_w: int | None = None,
):
    num_cores = grid_size[0] * grid_size[1]
    if per_core_N is None or in0_block_w is None:
        k_tiles = max(1, k // TILE_SIZE)
        n_tiles = max(1, n // TILE_SIZE)
        per_core_N = per_core_N or max(1, n_tiles // num_cores)
        kt_per_core = max(1, k_tiles // num_cores)
        in0_block_w = in0_block_w or _find_largest_divisor(kt_per_core)
    m_padded = nearest_32(m)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(m_padded / TILE_SIZE)),
        per_core_N=per_core_N,
        mcast_in0=True,
        fuse_batch=m_padded <= TILE_SIZE,
        fused_activation=None,
    )


def _width_shard_specs_match(a: ttnn.MemoryConfig, b: ttnn.MemoryConfig) -> bool:
    if a.memory_layout != b.memory_layout or a.buffer_type != b.buffer_type:
        return False
    try:
        sa, sb = a.shard_spec, b.shard_spec
        return sa is not None and sb is not None and sa == sb
    except Exception:
        return False


def _ensure_width_sharded_act(
    x: ttnn.Tensor,
    act_mc: ttnn.MemoryConfig,
) -> tuple[ttnn.Tensor, bool]:
    """Pad M→32 if needed and WIDTH_SHARD for 1D-mcast. Returns (x_sh, owns_tensor).

    When ``x`` is already WIDTH_SHARDED with M≥32 and a matching shard spec, returns
    it as-is (``owns_tensor=False``) so callers can leave resident embeddings alive.
    Otherwise consumes ``x`` (frees pad / I2S sources) and returns a fresh shard.
    """
    m_dim = int(list(x.shape)[-2])
    already_ws = ttnn.is_sharded(x) and x.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    if already_ws and m_dim >= TILE_SIZE and _width_shard_specs_match(x.memory_config(), act_mc):
        return x, False

    if already_ws:
        prev = x
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if prev is not x:
            ttnn.deallocate(prev)

    m_dim = int(list(x.shape)[-2])
    if m_dim < TILE_SIZE:
        # Pad M (the -2 dim) up to a full tile. Build the pad spec to match the
        # tensor's actual rank — the decode lm_head path feeds a 3D [B, 1, H] slice,
        # while the backbone linears feed 4D; ttnn.pad requires len(spec) == rank.
        rank = len(x.shape)
        pad_spec = [(0, 0)] * rank
        pad_spec[-2] = (0, TILE_SIZE - m_dim)
        padded = ttnn.pad(x, pad_spec, 0.0)
        if padded is not x:
            ttnn.deallocate(x)
        x = padded

    x_sh = ttnn.interleaved_to_sharded(x, act_mc)
    ttnn.deallocate(x)
    return x_sh, True


def spill_resident_emb_to_dram(x: ttnn.Tensor) -> ttnn.Tensor:
    """Move WIDTH_SHARDED resident t_emb to DRAM interleaved (keep M=32 pad).

    L1-resident emb must not survive across backbone MoE: expert
    ``l1_sharded_linear`` CBs clash with WIDTH_SHARDED emb on the same cores.
    ResBlock still skips FillPad when M≥32 and only pays InterleavedToSharded.
    """
    if not ttnn.is_sharded(x):
        return x
    out = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if out is not x:
        ttnn.deallocate(x)
    return out


def reshard_width_act_for_next_linear(
    x: ttnn.Tensor,
    *,
    next_n: int,
    device=None,
) -> ttnn.Tensor:
    """Reshard WIDTH_SHARDED ``[1,1,M,K]`` to the act layout for weight ``[K, next_n]``.

    Used so TimestepEmbedder's resident t_emb matches ResBlock emb_layers and
    the ResBlock linear can skip both FillPad and InterleavedToSharded.
    """
    if not ttnn.is_sharded(x):
        return x
    k = int(list(x.shape)[-1])
    m_work = int(list(x.shape)[-2])
    dev = device if device is not None else x.device()
    grid_size, _, _, _ = _fit_1d_matmul_grid(dev, k, next_n)
    num_cores = grid_size[0] * grid_size[1]
    act_mc = ttnn.create_sharded_memory_config_(
        [m_work, k // num_cores],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )
    if _width_shard_specs_match(x.memory_config(), act_mc):
        return x
    prev = x
    mid = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if prev is not mid:
        ttnn.deallocate(prev)
    out = ttnn.interleaved_to_sharded(mid, act_mc)
    ttnn.deallocate(mid)
    return out


def act_width_sharded_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor | None = None,
    batch_rows: int | None = None,
    compute_kernel_config=None,
    grid_size: tuple[int, int] | None = None,
    device=None,
    keep_sharded_output: bool = False,
) -> ttnn.Tensor:
    """Width-sharded L1 activation + interleaved weight (Tracy in0:width_sharded).

    If ``x`` is already WIDTH_SHARDED with M padded to 32 and the right shard
    layout, skip FillPad + InterleavedToSharded. When ``keep_sharded_output`` is
    True, leave the WIDTH_SHARDED M=32 result in L1 (for resident t_emb).
    Pass ``batch_rows`` explicitly when ``x`` may be M-padded (resident emb).
    """
    if not _l1_sharded_matmul_enabled():
        kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": compute_kernel_config}
        if bias is not None:
            kwargs["bias"] = bias
        return ttnn.linear(x, weight, **kwargs)

    _, k, n = infer_matmul_mkn(x, weight)
    if batch_rows is None:
        batch_rows = int(list(x.shape)[-2])

    dev = device if device is not None else weight.device()
    if grid_size is None:
        grid_size, _, per_core_N, in0_block_w = _fit_1d_matmul_grid(dev, k, n)
    else:
        per_core_N = None
        in0_block_w = None

    m_dim = int(list(x.shape)[-2])
    m_work = nearest_32(m_dim) if m_dim < TILE_SIZE else m_dim
    num_cores = grid_size[0] * grid_size[1]
    act_mc = ttnn.create_sharded_memory_config_(
        [m_work, k // num_cores],
        ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )

    x_sh, owns_x_sh = _ensure_width_sharded_act(x, act_mc)
    m_work = int(list(x_sh.shape)[-2])

    out_mc = _width_sharded_output_mem_config(m_work, n, grid_size)
    kwargs = {
        "memory_config": out_mc,
        "program_config": matmul_1d_program_config(
            m_work, k, n, grid_size, per_core_N=per_core_N, in0_block_w=in0_block_w
        ),
        "compute_kernel_config": compute_kernel_config,
    }
    if bias is not None:
        kwargs["bias"] = bias
    out = ttnn.linear(x_sh, weight, **kwargs)
    # Drop owned in0 after matmul. Never deallocate a caller-owned resident emb
    # (owns_x_sh=False), even when keep_sharded_output is False.
    if owns_x_sh and x_sh is not out:
        ttnn.deallocate(x_sh)

    if keep_sharded_output:
        return out

    out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if batch_rows < int(list(out.shape)[-2]):
        # Build the slice spec to match the tensor's rank (lm_head feeds 3D).
        out_rank = len(out.shape)
        begins = [0] * out_rank
        ends = list(out.shape)
        ends[-2] = batch_rows
        ends[-1] = n
        out = ttnn.slice(out, begins, ends)
    return out


def decode_width_sharded_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor | None = None,
    batch_rows: int | None = None,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Small-M linear: WIDTH_SHARDED in0/weight, L1 WIDTH_SHARDED out (decode path)."""
    if not _l1_sharded_matmul_enabled():
        kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": compute_kernel_config}
        if bias is not None:
            kwargs["bias"] = bias
        return ttnn.linear(x, weight, **kwargs)

    m, k, n = infer_matmul_mkn(x, weight)
    if batch_rows is None:
        batch_rows = int(list(x.shape)[-2])

    # DRAM-sharded matmul expects M padded to TILE_SIZE on the batch/seq dim.
    x_shape = list(x.shape)
    m_dim = int(x_shape[-2])
    if m_dim < TILE_SIZE:
        # Build the pad spec to match the tensor's rank (decode lm_head feeds 3D).
        pad_spec = [(0, 0)] * len(x_shape)
        pad_spec[-2] = (0, TILE_SIZE - m_dim)
        x = ttnn.pad(x, pad_spec, 0.0)

    act_mc, _, num_cores = width_sharded_act_mem_config(k)
    x_sh = ttnn.interleaved_to_sharded(x, act_mc)
    ttnn.deallocate(x)

    program_config = decode_width_sharded_matmul_program_config(m, k, n, num_cores)
    kwargs = {
        "memory_config": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        "program_config": program_config,
        "compute_kernel_config": compute_kernel_config,
    }
    if bias is not None:
        kwargs["bias"] = bias
    out = ttnn.linear(x_sh, weight, **kwargs)
    ttnn.deallocate(x_sh)

    out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if batch_rows < TILE_SIZE:
        # Build the slice spec to match the tensor's rank (decode lm_head feeds 3D).
        out_rank = len(out.shape)
        begins = [0] * out_rank
        ends = list(out.shape)
        ends[-2] = batch_rows
        ends[-1] = n
        out = ttnn.slice(out, begins, ends)
    return out


def l1_sharded_linear(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor | None = None,
    dtype=None,
    compute_kernel_config=None,
    program_config=None,
    decode: bool = False,
    batch_rows: int | None = None,
    allow_width_shard: bool = True,
) -> ttnn.Tensor:
    """ttnn.linear with a tuned matmul program config.

    Width-sharded small-M schedules (emb / lm_head) only run when ``program_config``
    is None and ``allow_width_shard=True``. An explicit ``program_config`` is always
    honored (L1-interleaved + MultiCast1D wins over width-shard for attn QKV/o_proj —
    see tests/perf/test_matmul_shard_sweep.py).

    ``allow_width_shard=False`` skips those schedules and, when no config is passed,
    installs ``decode_mm_program_config`` for Mt < 8 (attn o_proj was previously left
    on auto at ~39us; decode_mm is ~25us for 32x2048x4096).
    """
    m, k, n = infer_matmul_mkn(x, weight)
    small_m = math.ceil(m / TILE_SIZE) <= 1
    mt = math.ceil(m / TILE_SIZE)

    # Width-shard paths ignore program_config — only take them when none was passed.
    if (
        program_config is None
        and allow_width_shard
        and (
            decode
            or (
                _l1_sharded_matmul_enabled()
                and weight.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and small_m
            )
        )
    ):
        return decode_width_sharded_linear(
            x, weight, bias=bias, batch_rows=batch_rows, compute_kernel_config=compute_kernel_config
        )

    if (
        program_config is None
        and allow_width_shard
        and _l1_sharded_matmul_enabled()
        and small_m
        and weight.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        return act_width_sharded_linear(
            x, weight, bias=bias, batch_rows=batch_rows, compute_kernel_config=compute_kernel_config
        )

    # Schedule is M-dependent (tests/perf/test_expert_down_sweep.py):
    #   Mt >= 8 → wide_mm 2D-mcast (auto mis-schedules)
    #   Mt <  8 + allow_width_shard=False → decode_mm (attn QKV/o_proj; do NOT
    #     auto-install for arbitrary callers — lm_head N=133120 TT_THROWs on 1D
    #     for Mt>=3, and emb ops stay on the width-shard path above).
    if program_config is None and mt >= 8:
        from .parallel_utils import wide_mm_program_config

        program_config = wide_mm_program_config(x.device(), m, k, n)
    elif program_config is None and mt < 8 and not allow_width_shard:
        from .parallel_utils import decode_mm_program_config

        program_config = decode_mm_program_config(x.device(), m, k, n)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "compute_kernel_config": compute_kernel_config}
    if program_config is not None:
        kwargs["program_config"] = program_config
    if bias is not None:
        kwargs["bias"] = bias
    if dtype is not None:
        kwargs["dtype"] = dtype
    return ttnn.linear(x, weight, **kwargs)


def l1_sharded_matmul(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    compute_kernel_config=None,
    program_config=None,
) -> ttnn.Tensor:
    """ttnn.matmul — L1 only for small-M; large-M stays DRAM (same as l1_sharded_linear)."""
    m, k, n = infer_matmul_mkn(x, weight)
    small_m = math.ceil(m / TILE_SIZE) <= 1
    if not _l1_sharded_matmul_enabled() or not small_m:
        return ttnn.matmul(
            x, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config
        )

    if program_config is None:
        program_config = prefill_matmul_program_config(m, k, n)

    return ttnn.matmul(
        x,
        weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def to_interleaved_if_sharded(x: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """Convert a sharded tensor to interleaved when a downstream op requires it."""
    layout = x.memory_config().memory_layout
    if layout in (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED):
        return ttnn.sharded_to_interleaved(x, memory_config=memory_config)
    if x.memory_config().buffer_type == ttnn.BufferType.L1 and memory_config.buffer_type == ttnn.BufferType.DRAM:
        return ttnn.to_memory_config(x, memory_config)
    return x
