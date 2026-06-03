# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Seamless M4T v2 (grid, tile padding, attention masks)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import ttnn

from models.common.utility_functions import nearest_32


def _mesh_device_for_readback(t: ttnn.Tensor):
    """Best-effort lookup of the MeshDevice associated with a tensor.

    Device tensors carry the device directly. Host tensors (after ``ttnn.from_device``) lose that
    pointer, but the demo / perf tests always call ``ttnn.SetDefaultDevice(mesh_device)`` first,
    so the default device is the right composer target. Returns ``None`` if nothing is set.
    """
    if t.storage_type() == ttnn.StorageType.DEVICE:
        return t.device()
    try:
        return ttnn.GetDefaultDevice()
    except Exception:
        return None


def to_torch_replicated_first_shard(t: ttnn.Tensor) -> Any:
    """Read a replicated TTNN tensor back to torch, returning only the first device's data.

    The demo / generate path makes per-step host readbacks of replicated control tensors (token
    IDs, sequence lengths, vocoder cumsums, T2U duration counts, decoder logits, …). On a
    multi-device mesh every device sees the same data because inputs are replicated and ops are
    deterministic, so all shards are bit-identical and reading one is sufficient.

    Fast path: ``ttnn.to_torch(ttnn.get_device_tensors(t)[0])`` — pulls only shard 0. This is the
    same pattern used by ``models/demos/llama3_70b_galaxy/tt/llama_model.py::process_output_decode``
    and by the devstral2 generator. For per-step decoder logits readback (``[B, 1, V=256k]`` bf16)
    on a 1×4 mesh this cuts the device→host bytes by 4× vs the older ``ConcatMeshToTensor`` path
    and removes the host-side concat + slice. Replicated shards are identical by construction so
    the result matches the prior behaviour bit-for-bit.

    Slow-path fallback (kept for tensors where ``get_device_tensors`` is unavailable): use
    ``ConcatMeshToTensor(dim=0)`` and slice off the first per-device chunk.

    Accepts either a device tensor or a host tensor.
    """
    try:
        shards = ttnn.get_device_tensors(t)
        if shards:
            return ttnn.to_torch(shards[0])
    except Exception:
        pass

    dev = _mesh_device_for_readback(t)
    num_devices = 1
    if dev is not None and hasattr(dev, "get_num_devices"):
        try:
            num_devices = int(dev.get_num_devices())
        except Exception:
            num_devices = 1

    if num_devices > 1 and dev is not None:
        composer = ttnn.ConcatMeshToTensor(dev, dim=0)
        out = ttnn.to_torch(t, mesh_composer=composer)
        if out.dim() >= 2 and int(out.shape[0]) == 1 and int(out.shape[1]) % num_devices == 0:
            # Replicated ``[1, L]`` control tensors often concat along width → ``[1, N*L]``.
            out = out[:, : int(out.shape[1]) // num_devices]
        elif out.dim() >= 1 and int(out.shape[0]) >= num_devices:
            out = out[: int(out.shape[0]) // num_devices]
        return out

    host = ttnn.from_device(t) if t.storage_type() == ttnn.StorageType.DEVICE else t
    return ttnn.to_torch(host)


# ``torch.finfo(torch.bfloat16).min`` — the additive-mask "minus infinity" HF uses. Bf16-representable.
NEG_INF = -3.3895313892515355e38

# Tile alignment: TT SDPA must score against tile-aligned key sequences; pad to ``ceil(seq/32)*32``.
TILE = 32


def core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def determine_num_dram_shard_cores(shard_dim: int, max_dram_cores: int) -> int:
    """Largest core count ≤ ``max_dram_cores`` that evenly divides ``shard_dim`` (DRAM width shards)."""
    num_cores = max_dram_cores
    while shard_dim % num_cores != 0:
        assert num_cores > 0, "Unable to find DRAM shard core count"
        num_cores -= 1
    return num_cores


def find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest ``d`` such that ``n % d == 0`` and ``1 <= d <= cap``."""
    cap = max(1, cap)
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def pick_largest_height_shard_nhw_cores(nhw_tiles: int, device: ttnn.Device) -> int:
    """Largest NHW core count that divides ``nhw_tiles`` and fits the compute grid.

    Used for conformer depthwise Conv1d (height-sharded). TTNN auto-shard often picks very
    few cores when L1 is tight; ``override_sharding_config`` + ``core_grid`` can raise core
    count when a large divisor of ``nhw_tiles`` exists (e.g. 992 tiles → 31 or 62 cores).
    """
    grid = device.compute_with_storage_grid_size()
    max_cores = max(1, int(grid.x) * int(grid.y))
    return _largest_divisor_at_most(max(1, nhw_tiles), min(max_cores, max(1, nhw_tiles)))


def _pick_matmul_1d_grid(device: ttnn.Device, *, n_tiles: int) -> tuple[int, int]:
    """Pick a worker grid for 1D-on-N multicast matmul (Devstral-style).

    Chooses the smallest rectangle with at least ``n_tiles`` cores (up to the device grid) so
    ``per_core_N`` stays near 1 and N-parallelism is maximized.
    """
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    max_cores = max_x * max_y
    if n_tiles >= max_cores:
        return max_x, max_y
    for cores in range(n_tiles, max_cores + 1):
        for gx in range(min(max_x, cores), 0, -1):
            if cores % gx == 0 and cores // gx <= max_y:
                return gx, cores // gx
    return max_x, max_y


def _pick_matmul_2d_grid(device: ttnn.Device, *, m_tiles: int, n_tiles: int) -> tuple[int, int]:
    """Pick a 2D mcast worker grid (gx splits N, gy splits M) maximizing cores via exact divisors.

    The naive ``grid_x = cg.x`` choice collapses to a 1-row grid on devices whose grid width does
    not divide ``n_tiles`` (e.g. Blackhole cg.x=11 with n_tiles=16 → 11x1, ~9x slower). Prefer the
    largest ``gx*gy`` where ``gx | n_tiles`` and ``gy | m_tiles``; returns ``(0, 0)`` if no
    multi-core divisor grid exists so the caller can keep its fallback.
    """
    grid = device.compute_with_storage_grid_size()
    best_gx, best_gy, best_cores = 0, 0, 0
    for gx in range(1, int(grid.x) + 1):
        if n_tiles % gx:
            continue
        for gy in range(1, int(grid.y) + 1):
            if m_tiles % gy:
                continue
            cores = gx * gy
            if cores > best_cores or (cores == best_cores and gx > best_gx):
                best_gx, best_gy, best_cores = gx, gy, cores
    return best_gx, best_gy


def _pick_matmul_out_subblock_w(per_core_n: int) -> int:
    for w in (4, 3, 2, 1):
        if per_core_n % w == 0 and w <= 4:
            return w
    return 1


def matmul_multicast_1d_program_config(
    device: ttnn.Device,
    *,
    m: int,
    k: int,
    n: int,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (``mcast_in0=True``), aligned with Devstral2."""
    m_tiles = max(1, math.ceil(m / TILE))
    n_tiles = max(1, math.ceil(n / TILE))
    k_tiles = max(1, math.ceil(k / TILE))
    grid_x, grid_y = _pick_matmul_1d_grid(device, n_tiles=n_tiles)
    num_cores = grid_x * grid_y
    per_core_M = m_tiles
    per_core_N = max(1, math.ceil(n_tiles / num_cores))
    cap = min(
        8,
        max(1, 64 // per_core_M),
        max(1, 128 // per_core_N),
    )
    in0_block_w = _largest_divisor_at_most(k_tiles, cap)
    out_subblock_w = _largest_divisor_at_most(per_core_N, 4)
    out_subblock_h = _largest_divisor_at_most(per_core_M, max(1, 4 // out_subblock_w))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# Prefill rows at or below this use 1D-on-N; longer sequences use 2D multicast (speech-encoder path).
MATMUL_1D_SEQ_THRESHOLD = 128


def matmul_program_config(
    device: ttnn.Device,
    *,
    token_rows: int,
    in_dim: int,
    out_dim: int,
    matmul_1d_seq_threshold: int = MATMUL_1D_SEQ_THRESHOLD,
) -> ttnn.ProgramConfig:
    """Cached-friendly matmul PC factory: 1D multicast for short seq, 2D for long."""
    cg = device.compute_with_storage_grid_size()
    # Match tile-padded K (``matmul_multicast_1d_program_config`` uses ``ceil``).
    k_tiles = max(1, math.ceil(in_dim / TILE))
    in0_block_w = min(4, k_tiles)
    while in0_block_w > 1 and k_tiles % in0_block_w != 0:
        in0_block_w -= 1

    m_tiles = max(1, (token_rows + TILE - 1) // TILE)
    n_tiles = max(1, (out_dim + TILE - 1) // TILE)

    if token_rows <= matmul_1d_seq_threshold:
        return matmul_multicast_1d_program_config(
            device,
            m=max(TILE, m_tiles * TILE),
            k=in_dim,
            n=out_dim,
        )

    # Prefer the largest exact-divisor grid (gx splits N, gy splits M); the naive (cg.x, grid_y)
    # choice collapses to a 1-row grid when cg.x ∤ n_tiles (Blackhole 11x1 → ~6x slower on the
    # cross-attn KV-enc fill). Fall back to the original heuristic if no multi-core divisor grid.
    grid_x, grid_y = _pick_matmul_2d_grid(device, m_tiles=m_tiles, n_tiles=n_tiles)
    if grid_x * grid_y > 1:
        per_core_m = m_tiles // grid_y
        per_core_n = n_tiles // grid_x
    else:
        grid_x = cg.x
        grid_y = min(cg.y, m_tiles)
        while grid_y > 1 and n_tiles % (cg.x * grid_y) != 0:
            grid_y -= 1
        per_core_m = max(1, (m_tiles + grid_y - 1) // grid_y)
        per_core_n = max(1, (n_tiles + cg.x * grid_y - 1) // (cg.x * grid_y))
    out_subblock_w = _pick_matmul_out_subblock_w(per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


# Tuned 2D block-sharded program configs for the text-encoder TP linears (QKV / out_proj /
# fc1 / fc2), keyed by per-device (k, n).  Found by test_matmul_perf_report_sweep.py for
# M=4096, bf16 x bfp8_b -> bf16, LoFi on an 8x8 grid; the value is the winning in0_block_w.
# These beat the ttnn-default config (~8-10 TFLOPs, flagged SLOW in the perf report) by
# 10-27x.  in0 + out live in L1 BLOCK_SHARDED across the 8x8 grid.
_ENCODER_TP_BS_GRID = 8
_ENCODER_TP_BS_IBW = {
    (1024, 768): 4,  # QKV       (k = hidden, n = 3*hidden/tp); per-core Kt/8 = 4
    (256, 1024): 8,  # out_proj  (k = hidden/tp); per-core Kt/8 = 1 -> clamped to 1
    (1024, 2048): 4,  # fc1      (k = hidden, n = ffn_dim/tp)
    (2048, 1024): 8,  # fc2      (k = ffn_dim/tp); per-core Kt/8 = 8
}


def encoder_tp_block_sharded_matmul(
    device: ttnn.Device,
    m: int,
    k: int,
    n: int,
    *,
    fused_activation=None,
):
    """Tuned block-sharded 2D matmul for a text-encoder TP linear.

    Returns ``(program_config, in0_block_sharded_mem, out_block_sharded_mem)`` for a tuned
    ``(k, n)``, or ``None`` if the shape isn't tuned or doesn't tile-fit the 8x8 grid (caller
    then falls back to its default linear).  in0 and out are L1 ``BLOCK_SHARDED`` across the
    grid; the matmul height-shards M over grid rows and width-shards N over grid columns.
    """
    ibw_cap = _ENCODER_TP_BS_IBW.get((k, n))
    if ibw_cap is None:
        return None
    gx = gy = _ENCODER_TP_BS_GRID
    cg = device.compute_with_storage_grid_size()
    if gx > cg.x or gy > cg.y:
        return None
    if m % TILE or k % TILE or n % TILE:
        return None
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    # 2D block-shard divisibility: gy | Mt, gx | Nt, gx | Kt (K split across grid columns).
    if mt % gy or nt % gx or kt % gx:
        return None
    # in0 is BLOCK_SHARDED: each core owns kt_per_core = Kt/gx K-tiles (not full Kt).
    # Sweep ibw against kt_per_core, not global Kt — e.g. out_proj Kt=8, gx=8 -> ibw must be 1.
    kt_per_core = kt // gx
    ibw = _largest_divisor_at_most(kt_per_core, ibw_cap)
    if kt_per_core % ibw:
        return None
    per_core_m = mt // gy
    per_core_n = nt // gx
    # Block-sharded output requires out_subblock_h == 1; widen only along N (h*w <= 4).
    out_subblock_w = _largest_divisor_at_most(per_core_n, 4)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )
    grid = ttnn.CoreGrid(y=gy, x=gx)
    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    out_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, n),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    return program_config, in0_mem, out_mem


def speech_encoder_matmul_program_config(
    device: ttnn.Device,
    *,
    token_rows: int,
    in_dim: int,
    out_dim: int,
) -> ttnn.ProgramConfig:
    """Speech encoder prefill matmul PCs — tuned 1D multicast for hot conformer shapes."""
    if token_rows <= MATMUL_1D_SEQ_THRESHOLD and (
        (in_dim == 1024 and out_dim in (1024, 2048, 4096, 3072)) or (in_dim == 4096 and out_dim == 1024)
    ):
        m_tiles = max(1, (token_rows + TILE - 1) // TILE)
        n_tiles = max(1, (out_dim + TILE - 1) // TILE)
        k_tiles = in_dim // TILE
        grid_x, grid_y = _pick_matmul_1d_grid(device, n_tiles=n_tiles)
        num_cores = grid_x * grid_y
        per_core_m = m_tiles
        per_core_n = max(1, (n_tiles + num_cores - 1) // num_cores)
        in0_block_w = 8 if k_tiles % 8 == 0 else 4
        out_subblock_w = _pick_matmul_out_subblock_w(per_core_n)
        out_subblock_h = _largest_divisor_at_most(per_core_m, max(1, 4 // out_subblock_w))
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    return matmul_program_config(device, token_rows=token_rows, in_dim=in_dim, out_dim=out_dim)


def create_dram_sharded_mem_config(device: ttnn.Device, k: int, n: int) -> Tuple[ttnn.MemoryConfig, int]:
    """WIDTH-sharded DRAM config for linear weight ``[k, n]`` (``n`` may be padded)."""
    dram_cores = dram_matmul_shard_cores(device, k, n)
    assert device.dram_grid_size().y == 1, "DRAM sharding assumes dram grid y == 1"
    padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return mem_config, padded_n


def is_dram_width_sharded(tensor: ttnn.Tensor) -> bool:
    mc = tensor.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def dram_shard_core_count(device: ttnn.Device, n: int) -> int:
    return determine_num_dram_shard_cores(n, int(device.dram_grid_size().x))


def dram_matmul_shard_cores(device: ttnn.Device, k: int, n: int) -> int:
    """Largest DRAM core count valid for both in0 (K) and in1 (N) width shards."""
    max_dram = int(device.dram_grid_size().x)
    padded_n = math.ceil(n / (TILE * max_dram)) * (TILE * max_dram)
    for num_cores in range(max_dram, 0, -1):
        if padded_n % num_cores != 0:
            continue
        if (padded_n // num_cores) % TILE != 0:
            continue
        if k % (TILE * num_cores) != 0:
            continue
        return num_cores
    raise ValueError(
        f"Cannot DRAM width-shard matmul k={k}, n={n} on grid x={max_dram} "
        f"(need k % (32*cores)==0 and padded_n/cores tile-aligned)"
    )


def dram_linear_input_mem_config(device: ttnn.Device, m: int, k: int, n: int) -> ttnn.MemoryConfig:
    """L1 WIDTH-sharded activation layout for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``.

    The DRAM-sharded matmul kernel is hard-coded to ``M == TILE`` (one input-tile row).
    Long-seq callers (``m_actual > TILE``) must chunk the matmul into ``ceil(m_actual / TILE)``
    calls of this kernel; see ``TTSeamlessM4Tv2Encoder._linear``.
    """
    dram_cores = dram_matmul_shard_cores(device, k, n)
    if m > TILE:
        raise ValueError(f"dram_linear_input_mem_config expects m<={TILE}, got m={m} (chunk matmul instead)")
    return ttnn.create_sharded_memory_config(
        (m, k // dram_cores),
        core_grid=ttnn.CoreGrid(x=dram_cores, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def dram_matmul_program_config(
    device: ttnn.Device,
    m: int,
    k: int,
    n: int,
    *,
    fused_activation=None,
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    dram_cores = dram_matmul_shard_cores(device, k, n)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k // (TILE * dram_cores)),
        per_core_M=max(1, math.ceil(m / TILE)),
        per_core_N=math.ceil(n / (TILE * dram_cores)),
        fused_activation=fused_activation,
    )


def is_l1_width_sharded(tensor: ttnn.Tensor) -> bool:
    mc = tensor.memory_config()
    return mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def ensure_l1_width_sharded_activation(device: ttnn.Device, x: ttnn.Tensor, m: int, k: int, n: int) -> ttnn.Tensor:
    """Shard activations for ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` (in0 L1 width)."""
    target = dram_linear_input_mem_config(device, m, k, n)
    if is_l1_width_sharded(x):
        mc = x.memory_config()
        spec = mc.shard_spec
        if spec is not None and tuple(spec.shape) == tuple(target.shard_spec.shape):
            return x
    return ttnn.to_memory_config(x, target)


def width_sharded_to_l1_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    """Convert width-sharded L1 matmul output to interleaved L1 (SDPA / residual add)."""
    if x.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return x
    return ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)


def ensure_interleaved_bsh(
    x: ttnn.Tensor,
    *,
    batch: int,
    seq: int,
    channels: int,
) -> ttnn.Tensor:
    """Normalize activations to interleaved ``[B, S, C]`` (handles 2-D/4-D tile layouts)."""
    x = width_sharded_to_l1_interleaved(x)
    rank = len(x.shape)
    if rank == 4:
        flat_seq = int(x.shape[1]) * int(x.shape[2])
        c = int(x.shape[3])
        x = ttnn.reshape(x, (int(x.shape[0]), flat_seq, c))
        rank = 3
    if rank == 2:
        x = ttnn.reshape(x, (batch, seq, channels))
    elif rank == 3:
        if int(x.shape[0]) != batch or int(x.shape[1]) != seq or int(x.shape[2]) != channels:
            x = ttnn.slice(x, [0, 0, 0], [batch, seq, channels], [1, 1, 1])
    elif rank != 3:
        raise ValueError(f"Expected rank 2/3/4 for [B,S,C] normalize, got rank {rank} shape {x.shape}")
    return x


def ensure_tile_bf16_sdpa_mask(x: ttnn.Tensor) -> ttnn.Tensor:
    """SDPA requires a TILE bf16 mask; ``expand``/``add`` paths often yield ROW_MAJOR."""
    if x.get_layout() == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16:
        return x
    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    return out


def tile_align(seq: int) -> int:
    return ((seq + TILE - 1) // TILE) * TILE


def tt_position_ids(input_ids: ttnn.Tensor, pad_id: int) -> ttnn.Tensor:
    """HF ``create_position_ids_from_input_ids`` on device — ``cumsum`` of non-pad mask + offset."""
    ids_tile = (
        ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if input_ids.get_layout() != ttnn.TILE_LAYOUT
        else input_ids
    )
    mask = ttnn.ne(ids_tile, pad_id)
    if ids_tile is not input_ids:
        ttnn.deallocate(ids_tile)
    mask_i32 = ttnn.typecast(mask, ttnn.int32)
    ttnn.deallocate(mask)
    cumsum = ttnn.cumsum(mask_i32, dim=1, dtype=ttnn.int32)
    pos = ttnn.multiply(cumsum, mask_i32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum)
    ttnn.deallocate(mask_i32)
    pos = ttnn.add(pos, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos = ttnn.typecast(pos, ttnn.uint32)
    if pos.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        pos = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pos


def tt_position_ids_decode_step(
    input_ids: ttnn.Tensor,
    pad_id: int,
    past_key_values_length: int,
) -> ttnn.Tensor:
    """HF ``create_position_ids_from_input_ids`` for a single decode step ``[B, 1]``.

    Matches ``(cumsum(mask) + past_key_values_length) * mask + pad_id`` with ``mask = ids != pad``.
    """
    ids_tile = (
        ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if input_ids.get_layout() != ttnn.TILE_LAYOUT
        else input_ids
    )
    mask = ttnn.ne(ids_tile, pad_id)
    if ids_tile is not input_ids:
        ttnn.deallocate(ids_tile)
    mask_i32 = ttnn.typecast(mask, ttnn.int32)
    ttnn.deallocate(mask)
    cumsum = ttnn.cumsum(mask_i32, dim=1, dtype=ttnn.int32)
    past = ttnn.full(
        [int(input_ids.shape[0]), 1],
        float(past_key_values_length),
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=input_ids.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inc = ttnn.add(cumsum, past, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum)
    scaled = ttnn.multiply(inc, mask_i32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(inc)
    ttnn.deallocate(mask_i32)
    pos = ttnn.add(scaled, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(scaled)
    pos_u = ttnn.typecast(pos, ttnn.uint32)
    ttnn.deallocate(pos)
    if pos_u.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        pos_u = ttnn.to_layout(pos_u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pos_u


def tt_seq_position_ids(bsz: int, seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``create_position_ids_from_inputs_embeds`` on device — ``[pad+1, pad+2, …, pad+seq]``."""
    pos_1d = ttnn.arange(
        pad_id + 1,
        seq + pad_id + 1,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_2d = ttnn.reshape(pos_1d, [1, seq])
    if bsz <= 1:
        return pos_2d
    pos_out = ttnn.expand(pos_2d, [bsz, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pos_2d)
    return pos_out


def key_padding_additive(mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` 0/1 → ``[B, S]`` bf16 with ``0`` at real and ``NEG_INF`` at padded positions."""
    pad_bool = ttnn.eq(mask_2d, 0)
    pad_bf = ttnn.typecast(pad_bool, ttnn.bfloat16)
    ttnn.deallocate(pad_bool)
    additive = ttnn.multiply(pad_bf, NEG_INF, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_bf)
    return additive


def build_causal_mask_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal half only) on device."""
    full_neg = ttnn.full(
        [seq, seq],
        NEG_INF,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    causal_2d = ttnn.triu(full_neg, diagonal=1)
    ttnn.deallocate(full_neg)
    causal_4d = ttnn.reshape(causal_2d, [1, 1, seq, seq])
    if batch <= 1:
        return ensure_tile_bf16_sdpa_mask(causal_4d)
    expanded = ttnn.expand(causal_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    if causal_4d is not out:
        ttnn.deallocate(causal_4d)
    return out


def build_causal_with_padding_4d(
    attention_mask_2d: Optional[ttnn.Tensor], batch: int, seq: int, device: ttnn.Device
) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal + key padding) on device → ``[B, 1, S, S]`` bf16."""
    causal_4d = build_causal_mask_4d(batch, seq, device)
    if attention_mask_2d is None:
        return ensure_tile_bf16_sdpa_mask(causal_4d)
    pad_add_2d = key_padding_additive(attention_mask_2d, device=device)
    pad_add_4d = ttnn.reshape(pad_add_2d, [batch, 1, 1, seq])
    pad_add_expanded = ttnn.expand(pad_add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    combined = ttnn.add(causal_4d, pad_add_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(combined)
    ttnn.deallocate(pad_add_2d)
    for t in (pad_add_4d, pad_add_expanded, causal_4d):
        if t is not out:
            ttnn.deallocate(t)
    return out


def build_cross_attn_mask_4d(encoder_pad_mask_2d: ttnn.Tensor, *, tgt_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for cross-attn → ``[B, 1, tgt_seq, src_seq]`` bf16."""
    batch = int(encoder_pad_mask_2d.shape[0])
    src_seq = int(encoder_pad_mask_2d.shape[1])
    add_2d = key_padding_additive(encoder_pad_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, src_seq])
    if tgt_seq == 1:
        # ``ttnn.expand`` from ``[B, 1, 1, S]`` to ``[B, 1, 1, S]`` is a no-op view that shares
        # storage with ``add_2d``. Deallocating ``add_2d`` afterwards would free the returned
        # mask and trip SDPA's internal ``multiply(mask, scale)`` on the first decode step.
        # ``ensure_tile_bf16_sdpa_mask`` either returns ``add_4d`` as-is (already TILE bf16) or
        # allocates a tile-converted copy and deallocates ``add_4d`` itself — both ownership-safe.
        return ensure_tile_bf16_sdpa_mask(add_4d)
    expanded = ttnn.expand(add_4d, [batch, 1, tgt_seq, src_seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    ttnn.deallocate(add_2d)
    if add_4d is not out:
        ttnn.deallocate(add_4d)
    return out


def build_encoder_self_mask_4d(attention_mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for encoder self-attn → ``[B, 1, S, S]`` bf16."""
    batch = int(attention_mask_2d.shape[0])
    seq = int(attention_mask_2d.shape[1])
    add_2d = key_padding_additive(attention_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, seq])
    if seq == 1:
        # Same use-after-free guard as ``build_cross_attn_mask_4d``: when ``seq == 1`` the
        # ``ttnn.expand`` becomes a no-op view that shares storage with ``add_2d``, so a
        # later ``deallocate(add_2d)`` would free the returned mask. Defer ownership to
        # ``ensure_tile_bf16_sdpa_mask`` which either returns ``add_4d`` as-is or copies
        # and deallocates the input — both ownership-safe.
        return ensure_tile_bf16_sdpa_mask(add_4d)
    expanded = ttnn.expand(add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    ttnn.deallocate(add_2d)
    if add_4d is not out:
        ttnn.deallocate(add_4d)
    return out


def encoder_self_additive_mask_all_zeros_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Additive encoder self-attention mask when every position is valid (all keys visible)."""
    zeros = ttnn.zeros(
        [batch, 1, seq, seq],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return ensure_tile_bf16_sdpa_mask(zeros)


def pad_input_ids_to(input_ids: ttnn.Tensor, padded_seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 to ``[B, padded_seq]`` with ``pad_id`` (on device, ``ttnn.concat``)."""
    bsz = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    if padded_seq == seq:
        return input_ids
    pad_tail = ttnn.full(
        [bsz, padded_seq - seq],
        float(pad_id),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([input_ids, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_tail)
    return padded


def pad_mask_to(mask: ttnn.Tensor, padded_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 attention mask to ``[B, padded_seq]`` with 0 (on device)."""
    bsz = int(mask.shape[0])
    seq = int(mask.shape[1])
    if padded_seq == seq:
        return mask
    zeros = ttnn.full(
        [bsz, padded_seq - seq],
        0.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([mask, zeros], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(zeros)
    return padded


def ones_mask(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` uint32 all-ones (real-position mask) — used when caller omits ``attention_mask``."""
    return ttnn.full(
        [batch, seq],
        1.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def build_ln_sharded_config(
    device: ttnn.Device,
    m_tiles: int,
    n_tiles: int,
    cache: dict[Tuple[int, int], Tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]],
) -> Tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]:
    """Width-/block-sharded LN program config + memory config for ``[M_tiles, N_tiles]`` tile shape."""
    key = (m_tiles, n_tiles)
    cached = cache.get(key)
    if cached is not None:
        return cached

    device_grid = device.compute_with_storage_grid_size()
    grid_x = device_grid.x
    while grid_x > 1 and n_tiles % grid_x != 0:
        grid_x -= 1
    block_w = n_tiles // grid_x

    grid_y = min(device_grid.y, m_tiles)
    while grid_y > 1 and m_tiles % grid_y != 0:
        grid_y -= 1
    block_h = m_tiles // grid_y

    subblock_w = min(block_w, 4)
    while block_w % subblock_w != 0:
        subblock_w -= 1

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        [block_h * TILE, block_w * TILE],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED if grid_y == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    cached = (memory_config, program_config)
    cache[key] = cached
    return cached


def all_reduce_sum_replicate(
    x: ttnn.Tensor,
    mesh_device: ttnn.Device,
    *,
    cluster_axis: int = 1,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Sum-reduce TP partial results across devices on ``cluster_axis``; result replicated.

    For TP=1 (single device) returns ``x`` unchanged.  For TP>1, ``all_gather``
    stacks the per-device partial sums along the (unit) leading dim, then ``sum``
    over that dim produces the full all-reduce result.

    Each device starts with ``[1, ..., H]`` (a partial sum for the full output
    dimension), and after the all_reduce every device holds the full ``[1, ..., H]``.

    Why gather on dim 0 rather than the last dim: gathering on the last dim gives
    ``[..., tp*H]``, and separating the ``tp`` chunks then needs ``reshape [..., tp, H]``,
    which splits the tile-packed last dim → a physical re-tilize (a full copy ~27 µs/call,
    72×/decode step ≈ 25% of device time), plus a second reshape to restore ``[B, S, H]``
    and a ``fill_pad`` for the non-tile-aligned ``tp`` dim. Gathering on the leading
    (non-tiled) dim instead lets ``sum(dim=0)`` reduce across devices with no reshape,
    no fill_pad, and no re-tilize — and it is memory-safe at prefill/encoder lengths
    (``sum`` reads the gathered tensor and writes ``[1, ..., H]`` without a full extra copy),
    so the same path serves decode and prefill. Native ``ttnn.all_reduce`` was measured
    slower than ``all_gather`` + this local reduction on BH QB at H=1024.

    Assumes a unit leading dim (batch=1 decode/prefill here); see the assert below.
    """
    num_devices = 1
    if hasattr(mesh_device, "get_num_devices"):
        try:
            num_devices = int(mesh_device.get_num_devices())
        except Exception:
            num_devices = 1
    if num_devices <= 1:
        return x

    # Stack the per-device partials on the leading (non-tiled) dim: [1, ..., H] → [tp, ..., H].
    # Requires a unit leading dim so dim 0 sums purely across devices, not batch.
    assert int(x.shape[0]) == 1, f"all_reduce_sum_replicate expects a unit leading dim, got shape {x.shape}"
    try:
        gathered = ttnn.all_gather(
            x,
            dim=0,
            num_links=1,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            memory_config=memory_config,
        )
    except TypeError:
        # Newer TTNN all_gather API infers mesh from ``x`` and no longer accepts ``mesh_device``.
        gathered = ttnn.all_gather(
            x,
            dim=0,
            num_links=1,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
        )

    # Sum across devices: [tp, ..., H] → [1, ..., H]. keepdim preserves rank for downstream ops.
    acc = ttnn.sum(gathered, dim=0, keepdim=True, memory_config=memory_config)
    ttnn.deallocate(gathered)
    return acc


# TP encoder: large prefill activations in DRAM avoid L1 clashes with block-sharded matmul CBs.
ENCODER_TP_DRAM_TOKEN_THRESHOLD = 256


def encoder_tp_activation_memory_config(token_rows: int) -> ttnn.MemoryConfig:
    """Activation buffer type for encoder TP prefill (interleaved BSH)."""
    if token_rows >= ENCODER_TP_DRAM_TOKEN_THRESHOLD:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def encoder_all_reduce_sum_replicate(
    x: ttnn.Tensor,
    mesh_device: ttnn.Device,
    *,
    cluster_axis: int = 1,
    memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Text encoder only: sum row-parallel partials via ``ttnn.all_reduce`` (no ``all_gather``).

    Decoder / speech / T2U keep ``all_reduce_sum_replicate`` (gather + local sum).
    """
    num_devices = 1
    if hasattr(mesh_device, "get_num_devices"):
        try:
            num_devices = int(mesh_device.get_num_devices())
        except Exception:
            num_devices = 1
    if num_devices <= 1:
        return x

    mc = memory_config
    x_shape = list(x.shape)
    token_rows = 1
    for d in x_shape[:-1]:
        token_rows *= int(d)
    if mc.buffer_type == ttnn.BufferType.L1 and token_rows >= ENCODER_TP_DRAM_TOKEN_THRESHOLD:
        mc = ttnn.DRAM_MEMORY_CONFIG

    result = ttnn.all_reduce(
        x,
        cluster_axis=cluster_axis,
        memory_config=mc,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    if result is not x:
        ttnn.deallocate(x)
    return result


def sdpa_program_config(
    device: ttnn.Device,
    seq_q: int,
    seq_k: int,
    cache: dict[Any, ttnn.SDPAProgramConfig],
    *,
    large_chunks: bool = True,
) -> ttnn.SDPAProgramConfig:
    """Chunk sizes for ``ttnn.transformer.scaled_dot_product_attention`` (cached per caller dict)."""
    key: Any = (seq_q, seq_k) if large_chunks else (seq_q, seq_k, large_chunks)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if large_chunks:
        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
    else:
        q_chunk = max(32, min(256, nearest_32(seq_q)))
        k_chunk = max(32, min(256, nearest_32(seq_k)))
    out = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    cache[key] = out
    return out


# ============================================================================
# gather_in0 DRAM-width-sharded matmul (ported from devstral2_opt_17).
# ----------------------------------------------------------------------------
# Pattern: activations live in L1 WIDTH_SHARDED across the compute grid (K-dim
# sharded). Weights live in DRAM WIDTH_SHARDED across DRAM banks (N-dim sharded
# in the per-device slice). The matmul kernel reads activations across the
# core ring (``gather_in0=True``) so each core fetches its K shard from its
# neighbour without an explicit ``sharded_to_interleaved → interleaved_to_sharded``
# round trip. Output is L1 WIDTH_SHARDED on the same grid (now N-dim sharded).
#
# This pattern composes with TP because the gather happens *within* a device
# (across cores), while TP shards weights *across* devices. Existing seamless
# DRAM-sharded path (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``)
# is gated to ``tp == 1`` because its grid-vs-DRAM-bank coupling does not
# tolerate the per-device weight slice. gather_in0 does.
#
# Constraint per call: K_tiles % num_cores == 0 AND N_tiles % num_cores == 0
# (both must shard evenly across the same compute grid).
# ============================================================================


def find_grid_for_k_n(k_tiles: int, n_tiles: int, max_rows: int = 10, max_cols: int = 13) -> Tuple[int, int]:
    """Largest (grid_x, grid_y) where ``num_cores = grid_x * grid_y`` divides both K_tiles and N_tiles.

    Used to size the worker grid for ``gather_in0`` matmul. Picks the largest grid that
    satisfies both divisibility constraints so per-core work stays small.
    """
    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return cols, rows  # (grid_x, grid_y)
    raise AssertionError(f"No grid divides both K={k_tiles} and N={n_tiles} tiles within {max_rows}x{max_cols}.")


def gather_in0_matmul_program_config(
    *,
    grid_x: int,
    grid_y: int,
    m_seq: int,
    k_dim: int,
    n_dim: int,
    fuse_batch: bool,
    fused_activation=None,
    fp32_dest_acc_en: bool = True,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig(gather_in0=True)`` for DRAM-width-sharded
    weights + L1 WIDTH_SHARDED activations.

    Caller must ensure ``k_dim`` and ``n_dim`` are both divisible by ``grid_x * grid_y * TILE``
    (i.e. ``k_tiles`` and ``n_tiles`` divide ``num_cores`` evenly).
    """
    num_cores = grid_x * grid_y
    if m_seq % TILE != 0:
        raise ValueError(f"gather_in0 matmul requires M divisible by {TILE}, got m_seq={m_seq}")
    in0_block_w = k_dim // num_cores // TILE
    out_block_h = m_seq // TILE
    out_block_w = n_dim // num_cores // TILE
    if out_block_w * num_cores != n_dim // TILE:
        raise ValueError(
            f"gather_in0 shard mismatch: n_dim={n_dim} not divisible by num_cores*TILE = {num_cores * TILE}"
        )

    max_subblock = 4 if fp32_dest_acc_en else 8
    out_subblock_w = max((i for i in range(1, max_subblock + 1) if out_block_w % i == 0), default=1)
    out_subblock_h = max(
        (i for i in range(1, max_subblock + 1) if out_block_h % i == 0 and i * out_subblock_w <= max_subblock),
        default=1,
    )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=fuse_batch,
        fused_activation=fused_activation,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet(set()),
        num_global_cb_receivers=1,
    )


def width_sharded_l1_memcfg(m_tiles: int, k_tiles: int, grid_x: int, grid_y: int) -> ttnn.MemoryConfig:
    """L1 WIDTH_SHARDED memcfg for activations ``[1, 1, m_tiles*TILE, k_tiles*TILE]``.

    K-dim is sharded across ``num_cores = grid_x*grid_y`` cores (K_tiles must divide num_cores).
    """
    num_cores = grid_x * grid_y
    if k_tiles % num_cores != 0:
        raise ValueError(f"width_sharded_l1_memcfg: k_tiles={k_tiles} not divisible by num_cores={num_cores}")
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    shard_spec = ttnn.ShardSpec(
        core_grid,
        (m_tiles * TILE, (k_tiles // num_cores) * TILE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def dram_width_sharded_weight_memcfg(k: int, padded_n: int, device: ttnn.Device) -> ttnn.MemoryConfig:
    """DRAM WIDTH_SHARDED memcfg for a weight tensor of shape ``[K, padded_N]`` (per device).

    ``padded_n`` must be a multiple of ``TILE * dram_cores`` (use ``pad_n_for_dram_align`` first).
    Shards across the DRAM grid (y=1 on Blackhole).
    """
    dram_grid = device.dram_grid_size()
    dram_cores = int(dram_grid.x)
    assert int(dram_grid.y) == 1, "DRAM-width-sharded weights assume dram_grid.y == 1"
    if padded_n % dram_cores != 0:
        raise ValueError(f"padded_n={padded_n} not divisible by dram_cores={dram_cores}")
    core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(core_range, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def pad_n_for_dram_align(n: int, dram_cores: int) -> int:
    """Round ``n`` up to a multiple of ``TILE * dram_cores`` so weights split evenly across DRAM banks."""
    align = TILE * dram_cores
    return ((n + align - 1) // align) * align
