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
    IDs, sequence lengths, vocoder cumsums, T2U duration counts, …). On a multi-device mesh every
    device sees the same control-flow scalars because inputs are replicated and ops are
    deterministic, so all shards are identical and reading one is sufficient.

    ``ttnn.to_torch`` without a ``mesh_composer`` errors on a tensor with >1 shard
    (TT_FATAL: "Can't convert a tensor distributed on … mesh to row-major logical tensor.
    Supply a mesh composer …"), so we wire in ``ConcatMeshToTensor(dim=0)`` and slice the leading
    per-device chunk off the result. On a 1×1 mesh the composer path is skipped — behaviour is
    bit-identical to the original ``ttnn.to_torch`` call, so PCC tests (which all run on a
    single-device fixture) are unaffected.

    Accepts either a device tensor or a host tensor. Host tensors that came from a
    multi-device ``ttnn.from_device`` retain multi-shard storage; we look up the mesh device via
    ``GetDefaultDevice`` so we can still attach a composer.
    """
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
        if out.dim() >= 1 and out.shape[0] >= num_devices:
            out = out[: out.shape[0] // num_devices]
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
    k_tiles = max(1, in_dim // TILE)
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

    grid_y = min(cg.y, m_tiles)
    while grid_y > 1 and n_tiles % (cg.x * grid_y) != 0:
        grid_y -= 1
    per_core_m = max(1, (m_tiles + grid_y - 1) // grid_y)
    per_core_n = max(1, (n_tiles + cg.x * grid_y - 1) // (cg.x * grid_y))
    out_subblock_w = _pick_matmul_out_subblock_w(per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cg.x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def create_dram_sharded_mem_config(device: ttnn.Device, k: int, n: int) -> Tuple[ttnn.MemoryConfig, int]:
    """WIDTH-sharded DRAM config for linear weight ``[k, n]`` (``n`` may be padded)."""
    dram_cores = dram_shard_core_count(device, n)
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


def dram_linear_input_mem_config(device: ttnn.Device, m: int, k: int) -> ttnn.MemoryConfig:
    dram_cores = dram_shard_core_count(device, k)
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
    dram_cores = dram_shard_core_count(device, n)
    assert k % (TILE * dram_cores) == 0, f"k={k} must divide tile_size * dram_cores ({TILE * dram_cores})"
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(k // (TILE * dram_cores)),
        per_core_M=max(1, math.ceil(m / TILE)),
        per_core_N=math.ceil(n / (TILE * dram_cores)),
        fused_activation=fused_activation,
    )


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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
