# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral vision sequence chunk size for L1-bound matmuls (attention + MLP).

from __future__ import annotations

import os
from typing import Any, Tuple

import ttnn

from models.common.utility_functions import nearest_32

VISION_L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
TILE = 32


def vision_rms_norm_gamma_weight(gamma, device, memory_config, dtype=ttnn.bfloat16):
    """Tile-pad a 1D gamma and upload for ``ttnn.rms_norm`` (replaces removed ``pad_by_zero``)."""
    hidden = int(gamma.shape[0])
    tt = ttnn.from_torch(
        gamma.reshape(1, 1, 1, hidden),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    pad_h = TILE - 1
    pad_w = (TILE - (hidden % TILE)) % TILE
    if pad_h or pad_w:
        tt = ttnn.pad(tt, padding=[(0, 0), (0, 0), (0, pad_h), (0, pad_w)], value=0.0)
    tt = tt.to(ttnn.TILE_LAYOUT)
    is_mesh = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
    if is_mesh:
        tt = ttnn.from_torch(
            ttnn.to_torch(tt),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    else:
        tt = tt.to(device, memory_config)
    return tt


def vision_activation_memcfg(num_elements: int) -> ttnn.MemoryConfig:
    """L1 for vision ops when activation fits L1; else DRAM (avoids SDPA/CB clashes on large images)."""
    cap_raw = os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096")
    cap = max(32, nearest_32(int(cap_raw)))
    if int(num_elements) <= cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_slice_memcfg(num_patches: int) -> ttnn.MemoryConfig:
    """L1 for patch-grid slice/reshape only on small grids (agent full-res images stay DRAM)."""
    cap_raw = os.environ.get("PIXTRAL_VISION_L1_SLICE_CAP", "1024")
    cap = max(32, nearest_32(int(cap_raw)))
    if int(num_patches) <= cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_seq_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for seq×feature activations (norm/MLP/QKV) when seq and element volume fit L1."""
    seq_cap_raw = os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096")
    seq_cap = max(32, nearest_32(int(seq_cap_raw)))
    elem_cap_raw = os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576")
    elem_cap = max(seq_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(feature_dim))
    if int(seq_len) <= seq_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_rms_norm_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 RMSNorm when seq×feature fits; opt-out via PIXTRAL_VISION_NORM_DRAM=1 (CB clash fallback)."""
    if os.environ.get("PIXTRAL_VISION_NORM_DRAM", "").strip() in ("1", "true", "True"):
        return ttnn.DRAM_MEMORY_CONFIG
    return vision_seq_memcfg(seq_len, feature_dim)


def _vision_rms_norm_block_shard_enabled() -> bool:
    return os.environ.get("PIXTRAL_VISION_RMS_NORM_BLOCK_SHARD", "1").strip() not in ("0", "false", "False")


def _rms_norm_subblock_w(block_w: int) -> int:
    for s in (4, 3, 2, 1):
        if block_w % s == 0:
            return s
    return 1


def vision_rms_norm_block_shard_eligible(
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> bool:
    """True when 2D block-sharded RMSNorm matches 1024² sweep winner (8x8, bh=4, bw=4, sbw=4)."""
    if not _vision_rms_norm_block_shard_enabled():
        return False
    m, n = int(seq_len), int(feature_dim)
    if m % TILE or n % TILE:
        return False
    mt, nt = m // TILE, n // TILE
    if grid_x < 2 or mt % grid_y or nt % grid_x:
        return False
    return True


def vision_rms_norm_block_shard_memcfg(
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        (1, 1, int(seq_len), int(feature_dim)),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def vision_rms_norm_block_shard_program_config(
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.LayerNormShardedMultiCoreProgramConfig:
    mt, nt = int(seq_len) // TILE, int(feature_dim) // TILE
    block_h = mt // grid_y
    block_w = nt // grid_x
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid_x, grid_y],
        subblock_w=_rms_norm_subblock_w(block_w),
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )


def vision_rms_norm_prepare_block_shard_input(
    tensor: ttnn.Tensor,
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.Tensor:
    mem = vision_rms_norm_block_shard_memcfg(seq_len, feature_dim, grid_x, grid_y)
    if tensor.is_sharded():
        if tensor.memory_config() == mem:
            return tensor
        return ttnn.to_memory_config(tensor, mem)
    return ttnn.interleaved_to_sharded(tensor, mem)


def vision_rope_memcfg(seq_len: int, head_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for RoPE embed + rotary_embedding when seq×head fits."""
    rope_cap_raw = os.environ.get("PIXTRAL_VISION_L1_ROPE_SEQ_CAP", os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096"))
    rope_cap = max(32, nearest_32(int(rope_cap_raw)))
    elem_cap_raw = os.environ.get(
        "PIXTRAL_VISION_L1_ROPE_ELEM_CAP",
        os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576"),
    )
    elem_cap = max(rope_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(head_dim))
    if int(seq_len) <= rope_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def vision_collective_memcfg(seq_len: int, feature_dim: int = 1) -> ttnn.MemoryConfig:
    """L1 for all_gather/fast_reduce when seq×shard fits (norms/RoPE stay DRAM to avoid CB clash)."""
    ag_cap_raw = os.environ.get("PIXTRAL_VISION_L1_AG_SEQ_CAP", os.environ.get("PIXTRAL_VISION_L1_SEQ_CAP", "4096"))
    ag_cap = max(32, nearest_32(int(ag_cap_raw)))
    elem_cap_raw = os.environ.get(
        "PIXTRAL_VISION_L1_AG_ELEM_CAP",
        os.environ.get("PIXTRAL_VISION_L1_ELEM_CAP", "1048576"),
    )
    elem_cap = max(ag_cap, int(elem_cap_raw))
    elements = int(seq_len) * max(1, int(feature_dim))
    if int(seq_len) <= ag_cap and elements <= elem_cap:
        return VISION_L1_MEMCFG
    return ttnn.DRAM_MEMORY_CONFIG


def pixtral_vision_seq_chunk_len(configuration) -> int:
    """L1 matmul seq chunk (env PIXTRAL_VISION_MM_SEQ_CHUNK or min(cfg, cap))."""
    force = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK")
    if force is not None and str(force).strip() != "":
        chunk = max(32, nearest_32(int(force)))
    else:
        cap_raw = os.environ.get("PIXTRAL_VISION_MM_SEQ_CHUNK_CAP", "1024")
        cap = max(32, nearest_32(int(cap_raw)))
        cfg_chunk = getattr(configuration, "VISION_MAX_MM_SEQ", cap)
        if cfg_chunk is None:
            cfg_chunk = cap
        chunk = max(32, min(int(cfg_chunk), int(cap)))

    return chunk


def pixtral_effective_mm_seq_len(configuration, seq_len: int) -> int:
    """Matmul M: one kernel over full ``seq_len`` when it fits L1, else ``pixtral_vision_seq_chunk_len``."""
    chunk = pixtral_vision_seq_chunk_len(configuration)
    force = os.environ.get("PIXTRAL_VISION_MM_FULL_SEQ_CAP")
    if force is not None and str(force).strip() != "":
        full_cap = max(32, nearest_32(int(force)))
    else:
        full_cap = max(chunk, 1024)
    if seq_len <= full_cap:
        return seq_len
    return chunk


def pad_seq_to_chunk_multiple(x: ttnn.Tensor, seq_len: int, chunk: int) -> tuple[ttnn.Tensor, int, int]:
    """Pad dim=2 to a multiple of ``chunk`` so batched matmul avoids ``ttnn.concat``."""
    original = seq_len
    if seq_len <= chunk or seq_len % chunk == 0:
        return x, seq_len, original
    padded = ((seq_len + chunk - 1) // chunk) * chunk
    pad_len = padded - seq_len
    x = ttnn.pad(x, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
    return x, padded, original


def trim_seq_dim2(x: ttnn.Tensor, original_seq_len: int) -> ttnn.Tensor:
    if int(x.shape[2]) == original_seq_len:
        return x
    return x[:, :, :original_seq_len, :]


def _vision_compute_grid_size(configuration: Any) -> tuple[int, int]:
    grid = configuration.max_grid_size
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return int(grid.x), int(grid.y)
    if isinstance(grid, (tuple, list)) and len(grid) >= 2:
        return int(grid[0]), int(grid[1])
    return 8, 8


def _vision_nlp_shard_enabled() -> bool:
    return os.environ.get("PIXTRAL_VISION_NLP_SHARD", "1").strip() not in ("0", "false", "False")


def vision_nlp_qkv_shard_memcfgs(
    seq_len: int,
    qkv_width: int,
    n_local_kv_heads: int,
    configuration: Any,
    batch: int = 1,
) -> Tuple[ttnn.MemoryConfig, ttnn.MemoryConfig]:
    """WIDTH_SHARDED L1 in, HEIGHT_SHARDED L1 out (fast sharded NlpCreateHeads path)."""
    num_cores = int(n_local_kv_heads)
    compute_grid_size = _vision_compute_grid_size(configuration)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    shard_shape = [int(seq_len) * int(batch), int(qkv_width) // num_cores]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return in_mem, out_mem


def vision_use_sharded_nlp_create_qkv(
    chunk_seq_len: int,
    n_local_heads: int,
    n_local_kv_heads: int,
    head_dim: int,
    qkv_width: int,
) -> bool:
    """Sharded create_qkv when chunk seq fits HEIGHT-shard limits (seq * heads <= heads * TILE)."""
    if not _vision_nlp_shard_enabled():
        return False
    if int(chunk_seq_len) % TILE != 0 or int(head_dim) % TILE != 0:
        return False
    num_cores = int(n_local_kv_heads)
    if num_cores <= 0 or int(n_local_heads) % num_cores != 0 or int(n_local_kv_heads) % num_cores != 0:
        return False
    if int(qkv_width) % num_cores != 0:
        return False
    return int(chunk_seq_len) * int(n_local_heads) <= int(n_local_heads) * TILE


def vision_use_sharded_nlp_concat(
    seq_len: int,
    n_local_heads: int,
    head_dim: int,
    configuration: Any,
    batch: int = 1,
) -> bool:
    """HEIGHT-sharded concat (one core per head). Disabled on Blackhole (kernel hang)."""
    if os.environ.get("PIXTRAL_VISION_NLP_CONCAT_SHARD", "").strip() not in ("1", "true", "True"):
        return False
    if not _vision_nlp_shard_enabled():
        return False
    if int(seq_len) % TILE != 0 or int(head_dim) % TILE != 0:
        return False
    if int(batch) != 1:
        return False
    grid_x, _grid_y = _vision_compute_grid_size(configuration)
    return 0 < int(n_local_heads) <= grid_x


def _vision_mlp_block_shard_enabled() -> bool:
    return os.environ.get("PIXTRAL_VISION_MLP_BLOCK_SHARD", "1").strip() not in ("0", "false", "False")


def vision_mlp_block_shard_eligible(
    m: int,
    k: int,
    n: int,
    grid_x: int = 8,
    grid_y: int = 8,
    in0_block_w: int = 4,
) -> bool:
    """True when 2D block-sharded matmul matches 1024³ sweep winner (bs/dram/bs, w=4)."""
    if not _vision_mlp_block_shard_enabled():
        return False
    m, k, n = int(m), int(k), int(n)
    if m % TILE or k % TILE or n % TILE:
        return False
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    if mt % grid_y or nt % grid_x or kt % grid_x:
        return False
    return (kt // grid_x) % int(in0_block_w) == 0


def vision_mlp_block_shard_in0_memcfg(
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        (1, 1, int(seq_len), int(feature_dim)),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def vision_mlp_block_shard_out_memcfg() -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )


def vision_mlp_block_shard_matmul_progcfg(
    m: int,
    k: int,
    n: int,
    grid_x: int = 8,
    grid_y: int = 8,
    in0_block_w: int = 4,
    *,
    fuse_batch: bool = True,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    mt, kt, nt = m // TILE, k // TILE, n // TILE
    per_core_m = mt // grid_y
    per_core_n = nt // grid_x
    out_subblock_w = 4 if per_core_n % 4 == 0 else (2 if per_core_n % 2 == 0 else 1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=fuse_batch,
    )


def vision_mlp_prepare_block_shard_input(
    tensor: ttnn.Tensor,
    seq_len: int,
    feature_dim: int,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.Tensor:
    mem = vision_mlp_block_shard_in0_memcfg(seq_len, feature_dim, grid_x, grid_y)
    if tensor.is_sharded():
        if tensor.memory_config() == mem:
            return tensor
        return ttnn.to_memory_config(tensor, mem)
    return ttnn.interleaved_to_sharded(tensor, mem)


def vision_nlp_concat_input_memcfg(
    seq_len: int,
    head_dim: int,
    n_local_heads: int,
    configuration: Any,
    batch: int = 1,
) -> ttnn.MemoryConfig:
    """HEIGHT-sharded L1: one core per head, full seq_len rows (matches nlp_concat_heads sharded kernel)."""
    _ = configuration, batch
    num_cores = int(n_local_heads)
    return ttnn.create_sharded_memory_config(
        shape=(int(seq_len), int(head_dim)),
        core_grid=ttnn.CoreGrid(y=1, x=num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
