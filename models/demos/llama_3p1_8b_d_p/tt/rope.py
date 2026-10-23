# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side Llama-3.1 RoPE mathematics and coordinate conversion."""

import math

import torch

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


def _require_even_head_dim(head_dim: int) -> None:
    if head_dim % 2:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}")


def llama3_inv_freq(
    *,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    original_max_position_embeddings: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return Llama3-scaled inverse frequencies as a float32 host tensor."""
    _require_even_head_dim(head_dim)

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    wavelength = 2 * math.pi / inv_freq
    low_freq_wavelength = original_max_position_embeddings / low_freq_factor
    high_freq_wavelength = original_max_position_embeddings / high_freq_factor

    scaled_inv_freq = inv_freq / factor
    llama3_freq = torch.where(wavelength > low_freq_wavelength, scaled_inv_freq, inv_freq)
    smooth_factor = (original_max_position_embeddings / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * scaled_inv_freq + smooth_factor * inv_freq
    medium_freq = (wavelength >= high_freq_wavelength) & (wavelength <= low_freq_wavelength)
    return torch.where(medium_freq, smoothed_inv_freq, llama3_freq)


def build_llama3_cos_sin(
    seq_len: int,
    *,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    theta: float = Llama31_8BConfig.ROPE_THETA,
    factor: float = Llama31_8BConfig.ROPE_SCALING_FACTOR,
    low_freq_factor: float = Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
    high_freq_factor: float = Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
    original_max_position_embeddings: int = Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Meta-interleaved float32 cos/sin tables with shape [1, 1, S, D]."""
    inv_freq = llama3_inv_freq(
        head_dim=head_dim,
        theta=theta,
        factor=factor,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=inv_freq.device)
    angles = torch.repeat_interleave(torch.outer(positions, inv_freq), 2, dim=-1)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def hf_to_meta(tensor: torch.Tensor) -> torch.Tensor:
    """Convert final-axis HF half-split coordinates to Meta adjacent pairs."""
    head_dim = tensor.shape[-1]
    _require_even_head_dim(head_dim)
    half = head_dim // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).flatten(-2)


def meta_to_hf(tensor: torch.Tensor) -> torch.Tensor:
    """Convert final-axis Meta adjacent pairs to HF half-split coordinates."""
    _require_even_head_dim(tensor.shape[-1])
    return torch.cat((tensor[..., 0::2], tensor[..., 1::2]), dim=-1)


def indexed_rope_table_capacity(*, max_seq_len: int, chunk_size: int) -> int:
    """Return table rows needed for the final full physical indexed-RoPE read.

    ``max_seq_len`` remains the logical serving limit.  The indexed kernel always consumes a
    full physical chunk, including a padded tail, so its table needs one chunk beyond that limit
    and is rounded to whole global chunks.
    """
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len!r}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    required = max_seq_len + chunk_size
    return math.ceil(required / chunk_size) * chunk_size


def _indexed_rope_geometry(mesh_device, *, max_seq_len: int, chunk_size: int, sp_axis: int):
    mesh_shape = tuple(mesh_device.shape)
    if len(mesh_shape) != 2:
        raise ValueError(f"indexed RoPE requires a 2-D mesh, got {mesh_shape}")
    if sp_axis not in (0, 1):
        raise ValueError(f"sp_axis must be 0 or 1, got {sp_axis!r}")
    if any(not isinstance(dim, int) or dim <= 0 for dim in mesh_shape):
        raise ValueError(f"mesh dimensions must be positive integers, got {mesh_shape}")

    capacity = indexed_rope_table_capacity(max_seq_len=max_seq_len, chunk_size=chunk_size)
    sp = mesh_shape[sp_axis]
    if chunk_size % sp:
        raise ValueError(f"chunk_size ({chunk_size}) must be divisible by SP ({sp})")
    chunk_local = chunk_size // sp
    if chunk_local % 32:
        raise ValueError(f"per-device chunk ({chunk_local}) must be tile-aligned (a multiple of 32)")
    return mesh_shape, sp, chunk_local, capacity


def build_transformation_mat(mesh_device, *, dtype=None):
    """Build a replicated, single-tile RoPE transformation matrix in DRAM."""
    import ttnn
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    if dtype is None:
        dtype = ttnn.bfloat16
    return ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def build_indexed_rope(
    mesh_device,
    *,
    max_seq_len: int,
    chunk_size: int,
    sp_axis: int = 0,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    dtype=None,
):
    """Build persistent block-cyclic Llama3 cos/sin tables for indexed device RoPE.

    The sequence dimension is sharded across SP and replicated across the other mesh axis.  Table
    capacity covers padded physical reads, while ``max_seq_len`` remains only the logical context
    limit consumed by the model.
    """
    mesh_shape, sp, chunk_local, capacity = _indexed_rope_geometry(
        mesh_device,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        sp_axis=sp_axis,
    )
    _require_even_head_dim(head_dim)

    import ttnn
    from models.common.utils import block_cyclic_reorder

    if dtype is None:
        dtype = ttnn.bfloat16
    cos, sin = build_llama3_cos_sin(capacity, head_dim=head_dim)
    cos = block_cyclic_reorder(cos, chunk_local, sp, seq_dim=2)
    sin = block_cyclic_reorder(sin, chunk_local, sp, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(shard_dims))

    def to_device(table):
        return ttnn.from_torch(
            table,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    return [to_device(cos), to_device(sin)]


def apply_indexed_rope(
    tensor,
    rope_tables,
    transformation_mat,
    *,
    kv_actual_global: int,
    sp_axis: int,
):
    """Apply indexed RoPE with the stock scalar-offset TTNN operator."""
    if sp_axis not in (0, 1):
        raise ValueError(f"sp_axis must be 0 or 1, got {sp_axis!r}")
    if len(rope_tables) != 2:
        raise ValueError(f"rope_tables must contain cos and sin, got {len(rope_tables)} tensors")

    import ttnn

    return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
        tensor,
        rope_tables[0],
        rope_tables[1],
        transformation_mat,
        kv_actual_global=kv_actual_global,
        cluster_axis=sp_axis,
    )
