import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rope import get_cos_sin_matrix, get_rot_transformation_mat


def create_balanced_chunk_order(sp_factor: int) -> list[int]:
    """Create balanced chunk order for sequence reordering.

    For sp_factor=4, creates 2*4=8 chunks with order: 0,7,1,6,2,5,3,4
    This interleaves chunks from start and end to balance workload.
    """
    num_chunks = 2 * sp_factor
    balanced_order = []

    left = 0
    right = num_chunks - 1

    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1

    return balanced_order


def reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = 2) -> torch.Tensor:
    """Reorder tensor chunks along sequence dimension according to chunk_order."""
    seq_len = tensor.shape[seq_dim]
    num_chunks = len(chunk_order)
    chunk_size = seq_len // num_chunks

    # Split into chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
        else:
            raise NotImplementedError(f"Reordering for seq_dim={seq_dim} not implemented")

    # Reorder chunks according to chunk_order
    reordered_chunks = [chunks[i] for i in chunk_order]

    # Concatenate reordered chunks
    return torch.cat(reordered_chunks, dim=seq_dim)


def reverse_reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = 2) -> torch.Tensor:
    """Reverse the chunk reordering to restore original order."""
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)


def get_rope_tensors(
    hf_config: PretrainedConfig,
    seq_len: int,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int = 0,
    is_balanced: bool = False,
) -> dict[str, ttnn.Tensor]:
    cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(hf_config)

    assert (
        seq_len <= hf_config.max_seq_len
    ), f"seq_len {seq_len} must be less than or equal to max_seq_len {hf_config.max_seq_len}"
    cos_matrix_torch = cos_matrix_torch[..., :seq_len, :]
    sin_matrix_torch = sin_matrix_torch[..., :seq_len, :]

    if is_balanced:
        sp_factor = mesh_device.shape[sp_axis]
        chunk_order = create_balanced_chunk_order(sp_factor)
        cos_matrix_torch = reorder_tensor_chunks(cos_matrix_torch, chunk_order, seq_dim=2)
        sin_matrix_torch = reorder_tensor_chunks(sin_matrix_torch, chunk_order, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[sp_axis] = 2

    cos_matrix = ttnn.from_torch(
        cos_matrix_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    )
    sin_matrix = ttnn.from_torch(
        sin_matrix_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    )

    trans_mat_torch = get_rot_transformation_mat()
    trans_matrix = ttnn.from_torch(
        trans_mat_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return {"cos_matrix": cos_matrix, "sin_matrix": sin_matrix, "trans_matrix": trans_matrix}
