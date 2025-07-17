# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3YarnRotaryEmbedding
from models.utility_functions import nearest_32


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_cos_sin_matrix(hf_config):
    """Function to get the cos and sin matrices for rotary positional embeddings.

    Args:
        hf_config: HuggingFace configuration object containing model parameters.
    Returns:
        cos: Cosine matrix for rotary embeddings.
        sin: Sine matrix for rotary embeddings.
    This function uses the DeepseekV3YarnRotaryEmbedding class to generate the matrices
    based on the provided HuggingFace configuration.

    HuggingFace returns cos/sin matrices in the format of [max_seq_len, dim], where dim is [t1, .., td//2, t1, .., td//2].
    This is because HF is the format of [r, r, ..., i, i, ...] which requires cos/sin to be [t1, t2, ..., td//2, t1, t2, ..., td//2].
    However, the Meta-style frequencies are in the format of [r, i, r, i, ...], so the cos/sin need to be [t1, t1, ..., td//2, td//2].

    """
    args = {
        "dim": hf_config.qk_rope_head_dim,
        "max_position_embeddings": hf_config.max_seq_len,
        "base": hf_config.rope_theta * 1.0,
        "device": "cpu",
        "scaling_factor": hf_config.rope_scaling["factor"],
        "original_max_position_embeddings": hf_config.rope_scaling["original_max_position_embeddings"],
        "beta_fast": hf_config.rope_scaling["beta_fast"],
        "beta_slow": hf_config.rope_scaling["beta_slow"],
        "mscale": hf_config.rope_scaling["mscale"],
        "mscale_all_dim": hf_config.rope_scaling["mscale_all_dim"],
    }

    reference_rope = DeepseekV3YarnRotaryEmbedding(**args)

    # [max_seq_len, dim], where dim is [t1, .., td//2, t1, .., td//2]
    cos = reference_rope.cos_cached
    sin = reference_rope.sin_cached

    # Undo the HF permute
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]

    return cos, sin


class RotarySetup:
    """
    Class to set up rotary positional embeddings for DeepSeekV3 models.

    Duplicate + changes from TTT rope.py
    """

    def __init__(self, device, batch_size: int, hf_config, use_dp: bool = False):
        self.batch_size = batch_size
        self.hf_config = hf_config
        self.dim = hf_config.qk_rope_head_dim
        self.device = device
        self.use_dp = use_dp

        if self.use_dp:
            self.batch_size_per_device_group = max(self.batch_size // list(device.shape)[0], 1)
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        # Generate the cos/sin matrices needed for ttnn.embedding op
        self.cos_matrix, self.sin_matrix = self.get_rot_mats_table()

        self.batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)

        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat().repeat(
            1,
            1,
            batch_size,
            1,
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=trans_mat_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        prefill_trans_mat_torch = get_rot_transformation_mat()
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def get_rot_idxs(self, position_idxs, on_host=False):
        """
        Get the rotary positional embedding indices for the given position indices.
        Args:
            position_idxs: A tensor of shape [batch] containing the position indices.
            on_host: If True, the indices will be sent to the host device.
        Returns:
            rot_idxs: A tensor of shape [1, batch] containing the rotary positional embedding indices.
        """
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device,
                dims=(1, None) if self.use_dp else (None, None),
                mesh_shape=list(self.device.shape),
            ),
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
        )

        return rot_idxs

    def get_rot_mats_table(self, seq_len=None):
        """
        Get the cos and sin matrices for all positions in the sequence length.
        If seq_len is None, it will use the max position embeddings from the HF config.
        """

        cos_matrix_torch, sin_matrix_torch = get_cos_sin_matrix(self.hf_config)

        if seq_len is not None:
            assert seq_len <= self.hf_config.max_seq_len, "seq_len must be less than or equal to max_seq_len"
            cos_matrix_torch = cos_matrix_torch[..., :seq_len, :]
            sin_matrix_torch = sin_matrix_torch[..., :seq_len, :]

        cos_matrix = ttnn.from_torch(
            cos_matrix_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin_matrix = ttnn.from_torch(
            sin_matrix_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

        if seq_len is not None:
            return cos_matrix, sin_matrix, self.transformation_mat_prefill
        return cos_matrix, sin_matrix

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        """
        Get the cos and sin matrices for the given position indices.
        Args:
            position_idxs: A tensor of shape [1, batch] containing the position indices.
            return_rot_idxs: If True, the function will also return the rotary positional embedding indices.
        Returns:
            A list containing the cos matrix, sin matrix, and transformation matrix.
            If return_rot_idxs is True, it will also return the rotary positional embedding indices.
        """

        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=embedding_layout)  # [1, batch, dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=embedding_layout)  # [1, batch, dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.dim]

        if return_rot_idxs:
            return [cos, sin, self.transformation_mat], rot_idxs
        return [cos, sin, self.transformation_mat]
