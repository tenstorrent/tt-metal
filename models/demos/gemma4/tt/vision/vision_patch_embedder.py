# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Gemma-4 vision patch embedder.

Mirrors HF ``Gemma4VisionPatchEmbedder``:
  * ``input_proj``: a (bias-free) linear projection of the flattened patch pixels
    ``[batch, num_patches, 3*patch_size^2] -> [..., hidden_size]`` -- a real matmul on
    real pixel data, run on device.
  * position embeddings: the reference computes ``one_hot(position_ids) @ position_embedding_table``
    summed over the x/y spatial axes, which is exactly two embedding-table lookups, so we use
    ``ttnn.embedding`` against the two ``[position_embedding_size, hidden_size]`` tables and add.
  * padding patches are zeroed and the projected patches and position embeddings are summed.

All inputs (``pixel_values``, ``pixel_position_ids``, ``padding_positions``) are on-device
``ttnn`` tensors: the position-id clamp/slice, embedding lookups, padding mask, matmul and add
all run on device with no host round-trip.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionPatchEmbedder(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, state_dict_prefix="", weight_cache_path=None, dtype=ttnn.bfloat16
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype

        vision_config = args.hf_config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.position_embedding_size = vision_config.position_embedding_size

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

        def cache_name(name):
            if args.dummy_weights or weight_cache_path is None:
                return None
            return weight_cache_path / f"{state_dict_prefix}{name}"

        # input_proj weight is [hidden_size, in_dim]; transpose to [in_dim, hidden_size] for x @ W.
        proj_weight = state_dict[f"{state_dict_prefix}input_proj.weight"]
        self.input_proj_weight = ttnn.as_tensor(
            proj_weight.transpose(-2, -1).contiguous(),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_name("input_proj"),
        )

        # position_embedding_table is [2, position_embedding_size, hidden_size]: one table per spatial axis.
        position_embedding_table = state_dict[f"{state_dict_prefix}position_embedding_table"]
        self.position_tables = []
        for axis in range(2):
            self.position_tables.append(
                ttnn.as_tensor(
                    position_embedding_table[axis].contiguous(),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                    cache_file_name=cache_name(f"position_embedding_table_{axis}"),
                )
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        """Embed flattened image patches and add 2D positional embeddings.

        Args:
            pixel_values: ttnn.Tensor ``[1, batch, num_patches, 3*patch_size^2]`` flattened patch pixels.
            pixel_position_ids: ttnn.Tensor (int32, ROW_MAJOR) ``[batch, num_patches, 2]`` patch (x, y)
                positions (padding patches are ``(-1, -1)``).
            padding_positions: ttnn.Tensor (uint32/int32, ROW_MAJOR) ``[batch, num_patches]``
                (nonzero = padding patch).

        Returns:
            ttnn.Tensor ``[1, batch, num_patches, hidden_size]`` patch embeddings.
        """
        batch, num_patches, _ = pixel_position_ids.shape

        # Gemma4 applies no normalization, just a [-1, 1] rescale, then projects.
        pixel_values = ttnn.subtract(pixel_values, 0.5)
        pixel_values = ttnn.multiply(pixel_values, 2.0)
        hidden_states = ttnn.matmul(
            pixel_values,
            self.input_proj_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
        )

        # one_hot(position_ids) @ table, summed over the x/y axes == two embedding lookups + add.
        # Clamp the -1 padding ids to 0 so they form valid (uint32) gather indices; the resulting
        # padding rows are zeroed out afterwards via the validity mask.
        clamped_positions = ttnn.clamp(pixel_position_ids, min=0)
        pos_emb = None
        for axis in range(2):
            ids = ttnn.slice(clamped_positions, [0, 0, axis], [batch, num_patches, axis + 1])  # [b, p, 1]
            ids = ttnn.reshape(ids, [batch, num_patches])
            ids = ttnn.typecast(ids, ttnn.uint32)
            axis_emb = ttnn.embedding(ids, self.position_tables[axis], layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            ttnn.deallocate(ids)
            pos_emb = axis_emb if pos_emb is None else ttnn.add(pos_emb, axis_emb)
        ttnn.deallocate(clamped_positions)

        pos_emb = ttnn.unsqueeze_to_4D(pos_emb)  # [batch, num_patches, hidden] -> [1, batch, num_patches, hidden]

        # Zero out the position embeddings of padding patches (reference uses torch.where).
        # padding_positions is {0, 1}, so the validity mask is 1 - padding. (logical_not has no
        # uint8 kernel, so build it arithmetically in bf16.)
        valid = ttnn.reshape(padding_positions, [1, batch, num_patches, 1])  # ROW_MAJOR
        valid = ttnn.to_layout(valid, ttnn.TILE_LAYOUT)
        valid = ttnn.typecast(valid, self.dtype)
        valid = ttnn.multiply(valid, -1.0)
        valid = ttnn.add(valid, 1.0)
        pos_emb = ttnn.multiply(pos_emb, valid)
        ttnn.deallocate(valid)

        out = ttnn.add(hidden_states, pos_emb)
        ttnn.deallocate(pos_emb)
        return out
