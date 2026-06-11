# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Gemma-4 vision tower.

Mirrors HF ``Gemma4VisionModel``: the full image encoder that maps raw patch pixels to pooled
soft tokens. The pipeline is

    patch embed -> transformer encoder  (``VisionTransformer``, fully on device)
    -> spatial average pooling          (``VisionPooler``)
    -> (optional) ``standardize`` affine ``(x - std_bias) * std_scale`` per hidden dim

This module composes those pieces. ``standardize`` is enabled for Gemma-4 (``config.standardize``
is ``True``); the per-hidden-dim ``std_bias`` / ``std_scale`` buffers are loaded as ttnn tensors
and applied on device. The affine is per-token independent, so applying it to the full pooled
tensor (then stripping padded soft tokens) is identical to the reference, which strips first.

Interface: like ``Gemma4VisionModel.forward``, this takes the host (torch) ``pixel_values`` and
``pixel_position_ids`` and does the host->device upload internally. The encoder runs on device;
the pooler builds its (tiny, integer) pooling matrix on host from the position metadata and runs
the heavy contraction on device.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_encoder import VisionTransformer
from models.demos.gemma4.tt.vision.vision_pooler import VisionPooler


class VisionTower(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        state_dict,
        tt_ccl,
        weight_cache_path,
    ):
        """
        Args:
            args (VisionModelArgs): Model arguments.
            dtype (ttnn.dtype): Data type for the transformer blocks.
            state_dict (dict): State dictionary containing model weights.
            tt_ccl (TT_CCL): Collective-communication helper for the encoder blocks.
            weight_cache_path: Path to the weight cache.
        """
        super().__init__()
        self.args = args
        self.mesh_device = args.mesh_device

        vision_config = args.hf_config.vision_config
        self.pooling_kernel_size = vision_config.pooling_kernel_size

        self.encoder = VisionTransformer(
            args=args,
            dtype=dtype,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            weight_cache_path=weight_cache_path,
        )
        self.pooler = VisionPooler(mesh_device=self.mesh_device, args=args, dtype=ttnn.bfloat16)

        self.is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"

        # Optional post-pool standardization: (x - std_bias) * std_scale per hidden dim.
        self.standardize = getattr(vision_config, "standardize", False)
        self.std_bias = None
        self.std_scale = None
        if self.standardize:
            prefix = args.get_state_dict_prefix("VisionTransformer")  # "visual"
            self.std_bias = self._load_affine(state_dict[f"{prefix}.std_bias"], weight_cache_path, "std_bias")
            self.std_scale = self._load_affine(state_dict[f"{prefix}.std_scale"], weight_cache_path, "std_scale")

    def _load_affine(self, weight, weight_cache_path, name):
        """Load a ``[hidden_size]`` standardize buffer as a ``[1, 1, 1, hidden_size]`` ttnn tensor."""
        cache_name = None
        if not self.args.dummy_weights and weight_cache_path is not None:
            cache_name = weight_cache_path / f"visual.{name}"
        return ttnn.as_tensor(
            weight.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate_mapper(),
            cache_file_name=cache_name,
        )

    def _replicate_mapper(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device) if self.is_mesh_device else None

    def forward(self, pixel_values, pixel_position_ids, seq_len):
        """Encode image patches to pooled soft tokens.

        Args:
            pixel_values (torch.Tensor): Flattened patch pixels ``[batch, num_patches, 3*patch_size^2]``.
            pixel_position_ids (torch.LongTensor): Patch (x, y) positions ``[batch, num_patches, 2]``
                (padding patches are ``(-1, -1)``).
            seq_len (int): Padded sequence length the encoder blocks run at (>= num_patches).

        Returns:
            pooled (ttnn.Tensor): ``[1, batch, output_length, hidden_size]`` scaled soft tokens.
            mask (torch.BoolTensor): ``[batch, output_length]`` (True = valid token); use it to strip
                padded soft tokens (``pooled[mask]``), matching ``Gemma4VisionModel``.
        """
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [batch, num_patches]
        num_patches = pixel_position_ids.shape[1]
        output_length = num_patches // (self.pooling_kernel_size**2)

        mapper = self._replicate_mapper()
        pixel_values_tt = ttnn.from_torch(
            pixel_values.unsqueeze(0),  # [1, batch, num_patches, in_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        position_ids_tt = ttnn.from_torch(
            pixel_position_ids.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        padding_positions_tt = ttnn.from_torch(
            padding_positions.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # Encoder: patch embed + rotary + transformer blocks. Output is sliced back to the true
        # patch count (padding patches included; the pooler masks them out).
        encoder_output = self.encoder(
            pixel_values_tt,
            position_ids_tt,
            padding_positions_tt,
            unpadded_seq_len=num_patches,
            seq_len=seq_len,
        )
        ttnn.deallocate(pixel_values_tt)
        ttnn.deallocate(position_ids_tt)
        ttnn.deallocate(padding_positions_tt)

        # Pooler builds its weights on host from the (torch) position metadata.
        pooled, mask = self.pooler(
            encoder_output,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        ttnn.deallocate(encoder_output)

        # Standardize: (x - std_bias) * std_scale, broadcast over the per-hidden-dim buffers.
        if self.standardize:
            pooled = ttnn.subtract(pooled, self.std_bias)
            pooled = ttnn.multiply(pooled, self.std_scale)

        return pooled, mask
