# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of ``Lfm2VlMultiModalProjector``.

Unlike Gemma3's projector (average-pool downsample + RMSNorm + linear), LFM2-VL uses a
"pixel-unshuffle" (space-to-depth) downsample with **no** normalization
(``projector_use_layernorm=False``)::

    x = pixel_unshuffle(vision_features, factor=downsample_factor)   # [B, S/f^2, C*f^2]
    x = linear_2(gelu(linear_1(x)))

The pixel-unshuffle itself is a pure data-reshuffle (gather non-contiguous 2x2 patch
blocks into the channel dim) with no arithmetic; since it does not map cleanly onto a
sequence of 4D ttnn permute/reshape ops, it is done host-side (cheap: a handful of KB of
data) while the two projector matmuls (the actual FLOPs) run on-device via ttnn, matching
the pattern used for ``ShortConv``'s host-resident state bookkeeping.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.multimodal.lfm25_vl.reference.functional import pixel_unshuffle


class TtLfm2VlMultiModalProjector(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        vision_dim,
        projector_hidden_size,
        text_dim,
        downsample_factor,
        weight_cache_path,
        dtype,
        configuration,
        bias=True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.downsample_factor = downsample_factor
        self.vision_dim = vision_dim
        self.in_features = vision_dim * downsample_factor * downsample_factor

        cache_name = (
            (lambda _: None)
            if configuration.dummy_weights or weight_cache_path is None
            else (lambda name: weight_cache_path / f"{state_dict_prefix}.{name}")
        )

        def load_linear(name, in_features, out_features, has_bias):
            w = state_dict[f"{state_dict_prefix}.{name}.weight"]  # [out, in]
            assert w.shape == (out_features, in_features), (name, w.shape, out_features, in_features)
            weight = ttnn.as_tensor(
                w.transpose(-1, -2).contiguous(),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"{name}.weight"),
            )
            bias_t = None
            bias_key = f"{state_dict_prefix}.{name}.bias"
            if has_bias and bias_key in state_dict:
                bias_t = ttnn.as_tensor(
                    state_dict[bias_key].reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(f"{name}.bias"),
                )
            return weight, bias_t

        self.linear_1_w, self.linear_1_b = load_linear("linear_1", self.in_features, projector_hidden_size, bias)
        self.linear_2_w, self.linear_2_b = load_linear("linear_2", projector_hidden_size, text_dim, bias)

    def forward_sequence(self, unshuffled_features: torch.Tensor) -> ttnn.Tensor:
        """Run the two projector linears on an already pixel-unshuffled host sequence.

        Args:
            unshuffled_features: ``[B, S, vision_dim * factor**2]`` torch tensor
        Returns:
            ttnn tensor ``[B, S, text_dim]``
        """
        assert unshuffled_features.shape[-1] == self.in_features, (
            unshuffled_features.shape,
            self.in_features,
        )
        tt_x = ttnn.from_torch(
            unshuffled_features,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        h = ttnn.linear(tt_x, self.linear_1_w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.linear_1_b is not None:
            h = ttnn.add(h, self.linear_1_b)
        h = ttnn.gelu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(h, self.linear_2_w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.linear_2_b is not None:
            out = ttnn.add(out, self.linear_2_b)
        return out

    def forward(
        self,
        vision_features: ttnn.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            vision_features: ttnn tensor [B, num_patches, vision_dim] (SigLIP2 encoder output).
                When ``height``/``width`` are omitted, ``num_patches`` must be a perfect square.
            height / width: optional explicit patch-grid size (NaFlex rectangular tiles).
        Returns:
            ttnn tensor [B, (H/f)*(W/f), text_dim]
        """
        # Shards are identical (replicated mesh tensor), so read shard 0 instead of composing.
        torch_features = ttnn.to_torch(ttnn.get_device_tensors(vision_features)[0]).float()
        batch, num_patches, vision_dim = torch_features.shape
        if height is None or width is None:
            side = int(round(num_patches**0.5))
            assert side * side == num_patches, f"num_patches={num_patches} is not a perfect square"
            height = width = side
        else:
            assert height * width == num_patches, (height, width, num_patches)

        x = torch_features.reshape(batch, height, width, vision_dim)
        x = pixel_unshuffle(x, factor=self.downsample_factor)
        x = x.reshape(batch, -1, vision_dim * self.downsample_factor * self.downsample_factor)
        return self.forward_sequence(x)
