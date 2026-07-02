"""
Vision aligner for Janus-Pro-7B.
Projects vision encoder output (hidden_size) to text embedding dim (projection_dim).
HF reference: JanusVisionAlignerMLP in transformers.models.janus.modeling_janus
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProVisionAligner(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,  # "model.aligner."
        weight_cache_path,
        dtype,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args

        num_hidden = max(0, args.vision_aligner_depth - 1)

        def load_linear(name):
            w = torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)
            cache = None if args.dummy_weights else weight_cache_path / f"{state_dict_prefix}{name}.weight"
            weight = ttnn.as_tensor(
                w,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache,
            )
            bias = None
            if f"{state_dict_prefix}{name}.bias" in state_dict:
                b = state_dict[f"{state_dict_prefix}{name}.bias"]
                bias_cache = None if args.dummy_weights else weight_cache_path / f"{state_dict_prefix}{name}.bias"
                bias = ttnn.as_tensor(
                    b,
                    dtype=dtype,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=bias_cache,
                )
                bias = ttnn.reshape(bias, [1, -1])
            return weight, bias

        self.fc1_weight, self.fc1_bias = load_linear("fc1")
        self.hidden_layers = []
        for i in range(num_hidden):
            w, b = load_linear(f"hidden_layers.{i}")
            self.hidden_layers.append((w, b))

    def _linear(self, x, weight, bias):
        out = ttnn.linear(
            x,
            weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if bias is not None:
            out = ttnn.add(out, bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [1, B, seq, vision_dim] — output of TtJanusProVisionModel (ln_post)
        x = self._linear(x, self.fc1_weight, self.fc1_bias)
        for weight, bias in self.hidden_layers:
            x = ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = self._linear(x, weight, bias)
        return x
