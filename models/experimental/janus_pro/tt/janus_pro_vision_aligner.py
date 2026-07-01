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
        tt_ccl,
        args,
        state_dict,
        state_dict_prefix,  # "model.aligner."
        weight_cache_path,
        dtype,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.model_config = args.get_model_config()

        num_hidden = max(0, args.vision_aligner_depth - 1)

        def load_linear(name, weight_dim, bias_dim=None):
            w = torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)
            cache = None if args.dummy_weights else weight_cache_path / f"{state_dict_prefix}{name}.weight"
            weight = ttnn.as_tensor(
                w,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=weight_dim)
                if weight_dim is not None
                else ttnn.ReplicateTensorToMesh(mesh_device),
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
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=bias_dim)
                    if bias_dim is not None
                    else ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=bias_cache,
                )
                bias = ttnn.reshape(bias, [1, -1])
            return weight, bias

        self.fc1_weight, self.fc1_bias = load_linear("fc1", weight_dim=-1, bias_dim=-1)
        self.hidden_layers = []
        for i in range(num_hidden):
            w, b = load_linear(f"hidden_layers.{i}", weight_dim=-2, bias_dim=None)
            self.hidden_layers.append((w, b))

    def _linear_with_reduce(self, x, weight, bias):
        out = ttnn.linear(
            x,
            weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.args.num_devices > 1:
            out = ttnn.experimental.all_gather_async(
                out,
                persistent_output_buffer=None,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=4 if self.args.is_galaxy else 1,
                topology=ttnn.Topology.Ring,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            out = ttnn.experimental.fast_reduce_nc(out, dims=[1], output=None, compute_kernel_config=None)
        if bias is not None:
            out = ttnn.add(out, bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, seq, vision_dim] — output of TtJanusProVisionModel (ln_post)
        x = ttnn.linear(
            x,
            self.fc1_weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.fc1_bias is not None:
            x = ttnn.add(x, self.fc1_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        for weight, bias in self.hidden_layers:
            x = ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = self._linear_with_reduce(x, weight, bias)
        return x
