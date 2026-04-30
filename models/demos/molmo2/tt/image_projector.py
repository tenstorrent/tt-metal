# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 image projector: SwiGLU MLP (no biases) mapping adapter tokens to text dim.

Weight naming:
    image_projector.w1.weight  [12288, 1152]  gate
    image_projector.w3.weight  [12288, 1152]  up
    image_projector.w2.weight  [4096, 12288]  down

Forward: w2(silu(w1(x)) * w3(x))

Standard SwiGLU — no gate/value reversal (unlike the text MLP).
Runs replicated on all T3K devices.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMolmo2ImageProjector(LightweightModule):
    """SwiGLU projector [1152 → 12288 → 4096] for Molmo2 vision adapter."""

    def __init__(self, mesh_device, state_dict, cfg, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.compute_kernel_config = cfg.compute_kernel_config_hifi2_fp16

        prefix = "model.vision_backbone.image_projector"

        if cfg.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"molmo2_proj.{name}"

        def _tt(key, name):
            w = state_dict[f"{prefix}.{key}.weight"].T.unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=cache_name(name),
            )

        self.w1 = _tt("w1", "w1")  # gate
        self.w3 = _tt("w3", "w3")  # up
        self.w2 = _tt("w2", "w2")  # down

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, N_valid, 1152] → [1, 1, N_valid, 4096]."""
        seq_len = x.shape[-2]
        # Only chunk if evenly divisible — N_valid (e.g. 1316) may not be.
        chunked = seq_len >= 1024 and seq_len % 1024 == 0
        if chunked:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        gate = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # silu(gate) * up
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            hidden,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)

        if chunked:
            original_shape = out.shape
            out = ttnn.reshape(
                out,
                (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
            )
        return out
