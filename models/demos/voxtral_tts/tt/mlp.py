# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral TTS text decoder SwiGLU MLP for N150 (single device).

Standard SwiGLU: silu(w1(x)) * w3(x) → w2(...)

No tensor parallelism or CCL needed for N150 single device.

Weight keys:
  layers.{N}.feed_forward.w1.weight  [9216, 3072]  gate
  layers.{N}.feed_forward.w2.weight  [3072, 9216]  down
  layers.{N}.feed_forward.w3.weight  [9216, 3072]  value
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVoxtralTextMLP(LightweightModule):
    """Text decoder SwiGLU MLP — single device, no CCL."""

    def __init__(
        self,
        device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.device = device
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2

        pfx = f"layers.{layer_num}.feed_forward"
        cache_name = (
            (lambda n: weight_cache_path / f"{pfx}.{n}")
            if weight_cache_path and not configuration.dummy_weights
            else (lambda _: None)
        )

        def _up(w, name):
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
            )

        w1 = state_dict[f"{pfx}.w1.weight"]  # [9216, 3072]
        w3 = state_dict[f"{pfx}.w3.weight"]  # [9216, 3072]
        w2 = state_dict[f"{pfx}.w2.weight"]  # [3072, 9216]

        self.w1 = _up(w1, "w1_gate")
        self.w3 = _up(w3, "w3_value")
        self.w2 = _up(w2, "w2_down")

    def forward(self, x: ttnn.Tensor, mode=None) -> ttnn.Tensor:
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)
        return out
