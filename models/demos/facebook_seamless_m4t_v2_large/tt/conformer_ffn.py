# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Conformer feed-forward block used by the
SeamlessM4T-v2 speech encoder (W2v-BERT-2.0 style).

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_ffn_forward``,
which computes the inner FFN of ``SeamlessM4Tv2ConformerFeedForward``::

    h = intermediate_dense(x) + b1     # Linear(hidden, intermediate)
    h = swish(h)                       # SiLU
    y = output_dense(h) + b2           # Linear(intermediate, hidden)

For the ``-large`` configuration: ``hidden = 1024``,
``intermediate = config.speech_encoder_intermediate_size = 4096`` and the
activation is swish (silu). Both projections have a bias.

This block mirrors the structure of ``SeamlessFfn`` (same two-projection
layout) but swaps the activation for swish and uses the smaller 4x expansion
ratio characteristic of the Conformer FFN inside the speech encoder layer.

NOTE: the surrounding half-step residual (``x + 0.5 * FFN(LayerNorm(x))``)
lives at the parent encoder-layer level and is NOT part of this block.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class ConformerFfn(LightweightModule):
    """Two-projection feed-forward block with a swish activation in between.

    Args:
        device: ttnn device or mesh device.
        intermediate_weight: tensor of shape ``(intermediate, hidden)`` -- the
            first projection weight in PyTorch convention.
        intermediate_bias:   tensor of shape ``(intermediate,)``.
        output_weight:       tensor of shape ``(hidden, intermediate)`` -- the
            second projection weight in PyTorch convention.
        output_bias:         tensor of shape ``(hidden,)``.
        weight_dtype: storage dtype for weights / biases on device.
        weight_memory_config: where to store weights / biases (default DRAM).
    """

    def __init__(
        self,
        device,
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias,
        act_fn: str = "swish",
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        # Select activation in the same way as the reference
        # ``conformer_ffn_forward(act_fn=...)``. ``"swish"`` and ``"silu"`` are
        # aliases (both -> SiLU). The Conformer adapter layer uses ``"relu"``.
        if act_fn in ("swish", "silu"):
            self._act = ttnn.silu
        elif act_fn == "gelu":
            self._act = ttnn.gelu
        elif act_fn == "relu":
            self._act = ttnn.relu
        else:
            raise ValueError(f"Unsupported activation: {act_fn!r}")

        intermediate, hidden = intermediate_weight.shape
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)

        # ttnn.linear expects weights laid out as ``(in_features, out_features)``,
        # i.e. the transpose of the PyTorch ``nn.Linear.weight``.
        self.intermediate_weight = ttnn.from_torch(
            intermediate_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.intermediate_bias = ttnn.from_torch(
            intermediate_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.output_weight = ttnn.from_torch(
            output_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.output_bias = ttnn.from_torch(
            output_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # HiFi4 + fp32 dest accumulation: matches the per-block precision
        # contract documented in the bring-up rules for matmul-heavy ops.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the Conformer FFN forward pass on a TILE_LAYOUT ttnn tensor.

        ``x`` has shape ``[..., hidden]``. Output has the same shape.
        """
        hidden = ttnn.linear(
            x,
            self.intermediate_weight,
            bias=self.intermediate_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        activated = self._act(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        output = ttnn.linear(
            activated,
            self.output_weight,
            bias=self.output_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(activated)
        return output
