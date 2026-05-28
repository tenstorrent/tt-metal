# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Conformer feature-projection block.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_feature_projection_forward``,
which is the very front of the W2v-BERT-2.0 speech encoder::

    x = LayerNorm(x)            # nn.LayerNorm(feature_size, eps=1e-5)
    x = Linear(x)               # nn.Linear(feature_size -> hidden), bias=True
    x = Dropout(x)              # no-op at eval

For the -large config: ``feature_size = 160``, ``hidden = 1024``,
``layer_norm_eps = 1e-5``. Dropout is omitted (eval-time).

This module composes the shared ``LayerNorm`` TTNN block followed by a
``ttnn.linear`` up-projection. Weight loading mirrors
``tt/conformer_ffn.py`` (PyTorch ``nn.Linear.weight`` shape
``(out, in)`` is transposed to ``(in, out)`` for ``ttnn.linear``).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm


class ConformerFeatureProjection(LightweightModule):
    """LayerNorm + Linear front end of the W2v-BERT-2.0 speech encoder.

    Args:
        device: ttnn device or mesh device.
        layer_norm_weight: tensor of shape ``(feature_size,)`` -- LayerNorm gamma.
        layer_norm_bias:   tensor of shape ``(feature_size,)`` -- LayerNorm beta.
        projection_weight: tensor of shape ``(hidden, feature_size)`` -- PyTorch
            ``nn.Linear.weight`` convention.
        projection_bias:   tensor of shape ``(hidden,)``.
        eps: LayerNorm epsilon (default 1e-5, matches HF v2-Large).
        weight_dtype: storage dtype for weights / biases on device.
        weight_memory_config: where to place weights / biases (default DRAM).
    """

    def __init__(
        self,
        device,
        layer_norm_weight,
        layer_norm_bias,
        projection_weight,
        projection_bias,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        hidden, feature_size = projection_weight.shape
        self.feature_size = int(feature_size)
        self.hidden = int(hidden)

        # Reuse the shared LayerNorm TTNN block (matches the layernorm_forward
        # reference helper used in conformer_feature_projection_forward).
        self.layer_norm = LayerNorm(
            device=device,
            dim=self.feature_size,
            weight=layer_norm_weight,
            bias=layer_norm_bias,
            eps=eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # ttnn.linear expects weights laid out as ``(in_features, out_features)``,
        # i.e. the transpose of the PyTorch ``nn.Linear.weight``.
        self.projection_weight = ttnn.from_torch(
            projection_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.projection_bias = ttnn.from_torch(
            projection_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # HiFi4 + fp32 dest accumulation for the matmul, matching the other
        # matmul-heavy blocks (conformer_ffn, seamless_ffn) in this model.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the feature projection on a TILE_LAYOUT ttnn tensor.

        ``x`` has shape ``[..., feature_size]``. Output is ``[..., hidden]``.
        """
        normed = self.layer_norm(x)
        projected = ttnn.linear(
            normed,
            self.projection_weight,
            bias=self.projection_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(normed)
        return projected
