# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek transformer block, MLP, and normalization layers."""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter


class RMSNormLayer(AbstractModuleBase):
    """Root Mean Square Layer Normalization."""

    def __init__(self, features: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = Parameter(ttml.init.ones()((1, 1, 1, features)))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return ttml.ops.rmsnorm.rmsnorm(x, self.gamma.tensor, self.epsilon)


class DeepSeekMLP(AbstractModuleBase):
    """SwiGLU feed-forward network: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = LinearLayer(dim, inter_dim, has_bias=False)
        self.w3 = LinearLayer(dim, inter_dim, has_bias=False)
        self.w2 = LinearLayer(inter_dim, dim, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return self.w2(ttml.ops.binary.mul(ttml.ops.unary.silu(self.w1(x)), self.w3(x)))


class DeepSeekBlock(AbstractModuleBase):
    """Pre-norm residual transformer block.

    First n_dense_layers use dense MLP; remaining layers use MoE.
    """

    def __init__(self, layer_id: int, config, rope_params) -> None:
        # Lazy imports to avoid circular dependency (mla/moe import RMSNormLayer from here)
        import ttml as _ttml
        from .mla import MultiHeadLatentAttention
        from .moe import MoE
        from .moe_sparse import SparseMoE
        from .moe_sparse_tp import SparseMoETP

        super().__init__()
        self.attn = MultiHeadLatentAttention(config, rope_params)
        if layer_id < config.n_dense_layers:
            self.ffn = DeepSeekMLP(config.dim, config.inter_dim)
        else:
            moe_type = str(getattr(config, "moe_type", "sparse")).lower()
            if moe_type == "dense":
                self.ffn = MoE(config)
            elif moe_type == "sparse":
                tp_name = getattr(config, "moe_tp_axis_name", None)
                mesh = _ttml.maybe_mesh()
                use_moe_tp = (
                    tp_name is not None and mesh is not None and mesh.has_axis(tp_name) and mesh.axis_size(tp_name) > 1
                )
                self.ffn = SparseMoETP(config) if use_moe_tp else SparseMoE(config)
            else:
                raise ValueError(
                    f"DeepSeekBlock: unknown moe_type={moe_type!r}; expected 'sparse' or 'dense' "
                    f"(from DeepSeekConfig.moe_type)"
                )
            self.ffn._debug_layer_id = layer_id
        self.attn_norm = RMSNormLayer(config.dim)
        self.ffn_norm = RMSNormLayer(config.dim)

    def forward(self, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = ttml.ops.binary.add(x, self.attn(self.attn_norm(x), mask))
        x = ttml.ops.binary.add(x, self.ffn(self.ffn_norm(x)))
        return x
