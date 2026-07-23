# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek transformer block, MLP, and normalization layers."""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, Parameter, RowParallelLinear


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

    def __init__(self, dim: int, inter_dim: int, *, tp_axis_name: str | None = None) -> None:
        super().__init__()
        if tp_axis_name is not None:
            self.w1 = ColumnParallelLinear(
                dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                axis_name=tp_axis_name,
            )
            self.w3 = ColumnParallelLinear(
                dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                axis_name=tp_axis_name,
            )
            self.w2 = RowParallelLinear(
                inter_dim,
                dim,
                has_bias=False,
                input_is_parallel=True,
                axis_name=tp_axis_name,
            )
        else:
            self.w1 = LinearLayer(dim, inter_dim, has_bias=False)
            self.w3 = LinearLayer(dim, inter_dim, has_bias=False)
            self.w2 = LinearLayer(inter_dim, dim, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return ttml.ops.swiglu.swiglu(
            x,
            self.w1.weight.tensor,
            self.w2.weight.tensor,
            self.w3.weight.tensor,
        )


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
        from .moe_sparse_ep import SparseMoEEP

        super().__init__()
        self.attn = MultiHeadLatentAttention(config, rope_params)
        use_tp = bool(getattr(config, "use_tp", False))
        if layer_id < config.n_dense_layers:
            self.ffn = DeepSeekMLP(config.dim, config.inter_dim, tp_axis_name="tp" if use_tp else None)
        else:
            moe_type = str(getattr(config, "moe_type", "sparse_ep")).lower()
            if moe_type == "dense":
                self.ffn = MoE(config)
            elif moe_type == "sparse_ep":
                # Resolve the EP axis: full-model TP → "tp", else moe_axis_name
                # if it points at a real axis with size > 1, else no EP axis.
                mesh = _ttml.maybe_mesh()
                if use_tp:
                    moe_axis_name = "tp"
                else:
                    tp_name = getattr(config, "moe_axis_name", None)
                    moe_axis_name = (
                        tp_name
                        if (
                            tp_name is not None
                            and mesh is not None
                            and mesh.has_axis(tp_name)
                            and mesh.axis_size(tp_name) > 1
                        )
                        else None
                    )

                if moe_axis_name is None:
                    # No usable EP axis (single chip / pure replication):
                    # sparse_ep degenerates to single-device SparseMoE (EP size 1).
                    self.ffn = SparseMoE(config)
                else:
                    self.ffn = SparseMoEEP(config, axis_name=moe_axis_name)
            else:
                raise ValueError(
                    f"DeepSeekBlock: unknown moe_type={moe_type!r}; expected one of "
                    f"'dense', 'sparse_ep' (from DeepSeekConfig.moe_type)"
                )
            self.ffn._debug_layer_id = layer_id
        self.attn_norm = RMSNormLayer(config.dim)
        self.ffn_norm = RMSNormLayer(config.dim)

    def forward(self, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor = None) -> ttml.autograd.Tensor:
        # `mask` is accepted (and unused) only to satisfy the shared block(input, mask)
        # contract used by memory_efficient_runner. MLA is causal-only and generates its
        # causal mask on chip, so nothing is forwarded to attention.
        x = ttml.ops.binary.add(x, self.attn(self.attn_norm(x)))
        x = ttml.ops.binary.add(x, self.ffn(self.ffn_norm(x)))
        return x
