# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `u_m_t5_layer_norm`
(meituan-longcat/LongCat-Video's `text_encoder.encoder.block.0.layer.0.layer_norm`,
a real `transformers.models.umt5.modeling_umt5.UMT5LayerNorm` -- RMSNorm with a
per-channel scale and NO bias/mean-subtraction: `x * rsqrt(mean(x**2) + eps) * weight`).

Adapts the already-validated `T5RMSNorm` in
`models/tt_dit/encoders/t5/model_t5.py` -- the same class `u_m_t5_block`'s
`T5Attention`/`T5FF` use internally for their own layer norms -- same
rationale/precedent as `u_m_t5_block`/`u_m_t5_dense_gated_act_dense`. It has
no `mesh_axis`/TP concerns (norms are always replicated, never sharded), so
this graduates single-device only (no `_sharded` rung for this component).
"""

from __future__ import annotations

import ttnn
from models.tt_dit.encoders.t5.model_t5 import T5RMSNorm


class TtUMT5LayerNorm:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.module = T5RMSNorm(
            embedding_dim=torch_module.weight.shape[-1],
            norm_eps=torch_module.variance_epsilon,
            bias=False,
            mesh_device=device,
        )
        self.module.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self.module(hidden_states)


def build(device: ttnn.Device, torch_module) -> TtUMT5LayerNorm:
    return TtUMT5LayerNorm(device, torch_module)
