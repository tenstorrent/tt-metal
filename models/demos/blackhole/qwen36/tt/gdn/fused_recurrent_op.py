# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host wrapper for ttnn.transformer.fused_recurrent_gated_delta_rule. qwen36-only."""
import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn


def fused_recurrent_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    scale=None,
    initial_state=None,
    device=None,
    output_per_token_state=False,
    high_precision=True,
):
    """Fused recurrent gated delta rule. Returns (o [B, T, H, V], final or per-token state)."""
    Kd = q.shape[-1]
    if scale is None:
        scale = Kd**-0.5
    if high_precision:
        q = ttnn.typecast(q, ttnn.float32)
        k = ttnn.typecast(k, ttnn.float32)
        v = ttnn.typecast(v, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)
    qn = l2_norm_ttnn(q, dim=-1)
    kn = l2_norm_ttnn(k, dim=-1)
    o, state = ttnn.transformer.fused_recurrent_gated_delta_rule(
        qn,
        kn,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        output_per_token_state=output_per_token_state,
    )
    return o, state
