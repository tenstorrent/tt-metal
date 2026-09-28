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
    B, H, Kd = q.shape[0], q.shape[2], q.shape[-1]
    if device is not None:
        # The device op places one head per core and TT_FATALs deep in the program factory
        # otherwise; check it here so the failure names the caller's shape.
        cores = device.compute_with_storage_grid_size()
        assert (
            B * H <= cores.x * cores.y
        ), f"fused recurrent GDN needs B*H ({B}*{H}={B * H}) <= {cores.x * cores.y} compute cores"
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
