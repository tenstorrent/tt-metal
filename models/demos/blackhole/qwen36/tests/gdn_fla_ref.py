# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference and input/PCC helpers for the fused GDN kernels.

The reference is the op's registered golden, ``ttnn.get_golden_function(ttnn.transformer.fused_recurrent_gated_delta_rule)``
(``ttnn/ttnn/operations/transformer_golden.py``): FLA's ``naive_recurrent_gated_delta_rule`` math, the exact form the
vLLM ``fused_sigmoid_gating_delta_rule_update`` implements, with per-token states for the multi-token verify kernel. The
golden is checked against FLA naive itself in
``tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gdn_golden.py``.

Contract note (must match the device kernels):
  * FLA naive does NOT L2-normalize q/k and does NOT sigmoid beta — those are the layer's job
    (in FLA, ``use_qk_l2norm_in_kernel`` / ``use_beta_sigmoid_in_kernel``). Our device op does the
    L2-norm + query scale internally, so to compare against naive we L2-normalize q/k here
    (``l2norm_fla``) and pass beta already in (0,1) and g already as log-decay (g<0, decay=exp(g)).
  * scale defaults to Dk**-0.5, applied to q AFTER the L2-norm (matches gdn/tp.py and FLA).
"""

import torch
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule


def naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False):
    """FLA naive's signature (q, k, v, beta, g) over the registered golden's recurrence."""
    return recurrent_gated_delta_rule(
        q, k, v, beta, g, scale=scale, initial_state=initial_state, output_final_state=output_final_state
    )


def naive_recurrent_per_token_state(q, k, v, beta, g, scale=None, initial_state=None):
    """The recurrence with the state after every token: o [B,T,H,V], states [B,T,H,K,V] (state AFTER absorbing token
    t), final_state == states[:, -1]. q/k are expected already L2-normalized (see module note)."""
    return recurrent_gated_delta_rule(
        q, k, v, beta, g, scale=scale, initial_state=initial_state, output_final_state=True, output_per_token_state=True
    )


def l2norm_fla(x, eps=1e-6):
    """FLA in-kernel L2-norm: x / sqrt(sum(x^2) + eps) over the last dim. Matches l2_norm_ttnn
    (rms_norm(x, eps/K) * K**-0.5) used by recurrent_gated_delta_rule_decode_ttnn."""
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def make_gdn_inputs(T, H=32, Dk=128, Dv=128, B=1, seed=0, g_scale=2.0):
    """Post-conv, post-GQA-expand GDN recurrence inputs at real Qwen3.6-27B dims.

    q/k/v [B,T,H,D]; beta [B,T,H] in (0,1) (sigmoid range); g [B,T,H] negative log-decay
    (decay = exp(g) in (0,1)). Convention matches test_gdn_chunk_recurrent_parity.py.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, Dk, generator=gen, dtype=torch.float32)
    k = torch.randn(B, T, H, Dk, generator=gen, dtype=torch.float32)
    v = torch.randn(B, T, H, Dv, generator=gen, dtype=torch.float32)
    beta = torch.rand(B, T, H, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, H, generator=gen, dtype=torch.float32) * g_scale
    return q, k, v, beta, g


def pcc(a, b):
    """Pearson correlation of two tensors (flattened, double). Matches the repo's compute_pcc."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (a @ b / denom).item()
