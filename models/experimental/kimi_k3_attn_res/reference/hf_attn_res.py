# SPDX-FileCopyrightText: © 2026 Moonshot AI
# SPDX-License-Identifier: LicenseRef-Kimi-K3-License
#
# Kimi K3 License, https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE
#
# Permission is hereby granted, free of charge, to any person (the "Licensee")
# obtaining a copy of this software [...] to deal in the Software without
# restriction. [...] The above copyright notice and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Upstream Kimi K3 attention-residual read, vendored as the bringup oracle.

`_apply_attn_res` below is byte-identical to `modeling_kimi_linear.py` in the
`moonshotai/Kimi-K3` repository. It must stay that way: it is the only
independent ground truth for `torch_functional/attn_res.py`, and every numeric
gate in `API_SPEC.md` is rooted here. Edit it and the ladder measures nothing.

The two shim classes exist so the vendored function can be called without
importing `transformers` or instantiating a `KimiRMSNorm` — it only ever touches
`norm.weight`, `norm.variance_epsilon` and `proj.weight`.
"""

import torch


def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    """
    prefix_sum:     (num_tokens, hidden_size)
    block_residual: (num_tokens, num_blocks, hidden_size)
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


class _NormShim:
    def __init__(self, weight, variance_epsilon):
        self.weight = weight
        self.variance_epsilon = variance_epsilon


class _ProjShim:
    def __init__(self, weight):
        self.weight = weight


def hf_attn_res(prefix_sum, block_residual, norm_weight, proj_weight, eps):
    """Call the vendored read with plain tensors.

    Args:
        prefix_sum: [N, d] live residual stream.
        block_residual: [N, S, d] sealed snapshots. S == 0 is legal.
        norm_weight: [d] `*_res_norm.weight`.
        proj_weight: [1, d] `*_res_proj.weight`.
        eps: `rms_norm_eps`.

    Returns:
        [N, d] in `prefix_sum.dtype`.
    """
    return _apply_attn_res(
        prefix_sum,
        block_residual,
        _ProjShim(proj_weight),
        _NormShim(norm_weight, eps),
    )
