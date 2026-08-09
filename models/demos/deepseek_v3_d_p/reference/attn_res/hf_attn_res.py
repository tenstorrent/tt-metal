# SPDX-FileCopyrightText: Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team.
# SPDX-License-Identifier: LicenseRef-LICENSE-FILE
# All rights reserved.
#
# This source code is licensed under the terms described in LICENSE-Kimi-K3 in
# this folder. The header below is reproduced verbatim from the upstream file;
# `_apply_attn_res` is not part of the DeepSeek-V3-adapted code, so it falls under
# the Kimi K3 License arm rather than the Apache-2.0 arm.
#
# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The multi-head latent attention, MoE gating and sparse MoE block in this file are
# adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py). They have been
# extensively modified and extended for the Kimi-Linear architecture.
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the Kimi K3 License (see the LICENSE file in this repository).
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upstream Kimi K3 attention-residual read, vendored as the external anchor.

Provenance, so the extract is checkable rather than asserted:

  * source: `modeling_kimi_linear.py`, `moonshotai/Kimi-K3` on HuggingFace
  * upstream file sha256: `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`
  * `_apply_attn_res` occupies lines 1075-1088 of that 1314-line file
  * `_apply_attn_res` below is byte-identical to those lines — verified by diff,
    not by transcription. Keep it that way; edit it and the anchor measures nothing.
  * `LICENSE-Kimi-K3` beside this file is the repository's LICENSE, upstream sha256
    `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`. The in-tree copy
    hashes differently because `trailing-whitespace` and `end-of-file-fixer` took one
    trailing space off line 30 and added a final newline. Two whitespace bytes; the
    license text is unmodified. Every other vendored LICENSE under `models/` is
    normalized the same way — only `*.patch` is excluded from those hooks.

This is the *external* anchor, not the ladder's root. `reference/attn_res_reference.py`
is the fp64 ground truth and is pinned by closed forms; what this file adds is the one
thing no reference of ours can supply — evidence that upstream computes the equation
we believe it computes. It runs in fp32 whatever dtype it is handed, because it widens
with `.float()`, so it cannot serve as a precision reference itself.

The two shim classes exist so the vendored function can be called without importing
`transformers` or instantiating a `KimiRMSNorm` — it only ever touches `norm.weight`,
`norm.variance_epsilon` and `proj.weight`.
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
        [N, d] in `prefix_sum.dtype`, computed in fp32 regardless of that dtype.
    """
    return _apply_attn_res(
        prefix_sum,
        block_residual,
        _ProjShim(proj_weight),
        _NormShim(norm_weight, eps),
    )
