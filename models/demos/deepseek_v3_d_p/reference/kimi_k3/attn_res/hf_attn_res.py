# SPDX-FileCopyrightText: Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team.
# SPDX-License-Identifier: MIT
# All rights reserved.
#
# The governing terms are LICENSE-Kimi-K3 in this folder — MIT plus a Model-as-a-Service
# and an attribution condition — which has no SPDX identifier of its own, so the tag above
# names the license it extends. The header below is reproduced verbatim from the
# HuggingFace file; `_apply_attn_res` is not part of the DeepSeek-V3-adapted code, so it
# falls under the Kimi K3 License arm rather than the Apache-2.0 arm.
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

"""The Kimi K3 attention-residual read, vendored from HuggingFace.

Provenance, so the extract is checkable rather than asserted:

  * source: `modeling_kimi_linear.py`, `moonshotai/Kimi-K3` on HuggingFace
  * that file's sha256: `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`
  * `_apply_attn_res` occupies lines 1075-1088 of its 1314 lines
  * `_apply_attn_res` below is byte-identical to those lines — verified by diff,
    not by transcription. Keep it that way; edit it and it measures nothing.
  * `LICENSE-Kimi-K3` beside this file is the repository's LICENSE, whose published
    sha256 is `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887`. The
    in-tree copy hashes differently because `trailing-whitespace` and `end-of-file-fixer`
    took one trailing space off line 30 and added a final newline. Two whitespace bytes;
    the license text is unmodified. Every other vendored LICENSE under `models/` is
    normalized the same way — only `*.patch` is excluded from those hooks.

This is the only read in `reference/kimi_k3/attn_res/` written by someone other than us, which
is what makes it worth vendoring rather than paraphrasing: it is the evidence that the
equation we implement is the equation the model defines. Two things it cannot do, which
is why the other two files exist. It takes one softmax over the whole candidate set, so
it has no seam at which the device's split into sealed and live halves could be compared.
And it takes `res_norm.weight` and `res_proj.weight` separately, so it cannot check a
query that has folded them.

Nothing of ours lives here — the shims that let the function be called with plain
tensors are in `hf_walk.py` — so the license boundary above is the file boundary.
"""

import torch


def _apply_attn_res(running_sum, block_residual, proj, norm):
    """
    running_sum:     (num_tokens, hidden_size)
    block_residual: (num_tokens, num_blocks, hidden_size)
    """
    v = torch.cat((block_residual, running_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)
