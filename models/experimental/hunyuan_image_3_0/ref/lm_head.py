# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 text-generation LM head — the vocabulary
# projection that turns backbone hidden states into next-token logits. Golden for
# `tt/lm_head.py`.
#
# Extracted from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     HunyuanImage3ForCausalMM.__init__ : self.lm_head = nn.Linear(hidden, vocab, bias=False)
#     HunyuanImage3ForCausalMM.forward  : logits = self.lm_head(self.model.ln_f(hidden_states))
#
# `tie_word_embeddings=False`, so `lm_head.weight` ([V, H]) is a distinct checkpoint
# tensor. ln_f is applied by the backbone (apply_final_norm=True) before this head, so
# the reference here is exactly the bias-free linear projection.

import torch


def lm_head_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference LM-head projection: hidden [B, S, H] -> logits [B, S, V].

    Byte-for-byte the upstream `nn.Linear(bias=False)` forward — i.e.
    `F.linear(hidden, weight)` with `weight` in checkpoint layout [V, H]. This is the
    golden the device `HunyuanTtLMHead` (bf16 matmul) is PCC-gated against.
    """
    return torch.nn.functional.linear(hidden, weight)
