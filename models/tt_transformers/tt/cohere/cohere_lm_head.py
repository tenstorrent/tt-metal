# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal scaffold — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# CohereLMHead for Command-R (c4ai-command-r-v01) — bounty tt-metal#49307 track.
#
# Verified deltas vs the stock tt_transformers LMHead (sources in README.md):
#   1. logit_scale = 0.0625 applied POST-linear, pre-loss
#      (HF v4.39.3 modeling_cohere.py: line 1027 self.logit_scale = config.logit_scale;
#       line 1114 logits = logits * self.logit_scale).
#   2. Tied embeddings: config tie_word_embeddings=true — the checkpoint carries NO
#      separate lm_head.weight tensor (weight_map: 322 tensors); the LM head weight IS
#      model.embed_tokens.weight [256000, 8192] F16. TODO(P5): confirm how
#      load_checkpoints.py should alias the tied weight into the LMHead loader.
#
# Scaffold: wraps the proven LMHead (linear -> concat -> tt_all_reduce) and applies the
# scalar after its forward, mirroring HF's post-linear application point.

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.lm_head import LMHead


class CohereLMHead(LightweightModule):
    def __init__(self, *lm_head_args, logit_scale: float = 0.0625, **lm_head_kwargs):
        super().__init__()
        # config.logit_scale = 0.0625 (= 1/16), verified from on-box config.json.
        self.logit_scale = logit_scale
        self.lm_head = LMHead(*lm_head_args, **lm_head_kwargs)

    def forward(self, x: ttnn.Tensor, debug_input_torch=None, debug_weight_torch=None) -> ttnn.Tensor:
        logits = self.lm_head.forward(
            x, debug_input_torch=debug_input_torch, debug_weight_torch=debug_weight_torch
        )
        # HF applies the scale post-linear on fp32 logits before softmax/loss.
        # TODO(PCC): dtype of the multiply (logits leave LMHead as lm_head_dtype /
        # bfloat8_b per lm_head.py) — HF scales in fp32; if PCC < 0.99, cast
        # (ttnn.typecast to fp32) -> multiply -> cast back, and re-validate.
        return ttnn.multiply(logits, self.logit_scale)
