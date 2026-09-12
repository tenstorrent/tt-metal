# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# CohereLMHead for Command-R (c4ai-command-r-v01) — bounty tt-metal#49307 track.
#
# Verified deltas vs the stock tt_transformers LMHead (sources in README.md):
#   1. logit_scale = 0.0625 applied POST-linear, pre-loss
#      (HF v4.39.3 modeling_cohere.py: line 1027 self.logit_scale = config.logit_scale;
#       line 1114 logits = logits * self.logit_scale).
#   2. Tied embeddings: config tie_word_embeddings=true — the raw checkpoint carries NO
#      separate lm_head.weight tensor; the tied weight surfaces as BOTH
#      model.embed_tokens.weight and lm_head.weight in the HF model state_dict
#      (shared storage), so the standard conversion (standardize_hf_keys +
#      map_hf_to_meta_keys: embed_tokens -> tok_embeddings, lm_head -> output)
#      yields both TT keys unchanged — no load_checkpoints.py change needed.
#
# Wraps the proven LMHead (linear -> concat -> tt_all_reduce) and applies the
# scalar after its forward, mirroring HF's post-linear application point.
# Wired into Transformer via the family dispatch in model.py (args.model_type ==
# "cohere"); logit_scale resolves from ModelArgs (config.json logit_scale=0.0625).

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.lm_head import LMHead


class CohereLMHead(LightweightModule):
    def __init__(self, *lm_head_args, logit_scale=None, **lm_head_kwargs):
        super().__init__()
        # Resolution order: explicit kwarg -> ModelArgs (config.json) -> 0.0625.
        args = lm_head_kwargs.get("args", None)
        resolved = logit_scale
        if resolved is None and args is not None:
            resolved = getattr(args, "logit_scale", None)
        self.logit_scale = 0.0625 if resolved is None else float(resolved)
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
