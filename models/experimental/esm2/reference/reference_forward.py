# SPDX-License-Identifier: MIT
"""Oracle bridge: transformers.EsmForMaskedLM reference forward (FP32).

Used by tests and the baseline cross-check. This is the same code path the
porting evaluator uses as its immutable FP32 oracle.
"""
from __future__ import annotations

import torch


def hf_esm2_forward(weights_path: str, input_ids, attention_mask):
    """FP32 transformers forward -> (logits, final_hidden_state) torch fp32."""
    from transformers import EsmForMaskedLM

    model = EsmForMaskedLM.from_pretrained(weights_path).eval()
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    am = torch.as_tensor(attention_mask, dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, attention_mask=am).logits
        hidden = model.esm(ids, attention_mask=am).last_hidden_state
    return logits, hidden
