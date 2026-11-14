# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace reference for llama-nemotron-rerank-1b-v2 (CPU)."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, prompt_template


def load_tokenizer(model_id=MODEL_ID):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_reference(model_id=MODEL_ID, dtype=torch.float32):
    return AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).eval()


def tokenize_pairs(tokenizer, pairs, max_length=512):
    return [tokenizer(prompt_template(q, p), truncation=True, max_length=max_length)["input_ids"] for q, p in pairs]


@torch.inference_mode()
def reference_scores(model, batch_ids):
    """One sequence at a time, so no padding enters the reference."""
    out = []
    for ids in batch_ids:
        x = torch.tensor([ids])
        out.append(model(input_ids=x, attention_mask=torch.ones_like(x)).logits.float().item())
    return out
