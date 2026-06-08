# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch CPU golden reference for the Qwen3 text encoder from Tongyi-MAI/Z-Image-Turbo.

Loads the model from HuggingFace and runs inference on CPU for PCC comparison.
The pipeline uses hidden_states[-2] (pre-final-norm) as caption features for the DIT.
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128


def load_model():
    """Load Qwen3 text encoder from HuggingFace in bfloat16.

    Returns:
        model: Qwen3Model in eval mode.
    """
    model = AutoModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, use_cache=False
    ).eval()
    print(f"  Loaded text encoder ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return model


def tokenize(prompt, max_length=CAP_TOKENS):
    """Tokenize a prompt with chat template formatting.

    Returns:
        input_ids: [1, max_length] int64 tensor.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(formatted, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")[
        "input_ids"
    ]


@torch.no_grad()
def forward(model, input_ids):
    """Run text encoder and return pre-final-norm hidden states.

    Args:
        model: Qwen3Model from load_model().
        input_ids: [1, seq_len] token ids.

    Returns:
        [seq_len, 2560] float32 tensor (hidden_states[-2], squeezed).
    """
    out = model(input_ids, output_hidden_states=True)
    return out.hidden_states[-2].squeeze(0).float()
