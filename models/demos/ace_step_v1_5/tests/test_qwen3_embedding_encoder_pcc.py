# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC: TTNN Qwen3-0.6B encoder vs HuggingFace ``AutoModel`` (last hidden state)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

import ttnn
from models.demos.ace_step_v1_5.checkpoint_paths import ACE_STEP_CHECKPOINT_DIR_ENV, resolve_qwen3_embedding_model_dir
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder


def _ckpt_dir() -> Path | None:
    return resolve_qwen3_embedding_model_dir()


_SKIP_REASON = (
    "Qwen3-Embedding-0.6B not found. Clone ACE-Step next to tt-metal "
    "(…/ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/) or set "
    f"{ACE_STEP_CHECKPOINT_DIR_ENV} to the checkpoints directory."
)


@pytest.mark.skipif(_ckpt_dir() is None, reason=_SKIP_REASON)
def test_qwen3_encoder_pcc_vs_torch(device):
    ckpt = _ckpt_dir()
    assert ckpt is not None
    text_dir = ckpt

    tok = AutoTokenizer.from_pretrained(str(text_dir))
    prompt = "lofi hip hop, warm vinyl"
    tokens = tok(prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attn = tokens["attention_mask"]

    ref = AutoModel.from_pretrained(str(text_dir), torch_dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        y_ref = ref(input_ids=input_ids, attention_mask=attn).last_hidden_state.float().numpy()

    dev = device
    enc = TtQwen3EmbeddingEncoder(
        device=dev,
        hf_model_dir=str(text_dir),
        qwen_safetensors_path=str(text_dir / "model.safetensors"),
    )
    y_tt = enc.forward(input_ids.numpy().astype(np.uint32), attn.numpy().astype(np.float32))
    y_tt_np = ttnn.to_torch(y_tt).float().numpy()
    # TTNN returns [B,1,S,H] vs torch [B,S,H]
    y_tt_np = y_tt_np.reshape(y_ref.shape)

    a = y_ref.reshape(-1).astype(np.float64)
    b = y_tt_np.reshape(-1).astype(np.float64)
    pearson = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    max_abs = float(np.max(np.abs(a - b)))
    # ~0.99+ vs ``AutoModel`` in bf16 (matches TTNN weights/math width).
    assert pearson >= 0.99, f"Pearson {pearson:.4f} below threshold; rmse={rmse:.4g} max_abs={max_abs:.4g}"
