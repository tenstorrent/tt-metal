# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC: TTNN Qwen3-0.6B encoder vs HuggingFace ``AutoModel`` (last hidden state).

The PCC is computed over the **real (non-pad) token positions only**:

- Eager :meth:`AceStepQwen3Encoder.forward` delegates to
  :class:`~models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder.TtQwen3EmbeddingEncoder`,
  which applies the same HF-style causal + key-padding mask as ``AutoModel``.
- Pad-position hidden states can still differ slightly; PCC is computed on real tokens only.

So we slice both reference and TTNN outputs to the first ``int(attention_mask.sum())``
positions before PCC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step import AceStepQwen3Encoder as TtQwen3EmbeddingEncoder

# Same default checkpoint root as ``run_prompt_to_wav`` (HF hub cache layout).
_DEFAULT_QWEN_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints" / "Qwen3-Embedding-0.6B"


def _ckpt_dir() -> Path | None:
    d = _DEFAULT_QWEN_DIR.resolve()
    if (d / "model.safetensors").is_file():
        return d
    if any(d.glob("model-*.safetensors")):
        return d
    return None


_SKIP_REASON = (
    f"Qwen3-Embedding-0.6B not found at {_DEFAULT_QWEN_DIR}. "
    "Populate that directory (e.g. run the ACE-Step demo once so weights download there)."
)


@pytest.mark.skipif(_ckpt_dir() is None, reason=_SKIP_REASON)
def test_qwen3_encoder_pcc_vs_torch(device):
    """PCC: AceStepQwen3Encoder (per-token hidden states) vs HF AutoModel.last_hidden_state.

    Sliced to the real (non-pad) token positions before PCC; see module docstring for the
    rationale.
    """
    ckpt = _ckpt_dir()
    assert ckpt is not None
    text_dir = ckpt

    tok = AutoTokenizer.from_pretrained(str(text_dir))
    prompt = "lofi hip hop, warm vinyl"
    tokens = tok(prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attn = tokens["attention_mask"]
    real_len = int(attn[0].sum().item())
    assert real_len > 0, "tokenizer produced no real tokens (only pads); check the prompt"

    ref = AutoModel.from_pretrained(str(text_dir), torch_dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        y_ref = ref(input_ids=input_ids, attention_mask=attn).last_hidden_state.float().numpy()
    # Free the HF reference model before constructing the TTNN encoder so they don't compete for RAM.
    del ref

    mesh_desc = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "(unset)")
    print(
        f"[qwen3_encoder_pcc] TT_MESH_GRAPH_DESC_PATH={mesh_desc} "
        f"ACE_STEP_QWEN_DEBUG={os.environ.get('ACE_STEP_QWEN_DEBUG', '0')}",
        flush=True,
    )

    dev = device
    enc = TtQwen3EmbeddingEncoder(
        device=dev,
        hf_model_dir=str(text_dir),
        qwen_safetensors_path=str(text_dir / "model.safetensors"),
    )
    print(
        f"[qwen3_encoder_pcc] encoder eager={enc._eager_enc is not None} " f"trace_stack={enc._trace_stack_ready}",
        flush=True,
    )
    y_tt = enc.forward(input_ids.numpy().astype(np.uint32), attn.numpy().astype(np.float32))
    y_tt_np = ttnn.to_torch(y_tt).float().numpy()
    # TTNN returns [B, 1, S, H] vs torch [B, S, H] — drop the extra singleton.
    y_tt_np = y_tt_np.reshape(y_ref.shape)

    # Compare only the real (non-pad) positions.
    y_ref_real = y_ref[:, :real_len, :]
    y_tt_real = y_tt_np[:, :real_len, :]
    print(
        f"[qwen3_encoder_pcc] real_len={real_len}/{int(input_ids.shape[1])} "
        f"H={int(y_ref.shape[-1])} — comparing first {real_len} positions only",
        flush=True,
    )
    for ti in range(real_len):
        ok_t, pcc_t = comp_pcc(y_ref_real[:, ti : ti + 1, :], y_tt_real[:, ti : ti + 1, :], pcc=0.99)
        print(
            f"[qwen3_encoder_pcc] token[{ti}] pcc={float(pcc_t):.6f} ok={ok_t}",
            flush=True,
        )

    assert_pcc_print(
        "qwen3_embedding_encoder",
        torch.from_numpy(y_ref_real),
        torch.from_numpy(y_tt_real),
    )
