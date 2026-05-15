# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC checks for the lightweight TTNN ACE condition embedding path.

This intentionally tests the text-only path used by
``run_prompt_to_wav.py --fast-preprocess --ttnn-condition-embedding``:

* TTNN Qwen3 embedding encoder
* TTNN ``encoder.text_projector``
* TTNN context-latent concatenation

It does not test the full HF ``prepare_condition`` result because that result
also includes the lyric and timbre transformer encoders, which this lightweight
path explicitly skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

import ttnn
from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder
from models.demos.ace_step_v1_5.ttnn_impl.text_projector import TtAceStepTextProjector, load_text_projector_weight_numpy


def _ckpt_root() -> Path:
    return Path(
        os.environ.get("ACE_STEP_CHECKPOINT_DIR", "~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints")
    ).expanduser()


@pytest.mark.skipif(
    not (_ckpt_root() / "Qwen3-Embedding-0.6B" / "model.safetensors").is_file()
    or not (_ckpt_root() / "acestep-v15-base" / "model.safetensors").is_file()
    or not (_ckpt_root() / "acestep-v15-base" / "silence_latent.pt").is_file(),
    reason="ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.",
)
def test_ttnn_text_condition_embedding_pcc_vs_torch(device):
    ckpt = _ckpt_root()
    text_dir = ckpt / "Qwen3-Embedding-0.6B"
    model_path = ckpt / "acestep-v15-base" / "model.safetensors"

    prompt = "Electronic dance track with deep bass, punchy kick drum, bright synth lead, energetic rhythm"
    dit_instruction = "Fill the audio semantic mask based on the given conditions:"
    metas = {"caption": prompt, "duration": 15.0, "language": "en"}
    text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""

    tok = AutoTokenizer.from_pretrained(str(text_dir))
    tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attn = tokens["attention_mask"]

    ref_qwen = AutoModel.from_pretrained(str(text_dir), torch_dtype=torch.bfloat16).eval()
    with safe_open(str(model_path), framework="pt", device="cpu") as sf:
        text_projector_w = sf.get_tensor("encoder.text_projector.weight").to(torch.bfloat16)

    qwen_tt = TtQwen3EmbeddingEncoder(
        device=device,
        hf_model_dir=str(text_dir),
        qwen_safetensors_path=str(text_dir / "model.safetensors"),
    )
    text_tt = qwen_tt.forward(input_ids.numpy().astype(np.uint32), attn.numpy().astype(np.float32))

    with torch.inference_mode():
        text_ref_hf = ref_qwen(input_ids=input_ids, attention_mask=attn).last_hidden_state.float()

    text_tt_torch = ttnn.to_torch(text_tt).float()
    if text_tt_torch.ndim == 4:
        text_tt_torch = text_tt_torch.squeeze(1)

    # Qwen encoder parity (separate from projector; HF vs TTNN BF16 stack).
    assert_pcc_print("ttnn_qwen3_hidden_vs_hf", text_ref_hf, text_tt_torch, pcc=0.98)

    # Projector PCC: same TTNN Qwen hidden states as production (``run_prompt_to_wav`` path).
    with torch.inference_mode():
        projected_ref = torch.matmul(text_tt_torch.to(torch.bfloat16), text_projector_w.t()).float()

    projector_tt = TtAceStepTextProjector(
        device=device,
        weight_f32_numpy=load_text_projector_weight_numpy(str(model_path)),
        weights_dtype=getattr(ttnn, "bfloat16", None),
        weight_memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
    )
    projected_tt = projector_tt.forward_from_hidden(text_tt, activation_dtype=getattr(ttnn, "bfloat16", None))
    projected_tt_torch = ttnn.to_torch(projected_tt).float()
    if projected_tt_torch.ndim == 4:
        projected_tt_torch = projected_tt_torch.squeeze(1)

    assert_pcc_print("ttnn_text_projector", projected_ref, projected_tt_torch)


@pytest.mark.skipif(
    not (_ckpt_root() / "acestep-v15-base" / "silence_latent.pt").is_file(),
    reason="ACE-Step v1.5 silence_latent.pt not found; set ACE_STEP_CHECKPOINT_DIR.",
)
def test_ttnn_context_latents_match_torch_cat(device):
    ckpt = _ckpt_root()
    silence = torch.load(str(ckpt / "acestep-v15-base" / "silence_latent.pt"), map_location="cpu").to(torch.float32)
    if int(silence.shape[-1]) != 64:
        silence = silence.transpose(1, 2).contiguous()
    frames = 15 * 25
    src_latents = silence[:, :frames, :].contiguous()
    if src_latents.shape[1] < frames:
        rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
        src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()
    chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)
    ref = torch.cat([src_latents, chunk_masks], dim=-1).to(torch.bfloat16).float().numpy()

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    src_tt = ttnn.as_tensor(
        src_latents.numpy(), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
    )
    mask_tt = ttnn.as_tensor(
        chunk_masks.numpy(), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
    )
    ctx_tt = ttnn.concat([src_tt, mask_tt], dim=-1)
    got = ttnn.to_torch(ctx_tt).float().numpy()

    assert_pcc_print(
        "ttnn_context_latents_cat",
        torch.from_numpy(ref),
        torch.from_numpy(got),
    )
