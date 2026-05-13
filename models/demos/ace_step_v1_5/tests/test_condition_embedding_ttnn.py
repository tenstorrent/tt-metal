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
from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder
from models.demos.ace_step_v1_5.ttnn_impl.text_projector import TtAceStepTextProjector, load_text_projector_weight_numpy


def _ckpt_root() -> Path:
    return Path(
        os.environ.get("ACE_STEP_CHECKPOINT_DIR", "~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints")
    ).expanduser()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.array_equal(aa, bb):
        return 1.0
    return float(np.corrcoef(aa, bb)[0, 1])


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
        text_projector_w = sf.get_tensor("encoder.text_projector.weight").to(torch.float32)

    with torch.inference_mode():
        text_ref = ref_qwen(input_ids=input_ids, attention_mask=attn).last_hidden_state.to(torch.float32)
        # The TTNN path uses BF16 activations/weights, so compare against BF16 matmul rather than
        # an FP32 projector reference.
        projected_ref = torch.matmul(text_ref.to(torch.bfloat16), text_projector_w.to(torch.bfloat16).t())
        projected_ref = projected_ref.float().cpu().numpy()

    qwen_tt = TtQwen3EmbeddingEncoder(
        device=device,
        hf_model_dir=str(text_dir),
        qwen_safetensors_path=str(text_dir / "model.safetensors"),
    )
    text_tt = qwen_tt.forward(input_ids.numpy().astype(np.uint32), attn.numpy().astype(np.float32))
    projector_tt = TtAceStepTextProjector(
        device=device,
        weight_f32_numpy=load_text_projector_weight_numpy(str(model_path)),
        weights_dtype=getattr(ttnn, "bfloat16", None),
        weight_memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
    )
    projected_tt = projector_tt.forward_from_hidden(text_tt, activation_dtype=getattr(ttnn, "bfloat16", None))
    projected_tt_np = ttnn.to_torch(projected_tt).float().reshape(projected_ref.shape).numpy()

    pearson = _pearson(projected_ref, projected_tt_np)
    rmse = float(np.sqrt(np.mean((projected_ref.astype(np.float64) - projected_tt_np.astype(np.float64)) ** 2)))
    # The projector widens Qwen hidden states from 1024 -> 2048 and amplifies the existing BF16
    # Qwen encoder drift. This threshold is a regression guard for the implemented end-to-end
    # lightweight condition path, not a bit-exact projector-only test.
    assert pearson >= 0.98, f"projected condition PCC={pearson:.6f}, rmse={rmse:.6g}"


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

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)
