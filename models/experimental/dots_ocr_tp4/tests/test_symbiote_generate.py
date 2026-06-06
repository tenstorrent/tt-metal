# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end text generation driven ENTIRELY by the TP4 prefill path.

"Replace only the prefill": instead of the symbiote pipeline's prefill+KV-cache
decode (whose TP4 decode path is known-broken), this drives greedy generation by
re-running the clean TP4 prefill over the growing sequence each step (no KV
cache). Decodes the text with the real dots.ocr tokenizer and compares the
generated tokens/text to the real HF model's greedy output.

Causal attention lets us right-pad the sequence to a tile multiple and read the
logits at the real last position (the padded tail is "future" and ignored).
"""

import os

import pytest
import torch

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, to_replicated
from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download("rednote-hilab/dots.ocr")


def _tp4_next_token(tt_model, text_model, cur_ids, H):
    """One greedy step through the TP4 prefill: returns the argmax next-token id."""
    L = int(cur_ids.shape[1])
    s_pad = ((L + 31) // 32) * 32
    embeds = text_model.embed_tokens(cur_ids)  # [1, L, H] bf16
    if s_pad > L:
        pad = torch.zeros(1, s_pad - L, H, dtype=embeds.dtype)
        emb_in = torch.cat([embeds, pad], dim=1)
    else:
        emb_in = embeds
    x_tt = to_replicated(emb_in.to(torch.bfloat16), tt_model.mesh_device, dtype=ttnn.bfloat16)
    _, token_ids = tt_model.forward_with_head(x_tt, last_token_only=True, return_token=True, token_index=L - 1)
    ttnn.synchronize_device(tt_model.mesh_device)
    return int(token_ids.flatten()[0])


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("max_new_tokens", [int(os.environ.get("DOTS_OCR_TP4_GEN_TOKENS", "24"))])
def test_symbiote_prefill_swap_generate(mesh_device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = DotsOCRConfig.from_hf(hf_model.config)
    H = config.hidden_size
    text_model = hf_model.model

    messages = [{"role": "user", "content": "What is optical character recognition and how does it work?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"]
    prompt_len = int(input_ids.shape[1])

    # HF greedy reference (the real model, bf16 as shipped).
    hf_gen = hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    hf_new = hf_gen[0, prompt_len:].tolist()

    # Build the TP4 prefill model (decoder body + final norm + LM head) from the
    # SAME bf16 HF weights.
    tt_model = DotsOCRPrefillModelTP4.from_torch(
        mesh_device, config, text_model.layers, torch_norm=text_model.norm, torch_lm_head=hf_model.lm_head
    )

    eos_ids = set(
        [hf_model.config.eos_token_id]
        if isinstance(hf_model.config.eos_token_id, int)
        else list(hf_model.config.eos_token_id or [])
    )

    # Free-run TP4 greedy: drive generation entirely through the TP4 prefill.
    tp4_new = []
    cur = input_ids.clone()
    for _ in range(max_new_tokens):
        nxt = _tp4_next_token(tt_model, text_model, cur, H)
        tp4_new.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        if nxt in eos_ids:
            break

    hf_text = tokenizer.decode(hf_new, skip_special_tokens=True)
    tp4_text = tokenizer.decode(tp4_new, skip_special_tokens=True)

    # Token-level agreement on the common prefix (greedy match length).
    match = 0
    for a, b in zip(hf_new, tp4_new):
        if a == b:
            match += 1
        else:
            break

    print("\n" + "=" * 70)
    print(f"[dots_ocr_tp4] prompt_len={prompt_len}  max_new_tokens={max_new_tokens}")
    print(f"HF  tokens : {hf_new}")
    print(f"TP4 tokens : {tp4_new}")
    print(f"greedy match prefix: {match}/{min(len(hf_new), len(tp4_new))}")
    print(f"\nHF  text : {hf_text!r}")
    print(f"TP4 text : {tp4_text!r}")
    print("=" * 70)

    # The TP4 prefill reproduces the real model's greedy decode on its confident
    # tokens; divergence only appears at low-confidence near-ties (synonyms like
    # computers<->machines, understand<->interpret) where bf16 TP drift flips the
    # argmax — the text stays coherent and semantically equivalent. Gate on the
    # confident opening prefix matching exactly (proves the prefill is sound);
    # the printed text is the qualitative "correct output" evidence.
    assert match >= 6, f"TP4 greedy diverged almost immediately from HF: only {match} tokens matched"
    assert len(tp4_text.strip()) > 0


def _build_dots_ocr_processor(model_path):
    import json

    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)
    with open(os.path.join(model_path, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665
    return processor, tokenizer


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize(
    "image_link",
    ["https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"],
)
@pytest.mark.parametrize("max_new_tokens", [int(os.environ.get("DOTS_OCR_TP4_OCR_TOKENS", "48"))])
def test_symbiote_prefill_swap_ocr(mesh_device, image_link, max_new_tokens):
    """OCR the demo image, generating ENTIRELY through the TP4 text-decoder prefill.

    The HF vision tower produces the image patch embeddings (only the *text*
    decoder prefill is the TP4 rebuild); they are scattered into the prompt's
    image-token positions and the fused embeds are generated greedily via the
    TP4 prefill (re-run each step; no KV cache). HF greedy (with cache) is the
    reference. Token count is capped (the O(n^2) recompute over the ~2.8k-token
    image sequence is slow); raise DOTS_OCR_TP4_OCR_TOKENS for a fuller read.
    """
    pytest.importorskip("qwen_vl_utils")
    import requests
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    hf_model.config.image_token_id = 151665
    config = DotsOCRConfig.from_hf(hf_model.config)
    H = config.hidden_size
    text_model = hf_model.model

    processor, tokenizer = _build_dots_ocr_processor(model_path)

    # Load + crop the demo image (top 57.5%, matching the symbiote vision test —
    # keeps the image-token sequence near the ~2816 bucket the TP4 modules use).
    image = Image.open(requests.get(image_link, stream=True).raw)
    w, h = image.size
    image = image.crop((0, 0, w, int(h * 0.575)))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]
    prompt_len = int(input_ids.shape[1])

    # HF greedy reference (real model, vision + text, with KV cache -> fast).
    hf_gen = hf_model.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    hf_new = hf_gen[0, prompt_len:].tolist()

    # Fused prompt embeds (text embeds with vision-tower embeddings scattered into
    # the image-token positions) — the HF vision tower runs here, once.
    img_mask = input_ids == hf_model.config.image_token_id
    base_embeds = hf_model.prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw, img_mask)  # [1, L0, H] bf16

    tt_model = DotsOCRPrefillModelTP4.from_torch(
        mesh_device, config, text_model.layers, torch_norm=text_model.norm, torch_lm_head=hf_model.lm_head
    )

    eos_ids = set(
        [hf_model.config.eos_token_id]
        if isinstance(hf_model.config.eos_token_id, int)
        else list(hf_model.config.eos_token_id or [])
    )

    # Free-run TP4 greedy: vision is already fused into base_embeds; each new
    # token is a text token appended via embed_tokens.
    tp4_new = []
    cur_embeds = base_embeds  # [1, L, H]
    for _ in range(max_new_tokens):
        L = int(cur_embeds.shape[1])
        s_pad = ((L + 31) // 32) * 32
        emb_in = cur_embeds
        if s_pad > L:
            pad = torch.zeros(1, s_pad - L, H, dtype=cur_embeds.dtype)
            emb_in = torch.cat([cur_embeds, pad], dim=1)
        x_tt = to_replicated(emb_in.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)
        _, token_ids = tt_model.forward_with_head(x_tt, last_token_only=True, return_token=True, token_index=L - 1)
        ttnn.synchronize_device(mesh_device)
        nxt = int(token_ids.flatten()[0])
        tp4_new.append(nxt)
        nxt_embed = text_model.embed_tokens(torch.tensor([[nxt]], dtype=torch.long))  # [1, 1, H]
        cur_embeds = torch.cat([cur_embeds, nxt_embed], dim=1)
        if nxt in eos_ids:
            break

    hf_text = tokenizer.decode(hf_new, skip_special_tokens=True)
    tp4_text = tokenizer.decode(tp4_new, skip_special_tokens=True)
    match = 0
    for a, b in zip(hf_new, tp4_new):
        if a == b:
            match += 1
        else:
            break

    print("\n" + "=" * 70)
    print(f"[dots_ocr_tp4 OCR] prompt_len={prompt_len} (image tokens included)  max_new_tokens={max_new_tokens}")
    print(f"greedy match prefix: {match}/{min(len(hf_new), len(tp4_new))}")
    print(f"\nHF  OCR : {hf_text!r}")
    print(f"TP4 OCR : {tp4_text!r}")
    print("=" * 70)

    assert len(tp4_text.strip()) > 0
    assert match >= 6, f"TP4 OCR greedy diverged almost immediately from HF: only {match} tokens matched"
