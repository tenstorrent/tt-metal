# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 generation via a PAGED KV cache: prefill fills the cache, then cheap
single-token decode steps read/extend it (no O(n^2) recompute). Compares the
generated tokens to the real HF model's greedy output."""

import os

import pytest
import torch

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, to_replicated
from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache
from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download("rednote-hilab/dots.ocr")


def _pos_tensor(mesh_device, pos: int):
    return ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
    )


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("max_new_tokens", [int(os.environ.get("DOTS_OCR_TP4_DECODE_TOKENS", "24"))])
def test_tp4_paged_decode_generate(mesh_device, max_new_tokens):
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
    L0 = int(input_ids.shape[1])

    hf_gen = hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    hf_new = hf_gen[0, L0:].tolist()

    tt_model = DotsOCRPrefillModelTP4.from_torch(
        mesh_device, config, text_model.layers, torch_norm=text_model.norm, torch_lm_head=hf_model.lm_head
    )
    cache = create_paged_kv_cache(config, mesh_device, batch_size=1)

    eos_ids = set(
        [hf_model.config.eos_token_id]
        if isinstance(hf_model.config.eos_token_id, int)
        else list(hf_model.config.eos_token_id or [])
    )

    # ---- Prefill: fill the KV cache, read first token at the real last pos. ----
    s_pad = ((L0 + 31) // 32) * 32
    embeds = text_model.embed_tokens(input_ids)  # [1, L0, H]
    if s_pad > L0:
        embeds = torch.cat([embeds, torch.zeros(1, s_pad - L0, H, dtype=embeds.dtype)], dim=1)
    x_tt = to_replicated(embeds.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)
    _, tok = tt_model.prefill_with_head(x_tt, cache, token_index=L0 - 1, return_token=True)
    ttnn.synchronize_device(mesh_device)
    first = int(tok.flatten()[0])

    # ---- Decode: one token at a time, reading/extending the paged cache. ----
    tp4_new = [first]
    prev = first
    pos = L0  # the new token "first" sits at position L0
    for _ in range(max_new_tokens - 1):
        if prev in eos_ids:
            break
        emb = text_model.embed_tokens(torch.tensor([[prev]], dtype=torch.long))  # [1, 1, H]
        x_tt = to_replicated(emb.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)
        cp = _pos_tensor(mesh_device, pos)
        _, tok = tt_model.decode_with_head(x_tt, cache, cp, return_token=True)
        ttnn.synchronize_device(mesh_device)
        nxt = int(tok.flatten()[0])
        tp4_new.append(nxt)
        prev = nxt
        pos += 1

    hf_text = tokenizer.decode(hf_new, skip_special_tokens=True)
    tp4_text = tokenizer.decode(tp4_new, skip_special_tokens=True)
    match = 0
    for a, b in zip(hf_new, tp4_new):
        if a == b:
            match += 1
        else:
            break

    print("\n" + "=" * 70)
    print(f"[dots_ocr_tp4 PAGED DECODE] L0={L0}  max_new_tokens={max_new_tokens}")
    print(f"HF  tokens : {hf_new}")
    print(f"TP4 tokens : {tp4_new}")
    print(f"greedy match prefix: {match}/{min(len(hf_new), len(tp4_new))}")
    print(f"\nHF  text : {hf_text!r}")
    print(f"TP4 text : {tp4_text!r}")
    print("=" * 70)

    assert len(tp4_text.strip()) > 0
    assert match >= 6, f"TP4 paged decode diverged almost immediately from HF: only {match} matched"
