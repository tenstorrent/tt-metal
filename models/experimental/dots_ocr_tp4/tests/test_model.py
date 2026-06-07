# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full dots.ocr text-decoder prefill (N repeated blocks) TP4 vs torch."""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, from_replicated_to_torch, to_replicated
from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache
from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape
from models.experimental.dots_ocr_tp4.tests.torch_reference import TorchDecoderStack


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("num_layers", [int(os.environ.get("DOTS_OCR_TP4_NUM_LAYERS", "28"))])
@pytest.mark.parametrize("seq_len", [2816])
def test_dots_ocr_prefill_model_tp4(mesh_device, num_layers, seq_len):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    config = DotsOCRConfig(num_hidden_layers=num_layers)
    H = config.hidden_size

    torch_stack = TorchDecoderStack(config, num_layers=num_layers).eval()
    x = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    torch_out = torch_stack(x.to(torch.float32))

    tt_model = DotsOCRPrefillModelTP4.from_torch(mesh_device, config, torch_stack.layers)
    x_tt = to_replicated(x, mesh_device, dtype=ttnn.bfloat16)

    # Pass a paged KV cache so prefill populates it (each chip writes its one
    # KV head) -> the PagedFillCacheDeviceOperation kernel runs per layer in the
    # profiled prefill. The cache write is a side effect; PCC is unaffected.
    paged_cache = create_paged_kv_cache(config, mesh_device, batch_size=1)
    out_tt = tt_model.forward(x_tt, past_key_value=paged_cache)
    ttnn.synchronize_device(mesh_device)

    out_torch = from_replicated_to_torch(out_tt, mesh_device).to(torch.float32).reshape(torch_out.shape)

    passed, msg = assert_with_pcc(torch_out.to(torch.float32), out_torch, pcc=0.99)
    print(f"\n[dots_ocr_tp4] full prefill {num_layers} layers PCC: {msg}")


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download("rednote-hilab/dots.ocr")


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [int(os.environ.get("DOTS_OCR_TP4_HF_SEQ_LEN", "2816"))])
def test_dots_ocr_prefill_model_tp4_hf_weights(mesh_device, seq_len):
    """Full 28-layer prefill against the REAL HF dots.ocr text-decoder weights.

    Feeds the same embedded input to both. HF's final norm is swapped for
    Identity so ``last_hidden_state`` is exactly the 28-block output (matching
    this rebuild, which is the decoder body only). HF builds the causal mask
    internally; ``eager`` attention is used for a deterministic reference.
    """
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    config = DotsOCRConfig.from_hf(hf_model.config)
    assert config.num_hidden_layers == 28

    text_model = hf_model.model  # Qwen2Model

    # Same input for both: real embedded tokens (realistic hidden-state scale).
    input_ids = torch.randint(0, config_vocab(hf_model), (1, seq_len), dtype=torch.long)
    embeds = text_model.embed_tokens(input_ids)  # [1, S, H] bf16

    # 28-block output (drop the final norm so this matches the rebuild body).
    text_model.norm = torch.nn.Identity()
    # Ideal float32 reference: upcasting bf16 weights to float32 is exact, so the
    # device (bf16) keeps the identical weight VALUES (shard cast f32->bf16 round-
    # trips them) — PCC then cleanly measures the device bf16 path's error vs the
    # ground truth, the convention this repo's decode-PCC tests use.
    text_model.float()
    hf_out = text_model(inputs_embeds=embeds.to(torch.float32), use_cache=False).last_hidden_state

    tt_model = DotsOCRPrefillModelTP4.from_torch(mesh_device, config, text_model.layers)
    x_tt = to_replicated(embeds.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)

    out_tt = tt_model.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = from_replicated_to_torch(out_tt, mesh_device).to(torch.float32).reshape(hf_out.shape)

    # The prefill block uses the production low-precision recipe (BFP4/BFP8
    # weights, see attention.py/mlp.py). On real weights across 28 TP4 layers
    # that lands ~0.97 PCC -- the accuracy/perf tradeoff the original model
    # makes (its full-decoder bar is 0.95). bf16 weights reach ~0.993.
    passed, msg = assert_with_pcc(hf_out.to(torch.float32), out_torch, pcc=0.95)
    print(f"\n[dots_ocr_tp4] full prefill HF-weights {config.num_hidden_layers} layers (S={seq_len}) PCC: {msg}")


def config_vocab(hf_model):
    return int(getattr(hf_model.config, "vocab_size", 151936))


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [int(os.environ.get("DOTS_OCR_TP4_HF_SEQ_LEN", "2816"))])
def test_dots_ocr_prefill_full_with_head_hf_weights(mesh_device, seq_len):
    """End-to-end prefill on REAL HF weights: 28 blocks + final norm + LM head.

    Checks the last-token logits PCC and that the greedy next-token id matches
    the HF model exactly (the prefill -> first decode-token hand-off)."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    config = DotsOCRConfig.from_hf(hf_model.config)

    text_model = hf_model.model
    input_ids = torch.randint(0, config_vocab(hf_model), (1, seq_len), dtype=torch.long)
    embeds = text_model.embed_tokens(input_ids)  # [1, S, H] bf16

    # Ideal float32 reference over the full head (see other HF test for why).
    # Call the Qwen2Model (final norm included) + lm_head directly — the custom
    # DotsOCRForCausalLM.forward requires input_ids, and this mirrors exactly
    # what the TP4 head reproduces (final RMSNorm -> lm_head).
    hf_model.float()
    last_hidden = text_model(inputs_embeds=embeds.to(torch.float32), use_cache=False).last_hidden_state
    hf_logits = hf_model.lm_head(last_hidden[:, -1:, :])  # [1, 1, vocab]
    hf_token = int(hf_logits[0, -1].argmax())

    tt_model = DotsOCRPrefillModelTP4.from_torch(
        mesh_device, config, text_model.layers, torch_norm=text_model.norm, torch_lm_head=hf_model.lm_head
    )
    x_tt = to_replicated(embeds.to(torch.bfloat16), mesh_device, dtype=ttnn.bfloat16)

    logits_tt, token_ids = tt_model.forward_with_head(x_tt, last_token_only=True, return_token=True)
    ttnn.synchronize_device(mesh_device)

    logits_torch = from_replicated_to_torch(logits_tt, mesh_device).to(torch.float32).reshape(hf_logits.shape)
    tt_token = int(logits_torch[0, -1].argmax())
    device_token = int(token_ids.flatten()[0])

    from tests.ttnn.utils_for_testing import comp_pcc

    _, logits_pcc = comp_pcc(hf_logits.to(torch.float32), logits_torch, 0.0)
    # Greedy next-token id is the functional correctness gate for prefill. The
    # raw logits PCC over the 151936-way vocab is far more sensitive than the
    # hidden state (it amplifies the decoder's accumulated bf16 TP drift), so it
    # is reported but not the hard gate.
    print(
        f"\n[dots_ocr_tp4] full prefill+head HF-weights (S={seq_len}) last-token logits PCC: {logits_pcc} | "
        f"hf_token={hf_token} tt_token={tt_token} device_argmax={device_token}"
    )
    assert device_token == hf_token, f"next-token mismatch: HF {hf_token} vs device {device_token}"
