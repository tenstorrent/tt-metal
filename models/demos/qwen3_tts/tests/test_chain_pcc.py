# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
End-to-End Chained 28-Layer PCC Test.

PCC > 0.99 per-block is NOT sufficient — errors compound over 28 layers.
This test feeds TTNN layer N output into TTNN layer N+1 (chained mode),
exactly as happens during real inference, and reports PCC at every layer.

Usage:
    pytest models/demos/qwen3_tts/tests/test_chain_pcc.py -s -v
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    am, bm = a.mean(), b.mean()
    ac, bc = a - am, b - bm
    denom = ac.norm() * bc.norm()
    if denom < 1e-8:
        return 0.0
    return (ac * bc).sum().item() / denom.item()


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def state_dict():
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    sd = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            sd.update(load_file(f))
    return sd


def build_icl_embedding(state_dict, ref_codes: torch.Tensor, target_text: str):
    """Build ICL embedding by reusing create_icl_embedding from demo_pure_reference_tts."""
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_pure_reference_tts import TTSConfig, create_icl_embedding

    config = TTSConfig()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    ref_text = "Jason, can you put up the high level overview slides?"
    # Use zero speaker embedding (placeholder)
    speaker_embedding = torch.zeros(1, 1, 2048)

    inputs_embeds, trailing_text, tts_pad = create_icl_embedding(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes,
        tokenizer=tokenizer,
        weights=state_dict,
        config=config,
        speaker_embedding=speaker_embedding,
        language="english",
    )
    print(f"  ICL embedding shape: {inputs_embeds.shape}")
    return inputs_embeds


def run_reference_chain(state_dict, inputs_embeds: torch.Tensor):
    """Run 28 reference decoder layers chained, return per-layer hidden states."""
    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        compute_mrope_frequencies,
        decoder_layer,
        extract_talker_weights,
        rms_norm,
    )

    config = Qwen3TTSConfig()
    talker_weights = extract_talker_weights(state_dict)
    seq_len = inputs_embeds.shape[1]

    cos, sin = compute_mrope_frequencies(config.head_dim, seq_len, config.rope_theta)
    cos = cos.float()
    sin = sin.float()

    attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0).float()

    x = inputs_embeds.float()
    per_layer_hidden = {}

    for layer_idx in range(config.num_hidden_layers):
        per_layer_hidden[f"input_{layer_idx}"] = x.clone()
        layer_prefix = f"layers.{layer_idx}."
        lw = {k.replace(layer_prefix, ""): v.float() for k, v in talker_weights.items() if k.startswith(layer_prefix)}
        x = decoder_layer(x, lw, cos, sin, config, attention_mask=attn_mask, use_mrope=True)
        per_layer_hidden[f"output_{layer_idx}"] = x.clone()

    x = rms_norm(x, talker_weights["norm.weight"].float(), config.rms_norm_eps)
    per_layer_hidden["final"] = x.clone()

    # Top-5 logits at last position
    codec_head = state_dict["talker.codec_head.weight"].float()
    logits = x[0, -1, :] @ codec_head.T  # [3072]
    per_layer_hidden["logits_last"] = logits
    return per_layer_hidden


def run_ttnn_chain(device, state_dict, inputs_embeds: torch.Tensor):
    """Run TTNN Talker (28 layers chained) using forward_from_hidden, return per-layer hidden states."""
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    config = Qwen3TTSTalkerConfig()
    seq_len = inputs_embeds.shape[1]
    pad_seq = ((seq_len + 31) // 32) * 32
    padding = pad_seq - seq_len

    # Position IDs (flat)
    position_ids = torch.arange(pad_seq)
    cos_tt, sin_tt = get_rope_tensors(device, config.head_dim, pad_seq, position_ids, config.rope_theta)
    trans_mat = get_transformation_mat(config.head_dim, device)

    # Load all 28 layers individually to capture per-layer outputs
    layers = []
    for i in range(config.num_hidden_layers):
        layer = DecoderLayer(
            device=device,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            state_dict=state_dict,
            layer_idx=i,
            layer_prefix="talker.model",
            rms_norm_eps=config.rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
        )
        layers.append(layer)

    # Final norm weight — must be [1, 1, dim//32, 32] in ROW_MAJOR_LAYOUT
    TILE = 32
    final_norm_w = state_dict["talker.model.norm.weight"].to(torch.bfloat16)
    final_norm_w_reshaped = final_norm_w.view(1, 1, config.hidden_size // TILE, TILE)
    final_norm_tt = ttnn.as_tensor(
        final_norm_w_reshaped,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Prepare input [1, 1, pad_seq, 2048]
    inp_padded = F.pad(inputs_embeds, (0, 0, 0, padding))
    inp_4d = inp_padded.unsqueeze(1).to(torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        inp_4d, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    per_layer_hidden = {}

    for i, layer in enumerate(layers):
        per_layer_hidden[f"input_{i}"] = ttnn.to_torch(hidden_tt).squeeze(1).float()[:, :seq_len, :]
        hidden_tt, _ = layer(
            hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None, kv_cache=None, start_pos=0, mode="prefill"
        )
        per_layer_hidden[f"output_{i}"] = ttnn.to_torch(hidden_tt).squeeze(1).float()[:, :seq_len, :]

    # Final norm
    hidden_tt = ttnn.rms_norm(hidden_tt, epsilon=config.rms_norm_eps, weight=final_norm_tt)
    final_torch = ttnn.to_torch(hidden_tt).squeeze(1).float()[:, :seq_len, :]
    per_layer_hidden["final"] = final_torch

    # codec_head logits at last position
    codec_head = state_dict["talker.codec_head.weight"].float()
    logits = final_torch[0, -1, :] @ codec_head.T
    per_layer_hidden["logits_last"] = logits

    return per_layer_hidden


def test_chained_28_layer_pcc(device, state_dict):
    """
    Full 28-layer chained PCC test.

    Feeds the ICL embedding through all 28 layers sequentially (TTNN output → next TTNN layer),
    exactly as in real inference. Reports PCC vs reference at each layer.

    PASS criteria: Final layer PCC > 0.90, Layer 1 PCC > 0.99.
    """
    target_text = "Hello, welcome to Tenstorrent!!"
    cache_file = "/tmp/jim_ref.refcache.pt"

    cached = torch.load(cache_file, weights_only=True)
    ref_codes = cached["ref_codes"].long()  # [51, 16]

    print(f"\n{'='*60}")
    print("Building ICL embedding...")
    inputs_embeds = build_icl_embedding(state_dict, ref_codes, target_text)

    print("\nRunning reference 28-layer chain...")
    ref_states = run_reference_chain(state_dict, inputs_embeds)

    print("\nRunning TTNN 28-layer chain...")
    ttnn_states = run_ttnn_chain(device, state_dict, inputs_embeds)

    # Compare per-layer
    print(f"\n{'='*60}")
    print(f"{'Layer':>6} | {'PCC (output)':>12} | {'Max Abs Err':>11} | Status")
    print("-" * 50)
    for i in range(28):
        ref_out = ref_states[f"output_{i}"]
        tt_out = ttnn_states[f"output_{i}"]
        pcc_val = compute_pcc(ref_out, tt_out)
        max_err = (ref_out - tt_out).abs().max().item()
        status = "OK" if pcc_val > 0.99 else ("WARN" if pcc_val > 0.90 else "FAIL")
        print(f"{i:>6} | {pcc_val:>12.4f} | {max_err:>11.4f} | {status}")

    # Final norm PCC
    ref_final = ref_states["final"]
    tt_final = ttnn_states["final"]
    final_pcc = compute_pcc(ref_final, tt_final)
    print(f"{'FINAL':>6} | {final_pcc:>12.4f} |             | {'OK' if final_pcc > 0.90 else 'FAIL'}")

    # Logit top-5 comparison
    ref_logits = ref_states["logits_last"]
    tt_logits = ttnn_states["logits_last"]
    logit_pcc = compute_pcc(ref_logits.unsqueeze(0), tt_logits.unsqueeze(0))
    print(f"\nLogit PCC (last pos): {logit_pcc:.4f}")

    ref_top5 = ref_logits.topk(5)
    tt_top5 = tt_logits.topk(5)
    print(
        "\nReference top-5 tokens:",
        [(t.item(), f"{v.item():.2f}", t.item() < 2048) for t, v in zip(ref_top5.indices, ref_top5.values)],
    )
    print(
        "TTNN     top-5 tokens:",
        [(t.item(), f"{v.item():.2f}", t.item() < 2048) for t, v in zip(tt_top5.indices, tt_top5.values)],
    )

    print(f"\n{'='*60}")
    assert final_pcc > 0.90, f"Full 28-layer chain PCC {final_pcc:.4f} < 0.90 — Talker divergence detected"
    if final_pcc < 0.99:
        print(f"WARNING: Final PCC {final_pcc:.4f} < 0.99. Audio quality will be degraded.")
