# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
PCC Comparison Test: TTNN vs PyTorch Reference

Tests each model component with identical inputs to find where TTNN diverges from reference.
"""

import argparse
from pathlib import Path

import torch

import ttnn
from models.demos.qwen3_tts.reference.functional import (
    Qwen3TTSConfig,
    SpeechTokenizerDecoderConfig,
    attention,
    codebook_lookup_rvq,
    compute_mrope_frequencies,
    extract_speech_tokenizer_decoder_weights,
    extract_talker_weights,
    rms_norm,
)


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    a = a.flatten().float()
    b = b.flatten().float()

    a_mean = a.mean()
    b_mean = b.mean()

    a_centered = a - a_mean
    b_centered = b - b_mean

    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())

    if std_a == 0 or std_b == 0:
        return 0.0

    pcc = cov / (std_a * std_b)
    return pcc.item()


def load_weights(model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    """Load weights from HuggingFace."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Loading model weights from: {model_id}")
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    state_dict = {}
    for f in model_path.glob("*.safetensors"):
        state_dict.update(load_file(f))

    print(f"  Loaded {len(state_dict)} weight tensors")
    return state_dict


def load_speech_tokenizer_weights(model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    """Load Speech Tokenizer weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Loading speech tokenizer weights...")
    model_path = snapshot_download(model_id, allow_patterns=["speech_tokenizer/*.safetensors"])
    model_path = Path(model_path)

    st_path = model_path / "speech_tokenizer" / "model.safetensors"
    if not st_path.exists():
        return {}

    state_dict = load_file(st_path)
    decoder_weights = extract_speech_tokenizer_decoder_weights(state_dict)
    print(f"  Loaded {len(decoder_weights)} speech tokenizer decoder weights")
    return decoder_weights


def test_rms_norm_pcc(device):
    """Test RMSNorm PCC between TTNN and reference."""
    print("\n" + "=" * 80)
    print("TEST: RMSNorm PCC")
    print("=" * 80)

    # Create test input
    torch.manual_seed(42)
    batch, seq, hidden = 1, 32, 2048  # seq must be multiple of 32 for tile layout
    eps = 1e-6

    x = torch.randn(batch, seq, hidden)
    weight = torch.randn(hidden)

    # Reference implementation
    ref_output = rms_norm(x, weight, eps)

    # TTNN implementation
    # Input needs shape [1, 1, seq, hidden] for TTNN rms_norm
    x_4d = x.unsqueeze(0)  # [1, 1, seq, hidden]
    x_tt = ttnn.from_torch(x_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Weight needs shape [1, 1, 1, hidden] and proper tiling
    # For TTNN rms_norm, gamma should be [1, 1, 32, hidden] for proper tiling
    weight_4d = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, hidden]
    weight_4d = weight_4d.expand(1, 1, 32, hidden).contiguous()  # [1, 1, 32, hidden] for tile layout
    weight_tt = ttnn.from_torch(weight_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn.rms_norm(x_tt, epsilon=eps, weight=weight_tt)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)

    pcc = compute_pcc(ref_output, ttnn_output)
    print(f"  Input shape: {x.shape}")
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  TTNN output shape: {ttnn_output.shape}")
    print(f"  PCC: {pcc:.6f}")

    return pcc


def test_talker_layer_pcc(device, state_dict: dict, layer_idx: int = 0):
    """Test a single Talker layer PCC between TTNN and reference."""
    print(f"\n" + "=" * 80)
    print(f"TEST: Talker Layer {layer_idx} PCC")
    print("=" * 80)

    config = Qwen3TTSConfig()
    talker_weights = extract_talker_weights(state_dict)

    # Create test input (use bfloat16 to match weight dtype)
    torch.manual_seed(42)
    batch, seq = 1, 32  # Use 32 for tile alignment
    x = torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)

    # Get layer weights and convert to bfloat16 if needed
    layer_prefix = f"layers.{layer_idx}."
    layer_weights = {
        k.replace(layer_prefix, ""): v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in talker_weights.items()
        if k.startswith(layer_prefix)
    }

    # Reference: Pre-norm for attention
    hidden_states = rms_norm(x, layer_weights["input_layernorm.weight"], config.rms_norm_eps)

    # Compute RoPE
    cos, sin = compute_mrope_frequencies(config.head_dim, seq, config.rope_theta, x.device)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # Reference attention
    ref_attn_output = attention(
        hidden_states,
        q_proj_weight=layer_weights["self_attn.q_proj.weight"],
        k_proj_weight=layer_weights["self_attn.k_proj.weight"],
        v_proj_weight=layer_weights["self_attn.v_proj.weight"],
        o_proj_weight=layer_weights["self_attn.o_proj.weight"],
        q_norm_weight=layer_weights["self_attn.q_norm.weight"],
        k_norm_weight=layer_weights["self_attn.k_norm.weight"],
        cos=cos,
        sin=sin,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        attention_mask=None,
        use_mrope=True,
        mrope_section=config.mrope_section,
        mrope_interleaved=config.mrope_interleaved,
    )

    print(f"  Input shape: {x.shape}")
    print(f"  Reference attention output shape: {ref_attn_output.shape}")
    print(f"  Reference attention output stats: mean={ref_attn_output.mean():.4f}, std={ref_attn_output.std():.4f}")

    # For now, just return the reference output stats as we haven't loaded TTNN components
    # TODO: Load and compare with TTNN implementation
    print(f"  NOTE: Full TTNN comparison requires loading TtTalker class")

    return None


def test_speech_tokenizer_decoder_pcc(device, speech_weights: dict):
    """Test Speech Tokenizer Decoder PCC between TTNN and reference."""
    print("\n" + "=" * 80)
    print("TEST: Speech Tokenizer Decoder (Codebook Lookup) PCC")
    print("=" * 80)

    config = SpeechTokenizerDecoderConfig()

    # Create test token IDs
    torch.manual_seed(42)
    batch, num_quantizers, seq = 1, 16, 8
    token_ids = torch.randint(0, 2048, (batch, num_quantizers, seq))

    # Get RVQ codebooks and projections
    rvq_first_codebook = speech_weights.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_output_proj = speech_weights.get("quantizer.rvq_first.output_proj.weight")

    rvq_rest_codebooks = []
    for i in range(num_quantizers - 1):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        if key in speech_weights:
            rvq_rest_codebooks.append(speech_weights[key])
    rvq_rest_output_proj = speech_weights.get("quantizer.rvq_rest.output_proj.weight")

    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  RVQ First codebook: {rvq_first_codebook.shape if rvq_first_codebook is not None else None}")
    print(f"  RVQ Rest codebooks: {len(rvq_rest_codebooks)}")

    # Reference codebook lookup
    if rvq_first_output_proj is not None and rvq_rest_output_proj is not None:
        ref_embeddings = codebook_lookup_rvq(
            token_ids,
            rvq_first_codebook,
            rvq_rest_codebooks,
            rvq_first_output_proj,
            rvq_rest_output_proj,
        )
        print(f"  Reference embeddings shape: {ref_embeddings.shape}")
        print(f"  Reference embeddings stats: mean={ref_embeddings.mean():.4f}, std={ref_embeddings.std():.4f}")
    else:
        print("  WARNING: RVQ projections not found")
        return None

    # TODO: Compare with TTNN TtSpeechTokenizerDecoder
    print(f"  NOTE: Full TTNN comparison requires loading TtSpeechTokenizerDecoder class")

    return None


def run_all_tests(device):
    """Run all PCC comparison tests."""
    print("=" * 80)
    print("QWEN3-TTS: TTNN vs PyTorch Reference PCC Comparison")
    print("=" * 80)

    # Load weights
    state_dict = load_weights()
    speech_weights = load_speech_tokenizer_weights()

    # Run tests
    results = {}

    # Test RMSNorm
    pcc = test_rms_norm_pcc(device)
    results["rms_norm"] = pcc

    # Test Talker layer
    test_talker_layer_pcc(device, state_dict, layer_idx=0)

    # Test Speech Tokenizer Decoder
    test_speech_tokenizer_decoder_pcc(device, speech_weights)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, pcc in results.items():
        if pcc is not None:
            status = "PASS" if pcc > 0.99 else "FAIL"
            print(f"  {name}: PCC={pcc:.6f} [{status}]")
        else:
            print(f"  {name}: Not tested")


def main():
    parser = argparse.ArgumentParser(description="PCC Comparison Test")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID")
    args = parser.parse_args()

    # Initialize device
    device = ttnn.open_device(device_id=args.device_id)

    try:
        run_all_tests(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
