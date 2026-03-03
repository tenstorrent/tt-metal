# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test intermediate decoder stages to find mismatch.

Compares:
1. Codebook lookup
2. Pre-transformer output
3. Conv decoder output
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    cov = (a_centered * b_centered).sum()
    std_a = torch.sqrt((a_centered**2).sum())
    std_b = torch.sqrt((b_centered**2).sum())
    if std_a == 0 or std_b == 0:
        return 0.0
    return (cov / (std_a * std_b)).item()


def load_speech_tokenizer_weights():
    """Load speech tokenizer weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Strip "decoder." prefix from keys
    speech_tokenizer_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("decoder."):
            speech_tokenizer_dict[k[8:]] = v
        else:
            speech_tokenizer_dict[k] = v

    return speech_tokenizer_dict


def run_test(device_id: int = 0):
    """Test intermediate decoder stages."""
    print("=" * 80)
    print("Decoder Intermediate Stages Test")
    print("=" * 80)

    # Load reference codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print("ERROR: Reference codes not found.")
        return

    print("\nLoading data...")
    data = torch.load(voice_clone_path)
    ref_code = data["ref_code"]  # [101, 16]
    token_ids = ref_code.T.unsqueeze(0)  # [1, 16, 101]

    state_dict = load_speech_tokenizer_weights()
    print(f"  Token IDs shape: {token_ids.shape}")

    # =========================================
    # Test 1: Codebook Lookup
    # =========================================
    print("\n" + "=" * 80)
    print("Test 1: Codebook Lookup (RVQ)")
    print("=" * 80)

    from models.demos.qwen3_tts.reference.functional import SpeechTokenizerDecoderConfig, codebook_lookup_rvq

    config = SpeechTokenizerDecoderConfig()

    # PyTorch codebook lookup
    rvq_first_codebook = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")
    rvq_rest_codebooks = []
    for i in range(15):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        if key in state_dict:
            rvq_rest_codebooks.append(state_dict[key])
    rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

    pytorch_embeddings = codebook_lookup_rvq(
        token_ids,
        rvq_first_codebook,
        rvq_rest_codebooks,
        rvq_first_output_proj,
        rvq_rest_output_proj,
    )
    print(f"  PyTorch embeddings shape: {pytorch_embeddings.shape}")
    print(f"  PyTorch embeddings range: [{pytorch_embeddings.min():.4f}, {pytorch_embeddings.max():.4f}]")

    # TTNN codebook lookup (manual implementation)
    # The TTNN decoder has _codebook_lookup method
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

        ttnn_decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=state_dict,
        )

        # Call internal codebook lookup
        ttnn_embeddings = ttnn_decoder._codebook_lookup(token_ids)
        print(f"  TTNN embeddings shape: {ttnn_embeddings.shape}")
        print(f"  TTNN embeddings range: [{ttnn_embeddings.min():.4f}, {ttnn_embeddings.max():.4f}]")

        # Compare
        pcc1 = compute_pcc(pytorch_embeddings, ttnn_embeddings)
        print(f"  PCC (codebook lookup): {pcc1:.6f}")

        if pcc1 < 0.99:
            print("  *** BUG FOUND: Codebook lookup mismatch! ***")

        # =========================================
        # Test 2: Pre-transformer (if codebook matches)
        # =========================================
        print("\n" + "=" * 80)
        print("Test 2: Pre-transformer")
        print("=" * 80)

        if not ttnn_decoder.has_pre_transformer:
            print("  TTNN pre-transformer not loaded!")
        else:
            from models.demos.qwen3_tts.reference.functional import pre_transformer_forward

            # PyTorch pre-transformer
            pre_transformer_weights = {
                k.replace("pre_transformer.", ""): v for k, v in state_dict.items() if k.startswith("pre_transformer.")
            }

            pytorch_hidden = pre_transformer_forward(pytorch_embeddings, pre_transformer_weights, config)
            print(f"  PyTorch hidden shape: {pytorch_hidden.shape}")

            # TTNN pre-transformer - run forward manually
            # Convert pytorch_embeddings to TTNN and run pre_transformer
            seq_len = pytorch_embeddings.shape[1]
            pad_seq = ((seq_len + 31) // 32) * 32
            padding = pad_seq - seq_len

            embeddings_padded = F.pad(pytorch_embeddings, (0, 0, 0, padding))
            embeddings_ttnn = ttnn.from_torch(
                embeddings_padded.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

            # Get RoPE for pre-transformer
            cos, sin = ttnn_decoder._compute_rope(pad_seq, pytorch_embeddings.device)
            cos_ttnn = ttnn.from_torch(
                cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            sin_ttnn = ttnn.from_torch(
                sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

            ttnn_hidden = ttnn_decoder.pre_transformer(embeddings_ttnn, cos_ttnn, sin_ttnn)
            ttnn_hidden_torch = ttnn.to_torch(ttnn_hidden)[:, :seq_len, :]
            print(f"  TTNN hidden shape: {ttnn_hidden_torch.shape}")

            pcc2 = compute_pcc(pytorch_hidden, ttnn_hidden_torch)
            print(f"  PCC (pre-transformer): {pcc2:.6f}")

            if pcc2 < 0.99:
                print("  *** BUG FOUND: Pre-transformer mismatch! ***")

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"  Codebook Lookup PCC: {pcc1:.6f}")
        if "pcc2" in dir():
            print(f"  Pre-transformer PCC: {pcc2:.6f}")

        if pcc1 < 0.99:
            print("\n  --> Fix codebook lookup first!")
        elif "pcc2" in dir() and pcc2 < 0.99:
            print("\n  --> Fix pre-transformer!")
        else:
            print("\n  --> Bug is in conv decoder (PyTorch fallback)")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    run_test(device_id=args.device_id)


if __name__ == "__main__":
    main()
