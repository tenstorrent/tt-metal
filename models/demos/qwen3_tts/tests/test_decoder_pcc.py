# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Compare TTNN vs PyTorch Speech Tokenizer Decoder

This test compares the output of the TTNN decoder with the PyTorch
reference decoder to identify where differences occur.
"""

import argparse
from pathlib import Path

import torch

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
    """Compare TTNN vs PyTorch decoder."""
    print("=" * 80)
    print("TTNN vs PyTorch Decoder PCC Test")
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
    print(f"  Loaded {len(state_dict)} weight tensors")

    # Run PyTorch reference
    print("\n1. Running PyTorch reference decoder...")
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    config = SpeechTokenizerDecoderConfig()
    pytorch_audio = speech_tokenizer_decoder_forward(
        token_ids=token_ids,
        weights=state_dict,
        config=config,
    )
    print(f"  PyTorch audio shape: {pytorch_audio.shape}")

    # Run TTNN decoder
    print("\n2. Running TTNN decoder...")
    device = ttnn.open_device(device_id=device_id)

    try:
        from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

        ttnn_decoder = TtSpeechTokenizerDecoder(
            device=device,
            state_dict=state_dict,
        )

        ttnn_audio = ttnn_decoder.forward(token_ids)
        print(f"  TTNN audio shape: {ttnn_audio.shape}")

        # Compare
        print("\n3. Comparing outputs...")
        ttnn_audio_float = ttnn_audio.squeeze().detach().cpu().float()
        pytorch_audio_float = pytorch_audio.squeeze().detach().cpu().float()

        # Truncate to same length if different
        min_len = min(len(ttnn_audio_float), len(pytorch_audio_float))
        ttnn_audio_float = ttnn_audio_float[:min_len]
        pytorch_audio_float = pytorch_audio_float[:min_len]

        pcc = compute_pcc(ttnn_audio_float, pytorch_audio_float)
        max_diff = (ttnn_audio_float - pytorch_audio_float).abs().max().item()
        mean_diff = (ttnn_audio_float - pytorch_audio_float).abs().mean().item()

        print(f"  PCC: {pcc:.6f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        print(f"\n  PyTorch audio range: [{pytorch_audio_float.min():.4f}, {pytorch_audio_float.max():.4f}]")
        print(f"  TTNN audio range: [{ttnn_audio_float.min():.4f}, {ttnn_audio_float.max():.4f}]")

        # Save both for comparison
        import soundfile as sf

        sf.write("/tmp/decoder_pytorch.wav", pytorch_audio_float.numpy(), 24000)
        sf.write("/tmp/decoder_ttnn.wav", ttnn_audio_float.numpy(), 24000)
        print("\n  Saved: /tmp/decoder_pytorch.wav and /tmp/decoder_ttnn.wav")

        if pcc > 0.99:
            print("\n*** PASS: TTNN decoder matches PyTorch reference! ***")
        elif pcc > 0.95:
            print(f"\n*** WARN: PCC={pcc:.4f} - some differences ***")
        else:
            print(f"\n*** FAIL: PCC={pcc:.4f} - significant mismatch ***")

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    run_test(device_id=args.device_id)


if __name__ == "__main__":
    main()
