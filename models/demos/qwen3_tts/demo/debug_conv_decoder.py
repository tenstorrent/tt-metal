# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug conv decoder stages - upsampler and decoder.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    # Handle size mismatch
    min_len = min(a.numel(), b.numel())
    a = a.flatten().float()[:min_len]
    b = b.flatten().float()[:min_len]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug Conv Decoder Stages")
    print("=" * 80)

    # Load official model
    print("\n[1] Loading official model...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,
    )
    speech_tokenizer = model.model.speech_tokenizer.model
    official_decoder = speech_tokenizer.decoder

    # Load reference weights
    print("\n[2] Loading reference weights...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)

    speech_path = model_path / "speech_tokenizer" / "model.safetensors"
    speech_dict = load_file(speech_path)
    decoder_weights = {k[8:]: v.float() for k, v in speech_dict.items() if k.startswith("decoder.")}

    # Create test input at pre_transformer output stage
    print("\n[3] Creating test input...")
    torch.manual_seed(42)

    # Shape matches pre_transformer output: [batch, seq_len, hidden_size]
    # But we need channels-first for conv: [batch, channels, seq_len]
    hidden = torch.randn(1, 512, 20)  # [batch, 512, seq_len]
    print(f"  Input shape: {hidden.shape}")

    # Run official upsampler
    print("\n[4] Testing upsampler...")

    with torch.no_grad():
        official_upsample = hidden.clone()
        for i, block in enumerate(official_decoder.upsample):
            official_upsample = block[0](official_upsample)  # TransConv
            official_upsample = block[1](official_upsample)  # ConvNeXt
            print(f"  Official upsample block {i}: {official_upsample.shape}")

    # Reference upsampler
    from models.demos.qwen3_tts.reference.functional import upsample_block

    ref_upsample = hidden.clone()
    for i in range(2):  # 2 upsample blocks
        prefix = f"upsample.{i}."
        block_weights = {k.replace(prefix, ""): v for k, v in decoder_weights.items() if k.startswith(prefix)}
        ref_upsample = upsample_block(ref_upsample, block_weights, upsample_rate=2)
        print(f"  Reference upsample block {i}: {ref_upsample.shape}")

    pcc_upsample = compute_pcc(ref_upsample, official_upsample)
    print(f"\n  Upsampler PCC: {pcc_upsample:.6f}")

    if pcc_upsample < 0.99:
        print(f"  *** MISMATCH ***")
        print(f"  Ref: mean={ref_upsample.mean():.4f}, std={ref_upsample.std():.4f}")
        print(f"  Off: mean={official_upsample.mean():.4f}, std={official_upsample.std():.4f}")

    # Run official conv decoder
    print("\n[5] Testing conv decoder...")

    with torch.no_grad():
        official_dec = official_upsample.clone()
        for i, block in enumerate(official_decoder.decoder):
            official_dec = block(official_dec)
            print(f"  Official decoder block {i}: {official_dec.shape}")

    # Reference conv decoder - check step by step
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        conv_decoder_block,
        snake_activation,
    )

    config = SpeechTokenizerDecoderConfig()
    upsample_rates = config.upsample_rates  # [8, 5, 4, 3]
    channels = [512, 256, 128, 64, 32]

    ref_dec = ref_upsample.clone()

    # First conv (initial)
    init_conv_weight = decoder_weights["decoder.0.conv.weight"]
    init_conv_bias = decoder_weights.get("decoder.0.conv.bias")
    kernel_size = init_conv_weight.shape[-1]
    ref_dec = F.pad(ref_dec, (kernel_size - 1, 0), mode="constant", value=0)
    ref_dec = F.conv1d(ref_dec, init_conv_weight, init_conv_bias)
    print(f"  Reference decoder init conv: {ref_dec.shape}")

    # Compare with official after first block
    with torch.no_grad():
        off_first = official_decoder.decoder[0](official_upsample)
    pcc_first = compute_pcc(ref_dec, off_first)
    print(f"  PCC after first conv: {pcc_first:.6f}")

    # Upsample blocks
    for i, (rate, in_ch, out_ch) in enumerate(zip(upsample_rates, channels[:-1], channels[1:])):
        prefix = f"decoder.{i + 1}."
        block_weights = {k.replace(prefix, ""): v for k, v in decoder_weights.items() if k.startswith(prefix)}

        if not block_weights:
            print(f"  No weights for decoder.{i + 1}")
            continue

        ref_dec = conv_decoder_block(ref_dec, block_weights, upsample_rate=rate)
        print(f"  Reference decoder block {i + 1}: {ref_dec.shape}")

    # Final snake + conv
    if "decoder.5.snake.alpha" in decoder_weights:
        ref_dec = snake_activation(ref_dec, decoder_weights["decoder.5.snake.alpha"])

    final_conv_weight = decoder_weights.get("decoder.5.conv.weight")
    final_conv_bias = decoder_weights.get("decoder.5.conv.bias")
    if final_conv_weight is not None:
        kernel_size = final_conv_weight.shape[-1]
        ref_dec = F.pad(ref_dec, (kernel_size - 1, 0), mode="constant", value=0)
        ref_dec = F.conv1d(ref_dec, final_conv_weight, final_conv_bias)

    ref_dec = torch.tanh(ref_dec)
    print(f"  Reference decoder final: {ref_dec.shape}")
    print(f"  Official decoder final: {official_dec.shape}")

    pcc_dec = compute_pcc(ref_dec, official_dec)
    print(f"\n  Conv Decoder PCC: {pcc_dec:.6f}")

    if pcc_dec < 0.99:
        print(f"  *** MISMATCH ***")
        print(
            f"  Ref: mean={ref_dec.mean():.6f}, std={ref_dec.std():.4f}, range=[{ref_dec.min():.4f}, {ref_dec.max():.4f}]"
        )
        print(
            f"  Off: mean={official_dec.mean():.6f}, std={official_dec.std():.4f}, range=[{official_dec.min():.4f}, {official_dec.max():.4f}]"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
