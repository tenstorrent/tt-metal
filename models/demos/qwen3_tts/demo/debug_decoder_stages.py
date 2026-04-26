# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug decoder stages - step through each stage with hooks.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    min_len = min(a.numel(), b.numel())
    a = a.flatten().float()[:min_len]
    b = b.flatten().float()[:min_len]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug Decoder Stages")
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

    # Create test input (pre_transformer output shape)
    print("\n[3] Creating test input...")
    torch.manual_seed(42)

    # Shape after pre_transformer: [batch, seq_len, 1024]
    # Need to transpose to [batch, 1024, seq_len] for upsampler
    test_input = torch.randn(1, 1024, 20)
    print(f"  Input shape: {test_input.shape}")

    # Run official upsampler stages
    print("\n[4] Testing upsampler stages...")
    captured_official = {}

    with torch.no_grad():
        x = test_input.clone()
        for i, blocks in enumerate(official_decoder.upsample):
            x = blocks[0](x)  # TransConv
            captured_official[f"upsample.{i}.transconv"] = x.clone()
            x = blocks[1](x)  # ConvNeXt
            captured_official[f"upsample.{i}.convnext"] = x.clone()
            print(f"  Official upsample block {i}: {x.shape}")

    # Reference upsampler
    from models.demos.qwen3_tts.reference.functional import convnext_block

    ref_x = test_input.clone()
    for i in range(2):
        prefix = f"upsample.{i}."
        block_weights = {k.replace(prefix, ""): v for k, v in decoder_weights.items() if k.startswith(prefix)}

        # TransConv
        conv_weight = block_weights.get("0.conv.weight")
        conv_bias = block_weights.get("0.conv.bias")
        if conv_weight is not None:
            ref_x = F.conv_transpose1d(ref_x, conv_weight, conv_bias, stride=2)
            pad_to_trim = conv_weight.shape[-1] - 2
            if pad_to_trim > 0:
                ref_x = ref_x[..., :-pad_to_trim]

        off_tc = captured_official[f"upsample.{i}.transconv"]
        pcc_tc = compute_pcc(ref_x, off_tc)
        print(f"  Ref upsample {i} transconv: {ref_x.shape}, PCC: {pcc_tc:.6f}")

        # ConvNeXt - keys are like "1.dwconv.conv.weight"
        convnext_weights = {}
        for k, v in block_weights.items():
            if k.startswith("1."):
                new_k = k[2:]  # Remove "1."
                convnext_weights[new_k] = v
        if convnext_weights:
            ref_x = convnext_block(
                ref_x,
                dwconv_weight=convnext_weights.get("dwconv.conv.weight"),
                dwconv_bias=convnext_weights.get("dwconv.conv.bias"),
                pwconv1_weight=convnext_weights.get("pwconv1.weight"),
                pwconv1_bias=convnext_weights.get("pwconv1.bias"),
                pwconv2_weight=convnext_weights.get("pwconv2.weight"),
                pwconv2_bias=convnext_weights.get("pwconv2.bias"),
                norm_weight=convnext_weights.get("norm.weight"),
                norm_bias=convnext_weights.get("norm.bias"),
                gamma=convnext_weights.get("gamma"),
            )

        off_cnx = captured_official[f"upsample.{i}.convnext"]
        pcc_cnx = compute_pcc(ref_x, off_cnx)
        print(f"  Ref upsample {i} convnext: {ref_x.shape}, PCC: {pcc_cnx:.6f}")

    # Use official upsampler output to test conv decoder
    print("\n[5] Testing conv decoder stages (using official upsampler output)...")

    # Run official decoder stages
    with torch.no_grad():
        off_dec = captured_official["upsample.1.convnext"].clone()
        for i, block in enumerate(official_decoder.decoder):
            off_dec = block(off_dec)
            captured_official[f"decoder.{i}"] = off_dec.clone()
            print(f"  Official decoder[{i}]: {off_dec.shape}")

    # Reference decoder stages (starting from official upsampler output for fair comparison)
    from models.demos.qwen3_tts.reference.functional import conv_decoder_block, snake_activation

    ref_dec = captured_official["upsample.1.convnext"].clone()

    # decoder.0 - initial conv
    conv_weight = decoder_weights["decoder.0.conv.weight"]
    conv_bias = decoder_weights.get("decoder.0.conv.bias")
    kernel_size = conv_weight.shape[-1]
    ref_dec = F.pad(ref_dec, (kernel_size - 1, 0), mode="constant", value=0)
    ref_dec = F.conv1d(ref_dec, conv_weight, conv_bias)

    pcc_d0 = compute_pcc(ref_dec, captured_official["decoder.0"])
    print(f"\n  Ref decoder.0: {ref_dec.shape}, PCC: {pcc_d0:.6f}")

    # decoder.1-4 - upsampling blocks
    upsample_rates = [8, 5, 4, 3]
    for i, rate in enumerate(upsample_rates):
        block_prefix = f"decoder.{i + 1}."
        block_weights = {
            k.replace(block_prefix, ""): v for k, v in decoder_weights.items() if k.startswith(block_prefix)
        }
        if block_weights:
            ref_dec = conv_decoder_block(ref_dec, block_weights, rate)

        pcc_di = compute_pcc(ref_dec, captured_official[f"decoder.{i + 1}"])
        print(f"  Ref decoder.{i + 1}: {ref_dec.shape}, PCC: {pcc_di:.6f}")

        if pcc_di < 0.99:
            print(f"    *** DIVERGENCE at decoder.{i + 1} ***")
            print(f"    Ref: mean={ref_dec.mean():.4f}, std={ref_dec.std():.4f}")
            print(
                f"    Off: mean={captured_official[f'decoder.{i + 1}'].mean():.4f}, std={captured_official[f'decoder.{i + 1}'].std():.4f}"
            )

    # decoder.5 - snake activation
    if "decoder.5.alpha" in decoder_weights:
        ref_dec = snake_activation(ref_dec, decoder_weights["decoder.5.alpha"], decoder_weights["decoder.5.beta"])

    pcc_d5 = compute_pcc(ref_dec, captured_official["decoder.5"])
    print(f"  Ref decoder.5 (snake): {ref_dec.shape}, PCC: {pcc_d5:.6f}")

    # decoder.6 - final conv
    conv_weight = decoder_weights["decoder.6.conv.weight"]
    conv_bias = decoder_weights.get("decoder.6.conv.bias")
    kernel_size = conv_weight.shape[-1]
    ref_dec = F.pad(ref_dec, (kernel_size - 1, 0), mode="constant", value=0)
    ref_dec = F.conv1d(ref_dec, conv_weight, conv_bias)

    pcc_d6 = compute_pcc(ref_dec, captured_official["decoder.6"])
    print(f"  Ref decoder.6: {ref_dec.shape}, PCC: {pcc_d6:.6f}")

    # Final clamp
    ref_dec = ref_dec.clamp(min=-1, max=1)
    off_final = captured_official["decoder.6"].clamp(min=-1, max=1)

    pcc_final = compute_pcc(ref_dec, off_final)
    print(f"\n  Final audio PCC: {pcc_final:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
