# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Simple decoder debug - capture official outputs and compare with reference.
"""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Simple Decoder Debug")
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

    # Create test codes
    print("\n[3] Creating test codes...")
    torch.manual_seed(42)
    codes = torch.randint(0, 2048, (1, 16, 20))  # [batch, 16, seq_len]
    print(f"  Codes shape: {codes.shape}")

    # Capture official intermediate outputs
    print("\n[4] Running official decoder with hooks...")
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].clone().detach()
            elif hasattr(output, "last_hidden_state"):
                captured[name] = output.last_hidden_state.clone().detach()
            else:
                captured[name] = output.clone().detach()

        return hook

    hooks = []
    hooks.append(official_decoder.quantizer.register_forward_hook(make_hook("quantizer")))
    hooks.append(official_decoder.pre_conv.register_forward_hook(make_hook("pre_conv")))
    hooks.append(official_decoder.pre_transformer.register_forward_hook(make_hook("pre_transformer")))

    with torch.no_grad():
        official_audio = official_decoder(codes)

    for h in hooks:
        h.remove()

    print(f"  Captured stages: {list(captured.keys())}")
    for name, tensor in captured.items():
        print(f"    {name}: {tensor.shape}")

    # Now run reference
    print("\n[5] Running reference decoder...")
    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        codebook_lookup_rvq,
        pre_transformer_forward,
        speech_tokenizer_decoder_forward,
    )

    config = SpeechTokenizerDecoderConfig()

    # Stage 1: Quantizer
    print("\n  [Stage 1] Quantizer...")

    # Get weights
    rvq_first_emb = decoder_weights["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
    rvq_first_usage = decoder_weights["quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
    rvq_first_proj = decoder_weights["quantizer.rvq_first.output_proj.weight"]

    rvq_rest_codebooks = []
    rvq_rest_usages = []
    for i in range(15):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        rvq_rest_codebooks.append(decoder_weights[key])
        rvq_rest_usages.append(decoder_weights[usage_key])
    rvq_rest_proj = decoder_weights["quantizer.rvq_rest.output_proj.weight"]

    ref_quant = codebook_lookup_rvq(
        codes,
        rvq_first_emb,
        rvq_rest_codebooks,
        rvq_first_proj,
        rvq_rest_proj,
        rvq_first_usage,
        rvq_rest_usages,
    )
    print(f"    Reference quantizer: {ref_quant.shape}")

    # Run official quantizer directly
    with torch.no_grad():
        official_quant = official_decoder.quantizer.decode(codes)
    print(f"    Official quantizer: {official_quant.shape}")

    pcc_quant = compute_pcc(ref_quant, official_quant)
    print(f"    PCC: {pcc_quant:.6f}")

    if pcc_quant < 0.99:
        print(f"    *** MISMATCH ***")
        print(f"    Ref: mean={ref_quant.mean():.4f}, std={ref_quant.std():.4f}")
        print(f"    Off: mean={official_quant.mean():.4f}, std={official_quant.std():.4f}")

    # Stage 2: Pre-conv
    print("\n  [Stage 2] Pre-conv...")

    conv_weight = decoder_weights["pre_conv.conv.weight"]
    conv_bias = decoder_weights.get("pre_conv.conv.bias")
    kernel_size = conv_weight.shape[-1]

    # Causal padding
    ref_preconv_padded = F.pad(ref_quant, (kernel_size - 1, 0), mode="constant", value=0)
    ref_preconv = F.conv1d(ref_preconv_padded, conv_weight, conv_bias)

    print(f"    Reference pre_conv: {ref_preconv.shape}")
    print(f"    Official pre_conv: {captured['pre_conv'].shape}")

    pcc_preconv = compute_pcc(ref_preconv, captured["pre_conv"])
    print(f"    PCC: {pcc_preconv:.6f}")

    # Stage 3: Pre-transformer
    print("\n  [Stage 3] Pre-transformer...")

    pre_transformer_weights = {
        k.replace("pre_transformer.", ""): v for k, v in decoder_weights.items() if k.startswith("pre_transformer.")
    }

    # Transpose for transformer: [batch, seq_len, hidden]
    ref_pretrans_input = ref_preconv.transpose(1, 2)
    ref_pretrans = pre_transformer_forward(ref_pretrans_input, pre_transformer_weights, config)

    print(f"    Reference pre_transformer: {ref_pretrans.shape}")
    print(f"    Official pre_transformer: {captured['pre_transformer'].shape}")

    pcc_pretrans = compute_pcc(ref_pretrans, captured["pre_transformer"])
    print(f"    PCC: {pcc_pretrans:.6f}")

    if pcc_pretrans < 0.99:
        print(f"    *** MISMATCH ***")
        print(f"    Ref: mean={ref_pretrans.mean():.4f}, std={ref_pretrans.std():.4f}")
        print(f"    Off: mean={captured['pre_transformer'].mean():.4f}, std={captured['pre_transformer'].std():.4f}")

    # Final audio comparison
    print("\n  [Final] Audio...")

    ref_audio = speech_tokenizer_decoder_forward(codes, decoder_weights, config)
    print(f"    Reference audio: {ref_audio.shape}")
    print(f"    Official audio: {official_audio.shape}")

    # Handle size mismatch - truncate to shorter
    min_len = min(ref_audio.shape[-1], official_audio.shape[-1])
    ref_audio_trim = ref_audio[..., :min_len]
    off_audio_trim = official_audio[..., :min_len]

    pcc_audio = compute_pcc(ref_audio_trim, off_audio_trim)
    print(f"    Length diff: {ref_audio.shape[-1] - official_audio.shape[-1]} samples")
    print(f"    PCC (truncated): {pcc_audio:.6f}")

    if pcc_audio < 0.99:
        print(f"    *** MISMATCH ***")
        print(
            f"    Ref: mean={ref_audio.mean():.6f}, std={ref_audio.std():.4f}, range=[{ref_audio.min():.4f}, {ref_audio.max():.4f}]"
        )
        print(
            f"    Off: mean={official_audio.mean():.6f}, std={official_audio.std():.4f}, range=[{official_audio.min():.4f}, {official_audio.max():.4f}]"
        )

        # Find where the biggest differences are
        diff = (ref_audio - official_audio).abs()
        print(f"    Max diff: {diff.max():.6f} at position {diff.argmax().item()}")

    # Save both for listening comparison
    sf.write("/tmp/debug_ref.wav", ref_audio.squeeze().detach().cpu().numpy(), 24000)
    sf.write("/tmp/debug_off.wav", official_audio.squeeze().detach().cpu().numpy(), 24000)
    print(f"\n  Saved: /tmp/debug_ref.wav and /tmp/debug_off.wav")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Quantizer PCC:        {pcc_quant:.6f}")
    print(f"  Pre-conv PCC:         {pcc_preconv:.6f}")
    print(f"  Pre-transformer PCC:  {pcc_pretrans:.6f}")
    print(f"  Final Audio PCC:      {pcc_audio:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
