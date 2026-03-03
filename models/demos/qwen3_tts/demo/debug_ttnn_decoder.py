# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Debug TTNN Decoder vs Reference

Compares TTNN and Reference decoder at each step to find where they diverge.
"""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    print("=" * 80)
    print("Debug TTNN Decoder vs Reference")
    print("=" * 80)

    # Load saved codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if not voice_clone_path.exists():
        print("ERROR: No saved codes found")
        return

    data = torch.load(voice_clone_path, weights_only=False)
    codes = data["ref_code"].T.unsqueeze(0)  # [1, 16, 101]
    print(f"Codes shape: {codes.shape}")

    # Load decoder weights
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["speech_tokenizer/*"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Strip decoder. prefix for decoder weights
    decoder_weights = {k[8:]: v for k, v in raw_dict.items() if k.startswith("decoder.")}
    # Keep quantizer weights with original keys for both implementations
    quantizer_weights = {k: v for k, v in raw_dict.items() if k.startswith("quantizer.")}
    all_weights = {**decoder_weights, **quantizer_weights}

    print(f"Loaded {len(decoder_weights)} decoder weights, {len(quantizer_weights)} quantizer weights")

    # =========================================================================
    # Step 1: Compare codebook lookup
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Codebook Lookup")
    print("=" * 80)

    # TTNN codebook lookup
    import ttnn
    from models.demos.qwen3_tts.reference.functional import SpeechTokenizerDecoderConfig
    from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig, TtSpeechTokenizerDecoder

    device = ttnn.open_device(device_id=0)
    tt_config = SpeechTokenizerConfig()
    tt_decoder = TtSpeechTokenizerDecoder(device, all_weights, tt_config, use_reference=False)

    ttnn_embeddings = tt_decoder._codebook_lookup(codes)
    print(
        f"  TTNN embeddings: shape={ttnn_embeddings.shape}, range=[{ttnn_embeddings.min():.4f}, {ttnn_embeddings.max():.4f}]"
    )
    print(f"  TTNN embeddings mean: {ttnn_embeddings.mean():.6f}, std: {ttnn_embeddings.std():.6f}")

    # Reference config for later
    ref_config = SpeechTokenizerDecoderConfig()
    pcc = 0.0  # Will be computed when we have ref embeddings

    # =========================================================================
    # Step 2: Compare codebook lookup with reference
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Compare Codebook Lookup")
    print("=" * 80)

    # Reference codebook lookup (from functional.py full decoder)
    from models.demos.qwen3_tts.reference.functional import pre_transformer_forward

    # Get reference embeddings by calling the full decoder and capturing intermediate
    # Actually let's just compare the shapes and ranges
    print(f"  TTNN codebook output: shape={ttnn_embeddings.shape}")
    print(f"  Expected for pre-transformer input: [batch, seq_len, 1024]")

    if tt_decoder.has_pre_transformer:
        print("\n  TTNN pre-transformer available")
        # Run pre-transformer on TTNN embeddings
        import ttnn as ttnn_module

        embeddings_ttnn = ttnn_module.from_torch(
            ttnn_embeddings.detach(),
            dtype=tt_decoder.dtype,
            layout=ttnn_module.TILE_LAYOUT,
            device=device,
        )
        seq_len = ttnn_embeddings.shape[1]
        cos, sin = tt_decoder._compute_rope(seq_len, ttnn_embeddings.device)
        cos_ttnn = ttnn_module.from_torch(
            cos.unsqueeze(0).unsqueeze(0),
            dtype=tt_decoder.dtype,
            layout=ttnn_module.TILE_LAYOUT,
            device=device,
        )
        sin_ttnn = ttnn_module.from_torch(
            sin.unsqueeze(0).unsqueeze(0),
            dtype=tt_decoder.dtype,
            layout=ttnn_module.TILE_LAYOUT,
            device=device,
        )

        # Forward through pre-transformer
        hidden_ttnn = tt_decoder.pre_transformer(embeddings_ttnn, cos_ttnn, sin_ttnn)
        hidden_torch = ttnn_module.to_torch(hidden_ttnn)
        print(f"  TTNN pre-transformer output: shape={hidden_torch.shape}")
        print(f"  TTNN pre-transformer range: [{hidden_torch.min():.4f}, {hidden_torch.max():.4f}]")
        print(f"  TTNN pre-transformer mean: {hidden_torch.mean():.4f}, std: {hidden_torch.std():.4f}")
    else:
        print("  TTNN pre-transformer NOT available")

    # =========================================================================
    # Step 2.5: Compare Pre-transformer with Reference
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2.5: Reference Pre-transformer Output")
    print("=" * 80)

    # Run reference pre-transformer on the same embeddings
    ref_pre_trans = pre_transformer_forward(ttnn_embeddings.detach(), all_weights, ref_config)
    print(f"  Reference pre-transformer output: shape={ref_pre_trans.shape}")
    print(f"  Reference pre-transformer range: [{ref_pre_trans.min():.4f}, {ref_pre_trans.max():.4f}]")
    print(f"  Reference pre-transformer mean: {ref_pre_trans.mean():.4f}, std: {ref_pre_trans.std():.4f}")

    if tt_decoder.has_pre_transformer:
        # Compare TTNN vs reference pre-transformer
        pre_trans_pcc = compute_pcc(hidden_torch.detach(), ref_pre_trans.detach())
        print(f"  Pre-transformer PCC: {pre_trans_pcc:.6f}")

    # =========================================================================
    # Step 3: Full decoder comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Full Decoder Output")
    print("=" * 80)

    # Reference decoder
    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_decoder_forward

    ref_audio = speech_tokenizer_decoder_forward(codes, all_weights, ref_config)
    print(f"  Reference audio: shape={ref_audio.shape}, range=[{ref_audio.min():.4f}, {ref_audio.max():.4f}]")

    # TTNN decoder
    ttnn_audio = tt_decoder(codes)
    print(f"  TTNN audio: shape={ttnn_audio.shape}, range=[{ttnn_audio.min():.4f}, {ttnn_audio.max():.4f}]")

    # Compare
    min_len = min(ref_audio.shape[-1], ttnn_audio.shape[-1])
    audio_pcc = compute_pcc(ref_audio[..., :min_len], ttnn_audio[..., :min_len])
    print(f"  Audio PCC: {audio_pcc:.6f}")

    # Energy envelope comparison
    window_size = 480
    ref_env = F.avg_pool1d(
        ref_audio.squeeze().abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()
    ttnn_env = F.avg_pool1d(
        ttnn_audio.squeeze().abs().unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=window_size // 2
    ).squeeze()

    energy_pcc = compute_pcc(ref_env, ttnn_env)
    print(f"  Energy PCC: {energy_pcc:.6f}")

    # =========================================================================
    # Step 4: Save outputs for listening
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Save Outputs")
    print("=" * 80)

    sf.write("/tmp/debug_ref_audio.wav", ref_audio.squeeze().detach().cpu().numpy(), 24000)
    sf.write("/tmp/debug_ttnn_audio.wav", ttnn_audio.squeeze().detach().cpu().numpy(), 24000)
    print("  Reference: /tmp/debug_ref_audio.wav")
    print("  TTNN:      /tmp/debug_ttnn_audio.wav")

    ttnn.close_device(device)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Codebook Lookup PCC: {pcc:.6f}")
    print(f"  Final Audio PCC:     {audio_pcc:.6f}")
    print(f"  Energy PCC:          {energy_pcc:.6f}")
    print()
    if pcc < 0.99:
        print("  ISSUE: Codebook lookup diverges!")
    elif audio_pcc < 0.5:
        print("  ISSUE: Full decoder diverges after codebook lookup")
    else:
        print("  Both implementations match!")
    print("=" * 80)


if __name__ == "__main__":
    main()
