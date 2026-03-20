# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test the fixed speech tokenizer decoder.
"""

from pathlib import Path

import soundfile as sf
import torch


def test_fixed_decoder():
    """Test that the fixed reference decoder produces correct audio."""
    print("=" * 80)
    print("Testing Fixed Speech Tokenizer Decoder")
    print("=" * 80)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from models.demos.qwen3_tts.reference.functional import (
        SpeechTokenizerDecoderConfig,
        speech_tokenizer_decoder_forward,
    )

    # Load weights
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    model_path = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    model_path = Path(model_path)
    speech_tokenizer_path = model_path / "speech_tokenizer" / "model.safetensors"
    raw_dict = load_file(speech_tokenizer_path)

    # Strip decoder prefix
    state_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("decoder."):
            state_dict[k[8:]] = v

    config = SpeechTokenizerDecoderConfig()

    # Get test codes
    voice_clone_path = Path("/tmp/qwen_tts_tensors/voice_clone_prompt_full.pt")
    if voice_clone_path.exists():
        data = torch.load(voice_clone_path)
        ref_code = data["ref_code"]
        codes = ref_code.T.unsqueeze(0)  # [1, 16, 101]
        print(f"  Loaded ref_code from voice clone: {codes.shape}")
    else:
        codes = torch.randint(0, 2048, (1, 16, 50))
        print(f"  Using random codes: {codes.shape}")

    # Test intermediate steps
    print("\n--- Testing intermediate steps ---")

    # 1. Test codebook lookup
    from models.demos.qwen3_tts.reference.functional import codebook_lookup_rvq

    rvq_first_codebook = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    rvq_first_cluster_usage = state_dict.get("quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
    rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")

    rvq_rest_codebooks = []
    rvq_rest_cluster_usages = []
    for i in range(15):
        key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
        usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
        if key in state_dict:
            rvq_rest_codebooks.append(state_dict[key])
            rvq_rest_cluster_usages.append(state_dict.get(usage_key))
    rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

    embeddings = codebook_lookup_rvq(
        codes,
        rvq_first_codebook,
        rvq_rest_codebooks,
        rvq_first_output_proj,
        rvq_rest_output_proj,
        rvq_first_cluster_usage,
        rvq_rest_cluster_usages,
    )
    print(f"  Codebook embeddings shape: {embeddings.shape}")  # Should be [1, 256, seq_len]
    print(f"  Codebook embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

    # 2. Run full decoder
    print("\n--- Running full decoder ---")
    with torch.no_grad():
        audio = speech_tokenizer_decoder_forward(codes, state_dict, config)
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save audio
    audio_np = audio.squeeze().cpu().float().numpy()
    output_path = "/tmp/fixed_reference_decoder.wav"
    sf.write(output_path, audio_np, 24000)
    print(f"\n  Saved audio to: {output_path}")
    print("  LISTEN to verify it's correct speech!")

    # Check for noise characteristics
    audio_tensor = torch.tensor(audio_np)
    std = audio_tensor.std().item()
    mean = audio_tensor.abs().mean().item()
    print(f"\n  Audio statistics:")
    print(f"    Std: {std:.4f}")
    print(f"    Mean absolute: {mean:.4f}")

    # If std is very high and mean is near 0.5, it's likely noise
    if std > 0.5 and mean > 0.3:
        print("\n  WARNING: Audio statistics suggest this might be noise!")
        print("  Please listen to verify.")
    else:
        print("\n  Audio statistics look reasonable for speech.")

    return audio


if __name__ == "__main__":
    test_fixed_decoder()
