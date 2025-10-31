#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
HuggingFace SpeechT5 Reference Implementation.
This script helps us understand the expected behavior and outputs.
"""

import sys
import torch
import soundfile as sf

sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")

print("=" * 80)
print("HUGGINGFACE SPEECHT5 REFERENCE")
print("=" * 80)

# Step 1: Install dependencies (if needed)
print("\n[Step 1] Checking dependencies...")
try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset

    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease install:")
    print("  pip install transformers sentencepiece datasets[audio] soundfile")
    sys.exit(1)

# Step 2: Load models
print("\n[Step 2] Loading HuggingFace models...")
print("  - SpeechT5Processor...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
print("  - SpeechT5ForTextToSpeech...")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
print("  - SpeechT5HifiGan (vocoder)...")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
print("✓ Models loaded")

# Step 3: Load speaker embeddings
print("\n[Step 3] Loading speaker embeddings...")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
print(f"  Speaker embeddings shape: {speaker_embeddings.shape}")
print(f"  Speaker embeddings dtype: {speaker_embeddings.dtype}")
print("✓ Speaker embeddings loaded")

# Step 4: Process text
print("\n[Step 4] Processing input text...")
text = "Hello, my dog is cute."
print(f"  Input text: '{text}'")
inputs = processor(text=text, return_tensors="pt")
token_ids = inputs["input_ids"]
print(f"  Token IDs shape: {token_ids.shape}")
print(f"  Token IDs: {token_ids[0].tolist()[:20]}...")  # Show first 20 tokens
print(f"  Number of tokens: {token_ids.shape[1]}")
print("✓ Text processed")

# Step 5: Generate speech (with vocoder)
print("\n[Step 5] Generating speech with vocoder...")
with torch.no_grad():
    speech = model.generate_speech(token_ids, speaker_embeddings, vocoder=vocoder)
print(f"  Generated audio shape: {speech.shape}")
print(f"  Audio dtype: {speech.dtype}")
print(f"  Audio duration: {speech.shape[0] / 16000:.2f} seconds")
print(f"  Sample rate: 16000 Hz")

# Step 6: Save audio
print("\n[Step 6] Saving audio...")
output_file = "models/experimental/speecht5_tts/tests/speech_hf_reference.wav"
sf.write(output_file, speech.numpy(), samplerate=16000)
print(f"✓ Saved to {output_file}")

# Step 7: Analyze intermediate outputs (without vocoder)
print("\n[Step 7] Analyzing intermediate outputs...")
print("  Running model to get mel spectrogram...")

with torch.no_grad():
    # Get encoder output
    encoder_outputs = model.speecht5.encoder(input_values=token_ids, return_dict=True)
    encoder_hidden_states = encoder_outputs.last_hidden_state
    print(f"  Encoder output shape: {encoder_hidden_states.shape}")
    print(f"  Encoder output dtype: {encoder_hidden_states.dtype}")

    # Generate mel spectrogram (without vocoder)
    mel_outputs = model.generate_speech(token_ids, speaker_embeddings, vocoder=None)  # Return mel instead of audio
    print(f"  Mel spectrogram shape: {mel_outputs.shape}")
    print(f"  Mel spectrogram dtype: {mel_outputs.dtype}")
    print(f"  Number of mel frames: {mel_outputs.shape[0]}")
    print(f"  Number of mel bins: {mel_outputs.shape[1]}")

    # Save intermediate outputs
    torch.save(
        {
            "token_ids": token_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "mel_spectrogram": mel_outputs,
            "speaker_embeddings": speaker_embeddings,
            "text": text,
        },
        "models/experimental/speecht5_tts/tests/hf_reference_outputs.pt",
    )
    print(f"✓ Saved intermediate outputs to hf_reference_outputs.pt")

# Step 8: Model configuration
print("\n[Step 8] Model configuration...")
config = model.config
print(f"  Reduction factor: {config.reduction_factor}")
print(f"  Number of mel bins: {config.num_mel_bins}")
print(f"  Max decoder steps: {getattr(config, 'max_decoder_steps', 'N/A')}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Encoder layers: {config.encoder_layers}")
print(f"  Decoder layers: {config.decoder_layers}")
print(f"  Encoder attention heads: {config.encoder_attention_heads}")
print(f"  Decoder attention heads: {config.decoder_attention_heads}")
print(f"  Encoder FFN dim: {config.encoder_ffn_dim}")
print(f"  Decoder FFN dim: {config.decoder_ffn_dim}")

# Step 9: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n📝 Input:")
print(f"   Text: '{text}'")
print(f"   Tokens: {token_ids.shape[1]}")
print(f"\n🎯 Processing:")
print(f"   Encoder: {token_ids.shape} → {encoder_hidden_states.shape}")
print(f"   Decoder: Generated {mel_outputs.shape[0]} mel frames")
print(f"   Vocoder: {mel_outputs.shape} → {speech.shape[0]} audio samples")
print(f"\n🎵 Output:")
print(f"   Audio duration: {speech.shape[0] / 16000:.2f} seconds")
print(f"   Audio file: {output_file}")
print(f"\n✅ Reference generation complete!")
print(f"\n💡 Next steps:")
print(f"   1. Listen to {output_file}")
print(f"   2. Review intermediate outputs in hf_reference_outputs.pt")
print(f"   3. Compare with TTNN implementation")
print("=" * 80)
