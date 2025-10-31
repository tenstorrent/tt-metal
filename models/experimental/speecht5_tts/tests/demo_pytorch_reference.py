#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Reference Implementation Demo.

This script demonstrates SpeechT5 text-to-speech using the complete pure PyTorch
reference implementation. Unlike demo_hf_reference.py, this uses our op-by-op
PyTorch components instead of HuggingFace models.

The PyTorch reference is validated against HuggingFace (PCC ≈ 1.0) and serves
as the ground truth for TTNN implementation validation.
"""

import sys
import torch
import soundfile as sf
from datasets import load_dataset

sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")

print("=" * 80)
print("PYTORCH REFERENCE SPEECHT5 DEMO")
print("=" * 80)

# Step 1: Check dependencies
print("\n[Step 1] Checking dependencies...")
try:
    from transformers import SpeechT5Processor
    from models.experimental.speecht5_tts.reference import (
        load_full_reference_from_huggingface,
    )

    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease ensure the reference models are available.")
    sys.exit(1)

# Step 2: Load PyTorch reference model
print("\n[Step 2] Loading PyTorch reference model...")
print("  - This uses pure PyTorch implementations (no HF wrappers)")
print("  - All components validated against HF with PCC ≈ 1.0")

model = load_full_reference_from_huggingface()
print("✓ PyTorch reference model loaded")

# Step 3: Load processor and speaker embeddings
print("\n[Step 3] Loading processor and speaker embeddings...")
print("  - Processor: HuggingFace (for text tokenization)")
print("  - Speaker embeddings: From dataset")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
print(f"  Speaker embeddings shape: {speaker_embeddings.shape}")
print("✓ Processor and speaker embeddings loaded")

# Step 4: Process input text
print("\n[Step 4] Processing input text...")
text = "Hello, my dog is cute."
print(f"  Input text: '{text}'")

inputs = processor(text=text, return_tensors="pt")
token_ids = inputs["input_ids"]
print(f"  Token IDs shape: {token_ids.shape}")
print(f"  Token IDs: {token_ids[0].tolist()[:20]}...")  # Show first 20 tokens
print("✓ Text processed")

# Step 5: Generate speech
print("\n[Step 5] Generating speech with PyTorch reference...")
print("  - Using autoregressive generation")
print("  - All operations are explicit PyTorch implementations")

with torch.no_grad():
    # Generate speech with vocoder
    speech = model.generate_speech(token_ids, speaker_embeddings, vocoder=model.vocoder)

print(f"  Generated audio shape: {speech.shape}")
print(f"  Audio dtype: {speech.dtype}")
print(".2f")
print(f"  Sample rate: 16000 Hz")
print("✓ Speech generated")

# Step 6: Save audio
print("\n[Step 6] Saving audio...")
output_file = "models/experimental/speecht5_tts/tests/speech_pytorch_reference.wav"
sf.write(output_file, speech.numpy(), samplerate=16000)
print(f"✓ Saved to {output_file}")

# Step 7: Generate mel spectrogram only (without vocoder)
print("\n[Step 7] Analyzing intermediate mel spectrogram...")
with torch.no_grad():
    # Generate mel without vocoder
    mel_spectrogram = model.generate_speech(token_ids, speaker_embeddings, vocoder=None)  # Return mel instead of audio

print(f"  Mel spectrogram shape: {mel_spectrogram.shape}")
print(f"  Mel spectrogram dtype: {mel_spectrogram.dtype}")
print(f"  Number of mel frames: {mel_spectrogram.shape[0]}")
print(f"  Number of mel bins: {mel_spectrogram.shape[1]}")

# Step 8: Save intermediate outputs for comparison
print("\n[Step 8] Saving intermediate outputs...")
torch.save(
    {
        "token_ids": token_ids,
        "speaker_embeddings": speaker_embeddings,
        "mel_spectrogram": mel_spectrogram,
        "text": text,
        "model_config": model.config,
    },
    "models/experimental/speecht5_tts/tests/pytorch_reference_outputs.pt",
)
print("✓ Saved intermediate outputs to pytorch_reference_outputs.pt")

# Step 9: Model configuration summary
print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
config = model.config
print(f"  Hidden size: {config['hidden_size']}")
print(f"  Encoder layers: {config['encoder_layers']}")
print(f"  Decoder layers: {config['decoder_layers']}")
print(f"  Encoder attention heads: {config['encoder_attention_heads']}")
print(f"  Decoder attention heads: {config['decoder_attention_heads']}")
print(f"  Encoder FFN dim: {config['encoder_ffn_dim']}")
print(f"  Decoder FFN dim: {config['decoder_ffn_dim']}")
print(f"  Num mel bins: {config['num_mel_bins']}")
print(f"  Reduction factor: {config['reduction_factor']}")
print(f"  Speaker embedding dim: {config['speaker_embedding_dim']}")

# Step 10: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n📝 Input:")
print(f"   Text: '{text}'")
print(f"   Tokens: {token_ids.shape[1]}")
print("\n🎯 Processing:")
print("   Encoder: Pure PyTorch implementation (PCC=1.0 vs HF)")
print("   Decoder: Pure PyTorch implementation (PCC=1.0 vs HF)")
print("   Postnet: Pure PyTorch implementation (PCC=1.0 vs HF)")
print("   Vocoder: HiFiGAN (for audio generation)")
print(f"   Mel frames generated: {mel_spectrogram.shape[0]}")
print("\n🎵 Output:")
print(f"   Audio duration: {speech.shape[0] / 16000:.2f} seconds")
print(f"   Audio file: {output_file}")
print("\n✅ PyTorch reference generation complete!")
print("\n💡 Next steps:")
print(f"   1. Listen to {output_file}")
print(f"   2. Compare with HF reference: speech_hf_reference.wav")
print(f"   3. Review intermediate outputs in pytorch_reference_outputs.pt")
print(f"   4. Use this as ground truth for TTNN validation")
print("=" * 80)
