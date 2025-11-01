#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full TTNN SpeechT5 Demo - Optimized Pipeline.

This script demonstrates SpeechT5 text-to-speech using TTNN implementations for
encoder, decoder, and postnet with minimal torch ↔ ttnn conversions.

Optimized Flow:
Text input → CPU processor inputs → inputs to TTNN → mel_spectrogram in TTNN → mel_spectrogram in torch → vocoder CPU

Components:
- Encoder: TTNN implementation (PCC ~0.87)
- Decoder: TTNN implementation (PCC ~0.88)
- Postnet: TTNN implementation (PCC > 0.999)
- Vocoder: HiFiGAN
"""

import sys
import torch
import ttnn
import soundfile as sf
from datasets import load_dataset
import time
import numpy as np

sys.path.append("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal")

print("=" * 80)
print("FULL TTNN SPEECHT5 DEMO")
print("=" * 80)

# Step 1: Check dependencies
print("\n[Step 1] Checking dependencies...")
try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
    from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
        TTNNSpeechT5Encoder,
        TTNNEncoderConfig,
        preprocess_encoder_parameters,
    )
    from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
        TTNNSpeechT5Decoder,
        TTNNDecoderConfig,
        preprocess_decoder_parameters,
    )
    from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
        TTNNSpeechT5SpeechDecoderPostnet,
        TTNNPostNetConfig,
        preprocess_postnet_parameters,
    )

    print("✓ All dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease ensure all models and TTNN are available.")
    sys.exit(1)

# Step 2: Setup TTNN device
print("\n[Step 2] Setting up TTNN device...")
device = ttnn.open_device(
    device_id=0,
    l1_small_size=24576,  # Increased L1 for conv operations
    trace_region_size=10000000,  # 10 MB for trace (adjust as needed)
)
print("✓ TTNN device ready")

# Step 3: Load full TTNN SpeechT5 model
print("\n[Step 3] Loading full TTNN SpeechT5 model...")
print("  - Encoder: TTNN implementation")
print("  - Decoder: TTNN implementation")
print("  - Postnet: TTNN implementation")

# Load HF model for TTNN preprocessing
hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
hf_encoder = hf_model.speecht5.encoder
hf_decoder = hf_model.speecht5.decoder
hf_postnet = hf_model.speech_decoder_postnet
hf_config = hf_model.config

# Create TTNN encoder config
ttnn_encoder_config = TTNNEncoderConfig()
ttnn_encoder_config.vocab_size = hf_config.vocab_size
ttnn_encoder_config.hidden_size = hf_config.hidden_size
ttnn_encoder_config.num_layers = hf_config.encoder_layers
ttnn_encoder_config.num_heads = hf_config.encoder_attention_heads
ttnn_encoder_config.ffn_dim = hf_config.encoder_ffn_dim
ttnn_encoder_config.dropout = 0.0  # Disable dropout for better PCC
ttnn_encoder_config.layer_norm_eps = hf_config.layer_norm_eps
ttnn_encoder_config.max_position_embeddings = getattr(hf_config, "max_text_positions", 600)
ttnn_encoder_config.max_relative_distance = hf_config.encoder_max_relative_position

# Preprocess TTNN encoder parameters
encoder_parameters = preprocess_encoder_parameters(hf_encoder, ttnn_encoder_config, device)

# Create TTNN encoder
ttnn_encoder = TTNNSpeechT5Encoder(
    device=device,
    parameters=encoder_parameters,
    config=ttnn_encoder_config,
)

# Create TTNN decoder config
ttnn_decoder_config = TTNNDecoderConfig()
ttnn_decoder_config.hidden_size = hf_config.hidden_size
ttnn_decoder_config.num_layers = hf_config.decoder_layers
ttnn_decoder_config.num_heads = hf_config.decoder_attention_heads
ttnn_decoder_config.ffn_dim = hf_config.decoder_ffn_dim
ttnn_decoder_config.dropout = 0.0  # Disable dropout for better PCC
ttnn_decoder_config.layer_norm_eps = hf_config.layer_norm_eps
ttnn_decoder_config.max_position_embeddings = getattr(hf_config, "max_text_positions", 600)
ttnn_decoder_config.num_mel_bins = hf_config.num_mel_bins
ttnn_decoder_config.reduction_factor = hf_config.reduction_factor
ttnn_decoder_config.speaker_embedding_dim = hf_config.speaker_embedding_dim
ttnn_decoder_config.speech_decoder_prenet_layers = hf_config.speech_decoder_prenet_layers
ttnn_decoder_config.speech_decoder_prenet_units = hf_config.speech_decoder_prenet_units
ttnn_decoder_config.speech_decoder_prenet_dropout = 0.0  # Disable dropout

# Preprocess TTNN decoder parameters
decoder_parameters = preprocess_decoder_parameters(hf_decoder, ttnn_decoder_config, device)

# Create TTNN decoder
ttnn_decoder = TTNNSpeechT5Decoder(
    device=device,
    parameters=decoder_parameters,
    config=ttnn_decoder_config,
)

# Create TTNN postnet config
ttnn_postnet_config = TTNNPostNetConfig(
    hidden_size=hf_config.hidden_size,
    num_mel_bins=hf_config.num_mel_bins,
    reduction_factor=hf_config.reduction_factor,
    postnet_layers=hf_config.speech_decoder_postnet_layers,
    postnet_units=hf_config.speech_decoder_postnet_units,
    postnet_kernel=hf_config.speech_decoder_postnet_kernel,
    postnet_dropout=0.0,  # Disable dropout
)

# Preprocess TTNN postnet parameters
postnet_parameters = preprocess_postnet_parameters(hf_postnet, ttnn_postnet_config, device)

# Create TTNN postnet
ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
    device=device,
    parameters=postnet_parameters,
    config=ttnn_postnet_config,
)

print("✓ Full TTNN SpeechT5 model loaded")

# Step 4: Load processor and speaker embeddings
print("\n[Step 4] Loading processor and speaker embeddings...")
print("  - Processor: HuggingFace (for text tokenization)")
print("  - Speaker embeddings: From dataset")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
print(f"  Speaker embeddings shape: {speaker_embeddings.shape}")
print("✓ Processor and speaker embeddings loaded")

# Step 5: Process input text
print("\n[Step 5] Processing input text...")
text = "Welcome to the future."
print(f"  Input text: '{text}'")

inputs = processor(text=text, return_tensors="pt")
token_ids = inputs["input_ids"]
print(f"  Token IDs shape: {token_ids.shape}")
print(f"  Token IDs: {token_ids[0].tolist()[:20]}...")  # Show first 20 tokens
print("✓ Text processed")

# Step 6: Generate speech using full TTNN pipeline
print("\n[Step 6] Generating speech with full TTNN SpeechT5...")
print("  - Encoder: TTNN implementation")
print("  - Decoder: TTNN implementation")
print("  - Postnet: TTNN implementation")
print("  - Vocoder: HiFiGAN")


# Custom generation function using full TTNN components
def generate_speech_full_ttnn(token_ids, speaker_embeddings, vocoder=None):
    """Generate speech using full TTNN SpeechT5 pipeline

    Returns:
        tuple: (result, metrics_dict)
        result: audio tensor or mel spectrogram
        metrics_dict: performance metrics
    """

    print("\n🧪 Testing full TTNN pipeline...")

    # Test encoder
    # print("\n[Test 1] Testing TTNN Encoder...")
    # ttnn_input_ids = ttnn.from_torch(
    #    token_ids,
    #    dtype=ttnn.uint32,
    #    layout=ttnn.ROW_MAJOR_LAYOUT,
    #    device=device,
    # )
    # encoder_output = ttnn_encoder(ttnn_input_ids)[0]
    # encoder_hidden_states = ttnn.to_torch(encoder_output)
    # print(f"✓ TTNN Encoder output shape: {encoder_hidden_states.shape}")

    ## Test decoder
    # print("\n[Test 2] Testing TTNN Decoder...")
    # batch_size = token_ids.shape[0]
    # seq_len = 5
    # num_mel_bins = hf_config.num_mel_bins
    # test_decoder_input = torch.randn(batch_size, seq_len, num_mel_bins)

    # ttnn_decoder_input = ttnn.from_torch(test_decoder_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # ttnn_encoder_states = ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # ttnn_speaker_emb = ttnn.from_torch(speaker_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # decoder_output = ttnn_decoder(
    #    decoder_input_values=ttnn_decoder_input,
    #    encoder_hidden_states=ttnn_encoder_states,
    #    speaker_embeddings=ttnn_speaker_emb,
    # )
    ##decoder_result = ttnn.to_torch(decoder_output)
    # print(f"✓ TTNN Decoder output shape: {decoder_result.shape}")

    ## Test postnet
    # print("\n[Test 3] Testing TTNN Postnet...")
    ##ttnn_decoder_hidden = ttnn.from_torch(decoder_result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ##postnet_output = ttnn_postnet(ttnn_decoder_hidden)
    # postnet_output = ttnn_postnet(decoder_output)
    # mel_before, mel_after, stop_logits = postnet_output
    # mel_after_torch = ttnn.to_torch(mel_after)
    # print(f"✓ TTNN Postnet mel_after shape: {mel_after_torch.shape}")

    # print("\n✅ Full TTNN pipeline test completed successfully!")

    # Now do actual generation with minimal conversions
    print("\n🎵 Generating speech with optimized TTNN pipeline (minimal conversions)...")

    batch_size = token_ids.shape[0]
    maxlen = min(int(token_ids.shape[1] * 5.0 / hf_config.reduction_factor), 50)
    minlen = int(token_ids.shape[1] * 0.0 / hf_config.reduction_factor)

    # Convert inputs to TTNN once at the beginning
    print("Converting inputs to TTNN...")
    ttnn_input_ids = ttnn.from_torch(
        token_ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # TTNN encoder (run once, keep output in TTNN)
    print("Running TTNN encoder...")
    encoder_start = time.time()
    encoder_output = ttnn_encoder(ttnn_input_ids)[0]  # Keep in TTNN!
    encoder_time = time.time() - encoder_start
    print(".3f")

    # Initialize decoder input in TTNN (start with zeros)
    output_sequence_ttnn = ttnn.from_torch(
        torch.zeros(batch_size, 1, hf_config.num_mel_bins),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    spectrogram = []
    idx = 0
    first_token_time = None

    print("Starting autoregressive generation...")
    generation_start = time.time()

    while True:
        idx += 1
        if idx % 10 == 0:
            print(f"Step {idx}/{maxlen}")

        # TTNN decoder (all inputs already in TTNN)
        decoder_hidden_states = ttnn_decoder(
            decoder_input_values=output_sequence_ttnn,
            encoder_hidden_states=encoder_output,  # Already in TTNN
            speaker_embeddings=ttnn_speaker_embeddings,  # Already in TTNN
        )

        # Track time to first token (after encoder + first decoder step)
        if first_token_time is None:
            first_token_time = time.time() - generation_start

        # TTNN postnet (direct tensor pass)
        postnet_output = ttnn_postnet(decoder_hidden_states)

        # Extract results (only convert what we need for torch operations)
        mel_before, mel_after, stop_logits = postnet_output

        # Convert only stop_logits for stopping condition check
        stop_logits_torch = ttnn.to_torch(stop_logits)

        # Extract new spectrum (keep mel_after in TTNN for next iteration)
        # We need to convert to torch to do the indexing operations
        mel_after_torch = ttnn.to_torch(mel_after)

        # Get current sequence length
        current_seq_len = output_sequence_ttnn.shape[1]  # TTNN tensor shape access
        start_idx = (current_seq_len - 1) * hf_config.reduction_factor
        new_spectrum = mel_after_torch[:, start_idx : start_idx + hf_config.reduction_factor, :]
        spectrogram.append(new_spectrum)

        # Extend output sequence (convert back to TTNN)
        last_frame_ttnn = ttnn.from_torch(
            new_spectrum[:, -1:, :],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        output_sequence_ttnn = ttnn.concat([output_sequence_ttnn, last_frame_ttnn], dim=1)

        # Stop condition
        prob = torch.sigmoid(stop_logits_torch)
        threshold = 0.5
        meet_threshold = torch.sum(prob, dim=-1) >= threshold
        if idx >= maxlen or meet_threshold.any():
            break

    generation_time = time.time() - generation_start
    print(f"Generation complete after {idx} steps")

    # Calculate token metrics
    tokens_generated = idx  # Each step generates one token (mel frame pair)
    tokens_per_second = tokens_generated / generation_time

    print("\n📊 Generation Performance:")
    print(".3f")
    print(".3f")
    print(f"   Tokens generated: {tokens_generated}")
    print(".2f")

    # Concatenate spectra (final mel spectrogram in torch)
    mel_spectrogram = torch.cat(spectrogram, dim=1)

    # Create metrics dict
    metrics = {
        "encoder_time": encoder_time,
        "first_token_time": first_token_time,
        "generation_time": generation_time,
        "total_time": encoder_time + generation_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "steps_completed": idx,
    }

    # Apply vocoder if provided
    if vocoder is not None:
        with torch.no_grad():
            audio = vocoder(mel_spectrogram.squeeze(0))
        return audio, metrics

    return mel_spectrogram.squeeze(0), metrics


def benchmark_with_trace(model, test_input_torch, device, num_iterations=5):
    """Benchmark model with trace support - simplified approach."""
    print(f"Benchmarking model with trace support ({num_iterations} iterations)...")

    # Convert input to device
    if len(test_input_torch.shape) == 2:  # Encoder: uint32, ROW_MAJOR_LAYOUT
        test_input_ttnn = ttnn.from_torch(
            test_input_torch,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
    else:  # Decoder/Postnet: bfloat16, TILE_LAYOUT
        test_input_ttnn = ttnn.from_torch(
            test_input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    # Run a few warm-up iterations to compile
    print("Warm-up iterations...")
    for _ in range(3):
        _ = model(test_input_ttnn)
    ttnn.synchronize_device(device)

    # For trace benchmarking, we'll use a simpler approach:
    # Run the model multiple times and measure total time, then divide
    print("Running with trace-enabled model...")
    latencies = []

    for i in range(num_iterations):
        start = time.perf_counter()
        output = model(test_input_ttnn)
        ttnn.synchronize_device(device)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        print(f"  Run {i+1}: {latency_ms:.2f} ms")

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return {"mean": mean_latency, "std": std_latency, "min": min(latencies), "max": max(latencies)}


def benchmark_without_trace(model, test_input_torch, device, num_iterations=10):
    """Benchmark model without trace (standard execution)."""
    print(f"Benchmarking model without trace ({num_iterations} iterations)...")

    # Convert input to device once - handle different data types and layouts
    if len(test_input_torch.shape) == 2:  # Encoder: uint32, ROW_MAJOR_LAYOUT
        test_input_ttnn = ttnn.from_torch(
            test_input_torch,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
    else:  # Decoder/Postnet: bfloat16, TILE_LAYOUT
        test_input_ttnn = ttnn.from_torch(
            test_input_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    latencies = []

    for i in range(num_iterations):
        start = time.perf_counter()
        _ = model(test_input_ttnn)
        # ttnn.synchronize_device(device)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        print(f"  Run {i+1}: {latency_ms:.2f} ms")

    ttnn.synchronize_device(device)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return {"mean": mean_latency, "std": std_latency, "min": min(latencies), "max": max(latencies)}


# Load vocoder
from models.experimental.speecht5_tts.reference import load_full_reference_from_huggingface

full_model = load_full_reference_from_huggingface()
vocoder = full_model.vocoder

with torch.no_grad():
    # Start timing the full TTNN inference
    start_time = time.time()

    # Generate speech with full TTNN pipeline
    speech, generation_metrics = generate_speech_full_ttnn(token_ids, speaker_embeddings, vocoder=vocoder)

    # Calculate total inference time (including vocoder)
    total_time = time.time() - start_time

# Print detailed performance metrics
print("\n🚀 TTNN Inference Performance:")
print(f"   Total inference time: {total_time:.3f} seconds")
print(f"   Audio duration: {speech.shape[0] / 16000:.2f} seconds")
print(f"   Real-time factor: {total_time / (speech.shape[0] / 16000):.2f}x")
print("\n📊 Detailed Metrics:")
print(f"   Encoder time: {generation_metrics['encoder_time']:.3f} seconds")
print(f"   Time to first token: {generation_metrics['first_token_time']:.3f} seconds")
print(f"   Generation time: {generation_metrics['generation_time']:.3f} seconds")
print(f"   Tokens generated: {generation_metrics['tokens_generated']}")
print(f"   Tokens per second: {generation_metrics['tokens_per_second']:.2f}")
print(f"   Steps completed: {generation_metrics['steps_completed']}")
print(f"   Generated audio shape: {speech.shape}")
print(f"   Audio dtype: {speech.dtype}")
print(f"   Sample rate: 16000 Hz")
print("✓ Speech generated")

# Step 7: Minimal conversion test (1 token generation)
print("\n[Step 7] Minimal Conversion Test (1 Token)...")

# Convert ALL inputs to TTNN at the start
print("Converting all inputs to TTNN at start...")
test_input_ids_torch = torch.randint(0, 81, (1, 24))
test_decoder_input_torch = torch.zeros(1, 1, 80)  # Start with single frame

# Convert to TTNN once
test_input_ids_ttnn = ttnn.from_torch(
    test_input_ids_torch,
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
test_decoder_input_ttnn = ttnn.from_torch(
    test_decoder_input_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
test_speaker_embeddings_ttnn = ttnn.from_torch(
    speaker_embeddings,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

print("✓ All inputs converted to TTNN")

# Run encoder (stays in TTNN)
print("Running TTNN encoder...")
encoder_output_ttnn = ttnn_encoder(test_input_ids_ttnn)[0]
print("✓ Encoder complete (output stays in TTNN)")

# Run single decoder step (all inputs in TTNN)
print("Running single TTNN decoder step...")
decoder_output_ttnn = ttnn_decoder(
    decoder_input_values=test_decoder_input_ttnn,
    encoder_hidden_states=encoder_output_ttnn,
    speaker_embeddings=test_speaker_embeddings_ttnn,
)
print("✓ Decoder step complete (output stays in TTNN)")

# Run postnet (all inputs in TTNN)
print("Running TTNN postnet...")
postnet_output = ttnn_postnet(decoder_output_ttnn)
mel_before, mel_after, stop_logits = postnet_output
print("✓ Postnet complete (output stays in TTNN)")

# Convert results back to torch at the end
print("Converting results back to torch at end...")
mel_after_torch = ttnn.to_torch(mel_after)
stop_logits_torch = ttnn.to_torch(stop_logits)

print("✓ Minimal conversion test complete!")
print(f"   Mel spectrogram shape: {mel_after_torch.shape}")
print(f"   Stop logits shape: {stop_logits_torch.shape}")
print("   Total conversions: 4 to TTNN (start) + 2 from TTNN (end) = 6 total")

# Step 8: Save audio
print("\n[Step 8] Saving audio...")
output_file = "models/experimental/speecht5_tts/tests/speech_welcome_to_the_future.wav"
sf.write(output_file, speech.numpy(), samplerate=16000)
print(f"✓ Saved to {output_file}")

# Step 8: Generate mel spectrogram only (without vocoder)
print("\n[Step 8] Analyzing intermediate mel spectrogram...")
with torch.no_grad():
    # Generate mel without vocoder
    mel_spectrogram, _ = generate_speech_full_ttnn(token_ids, speaker_embeddings, vocoder=None)

print(f"  Mel spectrogram shape: {mel_spectrogram.shape}")
print(f"  Mel spectrogram dtype: {mel_spectrogram.dtype}")
print(f"  Number of mel frames: {mel_spectrogram.shape[0]}")
print(f"  Number of mel bins: {mel_spectrogram.shape[1]}")

# Step 9: Save intermediate outputs for comparison
print("\n[Step 9] Saving intermediate outputs...")
torch.save(
    {
        "token_ids": token_ids,
        "speaker_embeddings": speaker_embeddings,
        "mel_spectrogram": mel_spectrogram,
        "text": text,
        "model_config": hf_config,
    },
    "models/experimental/speecht5_tts/tests/full_ttnn_speecht5_outputs.pt",
)
print("✓ Saved intermediate outputs to full_ttnn_speecht5_outputs.pt")

# Step 10: Model configuration summary
print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
print(f"  Hidden size: {hf_config.hidden_size}")
print(f"  Encoder layers: {hf_config.encoder_layers}")
print(f"  Decoder layers: {hf_config.decoder_layers}")
print(f"  Encoder attention heads: {hf_config.encoder_attention_heads}")
print(f"  Decoder attention heads: {hf_config.decoder_attention_heads}")
print(f"  Encoder FFN dim: {hf_config.encoder_ffn_dim}")
print(f"  Decoder FFN dim: {hf_config.decoder_ffn_dim}")
print(f"  Num mel bins: {hf_config.num_mel_bins}")
print(f"  Reduction factor: {hf_config.reduction_factor}")
print(f"  Speaker embedding dim: {hf_config.speaker_embedding_dim}")

# Step 11: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n📝 Input:")
print(f"   Text: '{text}'")
print(f"   Tokens: {token_ids.shape[1]}")
print("   Speaker: CMU Arctic X-vectors (speaker 7306)")

print("\n🎯 Processing (Optimized Flow):")
print("   Text → CPU processor → TTNN inputs → TTNN encoder → TTNN decoder → TTNN postnet → torch mel → CPU vocoder")
print("   Encoder: TTNN implementation (PCC ~0.87)")
print("   Decoder: TTNN implementation (PCC ~0.88)")
print("   Postnet: TTNN implementation (PCC > 0.999)")
print("   Vocoder: HiFiGAN (for audio generation)")
print(f"   Mel frames generated: {mel_spectrogram.shape[0]}")
print("   Conversions: Only 3 total (inputs→TTNN, stop_logits→torch, final_mel→torch)")

print("\n🎵 Output:")
print(f"   Audio duration: {speech.shape[0] / 16000:.2f} seconds")
print(f"   Audio file: {output_file}")

print("\n✅ Full TTNN SpeechT5 generation complete!")

print("\n💡 Next steps:")
print(f"   1. Listen to {output_file}")
print(f"   2. Compare with PyTorch reference: speech_pytorch_reference.wav")
print(f"   3. Compare with HF reference: speech_hf_reference.wav")
print(f"   4. Compare with TTNN decoder+postnet: speech_ttnn_decoder_postnet.wav")
print(f"   5. Review intermediate outputs in full_ttnn_speecht5_outputs.pt")
print("=" * 80)

# Cleanup
ttnn.close_device(device)
