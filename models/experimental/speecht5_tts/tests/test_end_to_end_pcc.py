# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-End PCC Test with Real Text Input

Tests the complete SpeechT5 pipeline using the same inputs for both PyTorch and TTNN
to ensure accurate PCC measurements.

Pipeline:
1. Text: "Hello, my dog is cute."
2. Encoder: Same tokenized input → TTNN vs PyTorch encoder PCC
3. Decoder: PyTorch encoder output → TTNN vs PyTorch decoder PCC
4. Postnet: PyTorch decoder output → TTNN vs PyTorch postnet PCC

Setup:
    cd /home/ttuser/ssinghal/PR-fix/speecht5_tts_final/new/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    python models/experimental/speecht5_tts/tests/test_end_to_end_pcc.py
"""

import sys
from pathlib import Path
import torch
import ttnn

# Add tt-metal root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
from models.experimental.speecht5_tts.reference import (
    load_encoder_from_huggingface as load_encoder_ref,
    load_decoder_from_huggingface,
    load_postnet_from_huggingface as load_postnet_ref,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import TTNNSpeechT5Encoder, preprocess_encoder_parameters
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import TTNNSpeechT5Decoder, preprocess_decoder_parameters
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import TTNNEncoderConfig
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import TTNNDecoderConfig
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import TTNNPostNetConfig


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    tensor1_flat = tensor1.flatten().float()
    tensor2_flat = tensor2.flatten().float()

    mean1 = torch.mean(tensor1_flat)
    mean2 = torch.mean(tensor2_flat)

    tensor1_centered = tensor1_flat - mean1
    tensor2_centered = tensor2_flat - mean2

    numerator = torch.sum(tensor1_centered * tensor2_centered)
    denominator = torch.sqrt(torch.sum(tensor1_centered**2) * torch.sum(tensor2_centered**2))

    if denominator == 0:
        return 1.0 if torch.allclose(tensor1_flat, tensor2_flat) else 0.0

    return (numerator / denominator).item()


def test_end_to_end_pcc():
    """Test complete pipeline PCC with consistent inputs."""
    print("=" * 80)
    print("END-TO-END SPEECHT5 PCC TEST WITH REAL TEXT")
    print("=" * 80)

    # Test input
    text = "Hello, my dog is cute."
    print(f"\n📝 Input Text: '{text}'")

    # ========================================
    # Setup models and processor
    # ========================================
    print("\n🔧 Setting up models...")

    # Load processor and tokenize text
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Tokenized input shape: {input_ids.shape}")

    # Load HF model for reference
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    # Disable ALL dropout in HF model for fair comparison
    hf_model.speecht5.decoder.prenet.encode_positions.dropout.p = 0.0  # Positional dropout
    hf_model.speecht5.decoder.prenet.config.speech_decoder_prenet_dropout = 0.0  # Prenet consistent dropout

    # IMPORTANT: Monkey-patch HF's _consistent_dropout to actually skip dropout when p=0.0
    # HF's implementation zeros out values with p=0.0 (where(mask==1, x, 0) with all-zero mask)
    # but subsequent layer biases add back non-zero values, which is different from "no dropout"
    original_consistent_dropout = hf_model.speecht5.decoder.prenet._consistent_dropout

    def patched_consistent_dropout(inputs_embeds, p):
        if p == 0.0:
            return inputs_embeds  # Skip dropout entirely
        return original_consistent_dropout(inputs_embeds, p)

    hf_model.speecht5.decoder.prenet._consistent_dropout = patched_consistent_dropout

    # Load PyTorch reference models
    pytorch_encoder = load_encoder_ref()
    pytorch_decoder = load_decoder_from_huggingface()
    pytorch_postnet = load_postnet_ref()

    # Disable dropout on pytorch_decoder too (it loads a fresh HF model with default settings)
    pytorch_decoder.prenet.config.speech_decoder_prenet_dropout = 0.0  # Prenet consistent dropout
    pytorch_decoder.prenet.encode_positions.dropout.p = 0.0  # Positional dropout

    # Monkey-patch pytorch_decoder's _consistent_dropout too
    original_pt_consistent_dropout = pytorch_decoder.prenet._consistent_dropout

    def patched_pt_consistent_dropout(inputs_embeds, p):
        if p == 0.0:
            return inputs_embeds  # Skip dropout entirely
        return original_pt_consistent_dropout(inputs_embeds, p)

    pytorch_decoder.prenet._consistent_dropout = patched_pt_consistent_dropout

    # Setup TTNN device
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    # ========================================
    # Setup TTNN models
    # ========================================

    # TTNN Encoder
    ttnn_encoder_config = TTNNEncoderConfig(
        vocab_size=81,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.0,
        layer_norm_eps=1e-5,
        max_position_embeddings=600,
        max_relative_distance=160,
    )
    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, ttnn_encoder_config, device)
    ttnn_encoder = TTNNSpeechT5Encoder(device=device, parameters=encoder_params, config=ttnn_encoder_config)

    # TTNN Decoder
    ttnn_decoder_config = TTNNDecoderConfig(
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.0,
        layer_norm_eps=1e-5,
        max_position_embeddings=4000,
        num_mel_bins=80,
        reduction_factor=2,
        speech_decoder_prenet_units=256,
        speech_decoder_prenet_layers=2,
        speech_decoder_prenet_dropout=0.0,  # Disabled for fair PCC comparison with HF
    )
    max_sequence_length = 384

    # Create speaker embeddings for precomputation (random but deterministic)
    torch.manual_seed(42)
    speaker_embeddings = torch.randn(1, 512)  # batch_size=1, embedding_dim=512
    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder, ttnn_decoder_config, device, speaker_embeddings
    )
    ttnn_decoder = TTNNSpeechT5Decoder(
        device=device, parameters=decoder_params, config=ttnn_decoder_config, max_sequence_length=max_sequence_length
    )

    # TTNN Postnet
    ttnn_postnet_config = TTNNPostNetConfig(
        hidden_size=768,
        num_mel_bins=80,
        reduction_factor=2,
        postnet_layers=5,
        postnet_units=512,
        postnet_kernel=5,
        postnet_dropout=0.0,
    )
    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, ttnn_postnet_config, device)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
        device=device, parameters=postnet_params, config=ttnn_postnet_config
    )

    print("✅ All models loaded successfully")

    # ========================================
    # STAGE 1: Encoder PCC Test
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 1: ENCODER PCC")
    print("=" * 60)

    # PyTorch encoder forward
    pytorch_encoder.eval()
    with torch.no_grad():
        pytorch_encoder_output = pytorch_encoder(input_ids)
        if isinstance(pytorch_encoder_output, tuple):
            pytorch_encoder_output = pytorch_encoder_output[0]

    print(f"PyTorch encoder output shape: {pytorch_encoder_output.shape}")

    # TTNN encoder forward (use same conversion as working test)
    input_ids_ttnn = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_encoder_output = ttnn_encoder(input_ids_ttnn)
    # Handle tuple return (some models return tuples)
    if isinstance(ttnn_encoder_output, tuple):
        ttnn_encoder_output = ttnn_encoder_output[0]
    ttnn_encoder_output_torch = ttnn.to_torch(ttnn_encoder_output)

    print(f"TTNN encoder output shape: {ttnn_encoder_output_torch.shape}")

    # Compute encoder PCC
    encoder_pcc = compute_pcc(pytorch_encoder_output, ttnn_encoder_output_torch)
    print(f"\n📊 Encoder PCC: {encoder_pcc:.6f}")

    # ========================================
    # STAGE 2: Decoder PCC Test
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 2: DECODER PCC")
    print("=" * 60)

    # Create decoder inputs using PyTorch encoder output
    batch_size, seq_len, hidden_size = pytorch_encoder_output.shape
    decoder_seq_len = 10  # Same as in other tests

    # Create mel input (random but deterministic)
    torch.manual_seed(123)  # Use different seed to avoid interference with speaker_embeddings
    decoder_input = torch.randn(batch_size, decoder_seq_len, 80)  # mel bins

    # IMPORTANT: Reuse the SAME speaker_embeddings that was used for TTNN preprocessing (line 135)
    # The TTNN decoder precomputes normalized speaker embeddings, so we MUST use the same values
    # speaker_embeddings is already defined above (line 135) with torch.manual_seed(42)

    print(f"Decoder input (mel) shape: {decoder_input.shape}")
    print(f"Speaker embeddings shape: {speaker_embeddings.shape}")
    print(f"Encoder output shape: {pytorch_encoder_output.shape}")

    # PyTorch decoder forward
    pytorch_decoder.eval()
    with torch.no_grad():
        pytorch_decoder_output = pytorch_decoder(
            decoder_input_values=decoder_input,
            encoder_hidden_states=pytorch_encoder_output,
            speaker_embeddings=speaker_embeddings,
        )

    print(f"PyTorch decoder output shape: {pytorch_decoder_output.shape}")

    # TTNN decoder forward
    decoder_input_ttnn = ttnn.from_torch(decoder_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    encoder_output_ttnn = ttnn.from_torch(
        pytorch_encoder_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    speaker_embeddings_ttnn = ttnn.from_torch(
        speaker_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_decoder_output = ttnn_decoder(
        decoder_input_values=decoder_input_ttnn,
        encoder_hidden_states=encoder_output_ttnn,
        speaker_embeddings=speaker_embeddings_ttnn,
    )
    # Handle tuple return
    if isinstance(ttnn_decoder_output, tuple):
        ttnn_decoder_output = ttnn_decoder_output[0]
    ttnn_decoder_output_torch = ttnn.to_torch(ttnn_decoder_output)

    print(f"TTNN decoder output shape: {ttnn_decoder_output_torch.shape}")

    # Compute decoder PCC
    decoder_pcc = compute_pcc(pytorch_decoder_output, ttnn_decoder_output_torch)
    print(f"\n📊 Decoder PCC: {decoder_pcc:.6f}")

    # ========================================
    # STAGE 3: Postnet PCC Test
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 3: POSTNET PCC")
    print("=" * 60)

    # PyTorch postnet forward
    pytorch_postnet.eval()
    with torch.no_grad():
        pytorch_postnet_output = pytorch_postnet(pytorch_decoder_output)

    print(f"PyTorch postnet output shapes: {len(pytorch_postnet_output)} tensors")
    print(f"  Pre-postnet: {pytorch_postnet_output[0].shape}")
    print(f"  Post-postnet: {pytorch_postnet_output[1].shape}")
    print(f"  Stop logits: {pytorch_postnet_output[2].shape}")

    # TTNN postnet forward
    decoder_output_ttnn = ttnn.from_torch(
        pytorch_decoder_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_postnet_output = ttnn_postnet(decoder_output_ttnn)
    # Postnet always returns a 3-tuple: (pre_postnet, post_postnet, stop_logits)
    ttnn_pre_postnet = ttnn.to_torch(ttnn_postnet_output[0])
    ttnn_post_postnet = ttnn.to_torch(ttnn_postnet_output[1])
    ttnn_stop_logits = ttnn.to_torch(ttnn_postnet_output[2])

    print(f"TTNN postnet output shapes:")
    print(f"  Pre-postnet: {ttnn_pre_postnet.shape}")
    print(f"  Post-postnet: {ttnn_post_postnet.shape}")
    print(f"  Stop logits: {ttnn_stop_logits.shape}")

    # Compute postnet PCCs
    pre_postnet_pcc = compute_pcc(pytorch_postnet_output[0], ttnn_pre_postnet)
    post_postnet_pcc = compute_pcc(pytorch_postnet_output[1], ttnn_post_postnet)
    stop_logits_pcc = compute_pcc(pytorch_postnet_output[2], ttnn_stop_logits)

    print(f"\n📊 Postnet PCCs:")
    print(f"  Pre-postnet: {pre_postnet_pcc:.6f}")
    print(f"  Post-postnet: {post_postnet_pcc:.6f}")
    print(f"  Stop logits: {stop_logits_pcc:.6f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("END-TO-END PCC SUMMARY")
    print("=" * 80)
    print(f"Input Text: '{text}'")
    print(f"Tokenized Shape: {input_ids.shape}")
    print()
    print("PCC Results:")
    print(f"  Encoder: {encoder_pcc:.6f}")
    print(f"  Decoder: {decoder_pcc:.6f}")
    print(f"  Pre-Postnet: {pre_postnet_pcc:.6f}")
    print(f"  Post-Postnet: {post_postnet_pcc:.6f}")
    print(f"  Stop Logits: {stop_logits_pcc:.6f}")
    # Cleanup
    ttnn.close_device(device)

    return {
        "encoder_pcc": encoder_pcc,
        "decoder_pcc": decoder_pcc,
        "pre_postnet_pcc": pre_postnet_pcc,
        "post_postnet_pcc": post_postnet_pcc,
        "stop_logits_pcc": stop_logits_pcc,
    }


if __name__ == "__main__":
    results = test_end_to_end_pcc()
    print(f"\n✅ Test completed. Results: {results}")
