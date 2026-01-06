#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-Layer PCC Tracking During Autoregressive Generation

Tracks PCC at each autoregressive step:
- Encoder: final output (once at start)
- Decoder: final hidden state output
- MelBefore: after feat_out projection (before postnet convs)
- Postnet: conv 0-4 outputs (each layer)
- MelAfter: after residual addition (mel_before + conv_output)

Two modes:
1. Multi-step: Both use PyTorch output as ground truth input (fair comparison)
2. True autoregressive: Each feeds its own output back (shows divergence)

Setup:
    cd /home/ttuser/ssinghal/PR-fix/speecht5_tts_final/new/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    # Multi-step with 20 steps (default)
    python models/experimental/speecht5_tts/tests/test_autoregressive_layer_by_layer_pcc.py "Hello world"

    # Multi-step with custom steps
    python models/experimental/speecht5_tts/tests/test_autoregressive_layer_by_layer_pcc.py "Hello world" -m 30

    # True autoregressive (each feeds its own output)
    python models/experimental/speecht5_tts/tests/test_autoregressive_layer_by_layer_pcc.py "Hello world" -a 20

    # Show per-layer details at each step
    python models/experimental/speecht5_tts/tests/test_autoregressive_layer_by_layer_pcc.py "Hello world" -m 20 --verbose
"""

import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import pearsonr
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import ttnn
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset

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


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    if tensor1 is None or tensor2 is None:
        return 0.0

    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().float().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().float().numpy()

    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    if flat1.size == 0 or flat2.size == 0:
        return 1.0 if flat1.size == flat2.size == 0 else 0.0

    if np.std(flat1) < 1e-8 or np.std(flat2) < 1e-8:
        return 1.0 if np.allclose(flat1, flat2) else 0.0

    try:
        pcc = pearsonr(flat1, flat2)[0]
        return pcc if not np.isnan(pcc) else 0.0
    except:
        return 0.0


@dataclass
class LayerPCCHistory:
    """Track PCC history for each layer across steps."""

    encoder_final: List[float] = field(default_factory=list)
    decoder_prenet: List[float] = field(default_factory=list)
    decoder_layers: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(6)})
    decoder_final: List[float] = field(default_factory=list)
    postnet_convs: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(5)})
    postnet_residual: List[float] = field(default_factory=list)
    mel_before: List[float] = field(default_factory=list)
    mel_after: List[float] = field(default_factory=list)


def run_layer_by_layer_autoregressive(
    text: str, num_steps: int = 20, true_autoregressive: bool = False, verbose: bool = False
):
    """Run autoregressive generation with layer-by-layer PCC tracking."""

    mode = "TRUE AUTOREGRESSIVE" if true_autoregressive else "MULTI-STEP"
    print(f"LAYER-BY-LAYER PCC TRACKING ({mode})")
    print("=" * 80)
    print(f"Text: '{text}'")
    print(f"Steps: {num_steps}")

    # Load models
    print("\n1. Loading models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    hf_model.eval()

    # Disable dropout
    hf_model.speecht5.decoder.prenet.encode_positions.dropout.p = 0.0
    hf_model.speecht5.decoder.prenet.config.speech_decoder_prenet_dropout = 0.0
    original_dropout = hf_model.speecht5.decoder.prenet._consistent_dropout
    hf_model.speecht5.decoder.prenet._consistent_dropout = lambda x, p: x if p == 0.0 else original_dropout(x, p)

    # Speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Initialize TTNN
    print("\n2. Initializing TTNN...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    device.enable_program_cache()

    # Create configs
    encoder_config = TTNNEncoderConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.encoder_layers,
        num_heads=hf_model.config.encoder_attention_heads,
        ffn_dim=hf_model.config.encoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
    )

    decoder_config = TTNNDecoderConfig(
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.decoder_layers,
        num_heads=hf_model.config.decoder_attention_heads,
        ffn_dim=hf_model.config.decoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
        num_mel_bins=hf_model.config.num_mel_bins,
        reduction_factor=hf_model.config.reduction_factor,
        speech_decoder_prenet_units=hf_model.config.speech_decoder_prenet_units,
        speech_decoder_prenet_layers=hf_model.config.speech_decoder_prenet_layers,
        speech_decoder_prenet_dropout=0.0,
        speaker_embedding_dim=hf_model.config.speaker_embedding_dim,
    )

    postnet_config = TTNNPostNetConfig(
        hidden_size=hf_model.config.hidden_size,
        num_mel_bins=hf_model.config.num_mel_bins,
        reduction_factor=hf_model.config.reduction_factor,
        postnet_layers=hf_model.config.speech_decoder_postnet_layers,
        postnet_units=hf_model.config.speech_decoder_postnet_units,
        postnet_kernel=hf_model.config.speech_decoder_postnet_kernel,
    )

    # Create TTNN models
    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
    ttnn_encoder = TTNNSpeechT5Encoder(device, encoder_params, encoder_config)

    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder, decoder_config, device, speaker_embeddings
    )
    ttnn_decoder = TTNNSpeechT5Decoder(device, decoder_params, decoder_config, max_sequence_length=num_steps + 10)

    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device, postnet_params, postnet_config)

    # Process input
    inputs = processor(text=text, return_tensors="pt")

    # Run encoder once
    print("\n3. Running encoder...")
    with torch.no_grad():
        hf_encoder_output = hf_model.speecht5.encoder(inputs["input_ids"])[0]

    ttnn_input_ids = ttnn.from_torch(
        inputs["input_ids"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn_encoder_output = ttnn_encoder(ttnn_input_ids)
    if isinstance(ttnn_encoder_output, tuple):
        ttnn_encoder_output = ttnn_encoder_output[0]

    encoder_pcc = compute_pcc(hf_encoder_output, ttnn.to_torch(ttnn_encoder_output))
    print(f"   Encoder PCC: {encoder_pcc:.6f}")

    # Initialize sequences
    batch_size = 1
    num_mel_bins = 80
    reduction_factor = 2

    hf_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)
    ttnn_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)

    # PCC history
    history = LayerPCCHistory()

    # Autoregressive loop
    print(f"\n4. Running {num_steps} autoregressive steps...")
    print("-" * 100)

    if verbose:
        print(
            f"{'Step':>4} | {'Decoder':>8} | {'MelBefore':>9} | {'Conv0':>8} | {'Conv1':>8} | {'Conv2':>8} | {'Conv3':>8} | {'Conv4':>8} | {'MelAfter':>8}"
        )
        print("-" * 110)
    else:
        print(f"{'Step':>4} | {'Decoder':>10} | {'Postnet':>10} | {'Mel After':>10} | {'Min Layer PCC':>15}")
        print("-" * 80)

    for step in range(num_steps):
        # Get input for this step
        if true_autoregressive:
            hf_input = hf_mel_seq
            ttnn_input = ttnn_mel_seq
        else:
            # Both use HF output as ground truth
            hf_input = hf_mel_seq
            ttnn_input = hf_mel_seq

        # Convert TTNN input (use bfloat16 to match decoder)
        ttnn_mel_input = ttnn.from_torch(
            ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn_speaker_emb = ttnn.from_torch(
            speaker_embeddings,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        with torch.no_grad():
            # ========== HF Decoder ==========
            hf_decoder_output = hf_model.speecht5.decoder(
                input_values=hf_input,
                encoder_hidden_states=hf_encoder_output,
                speaker_embeddings=speaker_embeddings,
            )
            hf_hidden = hf_decoder_output[0]

            # ========== TTNN Decoder ==========
            ttnn_hidden = ttnn_decoder(
                decoder_input_values=ttnn_mel_input,
                encoder_hidden_states=ttnn_encoder_output,
                speaker_embeddings=ttnn_speaker_emb,
            )
            if isinstance(ttnn_hidden, tuple):
                ttnn_hidden = ttnn_hidden[0]

            # Decoder PCC
            decoder_pcc = compute_pcc(hf_hidden, ttnn.to_torch(ttnn_hidden))
            history.decoder_final.append(decoder_pcc)

            # ========== HF Postnet (layer by layer) ==========
            hf_feat_out = hf_model.speech_decoder_postnet.feat_out(hf_hidden)
            hf_mel_before = hf_feat_out.view(batch_size, -1, num_mel_bins)
            hf_residual = hf_mel_before
            hf_conv_input = hf_mel_before.transpose(1, 2)  # [B, 80, L]

            # ========== TTNN Postnet ==========
            ttnn_mel_before, ttnn_mel_after, ttnn_stop = ttnn_postnet(ttnn_hidden)
            ttnn_mel_before_torch = ttnn.to_torch(ttnn_mel_before)
            ttnn_mel_after_torch = ttnn.to_torch(ttnn_mel_after)

            # Mel before PCC
            mel_before_pcc = compute_pcc(hf_mel_before, ttnn_mel_before_torch)
            history.mel_before.append(mel_before_pcc)

            # Track conv layer PCCs using SAME input (hf_mel_before) for accurate conv measurement
            # This isolates conv layer accuracy from decoder/feat_out differences
            shared_input = hf_mel_before.transpose(1, 2)  # [B, 80, L]
            ttnn_conv_input = ttnn.from_torch(
                shared_input,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            conv_pccs = []
            for conv_idx in range(5):
                hf_layer = hf_model.speech_decoder_postnet.layers[conv_idx]
                ttnn_layer = ttnn_postnet.layers[conv_idx]

                # HF conv
                hf_conv_out = hf_layer.conv(hf_conv_input)
                hf_bn_out = hf_layer.batch_norm(hf_conv_out)
                if conv_idx < 4:
                    hf_layer_out = torch.tanh(hf_bn_out)
                else:
                    hf_layer_out = hf_bn_out

                # TTNN conv - using same input as HF for accurate conv measurement
                ttnn_layer_out = ttnn_layer(ttnn_conv_input)
                ttnn_layer_out_torch = ttnn.to_torch(ttnn_layer_out)

                # PCC - measures conv layer accuracy (same input)
                conv_pcc = compute_pcc(hf_layer_out, ttnn_layer_out_torch)
                history.postnet_convs[conv_idx].append(conv_pcc)
                conv_pccs.append(conv_pcc)

                # Update for next layer - both cascade from their own outputs
                hf_conv_input = hf_layer_out
                ttnn_conv_input = ttnn_layer_out

            # Residual addition
            hf_conv_output = hf_conv_input.transpose(1, 2)
            hf_mel_after = hf_residual + hf_conv_output

            # Residual PCC (mel_after)
            mel_after_pcc = compute_pcc(hf_mel_after, ttnn_mel_after_torch)
            history.mel_after.append(mel_after_pcc)
            history.postnet_residual.append(mel_after_pcc)

            # Print progress
            min_conv_pcc = min(conv_pccs)
            if verbose:
                print(
                    f"{step:>4} | {decoder_pcc:>8.4f} | {mel_before_pcc:>9.4f} | {conv_pccs[0]:>8.4f} | {conv_pccs[1]:>8.4f} | {conv_pccs[2]:>8.4f} | {conv_pccs[3]:>8.4f} | {conv_pccs[4]:>8.4f} | {mel_after_pcc:>8.4f}"
                )
            else:
                if step % 5 == 0 or step == num_steps - 1:
                    print(
                        f"{step:>4} | {decoder_pcc:>10.4f} | {min_conv_pcc:>10.4f} | {mel_after_pcc:>10.4f} | {min(decoder_pcc, min_conv_pcc, mel_after_pcc):>15.4f}"
                    )

            # Update mel sequences
            seq_len = hf_mel_seq.shape[1]
            start_idx = (seq_len - 1) * reduction_factor
            end_idx = start_idx + reduction_factor

            if end_idx <= hf_mel_after.shape[1]:
                hf_last_frame = hf_mel_after[:, end_idx - 1 : end_idx, :]
            else:
                hf_last_frame = hf_mel_after[:, -1:, :]

            if end_idx <= ttnn_mel_after_torch.shape[1]:
                ttnn_last_frame = ttnn_mel_after_torch[:, end_idx - 1 : end_idx, :]
            else:
                ttnn_last_frame = ttnn_mel_after_torch[:, -1:, :]

            hf_mel_seq = torch.cat([hf_mel_seq, hf_last_frame], dim=1)
            if true_autoregressive:
                ttnn_mel_seq = torch.cat([ttnn_mel_seq, ttnn_last_frame], dim=1)
            else:
                ttnn_mel_seq = hf_mel_seq.clone()

            # Cleanup
            ttnn.deallocate(ttnn_mel_input)
            ttnn.deallocate(ttnn_speaker_emb)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nEncoder PCC: {encoder_pcc:.6f}")

    print(f"\nDecoder (across {num_steps} steps):")
    print(f"  Mean: {np.mean(history.decoder_final):.6f}")
    print(f"  Min:  {np.min(history.decoder_final):.6f} (step {np.argmin(history.decoder_final)})")
    print(f"  Max:  {np.max(history.decoder_final):.6f}")

    print(f"\nPostnet Conv Layers (across {num_steps} steps):")
    for conv_idx in range(5):
        pccs = history.postnet_convs[conv_idx]
        print(f"  Conv{conv_idx}: Mean={np.mean(pccs):.4f}, Min={np.min(pccs):.4f}, Max={np.max(pccs):.4f}")

    print(f"\nMel After (with residual):")
    print(f"  Mean: {np.mean(history.mel_after):.6f}")
    print(f"  Min:  {np.min(history.mel_after):.6f} (step {np.argmin(history.mel_after)})")
    print(f"  Max:  {np.max(history.mel_after):.6f}")

    # Find worst performing layer
    all_min_pccs = {
        "decoder": np.min(history.decoder_final),
        "conv0": np.min(history.postnet_convs[0]),
        "conv1": np.min(history.postnet_convs[1]),
        "conv2": np.min(history.postnet_convs[2]),
        "conv3": np.min(history.postnet_convs[3]),
        "conv4": np.min(history.postnet_convs[4]),
        "mel_after": np.min(history.mel_after),
    }
    worst_layer = min(all_min_pccs, key=all_min_pccs.get)
    print(f"\nWorst performing component: {worst_layer} (min PCC: {all_min_pccs[worst_layer]:.4f})")

    ttnn.close_device(device)

    return history


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer-by-Layer PCC Tracking During Autoregressive Generation")
    parser.add_argument("text", help="Input text to synthesize")
    parser.add_argument("-m", "--multi-step", type=int, default=0, help="Run multi-step with N steps")
    parser.add_argument("-a", "--true-autoregressive", type=int, default=0, help="Run true autoregressive with N steps")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-layer details at each step")

    args = parser.parse_args()

    if args.true_autoregressive > 0:
        history = run_layer_by_layer_autoregressive(
            args.text, num_steps=args.true_autoregressive, true_autoregressive=True, verbose=args.verbose
        )
    elif args.multi_step > 0:
        history = run_layer_by_layer_autoregressive(
            args.text, num_steps=args.multi_step, true_autoregressive=False, verbose=args.verbose
        )
    else:
        # Default: 20 steps multi-step
        history = run_layer_by_layer_autoregressive(
            args.text, num_steps=20, true_autoregressive=False, verbose=args.verbose
        )

    print("\nTest completed!")


if __name__ == "__main__":
    main()
