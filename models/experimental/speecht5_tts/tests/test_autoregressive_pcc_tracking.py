# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test PCC differences after each autoregressive layer between PyTorch and TTNN

Setup:
    cd /home/ttuser/ssinghal/PR-fix/speecht5_tts_final/new/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    # Default: Multi-step with 20 steps
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world"

    # Multi-step with custom steps (PyTorch output as ground truth for both)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" --multi-step 50

    # True autoregressive (each feeds its own output back - shows divergence)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" --true-autoregressive 20

    # With dropout enabled (not recommended for PCC comparison)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" -m 20 --with-dropout
"""

import sys
from pathlib import Path
import torch
import ttnn
import numpy as np
from scipy.stats import pearsonr
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset

# Add tt-metal root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

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
from models.experimental.speecht5_tts.reference import (
    load_encoder_from_huggingface as load_encoder_ref,
    load_decoder_from_huggingface as load_decoder_ref,
    load_postnet_from_huggingface as load_postnet_ref,
)


def compute_pcc(tensor1, tensor2):
    """Compute PCC between two tensors."""
    if tensor1 is None or tensor2 is None:
        return 0.0

    # Convert to numpy for PCC calculation
    if hasattr(tensor1, "detach"):
        tensor1 = tensor1.detach().cpu().numpy()
    if hasattr(tensor2, "detach"):
        tensor2 = tensor2.detach().cpu().numpy()

    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    if tensor1_flat.size == 0 or tensor2_flat.size == 0:
        return 1.0 if tensor1_flat.size == tensor2_flat.size == 0 else 0.0

    try:
        pcc = pearsonr(tensor1_flat, tensor2_flat)[0]
        return pcc if not np.isnan(pcc) else 0.0
    except:
        return 0.0


def run_multi_step_pcc_test(text, num_steps=20, disable_dropout=True):
    """Run parallel PyTorch and TTNN inference for multiple autoregressive steps."""

    print(f"🔬 MULTI-STEP PCC ANALYSIS: '{text}' ({num_steps} steps)")
    print("=" * 80)
    if disable_dropout:
        print("⚠️  Prenet dropout DISABLED for deterministic comparison")

    # Load processor and models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    # Load PyTorch reference models
    print("\n1. Loading models...")
    pytorch_encoder = load_encoder_ref()
    pytorch_decoder = load_decoder_ref()
    pytorch_postnet = load_postnet_ref()

    # Load HF model for TTNN
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    # Process input
    inputs = processor(text=text, return_tensors="pt")
    batch_size = inputs["input_ids"].shape[0]

    # Load speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Disable dropout for PyTorch if requested
    if disable_dropout:
        pytorch_decoder.prenet._consistent_dropout = lambda inputs, p: inputs
        pytorch_decoder.prenet.config.speech_decoder_prenet_dropout = 0.0
        pytorch_decoder.prenet.encode_positions.dropout.p = 0.0
        print("   Dropout disabled for PyTorch prenet")

    # Initialize TTNN
    print("\n2. Initializing TTNN...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    # Create TTNN models
    encoder_config = TTNNEncoderConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.encoder_layers,
        num_heads=hf_model.config.encoder_attention_heads,
        ffn_dim=hf_model.config.encoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
    )

    # Set dropout to 0.0 for TTNN if disable_dropout is True
    prenet_dropout = 0.0 if disable_dropout else hf_model.config.speech_decoder_prenet_dropout

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
        speech_decoder_prenet_dropout=prenet_dropout,
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

    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
    ttnn_encoder = TTNNSpeechT5Encoder(device=device, parameters=encoder_params, config=encoder_config)

    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder, decoder_config, device, speaker_embeddings
    )
    ttnn_decoder = TTNNSpeechT5Decoder(
        device=device, parameters=decoder_params, config=decoder_config, max_sequence_length=num_steps + 10
    )

    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device=device, parameters=postnet_params, config=postnet_config)

    # Prepare inputs
    ttnn_input_ids = ttnn.from_torch(
        inputs["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Encoder
    print("\n3. Running encoder...")
    with torch.no_grad():
        pytorch_encoder_output = pytorch_encoder(inputs["input_ids"])
        if isinstance(pytorch_encoder_output, tuple):
            pytorch_encoder_output = pytorch_encoder_output[0]
    ttnn_encoder_output = ttnn_encoder(ttnn_input_ids)
    if isinstance(ttnn_encoder_output, tuple):
        ttnn_encoder_output = ttnn_encoder_output[0]
    encoder_pcc = compute_pcc(
        pytorch_encoder_output.cpu().numpy(), ttnn.to_torch(ttnn_encoder_output).float().cpu().numpy()
    )
    print(f"   Encoder PCC: {encoder_pcc:.6f}")

    # Multi-step autoregressive loop
    print(f"\n4. Running {num_steps} autoregressive steps...")
    print(f"{'Step':<6} {'Dec PCC':<10} {'Mel Pre':<10} {'Mel Post':<10} {'Stop PCC':<10}")
    print("-" * 50)

    num_mel_bins = 80
    reduction_factor = 2

    # Initialize mel sequences - using PyTorch output as ground truth input
    pytorch_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)

    pcc_history = {"decoder": [], "mel_pre": [], "mel_post": [], "stop": []}

    for step in range(num_steps):
        # Convert current mel sequence to TTNN
        ttnn_mel_seq = ttnn.from_torch(
            pytorch_mel_seq,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        with torch.no_grad():
            # PyTorch decoder
            pytorch_decoder_output = pytorch_decoder(
                decoder_input_values=pytorch_mel_seq,
                encoder_hidden_states=pytorch_encoder_output,
                speaker_embeddings=speaker_embeddings,
            )

            # TTNN decoder
            ttnn_decoder_output = ttnn_decoder(
                decoder_input_values=ttnn_mel_seq,
                encoder_hidden_states=ttnn_encoder_output,
                speaker_embeddings=ttnn_speaker_embeddings,
            )
            if isinstance(ttnn_decoder_output, tuple):
                ttnn_decoder_output = ttnn_decoder_output[0]

            # Compare decoder outputs
            pytorch_dec_np = pytorch_decoder_output.cpu().numpy()
            ttnn_dec_np = ttnn.to_torch(ttnn_decoder_output).float().cpu().numpy()
            dec_pcc = compute_pcc(pytorch_dec_np, ttnn_dec_np)

            # Postnet
            pytorch_mel_pre, pytorch_mel_post, pytorch_stop = pytorch_postnet(pytorch_decoder_output)
            ttnn_mel_pre_t, ttnn_mel_post_t, ttnn_stop_t = ttnn_postnet(ttnn_decoder_output)

            ttnn_mel_pre = ttnn.to_torch(ttnn_mel_pre_t).float()
            ttnn_mel_post = ttnn.to_torch(ttnn_mel_post_t).float()
            ttnn_stop = ttnn.to_torch(ttnn_stop_t).float()

            mel_pre_pcc = compute_pcc(pytorch_mel_pre.cpu().numpy(), ttnn_mel_pre.cpu().numpy())
            mel_post_pcc = compute_pcc(pytorch_mel_post.cpu().numpy(), ttnn_mel_post.cpu().numpy())
            stop_pcc = compute_pcc(pytorch_stop.cpu().numpy(), ttnn_stop.cpu().numpy())

            pcc_history["decoder"].append(dec_pcc)
            pcc_history["mel_pre"].append(mel_pre_pcc)
            pcc_history["mel_post"].append(mel_post_pcc)
            pcc_history["stop"].append(stop_pcc)

            if step % 5 == 0 or step == num_steps - 1:
                print(f"{step:<6} {dec_pcc:<10.4f} {mel_pre_pcc:<10.4f} {mel_post_pcc:<10.4f} {stop_pcc:<10.4f}")

            # Update mel sequence with PyTorch output (ground truth)
            seq_len = pytorch_mel_seq.shape[1]
            start_idx = (seq_len - 1) * reduction_factor
            end_idx = start_idx + reduction_factor
            if end_idx <= pytorch_mel_post.shape[1]:
                last_frame = pytorch_mel_post[:, end_idx - 1 : end_idx, :]
            else:
                last_frame = pytorch_mel_post[:, -1:, :]
            pytorch_mel_seq = torch.cat([pytorch_mel_seq, last_frame], dim=1)

            # Cleanup
            ttnn.deallocate(ttnn_mel_seq)

    # Summary
    print(f"\n5. Summary:")
    print(f"   Encoder PCC: {encoder_pcc:.6f}")
    print(
        f"   Decoder PCC: min={min(pcc_history['decoder']):.4f}, avg={np.mean(pcc_history['decoder']):.4f}, final={pcc_history['decoder'][-1]:.4f}"
    )
    print(f"   Mel Pre PCC: min={min(pcc_history['mel_pre']):.4f}, avg={np.mean(pcc_history['mel_pre']):.4f}")
    print(f"   Mel Post PCC: min={min(pcc_history['mel_post']):.4f}, avg={np.mean(pcc_history['mel_post']):.4f}")
    print(f"   Stop PCC: min={min(pcc_history['stop']):.4f}, avg={np.mean(pcc_history['stop']):.4f}")

    ttnn.close_device(device)
    return pcc_history


def run_true_autoregressive_test(text, num_steps=20, disable_dropout=True):
    """Test where TTNN feeds its own output back - true autoregressive mode."""

    print(f"🔬 TRUE AUTOREGRESSIVE TEST: '{text}' ({num_steps} steps)")
    print("=" * 80)
    print("Both TTNN and PyTorch will feed their own outputs back (independent loops)")
    if disable_dropout:
        print("⚠️  Prenet dropout DISABLED for deterministic comparison")

    # Load processor and models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    # Load models
    print("\n1. Loading models...")
    pytorch_encoder = load_encoder_ref()
    pytorch_decoder = load_decoder_ref()
    pytorch_postnet = load_postnet_ref()
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    inputs = processor(text=text, return_tensors="pt")
    batch_size = inputs["input_ids"].shape[0]

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Disable dropout for PyTorch if requested
    if disable_dropout:
        pytorch_decoder.prenet._consistent_dropout = lambda inputs, p: inputs
        pytorch_decoder.prenet.config.speech_decoder_prenet_dropout = 0.0
        pytorch_decoder.prenet.encode_positions.dropout.p = 0.0
        print("   Dropout disabled for PyTorch prenet")

    # Initialize TTNN
    print("\n2. Initializing TTNN...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    encoder_config = TTNNEncoderConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        num_layers=hf_model.config.encoder_layers,
        num_heads=hf_model.config.encoder_attention_heads,
        ffn_dim=hf_model.config.encoder_ffn_dim,
        max_position_embeddings=hf_model.config.max_length,
        layer_norm_eps=hf_model.config.layer_norm_eps,
    )

    # Set dropout to 0.0 for TTNN if disable_dropout is True
    prenet_dropout = 0.0 if disable_dropout else hf_model.config.speech_decoder_prenet_dropout

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
        speech_decoder_prenet_dropout=prenet_dropout,
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

    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
    ttnn_encoder = TTNNSpeechT5Encoder(device=device, parameters=encoder_params, config=encoder_config)
    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder, decoder_config, device, speaker_embeddings
    )
    ttnn_decoder = TTNNSpeechT5Decoder(
        device=device, parameters=decoder_params, config=decoder_config, max_sequence_length=num_steps + 10
    )
    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device=device, parameters=postnet_params, config=postnet_config)

    ttnn_input_ids = ttnn.from_torch(
        inputs["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Encoder
    print("\n3. Running encoder...")
    with torch.no_grad():
        pytorch_encoder_output = pytorch_encoder(inputs["input_ids"])
        if isinstance(pytorch_encoder_output, tuple):
            pytorch_encoder_output = pytorch_encoder_output[0]
    ttnn_encoder_output = ttnn_encoder(ttnn_input_ids)
    if isinstance(ttnn_encoder_output, tuple):
        ttnn_encoder_output = ttnn_encoder_output[0]
    encoder_pcc = compute_pcc(
        pytorch_encoder_output.cpu().numpy(), ttnn.to_torch(ttnn_encoder_output).float().cpu().numpy()
    )
    print(f"   Encoder PCC: {encoder_pcc:.6f}")

    # Independent autoregressive loops
    print(f"\n4. Running {num_steps} autoregressive steps (INDEPENDENT loops)...")
    print(f"{'Step':<6} {'Mel Seq PCC':<12} {'Stop PyT':<10} {'Stop TTNN':<10} {'Divergence':<10}")
    print("-" * 60)

    num_mel_bins = 80
    reduction_factor = 2

    # SEPARATE mel sequences for each - they will diverge
    pytorch_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)
    ttnn_mel_seq_torch = torch.zeros(batch_size, 1, num_mel_bins)

    pcc_history = []
    pytorch_stopped = False
    ttnn_stopped = False

    for step in range(num_steps):
        if pytorch_stopped and ttnn_stopped:
            break

        with torch.no_grad():
            # PyTorch (feeds its own output)
            if not pytorch_stopped:
                pytorch_decoder_output = pytorch_decoder(
                    decoder_input_values=pytorch_mel_seq,
                    encoder_hidden_states=pytorch_encoder_output,
                    speaker_embeddings=speaker_embeddings,
                )
                pytorch_mel_pre, pytorch_mel_post, pytorch_stop = pytorch_postnet(pytorch_decoder_output)
                pytorch_stop_prob = torch.sigmoid(pytorch_stop).mean().item()
                if pytorch_stop_prob > 0.5:
                    pytorch_stopped = True

            # TTNN (feeds its own output)
            if not ttnn_stopped:
                ttnn_mel_input = ttnn.from_torch(
                    ttnn_mel_seq_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn_decoder_output = ttnn_decoder(
                    decoder_input_values=ttnn_mel_input,
                    encoder_hidden_states=ttnn_encoder_output,
                    speaker_embeddings=ttnn_speaker_embeddings,
                )
                if isinstance(ttnn_decoder_output, tuple):
                    ttnn_decoder_output = ttnn_decoder_output[0]
                ttnn_mel_pre_t, ttnn_mel_post_t, ttnn_stop_t = ttnn_postnet(ttnn_decoder_output)
                ttnn_mel_post = ttnn.to_torch(ttnn_mel_post_t).float()
                ttnn_stop = ttnn.to_torch(ttnn_stop_t).float()
                ttnn_stop_prob = torch.sigmoid(ttnn_stop).mean().item()
                if ttnn_stop_prob > 0.5:
                    ttnn_stopped = True
                ttnn.deallocate(ttnn_mel_input)

            # Compare mel sequences
            mel_seq_pcc = compute_pcc(pytorch_mel_seq.numpy(), ttnn_mel_seq_torch.numpy()) if step > 0 else 1.0
            pcc_history.append(mel_seq_pcc)

            # Compute divergence (L2 norm of difference)
            if step > 0:
                divergence = torch.norm(pytorch_mel_seq - ttnn_mel_seq_torch).item()
            else:
                divergence = 0.0

            if step % 5 == 0 or step == num_steps - 1:
                print(
                    f"{step:<6} {mel_seq_pcc:<12.4f} {pytorch_stop_prob:<10.4f} {ttnn_stop_prob:<10.4f} {divergence:<10.4f}"
                )

            # Update mel sequences with their OWN outputs
            seq_len = pytorch_mel_seq.shape[1]
            start_idx = (seq_len - 1) * reduction_factor
            end_idx = start_idx + reduction_factor

            if not pytorch_stopped:
                if end_idx <= pytorch_mel_post.shape[1]:
                    pytorch_last_frame = pytorch_mel_post[:, end_idx - 1 : end_idx, :]
                else:
                    pytorch_last_frame = pytorch_mel_post[:, -1:, :]
                pytorch_mel_seq = torch.cat([pytorch_mel_seq, pytorch_last_frame], dim=1)

            if not ttnn_stopped:
                if end_idx <= ttnn_mel_post.shape[1]:
                    ttnn_last_frame = ttnn_mel_post[:, end_idx - 1 : end_idx, :]
                else:
                    ttnn_last_frame = ttnn_mel_post[:, -1:, :]
                ttnn_mel_seq_torch = torch.cat([ttnn_mel_seq_torch, ttnn_last_frame], dim=1)

    # Summary
    print(f"\n5. Summary:")
    print(f"   Encoder PCC: {encoder_pcc:.6f}")
    print(f"   PyTorch stopped at step: {'not stopped' if not pytorch_stopped else step}")
    print(f"   TTNN stopped at step: {'not stopped' if not ttnn_stopped else step}")
    print(
        f"   Mel Sequence PCC: min={min(pcc_history):.4f}, avg={np.mean(pcc_history):.4f}, final={pcc_history[-1]:.4f}"
    )
    min_len = min(pytorch_mel_seq.shape[1], ttnn_mel_seq_torch.shape[1])
    print(
        f"   Final divergence (L2): {torch.norm(pytorch_mel_seq[:, :min_len, :] - ttnn_mel_seq_torch[:, :min_len, :]).item():.4f}"
    )

    ttnn.close_device(device)
    return pcc_history


def main():
    """Main function with command line arguments"""

    import argparse

    parser = argparse.ArgumentParser(description="TTNN SpeechT5 Autoregressive PCC Tracking Test")
    parser.add_argument("text", help="Input text to analyze PCC degradation for")
    parser.add_argument(
        "--multi-step",
        "-m",
        type=int,
        default=0,
        help="Run multi-step analysis with N steps (default: 0 = single step only)",
    )
    parser.add_argument(
        "--true-autoregressive",
        "-a",
        type=int,
        default=0,
        help="Run true autoregressive test (TTNN feeds its own output) with N steps",
    )
    parser.add_argument(
        "--with-dropout",
        action="store_true",
        help="Enable prenet dropout (default: disabled for deterministic comparison)",
    )

    args = parser.parse_args()
    disable_dropout = not args.with_dropout

    try:
        if args.true_autoregressive > 0:
            pcc_history = run_true_autoregressive_test(
                args.text, num_steps=args.true_autoregressive, disable_dropout=disable_dropout
            )
        elif args.multi_step > 0:
            pcc_history = run_multi_step_pcc_test(args.text, num_steps=args.multi_step, disable_dropout=disable_dropout)
        else:
            # Default: run multi-step with 20 steps
            pcc_history = run_multi_step_pcc_test(args.text, num_steps=20, disable_dropout=disable_dropout)

        print("\n✅ PCC tracking completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during PCC tracking: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
