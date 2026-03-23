# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test PCC differences after each autoregressive layer between PyTorch and TTNN

Long texts are automatically split into chunks at sentence/word boundaries
(same logic as demo_ttnn.py) so the encoder never receives an overlong token
sequence.  Each chunk is processed independently and per-chunk metrics are
aggregated in the final summary.

Setup:
    cd /home/ttuser/ssinghal/PR-fix/speecht5_tts_final/new/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    # Default: Multi-step with 20 steps (short text)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world"

    # Long text: automatically chunked at sentence boundaries
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Very long text..."

    # Multi-step with custom steps (PyTorch output as ground truth for both)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" --multi-step 50

    # True autoregressive (each feeds its own output back - shows divergence)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" --true-autoregressive 20

    # Custom chunk size
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Long text..." --max-chunk-size 200

    # With dropout enabled (not recommended for PCC comparison)
    python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py "Hello world" -m 20 --with-dropout
"""

import sys
import re
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


DEFAULT_CHUNK_SIZE = 150


def chunk_text(text, max_chunk_size=DEFAULT_CHUNK_SIZE, processor=None):
    """Split text into chunks that always end at sentence boundaries.

    Sentences are packed greedily into chunks until adding the next sentence
    would exceed max_chunk_size characters. A single sentence that exceeds
    max_chunk_size is kept as one chunk (never split mid-sentence), since
    TTS quality degrades badly on sentence fragments.

    The only exception: if a sentence exceeds MAX_ENCODER_TOKENS tokens (hard
    device limit), it is split at the last clause boundary (,;) within that
    token budget to avoid L1 OOM.
    """
    MAX_ENCODER_TOKENS = 250

    if len(text) <= max_chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]

    def split_oversized(sentence):
        if processor is None:
            return [sentence]
        n_tokens = processor(text=sentence, return_tensors="pt")["input_ids"].shape[1]
        if n_tokens <= MAX_ENCODER_TOKENS:
            return [sentence]
        clauses = re.split(r"(?<=[,;])\s+", sentence)
        parts = []
        current = ""
        for clause in clauses:
            candidate = (current + " " + clause).strip() if current else clause
            n = processor(text=candidate, return_tensors="pt")["input_ids"].shape[1]
            if n <= MAX_ENCODER_TOKENS:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = clause
        if current:
            parts.append(current)
        return parts if parts else [sentence]

    flat_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            flat_sentences.extend(split_oversized(s))

    chunks = []
    current = ""
    for sentence in flat_sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chunk_size:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


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


def _run_multi_step_pcc_chunk(
    chunk_text_str,
    chunk_idx,
    num_chunks,
    num_steps,
    processor,
    pytorch_encoder,
    pytorch_decoder,
    pytorch_postnet,
    hf_model,
    speaker_embeddings,
    device,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    ttnn_speaker_embeddings,
):
    """Run multi-step PCC test for a single text chunk using pre-initialized models."""

    print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}: '{chunk_text_str}' ---")

    inputs = processor(text=chunk_text_str, return_tensors="pt")
    batch_size = inputs["input_ids"].shape[0]

    ttnn_input_ids = ttnn.from_torch(
        inputs["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"   Running encoder...")
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
    ttnn.deallocate(ttnn_input_ids)
    print(f"   Encoder PCC: {encoder_pcc:.6f}")

    print(f"   Running {num_steps} autoregressive steps...")
    print(f"   {'Step':<6} {'Dec PCC':<10} {'Mel Pre':<10} {'Mel Post':<10} {'Stop PCC':<10}")
    print("   " + "-" * 50)

    num_mel_bins = 80
    reduction_factor = 2

    pytorch_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)
    pcc_history = {"decoder": [], "mel_pre": [], "mel_post": [], "stop": []}

    for step in range(num_steps):
        ttnn_mel_seq = ttnn.from_torch(
            pytorch_mel_seq,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        with torch.no_grad():
            pytorch_decoder_output = pytorch_decoder(
                decoder_input_values=pytorch_mel_seq,
                encoder_hidden_states=pytorch_encoder_output,
                speaker_embeddings=speaker_embeddings,
            )

            ttnn_decoder_output = ttnn_decoder(
                decoder_input_values=ttnn_mel_seq,
                encoder_hidden_states=ttnn_encoder_output,
                speaker_embeddings=ttnn_speaker_embeddings,
            )
            if isinstance(ttnn_decoder_output, tuple):
                ttnn_decoder_output = ttnn_decoder_output[0]

            pytorch_dec_np = pytorch_decoder_output.cpu().numpy()
            ttnn_dec_np = ttnn.to_torch(ttnn_decoder_output).float().cpu().numpy()
            dec_pcc = compute_pcc(pytorch_dec_np, ttnn_dec_np)

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
                print(f"   {step:<6} {dec_pcc:<10.4f} {mel_pre_pcc:<10.4f} {mel_post_pcc:<10.4f} {stop_pcc:<10.4f}")

            seq_len = pytorch_mel_seq.shape[1]
            start_idx = (seq_len - 1) * reduction_factor
            end_idx = start_idx + reduction_factor
            if end_idx <= pytorch_mel_post.shape[1]:
                last_frame = pytorch_mel_post[:, end_idx - 1 : end_idx, :]
            else:
                last_frame = pytorch_mel_post[:, -1:, :]
            pytorch_mel_seq = torch.cat([pytorch_mel_seq, last_frame], dim=1)

            ttnn.deallocate(ttnn_mel_seq)

    ttnn.deallocate(ttnn_encoder_output)
    return encoder_pcc, pcc_history


def run_multi_step_pcc_test(text, num_steps=20, disable_dropout=True, max_chunk_size=DEFAULT_CHUNK_SIZE):
    """Run parallel PyTorch and TTNN inference for multiple autoregressive steps.

    Supports long texts by automatically splitting into chunks at sentence/word
    boundaries (mirrors the chunking behaviour of demo_ttnn.py).
    """

    chunks = chunk_text(text, max_chunk_size)
    num_chunks = len(chunks)

    print(f"MULTI-STEP PCC ANALYSIS ({num_steps} steps)")
    print("=" * 80)
    if num_chunks > 1:
        print(f"Text split into {num_chunks} chunks (max_chunk_size={max_chunk_size})")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i + 1}: '{c[:80]}{'...' if len(c) > 80 else ''}'")
    else:
        print(f"Input: '{text}'")
    if disable_dropout:
        print("WARNING: Prenet dropout DISABLED for deterministic comparison")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    print("\n1. Loading models...")
    pytorch_encoder = load_encoder_ref()
    pytorch_decoder = load_decoder_ref()
    pytorch_postnet = load_postnet_ref()
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    if disable_dropout:
        pytorch_decoder.prenet._consistent_dropout = lambda inputs, p: inputs
        pytorch_decoder.prenet.config.speech_decoder_prenet_dropout = 0.0
        pytorch_decoder.prenet.encode_positions.dropout.p = 0.0
        print("   Dropout disabled for PyTorch prenet")

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

    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"\n3. Processing {num_chunks} chunk(s)...")

    all_encoder_pccs = []
    all_pcc_history = {"decoder": [], "mel_pre": [], "mel_post": [], "stop": []}

    for chunk_idx, chunk in enumerate(chunks):
        enc_pcc, pcc_history = _run_multi_step_pcc_chunk(
            chunk_text_str=chunk,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
            num_steps=num_steps,
            processor=processor,
            pytorch_encoder=pytorch_encoder,
            pytorch_decoder=pytorch_decoder,
            pytorch_postnet=pytorch_postnet,
            hf_model=hf_model,
            speaker_embeddings=speaker_embeddings,
            device=device,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            ttnn_speaker_embeddings=ttnn_speaker_embeddings,
        )
        all_encoder_pccs.append(enc_pcc)
        for key in all_pcc_history:
            all_pcc_history[key].extend(pcc_history[key])

    print(f"\n4. Summary (all {num_chunks} chunk(s), {len(all_pcc_history['decoder'])} total steps):")
    print(f"   Encoder PCC: min={min(all_encoder_pccs):.6f}, avg={np.mean(all_encoder_pccs):.6f}")
    print(
        f"   Decoder PCC: min={min(all_pcc_history['decoder']):.4f}, avg={np.mean(all_pcc_history['decoder']):.4f}, final={all_pcc_history['decoder'][-1]:.4f}"
    )
    print(f"   Mel Pre PCC: min={min(all_pcc_history['mel_pre']):.4f}, avg={np.mean(all_pcc_history['mel_pre']):.4f}")
    print(
        f"   Mel Post PCC: min={min(all_pcc_history['mel_post']):.4f}, avg={np.mean(all_pcc_history['mel_post']):.4f}"
    )
    print(f"   Stop PCC: min={min(all_pcc_history['stop']):.4f}, avg={np.mean(all_pcc_history['stop']):.4f}")

    ttnn.close_device(device)
    return all_pcc_history


def _run_true_autoregressive_chunk(
    chunk_text_str,
    chunk_idx,
    num_chunks,
    num_steps,
    processor,
    pytorch_encoder,
    pytorch_decoder,
    pytorch_postnet,
    hf_model,
    speaker_embeddings,
    device,
    ttnn_encoder,
    ttnn_decoder,
    ttnn_postnet,
    ttnn_speaker_embeddings,
):
    """Run true autoregressive test for a single text chunk using pre-initialized models."""

    print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}: '{chunk_text_str}' ---")

    inputs = processor(text=chunk_text_str, return_tensors="pt")
    batch_size = inputs["input_ids"].shape[0]

    ttnn_input_ids = ttnn.from_torch(
        inputs["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"   Running encoder...")
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
    ttnn.deallocate(ttnn_input_ids)
    print(f"   Encoder PCC: {encoder_pcc:.6f}")

    print(f"   Running {num_steps} autoregressive steps (INDEPENDENT loops)...")
    print(f"   {'Step':<6} {'Mel Seq PCC':<12} {'Stop PyT':<10} {'Stop TTNN':<10} {'Divergence':<10}")
    print("   " + "-" * 60)

    num_mel_bins = 80
    reduction_factor = 2

    pytorch_mel_seq = torch.zeros(batch_size, 1, num_mel_bins)
    ttnn_mel_seq_torch = torch.zeros(batch_size, 1, num_mel_bins)

    pcc_history = []
    pytorch_stopped = False
    ttnn_stopped = False
    pytorch_stop_prob = 0.0
    ttnn_stop_prob = 0.0

    for step in range(num_steps):
        if pytorch_stopped and ttnn_stopped:
            break

        with torch.no_grad():
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

            mel_seq_pcc = compute_pcc(pytorch_mel_seq.numpy(), ttnn_mel_seq_torch.numpy()) if step > 0 else 1.0
            pcc_history.append(mel_seq_pcc)

            divergence = torch.norm(pytorch_mel_seq - ttnn_mel_seq_torch).item() if step > 0 else 0.0

            if step % 5 == 0 or step == num_steps - 1:
                print(
                    f"   {step:<6} {mel_seq_pcc:<12.4f} {pytorch_stop_prob:<10.4f} {ttnn_stop_prob:<10.4f} {divergence:<10.4f}"
                )

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

    ttnn.deallocate(ttnn_encoder_output)

    min_len = min(pytorch_mel_seq.shape[1], ttnn_mel_seq_torch.shape[1])
    final_divergence = torch.norm(pytorch_mel_seq[:, :min_len, :] - ttnn_mel_seq_torch[:, :min_len, :]).item()
    return encoder_pcc, pcc_history, pytorch_stopped, ttnn_stopped, step, final_divergence


def run_true_autoregressive_test(text, num_steps=20, disable_dropout=True, max_chunk_size=DEFAULT_CHUNK_SIZE):
    """Test where TTNN feeds its own output back - true autoregressive mode.

    Supports long texts by automatically splitting into chunks at sentence/word
    boundaries (mirrors the chunking behaviour of demo_ttnn.py).
    """

    chunks = chunk_text(text, max_chunk_size)
    num_chunks = len(chunks)

    print(f"TRUE AUTOREGRESSIVE TEST ({num_steps} steps)")
    print("=" * 80)
    print("Both TTNN and PyTorch will feed their own outputs back (independent loops)")
    if num_chunks > 1:
        print(f"Text split into {num_chunks} chunks (max_chunk_size={max_chunk_size})")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i + 1}: '{c[:80]}{'...' if len(c) > 80 else ''}'")
    else:
        print(f"Input: '{text}'")
    if disable_dropout:
        print("WARNING: Prenet dropout DISABLED for deterministic comparison")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    print("\n1. Loading models...")
    pytorch_encoder = load_encoder_ref()
    pytorch_decoder = load_decoder_ref()
    pytorch_postnet = load_postnet_ref()
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    if disable_dropout:
        pytorch_decoder.prenet._consistent_dropout = lambda inputs, p: inputs
        pytorch_decoder.prenet.config.speech_decoder_prenet_dropout = 0.0
        pytorch_decoder.prenet.encode_positions.dropout.p = 0.0
        print("   Dropout disabled for PyTorch prenet")

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

    ttnn_speaker_embeddings = ttnn.from_torch(
        speaker_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    print(f"\n3. Processing {num_chunks} chunk(s)...")

    all_encoder_pccs = []
    all_pcc_history = []

    for chunk_idx, chunk in enumerate(chunks):
        (
            enc_pcc,
            pcc_history,
            pytorch_stopped,
            ttnn_stopped,
            last_step,
            final_divergence,
        ) = _run_true_autoregressive_chunk(
            chunk_text_str=chunk,
            chunk_idx=chunk_idx,
            num_chunks=num_chunks,
            num_steps=num_steps,
            processor=processor,
            pytorch_encoder=pytorch_encoder,
            pytorch_decoder=pytorch_decoder,
            pytorch_postnet=pytorch_postnet,
            hf_model=hf_model,
            speaker_embeddings=speaker_embeddings,
            device=device,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            ttnn_speaker_embeddings=ttnn_speaker_embeddings,
        )
        all_encoder_pccs.append(enc_pcc)
        all_pcc_history.extend(pcc_history)
        print(f"   PyTorch stopped: {'yes at step ' + str(last_step) if pytorch_stopped else 'no'}")
        print(f"   TTNN stopped: {'yes at step ' + str(last_step) if ttnn_stopped else 'no'}")
        print(f"   Final divergence (L2): {final_divergence:.4f}")

    print(f"\n4. Summary (all {num_chunks} chunk(s), {len(all_pcc_history)} total steps):")
    print(f"   Encoder PCC: min={min(all_encoder_pccs):.6f}, avg={np.mean(all_encoder_pccs):.6f}")
    print(
        f"   Mel Sequence PCC: min={min(all_pcc_history):.4f}, avg={np.mean(all_pcc_history):.4f}, final={all_pcc_history[-1]:.4f}"
    )

    ttnn.close_device(device)
    return all_pcc_history


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
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per text chunk for long-text support (default: {DEFAULT_CHUNK_SIZE})",
    )

    args = parser.parse_args()
    disable_dropout = not args.with_dropout

    try:
        if args.true_autoregressive > 0:
            pcc_history = run_true_autoregressive_test(
                args.text,
                num_steps=args.true_autoregressive,
                disable_dropout=disable_dropout,
                max_chunk_size=args.max_chunk_size,
            )
        elif args.multi_step > 0:
            pcc_history = run_multi_step_pcc_test(
                args.text,
                num_steps=args.multi_step,
                disable_dropout=disable_dropout,
                max_chunk_size=args.max_chunk_size,
            )
        else:
            # Default: run multi-step with 20 steps
            pcc_history = run_multi_step_pcc_test(
                args.text, num_steps=20, disable_dropout=disable_dropout, max_chunk_size=args.max_chunk_size
            )

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
