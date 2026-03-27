#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 Voice Conversion (VC) bring-up on TTNN.

Stage-1 baseline pipeline:
1. Speech encoder prenet (HF reference, CPU)
2. Shared SpeechT5 encoder (TTNN)
3. Shared SpeechT5 decoder with speaker conditioning (TTNN)
4. Speech decoder postnet (TTNN)
5. HiFi-GAN vocoder (HF reference, CPU)

Notes:
- Input audio must be mono 16 kHz.
- This script is intentionally conservative for correctness bring-up.
- Advanced trace capture/sharding optimizations are expected in later stages.
"""

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import soundfile as sf
import torch
import ttnn
from datasets import load_dataset
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, SpeechT5Processor

from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNDecoderConfig,
    TTNNSpeechT5Decoder,
    init_kv_cache,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNEncoderConfig,
    TTNNSpeechT5Encoder,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNPostNetConfig,
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)


def load_audio_16khz_mono(path: str) -> torch.Tensor:
    """Load input audio as mono float32 tensor at 16 kHz."""
    waveform, sample_rate = sf.read(path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        raise ValueError(
            f"Input sample rate is {sample_rate} Hz. Please resample to 16000 Hz before running voice conversion."
        )

    return torch.tensor(waveform, dtype=torch.float32)


def load_speaker_embedding(
    speaker_embedding_path: Optional[str],
    speaker_index: int = 7306,
) -> torch.Tensor:
    """Load target speaker x-vector embedding [1, 512]."""
    if speaker_embedding_path:
        if speaker_embedding_path.endswith(".pt"):
            speaker_embeddings = torch.load(speaker_embedding_path, map_location="cpu")
        elif speaker_embedding_path.endswith(".npy"):
            import numpy as np

            speaker_embeddings = torch.tensor(np.load(speaker_embedding_path))
        else:
            raise ValueError("Unsupported speaker embedding file. Use .pt or .npy")

        if speaker_embeddings.ndim == 1:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        if speaker_embeddings.shape[-1] != 512:
            raise ValueError(
                f"Speaker embedding must have last dimension 512, got {speaker_embeddings.shape}."
            )
        return speaker_embeddings.float()

    # Default fallback: CMU Arctic x-vectors (same source used in the TTS demo).
    try:
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
    except NotImplementedError:
        import glob
        import numpy as np
        import pyarrow as pa

        cache_root = os.path.expanduser("~/.cache/huggingface/datasets/Matthijs___cmu-arctic-xvectors")
        arrow_files = glob.glob(os.path.join(cache_root, "**", "*validation*.arrow"), recursive=True)
        if not arrow_files:
            raise RuntimeError(
                "Could not find cached cmu-arctic-xvectors arrow file. "
                "Delete ~/.cache/huggingface/datasets/Matthijs___cmu-arctic-xvectors and re-run."
            )
        with pa.memory_map(arrow_files[0], "r") as src:
            table = pa.ipc.open_stream(src).read_all()
        speaker_embeddings = torch.tensor(np.array(table["xvector"][speaker_index].as_py())).unsqueeze(0)

    return speaker_embeddings.float()


def build_encoder_masks(
    reduced_attention_mask: Optional[torch.Tensor],
    device,
) -> Tuple[Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
    """
    Build encoder masks for:
    - encoder self-attention: [batch, 1, seq]
    - decoder cross-attention: [batch, 1, 1, seq]

    Mask convention: 0 for valid positions, -1e9 for padded positions.
    """
    if reduced_attention_mask is None:
        return None, None

    mask = (1.0 - reduced_attention_mask.float()) * -1e9

    encoder_self_mask = mask.unsqueeze(1)
    encoder_self_mask_ttnn = ttnn.from_torch(
        encoder_self_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    decoder_cross_mask = mask.unsqueeze(1).unsqueeze(1)
    decoder_cross_mask_ttnn = ttnn.from_torch(
        decoder_cross_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    return encoder_self_mask_ttnn, decoder_cross_mask_ttnn


def generate_mel_with_ttnn(
    input_values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    hf_model: SpeechT5ForSpeechToSpeech,
    ttnn_encoder: TTNNSpeechT5Encoder,
    ttnn_decoder: TTNNSpeechT5Decoder,
    ttnn_postnet: TTNNSpeechT5SpeechDecoderPostnet,
    decoder_config: TTNNDecoderConfig,
    device,
    max_steps: int,
    stop_threshold: float,
    min_steps: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Generate mel spectrogram autoregressively using TTNN shared encoder/decoder/postnet."""
    generation_start = time.time()

    with torch.no_grad():
        prenet_hidden_states, reduced_attention_mask = hf_model.speecht5.encoder.prenet(
            input_values=input_values,
            attention_mask=attention_mask,
        )

    encoder_self_mask_ttnn, decoder_cross_mask_ttnn = build_encoder_masks(reduced_attention_mask, device)

    ttnn_prenet_hidden = ttnn.from_torch(
        prenet_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        # Keep large encoder inputs in DRAM to reduce L1 pressure on long utterances.
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    encoder_start = time.time()
    encoder_output = ttnn_encoder.forward_from_hidden_states(
        ttnn_prenet_hidden,
        attention_mask=encoder_self_mask_ttnn,
    )[0]
    encoder_time = time.time() - encoder_start

    # Decoder expects [batch, 1, enc_seq, hidden]
    encoder_output = ttnn.unsqueeze(encoder_output, dim=1)

    batch_size = input_values.shape[0]
    encoder_seq_len = prenet_hidden_states.shape[1]

    kv_cache, cross_attn_cache = init_kv_cache(
        decoder_config,
        device,
        max_batch_size=batch_size,
        max_seq_len=max_steps + 10,
        encoder_seq_len=encoder_seq_len,
    )

    decoder_input = torch.zeros(batch_size, 1, decoder_config.num_mel_bins, dtype=torch.float32)
    all_mel_outputs = []

    total_decoder_time = 0.0
    total_postnet_time = 0.0
    ttft = None

    for step in range(max_steps):
        decoder_step_start = time.time()
        ttnn_decoder_input = ttnn.from_torch(
            decoder_input,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        current_decode_pos = ttnn.full(
            (batch_size, 1),
            step,
            dtype=ttnn.int32,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        decoder_hidden_states = ttnn_decoder(
            decoder_input_values=ttnn_decoder_input,
            encoder_hidden_states=encoder_output,
            speaker_embeddings=None,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=(step > 0),
            current_decode_pos=current_decode_pos,
            position_offset=step,
            encoder_attention_mask=decoder_cross_mask_ttnn,
        )
        total_decoder_time += time.time() - decoder_step_start

        postnet_start = time.time()
        _, outputs_after, stop_logits = ttnn_postnet(decoder_hidden_states)
        total_postnet_time += time.time() - postnet_start

        if step == 0:
            ttft = time.time() - generation_start

        mel_output_ttnn = ttnn.squeeze(outputs_after, 0)  # [reduction_factor, mel_bins]
        mel_output_torch = ttnn.to_torch(mel_output_ttnn)
        all_mel_outputs.append(mel_output_torch)

        if step >= min_steps:
            # Stop check on CPU avoids extra TT kernels in the hot loop and improves throughput.
            stop_probs = torch.sigmoid(ttnn.to_torch(stop_logits))
            if torch.any(torch.sum(stop_probs, dim=-1) >= stop_threshold).item():
                break

        # Feed last generated frame back into the decoder.
        # Keep this on CPU to avoid additional TT tensor ops every decode step.
        decoder_input = mel_output_torch[-1:, :].unsqueeze(0).to(torch.float32)

    mel_spectrogram = torch.cat(all_mel_outputs, dim=0).unsqueeze(0)

    total_time = time.time() - generation_start
    steps_completed = len(all_mel_outputs)
    avg_token_time = (total_decoder_time + total_postnet_time) / max(steps_completed, 1)

    stats = {
        "steps_completed": float(steps_completed),
        "mel_frames": float(mel_spectrogram.shape[1]),
        "ttft_s": float(ttft if ttft is not None else 0.0),
        "token_per_sec": float(1.0 / avg_token_time) if avg_token_time > 0 else 0.0,
        "encoder_time_s": float(encoder_time),
        "decoder_time_s": float(total_decoder_time),
        "postnet_time_s": float(total_postnet_time),
        "total_time_s": float(total_time),
    }
    return mel_spectrogram, stats


def convert_voice(
    input_audio_path: str,
    output_audio_path: str,
    speaker_embedding_path: Optional[str],
    device_id: int,
    max_steps: int,
    stop_threshold: float,
    min_steps: int,
    perf_report_path: Optional[str],
    l1_small_size: int,
    trace_region_size: int,
    num_command_queues: int,
):
    """Run end-to-end SpeechT5 voice conversion."""
    print("Loading HuggingFace SpeechT5-VC models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    hf_model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    hf_model.eval()

    print("Loading input audio and speaker embedding...")
    waveform = load_audio_16khz_mono(input_audio_path)
    speaker_embeddings = load_speaker_embedding(speaker_embedding_path)

    inputs = processor(audio=waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"]
    attention_mask = inputs.get("attention_mask", None)

    print("Initializing TTNN device...")
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
    )
    device.enable_program_cache()

    try:
        cfg = hf_model.config

        encoder_config = TTNNEncoderConfig(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.encoder_layers,
            num_heads=cfg.encoder_attention_heads,
            ffn_dim=cfg.encoder_ffn_dim,
            layer_norm_eps=cfg.layer_norm_eps,
            max_position_embeddings=getattr(cfg, "max_speech_positions", 600),
            max_relative_distance=getattr(cfg, "encoder_max_relative_position", 160),
        )

        decoder_config = TTNNDecoderConfig(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.decoder_layers,
            num_heads=cfg.decoder_attention_heads,
            ffn_dim=cfg.decoder_ffn_dim,
            max_position_embeddings=getattr(cfg, "max_speech_positions", 4000),
            layer_norm_eps=cfg.layer_norm_eps,
            num_mel_bins=cfg.num_mel_bins,
            reduction_factor=cfg.reduction_factor,
            speech_decoder_prenet_units=cfg.speech_decoder_prenet_units,
            speech_decoder_prenet_layers=cfg.speech_decoder_prenet_layers,
            speech_decoder_prenet_dropout=cfg.speech_decoder_prenet_dropout,
            speaker_embedding_dim=cfg.speaker_embedding_dim,
            use_fp32=True,
        )

        postnet_config = TTNNPostNetConfig(
            num_mel_bins=cfg.num_mel_bins,
            reduction_factor=cfg.reduction_factor,
            postnet_layers=cfg.speech_decoder_postnet_layers,
            postnet_units=cfg.speech_decoder_postnet_units,
            postnet_kernel=cfg.speech_decoder_postnet_kernel,
            postnet_dropout=cfg.speech_decoder_postnet_dropout,
        )

        print("Preprocessing TTNN parameters...")
        encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
        decoder_params = preprocess_decoder_parameters(
            hf_model.speecht5.decoder,
            decoder_config,
            device,
            speaker_embeddings,
        )
        postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)

        print("Creating TTNN modules...")
        ttnn_encoder = TTNNSpeechT5Encoder(device, encoder_params, encoder_config)
        ttnn_decoder = TTNNSpeechT5Decoder(device, decoder_params, decoder_config, max_sequence_length=max_steps + 16)
        ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device, postnet_params, postnet_config)

        print("Running TTNN voice conversion pipeline...")
        mel_spectrogram, stats = generate_mel_with_ttnn(
            input_values=input_values,
            attention_mask=attention_mask,
            hf_model=hf_model,
            ttnn_encoder=ttnn_encoder,
            ttnn_decoder=ttnn_decoder,
            ttnn_postnet=ttnn_postnet,
            decoder_config=decoder_config,
            device=device,
            max_steps=max_steps,
            stop_threshold=stop_threshold,
            min_steps=min_steps,
        )

        print("Running HiFi-GAN vocoder...")
        vocoder_start = time.time()
        with torch.no_grad():
            speech = vocoder(mel_spectrogram)
        stats["vocoder_time_s"] = float(time.time() - vocoder_start)
        stats["output_audio_seconds"] = float(speech.numel() / 16000.0)
        stats["rtf"] = (
            float(stats["total_time_s"] + stats["vocoder_time_s"]) / stats["output_audio_seconds"]
            if stats["output_audio_seconds"] > 0
            else 0.0
        )

        sf.write(output_audio_path, speech.squeeze().cpu().numpy(), samplerate=16000)

        print("\n=== Voice Conversion Summary ===")
        print(f"Output file: {output_audio_path}")
        print(f"Steps completed: {int(stats['steps_completed'])}")
        print(f"Token/s: {stats['token_per_sec']:.2f}")
        print(f"TTFT: {stats['ttft_s'] * 1000.0:.1f} ms")
        print(f"RTF: {stats['rtf']:.3f}")

        if perf_report_path:
            with open(perf_report_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"Perf report written to: {perf_report_path}")

    finally:
        ttnn.close_device(device)


def parse_args():
    parser = argparse.ArgumentParser(description="SpeechT5 Voice Conversion on TTNN")
    parser.add_argument("--input_wav", required=True, help="Path to input/source WAV (mono 16 kHz)")
    parser.add_argument("--output", default="converted_speech.wav", help="Output WAV path")
    parser.add_argument(
        "--speaker_embedding",
        default=None,
        help="Path to target speaker x-vector (.npy or .pt). If omitted, CMU Arctic default is used.",
    )
    parser.add_argument("--device_id", type=int, default=0, help="TT device id")
    parser.add_argument(
        "--l1_small_size",
        type=int,
        default=24576,
        help="L1 small allocation bytes (lower default helps avoid L1 clashes on multi-chip runs)",
    )
    parser.add_argument(
        "--trace_region_size",
        type=int,
        default=15000000,
        help="Trace region size in bytes",
    )
    parser.add_argument(
        "--num_command_queues",
        type=int,
        default=2,
        help="Number of command queues",
    )
    parser.add_argument("--max_steps", type=int, default=800, help="Maximum decoder steps")
    parser.add_argument("--stop_threshold", type=float, default=0.5, help="Stop token sigmoid threshold")
    parser.add_argument("--min_steps", type=int, default=10, help="Minimum generation steps before stop check")
    parser.add_argument(
        "--perf_report",
        default=None,
        help="Optional JSON path for perf stats",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_voice(
        input_audio_path=args.input_wav,
        output_audio_path=args.output,
        speaker_embedding_path=args.speaker_embedding,
        device_id=args.device_id,
        max_steps=args.max_steps,
        stop_threshold=args.stop_threshold,
        min_steps=args.min_steps,
        perf_report_path=args.perf_report,
        l1_small_size=args.l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=args.num_command_queues,
    )


if __name__ == "__main__":
    main()
