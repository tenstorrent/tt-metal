#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Cloud-ready Stage-1 validation script for SpeechT5-VC bring-up.

This script runs TTNN VC inference and generates a JSON report with:
- throughput (token/s)
- real-time factor (RTF)
- token-level accuracy proxy vs HF reference mel
- speaker similarity (cosine) against target speaker embedding
- content preservation WER

Notes:
- WER uses ASR transcripts. If no reference text is provided, it compares
  source-audio ASR transcript vs converted-audio ASR transcript.
- Token-level accuracy is reported as a mel-frame cosine-agreement proxy.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
import ttnn
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    SpeechT5ForSpeechToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    pipeline,
)

try:
    from transformers import AutoModelForAudioXVector
except ImportError:
    # Backward compatibility for older Transformers releases.
    from transformers import WavLMForXVector as AutoModelForAudioXVector

from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNDecoderConfig,
    TTNNSpeechT5Decoder,
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
from models.experimental.speecht5_vc.demo_ttnn import (
    generate_mel_with_ttnn,
    load_audio_16khz_mono,
    load_speaker_embedding,
)


@dataclass
class Thresholds:
    min_token_per_sec: float
    max_rtf: float
    min_speaker_cosine: float
    max_wer_percent: float
    min_token_accuracy_percent: float


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_hf_mel(output) -> torch.Tensor:
    if isinstance(output, tuple):
        mel = output[0]
    else:
        mel = output

    if mel.ndim == 3:
        mel = mel[0]
    return mel.float()


def _align_mels(tt_mel: torch.Tensor, ref_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if tt_mel.ndim == 3:
        tt_mel = tt_mel[0]
    if ref_mel.ndim == 3:
        ref_mel = ref_mel[0]

    min_len = min(tt_mel.shape[0], ref_mel.shape[0])
    return tt_mel[:min_len].float(), ref_mel[:min_len].float()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    if torch.std(a) == 0 or torch.std(b) == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def _token_accuracy_proxy(tt_mel: torch.Tensor, ref_mel: torch.Tensor, cosine_threshold: float) -> Dict[str, float]:
    tt_mel, ref_mel = _align_mels(tt_mel, ref_mel)

    if tt_mel.numel() == 0:
        return {
            "aligned_frames": 0.0,
            "avg_frame_cosine": 0.0,
            "token_accuracy_percent": 0.0,
            "mel_pcc": 0.0,
        }

    cos = F.cosine_similarity(tt_mel, ref_mel, dim=-1)
    token_acc = (cos >= cosine_threshold).float().mean() * 100.0

    return {
        "aligned_frames": float(tt_mel.shape[0]),
        "avg_frame_cosine": float(cos.mean().item()),
        "token_accuracy_percent": float(token_acc.item()),
        "mel_pcc": _pcc(tt_mel, ref_mel),
    }


def _build_asr_pipeline(asr_model_id: str):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    asr_device = 0 if use_cuda else -1

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(asr_model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=asr_device,
    )


def _transcribe(asr_pipe, wav_path: str, language: Optional[str], task: Optional[str]) -> str:
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task

    try:
        result = asr_pipe(wav_path, generate_kwargs=generate_kwargs if generate_kwargs else None)
    except TypeError:
        # Backward compatibility for older transformers pipeline signatures.
        result = asr_pipe(wav_path)

    text = result["text"] if isinstance(result, dict) else str(result)
    return _normalize_text(text)


def _build_speaker_embedder(model_id: str):
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioXVector.from_pretrained(model_id)
    model.eval()
    return extractor, model


def _extract_xvector(extractor, model, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    inputs = extractor(audio.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.embeddings.float()
    emb = F.normalize(emb, dim=-1)
    return emb[0]


def _avg(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _build_ttnn_modules(
    hf_model: SpeechT5ForSpeechToSpeech,
    speaker_embeddings: torch.Tensor,
    device,
    max_steps: int,
):
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

    encoder_params = preprocess_encoder_parameters(hf_model.speecht5.encoder, encoder_config, device)
    decoder_params = preprocess_decoder_parameters(
        hf_model.speecht5.decoder,
        decoder_config,
        device,
        speaker_embeddings,
    )
    postnet_params = preprocess_postnet_parameters(hf_model.speech_decoder_postnet, postnet_config, device)

    ttnn_encoder = TTNNSpeechT5Encoder(device, encoder_params, encoder_config)
    ttnn_decoder = TTNNSpeechT5Decoder(device, decoder_params, decoder_config, max_sequence_length=max_steps + 16)
    ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(device, postnet_params, postnet_config)

    return ttnn_encoder, ttnn_decoder, ttnn_postnet, decoder_config


def _open_ttnn_device(
    device_id: int,
    l1_small_size: int,
    trace_region_size: int,
    num_command_queues: int,
    enable_program_cache: bool,
    open_device_retries: int = 3,
    open_device_retry_sleep_s: float = 5.0,
):
    last_error: Optional[Exception] = None
    retries = max(1, int(open_device_retries))
    sleep_s = max(0.0, float(open_device_retry_sleep_s))

    for attempt in range(1, retries + 1):
        try:
            device = ttnn.open_device(
                device_id=device_id,
                l1_small_size=l1_small_size,
                trace_region_size=trace_region_size,
                num_command_queues=num_command_queues,
            )
            if enable_program_cache:
                device.enable_program_cache()
            return device
        except RuntimeError as err:
            last_error = err
            err_text = str(err)
            transient_open_failure = any(
                marker in err_text
                for marker in (
                    "failed to start",
                    "NoAccess error",
                    "ARC core",
                    "run_mailbox",
                )
            )

            can_retry = transient_open_failure and attempt < retries
            if not can_retry:
                raise

            print(
                f"Device open failed (attempt {attempt}/{retries}): {err_text}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to open TT device")


def parse_args():
    parser = argparse.ArgumentParser(description="SpeechT5-VC Stage-1 cloud validation")
    parser.add_argument("--input_wavs", nargs="+", required=True, help="Input source wav files (mono 16 kHz)")
    parser.add_argument(
        "--reference_texts",
        nargs="*",
        default=None,
        help="Optional reference transcript(s), one per input wav",
    )
    parser.add_argument("--speaker_embedding", default=None, help="Target speaker x-vector path (.npy/.pt)")
    parser.add_argument("--speaker_index", type=int, default=7306, help="Default CMU-Arctic x-vector index")
    parser.add_argument(
        "--target_speaker_wavs",
        nargs="*",
        default=None,
        help=(
            "Optional target speaker reference wav(s) for speaker similarity. "
            "Use one path (applies to all inputs) or one path per input wav."
        ),
    )
    parser.add_argument("--output_dir", default="./vc_stage1_outputs", help="Directory for converted wav outputs")
    parser.add_argument("--report_json", default="./vc_stage1_report.json", help="Output report JSON path")

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
    parser.add_argument(
        "--open_device_retries",
        type=int,
        default=3,
        help="Retry count for transient TT device open failures (ARC/no-access startup races)",
    )
    parser.add_argument(
        "--open_device_retry_sleep_s",
        type=float,
        default=5.0,
        help="Sleep duration between open-device retries in seconds",
    )
    parser.add_argument(
        "--disable_program_cache",
        action="store_true",
        help="Disable TT program cache (useful for debugging memory/program-cache issues)",
    )
    parser.add_argument(
        "--retry_on_l1_clash",
        action="store_true",
        help="If L1 clash occurs, reopen device with smaller l1_small_size and retry once",
    )
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--stop_threshold", type=float, default=0.5)
    parser.add_argument("--min_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--token_cosine_threshold", type=float, default=0.90)
    parser.add_argument(
        "--strict_stage1",
        action="store_true",
        help=(
            "Use strict Stage-1 pass gates (includes WER + speaker checks and end-to-end output-duration RTF). "
            "Default mode is bring-up focused and uses decoder-oriented perf gates."
        ),
    )

    parser.add_argument("--skip_wer", action="store_true")
    parser.add_argument("--skip_speaker_similarity", action="store_true")
    parser.add_argument("--asr_model_id", default="openai/whisper-small")
    parser.add_argument("--asr_language", default="en")
    parser.add_argument("--asr_task", default="transcribe")
    parser.add_argument("--speaker_model_id", default="microsoft/wavlm-base-plus-sv")

    parser.add_argument("--min_token_per_sec", type=float, default=30.0)
    parser.add_argument("--max_rtf", type=float, default=0.5)
    parser.add_argument("--min_speaker_cosine", type=float, default=0.70)
    parser.add_argument("--max_wer_percent", type=float, default=3.0)
    parser.add_argument("--min_token_accuracy_percent", type=float, default=95.0)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference_texts is not None and len(args.reference_texts) not in (0, len(args.input_wavs)):
        raise ValueError("--reference_texts must be omitted or have exactly one entry per input wav")
    if args.target_speaker_wavs is not None and len(args.target_speaker_wavs) not in (0, 1, len(args.input_wavs)):
        raise ValueError("--target_speaker_wavs must be omitted, a single path, or one path per input wav")

    thresholds = Thresholds(
        min_token_per_sec=args.min_token_per_sec,
        max_rtf=args.max_rtf,
        min_speaker_cosine=args.min_speaker_cosine,
        max_wer_percent=args.max_wer_percent,
        min_token_accuracy_percent=args.min_token_accuracy_percent,
    )

    _ensure_dir(args.output_dir)
    torch.manual_seed(args.seed)

    print("Loading models (HF + TTNN)...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    hf_model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    hf_model.eval()

    speaker_embeddings = load_speaker_embedding(args.speaker_embedding, speaker_index=args.speaker_index)
    target_spk_norm = F.normalize(speaker_embeddings.float(), dim=-1)[0]

    asr_pipe = None
    if not args.skip_wer:
        print(f"Loading ASR model for WER: {args.asr_model_id}")
        asr_pipe = _build_asr_pipeline(args.asr_model_id)

    speaker_extractor = None
    speaker_model = None
    if not args.skip_speaker_similarity:
        print(f"Loading speaker model: {args.speaker_model_id}")
        speaker_extractor, speaker_model = _build_speaker_embedder(args.speaker_model_id)

    target_reference_vectors: Optional[List[torch.Tensor]] = None
    speaker_metric_mode = "cross_space_xvector_vs_audio_encoder"
    if not args.skip_speaker_similarity and args.target_speaker_wavs:
        # Compute target speaker vectors in the same embedding space as output-audio vectors.
        target_reference_vectors = []
        for target_wav in args.target_speaker_wavs:
            target_audio = load_audio_16khz_mono(target_wav)
            target_vec = _extract_xvector(speaker_extractor, speaker_model, target_audio, sample_rate=16000)
            target_reference_vectors.append(target_vec)
        speaker_metric_mode = "same_space_audio_encoder"

    current_l1_small_size = int(args.l1_small_size)
    device = _open_ttnn_device(
        device_id=args.device_id,
        l1_small_size=current_l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=args.num_command_queues,
        enable_program_cache=not args.disable_program_cache,
        open_device_retries=args.open_device_retries,
        open_device_retry_sleep_s=args.open_device_retry_sleep_s,
    )

    try:
        ttnn_encoder, ttnn_decoder, ttnn_postnet, decoder_config = _build_ttnn_modules(
            hf_model=hf_model,
            speaker_embeddings=speaker_embeddings,
            device=device,
            max_steps=args.max_steps,
        )

        sample_reports: List[Dict] = []

        for idx, wav_path in enumerate(args.input_wavs):
            print(f"[{idx+1}/{len(args.input_wavs)}] Processing: {wav_path}")
            waveform = load_audio_16khz_mono(wav_path)
            source_audio_seconds = float(waveform.shape[0] / 16000.0)
            inputs = processor(audio=waveform.numpy(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"]
            attention_mask = inputs.get("attention_mask", None)

            # TTNN path
            run_error = None
            tt_mel = None
            tt_stats = None
            for attempt in range(2):
                try:
                    tt_mel, tt_stats = generate_mel_with_ttnn(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        hf_model=hf_model,
                        ttnn_encoder=ttnn_encoder,
                        ttnn_decoder=ttnn_decoder,
                        ttnn_postnet=ttnn_postnet,
                        decoder_config=decoder_config,
                        device=device,
                        max_steps=args.max_steps,
                        stop_threshold=args.stop_threshold,
                        min_steps=args.min_steps,
                    )
                    run_error = None
                    break
                except RuntimeError as err:
                    run_error = err
                    err_text = str(err)
                    l1_clash = "clash with L1 buffers" in err_text
                    can_retry = attempt == 0 and args.retry_on_l1_clash and l1_clash and current_l1_small_size > 8192
                    if not can_retry:
                        break

                    next_l1_small_size = max(8192, current_l1_small_size // 2)
                    if next_l1_small_size == current_l1_small_size:
                        break

                    print(
                        f"L1 clash detected. Retrying sample with smaller l1_small_size: "
                        f"{current_l1_small_size} -> {next_l1_small_size}"
                    )
                    ttnn.close_device(device)
                    current_l1_small_size = next_l1_small_size
                    device = _open_ttnn_device(
                        device_id=args.device_id,
                        l1_small_size=current_l1_small_size,
                        trace_region_size=args.trace_region_size,
                        num_command_queues=args.num_command_queues,
                        enable_program_cache=not args.disable_program_cache,
                        open_device_retries=args.open_device_retries,
                        open_device_retry_sleep_s=args.open_device_retry_sleep_s,
                    )
                    ttnn_encoder, ttnn_decoder, ttnn_postnet, decoder_config = _build_ttnn_modules(
                        hf_model=hf_model,
                        speaker_embeddings=speaker_embeddings,
                        device=device,
                        max_steps=args.max_steps,
                    )

            if run_error is not None:
                raise run_error

            vocoder_start = time.time()
            with torch.no_grad():
                tt_speech = vocoder(tt_mel)
            vocoder_time = time.time() - vocoder_start

            output_path = os.path.join(args.output_dir, f"converted_{idx:03d}.wav")
            sf.write(output_path, tt_speech.squeeze().cpu().numpy(), samplerate=16000)

            output_audio_seconds = float(tt_speech.numel() / 16000.0)
            rtf_end_to_end_output = (
                float(tt_stats["total_time_s"] + vocoder_time) / output_audio_seconds
                if output_audio_seconds > 0
                else 0.0
            )
            rtf_end_to_end_input = (
                float(tt_stats["total_time_s"] + vocoder_time) / source_audio_seconds
                if source_audio_seconds > 0
                else 0.0
            )
            rtf_decoder_input = (
                float(tt_stats["decoder_time_s"] + tt_stats["postnet_time_s"]) / source_audio_seconds
                if source_audio_seconds > 0
                else 0.0
            )

            decoder_token_per_sec = float(tt_stats.get("decoder_token_per_sec", tt_stats["token_per_sec"]))

            # HF reference mel path for parity metrics
            torch.manual_seed(args.seed)
            with torch.no_grad():
                hf_ref_out = hf_model.generate_speech(
                    input_values=input_values,
                    speaker_embeddings=speaker_embeddings,
                    attention_mask=attention_mask,
                    vocoder=None,
                )
            hf_ref_mel = _extract_hf_mel(hf_ref_out)

            parity = _token_accuracy_proxy(tt_mel, hf_ref_mel, cosine_threshold=args.token_cosine_threshold)

            # WER metric
            source_asr_text = None
            output_asr_text = None
            wer_ratio = None
            wer_percent = None
            if asr_pipe is not None:
                source_asr_text = _transcribe(asr_pipe, wav_path, args.asr_language, args.asr_task)
                output_asr_text = _transcribe(asr_pipe, output_path, args.asr_language, args.asr_task)

                reference_text = (
                    _normalize_text(args.reference_texts[idx])
                    if args.reference_texts and len(args.reference_texts) == len(args.input_wavs)
                    else source_asr_text
                )

                try:
                    import evaluate

                    wer_metric = evaluate.load("wer")
                    wer_ratio = float(wer_metric.compute(predictions=[output_asr_text], references=[reference_text]))
                    wer_percent = wer_ratio * 100.0
                except Exception:
                    wer_ratio = None
                    wer_percent = None

            # Speaker cosine metric
            speaker_cosine = None
            if speaker_extractor is not None and speaker_model is not None:
                out_audio = load_audio_16khz_mono(output_path)
                out_vec = _extract_xvector(speaker_extractor, speaker_model, out_audio, sample_rate=16000)
                if target_reference_vectors is not None:
                    ref_idx = idx if len(target_reference_vectors) == len(args.input_wavs) else 0
                    target_vec = target_reference_vectors[ref_idx]
                    speaker_cosine = float(F.cosine_similarity(out_vec, target_vec, dim=0).item())
                else:
                    # Legacy fallback: target x-vector and output-audio encoder spaces may differ.
                    speaker_cosine = float(F.cosine_similarity(out_vec, target_spk_norm, dim=0).item())

            sample_report = {
                "input_wav": wav_path,
                "output_wav": output_path,
                "source_audio_seconds": source_audio_seconds,
                "decoder_token_per_sec": decoder_token_per_sec,
                "token_per_sec": float(tt_stats["token_per_sec"]),
                "ttft_ms": float(tt_stats["ttft_s"] * 1000.0),
                "rtf": float(rtf_end_to_end_output),
                "rtf_end_to_end_output": float(rtf_end_to_end_output),
                "rtf_end_to_end_input": float(rtf_end_to_end_input),
                "rtf_decoder_input": float(rtf_decoder_input),
                "output_audio_seconds": output_audio_seconds,
                "steps_completed": int(tt_stats["steps_completed"]),
                "parity": parity,
                "speaker_cosine": speaker_cosine,
                "wer_ratio": wer_ratio,
                "wer_percent": wer_percent,
                "source_asr_text": source_asr_text,
                "output_asr_text": output_asr_text,
            }
            sample_reports.append(sample_report)

        avg_token_per_sec = _avg([s["token_per_sec"] for s in sample_reports]) or 0.0
        avg_decoder_token_per_sec = _avg([s["decoder_token_per_sec"] for s in sample_reports]) or 0.0
        avg_rtf_end_to_end_output = _avg([s["rtf_end_to_end_output"] for s in sample_reports]) or 0.0
        avg_rtf_end_to_end_input = _avg([s["rtf_end_to_end_input"] for s in sample_reports]) or 0.0
        avg_rtf_decoder_input = _avg([s["rtf_decoder_input"] for s in sample_reports]) or 0.0
        rtf_for_gate = avg_rtf_end_to_end_output if args.strict_stage1 else avg_rtf_decoder_input
        avg_token_accuracy = _avg([s["parity"]["token_accuracy_percent"] for s in sample_reports]) or 0.0
        avg_speaker_cosine = _avg([s["speaker_cosine"] for s in sample_reports if s["speaker_cosine"] is not None])
        avg_wer_percent = _avg([s["wer_percent"] for s in sample_reports if s["wer_percent"] is not None])

        checks = {
            "token_per_sec": {
                "value": avg_decoder_token_per_sec,
                "threshold": thresholds.min_token_per_sec,
                "pass": avg_decoder_token_per_sec >= thresholds.min_token_per_sec,
            },
            "rtf": {
                "value": rtf_for_gate,
                "threshold": thresholds.max_rtf,
                "pass": rtf_for_gate <= thresholds.max_rtf,
            },
            "token_accuracy_percent": {
                "value": avg_token_accuracy,
                "threshold": thresholds.min_token_accuracy_percent,
                "pass": avg_token_accuracy >= thresholds.min_token_accuracy_percent,
            },
            "speaker_cosine": {
                "value": avg_speaker_cosine,
                "threshold": thresholds.min_speaker_cosine,
                "pass": (avg_speaker_cosine is not None and avg_speaker_cosine >= thresholds.min_speaker_cosine)
                if not args.skip_speaker_similarity
                else None,
            },
            "wer_percent": {
                "value": avg_wer_percent,
                "threshold": thresholds.max_wer_percent,
                "pass": (avg_wer_percent is not None and avg_wer_percent <= thresholds.max_wer_percent)
                if not args.skip_wer
                else None,
            },
        }

        required_checks = [
            checks["token_per_sec"]["pass"],
            checks["rtf"]["pass"],
            checks["token_accuracy_percent"]["pass"],
        ]
        if args.strict_stage1 and not args.skip_speaker_similarity:
            required_checks.append(bool(checks["speaker_cosine"]["pass"]))
        if args.strict_stage1 and not args.skip_wer:
            required_checks.append(bool(checks["wer_percent"]["pass"]))

        overall_pass = all(required_checks)

        report = {
            "summary": {
                "num_samples": len(sample_reports),
                "overall_pass": overall_pass,
                "validation_mode": "strict_stage1" if args.strict_stage1 else "bringup",
                "avg_token_per_sec": avg_token_per_sec,
                "avg_decoder_token_per_sec": avg_decoder_token_per_sec,
                "avg_rtf": rtf_for_gate,
                "avg_rtf_end_to_end_output": avg_rtf_end_to_end_output,
                "avg_rtf_end_to_end_input": avg_rtf_end_to_end_input,
                "avg_rtf_decoder_input": avg_rtf_decoder_input,
                "avg_token_accuracy_percent": avg_token_accuracy,
                "avg_speaker_cosine": avg_speaker_cosine,
                "avg_wer_percent": avg_wer_percent,
            },
            "threshold_checks": checks,
            "samples": sample_reports,
            "config": {
                "input_wavs": args.input_wavs,
                "speaker_embedding": args.speaker_embedding,
                "speaker_index": args.speaker_index,
                "target_speaker_wavs": args.target_speaker_wavs,
                "speaker_metric_mode": speaker_metric_mode,
                "device_id": args.device_id,
                "l1_small_size": current_l1_small_size,
                "trace_region_size": args.trace_region_size,
                "num_command_queues": args.num_command_queues,
                "open_device_retries": args.open_device_retries,
                "open_device_retry_sleep_s": args.open_device_retry_sleep_s,
                "program_cache_enabled": not args.disable_program_cache,
                "strict_stage1": args.strict_stage1,
                "asr_model_id": None if args.skip_wer else args.asr_model_id,
                "asr_language": None if args.skip_wer else args.asr_language,
                "asr_task": None if args.skip_wer else args.asr_task,
                "speaker_model_id": None if args.skip_speaker_similarity else args.speaker_model_id,
                "token_cosine_threshold": args.token_cosine_threshold,
                "max_steps": args.max_steps,
                "stop_threshold": args.stop_threshold,
                "min_steps": args.min_steps,
                "seed": args.seed,
            },
        }

        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\n=== Stage-1 Validation Summary ===")
        print(f"Validation mode: {'strict_stage1' if args.strict_stage1 else 'bringup'}")
        print(f"Samples: {len(sample_reports)}")
        print(f"Avg token/s (decoder): {avg_decoder_token_per_sec:.2f}")
        print(f"Avg token/s (decoder+postnet): {avg_token_per_sec:.2f}")
        print(f"Avg RTF (gate): {rtf_for_gate:.3f}")
        print(f"Avg RTF (end-to-end/output-sec): {avg_rtf_end_to_end_output:.3f}")
        print(f"Avg RTF (end-to-end/input-sec): {avg_rtf_end_to_end_input:.3f}")
        print(f"Avg RTF (decoder/input-sec): {avg_rtf_decoder_input:.3f}")
        print(f"Avg token accuracy (%): {avg_token_accuracy:.2f}")
        if not args.skip_speaker_similarity:
            print(f"Avg speaker cosine: {avg_speaker_cosine if avg_speaker_cosine is not None else 'N/A'}")
            print(f"Speaker metric mode: {speaker_metric_mode}")
        if not args.skip_wer:
            print(f"Avg WER (%): {avg_wer_percent if avg_wer_percent is not None else 'N/A'}")
        print(f"Overall pass: {overall_pass}")
        print(f"Report: {args.report_json}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
