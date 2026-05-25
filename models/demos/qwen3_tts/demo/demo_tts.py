# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS text-to-speech demo.

Usage (GPU reference):
    python models/demos/qwen3_tts/demo/demo_tts.py \
        --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --text "こんにちは、世界" \
        --language japanese \
        --output output.wav \
        --backend reference

Usage (TT device — full pipeline):
    python models/demos/qwen3_tts/demo/demo_tts.py \
        --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --text "こんにちは、世界" \
        --language japanese \
        --output output.wav \
        --backend tt

Usage (TT device — voice cloning):
    python models/demos/qwen3_tts/demo/demo_tts.py \
        --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --text "こんにちは、世界" \
        --ref_audio reference.wav \
        --output output.wav \
        --backend tt
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch


def run_reference(args):
    """Run TTS using the HuggingFace reference model on GPU."""
    from models.demos.qwen3_tts.reference.functional import (
        ActivationCapture,
        generate_reference,
        load_reference_model,
    )

    print(f"Loading reference model from {args.model_path}...")
    model = load_reference_model(
        args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
    )

    capture = ActivationCapture() if args.dump_activations else None

    print(f"Generating speech for: '{args.text}' (language={args.language})")
    t0 = time.time()
    wavs, sr = generate_reference(
        model,
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        capture=capture,
    )
    elapsed = time.time() - t0

    wav = wavs[0].cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
    duration = len(wav) / sr
    rtf = elapsed / duration if duration > 0 else float("inf")

    sf.write(args.output, wav, sr)
    print(f"Saved {args.output} ({duration:.2f}s audio, {elapsed:.2f}s elapsed, RTF={rtf:.3f})")

    if capture and args.dump_activations:
        capture.save(args.activation_dir)
        print(f"Saved {len(capture.activations)} activation tensors to {args.activation_dir}")


def run_tt(args):
    """Run full TTS pipeline on TT device.

    Pipeline: Text → Talker (CB0) → Code Predictor (CB1-15) → Vocoder → WAV
    Optional: reference audio → Speaker Encoder → speaker embedding → Talker
    """
    import os

    import ttnn
    from models.demos.qwen3_tts.tt.generator import TTSGenerator

    os.environ["HF_MODEL"] = args.model_path

    device_ids = ttnn.get_device_ids()
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.ETH if len(device_ids) > 1 else ttnn.DispatchCoreType.WORKER
        ),
    )
    ttnn.enable_program_cache(mesh_device)

    print(f"Building TTS generator from {args.model_path}...")
    generator = TTSGenerator.build(args.model_path, mesh_device, max_seq_len=args.max_new_tokens + 512)

    ref_audio = None
    ref_sr = 24000
    if args.ref_audio:
        ref_audio_raw, ref_sr = sf.read(args.ref_audio, dtype="float32")
        if ref_audio_raw.ndim > 1:
            ref_audio_raw = ref_audio_raw.mean(axis=1)
        ref_audio = ref_audio_raw

    print(f"Generating speech for: '{args.text}' (language={args.language})")
    t0 = time.time()
    waveform, sr = generator.generate(
        text=args.text,
        language=args.language,
        ref_audio=ref_audio,
        ref_sr=ref_sr,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    elapsed = time.time() - t0

    duration = len(waveform) / sr
    rtf = elapsed / duration if duration > 0 else float("inf")

    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({duration:.2f}s audio, {elapsed:.2f}s elapsed, RTF={rtf:.3f})")

    ttnn.close_mesh_device(mesh_device)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS text-to-speech demo")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--text", type=str, default="こんにちは、世界。今日はいい天気ですね。")
    parser.add_argument("--language", type=str, default="japanese")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--backend", type=str, choices=["reference", "tt"], default="reference")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--ref_text", type=str, default=None, help="Transcript of reference audio")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dump_activations", action="store_true", help="Dump intermediate activations")
    parser.add_argument("--activation_dir", type=str, default="qwen3_tts_activations")

    args = parser.parse_args()

    if args.backend == "reference":
        run_reference(args)
    else:
        run_tt(args)


if __name__ == "__main__":
    main()
