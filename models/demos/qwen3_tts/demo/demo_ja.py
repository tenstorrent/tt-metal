# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS 日本語 text-to-speech demo.

Usage (GPU reference):
    python models/demos/qwen3_tts/demo/demo_ja.py \
        --text "こんにちは、世界" --backend reference --output output.wav

Usage (TT device — full pipeline):
    python models/demos/qwen3_tts/demo/demo_ja.py \
        --text "こんにちは、世界" --backend tt --output output.wav

Usage (TT device — voice cloning):
    python models/demos/qwen3_tts/demo/demo_ja.py \
        --text "こんにちは" --ref_audio reference.wav --backend tt

Usage (TT device — save speaker embedding for reuse):
    python models/demos/qwen3_tts/demo/demo_ja.py \
        --ref_audio reference.wav --save_speaker speaker.safetensors \
        --text "こんにちは" --backend tt

Usage (TT device — reuse saved speaker embedding):
    python models/demos/qwen3_tts/demo/demo_ja.py \
        --load_speaker speaker.safetensors --text "こんにちは" --backend tt

Interactive mode (continuous text input):
    python models/demos/qwen3_tts/demo/demo_ja.py --interactive --backend tt

Preset demo (batch generate from preset Japanese texts):
    python models/demos/qwen3_tts/demo/demo_ja.py --preset all --backend tt --output_dir demo_output/
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

PRESET_TEXTS = {
    "greeting": "こんにちは、世界。今日はいい天気ですね。",
    "news": "本日の東京の最高気温は三十二度で、各地で猛暑日となりました。",
    "numbers": "電話番号は〇三の一二三四の五六七八です。",
    "keigo": "お忙しいところ恐れ入りますが、少々お時間をいただけますでしょうか。",
    "long": "昔々、あるところにおじいさんとおばあさんが住んでいました。おじいさんは山へ芝刈りに、おばあさんは川へ洗濯に行きました。",
    "mixed": "人工知能は二〇二六年に大きな進歩を遂げ、音声合成の品質も飛躍的に向上しました。",
    "emotion": "やったー！ついに完成したよ！信じられない！",
}


def generate_one_reference(model, text, language, gen_config):
    """Generate one utterance with HF reference model."""
    from models.demos.qwen3_tts.reference.functional import generate_reference

    t0 = time.time()
    wavs, sr = generate_reference(
        model,
        text=text,
        language=language,
        max_new_tokens=gen_config.get("max_new_tokens", 2048),
        temperature=gen_config.get("temperature", 0.9),
        top_k=gen_config.get("top_k", 50),
        top_p=gen_config.get("top_p", 1.0),
        repetition_penalty=gen_config.get("repetition_penalty", 1.05),
    )
    elapsed = time.time() - t0
    wav = wavs[0].cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
    return wav, sr, elapsed


def generate_one_tt(generator, text, language, gen_config, ref_audio=None, ref_sr=24000, speaker_emb_tt=None):
    """Generate one utterance with TT pipeline."""
    t0 = time.time()
    waveform, sr = generator.generate(
        text=text,
        language=language,
        ref_audio=ref_audio,
        ref_sr=ref_sr,
        speaker_emb_tt=speaker_emb_tt,
        max_new_tokens=gen_config.get("max_new_tokens", 2048),
        temperature=gen_config.get("temperature", 0.9),
        top_k=gen_config.get("top_k", 50),
        top_p=gen_config.get("top_p", 1.0),
        repetition_penalty=gen_config.get("repetition_penalty", 1.05),
    )
    elapsed = time.time() - t0
    return waveform, sr, elapsed


def save_and_report(wav, sr, elapsed, output_path, text):
    """Save WAV and print stats."""
    sf.write(output_path, wav, sr)
    duration = len(wav) / sr
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"  -> {output_path} ({duration:.2f}s audio, {elapsed:.2f}s elapsed, RTF={rtf:.3f})")
    return {"text": text, "duration": duration, "elapsed": elapsed, "rtf": rtf, "path": output_path}


def build_reference_model(model_path, device):
    """Load HF reference model."""
    from models.demos.qwen3_tts.reference.functional import load_reference_model

    print(f"Loading reference model from {model_path}...")
    return load_reference_model(model_path, device=device, dtype=torch.bfloat16)


def build_tt_generator(model_path, max_new_tokens):
    """Build TT pipeline."""
    import ttnn
    from models.demos.qwen3_tts.tt.generator import TTSGenerator

    os.environ["HF_MODEL"] = model_path

    device_ids = ttnn.get_device_ids()
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.ETH if len(device_ids) > 1 else ttnn.DispatchCoreType.WORKER
        ),
    )
    try:
        mesh_device.enable_program_cache()
    except AttributeError:
        ttnn.enable_program_cache(mesh_device)

    print(f"Building TTS generator from {model_path}...")
    generator = TTSGenerator.build(model_path, mesh_device, max_seq_len=max_new_tokens + 512)
    return generator, mesh_device


def run_single(args):
    """Generate speech for a single text."""
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    ref_audio = None
    ref_sr = 24000
    speaker_emb_tt = None

    print(f"Text: {args.text}")
    print(f"Language: {args.language}, Backend: {args.backend}")

    if args.backend == "reference":
        if args.ref_audio:
            ref_audio_raw, ref_sr = sf.read(args.ref_audio, dtype="float32")
            if ref_audio_raw.ndim > 1:
                ref_audio_raw = ref_audio_raw.mean(axis=1)
            ref_audio = ref_audio_raw
        model = build_reference_model(args.model_path, args.device)
        wav, sr, elapsed = generate_one_reference(model, args.text, args.language, gen_config)
    else:
        generator, mesh_device = build_tt_generator(args.model_path, args.max_new_tokens)

        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        if args.load_speaker:
            speaker_emb_tt = SpeakerEncoder.load_embedding(args.load_speaker, mesh_device)
            print(f"Loaded speaker embedding from {args.load_speaker}")
        elif args.ref_audio:
            ref_audio_raw, ref_sr = sf.read(args.ref_audio, dtype="float32")
            if ref_audio_raw.ndim > 1:
                ref_audio_raw = ref_audio_raw.mean(axis=1)
            ref_audio = ref_audio_raw
            if args.save_speaker:
                generator.speaker_encoder.save_embedding(ref_audio, ref_sr, args.save_speaker)
                print(f"Saved speaker embedding to {args.save_speaker}")

        wav, sr, elapsed = generate_one_tt(
            generator, args.text, args.language, gen_config, ref_audio, ref_sr, speaker_emb_tt
        )

    save_and_report(wav, sr, elapsed, args.output, args.text)

    if args.backend == "tt":
        import ttnn
        ttnn.close_mesh_device(mesh_device)


def run_preset(args):
    """Generate speech for preset Japanese texts."""
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    output_dir = args.output_dir or "demo_ja_output"
    os.makedirs(output_dir, exist_ok=True)

    if args.preset == "all":
        texts = PRESET_TEXTS
    elif args.preset in PRESET_TEXTS:
        texts = {args.preset: PRESET_TEXTS[args.preset]}
    else:
        print(f"Unknown preset: {args.preset}. Available: {', '.join(PRESET_TEXTS.keys())}, all")
        return

    mesh_device = None
    speaker_emb_tt = None
    if args.backend == "reference":
        model = build_reference_model(args.model_path, args.device)
    else:
        generator, mesh_device = build_tt_generator(args.model_path, args.max_new_tokens)
        if args.load_speaker:
            from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

            speaker_emb_tt = SpeakerEncoder.load_embedding(args.load_speaker, mesh_device)
            print(f"Loaded speaker embedding from {args.load_speaker}")

    results = []
    print(f"\nGenerating {len(texts)} preset texts ({args.backend})...\n")
    for name, text in texts.items():
        print(f"[{name}] {text}")
        output_path = os.path.join(output_dir, f"{name}.wav")

        if args.backend == "reference":
            wav, sr, elapsed = generate_one_reference(model, text, args.language, gen_config)
        else:
            wav, sr, elapsed = generate_one_tt(generator, text, args.language, gen_config, speaker_emb_tt=speaker_emb_tt)

        result = save_and_report(wav, sr, elapsed, output_path, text)
        result["name"] = name
        results.append(result)

    print(f"\n{'='*60}")
    print(f"{'Name':<12} {'Duration':>8} {'Elapsed':>8} {'RTF':>6}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        print(f"{r['name']:<12} {r['duration']:>7.2f}s {r['elapsed']:>7.2f}s {r['rtf']:>6.3f}")
    mean_rtf = np.mean([r["rtf"] for r in results])
    total_dur = sum(r["duration"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*6}")
    print(f"{'Total':<12} {total_dur:>7.2f}s {total_elapsed:>7.2f}s {mean_rtf:>6.3f}")
    print(f"{'='*60}")

    if mesh_device is not None:
        import ttnn
        ttnn.close_mesh_device(mesh_device)


def run_interactive(args):
    """Interactive mode: continuously read text from stdin and generate speech."""
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    output_dir = args.output_dir or "demo_ja_output"
    os.makedirs(output_dir, exist_ok=True)

    mesh_device = None
    speaker_emb_tt = None
    if args.backend == "reference":
        model = build_reference_model(args.model_path, args.device)
    else:
        generator, mesh_device = build_tt_generator(args.model_path, args.max_new_tokens)
        if args.load_speaker:
            from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

            speaker_emb_tt = SpeakerEncoder.load_embedding(args.load_speaker, mesh_device)
            print(f"Loaded speaker embedding from {args.load_speaker}")

    print("\nQwen3-TTS Interactive Mode (Japanese)")
    print("Type text to synthesize. Commands: /quit, /preset <name|all>")
    print(f"Output directory: {output_dir}\n")

    count = 0
    try:
        while True:
            try:
                text = input(">> ").strip()
            except EOFError:
                break

            if not text:
                continue
            if text == "/quit":
                break

            if text.startswith("/preset"):
                parts = text.split(maxsplit=1)
                preset_name = parts[1] if len(parts) > 1 else "all"
                if preset_name == "all":
                    items = PRESET_TEXTS.items()
                elif preset_name in PRESET_TEXTS:
                    items = [(preset_name, PRESET_TEXTS[preset_name])]
                else:
                    print(f"Unknown preset: {preset_name}")
                    continue
                for name, ptext in items:
                    print(f"[{name}] {ptext}")
                    output_path = os.path.join(output_dir, f"preset_{name}.wav")
                    if args.backend == "reference":
                        wav, sr, elapsed = generate_one_reference(model, ptext, args.language, gen_config)
                    else:
                        wav, sr, elapsed = generate_one_tt(generator, ptext, args.language, gen_config, speaker_emb_tt=speaker_emb_tt)
                    save_and_report(wav, sr, elapsed, output_path, ptext)
                continue

            count += 1
            output_path = os.path.join(output_dir, f"interactive_{count:04d}.wav")
            print(f"Generating: {text}")
            if args.backend == "reference":
                wav, sr, elapsed = generate_one_reference(model, text, args.language, gen_config)
            else:
                wav, sr, elapsed = generate_one_tt(generator, text, args.language, gen_config, speaker_emb_tt=speaker_emb_tt)
            save_and_report(wav, sr, elapsed, output_path, text)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if mesh_device is not None:
        import ttnn
        ttnn.close_mesh_device(mesh_device)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 日本語 text-to-speech demo")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--text", type=str, default="こんにちは、世界。今日はいい天気ですね。")
    parser.add_argument("--language", type=str, default="japanese")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["reference", "tt"], default="reference")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio for voice cloning")
    parser.add_argument("--save_speaker", type=str, default=None, help="Save speaker embedding to .safetensors file")
    parser.add_argument("--load_speaker", type=str, default=None, help="Load speaker embedding from .safetensors file")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--preset", type=str, default=None,
        help=f"Run preset texts: {', '.join(PRESET_TEXTS.keys())}, all",
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
    elif args.preset:
        run_preset(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
