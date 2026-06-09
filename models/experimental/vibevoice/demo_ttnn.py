# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice TTNN demo.

Loads TTVibeVoiceModel, runs generate(), writes output WAV files.
The VibeVoiceProcessor (HF) and WAV writing (soundfile) stay here — NOT in tt/.

Usage:
    export PYTHONPATH=$(pwd)
    python models/experimental/vibevoice/demo_ttnn.py \
        --text "Hello, how are you today?" \
        --voice models/experimental/vibevoice/resources/voices/en-Alice_woman.wav \
        --output_dir /tmp/vv_out
"""

import argparse
import sys
from pathlib import Path

import torch
import ttnn

from models.experimental.vibevoice.common.config import (
    DEFAULT_TXT_PATH,
    MODEL_PATH,
    QWEN_TOKENIZER,
    VOICES_DIR,
)
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel


def _load_audio(wav_path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample WAV file to mono float32 tensor [T]."""
    try:
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            try:
                import resampy

                data = resampy.resample(data, sr, target_sr)
            except ImportError:
                from scipy.signal import resample_poly

                gcd = __import__("math").gcd(target_sr, sr)
                data = resample_poly(data, target_sr // gcd, sr // gcd)
        return torch.tensor(data, dtype=torch.float32)
    except ImportError:
        # fallback: torchaudio
        import torchaudio

        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0)


def _write_wav(path: str, audio: torch.Tensor, sr: int = 24000):
    try:
        import soundfile as sf

        sf.write(path, audio.numpy(), sr)
    except ImportError:
        import torchaudio

        torchaudio.save(path, audio.unsqueeze(0), sr)


def _load_tokenizer_and_encode(text: str, model_path: str) -> torch.Tensor:
    """Tokenize text using Qwen tokenizer. Returns [1, S] LongTensor."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, trust_remote_code=True)
    ids = tok.encode(text, return_tensors="pt")  # [1, S]
    return ids


def main():
    parser = argparse.ArgumentParser(description="VibeVoice TTNN demo")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--text_file", type=str, default=str(DEFAULT_TXT_PATH))
    parser.add_argument("--voice", type=str, default=str(VOICES_DIR / "en-Alice_woman.wav"))
    parser.add_argument("--output_dir", type=str, default="/tmp/vv_ttnn_out")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--cfg_scale", type=float, default=1.3)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    try:
        args.model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"[demo_ttnn] ERROR: {exc}", file=sys.stderr)
        print("[demo_ttnn] Set VIBEVOICE_MODEL_PATH or pass --model_path", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Open device ───────────────────────────────────────────────────────
    print("[demo_ttnn] Opening device...")
    mesh_device = ttnn.open_device(device_id=0, l1_small_size=32768)

    try:
        # ── Load model ────────────────────────────────────────────────────
        print("[demo_ttnn] Loading TTVibeVoiceModel from checkpoint...")
        model = TTVibeVoiceModel.from_checkpoint(
            mesh_device=mesh_device,
            model_path=args.model_path,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
        )
        print("[demo_ttnn] Model loaded.")

        # ── Prepare inputs ────────────────────────────────────────────────
        text = args.text
        if text is None:
            with open(args.text_file) as f:
                text = f.read().strip()
        print(f"[demo_ttnn] Text: {text[:100]}...")

        # Tokenize
        input_ids = _load_tokenizer_and_encode(text, args.model_path)  # [1, S]

        # Load voice audio
        print(f"[demo_ttnn] Loading voice from {args.voice}")
        voice_audio = _load_audio(args.voice)  # [T]
        voice_audio_4d = voice_audio.view(1, 1, 1, -1).to(torch.bfloat16)
        voice_audio_tt = ttnn.as_tensor(
            voice_audio_4d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ── Generate ──────────────────────────────────────────────────────
        print("[demo_ttnn] Generating...")
        output = model.generate(
            input_ids=input_ids,
            voice_audio_tt=voice_audio_tt,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens,
        )

        # ── Write outputs ─────────────────────────────────────────────────
        print(f"[demo_ttnn] Generated {len(output.speech_outputs)} speech segment(s)")
        for i, wav in enumerate(output.speech_outputs):
            out_path = str(Path(args.output_dir) / f"output_{i:03d}.wav")
            _write_wav(out_path, wav.cpu().float())
            print(f"[demo_ttnn] Wrote {out_path}")

        if not output.speech_outputs:
            print("[demo_ttnn] No speech outputs generated (check token constraint config).")

    finally:
        ttnn.close_device(mesh_device)
        print("[demo_ttnn] Device closed.")


if __name__ == "__main__":
    main()
