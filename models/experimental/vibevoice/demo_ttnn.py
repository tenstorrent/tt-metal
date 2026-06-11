# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice TTNN demo — run on-device inference and compare against website golden audio.

Defaults to the shortest golden clip with a local text script (1p_CH2EN /
resources/text/1p_Ch2EN.txt vs resources/golden/1p_CH2EN.wav from
https://microsoft.github.io/VibeVoice/).

This script runs TT inference only (no HuggingFace reference model). For an optional
PyTorch baseline on the same inputs, use demo_hf.py or reference/run_inference.py.

Multi-speaker climate demos auto-enable voice cloning from resources/voices/:
  Speaker 1 Alice  -> en-Alice_woman.wav
  Speaker 2 Carter -> en-Carter_man.wav
  Speaker 3 Frank  -> en-Frank_man.wav
  Speaker 4 Maya   -> en-Maya_woman.wav

Usage (from tt-metal root):
    python models/experimental/vibevoice/demo_ttnn.py
    python models/experimental/vibevoice/demo_ttnn.py --demo 2p_goat
    python models/experimental/vibevoice/demo_ttnn.py --demo 4p_climate_45min --output_dir ~/vv_ttnn_long
    python models/experimental/vibevoice/demo_ttnn.py --demo 4p_climate_45min --max_new_tokens 256
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
import ttnn

from models.experimental.vibevoice.common.golden_audio_utils import (
    MINIMAL_GOLDEN_DEMO_ID,
    download_golden_demo,
    get_golden_demo,
    minimal_golden_demo,
    text_path_for_demo,
)
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import (
    DEMO_VOICE_CLONES,
    build_voice_samples,
    ensure_demo_resources,
    load_script,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VV_ROOT = Path(__file__).resolve().parent
for _p in (_VV_ROOT / "reference", _VV_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SR = 24000


def _write_wav(path: Path, audio_1d: torch.Tensor) -> None:
    import soundfile as sf

    sf.write(str(path), audio_1d.detach().to(torch.float32).numpy(), SR)


def _load_wav(path: Path) -> torch.Tensor:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.tensor(data, dtype=torch.float32)
    if file_sr != SR:
        import torchaudio

        wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, SR).squeeze(0)
    return wav


def _demo_output_paths(out_dir: Path, demo_id: str) -> dict[str, Path]:
    """Per-demo output layout: ``{out_dir}/{demo_id}/{demo_id}_*.wav``."""
    demo_dir = out_dir / demo_id
    demo_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": demo_dir,
        "tt": demo_dir / f"{demo_id}_tt.wav",
        "golden": demo_dir / f"{demo_id}_golden.wav",
        "script": demo_dir / f"{demo_id}_script.txt",
        "meta": demo_dir / f"{demo_id}_meta.json",
    }


def _compare_audio(golden: torch.Tensor, tt: torch.Tensor) -> dict:
    dur_g = golden.numel() / SR
    dur_t = tt.numel() / SR
    n = min(golden.numel(), tt.numel())
    prefix_rms = (golden[:n] - tt[:n]).pow(2).mean().sqrt().item()
    log_mel_l1 = float("nan")
    try:
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=256, n_mels=80)
        lm_g = torch.log(mel(golden[:n]) + 1e-5)
        lm_t = torch.log(mel(tt[:n]) + 1e-5)
        log_mel_l1 = (lm_g - lm_t).abs().mean().item()
    except (ImportError, OSError):
        pass
    return {
        "golden_sec": dur_g,
        "tt_sec": dur_t,
        "duration_ratio": dur_t / dur_g if dur_g > 0 else float("nan"),
        "prefix_rms": prefix_rms,
        "log_mel_l1": log_mel_l1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="VibeVoice TTNN demo vs website golden audio")
    ap.add_argument(
        "--demo",
        default=None,
        help=f"Golden demo id (default: shortest with text, usually {MINIMAL_GOLDEN_DEMO_ID})",
    )
    ap.add_argument(
        "--output_dir",
        default="/tmp/vv_ttnn_out",
        help="Root output dir; writes {output_dir}/{demo_id}/{demo_id}_tt.wav etc.",
    )
    ap.add_argument("--model_path", default=None, help="VibeVoice checkpoint (auto-download if omitted)")
    ap.add_argument(
        "--no-voice-cloning",
        action="store_true",
        help="Disable voice cloning even when the demo has a speaker preset",
    )
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional AR cap (default: until EOS, bounded by max_length_times)",
    )
    ap.add_argument(
        "--max_length_times",
        type=float,
        default=2.0,
        help="Max AR steps ≈ max_length_times × prefill token length (HF default: 2)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    demo_entry = get_golden_demo(args.demo) if args.demo else minimal_golden_demo()
    demo_id = demo_entry.id
    golden_wav = download_golden_demo(demo_id)
    text_path = text_path_for_demo(demo_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_demo_resources()
        model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"[demo_ttnn] ERROR: {exc}", file=sys.stderr)
        return 1

    script = load_script(text_path)
    paths = _demo_output_paths(out_dir, demo_id)
    paths["script"].write_text(script + "\n", encoding="utf-8")
    shutil.copy2(golden_wav, paths["golden"])

    use_voice_cloning = not args.no_voice_cloning and demo_id in DEMO_VOICE_CLONES
    voice_mapping: Optional[list[dict[str, str]]] = None
    voice_samples: Optional[list[str]] = None
    if use_voice_cloning:
        voice_samples, voice_mapping = build_voice_samples(script, demo_id)

    print(f"[demo_ttnn] demo={demo_id}  text={text_path.name}  golden={golden_wav.name}", flush=True)
    print(f"[demo_ttnn] {demo_entry.website_title} ({demo_entry.website_section})", flush=True)
    print(f"[demo_ttnn] output dir: {paths['dir']}", flush=True)
    print(f"[demo_ttnn] golden reference → {paths['golden']}", flush=True)
    print("[demo_ttnn] TT-only inference (no HuggingFace reference model)", flush=True)
    if use_voice_cloning:
        print("[demo_ttnn] voice cloning enabled (on-device speech prefill):", flush=True)
        for entry in voice_mapping or []:
            print(
                f"  Speaker {entry['speaker_id']} ({entry['name']}) → {entry['voice_file']}",
                flush=True,
            )
    else:
        print("[demo_ttnn] text-only prompt (no voice cloning samples)", flush=True)

    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    processor_kwargs = {
        "text": [script],
        "padding": True,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    if voice_samples:
        processor_kwargs["voice_samples"] = [voice_samples]
    inputs = processor(**processor_kwargs)
    prefill_len = inputs["input_ids"].shape[1]

    max_ar_steps = args.max_new_tokens
    if max_ar_steps is None:
        max_ar_steps = int(args.max_length_times * prefill_len)
    print(
        f"[demo_ttnn] prefill tokens={prefill_len}  max AR steps≈{max_ar_steps} (max_length_times={args.max_length_times})",
        flush=True,
    )

    mesh = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        print("[demo_ttnn] Loading TTVibeVoiceModel...", flush=True)
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh,
            model_path,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
        )

        torch.manual_seed(args.seed)
        print("[demo_ttnn] TT generate...", flush=True)
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "speech_input_mask": inputs["speech_input_mask"],
            "tokenizer": processor.tokenizer,
            "cfg_scale": args.cfg_scale,
            "num_diffusion_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "max_length_times": args.max_length_times,
        }
        if voice_samples and inputs.get("speech_tensors") is not None:
            generate_kwargs["speech_tensors"] = inputs["speech_tensors"]
            generate_kwargs["speech_masks"] = inputs["speech_masks"]
        tt_out = tt_model.generate(**generate_kwargs)
        tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
        tt_gen = tt_out.sequences[0, prefill_len:]
    finally:
        ttnn.close_device(mesh)

    tt_path = paths["tt"]
    golden_out = paths["golden"]
    _write_wav(tt_path, tt_speech)

    golden = _load_wav(golden_wav)
    metrics = _compare_audio(golden, tt_speech)

    meta = {
        "demo_id": demo_id,
        "website_title": demo_entry.website_title,
        "website_section": demo_entry.website_section,
        "text_file": text_path.name,
        "voice_cloning": use_voice_cloning,
        "voice_mapping": voice_mapping,
        "prefill_tokens": prefill_len,
        "ar_tokens_generated": int(tt_gen.numel()),
        "max_length_times": args.max_length_times,
        "max_new_tokens": args.max_new_tokens,
        "tt_wav": str(tt_path),
        "golden_wav": str(golden_out),
        "script_copy": str(paths["script"]),
        **metrics,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    mel_msg = (
        f"log-mel L1={metrics['log_mel_l1']:.4f}"
        if metrics["log_mel_l1"] == metrics["log_mel_l1"]
        else f"prefix RMS={metrics['prefix_rms']:.4f}"
    )
    print(
        f"[demo_ttnn] TT:  {tt_gen.numel()} AR tokens, {metrics['tt_sec']:.2f}s → {tt_path}\n"
        f"[demo_ttnn] Golden: {metrics['golden_sec']:.2f}s → {golden_out}\n"
        f"[demo_ttnn] Compare (prefix {min(golden.numel(), tt_speech.numel())/SR:.2f}s): "
        f"{mel_msg}  duration ratio={metrics['duration_ratio']:.3f}"
    )
    if abs(metrics["duration_ratio"] - 1.0) > 0.05:
        print(
            "[demo_ttnn] note: duration differs from golden — website audio used a fixed Microsoft "
            "demo run; TT free-running LM may stop at a different EOS step."
        )

    print(f"[demo_ttnn] DONE → {tt_path.name}  vs  {golden_out.name}  under {paths['dir']}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
