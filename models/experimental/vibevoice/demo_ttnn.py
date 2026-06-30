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
    python models/experimental/vibevoice/demo_ttnn.py --text ... --voice alice.wav carter.wav frank.wav --max_new_tokens 64 --debug
"""

from __future__ import annotations

import argparse
import json
import os
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
    voice_preset_demo_id,
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
        default=str(_VV_ROOT / "output"),
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
    ap.add_argument("--text", default=None, help="Custom script path (overrides --demo text)")
    ap.add_argument(
        "--voice",
        nargs="+",
        default=None,
        metavar="WAV",
        help="Voice clone WAV(s) for Speaker 1, 2, 3, … in order (repeatable path list)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Verbose stage logs (VV_DEBUG=1) + device-synced timing breakdown (VV_PROFILE=1)",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="ttnn-trace the post-diffusion block (VV_TRACE_POSTDIFF=1) + open device with "
        "trace region & 2 command queues. EVAL ONLY: not yet bit-exact (~0.985 PCC vs eager).",
    )
    args = ap.parse_args()

    if args.debug:
        os.environ["VV_DEBUG"] = "1"
        os.environ["VV_PROFILE"] = "1"
        print("[demo_ttnn] debug enabled (VV_DEBUG=1 VV_PROFILE=1)", flush=True)

    if args.trace:
        os.environ["VV_TRACE_POSTDIFF"] = "1"
        print("[demo_ttnn] post-diffusion trace enabled (VV_TRACE_POSTDIFF=1)", flush=True)

    if args.text:
        text_path = Path(args.text)
        if not text_path.is_file():
            print(f"[demo_ttnn] ERROR: text file not found: {text_path}", file=sys.stderr)
            return 1
        demo_id = text_path.stem
        demo_entry = None
        golden_wav = None
    else:
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
    if golden_wav is not None:
        shutil.copy2(golden_wav, paths["golden"])

    use_voice_cloning = False
    voice_mapping: Optional[list[dict[str, str]]] = None
    voice_samples: Optional[list[str]] = None
    if args.voice:
        voice_paths: list[Path] = []
        for voice_ref in args.voice:
            voice_path = Path(voice_ref)
            if not voice_path.is_file():
                print(f"[demo_ttnn] ERROR: voice file not found: {voice_path}", file=sys.stderr)
                return 1
            voice_paths.append(voice_path)
        use_voice_cloning = not args.no_voice_cloning
        voice_samples = [str(p) for p in voice_paths]
        voice_mapping = [
            {
                "speaker_id": str(speaker_idx),
                "name": voice_path.stem,
                "voice_file": voice_path.name,
            }
            for speaker_idx, voice_path in enumerate(voice_paths, start=1)
        ]
    elif not args.no_voice_cloning and voice_preset_demo_id(demo_id) in DEMO_VOICE_CLONES:
        use_voice_cloning = True
        voice_samples, voice_mapping = build_voice_samples(script, voice_preset_demo_id(demo_id))

    print(f"[demo_ttnn] demo={demo_id}  text={text_path.name}", flush=True)
    if demo_entry is not None:
        print(f"[demo_ttnn] {demo_entry.website_title} ({demo_entry.website_section})", flush=True)
    print(f"[demo_ttnn] output dir: {paths['dir']}", flush=True)
    if golden_wav is not None:
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

    if args.debug:
        speech_slots = int(inputs["speech_input_mask"][0].sum().item()) if "speech_input_mask" in inputs else 0
        voice_samples_sec = None
        if voice_samples and inputs.get("speech_tensors") is not None:
            voice_samples_sec = inputs["speech_tensors"].shape[-1] / SR
        print(
            "[demo_ttnn] stage 1/5 processor: "
            f"input_ids={tuple(inputs['input_ids'].shape)} "
            f"speech_slots={speech_slots} "
            f"voice_audio_sec={voice_samples_sec}",
            flush=True,
        )

    max_ar_steps = args.max_new_tokens
    if max_ar_steps is None:
        max_ar_steps = int(args.max_length_times * prefill_len)
    print(
        f"[demo_ttnn] prefill tokens={prefill_len}  max AR steps≈{max_ar_steps} (max_length_times={args.max_length_times})",
        flush=True,
    )

    import time as _time

    _open_kwargs = dict(device_id=0, l1_small_size=32768)
    if args.trace:
        # Reserve a trace buffer + a 2nd command queue for the post-diffusion trace.
        _open_kwargs.update(trace_region_size=200_000_000, num_command_queues=2)
    mesh = ttnn.open_device(**_open_kwargs)
    try:
        if args.debug:
            print("[demo_ttnn] stage 2/5 open_device: device_id=0 l1_small_size=32768", flush=True)
        print("[demo_ttnn] Loading TTVibeVoiceModel...", flush=True)
        _t_load0 = _time.perf_counter()
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh,
            model_path,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
        )
        print(f"[demo_ttnn] model load: {_time.perf_counter() - _t_load0:.1f}s", flush=True)
        if args.debug:
            print(
                "[demo_ttnn] stage 3/5 model loaded: LM + connectors + diffusion_head + "
                "acoustic/semantic tokenizers + DPM scheduler",
                flush=True,
            )

        torch.manual_seed(args.seed)
        print("[demo_ttnn] TT generate...", flush=True)
        if args.debug:
            print(
                "[demo_ttnn] stage 4/5 generate: prefill (voice encode + LM) → "
                f"AR loop up to {max_ar_steps} steps (see [VV_DEBUG] per step)",
                flush=True,
            )
        _t_gen0 = _time.perf_counter()
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
        _generate_wall = _time.perf_counter() - _t_gen0
        print(f"[demo_ttnn] generate wall: {_generate_wall:.1f}s", flush=True)
        tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)
        tt_gen = tt_out.sequences[0, prefill_len:]
        _ar_tokens = int(tt_gen.numel())
        _decode_tps = _ar_tokens / tt_out.decode_wall_s if tt_out.decode_wall_s > 0 else 0.0
        _prefill_tps = prefill_len / tt_out.prefill_wall_s if tt_out.prefill_wall_s > 0 else 0.0
        print(
            f"[demo_ttnn] prefill_tokens={prefill_len}  "
            f"TTFT={tt_out.prefill_wall_s:.2f}s  "
            f"decode={tt_out.decode_wall_s:.2f}s  "
            f"decode={_decode_tps:.1f} tok/s  "
            f"prefill={_prefill_tps:.0f} tok/s",
            flush=True,
        )
    finally:
        ttnn.close_device(mesh)

    if args.debug:
        print(
            f"[demo_ttnn] stage 5/5 save: wav + meta under {paths['dir']}",
            flush=True,
        )

    tt_path = paths["tt"]
    golden_out = paths["golden"]
    _write_wav(tt_path, tt_speech)

    metrics: dict = {}
    if golden_wav is not None:
        golden = _load_wav(golden_wav)
        metrics = _compare_audio(golden, tt_speech)

    meta = {
        "demo_id": demo_id,
        "website_title": demo_entry.website_title if demo_entry else None,
        "website_section": demo_entry.website_section if demo_entry else None,
        "text_file": text_path.name,
        "voice_cloning": use_voice_cloning,
        "voice_mapping": voice_mapping,
        "prefill_tokens": prefill_len,
        "ar_tokens_generated": int(tt_gen.numel()),
        "ttft_s": round(tt_out.prefill_wall_s, 3),
        "decode_wall_s": round(tt_out.decode_wall_s, 3),
        "decode_toks_per_s": round(_decode_tps, 2),
        "prefill_toks_per_s": round(_prefill_tps, 1),
        "generate_wall_s": round(_generate_wall, 3),
        "max_length_times": args.max_length_times,
        "max_new_tokens": args.max_new_tokens,
        "tt_wav": str(tt_path),
        "golden_wav": str(golden_out),
        "script_copy": str(paths["script"]),
        **metrics,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"[demo_ttnn] TT: {tt_gen.numel()} AR tokens, {tt_speech.numel() / SR:.2f}s → {tt_path}",
        flush=True,
    )
    if metrics:
        mel_msg = (
            f"log-mel L1={metrics['log_mel_l1']:.4f}"
            if metrics["log_mel_l1"] == metrics["log_mel_l1"]
            else f"prefix RMS={metrics['prefix_rms']:.4f}"
        )
        print(
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
    else:
        print(f"[demo_ttnn] DONE → {tt_path.name}  under {paths['dir']}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
