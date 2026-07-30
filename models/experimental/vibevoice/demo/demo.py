# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice TTNN demo — run on-device inference and write the generated audio.

Text-driven: pass a script with --text <path>, or --demo <id> as a shortcut for
resources/text/<id>.txt. With neither, the default script (1p_vibevoice.txt) is used.

This script runs TT inference only (no HuggingFace reference model).

Multi-speaker climate demos auto-enable voice cloning from resources/voices/:
  Speaker 1 Alice  -> en-Alice_woman.wav
  Speaker 2 Carter -> en-Carter_man.wav
  Speaker 3 Frank  -> en-Frank_man.wav
  Speaker 4 Maya   -> en-Maya_woman.wav

Usage (from tt-metal root):
    python models/experimental/vibevoice/demo/demo.py
    python models/experimental/vibevoice/demo/demo.py --demo 2p_goat
    python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --output_dir ~/vv_ttnn_long
    python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 256
    python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32
    python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_100min --isl 1024 --warmup
    python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --no-trace
    python models/experimental/vibevoice/demo/demo.py --text ... --voice alice.wav carter.wav frank.wav --max_new_tokens 64 --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import ttnn

from models.experimental.vibevoice.common.config import DEFAULT_TXT_PATH, TEXT_EXAMPLES_DIR, VIBEVOICE_ROOT
from models.experimental.vibevoice.common.safe_paths import safe_join, safe_output_path
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import (
    DEMO_VOICE_CLONES,
    build_voice_samples,
    voice_preset_demo_id,
    ensure_demo_resources,
    load_script,
)
from models.experimental.vibevoice.demo.perf_metrics import (
    crop_processor_inputs_to_isl,
    format_perf_line,
    summarize_generate_perf,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

SR = 24000


def _write_wav(path: Path, audio_1d: torch.Tensor) -> None:
    import soundfile as sf

    sf.write(str(path), audio_1d.detach().to(torch.float32).numpy(), SR)


def _split_script(script: str, n: int) -> list[str]:
    """Split a multi-speaker script into ``n`` roughly equal parts at speaker-turn boundaries.

    Each part is prefilled and generated independently, so the AR position resets per part.  A
    single pass is capped by max_position_embeddings (65536 - prefill frames, ~93 min for the
    100-min script), so chunking is the way to render a script that needs longer than that.
    Balancing is by character count, so parts are only approximately equal in duration.

    Chunking is only needed to exceed the single-pass position limit, not for output quality: a
    single pass renders 93 min clean.  Each extra boundary costs ~1 garbled minute while the new
    prefill settles, so prefer the fewest chunks that fit.

    Split on single newlines, not blank lines: ``load_script`` collapses the blank lines that
    separate turns in the resource .txt files, so by the time the script reaches here one turn is
    one non-empty line.
    """
    if n <= 1:
        return [script]
    turns = [t for t in script.split("\n") if t.strip()]
    if len(turns) < n:
        raise ValueError(f"--chunks {n} requested but the script only has {len(turns)} speaker turns")
    target = sum(len(t) for t in turns) / n
    parts: list[list[str]] = [[]]
    run = 0
    for turn in turns:
        # Start a new part once this one is past its share, while parts remain to be filled.
        if run >= target and len(parts) < n:
            parts.append([])
            run = 0
        parts[-1].append(turn)
        run += len(turn)
    return ["\n".join(p) for p in parts if p]


def _demo_output_paths(out_dir: Path, demo_id: str) -> dict[str, Path]:
    """Per-demo output layout: ``{out_dir}/{demo_id}/{demo_id}_*.wav``.

    ``demo_id`` is derived from a ``--demo`` / ``--text`` argument, so every artifact is
    pinned under ``out_dir``.
    """
    demo_dir = safe_join(out_dir, demo_id)
    demo_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": demo_dir,
        "tt": safe_join(demo_dir, f"{demo_id}_tt.wav"),
        "script": safe_join(demo_dir, f"{demo_id}_script.txt"),
        "meta": safe_join(demo_dir, f"{demo_id}_meta.json"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="VibeVoice TTNN demo")
    ap.add_argument(
        "--demo",
        default=None,
        help="Script id — shortcut for resources/text/<id>.txt (e.g. 4p_climate_45min)",
    )
    ap.add_argument(
        "--output_dir",
        default=str(VIBEVOICE_ROOT / "output"),
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
    ap.add_argument(
        "--isl",
        type=int,
        default=None,
        help="Crop processor batch to the first N tokens after tokenization (input sequence length)",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="Run an untimed short generate before the measured pass (warm program cache)",
    )
    ap.add_argument(
        "--warmup_tokens",
        type=int,
        default=4,
        help="AR steps for --warmup generate (default: 4)",
    )
    ap.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="Render the script as N independently-prefilled chunks and concatenate (default 1). "
        "Only needed for scripts longer than one pass allows (max_position_embeddings - prefill, "
        "~93 min for the 100-min script) — a single pass is quality-clean throughout. Each extra "
        "boundary costs ~1 garbled minute. Splits only at speaker-turn boundaries.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", default=None, help="Custom script path (overrides --demo)")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ttnn-trace the whole steady-state speech-diffusion frame as ONE device-driven graph "
        "(VV_TRACE_SEGMENT=1), the llama shape: neg-LM + diffusion + post-diffusion + pos-LM fused "
        "and replayed per frame — positions self-advance (ttnn.plus_one), RoPE is gathered on device "
        "(bf16), the neg embed is a per-frame input (a segment's first frame folds the negative "
        "prefill), pos hidden is loop-carried on device. Lifecycle is warmup -> throwaway capture -> "
        "reset (no capture-poison re-run). Reserves a large trace region & 2 command queues. "
        "On by default; pass --no-trace for eager decode.",
    )
    args = ap.parse_args()

    if args.debug:
        os.environ["VV_DEBUG"] = "1"
        os.environ["VV_PROFILE"] = "1"
        print("[vibevoice_demo] debug enabled (VV_DEBUG=1 VV_PROFILE=1)", flush=True)

    if args.trace:
        os.environ["VV_TRACE_SEGMENT"] = "1"
        print("[vibevoice_demo] trace enabled: whole-segment fused frame (VV_TRACE_SEGMENT=1, llama shape)", flush=True)
    else:
        os.environ["VV_TRACE_SEGMENT"] = "0"
        print("[vibevoice_demo] trace disabled (--no-trace): eager decode", flush=True)

    if args.text:
        # An explicitly named script file; normalized, with no base to pin it under.
        text_path = safe_output_path(args.text)
    elif args.demo:
        # A script id indexing into the bundled resources — pin it inside that tree.
        text_path = safe_join(TEXT_EXAMPLES_DIR, f"{args.demo}.txt")
    else:
        text_path = Path(DEFAULT_TXT_PATH)
    demo_id = text_path.stem

    out_dir = safe_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_demo_resources()
        model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"[vibevoice_demo] ERROR: {exc}", file=sys.stderr)
        return 1

    if not text_path.is_file():
        print(f"[vibevoice_demo] ERROR: text file not found: {text_path}", file=sys.stderr)
        return 1

    script = load_script(text_path)
    paths = _demo_output_paths(out_dir, demo_id)
    paths["script"].write_text(script + "\n", encoding="utf-8")

    use_voice_cloning = False
    voice_mapping: Optional[list[dict[str, str]]] = None
    voice_samples: Optional[list[str]] = None
    if args.voice:
        voice_paths: list[Path] = []
        for voice_ref in args.voice:
            voice_path = Path(voice_ref)
            if not voice_path.is_file():
                print(f"[vibevoice_demo] ERROR: voice file not found: {voice_path}", file=sys.stderr)
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

    print(f"[vibevoice_demo] demo={demo_id}  text={text_path.name}", flush=True)
    print(f"[vibevoice_demo] output dir: {paths['dir']}", flush=True)
    print("[vibevoice_demo] TT-only inference (no HuggingFace reference model)", flush=True)
    if use_voice_cloning:
        print("[vibevoice_demo] voice cloning enabled (on-device speech prefill):", flush=True)
        for entry in voice_mapping or []:
            print(
                f"  Speaker {entry['speaker_id']} ({entry['name']}) → {entry['voice_file']}",
                flush=True,
            )
    else:
        print("[vibevoice_demo] text-only prompt (no voice cloning samples)", flush=True)

    from models.experimental.vibevoice.reference.processor.vibevoice_processor import VibeVoiceProcessor

    processor = VibeVoiceProcessor.from_pretrained(model_path)

    def _build_inputs(text: str):
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }
        if voice_samples:
            processor_kwargs["voice_samples"] = [voice_samples]
        return processor(**processor_kwargs)

    scripts = _split_script(script, max(1, args.chunks))
    if len(scripts) > 1:
        print(
            f"[vibevoice_demo] chunked render: {len(scripts)} parts "
            f"({', '.join(f'{len(s)}ch' for s in scripts)}), each prefilled independently",
            flush=True,
        )
    inputs = _build_inputs(scripts[0])
    full_prefill_len = int(inputs["input_ids"].shape[1])
    if args.isl is not None:
        inputs = crop_processor_inputs_to_isl(inputs, args.isl)
        print(
            f"[vibevoice_demo] ISL crop: {full_prefill_len} → {args.isl} tokens (post-tokenization)",
            flush=True,
        )
    prefill_len = int(inputs["input_ids"].shape[1])

    if args.debug:
        speech_slots = int(inputs["speech_input_mask"][0].sum().item()) if "speech_input_mask" in inputs else 0
        voice_samples_sec = None
        if voice_samples and inputs.get("speech_tensors") is not None:
            voice_samples_sec = inputs["speech_tensors"].shape[-1] / SR
        print(
            "[vibevoice_demo] stage 1/5 processor: "
            f"input_ids={tuple(inputs['input_ids'].shape)} "
            f"speech_slots={speech_slots} "
            f"voice_audio_sec={voice_samples_sec}",
            flush=True,
        )

    max_ar_steps = args.max_new_tokens
    if max_ar_steps is None:
        max_ar_steps = int(args.max_length_times * prefill_len)
    print(
        f"[vibevoice_demo] prefill tokens={prefill_len}  max AR steps≈{max_ar_steps} "
        f"(max_new_tokens={args.max_new_tokens} max_length_times={args.max_length_times})",
        flush=True,
    )

    import time as _time

    _open_kwargs = dict(device_id=0, l1_small_size=32768)
    if args.trace:
        # Reserve a trace buffer + a 2nd command queue.  --trace holds one large fused-frame
        # capture (neg-LM + diffusion + post-diff + pos-LM).
        _open_kwargs.update(trace_region_size=1_400_000_000, num_command_queues=2)
    mesh = ttnn.open_device(**_open_kwargs)
    try:
        if args.debug:
            print("[vibevoice_demo] stage 2/5 open_device: device_id=0 l1_small_size=32768", flush=True)
        print("[vibevoice_demo] Loading TTVibeVoiceModel...", flush=True)
        _t_load0 = _time.perf_counter()
        tt_model = TTVibeVoiceModel.from_checkpoint(
            mesh,
            model_path,
            cfg_scale=args.cfg_scale,
            num_diffusion_steps=args.num_steps,
        )
        print(f"[vibevoice_demo] model load: {_time.perf_counter() - _t_load0:.1f}s", flush=True)
        if args.debug:
            print(
                "[vibevoice_demo] stage 3/5 model loaded: LM + connectors + diffusion_head + "
                "acoustic/semantic tokenizers + DPM scheduler",
                flush=True,
            )

        torch.manual_seed(args.seed)
        print("[vibevoice_demo] TT generate...", flush=True)
        if args.debug:
            print(
                "[vibevoice_demo] stage 4/5 generate: prefill (voice encode + LM) → "
                f"AR loop up to {max_ar_steps} steps (see [VV_DEBUG] per step)",
                flush=True,
            )
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

        if args.warmup:
            warm_n = max(1, int(args.warmup_tokens))
            print(f"[vibevoice_demo] warmup generate (max_new_tokens={warm_n}, untimed)...", flush=True)
            warm_kw = dict(generate_kwargs)
            warm_kw["max_new_tokens"] = warm_n
            torch.manual_seed(args.seed)
            _ = tt_model.generate(**warm_kw)
            ttnn.synchronize_device(mesh)
            print("[vibevoice_demo] warmup done; starting timed generate", flush=True)

        speech_parts: list[torch.Tensor] = []
        for chunk_idx, chunk_script in enumerate(scripts):
            if chunk_idx > 0:
                # Fresh prefill for this chunk.  generate() builds its own generator (and so its
                # own KV caches + fused-frame trace) per call and releases the trace on exit, so
                # nothing carries over between chunks except the loaded weights.
                inputs = _build_inputs(chunk_script)
                prefill_len = int(inputs["input_ids"].shape[1])
                generate_kwargs["input_ids"] = inputs["input_ids"]
                generate_kwargs["attention_mask"] = inputs["attention_mask"]
                generate_kwargs["speech_input_mask"] = inputs["speech_input_mask"]
                if voice_samples and inputs.get("speech_tensors") is not None:
                    generate_kwargs["speech_tensors"] = inputs["speech_tensors"]
                    generate_kwargs["speech_masks"] = inputs["speech_masks"]
            if len(scripts) > 1:
                print(
                    f"[vibevoice_demo] --- chunk {chunk_idx + 1}/{len(scripts)}: " f"prefill={prefill_len} tokens ---",
                    flush=True,
                )
            torch.manual_seed(args.seed)
            _t_gen0 = _time.perf_counter()
            tt_out = tt_model.generate(**generate_kwargs)
            ttnn.synchronize_device(mesh)
            _generate_wall = _time.perf_counter() - _t_gen0
            print(f"[vibevoice_demo] generate wall: {_generate_wall:.1f}s", flush=True)
            speech_parts.append(tt_out.speech_outputs[0].to(torch.float32).reshape(-1))
            tt_gen = tt_out.sequences[0, prefill_len:]
            _ar_tokens = int(tt_gen.numel())
            perf = summarize_generate_perf(
                prefill_len=prefill_len,
                ar_tokens=_ar_tokens,
                prefill_wall_s=tt_out.prefill_wall_s,
                decode_wall_s=tt_out.decode_wall_s,
                generate_wall_s=_generate_wall,
                steady_decode_s=tt_out.steady_decode_s,
                steady_decode_frames=tt_out.steady_decode_frames,
            )
            print(f"[vibevoice_demo] {format_perf_line(perf)}", flush=True)
        tt_speech = speech_parts[0] if len(speech_parts) == 1 else torch.cat(speech_parts)
        if len(speech_parts) > 1:
            print(
                f"[vibevoice_demo] concatenated {len(speech_parts)} chunks → " f"{tt_speech.numel() / SR / 60:.2f} min",
                flush=True,
            )
    finally:
        ttnn.close_device(mesh)

    if args.debug:
        print(
            f"[vibevoice_demo] stage 5/5 save: wav + meta under {paths['dir']}",
            flush=True,
        )

    tt_path = paths["tt"]
    _write_wav(tt_path, tt_speech)

    meta = {
        "demo_id": demo_id,
        "text_file": text_path.name,
        "voice_cloning": use_voice_cloning,
        "voice_mapping": voice_mapping,
        "isl": args.isl,
        "full_prefill_tokens": full_prefill_len,
        "warmup": bool(args.warmup),
        "warmup_tokens": args.warmup_tokens if args.warmup else None,
        "max_length_times": args.max_length_times,
        "max_new_tokens": args.max_new_tokens,
        "tt_wav": str(tt_path),
        "script_copy": str(paths["script"]),
        **perf,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"[vibevoice_demo] TT: {tt_gen.numel()} AR tokens, {tt_speech.numel() / SR:.2f}s → {tt_path}",
        flush=True,
    )
    print(f"[vibevoice_demo] DONE → {tt_path.name}  under {paths['dir']}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
