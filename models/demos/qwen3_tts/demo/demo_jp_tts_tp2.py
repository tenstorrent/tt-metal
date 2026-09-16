# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Japanese Qwen3-TTS: 1.7B and 0.6B on N300 TP=2.

Requires an N300 (or 2-chip mesh). Set MESH_DEVICE=N300.

    MESH_DEVICE=N300 PYTHONPATH=$PWD python models/demos/qwen3_tts/demo/demo_jp_tts_tp2.py

Writes:
    qwen3-tts-1.7B-tp2.wav
    qwen3-tts-0.6B-tp2.wav
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
    _load_ref_text_for,
    get_default_reference_path,
    run_full_ttnn_tts,
)

# 12 Hz codec: one frame is 80 ms of audio, so RTF = ms per frame / 80.
_MS_PER_FRAME_REALTIME = 80.0


def _run_one(queue, kwargs):
    """Child-process entry point. Kept at module level so "spawn" can pickle it."""
    try:
        queue.put(("ok", run_full_ttnn_tts(**kwargs)))
    except BaseException as exc:  # noqa: BLE001 - report, do not kill the parent
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def _run_isolated(kwargs):
    """Run one model in a fresh process.

    Two ``run_full_ttnn_tts`` calls in one process corrupt the second one: the 0.6B
    generates to the ``max_new_tokens`` cap (256 frames, 20.5 s of garbled audio)
    whenever it is not the first model, and running the 0.6B twice reproduces it with a
    bit-identical speaker embedding — so the state that leaks is downstream of the
    model, not in the weights or the encoder. ``run_full_ttnn_tts`` closes its device,
    but Python- and ttnn-level state survives. A fresh process is the reliable fix.
    Set QWEN3_TTS_JP_IN_PROCESS=1 to opt out (and reproduce the bug).
    """
    if os.environ.get("QWEN3_TTS_JP_IN_PROCESS", "0") != "0":
        return run_full_ttnn_tts(**kwargs)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_one, args=(queue, kwargs))
    proc.start()
    status, payload = queue.get()
    proc.join()
    if status != "ok":
        raise RuntimeError(f"model run failed in child process: {payload}")
    return payload


MODELS = (
    ("1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3-tts-1.7B-tp2.wav"),
    ("0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen3-tts-0.6B-tp2.wav"),
)


def _load_default_text() -> str:
    p = Path(__file__).with_name("sample_ja.txt")
    return p.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description="Japanese Qwen3-TTS 1.7B and 0.6B, N300 TP=2")
    parser.add_argument("--text", type=str, default=None, help="Japanese text (default: sample_ja.txt)")
    parser.add_argument("--output-dir", type=str, default="/tmp", help="Directory for wavs")
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MESH_DEVICE", "N300")
    text = args.text if args.text else _load_default_text()
    ref_audio = args.ref_audio if args.ref_audio else get_default_reference_path()
    ref_text = args.ref_text if args.ref_text else _load_ref_text_for(ref_audio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, hf_id, wav_name in MODELS:
        output_path = str(out_dir / wav_name)
        print("\n" + "#" * 80)
        print(f"# {label}  {hf_id}  →  {output_path}")
        print("#" * 80)
        result = _run_isolated(
            dict(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                output_path=output_path,
                language="japanese",
                seed=args.seed,
                device_id=args.device_id,
                hf_id=hf_id,
                use_2cq=True,
            )
        )
        rows.append((label, result))

    print("\n" + "=" * 80)
    print("1.7B vs 0.6B  (N300 TP=2)")
    print("=" * 80)
    print(f"{'Model':<8} {'Prefill':>10} {'Steady AR':>14} {'fps':>8} {'RTF':>7} {'frames':>7}  wav")
    print("-" * 80)
    for label, r in rows:
        # run_full_ttnn_tts does not return an "rtf" key — deriving it here is what the
        # old r["rtf"] lookup meant, and it used to KeyError after both wavs were written.
        rtf = r["steady_ms_per_frame"] / _MS_PER_FRAME_REALTIME
        print(
            f"{label:<8} {r['prefill_ms']:>8.1f} ms {r['steady_ms_per_frame']:>8.1f} ms/fr "
            f"{r['steady_frames_per_sec']:>8.2f} {rtf:>7.2f} {r['num_frames']:>7d}  {r['output_wav']}"
        )
    print("=" * 80)
    print("RTF vs 80 ms/frame (12 Hz). <1 is faster than real-time.")
    for label, r in rows:
        if r["num_frames"] >= 256:
            print(f"  WARNING {label}: hit the {r['num_frames']}-frame cap — EOS never fired, audio is garbled.")
    for label, r in rows:
        print(f"  {label}: {r['output_wav']}")


if __name__ == "__main__":
    main()
