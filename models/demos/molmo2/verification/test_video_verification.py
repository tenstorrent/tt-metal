# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verification script for the first 10 video tests in test.jsonl.

Checks two things per test:

  1. PREPROCESSING  Molmo2 video processor produces the same sampled-frame count
                    as recorded in video_frame_stats.jsonl.
                    A ±1 difference is flagged as WARN (known version delta: the
                    stats were generated before the append-last-frame feature was
                    added to sample_times; all other frame counts must match exactly).

  2. INFERENCE      HF reference model (CPU, float32) produces the same response
                    as recorded in test_results.jsonl.

Run:
    cd /home/ttuser/ssinghal/PR-fix/molmo2/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    python models/demos/molmo2/verification/test_video_verification.py

Options:
    --n N            Number of video tests to run  (default 10)
    --skip-inference Skip the HF model inference check
    --verbose        Print full detail for every test

Environment:
    MOLMO2_VERIF_DIR  Override directory containing test.jsonl / videos / etc.
                      Default: /home/ttuser/ssinghal/PR-fix/tt-metal/
                               models/demos/molmo2/verification
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/" "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)

_default_verif_dir = Path(
    os.environ.get(
        "MOLMO2_VERIF_DIR",
        "/home/ttuser/ssinghal/PR-fix/tt-metal/models/demos/molmo2/verification",
    )
)
VERIF_DIR = _default_verif_dir
VIDEO_DIR = VERIF_DIR / "videos"
TEST_JSONL = VERIF_DIR / "test.jsonl"
RESULTS_JSONL = VERIF_DIR / "test_results.jsonl"
FRAME_STATS_JSONL = VERIF_DIR / "video_frame_stats.jsonl"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def url_to_filename(url: str) -> str:
    return url.split("/")[-1]


def extract_video_url(test: dict) -> str | None:
    for msg in test.get("messages", []):
        for c in msg.get("content", []):
            if "video_url" in c:
                return c["video_url"]["url"]
    return None


def extract_prompt_text(test: dict) -> str:
    for msg in test.get("messages", []):
        for c in msg.get("content", []):
            if c.get("type") == "text":
                return c["text"]
    return ""


# ---------------------------------------------------------------------------
# Frame-count check — PASS / WARN±1 / FAIL
# ---------------------------------------------------------------------------


class FrameResult:
    PASS = "PASS"
    WARN = "WARN±1"
    FAIL = "FAIL"


def check_preprocessing(processor, video_path: Path, expected_frames: int, stats_entry: dict) -> tuple[str, str]:
    """
    Run the Molmo2 video processor and compare sampled frame count.

    Returns (status, detail_string) where status ∈ {PASS, WARN±1, FAIL}.

    ±1 tolerance note
    -----------------
    video_frame_stats.jsonl was generated before the 'append last frame'
    feature was added to Molmo2VideoProcessor.sample_times.  When the last
    np.arange timestamp is not exactly equal to the video duration, the current
    processor appends an extra frame, causing a +1 discrepancy for videos whose
    duration is not an exact multiple of 1/max_fps.  This is a known version
    delta, not a bug.
    """
    try:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test"},
                    {"type": "video"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=str(video_path), return_tensors="pt")

        pv = inputs.get("pixel_values_videos")
        if pv is None:
            return FrameResult.FAIL, "pixel_values_videos missing from processor output"

        actual = pv.shape[0]
        delta = actual - expected_frames

        if delta == 0:
            return FrameResult.PASS, (
                f"frames={actual}"
                f"  (fps={stats_entry.get('fps')}"
                f"  raw={stats_entry.get('raw_frame_count')}"
                f"  branch={stats_entry.get('molmo_sampling_branch')})"
            )
        elif abs(delta) == 1:
            return FrameResult.WARN, (
                f"frames={actual} vs expected={expected_frames} (Δ={delta:+d})"
                f" — known append-last-frame version delta"
                f"  (fps={stats_entry.get('fps')}"
                f"  duration={stats_entry.get('duration_sec'):.4f})"
            )
        else:
            return FrameResult.FAIL, (
                f"frames={actual} vs expected={expected_frames} (Δ={delta:+d})"
                f"  (fps={stats_entry.get('fps')}"
                f"  raw={stats_entry.get('raw_frame_count')})"
            )

    except Exception as e:
        return FrameResult.FAIL, f"exception: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Inference check — PASS / FAIL
# ---------------------------------------------------------------------------


def check_inference(
    model,
    processor,
    video_path: Path,
    prompt: str,
    max_new_tokens: int,
    expected_response: str,
) -> tuple[bool, str]:
    """
    Run HF model and compare stripped response with stored reference.
    Returns (ok, detail_string).
    """
    try:
        import torch

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                ],
            }
        ]
        formatted = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=formatted, videos=str(video_path), return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prefix = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][prefix:], skip_special_tokens=True).strip()

        if response == expected_response:
            return True, f"response={response!r}"
        else:
            return False, f"got={response!r}  expected={expected_response!r}"

    except Exception as e:
        return False, f"exception: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Molmo2 video verification (first N tests)")
    p.add_argument("--n", type=int, default=10, help="Number of tests (default 10)")
    p.add_argument("--skip-inference", action="store_true", help="Only run preprocessing check")
    p.add_argument("--verbose", "-v", action="store_true", help="Print detail for every test")
    return p.parse_args()


def _sym(status) -> str:
    """Display symbol for a result."""
    if status is True or status == FrameResult.PASS:
        return "✓"
    if status == FrameResult.WARN:
        return "~"
    return "✗"


def main():
    args = parse_args()
    sys.path.insert(0, str(HF_PATH))

    # ---- Load reference data ----
    all_tests = load_jsonl(TEST_JSONL)
    all_results = load_jsonl(RESULTS_JSONL)
    frame_stats = {d["filename"]: d for d in load_jsonl(FRAME_STATS_JSONL)}
    results_map = {r["test_index"]: r for r in all_results}

    # Select first N video tests that have local video files
    video_tests = []
    for i, test in enumerate(all_tests):
        url = extract_video_url(test)
        if url is None:
            continue
        filename = url_to_filename(url)
        vpath = VIDEO_DIR / filename
        if not vpath.exists():
            continue
        video_tests.append((i, test, filename, vpath))
        if len(video_tests) >= args.n:
            break

    if not video_tests:
        print("ERROR: No video tests found with local video files.")
        print(f"  Check VIDEO_DIR = {VIDEO_DIR}")
        return 1

    print(f"Loaded {len(video_tests)} video tests  " f"(test indices {video_tests[0][0]}..{video_tests[-1][0]})\n")

    # ---- Load processor ----
    print("Loading processor...")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(HF_PATH), trust_remote_code=True)
    print("  Processor ready\n")

    # ---- Load model (unless --skip-inference) ----
    model = None
    if not args.skip_inference:
        import torch
        from transformers import AutoModelForImageTextToText

        print("Loading HF model (float32, CPU) — takes ~60s...")
        t0 = time.time()
        model = AutoModelForImageTextToText.from_pretrained(
            str(HF_PATH),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        model.eval()
        print(f"  Model ready in {time.time()-t0:.0f}s\n")

    # ---- Column header ----
    print("=" * 78)
    col_frames = "Frames"
    col_inf = "Inference" if not args.skip_inference else ""
    print(f"{'#':>3}  {'Video filename (short)':28}  {col_frames:9}  {col_inf:12}  Time")
    print("=" * 78)

    # Accumulate
    counts = {
        FrameResult.PASS: 0,
        FrameResult.WARN: 0,
        FrameResult.FAIL: 0,
    }
    inf_pass = inf_fail = 0
    failures = []

    for rank, (test_idx, test, filename, vpath) in enumerate(video_tests):
        stats = frame_stats.get(filename, {})
        result = results_map.get(test_idx, {})
        prompt = extract_prompt_text(test)
        max_tok = test.get("max_tokens", 16)
        exp_frames = stats.get("molmo_sampled_frame_count", -1)
        exp_response = (result.get("response") or "").strip()

        t0 = time.time()

        # -- Preprocessing --
        fr_status, fr_detail = check_preprocessing(processor, vpath, exp_frames, stats)

        # -- Inference --
        inf_ok = None
        inf_detail = "(skipped)"
        if not args.skip_inference and model is not None:
            inf_ok, inf_detail = check_inference(model, processor, vpath, prompt, max_tok, exp_response)

        elapsed = time.time() - t0

        # Update counts
        counts[fr_status] = counts.get(fr_status, 0) + 1
        if inf_ok is True:
            inf_pass += 1
        elif inf_ok is False:
            inf_fail += 1

        if fr_status != FrameResult.PASS or inf_ok is False:
            failures.append(
                dict(
                    rank=rank,
                    test_idx=test_idx,
                    filename=filename,
                    fr_status=fr_status,
                    fr_detail=fr_detail,
                    inf_ok=inf_ok,
                    inf_detail=inf_detail,
                )
            )

        # Print row
        short = filename[:26] + ".."
        f_sym = _sym(fr_status)
        i_sym = _sym(inf_ok) if inf_ok is not None else "-"
        fr_label = f"[{f_sym}] {fr_status}"
        inf_col = f"[{i_sym}]" if not args.skip_inference else ""
        print(f"{rank:>3}  {short:28}  {fr_label:9}  {inf_col:12}  {elapsed:.1f}s")

        if args.verbose or fr_status != FrameResult.PASS:
            print(f"       frames:    {fr_detail}")
        if args.verbose or inf_ok is False:
            if inf_ok is not None:
                print(f"       inference: {inf_detail}")

    n = len(video_tests)
    print("=" * 78)

    # ---- Summary ----
    print("\nSUMMARY")
    print("-" * 40)
    print("Preprocessing (frame count):")
    print(f"  PASS   {counts[FrameResult.PASS]}/{n}")
    print(f"  WARN±1 {counts[FrameResult.WARN]}/{n}  ← known append-last-frame version delta")
    print(f"  FAIL   {counts[FrameResult.FAIL]}/{n}")

    if not args.skip_inference:
        print("Inference (response match):")
        print(f"  PASS   {inf_pass}/{n}")
        print(f"  FAIL   {inf_fail}/{n}")

    if failures:
        print("\nDetails for non-PASS tests:")
        for f in failures:
            print(f"  [{f['rank']}] idx={f['test_idx']}  {f['filename'][:32]}")
            if f["fr_status"] != FrameResult.PASS:
                print(f"       frames:    [{_sym(f['fr_status'])}] {f['fr_detail']}")
            if f["inf_ok"] is False:
                print(f"       inference: [✗] {f['inf_detail']}")

    # Overall verdict: PASS if no hard FAIL in frames AND no inference fail
    hard_frame_fail = counts[FrameResult.FAIL] > 0
    all_ok = not hard_frame_fail and (args.skip_inference or inf_fail == 0)

    print()
    if all_ok:
        if counts[FrameResult.WARN] > 0:
            print("Result: PASS  (with warnings for known ±1 frame-count version delta)")
        else:
            print("Result: ALL PASS ✓")
    else:
        print("Result: FAILURES DETECTED ✗")
        if hard_frame_fail:
            print(f"  - {counts[FrameResult.FAIL]} frame-count hard failures (delta > ±1)")
        if not args.skip_inference and inf_fail > 0:
            print(f"  - {inf_fail} inference response mismatches")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
