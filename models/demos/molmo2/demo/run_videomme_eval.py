#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Video-MME evaluation for Molmo2-8B on TT hardware.

Loads the model ONCE, captures warmup/traces ONCE, then iterates over
Video-MME samples — same approach as demo.py but in a batch loop.

Usage:
    python -m models.demos.molmo2.demo.run_videomme_eval \\
        --video-dir ~/.cache/huggingface/videomme/data \\
        --output results/videomme_results.jsonl \\
        --limit 50 \\
        --use-decode-trace

    # Run specific duration split:
    python -m models.demos.molmo2.demo.run_videomme_eval \\
        --duration short --limit 100 --output results/videomme_short.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn

# Reuse all demo infrastructure
from models.demos.molmo2.demo.demo import (
    PREFILL_SEQ_BUCKETS,
    VIDEO_MAX_FPS,
    Molmo2Generator,
    create_model,
    load_model_weights,
    load_processor,
    preprocess_video,
)

# ── Dataset helpers ────────────────────────────────────────────────────────────


def load_videomme_dataset(duration_filter: Optional[str] = None):
    """Load Video-MME parquet, optionally filtering by duration (short/medium/long)."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/Video-MME", "videomme", split="test")
    if duration_filter:
        ds = ds.filter(lambda x: x["duration"] == duration_filter)
    return ds


def build_prompt(question: str, options: list) -> str:
    """Build the multiple-choice prompt matching lmms-eval's videomme format."""
    options_str = "\n".join(options)
    return f"<|video|>\n{question}\n{options_str}\n" "Answer with the option's letter from the given choices directly."


def extract_answer_letter(response: str) -> str:
    """Extract A/B/C/D from model response."""
    response = response.strip()
    # Direct single letter
    if response and response[0].upper() in "ABCD":
        return response[0].upper()
    # Match "Answer: X" or "(X)" or "option X"
    m = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return response[:1].upper() if response else "?"


def score_response(predicted: str, ground_truth: str) -> bool:
    return predicted.upper() == ground_truth.upper()


# ── Main eval loop ─────────────────────────────────────────────────────────────


def run_eval(
    video_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    duration_filter: Optional[str] = None,
    max_new_tokens: int = 16,
    max_frames: int = 196,
    max_fps: float = VIDEO_MAX_FPS,
    max_seq_len: int = 16384,
    num_devices: int = 8,
    frames_per_device: int = 8,
    use_decode_trace: bool = True,
    use_vision_trace: bool = False,
    use_trace: bool = False,
    use_dp_vision_trace: bool = False,
    num_layers: Optional[int] = None,
):
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The HF processor evenly spaces num_frames across the full video duration regardless of length.
    # So num_frames=196 gives 196 evenly-spaced frames from a 5s video or a 60min video alike.
    # VIDEO_MAX_FRAMES=384 is the HF default (published score). We cap at 196 (HW ISL limit).
    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading Video-MME dataset...")
    ds = load_videomme_dataset(duration_filter)
    samples = list(ds)
    if limit:
        samples = samples[:limit]
    logger.info(f"Evaluating {len(samples)} samples (duration={duration_filter or 'all'})")

    # ── Initialize model (ONCE) ───────────────────────────────────────────────
    tokenizer = load_processor()
    state_dict = load_model_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    results = []
    n_correct = 0
    n_total = 0
    n_skipped = 0

    try:
        model = create_model(
            device,
            state_dict,
            num_layers,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            use_async_ccl=False,
        )
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=max_seq_len,
            use_paged_attention=True,
        )

        # ── Warmup / trace capture (ONCE) ─────────────────────────────────────
        # Use first available video to determine n_out and k_pool
        logger.info("Running warmup with trace capture (once for all videos)...")
        warmup_video_id = samples[0]["videoID"]
        warmup_video_path = video_dir / f"{warmup_video_id}.mp4"
        if not warmup_video_path.exists():
            # Try finding any video for warmup
            mp4s = list(video_dir.glob("*.mp4"))
            if not mp4s:
                raise FileNotFoundError(f"No .mp4 files found in {video_dir}")
            warmup_video_path = mp4s[0]
            logger.info(f"Using {warmup_video_path.name} for warmup")

        warmup_prompt = build_prompt(samples[0]["question"], samples[0]["options"])
        warmup_inputs = preprocess_video(
            str(warmup_video_path),
            warmup_prompt,
            num_frames=max_frames,
            apply_template=False,
        )

        from models.demos.molmo2.tt.vision_backbone import MAX_VIT_FRAMES_FOR_POOL

        _n_out = warmup_inputs["n_tokens"] // warmup_inputs["n_frames"]
        _k_pool = warmup_inputs["k_pool"]

        # Always initialize KV cache (allocates rot_mat_idxs, current_pos, page tables).
        generator.init_kv_cache()

        # Capture decode trace (and optionally prefill traces).
        if use_decode_trace or use_trace:
            generator.warmup_video_traces(
                frames_per_device=frames_per_device,
                num_devices=num_devices,
                prefill_buckets=[b for b in PREFILL_SEQ_BUCKETS if b <= generator.max_seq_len],
                pool_n_out=_n_out,
                pool_k_pool=_k_pool,
                max_vit_frames=MAX_VIT_FRAMES_FOR_POOL,
                use_prefill_trace=use_trace,
                use_decode_trace=use_decode_trace,
            )

        logger.info("Warmup complete. Starting eval loop...")

        # ── Eval loop ─────────────────────────────────────────────────────────
        for idx, sample in enumerate(samples):
            video_id = sample["videoID"]
            video_path = video_dir / f"{video_id}.mp4"

            if not video_path.exists():
                logger.warning(f"[{idx+1}/{len(samples)}] Video not found: {video_path} — skipping")
                n_skipped += 1
                results.append(
                    {
                        "idx": idx,
                        "question_id": sample["question_id"],
                        "video_id": video_id,
                        "duration": sample["duration"],
                        "domain": sample["domain"],
                        "question": sample["question"],
                        "options": sample["options"],
                        "ground_truth": sample["answer"],
                        "predicted": None,
                        "correct": False,
                        "skipped": True,
                        "error": "video_not_found",
                    }
                )
                continue

            prompt = build_prompt(sample["question"], sample["options"])

            try:
                t0 = time.perf_counter()

                # Preprocess video
                video_inputs = preprocess_video(
                    str(video_path),
                    prompt,
                    num_frames=max_frames,
                    apply_template=False,
                )
                preprocess_ms = (time.perf_counter() - t0) * 1000

                # Reset generator state between samples
                generator.reset_state()

                # Run inference
                response, perf_metrics = generator.run_video_inference(
                    video_inputs=video_inputs,
                    max_new_tokens=max_new_tokens,
                    use_trace=use_trace,
                    use_decode_trace=use_decode_trace,
                    use_vision_trace=use_vision_trace,
                    use_unified_trace=False,
                    use_dp_vision_trace=use_dp_vision_trace,
                    use_data_parallel=False,
                    frames_per_device=frames_per_device,
                )

                total_ms = (time.perf_counter() - t0) * 1000
                predicted = extract_answer_letter(response)
                correct = score_response(predicted, sample["answer"])
                n_correct += correct
                n_total += 1

                result = {
                    "idx": idx,
                    "question_id": sample["question_id"],
                    "video_id": video_id,
                    "duration": sample["duration"],
                    "domain": sample["domain"],
                    "question": sample["question"],
                    "options": sample["options"],
                    "ground_truth": sample["answer"],
                    "predicted": predicted,
                    "response": response,
                    "correct": correct,
                    "skipped": False,
                    "n_frames": perf_metrics.get("n_frames", 0),
                    "tokens_per_sec": perf_metrics.get("tokens_per_sec", 0),
                    "total_ms": total_ms,
                }
                results.append(result)

                running_acc = n_correct / n_total * 100 if n_total > 0 else 0
                logger.info(
                    f"[{idx+1}/{len(samples)}] {video_id} | "
                    f"GT={sample['answer']} Pred={predicted} {'✓' if correct else '✗'} | "
                    f"Acc={running_acc:.1f}% ({n_correct}/{n_total}) | "
                    f"{perf_metrics.get('tokens_per_sec', 0):.1f} tok/s | "
                    f"{total_ms/1000:.1f}s"
                )

            except Exception as e:
                logger.error(f"[{idx+1}/{len(samples)}] Error on {video_id}: {e}")
                n_skipped += 1
                results.append(
                    {
                        "idx": idx,
                        "question_id": sample["question_id"],
                        "video_id": video_id,
                        "duration": sample.get("duration"),
                        "domain": sample.get("domain"),
                        "question": sample["question"],
                        "options": sample["options"],
                        "ground_truth": sample["answer"],
                        "predicted": None,
                        "correct": False,
                        "skipped": True,
                        "error": str(e),
                    }
                )

            # Write results incrementally (crash-safe)
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0
    summary = {
        "task": "videomme",
        "duration_filter": duration_filter or "all",
        "total_samples": len(samples),
        "evaluated": n_total,
        "skipped": n_skipped,
        "correct": n_correct,
        "accuracy": accuracy,
        "published_score": 69.9,
    }
    logger.info("=" * 60)
    logger.info(f"Video-MME Results:")
    logger.info(f"  Accuracy: {accuracy:.2f}% ({n_correct}/{n_total})")
    logger.info(f"  Skipped:  {n_skipped}")
    logger.info(f"  Published (Molmo2-8B): 69.9%")
    logger.info("=" * 60)

    summary_path = output_path.parent / (output_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results:  {output_path}")
    logger.info(f"Summary:  {summary_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Video-MME evaluation for Molmo2-8B")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(Path.home() / ".cache/huggingface/videomme/data"),
        help="Directory containing extracted .mp4 files from Video-MME",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/videomme_results.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--duration",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Filter by video duration split",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens to generate per answer (default: 16, sufficient for A/B/C/D)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=196,
        help="Number of frames to evenly sample from each video (default: 196, HW ISL limit). Published score uses 384.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="Max sequence length for KV cache (default: 16384)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=8,
        help="Number of TT devices (default: 8 for T3K)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of text layers (default: 36)",
    )
    parser.add_argument(
        "--use-decode-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decode trace (default: ON)",
    )
    parser.add_argument(
        "--use-vision-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use non-DP vision trace (default: OFF — hangs on large frame counts; use --use-dp-vision-trace for video)",
    )
    parser.add_argument(
        "--use-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use prefill trace (default: OFF)",
    )
    parser.add_argument(
        "--use-dp-vision-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DP=8 vision trace (default: OFF)",
    )
    args = parser.parse_args()

    run_eval(
        video_dir=args.video_dir,
        output_path=args.output,
        limit=args.limit,
        duration_filter=args.duration,
        max_new_tokens=args.max_tokens,
        max_frames=args.max_frames,
        max_seq_len=args.max_seq_len,
        num_devices=args.num_devices,
        num_layers=args.num_layers,
        use_decode_trace=args.use_decode_trace,
        use_vision_trace=args.use_vision_trace,
        use_trace=args.use_trace,
        use_dp_vision_trace=args.use_dp_vision_trace,
    )


if __name__ == "__main__":
    main()
