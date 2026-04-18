"""
Molmo2-8B Video Evaluation Script

Runs batch inference on test.jsonl video entries and reports:
  - Per-video: frames, TTFT, tok/s, predicted answer
  - Aggregate: avg frames/sec, avg TTFT, avg tok/s, total processed
  - Optional ``--output-jsonl``: one JSON object per line with question and answer letter
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

from loguru import logger

from models.demos.molmo2.tests.verification_jsonl import extract_prompt_video_max, find_local_video_for_url
from models.demos.molmo2.tt.hf_processor import preprocess_video
from models.demos.molmo2.tt.utils import PREFILL_SEQ_BUCKETS, VIDEO_PROMPT

# Default paths
SCRIPT_DIR = Path(__file__).parent
MOLMO_DIR = SCRIPT_DIR.parent
DEFAULT_TEST_JSONL = MOLMO_DIR / "verification" / "test.jsonl"
DEFAULT_CACHE_DIR = MOLMO_DIR / "verification" / "video_cache"


def download_video(url: str, cache_dir: Path) -> str:
    """
    Download a video from a URL to a local cache directory.

    Returns the local file path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use URL filename (last path segment without query string)
    url_path = url.split("?")[0]
    filename = url_path.rstrip("/").split("/")[-1]
    if not filename:
        filename = "video.mp4"

    local_path = cache_dir / filename
    if local_path.exists():
        logger.debug(f"Using cached video: {local_path}")
        return str(local_path)

    logger.info(f"Downloading: {url} -> {local_path}")
    try:
        urllib.request.urlretrieve(url, str(local_path))
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    return str(local_path)


def resolve_video_local_path(url: str, cache_path: Path, skip_download: bool) -> Optional[str]:
    """Prefer repo-local copy (hash name), else cache path, else download."""
    local = find_local_video_for_url(url)
    if local is not None and local.is_file():
        return str(local)
    if skip_download:
        url_path = url.split("?")[0]
        filename = url_path.rstrip("/").split("/")[-1] or "video.mp4"
        p = cache_path / filename
        return str(p) if p.is_file() else None
    try:
        return download_video(url, cache_path)
    except Exception as e:
        logger.warning(f"Could not resolve video for {url}: {e}")
        return None


def extract_question_and_choices(text: str) -> tuple[str, list[str]]:
    """
    Parse the question text to find the question body and multiple-choice options.

    Returns (question_text, [choice_letters])
    """
    lines = text.strip().split("\n")
    choices = []
    for line in lines:
        m = re.match(r"^([A-Z])\.\s", line)
        if m:
            choices.append(m.group(1))
    return text, choices


def extract_answer_letter(response: str) -> Optional[str]:
    """
    Extract a single letter answer from the model's response.

    Looks for standalone A/B/C/D etc. at the start of the response.
    """
    text = response.strip()
    m = re.match(r"^([A-Z])[\.\s,]?", text)
    if m:
        return m.group(1)
    # Try finding any uppercase letter surrounded by whitespace
    m = re.search(r"\b([A-Z])\b", text)
    if m:
        return m.group(1)
    return None


def run_eval(
    test_jsonl: str,
    cache_dir: str,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 16,
    max_seq_len: int = 16384,
    max_frames: int = 8,
    max_fps: Optional[float] = None,
    video_fps: Optional[float] = None,
    video_sampling_fps: Optional[float] = None,
    num_layers: Optional[int] = None,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
    use_dp_vision_trace: bool = True,
    skip_download: bool = False,
    exact_jsonl: bool = True,
    unique_video_urls_only: bool = False,
    output_jsonl: Optional[str] = None,
):
    """
    Run batch video evaluation.

    Args:
        test_jsonl: Path to test.jsonl file
        cache_dir: Directory to cache downloaded videos
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Max tokens per sample when ``unique_video_urls_only`` is True;
            ignored in exact JSONL mode (each row's ``max_tokens`` is used).
        max_seq_len: KV cache sequence length (16384 for video)
        max_frames: Max frames passed to HF ``preprocess_video`` (``num_frames``).
        max_fps: HF ``max_fps`` cap; ``None`` omits the argument (HF processor default).
        video_fps / video_sampling_fps: Optional HF ``fps`` / ``sampling_fps`` overrides.
        num_layers: Number of model layers (None = 36)
        use_trace: Enable prefill tracing
        use_decode_trace: Enable decode tracing
        use_vision_trace: Enable vision backbone tracing
        use_unified_trace: Enable unified Vision+Prefill trace
        use_dp_vision_trace: Capture ViT + pool-chunk traces before inference (fast vision path).
            When True, runs ``warmup_video_traces`` ViT+pool steps even if prefill/decode traces are off.
        skip_download: Skip downloading videos (assume they are already cached)
        exact_jsonl: If True (default), one run per JSONL line with that row's
            prompt and ``max_tokens`` (matches ``verification/test.jsonl`` harness).
        unique_video_urls_only: If True, one run per unique video URL and global
            ``max_new_tokens`` (legacy / faster smoke path).
        output_jsonl: If set, append one JSON object per run with ``question``,
            ``answer_letter`` (predicted, or null on failure), and ``jsonl_line``.
    """
    import ttnn

    # Import demo functions here to avoid circular imports
    from models.demos.molmo2.demo.demo import Molmo2Generator, create_model, load_model_weights, load_processor

    cache_path = Path(cache_dir)
    test_path = Path(test_jsonl)

    with open(test_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if num_samples is not None:
        entries = entries[:num_samples]

    logger.info(f"Loaded {len(entries)} test entries from {test_path}")

    if unique_video_urls_only:
        exact_jsonl = False

    url_to_local: dict[str, str] = {}

    def get_or_resolve_local(video_url: str) -> Optional[str]:
        if video_url in url_to_local:
            return url_to_local[video_url]
        lp = resolve_video_local_path(video_url, cache_path, skip_download)
        if lp and Path(lp).exists():
            url_to_local[video_url] = lp
            return lp
        return None

    runnable = []

    if exact_jsonl:
        logger.info("Mode: exact test.jsonl (one run per line, per-row max_tokens)")
        for line_idx, entry in enumerate(entries):
            try:
                prompt_text, video_url, row_max_tokens = extract_prompt_video_max(entry)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping line {line_idx}: {e}")
                continue
            local_path = get_or_resolve_local(video_url)
            if not local_path:
                logger.warning(f"Skipping line {line_idx}: no local file for {video_url}")
                continue
            runnable.append(
                {
                    "line_idx": line_idx,
                    "prompt_text": prompt_text,
                    "url": video_url,
                    "max_new_tokens": row_max_tokens,
                    "local_path": local_path,
                }
            )
    else:
        logger.info("Mode: unique video URLs only (global --max-tokens)")
        seen_urls = set()
        for line_idx, entry in enumerate(entries):
            msg = entry["messages"][0]
            url = None
            text = None
            for part in msg["content"]:
                if part["type"] == "text":
                    text = part["text"]
                elif part["type"] == "video_url":
                    url = part["video_url"]["url"]
            if url and url not in seen_urls:
                seen_urls.add(url)
                local_path = get_or_resolve_local(url)
                if not local_path:
                    logger.warning(f"Skipping first row for {url}: no local file")
                    continue
                runnable.append(
                    {
                        "line_idx": line_idx,
                        "prompt_text": text,
                        "url": url,
                        "max_new_tokens": max_new_tokens,
                        "local_path": local_path,
                    }
                )

        logger.info(f"Unique videos with local files: {len(runnable)}")

    logger.info(f"Runs scheduled: {len(runnable)}")

    if not runnable:
        logger.error("No videos available. Check cache dir or download them first.")
        return

    # Load model
    logger.info("Loading tokenizer and model weights...")
    tokenizer = load_processor()
    state_dict = load_model_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        model = create_model(device, state_dict, num_layers)
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=max_seq_len,
        )

        # Upfront warmup: ViT+pool traces are required for fast DP vision (else eager path can be ~tens of seconds).
        # Prefill/decode traces are optional and only captured when use_trace / use_decode_trace.
        if runnable and (use_dp_vision_trace or use_trace or use_decode_trace):
            from models.demos.molmo2.tt.vision_backbone import MAX_VIT_FRAMES_FOR_POOL

            first = runnable[0]
            warm_prompt = f"{VIDEO_PROMPT}\n{first['prompt_text']}"
            warm_inputs = preprocess_video(
                first["local_path"],
                warm_prompt,
                num_frames=max_frames,
                max_fps=max_fps,
                fps=video_fps,
                sampling_fps=video_sampling_fps,
            )
            _n_out = warm_inputs["n_tokens"] // warm_inputs["n_frames"]
            _k_pool = warm_inputs["k_pool"]
            _buckets = [b for b in PREFILL_SEQ_BUCKETS if b <= max_seq_len]

            generator.init_kv_cache()
            generator.warmup_video_traces(
                frames_per_device=8,
                num_devices=8,
                prefill_buckets=_buckets,
                max_frames_per_pool_chunk=16,
                pool_n_out=_n_out,
                pool_k_pool=_k_pool,
                max_vit_frames=MAX_VIT_FRAMES_FOR_POOL,
                use_prefill_trace=use_trace,
                use_decode_trace=use_decode_trace,
            )
            if use_trace or use_decode_trace:
                logger.info("Upfront video trace warmup complete (ViT + pool + prefill/decode as requested).")
            else:
                logger.info(
                    "Upfront ViT + pool trace warmup complete (prefill/decode still eager; "
                    "add --use-trace / --use-decode-trace to capture those traces)."
                )

        results = []
        total_vision_ms = 0.0
        total_ttft_ms = 0.0
        total_decode_ms = 0.0
        total_frames = 0
        total_tokens = 0
        num_succeeded = 0

        logger.info("\n" + "=" * 70)
        logger.info("Starting video evaluation")
        logger.info("=" * 70)

        jsonl_file = None
        if output_jsonl:
            jsonl_path = Path(output_jsonl)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_file = open(jsonl_path, "w", encoding="utf-8")

        for idx, item in enumerate(runnable):
            video_path = item["local_path"]
            question_text = item["prompt_text"]
            url = item["url"]
            row_max_tokens = item["max_new_tokens"]
            line_idx = item["line_idx"]

            logger.info(f"\n[{idx+1}/{len(runnable)}] line={line_idx} {Path(video_path).name}")
            logger.info(f"  Question: {question_text[:100]}...")
            logger.info(f"  max_new_tokens: {row_max_tokens}")

            try:
                # HF processor: same as demo / verification_jsonl (newline after <|video|>)
                hf_prompt = f"{VIDEO_PROMPT}\n{question_text}"
                t0 = time.perf_counter()
                video_inputs = preprocess_video(
                    video_path,
                    hf_prompt,
                    num_frames=max_frames,
                    max_fps=max_fps,
                    fps=video_fps,
                    sampling_fps=video_sampling_fps,
                )
                extract_ms = (time.perf_counter() - t0) * 1000
                n_frames = video_inputs["n_frames"]

                generator.init_kv_cache()
                generator.reset_kv_cache(start_pos=0)

                response, perf = generator.run_video_inference(
                    video_inputs=video_inputs,
                    max_new_tokens=row_max_tokens,
                    use_trace=use_trace,
                    use_decode_trace=use_decode_trace,
                    use_vision_trace=use_vision_trace,
                    use_unified_trace=use_unified_trace,
                    use_dp_vision_trace=use_dp_vision_trace,
                )

                predicted = extract_answer_letter(response)
                _, choices = extract_question_and_choices(question_text)

                total_frames += n_frames
                total_vision_ms += perf["vision_ms"]
                total_ttft_ms += perf["ttft_ms"]
                total_decode_ms += perf["total_decode_ms"]
                total_tokens += perf["generated_tokens"]
                num_succeeded += 1

                result = {
                    "idx": idx,
                    "jsonl_line": line_idx,
                    "url": url,
                    "max_new_tokens": row_max_tokens,
                    "n_frames": n_frames,
                    "extract_ms": extract_ms,
                    "vision_ms": perf["vision_ms"],
                    "compile_vision_ms": perf.get("compile_vision_ms", 0),
                    "frames_per_sec": perf["frames_per_sec"],
                    "ttft_ms": perf["ttft_ms"],
                    "total_decode_ms": perf["total_decode_ms"],
                    "total_e2e_ms": extract_ms + perf["vision_ms"] + perf["ttft_ms"] + perf["total_decode_ms"],
                    "tokens_per_sec": perf["tokens_per_sec"],
                    "input_tokens": perf["input_tokens"],
                    "generated_tokens": perf["generated_tokens"],
                    "response": response,
                    "predicted_answer": predicted,
                    "choices": choices,
                }
                results.append(result)

                if jsonl_file:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "jsonl_line": line_idx,
                                "question": question_text,
                                "answer_letter": predicted,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                compile_vision_ms = perf.get("compile_vision_ms", 0)
                compile_str = f" [compile:{compile_vision_ms:.0f}ms]" if compile_vision_ms > 0 else ""
                total_e2e_ms = extract_ms + perf["vision_ms"] + perf["ttft_ms"] + perf["total_decode_ms"]
                logger.info(
                    f"  Frames: {n_frames} | Vision: {perf['vision_ms']:.0f}ms{compile_str} "
                    f"({perf['frames_per_sec']:.1f} fps) | TTFT: {perf['ttft_ms']:.0f}ms | "
                    f"Decode: {perf['tokens_per_sec']:.1f} tok/s | "
                    f"Answer: {predicted}"
                )
                logger.info(
                    f"  Total E2E (preproc+vision+prefill+decode): "
                    f"preproc={extract_ms:.0f}ms + vision={perf['vision_ms']:.0f}ms + "
                    f"prefill={perf['ttft_ms']:.0f}ms + decode={perf['total_decode_ms']:.0f}ms "
                    f"= {total_e2e_ms:.0f}ms ({total_e2e_ms/1000:.2f}s)"
                )

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results.append(
                    {
                        "idx": idx,
                        "jsonl_line": line_idx,
                        "url": url,
                        "error": str(e),
                    }
                )
                if jsonl_file:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "jsonl_line": line_idx,
                                "question": question_text,
                                "answer_letter": None,
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if jsonl_file:
            jsonl_file.close()
            logger.info(f"Question / answer JSONL written to {output_jsonl}")

        # Aggregate report
        logger.info("\n" + "=" * 70)
        logger.info("Video Evaluation Results:")
        logger.info(f"  Videos processed:    {num_succeeded}/{len(runnable)}")
        if num_succeeded > 0:
            avg_frames = total_frames / num_succeeded
            avg_vision_ms = total_vision_ms / num_succeeded
            avg_fps = total_frames / (total_vision_ms / 1000) if total_vision_ms > 0 else 0
            avg_ttft = total_ttft_ms / num_succeeded
            avg_decode_ms = total_decode_ms / total_tokens * 1000 if total_tokens > 0 else 0
            avg_tok_s = total_tokens / (total_decode_ms / 1000) if total_decode_ms > 0 else 0

            logger.info(f"  Avg frames/video:    {avg_frames:.1f}")
            logger.info(f"  Avg vision time:     {avg_vision_ms:.0f}ms ({avg_fps:.2f} frames/sec)")
            logger.info(f"  Avg TTFT:            {avg_ttft:.0f}ms")
            logger.info(f"  Avg decode:          {avg_decode_ms/1000:.2f}ms/token ({avg_tok_s:.1f} tok/s)")
        logger.info("=" * 70)

        return results

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")


def main():
    parser = argparse.ArgumentParser(description="Molmo2-8B Video Evaluation")
    parser.add_argument(
        "--test-jsonl",
        type=str,
        default=str(DEFAULT_TEST_JSONL),
        help="Path to test.jsonl file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory to cache downloaded videos",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="First N JSONL rows to load (default: all rows)",
    )
    parser.add_argument(
        "--unique-video-urls-only",
        action="store_true",
        help="One run per unique video URL with global --max-tokens (default: one run per JSONL line, per-row max_tokens).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max tokens when --unique-video-urls-only; ignored in exact per-line mode (uses each row's max_tokens).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="Maximum sequence length for KV cache",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=8,
        help="Maximum frames to extract per video",
    )
    parser.add_argument(
        "--max-video-fps",
        type=float,
        default=None,
        help="HF max_fps cap for sampling (default: omit; HF uses processor default, often 2.0)",
    )
    parser.add_argument(
        "--native-video-fps",
        action="store_true",
        help="Do not pass max_fps (same as omitting --max-video-fps when you want HF default).",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Optional HF fps kwarg.",
    )
    parser.add_argument(
        "--video-sampling-fps",
        type=float,
        default=None,
        help="Optional HF sampling_fps kwarg.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of model layers (default: 36)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading videos (assume they are already cached)",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable tracing for prefill",
    )
    parser.add_argument(
        "--use-decode-trace",
        action="store_true",
        help="Enable tracing for decode",
    )
    parser.add_argument(
        "--use-vision-trace",
        action="store_true",
        help="Enable tracing for vision backbone",
    )
    parser.add_argument(
        "--use-unified-trace",
        action="store_true",
        help="Enable unified Vision+Prefill trace",
    )
    parser.add_argument(
        "--use-dp-vision-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture DP ViT + pool-chunk traces before eval (recommended; avoids slow eager vision). "
            "Use --no-use-dp-vision-trace to force the eager vision path for debugging."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save per-video results to JSON file",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Save one JSON object per line: question, answer_letter, jsonl_line (and error if failed)",
    )

    args = parser.parse_args()

    results = run_eval(
        test_jsonl=args.test_jsonl,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        max_frames=args.max_video_frames,
        max_fps=None if args.native_video_fps else args.max_video_fps,
        video_fps=args.video_fps,
        video_sampling_fps=args.video_sampling_fps,
        num_layers=args.num_layers,
        use_trace=args.use_trace,
        use_decode_trace=args.use_decode_trace,
        use_vision_trace=args.use_vision_trace,
        use_unified_trace=args.use_unified_trace,
        use_dp_vision_trace=args.use_dp_vision_trace,
        skip_download=args.skip_download,
        exact_jsonl=not args.unique_video_urls_only,
        unique_video_urls_only=args.unique_video_urls_only,
        output_jsonl=args.output_jsonl,
    )

    if results and args.output_json:
        import json as _json

        with open(args.output_json, "w") as f:
            _json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
