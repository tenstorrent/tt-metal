"""
Molmo2-8B Video Evaluation Script

Runs batch inference on test.jsonl video entries and reports:
  - Per-video: frames, TTFT, tok/s, predicted answer
  - Aggregate: avg frames/sec, avg TTFT, avg tok/s, total processed
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

from loguru import logger

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
    max_fps: float = 2.0,
    num_layers: Optional[int] = None,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
    skip_download: bool = False,
):
    """
    Run batch video evaluation.

    Args:
        test_jsonl: Path to test.jsonl file
        cache_dir: Directory to cache downloaded videos
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Max tokens to generate per sample
        max_seq_len: KV cache sequence length (16384 for video)
        max_frames: Max frames to extract per video
        max_fps: Max FPS for frame sampling
        num_layers: Number of model layers (None = 36)
        use_trace: Enable prefill tracing
        use_decode_trace: Enable decode tracing
        use_vision_trace: Enable vision backbone tracing
        use_unified_trace: Enable unified Vision+Prefill trace
        skip_download: Skip downloading videos (assume they are already cached)
    """
    import ttnn

    # Import demo functions here to avoid circular imports
    from models.demos.molmo2.demo.demo import (
        VIDEO_PROMPT,
        Molmo2Generator,
        create_model,
        load_model_weights,
        load_processor,
        preprocess_video_molmo2,
    )

    cache_path = Path(cache_dir)
    test_path = Path(test_jsonl)

    # Load test entries
    with open(test_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if num_samples is not None:
        entries = entries[:num_samples]

    logger.info(f"Loaded {len(entries)} test entries from {test_path}")

    # Extract unique video URLs (entries may repeat the same video)
    seen_urls = set()
    unique_entries = []
    for entry in entries:
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
            unique_entries.append({"url": url, "text": text, "entry": entry})

    logger.info(f"Unique videos: {len(unique_entries)}")

    # Download all videos first (unless skip_download)
    if not skip_download:
        logger.info("Downloading videos...")
        for item in unique_entries:
            try:
                item["local_path"] = download_video(item["url"], cache_path)
            except Exception as e:
                logger.warning(f"Skipping {item['url']}: {e}")
                item["local_path"] = None
    else:
        for item in unique_entries:
            url_path = item["url"].split("?")[0]
            filename = url_path.rstrip("/").split("/")[-1]
            item["local_path"] = str(cache_path / filename)

    # Filter entries that have local paths
    runnable = [e for e in unique_entries if e.get("local_path") and Path(e["local_path"]).exists()]
    logger.info(f"Videos available for inference: {len(runnable)}")

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
        model = create_model(device, state_dict, num_layers, max_seq_len=max_seq_len)
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=max_seq_len,
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

        for idx, item in enumerate(runnable):
            video_path = item["local_path"]
            question_text = item["text"]
            url = item["url"]

            logger.info(f"\n[{idx+1}/{len(runnable)}] {Path(video_path).name}")
            logger.info(f"  Question: {question_text[:100]}...")

            try:
                # Preprocess video
                t0 = time.perf_counter()
                video_inputs = preprocess_video_molmo2(
                    video_path,
                    max_frames=max_frames,
                    max_fps=max_fps,
                )
                extract_ms = (time.perf_counter() - t0) * 1000
                n_frames = video_inputs["n_frames"]

                # Build prompt with video token + question
                prompt = f"{VIDEO_PROMPT} {question_text}"

                # Reset KV cache for new video
                generator.init_kv_cache()
                generator.reset_kv_cache(start_pos=0)

                # Run inference
                response, perf = generator.run_video_inference(
                    video_inputs=video_inputs,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    use_trace=use_trace,
                    use_decode_trace=use_decode_trace,
                    use_vision_trace=use_vision_trace,
                    use_unified_trace=use_unified_trace,
                )

                predicted = extract_answer_letter(response)
                _, choices = extract_question_and_choices(question_text)

                # Accumulate metrics
                total_frames += n_frames
                total_vision_ms += perf["vision_ms"]
                total_ttft_ms += perf["ttft_ms"]
                total_decode_ms += perf["total_decode_ms"]
                total_tokens += perf["generated_tokens"]
                num_succeeded += 1

                result = {
                    "idx": idx,
                    "url": url,
                    "n_frames": n_frames,
                    "extract_ms": extract_ms,
                    "vision_ms": perf["vision_ms"],
                    "frames_per_sec": perf["frames_per_sec"],
                    "ttft_ms": perf["ttft_ms"],
                    "total_decode_ms": perf["total_decode_ms"],
                    "tokens_per_sec": perf["tokens_per_sec"],
                    "input_tokens": perf["input_tokens"],
                    "generated_tokens": perf["generated_tokens"],
                    "response": response,
                    "predicted_answer": predicted,
                    "choices": choices,
                }
                results.append(result)

                logger.info(
                    f"  Frames: {n_frames} | Vision: {perf['vision_ms']:.0f}ms "
                    f"({perf['frames_per_sec']:.1f} fps) | TTFT: {perf['ttft_ms']:.0f}ms | "
                    f"Decode: {perf['tokens_per_sec']:.1f} tok/s | "
                    f"Answer: {predicted}"
                )

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results.append({"idx": idx, "url": url, "error": str(e)})

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
        help="Number of samples to evaluate (default: all unique videos)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per sample",
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
        default=2.0,
        help="Maximum FPS for frame sampling",
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
        "--output-json",
        type=str,
        default=None,
        help="Save per-video results to JSON file",
    )

    args = parser.parse_args()

    results = run_eval(
        test_jsonl=args.test_jsonl,
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        max_frames=args.max_video_frames,
        max_fps=args.max_video_fps,
        num_layers=args.num_layers,
        use_trace=args.use_trace,
        use_decode_trace=args.use_decode_trace,
        use_vision_trace=args.use_vision_trace,
        use_unified_trace=args.use_unified_trace,
        skip_download=args.skip_download,
    )

    if results and args.output_json:
        import json as _json

        with open(args.output_json, "w") as f:
            _json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
