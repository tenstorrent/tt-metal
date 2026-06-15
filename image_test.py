#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Test client for the standalone media inference server (server.py).

Targets the standalone server's *synchronous* endpoints:
  - images: POST /image/generations  (SDXL / SD3.5 Large)        -> base64 image(s) in body
  - videos: POST /video/generations  (Wan2.2 T2V)                -> base64 PNG frames in body
  - health: GET  /health
No auth (the standalone server requires none).

Generation is followed by layered quality checks:
  Layer A — noise gate (always): flags garbage/noise output with no reference needed.
  Layer B — SSIM/MSE reproducibility gate (only with --compare <ref>).
  Layer C — CLIP image<->text similarity (opt-in --use-clip, informational; never gates).
"""

import sys
import argparse
import base64
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

try:
    import imageio.v3 as _iio

    _IMAGEIO_AVAILABLE = True
except ImportError:
    _iio = None
    _IMAGEIO_AVAILABLE = False

try:
    import imageio_ffmpeg  # noqa: F401

    _FFMPEG_AVAILABLE = True
except ImportError:
    _FFMPEG_AVAILABLE = False

try:
    # CLIP encoder is optional and may not be present in this repo; degrade gracefully.
    from utils.sdxl_accuracy_utils.clip_encoder import CLIPEncoder as _CLIPEncoder

    _CLIP_AVAILABLE = True
except Exception:
    _CLIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model aliases and defaults
# ---------------------------------------------------------------------------

# Image models served at /image/generations. Values are the server's model label
# (from /health) used for the request-vs-server consistency warning.
_IMAGE_MODEL_LABELS = {
    "sdxl": "SDXL",
    "sd35": "SD3.5 Large",
}
# Video models served at /video/generations.
_VIDEO_MODEL_LABELS = {
    "wan-t2v": "Wan2.2 T2V",
}

_SDXL_DEFAULTS = {
    "steps": 50,
    "guidance": 5.0,
    "seed": 14241,
    "negative": "cartoon, drawing, low quality, bad quality, distorted, noise, watermark, text, signature",
    "output": "output_sdxl.png",
}

_SD35_DEFAULTS = {
    "steps": 28,
    "guidance": 3.5,
    "seed": 42,
    "negative": "",
    "output": "output_sd35.png",
}

_WAN_T2V_DEFAULTS = {
    "steps": 40,
    "guidance": None,  # let the server use its configured two-stage guidance scales
    "seed": 42,
    "negative": "",
    "output": "output_wan_t2v.mp4",
    "fps": 16,  # matches models/tt_dit/tests/models/wan2_2/test_performance_wan.py
}

_DEFAULTS_BY_ALIAS = {
    "sdxl": _SDXL_DEFAULTS,
    "sd35": _SD35_DEFAULTS,
    "wan-t2v": _WAN_T2V_DEFAULTS,
}

_VIDEO_FRAME_SAMPLE_RATE = 8  # sample every Nth reference frame for comparison


# ---------------------------------------------------------------------------
# Image comparison (inlined — no cross-repo import)
# ---------------------------------------------------------------------------

def _calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size)
        arr2 = np.array(img2).astype(np.float32)
    return float(np.mean((arr1 - arr2) ** 2))


def _calculate_ssim(img1: Image.Image, img2: Image.Image) -> float:
    try:
        from skimage.metrics import structural_similarity as skimage_ssim

        arr1 = np.array(img1.convert("L"))
        arr2 = np.array(img2.convert("L"))
        if arr1.shape != arr2.shape:
            img2 = img2.resize(img1.size)
            arr2 = np.array(img2.convert("L"))
        return float(skimage_ssim(arr1, arr2))
    except ImportError:
        # Simplified global SSIM fallback (no scipy/skimage dependency).
        arr1 = np.array(img1.convert("L")).astype(np.float64)
        arr2 = np.array(img2.convert("L")).astype(np.float64)
        if arr1.shape != arr2.shape:
            img2 = img2.resize(img1.size)
            arr2 = np.array(img2.convert("L")).astype(np.float64)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu1, mu2 = arr1.mean(), arr2.mean()
        sigma1, sigma2 = arr1.std(), arr2.std()
        sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
        return float(numerator / denominator)


def compare_images(img1: Image.Image, img2: Image.Image, ssim_threshold: float = 0.85) -> dict:
    mse = _calculate_mse(img1, img2)
    ssim_score = _calculate_ssim(img1, img2)
    return {"mse": mse, "ssim": ssim_score, "similar": ssim_score >= ssim_threshold}


# ---------------------------------------------------------------------------
# Noise gate (Layer A) — numpy + PIL only, no new deps
# Thresholds tuned for ~1024x1024 SDXL output; adjust _NOISE_* if needed.
# ---------------------------------------------------------------------------

_NOISE_ENTROPY_MAX = 7.9   # bits; random noise ~7.95, natural images ~6.0-7.6
_NOISE_STDDEV_MAX = 72.0   # per-channel; random noise ~73, natural images ~40-70
_NOISE_AUTOCORR_MIN = 0.5  # Pearson r with 1-px shift; noise <0.05, natural >0.85


def _image_entropy(img: Image.Image) -> float:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    counts, _ = np.histogram(luma.ravel(), bins=256, range=(0, 256))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _channel_stddev(img: Image.Image) -> tuple:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    return (float(arr[:, :, 0].std()), float(arr[:, :, 1].std()), float(arr[:, :, 2].std()))


def _spatial_autocorrelation(img: Image.Image) -> float:
    arr = np.array(img.convert("L")).astype(np.float64)
    flat = arr[:, :-1].ravel()
    shifted = arr[:, 1:].ravel()
    if flat.std() == 0 or shifted.std() == 0:
        return 0.0
    return float(np.corrcoef(flat, shifted)[0, 1])


def noise_gate_check(img: Image.Image) -> tuple:
    """Return (passed, metrics_dict). Fails on garbage/noise output."""
    entropy = _image_entropy(img)
    stddevs = _channel_stddev(img)
    autocorr = _spatial_autocorrelation(img)
    metrics = {"entropy": entropy, "stddev": stddevs, "autocorr": autocorr}
    noise_by_stats = entropy > _NOISE_ENTROPY_MAX and any(s > _NOISE_STDDEV_MAX for s in stddevs)
    noise_by_autocorr = autocorr < _NOISE_AUTOCORR_MIN
    passed = not (noise_by_stats or noise_by_autocorr)
    return passed, metrics


# ---------------------------------------------------------------------------
# Optional CLIP layer (Layer C) — informational only, never gates exit code
# ---------------------------------------------------------------------------

def clip_similarity_check(img: Image.Image, ref_img: Image.Image, prompt: str):
    if not _CLIP_AVAILABLE:
        print("  CLIP: unavailable (clip_encoder not installed) — skipping")
        return None
    try:
        encoder = _CLIPEncoder()
        return {
            "img_to_text": float(encoder.get_clip_score(prompt, img).mean()),
            "ref_to_text": float(encoder.get_clip_score(prompt, ref_img).mean()),
        }
    except Exception as e:
        print(f"  CLIP: warning — {e}")
        return None


# ---------------------------------------------------------------------------
# Comparison dispatchers
# ---------------------------------------------------------------------------

def run_comparison_layers(image: Image.Image, ref_image, prompt: str, args) -> int:
    """Layered checks on a single image. Returns 0 (pass) or 1 (fail)."""
    threshold = args.ssim_threshold

    # Layer A — noise gate (always)
    passed, m = noise_gate_check(image)
    print(f"  Noise gate:   entropy={m['entropy']:.3f}  "
          f"stddev=({m['stddev'][0]:.1f},{m['stddev'][1]:.1f},{m['stddev'][2]:.1f})  "
          f"autocorr={m['autocorr']:.3f}")
    if not passed:
        print("FAIL: output image appears to be noise/garbage (Layer A noise gate)")
        return 1

    # Layer B — SSIM/MSE reproducibility gate (only with reference)
    if ref_image is not None:
        metrics = compare_images(image, ref_image, ssim_threshold=threshold)
        print(f"  MSE:          {metrics['mse']:.2f}")
        print(f"  SSIM:         {metrics['ssim']:.4f}  (threshold: {threshold})")
        if not metrics["similar"]:
            print(f"FAIL: images differ significantly (SSIM {metrics['ssim']:.4f} < {threshold})")
            return 1
        print(f"PASS: images are similar (SSIM {metrics['ssim']:.4f} >= {threshold})")

        # Layer C — CLIP (informational, requires reference)
        if args.use_clip:
            clip = clip_similarity_check(image, ref_image, prompt)
            if clip:
                print(f"  CLIP img→text: {clip['img_to_text']:.4f}  ref→text: {clip['ref_to_text']:.4f}")

    return 0


def _extract_ref_frames(video_path, frame_sample_rate=_VIDEO_FRAME_SAMPLE_RATE) -> list:
    """Return sampled PIL Images from a reference video file. Empty if unreadable."""
    if not _IMAGEIO_AVAILABLE:
        print("  Warning: imageio not installed — reference frame extraction skipped")
        return []
    try:
        frames = []
        for i, frame in enumerate(_iio.imiter(str(video_path))):
            if i % frame_sample_rate == 0:
                frames.append(Image.fromarray(frame))
        return frames
    except Exception as e:
        print(f"  Warning: could not read reference video ({e}) — skipping Layer B")
        return []


def run_video_comparison_layers(frames: list, ref_path, prompt: str, args) -> int:
    """Layered checks on generated video frames (PIL list). Returns 0/1.

    Noise-gate thresholds are image-tuned; Wan T2V dimensions/codec may shift
    per-frame metrics, so Layer A uses a >50% majority rule as a safety margin.
    """
    if not frames:
        print("  Warning: no frames to check")
        return 0

    # Layer A — noise gate (majority rule: >50% must pass)
    failed = sum(1 for f in frames if not noise_gate_check(f)[0])
    ratio = (len(frames) - failed) / len(frames)
    print(f"  Noise gate: {len(frames) - failed}/{len(frames)} frames passed ({ratio:.0%})")
    if ratio < 0.5:
        print("FAIL: majority of video frames failed noise gate (Layer A)")
        return 1

    # Layer B — mean SSIM across paired frames (only with reference)
    if ref_path is not None:
        ref_frames = _extract_ref_frames(ref_path)
        if ref_frames:
            n = min(len(frames), len(ref_frames))
            scores = [compare_images(frames[i], ref_frames[i])["ssim"] for i in range(n)]
            mean_ssim = float(np.mean(scores))
            print(f"  Mean SSIM ({n} frame pairs): {mean_ssim:.4f}  (threshold: {args.ssim_threshold})")
            if mean_ssim < args.ssim_threshold:
                print(f"FAIL: mean frame SSIM {mean_ssim:.4f} < {args.ssim_threshold} (Layer B)")
                return 1
            print(f"PASS: mean frame SSIM {mean_ssim:.4f} >= {args.ssim_threshold}")

            # Layer C — CLIP (informational, first 5 frames)
            if args.use_clip:
                clip_scores = [clip_similarity_check(f, ref_frames[0], prompt) for f in frames[:5]]
                valid = [c["img_to_text"] for c in clip_scores if c]
                if valid:
                    print(f"  CLIP mean img→text (sampled): {float(np.mean(valid)):.4f}")

    return 0


# ---------------------------------------------------------------------------
# Video writing (adaptive: MP4 if ffmpeg available, else animated GIF)
# ---------------------------------------------------------------------------

def _write_video(frames: list, output_path: Path, fps: int) -> Path:
    """Write PIL frames to output_path. Falls back to GIF if MP4/ffmpeg unavailable.

    Returns the path actually written (may differ from output_path on fallback).
    """
    if output_path.suffix.lower() == ".mp4" and _IMAGEIO_AVAILABLE and _FFMPEG_AVAILABLE:
        arr = np.stack([np.array(f.convert("RGB")) for f in frames]).astype(np.uint8)
        _iio.imwrite(str(output_path), arr, fps=fps, codec="libx264")
        return output_path

    # Fallback: animated GIF via PIL (always available, no ffmpeg needed).
    if output_path.suffix.lower() == ".mp4":
        gif_path = output_path.with_suffix(".gif")
        print(f"  Note: ffmpeg unavailable — writing GIF instead: {gif_path.name} "
              f"(pip install imageio-ffmpeg for MP4)")
    else:
        gif_path = output_path
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return gif_path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    model: str = "sdxl",
    server_url: str = "http://127.0.0.1:8000",
    num_inference_steps: int = None,
    guidance_scale: float = None,
    seed: int = None,
    negative_prompt: str = None,
    guidance_rescale: float = 0.0,
    prompt_2: str = None,
    negative_prompt_2: str = None,
) -> tuple:
    """POST /image/generations (synchronous) and return (PIL Image, inference_time, model_label)."""
    defaults = _DEFAULTS_BY_ALIAS.get(model, _SDXL_DEFAULTS)
    if num_inference_steps is None:
        num_inference_steps = defaults["steps"]
    if guidance_scale is None:
        guidance_scale = defaults["guidance"]
    if seed is None:
        seed = defaults["seed"]
    if negative_prompt is None:
        negative_prompt = defaults["negative"]

    print(f"Sending request to {server_url}...")
    print(f"Prompt: {prompt}")
    if model == "sdxl" and prompt_2:
        print(f"Prompt 2: {prompt_2}")

    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    # SDXL dual-encoder fields (the server ignores them for SD3.5).
    if model == "sdxl":
        request_data["guidance_rescale"] = guidance_rescale
        if prompt_2 is not None:
            request_data["prompt_2"] = prompt_2
        if negative_prompt_2 is not None:
            request_data["negative_prompt_2"] = negative_prompt_2

    response = requests.post(f"{server_url}/image/generations", json=request_data, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(f"Server error ({response.status_code}): {response.text}")

    data = response.json()
    inference_time = data["inference_time"]
    server_model = data.get("model", "unknown")
    print(f"Inference completed in {inference_time:.2f}s (server model: {server_model})")

    image = Image.open(BytesIO(base64.b64decode(data["images"][0])))
    return image, inference_time, server_model


def generate_video(
    prompt: str,
    server_url: str = "http://127.0.0.1:8000",
    num_inference_steps: int = None,
    seed: int = None,
    negative_prompt: str = None,
    guidance_scale: float = None,
    guidance_scale_2: float = None,
    timeout: int = 1800,
) -> tuple:
    """POST /video/generations (synchronous) → (frames[PIL], inference_time, model_label, width, height).

    The standalone server returns base64 PNG frames inline (no async job/poll/download).
    num_frames/height/width are fixed at pipeline creation, so they are not sent.
    """
    defaults = _WAN_T2V_DEFAULTS
    if num_inference_steps is None:
        num_inference_steps = defaults["steps"]
    if seed is None:
        seed = defaults["seed"]
    if negative_prompt is None:
        negative_prompt = defaults["negative"]

    print(f"Sending video request to {server_url}...")
    print(f"Prompt: {prompt}")

    payload = {"prompt": prompt, "num_inference_steps": num_inference_steps}
    if seed is not None:
        payload["seed"] = seed
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if guidance_scale_2 is not None:
        payload["guidance_scale_2"] = guidance_scale_2

    response = requests.post(f"{server_url}/video/generations", json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Server error ({response.status_code}): {response.text}")

    data = response.json()
    inference_time = data.get("inference_time")
    server_model = data.get("model", "unknown")
    width, height = data.get("width"), data.get("height")
    frames = [Image.open(BytesIO(base64.b64decode(b))) for b in data["frames"]]
    for f in frames:  # force-load so the underlying BytesIO can be released
        f.load()

    if inference_time is not None:
        print(f"Inference completed in {inference_time:.2f}s (server model: {server_model}, "
              f"{len(frames)} frames @ {width}x{height})")
    return frames, inference_time, server_model, width, height


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test image/video generation against the standalone media inference server"
    )
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=list(_IMAGE_MODEL_LABELS.keys()) + list(_VIDEO_MODEL_LABELS.keys()),
        help="Model: 'sdxl' (default), 'sd35', or 'wan-t2v' (video)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: output_<model>.<ext>)")
    parser.add_argument("--compare", type=str, help="Reference image/video for SSIM comparison (Layer B)")
    parser.add_argument("--no-compare", action="store_true", help="Skip reference comparison (Layer A only)")
    parser.add_argument("--ssim-threshold", type=float, default=0.85, help="SSIM pass/fail threshold (default: 0.85)")
    parser.add_argument("--use-clip", action="store_true",
                        help="Log CLIP image→text similarity (informational, requires reference)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt (model default if unset)")
    # Video-only
    parser.add_argument("--guidance2", type=float, default=None, help="[wan-t2v] second-stage guidance scale")
    parser.add_argument("--fps", type=int, default=_WAN_T2V_DEFAULTS["fps"], help="[wan-t2v] output frame rate")
    # SDXL-only (ignored when --model sd35)
    parser.add_argument("--rescale", type=float, default=0.0, help="[SDXL] guidance rescale factor (0.0-1.0)")
    parser.add_argument("--prompt2", type=str, help="[SDXL] secondary prompt for second text encoder")
    parser.add_argument("--negative2", type=str, help="[SDXL] negative prompt for second text encoder")

    args = parser.parse_args()
    is_video = args.model in _VIDEO_MODEL_LABELS

    defaults = _DEFAULTS_BY_ALIAS[args.model]
    output_file = args.output if args.output is not None else defaults["output"]

    # -----------------------------------------------------------------------
    # Health check (/health) + request-vs-server model consistency warning
    # -----------------------------------------------------------------------
    try:
        health = requests.get(f"{args.server}/health", timeout=5)
        if health.status_code == 200:
            hd = health.json()
            server_model = hd.get("model", "unknown")
            print(f"Server status: {hd.get('status')}")
            print(f"Server model:  {server_model}")
            print(f"Workers: {hd.get('workers_alive')}/{hd.get('workers_total')}")
            expected = {**_IMAGE_MODEL_LABELS, **_VIDEO_MODEL_LABELS}.get(args.model, args.model)
            if server_model != expected:
                print(f"WARNING: --model {args.model} expects server model '{expected}' "
                      f"but server reports '{server_model}'. Results may be unexpected.")
        else:
            print("Warning: Server health check returned non-200 status")
    except Exception as e:
        print(f"Error: Cannot connect to server at {args.server}: {e}")
        print(f"Make sure the server is running: ./launch_server.sh --model "
              f"{'wan22 --board <board>' if is_video else '<sdxl|sd35>'}")
        sys.exit(1)

    print("")
    if is_video:
        # ----- Video path -----
        try:
            frames, inference_time, _model, _w, _h = generate_video(
                prompt=args.prompt,
                server_url=args.server,
                num_inference_steps=args.steps,
                seed=args.seed,
                negative_prompt=args.negative,
                guidance_scale=args.guidance,
                guidance_scale_2=args.guidance2,
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        written = _write_video(frames, Path(output_file), args.fps)
        print(f"Video saved to: {written}")

        if args.no_compare:
            ref_path = None
            print("Reference comparison skipped (--no-compare) — Layer A only")
        elif args.compare:
            ref_path = Path(args.compare)
            if not ref_path.exists():
                print(f"Warning: Reference video not found: {ref_path}")
                ref_path = None
            else:
                print(f"\nComparing with reference: {ref_path}")
        else:
            ref_path = None

        sys.exit(run_video_comparison_layers(frames, ref_path, args.prompt, args))

    else:
        # ----- Image path -----
        try:
            image, inference_time, _model = generate_image(
                prompt=args.prompt,
                model=args.model,
                server_url=args.server,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                negative_prompt=args.negative,
                guidance_rescale=args.rescale,
                prompt_2=args.prompt2,
                negative_prompt_2=args.negative2,
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        output_path = Path(output_file)
        image.save(output_path)
        print(f"Image saved to: {output_path}")

        if args.no_compare:
            ref_image = None
            print("Reference comparison skipped (--no-compare) — Layer A only")
        elif args.compare:
            ref_path = Path(args.compare)
            if not ref_path.exists():
                print(f"Warning: Reference image not found: {ref_path}")
                ref_image = None
            else:
                print(f"\nComparing with reference: {ref_path}")
                ref_image = Image.open(ref_path)
        else:
            ref_image = None

        sys.exit(run_comparison_layers(image, ref_image, args.prompt, args))


if __name__ == "__main__":
    main()
