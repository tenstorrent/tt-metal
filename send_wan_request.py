# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
send_wan_request.py — minimal client for the WAN 2.2 text-to-video server.

Usage
-----
  # Hardcoded example prompt (no args):
  python send_wan_request.py

  # Custom prompt:
  python send_wan_request.py "A cat surfing on ocean waves at sunset"

  # Full options:
  python send_wan_request.py "A cat surfing" \
      --host 127.0.0.1 \
      --port 8000 \
      --output my_video.mp4 \
      --seed 42 \
      --steps 20

Prerequisites
-------------
  pip install requests
  ./launch_server.sh --model wan   # wait for "All workers ready" in logs

Notes
-----
  - The server encodes the MP4 as base64; this script decodes and writes it to disk.
  - Default timeout matches the server's inference_timeout_seconds (900 s / 15 min).
  - /health is checked before submitting; the script exits early if the server is
    not yet ready or is running a different model (e.g. sdxl / sd35).
"""

import argparse
import base64
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Defaults — mirror wan_config.py values so the script works out of the box
# ---------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_OUTPUT = "output_wan.mp4"
DEFAULT_STEPS = 20
REQUEST_TIMEOUT_S = 960  # 900 s server timeout + 60 s network headroom

EXAMPLE_PROMPT = "A golden retriever runs through a sunlit autumn forest, leaves swirling in slow motion"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_health(base_url: str) -> None:
    """GET /health and exit with a clear message if the server is not ready."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: cannot reach server at {base_url}  (connection refused).")
        print("Make sure you have run:  ./launch_server.sh --model wan")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"Error: health check failed — {exc}")
        sys.exit(1)

    data = resp.json()
    status = data.get("status", "unknown")
    model = data.get("model", "unknown")
    workers_alive = data.get("workers_alive", 0)

    if status != "healthy" or workers_alive == 0:
        print(f"Error: server is not healthy (status={status!r}, workers_alive={workers_alive}).")
        print("Wait until logs show: 'All workers ready. WAN 2.2 server is accepting requests.'")
        sys.exit(1)

    if model != "WAN 2.2":
        print(f"Error: server is running model '{model}', not 'WAN 2.2'.")
        print("Restart with:  ./launch_server.sh --model wan")
        sys.exit(1)

    print(f"Server ready  (model={model!r}, workers_alive={workers_alive})")


def build_payload(args: argparse.Namespace) -> dict:
    payload: dict = {
        "prompt": args.prompt,
        "negative_prompt": "",
        "num_inference_steps": args.steps,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    return payload


def send_request(base_url: str, payload: dict) -> dict:
    """POST to /video/generations and return the parsed JSON response."""
    print(f"Sending request to {base_url}/video/generations ...")
    print(f"  prompt : {payload['prompt'][:80]}")
    print(f"  steps  : {payload['num_inference_steps']}")
    if "seed" in payload:
        print(f"  seed   : {payload['seed']}")

    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/video/generations",
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"Error: request timed out after {REQUEST_TIMEOUT_S} s.")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"Error: HTTP {exc.response.status_code} — {exc.response.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"Response received in {elapsed:.1f} s (round-trip)")
    return resp.json()


def save_video(data: dict, output_path: str) -> None:
    """Decode the base64 MP4 from the response and write it to disk."""
    encoded = data.get("video", "")
    if not encoded:
        print("Error: response contained no 'video' field.")
        sys.exit(1)

    mp4_bytes = base64.b64decode(encoded)
    with open(output_path, "wb") as fh:
        fh.write(mp4_bytes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a text-to-video request to the WAN 2.2 inference server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=EXAMPLE_PROMPT,
        help="Text prompt for video generation (default: built-in example)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output MP4 path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_STEPS})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    check_health(base_url)
    payload = build_payload(args)
    data = send_request(base_url, payload)

    save_video(data, args.output)

    inference_time = data.get("inference_time", 0.0)
    num_frames = data.get("num_frames", "?")
    fps = data.get("fps", "?")
    model = data.get("model", "?")

    print()
    print("Done.")
    print(f"  model          : {model}")
    print(f"  inference time : {inference_time:.1f} s")
    print(f"  frames / fps   : {num_frames} @ {fps} fps")
    print(f"  output file    : {args.output}")


if __name__ == "__main__":
    main()
