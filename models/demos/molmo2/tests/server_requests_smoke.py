# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-API smoke test for a running Molmo2 server (vllm-tt-plugin or tt-inference-server).

Runs the request types from the bring-up instructions — text, image URL, image as base64 data URL,
video from a local file:// path, video from an https URL — plus a two-image request, and checks status,
non-empty output and a few content expectations.

    python models/demos/molmo2/tests/server_requests_smoke.py --port 8001 \
        --image /path/photo.jpg --video /abs/path/clip.mp4 [--token $JWT]
"""

import argparse
import base64
import json
import sys
import time
import urllib.request

MODEL = "allenai/Molmo2-8B"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
VIDEO_URL = "https://storage.googleapis.com/oe-training-public/molmo2-eval-media/d02d399003cca14e2a5c822d389821e553e6b7942c5640d48c075703e533d1dc.mp4"


def chat(base, content, max_tokens=64, token=None):
    body = json.dumps(
        {"model": MODEL, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers=headers)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.load(r)
    except Exception as e:  # noqa: BLE001
        return None, f"request failed: {e}", time.time() - t
    msg = data["choices"][0]["message"]["content"]
    return data, msg, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--image", help="local image for the data-URL request")
    ap.add_argument("--video", help="absolute path of a local video for the file:// request")
    ap.add_argument("--token", help="JWT bearer token, if the server requires one")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    text = lambda s: {"type": "text", "text": s}  # noqa: E731

    cases = [("text", [text("What is 2 + 2?")], 32, ["4"])]
    cases.append(
        (
            "image-url",
            [{"type": "image_url", "image_url": {"url": IMAGE_URL}}, text("Describe this image in one sentence.")],
            64,
            ["cat"],
        )
    )
    if args.image:
        b64 = base64.b64encode(open(args.image, "rb").read()).decode()
        cases.append(
            (
                "image-data-url",
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    text("What is in this image?"),
                ],
                64,
                ["cat"],
            )
        )
        cases.append(
            (
                "two-images",
                [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    text("How many images are there and what do they show?"),
                ],
                96,
                [],
            )
        )
    if args.video:
        cases.append(
            (
                "video-file",
                [
                    {"type": "video_url", "video_url": {"url": f"file://{args.video}"}},
                    text("Describe what is happening in this video."),
                ],
                64,
                [],
            )
        )
    cases.append(
        (
            "video-https",
            [{"type": "video_url", "video_url": {"url": VIDEO_URL}}, text("What action is shown?")],
            64,
            [],
        )
    )

    failed = 0
    for name, content, max_tokens, expect in cases:
        data, msg, dt = chat(base, content, max_tokens, args.token)
        ok = data is not None and msg.strip() != "" and all(w in msg.lower() for w in expect)
        usage = (data or {}).get("usage") or {}
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:15s} {dt:6.1f}s prompt_tokens={usage.get('prompt_tokens')} -> {msg[:160]!r}"
        )
        failed += not ok
    print(f"{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
