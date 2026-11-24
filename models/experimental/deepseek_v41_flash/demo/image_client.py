# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Image prompts against ``server.py``: synthetic images whose answer is known, checked in the replies.

Each case sends one chat request (non-streaming, chat mode) with images of a large digit on a plain
background, inline as base64 ``image_url`` parts, and checks that the reply names the digits:

* ``one``: one image and the question.
* ``text-first``: ``--text-tokens`` of text, then the image and the question. With a prefill server that takes
  images the whole prompt but its tail is the server's; with one that does not (or the persistent stage's own
  prefill), the text before the image is the prefill model's and decode replays the image (server.py's images item).
* ``image-then-text``: the image, then ``--text-tokens`` of text, then the question: with a prefill server that takes
  images, the image's KV is all the server's.
* ``two``: two images, read left to right.
* ``follow-up``: the ``one`` conversation continued with a text-only turn about the same image.
* ``sizes``: small, wide and tall images (other spans: 184 tokens and up).

Needs no device or ttnn: PIL and the standard library only.

    python models/experimental/deepseek_v41_flash/demo/image_client.py --url http://$DECODE_IP:$DECODE_PORT
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

QUESTION = "What digit is written in the image? Reply with the digit only."
TWO_QUESTION = "Each image shows one digit. Reply with the two digits in order, separated by a space, and nothing else."
FOLLOW_UP = "Repeat that digit, then add one to it. Reply with the two numbers separated by a space, and nothing else."
FILLER = (
    "The committee met on Tuesday to review the quarterly maintenance schedule for the river locks. "
    "Each lock is inspected in turn, its gates greased, its seals checked and its logs updated. "
)


def digit_png(digit: str, size=(448, 448), background=(200, 30, 30), foreground=(255, 255, 255)) -> bytes:
    """A PNG of ``digit``, three quarters of the image's shorter side tall, centred on a plain background."""
    w, h = size
    image = Image.new("RGB", size, background)
    font = ImageFont.load_default(size=min(w, h) * 3 // 4)
    ImageDraw.Draw(image).text((w / 2, h / 2), digit, fill=foreground, font=font, anchor="mm")
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def chat(url: str, messages: list, max_tokens: int, timeout: float) -> tuple[str, dict, float]:
    """``(reply text, usage, seconds)`` of one non-streaming chat request."""
    # No temperature: a server that samples on the device (--persistent-moe) refuses one; it decodes greedily there.
    body = {"messages": messages, "max_tokens": max_tokens, "thinking": False}
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    seconds = time.perf_counter() - t0
    return reply["choices"][0]["message"].get("content") or "", reply.get("usage", {}), seconds


def digits_of(text: str) -> list[str]:
    return [c for c in text if c.isdigit()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--text-tokens", type=int, default=2000, help="about how much text precedes the image in text-first")
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--cases",
        default="one,text-first,image-then-text,two,follow-up,sizes",
        help="comma-separated cases to run (see above)",
    )
    args = p.parse_args(argv)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    results = []

    def check(name: str, messages: list, expect: list[str]) -> str:
        try:
            text, usage, seconds = chat(args.url, messages, args.max_tokens, args.timeout)
        except Exception as e:  # noqa: BLE001 - reported as a failed case
            results.append((name, False))
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            return ""
        ok = digits_of(text)[: len(expect)] == expect
        results.append((name, ok))
        print(
            f"{'PASS' if ok else 'FAIL'} {name}: expected {' '.join(expect)}, got {text.strip()!r} "
            f"({usage.get('prompt_tokens', '?')} prompt tokens, {seconds:.2f} s)",
            flush=True,
        )
        return text

    if "one" in cases or "follow-up" in cases:
        first = [{"role": "user", "content": [image_part(digit_png("7")), {"type": "text", "text": QUESTION}]}]
        reply = check("one", first, ["7"])
        if "follow-up" in cases and reply:
            messages = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": FOLLOW_UP}]
            check("follow-up", messages, ["7", "8"])
    if "text-first" in cases:
        filler = FILLER * max(1, args.text_tokens // 40)
        content = [
            {"type": "text", "text": "Read this note, then answer the question about the image after it.\n\n" + filler},
            image_part(digit_png("4", background=(30, 60, 200), foreground=(255, 220, 0))),
            {"type": "text", "text": QUESTION},
        ]
        check("text-first", [{"role": "user", "content": content}], ["4"])
    if "image-then-text" in cases:
        filler = FILLER * max(1, args.text_tokens // 40)
        content = [
            image_part(digit_png("8", background=(150, 90, 20))),
            {"type": "text", "text": "Remember the image above. Now read this note.\n\n" + filler + "\n\n" + QUESTION},
        ]
        check("image-then-text", [{"role": "user", "content": content}], ["8"])
    if "two" in cases:
        content = [
            image_part(digit_png("2", background=(20, 120, 40))),
            image_part(digit_png("9", background=(90, 20, 120))),
            {"type": "text", "text": TWO_QUESTION},
        ]
        check("two", [{"role": "user", "content": content}], ["2", "9"])
    if "sizes" in cases:
        for digit, size in (("5", (96, 96)), ("6", (1600, 500)), ("3", (500, 1400))):
            content = [image_part(digit_png(digit, size=size)), {"type": "text", "text": QUESTION}]
            check(f"sizes {size[0]}x{size[1]}", [{"role": "user", "content": content}], [digit])

    passed = sum(ok for _, ok in results)
    print(f"{passed}/{len(results)} passed")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
