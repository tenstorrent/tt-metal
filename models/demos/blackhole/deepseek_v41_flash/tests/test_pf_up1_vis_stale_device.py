# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device regression test (against a RUNNING server, gated by DSV41_UP1_SERVER=host:port) of the stale vision-buffer bug of the one-slot (Up=1) context.

Sequence per trial: (1) an image request (Up=U context), (2) a short text request (one-slot context), (3) a 1200-token needle prompt (not eligible for the one-slot context -> text-only Up=U windows).
Before the fix the needle was lost in 24/24 trials (the Up=U vision buffers kept the image mask / embedding rows). Needs a server with DSV41_VISION=1 and DSV41_PF_UP1_SHORT=1; the greedy answer must
contain the needle code. Run: DSV41_UP1_SERVER=127.0.0.1:8000 pytest -q tests/test_pf_up1_vis_stale_device.py
"""

import json
import os
import random
import sys
import urllib.request

import pytest

SERVER = os.environ.get("DSV41_UP1_SERVER")
pytestmark = pytest.mark.skipif(not SERVER, reason="needs a running DSV4.1 server (DSV41_UP1_SERVER=host:port)")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "serving_tests"))
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
WORDS = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()


def post(path, body):
    req = urllib.request.Request(
        f"http://{SERVER}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def image_request(word):
    from client_img import card, img, txt

    content = [
        img(card(word, "square", (220, 30, 30))),
        txt("What text is written in this image? Answer with the text only."),
    ]
    return post(
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0},
    )


def short_text(i):
    return post(
        "/v1/completions", {"model": MODEL, "prompt": f"Count from {i} to {i + 10}:", "max_tokens": 4, "temperature": 0}
    )


def needle_prompt(rng, code):
    filler = lambda n: " ".join(rng.choice(WORDS) for _ in range(n))
    return (
        filler(500)
        + f"\nThe secret access code of vault alpha is {code}.\n"
        + filler(500)
        + "\nQuestion: What is the secret access code of vault alpha? Answer with the number only.\nAnswer:"
    )


@pytest.mark.parametrize("with_short_between", [True, False])
def test_needle_survives_image_then_short_then_long(with_short_between):
    rng = random.Random(7)
    lost = []
    for i in range(24):
        image_request(f"CAT {rng.randint(10, 99)}")
        if with_short_between:
            short_text(i)
        code = str(rng.randint(1000, 9999))
        out = post(
            "/v1/completions", {"model": MODEL, "prompt": needle_prompt(rng, code), "max_tokens": 12, "temperature": 0}
        )
        if code not in out["choices"][0]["text"]:
            lost.append((i, out["choices"][0]["text"]))
    assert not lost, f"needle lost in {len(lost)} of 24 trials (stale image rows?): {lost[:3]}"
