# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP coverage for the image server, with pinned published reference images."""

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.experimental.qwen_image_2_1.common.config import HF_REVISION, PROMPT_DEMO

SOURCE_REVISION = "d8befe24901fbdbf0feeb13964b6f175989f5758"
ASSETS = {
    "reference_rtx5090.png": "0018e3043106f231475b576e4f88649fae0fa059ba34bfadda255bda74f8e9d1",
    "edit_demo_p150.png": "2506929fb423c4143395c3f48cd62a893e1cb324a0abd5f6e6bbc1b9e3d0ba8b",
    "edit_ref_llama.jpg": "acb2dba9fe4966197f18c16f166dcfb9c717b974b6fa08c2004c111d8fc17b37",
    "edit_ref_gongnyang.jpg": "405548c6515da0d525a225669582fe07a184da1315d388bbba7928f879e3f8ed",
}
EDIT_PROMPT = (
    'Character 1 - "White furry llama with sunglasses", Character 2 - "Cat with glasses wearing a hoodie". '
    "These two characters are jumping, hooraying together. Happy"
)


def request(path, body=None):
    url = os.environ.get("QWEN_IMAGE_SERVER_URL", "http://127.0.0.1:20000") + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


@pytest.fixture(scope="module")
def assets():
    root = Path(os.environ["QWEN_IMAGE_FIXTURES"])
    for name, digest in ASSETS.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert request("/health")["status"] == "ok"
    assert request("/info")["weights_revision"] == HF_REVISION
    return root


def decoded_image(response):
    image = Image.open(io.BytesIO(base64.b64decode(response["image"], validate=True))).convert("RGB")
    assert image.size == (response["width"], response["height"])
    return image


def pixel_pcc(image, reference):
    assert image.size == reference.size, "reference dimensions must match; resizing can hide a shape error"
    arrays = []
    for value in (image, reference):
        rgba = value.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        arrays.append(np.asarray(rgb, dtype=np.float32).ravel())
    pcc = float(np.corrcoef(*arrays)[0, 1])
    assert np.isfinite(pcc)
    return pcc


@pytest.mark.parametrize("case", ["text", "edit"])
def test_published_reference(case, assets, record_property):
    body = {"prompt": PROMPT_DEMO, "seed": 42, "num_steps": 40, "return_rgba": True}
    reference_name = "reference_rtx5090.png"
    if case == "edit":
        body["prompt"] = EDIT_PROMPT
        body["images"] = [
            base64.b64encode((assets / name).read_bytes()).decode()
            for name in ("edit_ref_llama.jpg", "edit_ref_gongnyang.jpg")
        ]
        reference_name = "edit_demo_p150.png"
    started = datetime.now(timezone.utc).isoformat()
    first = request("/predict", body)
    timings = []
    for _ in range(3):
        t0 = time.monotonic()
        response = request("/predict", body)
        elapsed = time.monotonic() - t0
        assert response["image"] == first["image"], "repeated request changed pixels"
        assert response["image_rgba"] == first["image_rgba"]
        assert response["seed"] == 42 and response["num_steps"] == 40
        timings.append({"wall_s": elapsed, "server_ms": response["timing_ms"]})
    actual = decoded_image(response)
    reference = Image.open(assets / reference_name)
    score = pixel_pcc(actual, reference)
    out = Path(os.environ["QWEN_IMAGE_RESULTS"])
    out.mkdir(parents=True, exist_ok=True)
    actual.save(out / f"served-{case}.png")
    summary = {
        "case": case,
        "run_start": started,
        "run_end": datetime.now(timezone.utc).isoformat(),
        "reference_kind": "independent CUDA bf16" if case == "text" else "original P150 regression",
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": HF_REVISION,
        "reference_sha256": ASSETS[reference_name],
        "image_pcc": score,
        "warmup_requests": 1,
        "measured_requests": 3,
        "mean_wall_s": sum(row["wall_s"] for row in timings) / len(timings),
        "timings": timings,
    }
    (out / f"served-{case}.json").write_text(json.dumps(summary, indent=2))
    record_property("image_pcc", score)
    print(json.dumps(summary), flush=True)
    assert score >= 0.97  # The source package's published smoke-test threshold.


def test_request_isolation_and_prompt_eviction(assets):
    prompts = [f"A red cube on a white table, labeled {index}" for index in range(5)]

    def predict(prompt, seed=42):
        return request("/predict", {"prompt": prompt, "seed": seed, "num_steps": 2})["image"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        expected = list(pool.map(predict, prompts[:2]))
    assert expected[0] != expected[1]
    for prompt in prompts[2:]:
        predict(prompt)
    assert predict(prompts[0]) == expected[0]
    assert predict(prompts[1]) == expected[1]
    assert predict(prompts[0], seed=43) != expected[0]


@pytest.mark.parametrize(
    "body,status",
    [
        ({"prompt": "test", "num_steps": 0}, 422),
        ({"prompt": "test", "images": ["not valid base64"]}, 400),
    ],
)
def test_invalid_request(body, status, assets, expect_error):
    with expect_error(urllib.error.HTTPError, str(status)) as error:
        request("/predict", body)
    assert error.value.code == status


@pytest.mark.parametrize("aspect", ["portrait", "landscape"])
def test_aspect_change_returns_to_cached_text(assets, aspect):
    body = {"prompt": PROMPT_DEMO, "seed": 42, "num_steps": 2}
    expected = request("/predict", body)
    image = Image.open(assets / "edit_ref_llama.jpg").convert("RGBA")
    box = (0, 0, image.width // 2, image.height) if aspect == "portrait" else (0, 0, image.width, image.height // 2)
    condition = image.crop(box)
    encoded = io.BytesIO()
    condition.save(encoded, format="PNG")
    edited = request(
        "/predict",
        {
            "prompt": "Keep this llama on a sunny beach",
            "seed": 42,
            "num_steps": 2,
            "images": [base64.b64encode(encoded.getvalue()).decode()],
        },
    )
    assert (edited["height"] > edited["width"]) == (aspect == "portrait")
    decoded_image(edited)
    repeated = request("/predict", body)
    assert (repeated["width"], repeated["height"]) == (1024, 1024)
    assert repeated["image"] == expected["image"]
