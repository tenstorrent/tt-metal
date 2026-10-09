# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed image-decision request latency (stage 12B), batch 1, the 8 golden image rows (bucket 1024).

Per row, ``BURST_RUNS`` passes each after ``BURST_PAUSE_S`` s idle (the stage-6 ``burst`` mode; the
device runs slower back to back, doc/full_model/README.md), median reported:
- ``request``: ``TTDecider.predict_probabilities(state, question, images=[png])`` end to end
  (open image + chat template + processor, input prep + uploads, device forward, readback);
- ``phases``: the same request with a ``synchronize_device`` after each phase:
  ``processor`` (open image, chat template, processor), ``input_prep`` (3D position ids, cos/sin,
  splice index, vision host tables and every upload; ``prepare_images``), ``vision_tower``,
  ``splice``, ``text_forward`` (64 layers + head), ``readback`` (probabilities to host).
A text-only request at the same bucket (decision-golden row s01, 178 tokens -> 1024) is measured
the same way for comparison (phases ``tokenize``, ``upload``, ``text_forward``, ``readback``).

Results: ``$PPLX_DECIDER_STAGE12B_DIR/image_perf.json``.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider
from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel

WARMUP = 2
BURST_RUNS = 5
BURST_PAUSE_S = 10.0
OUT_DIR = Path(
    os.environ.get("PPLX_DECIDER_STAGE12B_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/stage12b/image_e2e")
)
IMAGE_PROMPTS = (
    Path(os.environ.get("PPLX_DECIDER_IMAGE_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/e2e"))
    / "prompts.jsonl"
)
TEXT_PROMPTS = (
    Path(
        os.environ.get("PPLX_DECIDER_DECISION_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/decisions")
    )
    / "prompts.jsonl"
)
TEXT_ROW = "s01_ticket_routing"

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})


@pytest.fixture(scope="module")
def decider(_device_module_impl):
    return TTDecider(PplxDeciderModel.from_snapshot(_device_module_impl, vision=True), AppTokenizer())


def stats(samples: list[float]) -> dict:
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
        "max_ms": max(samples) * 1e3,
        "samples_ms": [round(s * 1e3, 3) for s in samples],
    }


def burst(fn) -> list:
    for _ in range(WARMUP):
        fn()
    out = []
    for _ in range(BURST_RUNS):
        time.sleep(BURST_PAUSE_S)
        out.append(fn())
    return out


def image_request_phases(decider: TTDecider, row: dict) -> dict:
    model, device, clock = decider.model, decider.model.mesh_device, time.perf_counter
    ttnn.synchronize_device(device)
    t = [clock()]
    enc = decider.encode(row)
    count = decider.tokenizer.count(row)
    t.append(clock())
    tokens, last_index, images = model.prepare_images(enc)
    ttnn.synchronize_device(device)
    t.append(clock())
    features = model.image_features(images)
    ttnn.synchronize_device(device)
    t.append(clock())
    x = model.splice(tokens, images, features)
    ttnn.synchronize_device(device)
    t.append(clock())
    probs, logits, _ = model(tokens, last_index, count, images=images, embeds=x)
    ttnn.synchronize_device(device)
    t.append(clock())
    ttnn.to_torch(probs).reshape(-1)[:count].tolist()
    t.append(clock())
    for tensor in (tokens, probs, logits):
        ttnn.deallocate(tensor)
    images.deallocate()
    names = ("processor", "input_prep", "vision_tower", "splice", "text_forward", "readback")
    return {name: t[i + 1] - t[i] for i, name in enumerate(names)}


def text_request_phases(decider: TTDecider, row: dict) -> dict:
    model, device, clock = decider.model, decider.model.mesh_device, time.perf_counter
    ttnn.synchronize_device(device)
    t = [clock()]
    ids, count = decider.tokenizer.input_ids(row), decider.tokenizer.count(row)
    t.append(clock())
    tokens, last_index = model.upload_tokens(ids)
    ttnn.synchronize_device(device)
    t.append(clock())
    probs, logits, _ = model(tokens, last_index, count)
    ttnn.synchronize_device(device)
    t.append(clock())
    ttnn.to_torch(probs).reshape(-1)[:count].tolist()
    t.append(clock())
    for tensor in (tokens, probs, logits):
        ttnn.deallocate(tensor)
    names = ("tokenize", "upload", "text_forward", "readback")
    return {name: t[i + 1] - t[i] for i, name in enumerate(names)}


def timed_request(fn):
    def run():
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    return run


def summarize(phase_runs: list[dict]) -> dict:
    return {name: stats([r[name] for r in phase_runs]) for name in phase_runs[0]}


@pytest.mark.timeout(7200)
def test_image_request_latency(decider):
    image_rows = [json.loads(line) for line in IMAGE_PROMPTS.read_text().splitlines()]
    text_row = next(
        json.loads(l)["row"] for l in TEXT_PROMPTS.read_text().splitlines() if json.loads(l)["id"] == TEXT_ROW
    )
    results = {}
    for r in image_rows:
        row = r["row"]
        request = stats(
            burst(
                timed_request(
                    lambda: decider.predict_probabilities(row["state"], row["question"], images=row["images"])
                )
            )
        )
        phases = summarize(burst(lambda: image_request_phases(decider, row)))
        results[r["id"]] = {
            "seq_len": r["seq_len"],
            "bucket": r["bucket"],
            "patches": r["patches"],
            "image_tokens": r["image_tokens"],
            "request_burst": request,
            "phases_burst": phases,
        }
        print(
            f"PERF {r['id']:<22} S={r['seq_len']:<4} patches={r['patches']:<5} request {request['median_ms']:8.2f} ms | "
            + " ".join(f"{k} {v['median_ms']:.2f}" for k, v in phases.items())
        )
    text_request = stats(
        burst(timed_request(lambda: decider.predict_probabilities(text_row["state"], text_row["question"])))
    )
    text_phases = summarize(burst(lambda: text_request_phases(decider, text_row)))
    results[f"text_only_{TEXT_ROW}"] = {
        "seq_len": len(decider.tokenizer.input_ids(text_row)),
        "bucket": 1024,
        "request_burst": text_request,
        "phases_burst": text_phases,
    }
    print(
        f"PERF text-only {TEXT_ROW} request {text_request['median_ms']:8.2f} ms | "
        + " ".join(f"{k} {v['median_ms']:.2f}" for k, v in text_phases.items())
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "image_perf.json").write_text(
        json.dumps(
            {
                "method": (
                    f"{WARMUP} warm-ups, then median of {BURST_RUNS} passes each after {BURST_PAUSE_S} s idle; "
                    "batch 1; eager; phases with synchronize_device after each phase (request without)"
                ),
                "policy": decider.model.config.optimizations.policy.name,
                "vision_policy": decider.model.vision.config.optimizations.policy.name,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": results,
            },
            indent=2,
        )
        + "\n"
    )
