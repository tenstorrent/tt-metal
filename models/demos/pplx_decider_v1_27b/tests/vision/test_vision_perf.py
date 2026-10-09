# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision tower latency per patch bucket and device DRAM, alone and next to the full text model.

- ``test_tower_latency``: per bucket one golden image; ``WARMUP`` untimed passes, then ``TIMED`` passes,
  each ``synchronize -> t0 -> tower(inputs) -> synchronize -> t1`` (host dispatch + device execution of
  patch embed, 27 blocks, merger and the output slice). ``prepare_inputs`` (host tables + upload) is
  timed separately. Also the DRAM the vision weights take (allocator view before / after the load).
- ``test_dram_with_text_model``: the full 64-layer text model (default policy = stage-8 C0, from the
  disk weight cache) and the vision tower resident together; one 1024-bucket tower forward and one
  8192-bucket text forward run with both resident; allocator view after each step.

Results: ``$PPLX_DECIDER_STAGE12A_DIR/vision_perf.json`` and ``vision_dram_with_text.json``.
"""

from __future__ import annotations

import json
import statistics
import time

import pytest

import ttnn
from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import (
    DEVICE_PARAMS,
    STAGE_DIR,
    build_tower,
    golden_inputs,
    prepare,
)
from models.demos.pplx_decider_v1_27b.tt.model import dram_view

WARMUP = 2
TIMED = 7
# bucket -> golden image (patches): exact-bucket and padded cases.
BUCKET_IMAGES = {
    256: "v01_dominant_color",  # 256
    512: "v05_red_circle_yes",  # 320
    768: "v03_receipt_total",  # 748
    1024: "v04_tallest_bar",  # 1024
}
PADDED_1024 = "v02_count_circles"  # 936 in 1024

pytestmark = pytest.mark.use_module_device(DEVICE_PARAMS)


def _save(name: str, payload) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STAGE_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")


def _stats(samples):
    ms = [s * 1e3 for s in samples]
    return dict(median_ms=statistics.median(ms), min_ms=min(ms), max_ms=max(ms), runs=len(ms), samples_ms=ms)


@pytest.mark.timeout(1800)
def test_tower_latency(device):
    before = dram_view(device)
    t0 = time.perf_counter()
    tower = build_tower(device)
    load_s = time.perf_counter() - t0
    after = dram_view(device)
    weights_gib = after["allocated_gib"] - before["allocated_gib"]
    results = {
        "load_s": load_s,
        "dram_empty": before,
        "dram_after_vision_weights": after,
        "vision_weights_gib": weights_gib,
        "warmup": WARMUP,
        "timed": TIMED,
        "buckets": {},
    }
    cases = [(b, img) for b, img in BUCKET_IMAGES.items()] + [(1024, PADDED_1024)]
    for bucket, image in cases:
        enc = golden_inputs(image)
        prep = []
        for _ in range(3):
            s = time.perf_counter()
            inputs = tower.prepare_inputs(enc["pixel_values"], enc["image_grid_thw"])
            ttnn.synchronize_device(device)
            prep.append(time.perf_counter() - s)
            inputs.deallocate()
        inputs = prepare(tower, image)
        assert inputs.bucket == bucket
        for _ in range(WARMUP):
            ttnn.deallocate(tower(inputs))
        ttnn.synchronize_device(device)
        samples = []
        for _ in range(TIMED):
            ttnn.synchronize_device(device)
            s = time.perf_counter()
            out = tower(inputs)
            ttnn.synchronize_device(device)
            samples.append(time.perf_counter() - s)
            ttnn.deallocate(out)
        dram_live = dram_view(device)
        inputs.deallocate()
        key = f"{bucket}" if image != PADDED_1024 else f"{bucket}_padded"
        results["buckets"][key] = {
            "image": image,
            "patches": inputs.num_patches,
            "forward": _stats(samples),
            "prepare_inputs": _stats(prep),
            "dram_with_inputs": dram_live,
        }
    _save("vision_perf.json", results)
    for key, r in results["buckets"].items():
        print(
            f"bucket {key} ({r['patches']} patches): forward median {r['forward']['median_ms']:.2f} ms, "
            f"prepare {r['prepare_inputs']['median_ms']:.2f} ms"
        )
    print(f"vision weights {weights_gib:.3f} GiB, load {load_s:.1f} s")


@pytest.mark.timeout(3600)
def test_dram_with_text_model(device):
    from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel

    steps = {"empty": dram_view(device)}
    t0 = time.perf_counter()
    model = PplxDeciderModel.from_snapshot(device)  # default policy (stage-8 selection, C0), disk cache
    steps["text_model_loaded"] = dram_view(device)
    text_load_s = time.perf_counter() - t0
    tower = build_tower(device)
    steps["text_plus_vision_weights"] = dram_view(device)
    # Vision forward at the largest bucket with the text model resident.
    inputs = prepare(tower, "v04_tallest_bar")
    features = tower(inputs)
    ttnn.synchronize_device(device)
    steps["after_vision_forward_1024_features_live"] = dram_view(device)
    inputs.deallocate()
    # Text forward at the largest bucket (8192) with vision weights and the image features resident.
    ids = [151644] * 8000  # synthetic ids (8192 bucket); this step measures memory, not accuracy
    tokens, last_index = model.upload_tokens(ids)
    probs, logits, _ = model(tokens, last_index, 2)
    ttnn.synchronize_device(device)
    steps["after_text_forward_8192"] = dram_view(device)
    for t in (probs, logits, tokens, features):
        ttnn.deallocate(t)
    result = {
        "text_policy": model.config.optimizations.policy.name,
        "text_load_s": text_load_s,
        "text_weights_gib": steps["text_model_loaded"]["allocated_gib"] - steps["empty"]["allocated_gib"],
        "vision_weights_gib": steps["text_plus_vision_weights"]["allocated_gib"]
        - steps["text_model_loaded"]["allocated_gib"],
        "free_gib_with_both_resident": steps["text_plus_vision_weights"]["free_gib"],
        "steps": steps,
        "note": "allocator view between steps; both forwards completed with text + vision resident",
    }
    _save("vision_dram_with_text.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))
