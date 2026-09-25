# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optional real-activation replay; raw model captures are not repository assets.

Set SDPA_MODEL_CAPTURE_DIR to the frozen FLUX.2 block-sweep-source directory.
This is an attention integration check, not a new end-to-end model quality run.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, metrics, prepare, reference

CAPTURES = {
    "qkv-dual.0.pt": "69ef1a3b0ae8508f758dd9b73bceb9a9e5f114ccda1fc2a698333d93a209b62c",
    "qkv-single.47.pt": "1f04269fe90447c593ed8d92ec5ad861d4c9cb39e5cdc915b9f84e615f721bea",
}


@pytest.fixture(scope="module", params=CAPTURES)
def model_capture(request):
    folder = os.getenv("SDPA_MODEL_CAPTURE_DIR")
    if not folder or not is_blackhole():
        pytest.skip("Requires Blackhole and SDPA_MODEL_CAPTURE_DIR")
    path = Path(folder) / request.param
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CAPTURES[request.param]
    data = torch.load(path, map_location="cpu", weights_only=True)
    host = [data[name].contiguous() for name in ("q", "k", "v")]
    assert all(x.shape == (1, 4, 4608, 128) and x.dtype == torch.bfloat16 for x in host)
    expected = torch.cat(
        [reference(host[0][..., start : start + 256, :], *host[1:]) for start in range(0, 4608, 256)], dim=2
    )
    return request.param, host, expected


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_sdpa_recipe_model_capture(device, model_capture, variant, record_property):
    name, host, expected = model_capture
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    prepared = prepare(inputs, variant)

    def invoke():
        return ttnn.transformer.scaled_dot_product_attention(
            *prepared,
            is_causal=False,
            precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
            inputs_prepared=variant.startswith("E_"),
        )

    output = invoke()
    actual = ttnn.to_torch(output)
    observed = metrics(actual, expected)
    record_property("capture", name)
    record_property("capture_sha256", CAPTURES[name])
    record_property("variant", variant)
    for key, value in observed.items():
        record_property(key, value)
    # Broad integration smoke, not a substitute for per-case frozen regression
    # limits or full-model acceptance. Retain every measured error in the report.
    assert observed["pcc"] > 0.9
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(ttnn.to_torch(traced)) == digest(actual)
    finally:
        ttnn.release_trace(device, trace)
    assert [digest(ttnn.to_torch(x)) for x in inputs] == [digest(x) for x in host]
