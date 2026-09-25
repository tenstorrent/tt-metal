# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import load_baseline, VARIANTS, digest, make_inputs, metrics, prepare, reference, run

BASELINE = load_baseline()


@pytest.fixture(scope="module", params=BASELINE["cases"], ids=lambda c: f"{c['k_length']}-{c['distribution']}")
def accuracy_case(request):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    case = request.param
    host = make_inputs(case["k_length"], case["distribution"])
    assert [digest(x) for x in host] == case["input_sha256"], "Inputs differ from the immutable qualification evidence"
    return case, host, reference(*host)


@pytest.mark.parametrize("variant", VARIANTS)
def test_sdpa_frozen_accuracy(device, accuracy_case, variant, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    case, host, expected = accuracy_case
    frozen = case["variants"][variant]
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    prepared = prepare(inputs, variant)

    # Frozen E evidence hashes decoded FP32 values; A-D hash BF16 storage.
    def prepared_digest(x):
        host = ttnn.to_torch(x)
        return digest(host.float() if variant.startswith("E_") else host)

    assert [prepared_digest(x) for x in prepared] == frozen["prepared_sha256"]
    actual = ttnn.to_torch(run(prepared, variant))
    observed = metrics(actual, expected)
    for name, value in observed.items():
        record_property(name, value)
    record_property("variant", variant)
    record_property("suite", case["suite"])
    record_property("frozen_l2_pct", frozen["metrics"]["l2_pct"])
    record_property("frozen_output_equal", digest(actual) == frozen["output_sha256"])
    record_property("output_sha256", digest(actual))
    # Relative per-case regression budget, not a universal absolute guarantee.
    # Common modes remain visible stress cases; they do not get excluded.
    assert observed["l2_pct"] <= 1.05 * frozen["metrics"]["l2_pct"] + 0.0001
    assert [digest(ttnn.to_torch(x)) for x in inputs] == case["input_sha256"]
    assert [prepared_digest(x) for x in prepared] == frozen["prepared_sha256"]


def test_sdpa_fp64_reference():
    host = make_inputs(512, "normal")
    q, k, v = [x.double() for x in host]
    dense = (q @ k.transpose(-1, -2) / 128**0.5).softmax(-1) @ v
    torch.testing.assert_close(reference(*host, block=128), dense, rtol=1e-12, atol=1e-12)
