# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive MXFP8 reconstruction survey; MXFP8_ACCURACY_REPORT selects the JSON artifact."""

import json
import os

import pytest
import torch

from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.mxfp8_reference import (
    CONTAINERS,
    DEST_MODES,
    E4M3_NAN,
    PRIMARY,
    corpus,
    decode_e4m3,
    kernel_hashes,
    metrics,
    reference,
    run_decode,
)

import ttnn


def test_mxfp8_reference_matches_torch_e4m3fn():
    # Host-only: the independent decoder agrees with PyTorch on every finite code.
    codes = torch.tensor([code for code in range(256) if code not in E4M3_NAN], dtype=torch.int32)
    expected = codes.to(torch.uint8).view(torch.float8_e4m3fn).double()
    torch.testing.assert_close(decode_e4m3(codes), expected, rtol=0, atol=0)
    assert float(decode_e4m3(torch.tensor([0x7E]))) == 448.0
    assert float(decode_e4m3(torch.tensor([0x01]))) == 2.0**-9


def test_mxfp8_corpus_covers_every_pair():
    codes, scales = corpus()
    assert not bool(((codes == E4M3_NAN[0]) | (codes == E4M3_NAN[1])).any())
    pairs = set(zip(codes[:255].flatten().tolist(), scales[:255].repeat_interleave(32, dim=-1).flatten().tolist()))
    finite_codes = 256 - len(E4M3_NAN)
    assert len(pairs) == finite_codes * 255, "Every finite (code, scale) pair must be present"
    exact = reference(codes, scales)
    assert bool((exact.abs() >= 2.0**128).any()) and bool(((exact != 0) & (exact.abs() < 2.0**-126)).any())


@pytest.mark.skipif(not is_blackhole(), reason="MXFP8 decoder requires Blackhole")
def test_mxfp8_accuracy_survey(device, tmp_path):
    torch.set_num_threads(4)
    codes, scales = corpus()
    exact = reference(codes, scales)
    cases, failures = [], []
    for container in CONTAINERS:
        for fp32_dest_acc_en, dtype in DEST_MODES:
            config = dict(container=container, fp32_dest_acc_en=fp32_dest_acc_en, dtype=dtype)
            actual, keep_alive = run_decode(device, codes, scales, **config)
            for tensor in keep_alive:
                ttnn.deallocate(tensor)
            result = metrics(actual, exact, dtype, codes, scales)
            cases.append(dict(config=config, primary=config == PRIMARY, metrics=result))
            if result["mismatches"]:
                failures.append(f"{config}: {result['mismatches']} mismatches, e.g. {result['mismatch_examples'][:2]}")
    report = dict(
        kernel_hashes=kernel_hashes(),
        torch_version=torch.__version__,
        contract="exact E4M3*2^(E8M0-127); >=2^128 -> inf; <2^-126 flush or correctly rounded subnormal",
        valid_values_per_case=int(codes.numel()),
        cases=cases,
        accuracy_gate=dict(
            path="mxfp8_exact",
            passed=not failures,
            observed=dict(
                mismatches=sum(case["metrics"]["mismatches"] for case in cases),
                cases=len(cases),
            ),
        ),
    )
    destination = os.environ.get("MXFP8_ACCURACY_REPORT", str(tmp_path / "accuracy.json"))
    with open(destination, "w") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert not failures, "\n".join(failures)
