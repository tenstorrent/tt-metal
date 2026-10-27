# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU dataflow vs the fp32 reference goldens (CONTRACT §6A): within 1e-5 relative; fallback decision exact."""

import math

import pytest
import torch

from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden

TEST_CASES = ["single", "single_long", "ragged4", "model16", "fallback", "horizon4"]


@pytest.fixture(scope="module")
def weights():
    torch.set_num_threads(16)
    return df.Weights()


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


@pytest.mark.parametrize("case", TEST_CASES)
@pytest.mark.parametrize("bucket", [False, True], ids=["exact", "bucketed"])
def test_forecast(weights, case, bucket):
    g = load_golden(case)
    heads = []
    q = df.forecast(weights, g["x"], g["horizon"], bucket=bucket, heads=heads)
    ref = g["quantiles"]
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(q))
    r = rel(q[fin], ref[fin])
    for k, (h, f) in enumerate(zip(heads, g["forwards"])):
        n = h.shape[1]
        rh = rel(h, f["head"][:, :n])
        print(f"{case} bucket={bucket} forward {k}: head maxrel {rh:.2e}")
        assert rh < 1e-5
    print(f"{case} bucket={bucket}: quantiles maxrel {r:.2e}")
    assert r < 1e-5


@pytest.mark.parametrize("case", TEST_CASES)
def test_front_end(case):
    g = load_golden(case)
    f = g["forwards"][0]
    p = df.prepare_inputs(g["x"], bucket=False)
    assert torch.equal(p.fb_mask, f["fb_mask"])
    assert torch.equal(torch.nan_to_num(p.fb_value, 7.0), torch.nan_to_num(f["fb_value"][:, 0], 7.0))
    assert torch.equal(p.key_valid, f["key_valid"])
    torch.testing.assert_close(p.loc, f["loc"], rtol=1e-5, atol=0)
    torch.testing.assert_close(p.scale, f["scale"], rtol=1e-5, atol=0)
    if "tokens" in f:
        torch.testing.assert_close(p.tokens, f["tokens"], rtol=1e-5, atol=1e-6)
    print(f"{case}: fallback rows {p.fb_mask.nonzero().flatten().tolist()}")


def test_bucket_table():
    for b in df.BUCKETS:
        s = b // df.PATCH + df.NFUT + 1
        s_pad, chunk = df.PAD_TABLE[b]
        assert s <= s_pad and s_pad % chunk == 0 and s_pad % 32 == 0
    assert df.bucket_len(2049) == 3072 and df.bucket_len(4931) == 8192 and df.bucket_len(40000) == 32768
    assert math.ceil(1100 / 32) == 35
