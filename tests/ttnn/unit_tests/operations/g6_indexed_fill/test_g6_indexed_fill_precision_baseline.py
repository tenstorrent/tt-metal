# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_indexed_fill.

The op moves uint32 values and computes none, so the gate is bit equality. PCC, abs error, relative
RMS, integer ULP distance and the got/true ratio spread are recorded for the report; any nonzero
error is a bug, never precision loss.

Shapes: the production case, the rule-9 shape, and one shape per remaining alignment tag
(w_non_aligned, tile_aligned), which also backs those SUPPORTED values on the device.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.g6_indexed_fill.helpers import (
    UINT32_MAX,
    create_ttnn_input_tensor,
    make_host_inputs,
    pytorch_g6_indexed_fill,
)
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_indexed_fill import INPUT_TAGGERS, g6_indexed_fill

pytestmark = pytest.mark.use_module_device

# (ids, input_shape, expected alignment tag)
CASES = {
    "production": ([3], (7, 1, 1, 32), "h_non_aligned"),
    "rule9": ([4, 1], (5, 2, 1, 64), "h_non_aligned"),
    "w17": ([4, UINT32_MAX, 1], (5, 1, 1, 17), "w_non_aligned"),
    "d2_32": ([2, 0], (3, 1, 32, 32), "tile_aligned"),
}


def _metrics(got, want):
    """int64 views of uint32 values -> the baseline row (float64 only for the PCC/RMS figures)."""
    diff = (got - want).abs()
    got_f, want_f = got.to(torch.float64), want.to(torch.float64)
    _, pcc = comp_pcc(want_f, got_f, pcc=1.0)
    _, allclose = comp_allclose(want_f, got_f, rtol=0.0, atol=0.0)
    std = want_f.std().item()
    rel_rms = (got_f - want_f).pow(2).mean().sqrt().item() / std if std > 0 else 0.0
    nonzero = want != 0
    ratio = got_f[nonzero] / want_f[nonzero]
    return {
        "pcc": pcc,
        "max_abs": int(diff.max()),
        "mean_abs": diff.to(torch.float64).mean().item(),
        "rel_rms": rel_rms,
        "max_ulp": int(diff.max()),  # one integer step is one ULP of uint32
        "mismatched": int((diff != 0).sum()),
        "ratio_median": ratio.median().item(),
        "ratio_p5_p95": (ratio.quantile(0.05).item(), ratio.quantile(0.95).item()),
        "allclose": allclose,
    }


def _to_device(values, device):
    return create_ttnn_input_tensor(values, device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _read_back(tensor):
    # The mask keeps the readback unsigned whether to_torch returns uint32 or int32.
    return ttnn.to_torch(tensor).to(torch.int64) & 0xFFFFFFFF


@pytest.mark.parametrize("case", sorted(CASES))
def test_precision_baseline(device, case):
    ids, input_shape, tag = CASES[case]
    value_shape = (len(ids),) + input_shape[1:]
    assert INPUT_TAGGERS["alignment"](((len(ids),), input_shape, value_shape), {}) == tag
    torch.manual_seed(42)
    x, s = make_host_inputs(input_shape, value_shape)
    ids_host = torch.tensor(ids, dtype=torch.int64).to(torch.uint32)
    x_dev, s_dev, ids_dev = _to_device(x, device), _to_device(s, device), _to_device(ids_host, device)

    output = g6_indexed_fill(ids_dev, x_dev, s_dev, dim=0)
    got = _read_back(output)
    want = pytorch_g6_indexed_fill(ids_host, x, s, dim=0).to(torch.int64)
    m = _metrics(got, want)
    print(
        f"\nBASELINE {case} {list(input_shape)} ids={ids} tag={tag}: pcc={m['pcc']} max_abs={m['max_abs']} "
        f"mean_abs={m['mean_abs']} rel_rms={m['rel_rms']} max_ulp={m['max_ulp']} mismatched={m['mismatched']} "
        f"ratio_median={m['ratio_median']} ratio_p5_p95={m['ratio_p5_p95']} {m['allclose']}"
    )
    assert m["mismatched"] == 0, f"[{case}] {m['mismatched']} elements differ: {m}"
    unchanged = torch.equal(_read_back(x_dev), x.to(torch.int64)) and torch.equal(_read_back(s_dev), s.to(torch.int64))
    assert unchanged, f"[{case}] an input changed"
