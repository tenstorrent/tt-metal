# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU fp32 dataflow vs the reference on the test goldens: every forward's head output within 1e-5 relative."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
import dataflow as DF  # noqa: E402

GOLD = Path("/home/andy/bringup/goldens/kairos/test")
CASES = ["single", "single_gift", "ragged4", "forced_n16", "forced_n32", "forced_n64", "forced_mixed", "rollout2",
         "bias_5call", "front_end_long"]


def fronts(rec):
    """One dict per forward, from the captured host front-end outputs, plus that forward's head output."""
    out, cur = [], {}
    for name, _a, _k, o in rec["firings"]:
        if name == "instance_norm":
            cur = {"loc_scale": o[1]}
        elif name == "fft_norm":
            cur["s_features"] = o
        elif name == "patch":
            px, pm, size, w, idx, xf = o
            B, N = size.shape
            cur.update(x_final=xf.view(3, B, N, -1), weights=w, indices=idx,
                       attn_mask=pm.nan_to_num(0.0).sum(-1) > 0)
        elif name == "output_patch_embedding":
            cur["head"] = o
            out.append(cur)
    return out


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("case", CASES)
def test_dataflow_head(W, case):
    rec = torch.load(GOLD / f"{case}.pt", weights_only=False)
    fs = fronts(rec)
    assert len(fs) == len(rec["gate"])
    for i, f in enumerate(fs):
        with torch.no_grad():
            got = DF.forward(W, f)
        r = rel(got, f["head"])
        print(f"{case} forward {i}: N={f['indices'].shape[1]} head max rel {r:.2e}")
        assert r < 1e-5, (case, i, r)


def test_dataflow_quantiles_single(W):
    rec = torch.load(GOLD / "single.pt", weights_only=False)
    f = fronts(rec)[0]
    q = DF.head_to_quantiles(DF.forward(W, f), *f["loc_scale"])[..., : rec["prediction_length"]]
    assert rel(q, rec["outputs"][0]) < 1e-5
