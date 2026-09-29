# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attn_collapse of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.attn_collapse.test.1): attn_in [S, H] = sum_n pre[:, n] * x[:, n], x = `in` [S * 4, H] token-major
(stream n of token s at row 4s + n), pre = attn_hc[:, 0:4]. At layer 0 the four streams are identical copies of the
embedding, so the golden cannot see stream order or pre column order, and whole-output PCC passes real bugs. Measured
on this golden (s4096 chunk 1, 2048 rows), PCC / rel L2 / per-token norm ratio: last stream dropped 0.9998 / 0.021 /
[0.80, 1.00], output x1.02 0.999998 / 0.020 / [1.017, 1.023], last 32 rows zero 0.9934 / 0.115 / [0, 1.0], pre
reversed or stream-major rows at layer 0 undetectable or 0.047. Device noise: bf16 products accumulated in bf16 give
rel 0.0036, ratio [0.994, 1.004] (fp32 reference vs the bf16 golden: rel 0.0021).
Extra checks: rel L2 <= 0.01 and per-token norm ratio in [0.985, 1.015] at layer 0; and the same module (the step has
no weights) on layer 1's golden `in` / `attn_hc` / `attn_in`, whose streams differ (rel 0.8..1.9 from stream 0):
there pre reversed gives rel 1.27, pre columns 0/1 swapped 0.29, stream-major rows 1.28, last stream dropped 0.38.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "attn_collapse"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
DISTINCT_LAYER = 1  # a kda_dense layer whose streams differ (checks stream / pre order)
MAX_REL_L2 = 0.01  # ||got - want|| / ||want||
RATIO = (0.985, 1.015)  # per-token ||got|| / ||want||


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    assert out.numel() == want.numel(), f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    rel = ((got - w).norm() / w.norm()).item()
    r = got.norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-30)
    lo, hi = r.min().item(), r.max().item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    print(f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO})")
    fails = []
    if rel > MAX_REL_L2:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {MAX_REL_L2}")
    if lo < RATIO[0] or hi > RATIO[1]:
        fails.append(f"{tag}: per-token norm ratio [{lo:.4f}, {hi:.4f}] outside {RATIO}")
    return fails


def _run(fn, ref, g, c, gl, st, layer):
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    return fn(reference_ctx(ref, LAYER, g, c), device_ctx(layer, g, c), *inputs)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out = _run(fn, ref, g, c, gl, st, LAYER)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    fails = _checks(f"L{LAYER:02d}", out, want)

    # Same module on a layer with distinct streams (layer 0's four streams are identical).
    assert S.block_type_of(DISTINCT_LAYER) == S.block_type_of(LAYER)
    gd = g.layer(c, DISTINCT_LAYER)
    out_d = _run(fn, ref, g, c, gd, st, DISTINCT_LAYER)
    fails += _checks(f"L{DISTINCT_LAYER:02d}_distinct_streams", out_d, gd[st.output])
    assert not fails, "; ".join(fails)
