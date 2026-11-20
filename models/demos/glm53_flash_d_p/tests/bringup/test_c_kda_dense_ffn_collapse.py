# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_collapse of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.ffn_collapse.test.1): ffn_in [S, H] = sum_n pre[:, n] * x[:, n], x = `h_mid` [S * 4, H]
token-major (stream n of token s at row 4s + n), pre = ffn_hc[:, 0:4]. Unlike attn_collapse, h_mid's streams already
differ at layer 0 (rel 0.6..1.5 from stream 0), so stream and pre order are visible to PCC there. PCC still passes
scale and sparse bugs. Measured on this golden (s4096 chunk 1, 2048 rows), PCC / rel L2 / per-token norm ratio:
mean instead of sum 0.999997 / 0.75 / [0.25, 0.25], output x1.02 0.999997 / 0.020 / [1.018, 1.022], x1.01 0.0103 /
[1.008, 1.012], last row zeroed 0.99993 / 0.012 / [0, 1.0], last 32 rows zeroed 0.9932 / 0.116, one row with pre
reversed rel 0.0104 (worst row 0.38), one row duplicated from its neighbour worst row 0.93, pre 0/1 swapped 0.9877.
Device noise (bf16 h_mid, the golden's bf16 ffn_hc): fp32 accumulate rel 0.0027 / worst row 0.0041 / ratio
[0.9978, 1.0022]; bf16 products and sums rel 0.0038..0.0047 / worst row <= 0.0076 / ratio [0.9974, 1.0024].
Extra checks: rel L2 <= 0.008, worst per-token rel L2 <= 0.03, per-token norm ratio in [0.99, 1.01]; and the same
module (the step has no weights) on layer 1's golden `h_mid` / `ffn_hc` / `ffn_in`, where pre reversed gives rel 1.13
and pre 0/1 swapped 1.94 (reference noise there: rel 0.0035, worst row 0.0067).
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
STEP = "ffn_collapse"
LAYER = 0
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
SECOND_LAYER = 1  # a second kda_dense layer (same weightless module, different streams and pre)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want||
MAX_ROW_REL_L2 = 0.03  # worst per-token ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    assert out.numel() == want.numel(), f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    rel = ((got - w).norm() / w.norm()).item()
    wn = w.norm(dim=-1).clamp_min(1e-30)
    row = ((got - w).norm(dim=-1) / wn).max().item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"row_rel_l2_max_{STEP}_{tag}", row)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    print(
        f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) worst row {row:.5f} (<= {MAX_ROW_REL_L2}) "
        f"norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO})"
    )
    fails = []
    if rel > MAX_REL_L2:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {MAX_REL_L2}")
    if row > MAX_ROW_REL_L2:
        fails.append(f"{tag}: worst per-token rel L2 {row:.4f} > {MAX_ROW_REL_L2}")
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

    # Same weightless module on a second layer of the block type.
    assert S.block_type_of(SECOND_LAYER) == S.block_type_of(LAYER)
    gd = g.layer(c, SECOND_LAYER)
    out_d = _run(fn, ref, g, c, gd, st, SECOND_LAYER)
    fails += _checks(f"L{SECOND_LAYER:02d}", out_d, gd[st.output])
    assert not fails, "; ".join(fails)
