# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: ffn_collapse of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.ffn_collapse.test.1): the same weightless mHC collapse as dsa_moe ffn_collapse, ffn_in [S, H] =
sum_n pre[:, n] * x[:, n], x = `h_mid` [S * 4, H] token-major (stream n of token s at row 4s + n), pre = ffn_hc[:, 0:4].
At layer 4 the streams differ (rel ~1.0 from stream 0) but only pre columns 0 / 1 are live (means 0.008 / 0.98);
columns 2 / 3 are ~1e-6 (hc_eps). So on this golden streams 2 / 3 are invisible: last stream dropped, streams 2 / 3
swapped and pre col 2 or 3 x1.01 all score exactly the correct output. The same module therefore also runs on layer
3's golden (dsa_moe, the other MoE layer in 0-4, same op), where pre col 3 is live (mean 0.98): streams 2 / 3 swapped
PCC 0.077, last stream dropped 0.45, pre col 3 x1.01 rel 0.0096 / ratio 1.011 / coefficient 1.0083.
Measured on the layer-4 golden (s4096 chunk 1, 2048 rows, row RMS 0.0014..0.018), PCC / rel L2 / worst row rel L2 /
per-token norm ratio / coefficient: pre reversed 0.62, pre 0/1 swapped 0.32, stream-major rows 0.009, post instead
of pre 0.31, unweighted mean 0.37, rows shifted by one 0.22 (all fail PCC). Bugs that pass PCC 0.99: pre normalized to
sum 1 0.998 / 0.063 / 0.50; output x1.005 0.0057 / 0.0079 / [1.0023, 1.0075] / 1.0051; x0.995 coefficient 0.9951;
x1.01 0.0104 / 0.0128 / [1.007, 1.013]; pre col 0 x1.01 0.0036 / 0.0079 (chunk 0 0.0098); pre col 1 x1.01 0.0098 /
0.0121 / [1.001, 1.012] / 1.0092; last row zero 0.0114 / 1.0; last 32 rows zero 0.118; last 32 columns zero 0.088 /
0.120; one row's pre reversed 0.021 / 0.82; one row duplicated from its neighbour 0.024 / 3.1.
Noise: fp32 reference vs the bf16 golden 0.0026 / 0.0035 / [0.9973, 1.0025] / 1.0001; fp32 rounded to bf16 0.0030 /
0.0040; bf16 products and bf16 sums 0.0034 / 0.0049 / [0.9963, 1.0024] / 0.9998; 0.3% element noise 0.0039 / 0.0046.
Chunk 0 of layer 4 is the same (fp32 0.0026 / 0.0034, bf16 0.0034 / 0.0049 / [0.9963, 1.0022]).
Extra checks (asserted, NaN fails every one): finite; rel L2 <= 0.008; worst per-token rel L2 <= 0.008; per-token
norm ratio in [0.99, 1.01]; coefficient <got, want> / <want, want> in [0.996, 1.004]; on chunk 1 and chunk 0 of
layer 4 and chunk 1 of layer 3 (limits from the frozen dsa_moe test, measured there). Not caught: pre col 2 (dead in
both layers).
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
LAYER = 4
SECOND_LAYER = 3  # dsa_moe: the same weightless collapse, with pre col 3 live
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.008  # ||got - want|| / ||want||
MAX_ROW_REL_L2 = 0.008  # worst per-token ||got - want|| / ||want||
RATIO = (0.99, 1.01)  # per-token ||got|| / ||want||
COEF = (0.996, 1.004)  # <got, want> / <want, want>


def _checks(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    if out.numel() != want.numel():
        return [f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"]
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-30)
    rel = ((got - w).norm() / w.norm()).item()
    row = ((got - w).norm(dim=-1) / wn).max().item()
    r = got.norm(dim=-1) / wn
    lo, hi = r.min().item(), r.max().item()
    coef = ((got * w).sum() / (w * w).sum()).item()
    metrics.record(f"rel_l2_{STEP}_{tag}", rel)
    metrics.record(f"row_rel_l2_max_{STEP}_{tag}", row)
    metrics.record(f"norm_ratio_min_{STEP}_{tag}", lo)
    metrics.record(f"norm_ratio_max_{STEP}_{tag}", hi)
    metrics.record(f"coef_{STEP}_{tag}", coef)
    print(
        f"{tag}: rel_l2={rel:.5f} (<= {MAX_REL_L2}) worst row {row:.5f} (<= {MAX_ROW_REL_L2}) "
        f"norm ratio [{lo:.4f}, {hi:.4f}] (in {RATIO}) coef {coef:.5f} (in {COEF})"
    )
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"{tag}: rel L2 {rel:.4f} > {MAX_REL_L2}")
    if not row <= MAX_ROW_REL_L2:
        fails.append(f"{tag}: worst per-token rel L2 {row:.4f} > {MAX_ROW_REL_L2}")
    if not (RATIO[0] <= lo and hi <= RATIO[1]):
        fails.append(f"{tag}: per-token norm ratio [{lo:.4f}, {hi:.4f}] outside {RATIO}")
    if not (COEF[0] <= coef <= COEF[1]):
        fails.append(f"{tag}: coefficient {coef:.5f} outside {COEF}")
    return fails


def _run(fn, ref, g, chunk, layer=LAYER):
    gl = g.layer(chunk, layer)
    st = _step(ref, LAYER, STEP)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    return fn(reference_ctx(ref, LAYER, g, chunk), device_ctx(layer, g, chunk), *inputs), gl[st.output]


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    out, want = _run(fn, ref, g, c)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    fails = _checks(f"L{LAYER:02d}", out, want)

    # Same module on the layer's other dumped chunk (start 0).
    if c != 0:
        out0, want0 = _run(fn, ref, g, 0)
        _, ok0 = compare(f"pcc_{STEP}_L{LAYER:02d}_c0", out0, want0, mode, thr)
        if not ok0:
            fails.append(f"chunk 0: PCC below {thr}")
        fails += _checks(f"L{LAYER:02d}_c0", out0, want0)

    # Same weightless module on layer 3's golden, where stream 3 carries weight.
    out3, want3 = _run(fn, ref, g, c, SECOND_LAYER)
    _, ok3 = compare(f"pcc_{STEP}_L{LAYER:02d}_on_L{SECOND_LAYER:02d}", out3, want3, mode, thr)
    if not ok3:
        fails.append(f"layer {SECOND_LAYER}: PCC below {thr}")
    fails += _checks(f"L{LAYER:02d}_on_L{SECOND_LAYER:02d}", out3, want3)
    assert not fails, "; ".join(fails)
