# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type kda_moe (layer 4) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 4, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_moe.attention.test.1): the same KDA module and checks as the layer-0 test
(test_c_kda_dense_attention.py), with every limit re-measured on this layer's golden and kept. Golden: s4096 chunk 1
(start 2048, 2048 rows), ``attn_norm`` [2048, 4096] -> ``attn_out`` [2048, 4096] bf16; KDA over the golden prefix state
(kda_recurrent [64, 128, 128] fp32, kda_conv [3, 24576]). Measured on this golden, PCC / rel L2 / worst 128-row block /
worst per-token rel L2 / per-token norm ratio | post-chunk recurrent rel / worst head / conv rel: fp32 CPU reference
0.999998 / 0.0019 / 0.0019 / 0.0021 / [0.9993, 1.0006] | 0.0008 / 0.0014 / 0.0019; bf16 q/k/v, gate, beta, core and
output 0.99999 / 0.0044 / 0.0045 / 0.0072 / [0.998, 1.002] | 0.0029 / 0.0048; + 1% element noise on q/k/v/core/out
0.99984 / 0.018 / 0.018 / 0.026 / [0.992, 1.006] | 0.011 / 0.017; + 3% 0.99862 / 0.053 / 0.053 / 0.079 | 0.033 / 0.050.
Device (ttKDA 2x2, HiFi4, precise decay) 0.99998 / 0.0073 / 0.0077 / 0.029 / [0.9887, 1.0001] | 0.0166 / 0.047 / 0.0017:
the worst head (20, scale 0.962) is one slow-decay key row (k 124, mean g -1e-4, row scale 0.949), not the fast-decay
bias of layer 0; every other head is <= 0.021. State bugs (all pass PCC): recurrent prefix zeroed 0.99792 / 0.064 /
0.20 / 0.98 | 0.149 / 0.41; conv tail zeroed 0.99993 / 0.012 / 0.047 / 0.43; both zeroed 0.99799 / 0.063 / 0.20 / 1.0;
recurrent state transposed 0.99734 / 0.073 / 0.24 / 1.17 | 0.177 / 0.53; SP: second half from the prefix state
0.99880 / 0.049 / 0.19 / 1.0 | 0.126 / 0.27, from zero 0.99735 / 0.073 / 0.24 / 1.21 | 0.256 / 0.56, conv halo at the
half zeroed 0.99967 / 0.026 / 0.11 / 0.80. Scale: x1.02 1.0 / 0.020 / ratio [1.019, 1.021] (fails rel), x0.99 1.0 /
0.010 / [0.989, 0.991] (passes every check, as at layer 0). o_norm: dropped 0.99987 / rel 0.997, weight dropped
0.99939 / 2.9, no conv silu 0.99296 / 0.97, eps 1.2e-5 0.999995 / 0.086, eps 1e-6 0.99716 / 1.94 (all pass PCC here).
Caught by PCC: q scale dropped 0.959, gate bound -1 0.925, no dt_bias 0.749, A_log not exponentiated 0.803, beta = 1
0.989.
Extra checks (asserted; informational metrics): finite; rel L2 <= 0.02; per-token norm ratio in [0.98, 1.02]; rel L2
of every 128-row block <= 0.03 (a state or SP-carry bug concentrates on the first rows of the chunk or of an SP
segment); worst per-token rel L2 <= 0.1.
State after the chunk: checked against the golden state at start + chunk when the module exposes it
(``dctx.extra["state_out"] = {"kda_recurrent": [64, 128, 128], "kda_conv": [3, 24576]}``, reference layout, torch, read
back at the harness boundary; the glm53 KDA host wrapper does); the reference state is always checked. Limits:
recurrent rel L2 <= 0.03, worst per-head rel L2 <= 0.05, conv tail rel L2 <= 0.02. The device's worst head (0.047)
sits close to 0.05; the smallest state-carry bug is 0.27.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    impl_mode,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "attention"
LAYER = 4
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.02  # ||got - want|| / ||want||, whole chunk
ROW_NORM_RATIO = (0.98, 1.02)  # per-token ||got|| / ||want||
BLOCK_ROWS = 128
MAX_BLOCK_REL_L2 = 0.03  # every 128-row block: state / SP-carry / conv-tail bugs hit the first rows of a segment
MAX_ROW_REL_L2 = 0.1  # worst per-token rel L2
MAX_STATE_REL_L2 = {"kda_recurrent": 0.03, "kda_conv": 0.02}
MAX_STATE_HEAD_REL_L2 = 0.05  # kda_recurrent, worst head


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _check_state(got_state: dict, want_state: dict, fails: list) -> None:
    for name, lim in MAX_STATE_REL_L2.items():
        if name not in got_state:
            fails.append(f"state_out has no {name}")
            continue
        want = want_state[name].float()
        got = got_state[name].float()
        if got.numel() != want.numel():
            fails.append(f"state {name}: {tuple(got.shape)} elements, want {tuple(want.shape)}")
            continue
        got = got.reshape(want.shape)
        rel = _rel(got, want)
        metrics.record(f"rel_l2_state_{name}_L{LAYER:02d}", rel)
        msg = f"state {name}: rel_l2={rel:.5f} (<= {lim})"
        if not torch.isfinite(got).all():
            fails.append(f"state {name} not finite")
        if rel > lim:
            fails.append(f"state {name} rel L2 {rel:.4f} > {lim}")
        if name == "kda_recurrent":
            d = (got - want).flatten(1).norm(dim=1) / want.flatten(1).norm(dim=1).clamp_min(1e-12)
            head = d.max().item()
            metrics.record(f"max_head_rel_l2_state_{name}_L{LAYER:02d}", head)
            msg += f" worst_head_rel_l2={head:.5f} (<= {MAX_STATE_HEAD_REL_L2})"
            if head > MAX_STATE_HEAD_REL_L2:
                fails.append(f"state {name} worst head rel L2 {head:.4f} > {MAX_STATE_HEAD_REL_L2}")
        print(msg)


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    start = c * g.chunk
    assert start > 0, "component chunk must start after 0 so KDA reads a real prefix state"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    rctx, dctx = reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c)
    out = fn(rctx, dctx, *inputs)

    mode = COMPARE or default_mode(want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
    assert ok, "PCC below threshold"

    # Error-size checks (PCC misses every state bug on this golden). Informational metrics.
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    got = out.float().reshape(want.shape)
    w = want.float()
    assert torch.isfinite(got).all(), "non-finite output"
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(got, w)
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    row_rel = ((got - w).norm(dim=-1) / wn).max().item()
    blocks = [_rel(got[i : i + BLOCK_ROWS], w[i : i + BLOCK_ROWS]) for i in range(0, w.shape[0], BLOCK_ROWS)]
    blk = max(blocks)
    blk_at = blocks.index(blk) * BLOCK_ROWS
    metrics.record(f"rel_l2_{STEP}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_L{LAYER:02d}", rmax)
    metrics.record(f"max_block_rel_l2_{STEP}_L{LAYER:02d}", blk)
    metrics.record(f"max_row_rel_l2_{STEP}_L{LAYER:02d}", row_rel)
    print(
        f"rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_block_rel_l2={blk:.4f} at rows {blk_at}.. (<= {MAX_BLOCK_REL_L2}) "
        f"max_row_rel_l2={row_rel:.4f} (<= {MAX_ROW_REL_L2})"
    )
    print("block rel_l2: " + " ".join(f"{b:.4f}" for b in blocks))
    fails = []
    if rel > MAX_REL_L2:
        fails.append(f"relative L2 error {rel:.4f} > {MAX_REL_L2}")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if blk > MAX_BLOCK_REL_L2:
        fails.append(
            f"rel L2 of rows {blk_at}..{blk_at + BLOCK_ROWS - 1} is {blk:.4f} > {MAX_BLOCK_REL_L2} "
            "(prefix recurrent/conv state, SP carry or conv halo mishandled?)"
        )
    if row_rel > MAX_ROW_REL_L2:
        fails.append(f"worst per-token rel L2 {row_rel:.4f} > {MAX_ROW_REL_L2}")

    # State after the chunk (fixed-size, snapshot at start + chunk).
    want_state = g.state(LAYER, at=start + g.chunk)
    if impl_mode() == "device":
        got_state = dctx.extra.get("state_out")
        if got_state is None:
            print("state: the device module does not expose dctx.extra['state_out']; the ladder checks the state")
        else:
            _check_state(got_state, want_state, fails)
    elif impl_mode() == "reference":
        _check_state(ref.state_tensors(rctx.state, LAYER, g.seq), want_state, fails)
    assert not fails, "; ".join(fails)
