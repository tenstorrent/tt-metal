# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type kda_dense (layer 0) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 0, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.kda_dense.attention.test.1): golden is s4096 chunk 1 (start 2048, 2048 rows), ``attn_norm`` [2048, 4096]
-> ``attn_out`` [2048, 4096] bf16; KDA over the golden prefix state (kda_recurrent [64, 128, 128] fp32, kda_conv
[3, 24576] = the last 3 pre-conv q|k|v rows). PCC is too weak here: the state carries little of the output
norm, so every state bug passes 0.99. Measured on this golden, PCC / rel L2 / rel L2 of the worst 128-row block / worst
per-token rel L2 / per-token norm ratio: fp32 CPU reference 0.999998 / 0.0020 / 0.0020 / 0.0024 / [0.9996, 1.0004];
bf16 activations, projections, q/k/v/g/beta and core output 0.99999 / 0.0047 / 0.0047 / 0.0063 / [0.998, 1.002];
the same + 1% element noise 0.99994 / 0.011 / 0.011 / 0.015 / [0.996, 1.005]; + 3% noise 0.99955 / 0.030 / 0.030 /
0.039. State bugs (all pass PCC): recurrent prefix zeroed 0.99935 / 0.036 / 0.097 / 0.63; conv tail zeroed 0.99992 /
0.012 / 0.049 / 0.41; both zeroed (start treated as 0) 0.99926 / 0.039 / 0.11 / 0.64; recurrent state transposed
0.9982 / 0.059 / 0.15 / 0.95; SP: second half of the chunk from the prefix state instead of the first half's carry
0.99987 / 0.016 / 0.048 / 0.44, from zero 0.99941 / 0.035 / 0.097 / 0.66, conv halo at the half zeroed 0.99994 / 0.011 /
0.043 / 0.38; output x1.02 0.999998 / 0.020 / ratio [1.0196, 1.0204]; o_norm dropped 0.99942 / 0.997 (ratio 0.003:
the core output RMS is far below sqrt(eps), so eps sets the o_norm scale; eps 1e-6 gives rel 1.9). Caught by PCC
already: q scale dropped 0.958, gate bound -1 0.985, logsigmoid gate 0.905, no dt_bias 0.979, A_log not exponentiated
0.982, beta = 1 0.987, silu gate 0.30, no gate 0.91, no o_norm weight 0.988, no conv silu 0.981, TP head halves
swapped 0.005.
Extra checks (asserted; informational metrics): finite; rel L2 <= 0.02; per-token norm ratio in [0.98, 1.02]; rel L2
of every 128-row block <= 0.03 (a state or SP-carry bug concentrates on the first rows of the chunk or of an SP
segment); worst per-token rel L2 <= 0.1.
State after the chunk: checked against the golden state at start + chunk when the module exposes it. The reference
state is always checked (recurrent rel 0.0008, conv 0.0019). A device module opts in by putting
``{"kda_recurrent": [64, 128, 128], "kda_conv": [3, 24576]}`` torch tensors (the reference layout, read back at the
harness boundary) in ``dctx.extra["state_out"]``; otherwise the ladder's state metrics check it. Limits:
recurrent rel L2 <= 0.03 and worst per-head rel L2 <= 0.05 (bf16 state 0.0017 / 0.0019, 1% noise 0.010 / 0.014; zero
prefix 0.125 / 0.24, second half only 0.099 / 0.20, transposed 1.41), conv tail rel L2 <= 0.02 (prefix tail 1.05).
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
LAYER = 0
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
