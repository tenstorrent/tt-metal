# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: attention of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.attention.test.1): attn_out = MLA(attn_norm [S, 4096], q_resid [S, 1536], topk int32 [S, 2051])
-> [S, 4096] bf16 golden. NoPE absorbed MLA: latent = kv_a_layernorm(kv_a_proj(x)) [S, 512], written to the kv_latent
cache at rows [start, start + S); 64 heads, qk 256 / v 256, scale 256^-0.5 (sparse_sdpa's default is 512^-0.5), softmax
over the ids in topk only (-1 = none), o_proj. Golden s4096: chunk 1 (start 2048, gated, prefix latent from the golden
state; its only -1 ids are in the 3 tail columns) and chunk 0 (start 0: rows q < 2047 select every token 0..q, the rest
of the row is -1). Row norms of attn_out 11.9..27.4.
Measured on this golden (CPU host script, not kept), chunk 1 / chunk 0: PCC, rel L2, per-token norm ratio, worst
per-token rel L2; kv = the chunk's latent rows vs the golden state (rel L2, ratio):
fp32 reference 0.999998 / 0.0017 / [0.9997, 1.0003] / 0.0019, kv 0.0020 / [0.9996, 1.0005]; bf16 latent, q, q_lat,
probabilities, o_lat and heads 0.0030 / [0.9991, 1.0013] / 0.0040 (c0 0.0032 / [0.9989, 1.0010]); 1% noise on the
scores 0.0027 / [0.9989, 1.0015] (c0 0.0050 / [0.9975, 1.0022] / 0.012); 1% on q_lat 0.0036; 1% on the output 0.010 /
0.014. Caught by PCC (both chunks): scale 512^-0.5 0.971, latent cache written 32 rows late / early 0.966 / 0.970,
written at row 0 0.60, prefix latent zero 0.55, no tail (topk[:, 2048:] dropped: a 128-aligned truncation to 2048
columns) 0.983, tail one past q 0.979, w_uk / w_uv swapped 0.04, heads reversed or halves swapped ~0, rows 512..1023
swapped with 1024..1535 (chip order) 0.81, rows shifted 32 0.66, no kv norm 0.67, no kv norm weight 0.82; -1 ids
not masked (attend to token 0) 0.99983 on chunk 1 but 0.52 on chunk 0. They pass PCC: output x1.005 1.0000 / 0.0053 /
[1.0047, 1.0053]; x1.01 0.0102 / [1.0097, 1.0103]; score scale x1.02 0.0197 / [1.0085, 1.0215]; latent x1.01 0.0132 /
[1.0008, 1.0174], kv [1.0096, 1.0105]; kv eps 0 0.0063 / [1.0003, 1.0076] (c0 max 1.0090), kv 0.0053 / [1.0025,
1.0096]; kv eps 1e-6 0.0057 / [1.0003, 1.0069] (c0 max 1.0081), kv 0.0049 / [1.0022, 1.0086]; kv eps 1e-4 0.050;
-1 not masked (c1) 0.0185 / worst row 0.086; last complete pool of each row dropped 0.0036 / worst row 0.135; first
or last row zero 0.025 / 0.020, worst row 1.0.
Extra checks on both chunks (asserted; informational metrics): finite; PCC >= the component threshold also on chunk 0;
rel L2 <= 0.012; per-token norm ratio in [0.994, 1.006]; worst per-token rel L2 <= 0.03; rel L2 of every 128-row block
<= 0.02. Latent cache, from ``dctx.extra["state_out"]["kv_latent"]`` ([n >= start + S, 512], the reference layout, read
back at the harness boundary; the test fails without it): the chunk's rows vs the golden state rel L2 <= 0.005, ratio
in [0.997, 1.003], worst row <= 0.008 (the device q_a, the same fp8 projection + RMSNorm, scores 0.0027 / [0.9989,
1.0005] / 0.0032); prefix rows [0, start) equal to the loaded golden prefix (rel <= 1e-3). Every bug above fails at
least one check except output x1.005. Limits are written ``not x <= lim`` so NaN fails.
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
LAYER = 3
COMPARE = None  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MAX_REL_L2 = 0.012  # ||got - want|| / ||want||, whole chunk
ROW_NORM_RATIO = (0.994, 1.006)  # per-token ||got|| / ||want||
MAX_ROW_REL_L2 = 0.03  # worst per-token rel L2
BLOCK_ROWS = 128
MAX_BLOCK_REL_L2 = 0.02  # every 128-row block
KV_MAX_REL_L2 = 0.005  # the chunk's latent rows vs the golden state
KV_ROW_NORM_RATIO = (0.997, 1.003)
KV_MAX_ROW_REL_L2 = 0.008
KV_PREFIX_MAX_REL_L2 = 1e-3  # prefix rows must stay the loaded golden prefix


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _check_out(tag: str, out: torch.Tensor, want: torch.Tensor) -> list[str]:
    if out.numel() != want.numel():
        return [f"{tag}: output has {out.numel()} elements, want {tuple(want.shape)}"]
    got = out.float().reshape(want.shape)
    w = want.float()
    if not torch.isfinite(got).all():
        return [f"{tag}: non-finite output"]
    wn = w.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(got, w)
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    rows = (got - w).norm(dim=-1) / wn
    row_rel, row_at = rows.max().item(), int(rows.argmax())
    blocks = [_rel(got[i : i + BLOCK_ROWS], w[i : i + BLOCK_ROWS]) for i in range(0, w.shape[0], BLOCK_ROWS)]
    blk = max(blocks)
    blk_at = blocks.index(blk) * BLOCK_ROWS
    metrics.record(f"rel_l2_{STEP}_{tag}_L{LAYER:02d}", rel)
    metrics.record(f"row_norm_ratio_min_{STEP}_{tag}_L{LAYER:02d}", rmin)
    metrics.record(f"row_norm_ratio_max_{STEP}_{tag}_L{LAYER:02d}", rmax)
    metrics.record(f"max_row_rel_l2_{STEP}_{tag}_L{LAYER:02d}", row_rel)
    metrics.record(f"max_block_rel_l2_{STEP}_{tag}_L{LAYER:02d}", blk)
    print(
        f"{tag}: rel_l2={rel:.6f} (<= {MAX_REL_L2}) row_norm_ratio=[{rmin:.4f}, {rmax:.4f}] (in {ROW_NORM_RATIO}) "
        f"max_row_rel_l2={row_rel:.4f} at row {row_at} (<= {MAX_ROW_REL_L2}) "
        f"max_block_rel_l2={blk:.4f} at rows {blk_at}.. (<= {MAX_BLOCK_REL_L2})"
    )
    print(f"{tag} block rel_l2: " + " ".join(f"{b:.4f}" for b in blocks))
    fails = []
    if not rel <= MAX_REL_L2:
        fails.append(f"{tag}: relative L2 error {rel:.4f} > {MAX_REL_L2}")
    if not (ROW_NORM_RATIO[0] <= rmin and rmax <= ROW_NORM_RATIO[1]):
        fails.append(f"{tag}: per-token norm ratio [{rmin:.4f}, {rmax:.4f}] outside {ROW_NORM_RATIO}")
    if not row_rel <= MAX_ROW_REL_L2:
        fails.append(
            f"{tag}: worst per-token rel L2 {row_rel:.4f} at row {row_at} > {MAX_ROW_REL_L2} "
            "(masking of -1 ids, a dropped pool or a zeroed row?)"
        )
    if not blk <= MAX_BLOCK_REL_L2:
        fails.append(f"{tag}: rel L2 of rows {blk_at}..{blk_at + BLOCK_ROWS - 1} is {blk:.4f} > {MAX_BLOCK_REL_L2}")
    return fails


def _check_latent(tag: str, got_state, want_state, start: int, s: int) -> list[str]:
    if got_state is None or "kv_latent" not in got_state:
        return [f"{tag}: no dctx.extra['state_out']['kv_latent'] from the device module (latent cache unchecked)"]
    want = want_state["kv_latent"].float()
    r = want.shape[-1]
    got = got_state["kv_latent"].float()
    if got.numel() % r:
        return [f"{tag}: kv_latent shape {tuple(got.shape)}, want [>= {start + s}, {r}]"]
    got = got.reshape(-1, r)
    if got.shape[0] < start + s:
        return [f"{tag}: kv_latent has {got.shape[0]} rows, want >= {start + s}"]
    fails = []
    g_rows, w_rows = got[start : start + s], want[start : start + s]
    if not torch.isfinite(g_rows).all():
        return [f"{tag}: kv_latent not finite"]
    wn = w_rows.norm(dim=-1).clamp_min(1e-12)
    rel = _rel(g_rows, w_rows)
    row = ((g_rows - w_rows).norm(dim=-1) / wn).max().item()
    ratio = g_rows.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_kv_latent_{tag}_L{LAYER:02d}", rel)
    metrics.record(f"max_row_rel_l2_kv_latent_{tag}_L{LAYER:02d}", row)
    msg = (
        f"{tag} kv_latent rows [{start}, {start + s}): rel_l2={rel:.5f} (<= {KV_MAX_REL_L2}) worst_row={row:.4f} "
        f"(<= {KV_MAX_ROW_REL_L2}) ratio=[{rmin:.4f}, {rmax:.4f}] (in {KV_ROW_NORM_RATIO})"
    )
    if not rel <= KV_MAX_REL_L2:
        fails.append(f"{tag}: latent rows rel L2 {rel:.4f} > {KV_MAX_REL_L2}")
    if not row <= KV_MAX_ROW_REL_L2:
        fails.append(f"{tag}: latent worst row rel L2 {row:.4f} > {KV_MAX_ROW_REL_L2}")
    if not (KV_ROW_NORM_RATIO[0] <= rmin and rmax <= KV_ROW_NORM_RATIO[1]):
        fails.append(f"{tag}: latent row norm ratio [{rmin:.4f}, {rmax:.4f}] outside {KV_ROW_NORM_RATIO}")
    if start:
        prel = _rel(got[:start], want[:start])
        metrics.record(f"rel_l2_kv_latent_prefix_{tag}_L{LAYER:02d}", prel)
        msg += f"; prefix rows [0, {start}) rel_l2={prel:.2e} (<= {KV_PREFIX_MAX_REL_L2})"
        if not prel <= KV_PREFIX_MAX_REL_L2:
            fails.append(f"{tag}: latent prefix rows changed (rel L2 {prel:.2e} vs the loaded golden prefix)")
    print(msg)
    return fails


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    want_state = g.state(LAYER)
    chunks = sorted(set(g.dumped_chunks) | {c})
    assert c == chunks[-1] and c > 0, f"component chunk {c} of dumped {g.dumped_chunks}"
    gate_ok, fails = True, []
    for ch in chunks:
        tag = f"c{ch}"
        gl = g.layer(ch, LAYER)
        inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
        want = gl[st.output]
        start = ch * g.chunk
        rctx, dctx = reference_ctx(ref, LAYER, g, ch), device_ctx(LAYER, g, ch)
        out = fn(rctx, dctx, *inputs)
        mode = COMPARE or default_mode(want)
        if ch == c:
            _, gate_ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, want, mode, thr)
        else:
            _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}_{tag}", out, want, mode, thr)
            if not ok:
                fails.append(f"{tag}: PCC below {thr}")
        fails += _check_out(tag, out, want)
        m = impl_mode()
        if m == "device":
            fails += _check_latent(tag, dctx.extra.get("state_out"), want_state, start, want.shape[0])
        elif m == "reference":
            fails += _check_latent(tag, ref.state_tensors(rctx.state, LAYER, g.seq), want_state, start, want.shape[0])
    assert gate_ok, "PCC below threshold"
    assert not fails, "; ".join(fails)
