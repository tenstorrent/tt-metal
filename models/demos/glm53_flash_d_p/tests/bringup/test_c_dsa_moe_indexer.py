# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: indexer of block type dsa_moe (layer 3) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 3, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.dsa_moe.indexer.test.1): topk = indexer(attn_norm, q_resid), int32 [2048, 2051]: token ids of the top
512 pools (4 tokens each) + up to 3 tail tokens of the query's incomplete pool, -1 = none. The golden is fp32 compute.
Exact match is the wrong mode: the reference itself scores 0.525 on chunk 1 (order and near-ties; 0.965 on chunk 0,
order only). The gated metric ``pcc_indexer_L03`` is the harness ``topk_overlap`` (per-row set overlap) on chunk 1.
Measured on this golden (CPU host script, not kept), chunk 1 overlap mean / worst row:
fp32 reference from the bf16 golden inputs 0.99903 / 0.9922; bf16 q, pooled keys, weights and scores 0.99874 /
0.9902; the same + 1% score noise 0.99831 / 0.9902. Bugs: no relu 0.937, relu after the head weights 0.898, no head
weights 0.801, no k_norm bias 0.725, no k_norm weight 0.939, RMSNorm for k 0.726, pool mean instead of softmax 0.976,
softmax over head_dim 0.924, last token of the pool 0.940, strided pooling 0.925, chunk's pools zero 0.912, prefix
keys zero 0.761, lowest scores 0.469: all fail 0.99. They pass 0.99: no ape 0.99655 / 0.9824, ape reversed 0.99589 /
0.9766, per-pool key scale 0.97..1.03 0.99844, pool visible one token early (4p + 2 <= q) 0.99868, 4 tokens early
0.99762 (also duplicates), 4 late 0.99764, 511 pools 0.99762, no tail 0.99830, tail one token past q 0.99903.
On chunk 0 (start 0, at most 512 visible pools, every one selected) every score bug above scores overlap 1.0, so
chunk 0 checks the selection structure only (exact sets), and chunk 1's pooled keys check the pooling.
Extra checks, on both chunks (0 then 1; chunk 1 loads the golden prefix):
- structure, exact: ids in [-1, start + S); every id <= its query position; no duplicate id in a row; per-row count of
  ids equals the golden's; ids below the query's incomplete pool come in complete pools of 4 (every pool id 0 or 4
  times); rows where every visible pool is selected (<= 512 visible) have exactly the golden's set. Catches every
  causal, tail and count bug above (the reference passes all of them).
- chunk 1 overlap >= 0.9975 and worst row >= 0.98 (no ape and ape reversed fail; 1% score noise passes).
- pooled keys (the index_key rows this chunk writes, [start/4, (start+S)/4) x 128) vs the golden state: rel L2 <=
  0.008, worst row rel L2 <= 0.02, per-row norm ratio in [0.995, 1.005]. Reference 0.0017 / 0.0022; bf16 k, gate,
  probabilities and pooled key 0.0028 / 0.0038; + 0.3% noise 0.0041 / 0.0058 / [0.9983, 1.0021]. Bugs: x1.01 0.0101 /
  ratio 1.0107, 1% noise 0.0101, ape column 0 only 0.025 / 0.068, last pool zero worst row 1.0, pools shifted by one
  0.62, no ape 0.063. The device module must expose ``{"index_key": [n >= (start+S)/4, 128]}`` (the reference layout,
  read back at the harness boundary) in ``dctx.extra["state_out"]``; the test fails without it.
Limits are written ``not x <= lim`` so NaN fails.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    device_ctx,
    impl_mode,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "indexer"
LAYER = 3
COMPARE = "topk_overlap"  # per-row set overlap: order-free, near-tie flips allowed
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MIN_OVERLAP = 0.9975  # chunk 1 mean per-row overlap (extra, tighter than the gated metric)
MIN_ROW_OVERLAP = 0.98  # chunk 1 worst row
KP = 4  # tokens per pool
SEL_POOLS = 512
KEY_MAX_REL_L2 = 0.008
KEY_MAX_ROW_REL_L2 = 0.02
KEY_ROW_NORM_RATIO = (0.995, 1.005)


def _rowsets(t: torch.Tensor, n: int) -> torch.Tensor:
    """[S, W] ids (-1 = none) -> [S, n] bool membership; ids outside [0, n) count as none (_structure flags them)."""
    t = torch.where((t >= 0) & (t < n), t, torch.full_like(t, -1))
    return torch.zeros(t.shape[0], n + 1, dtype=torch.bool).scatter_(1, t + 1, True)[:, 1:]


def _overlap(got: torch.Tensor, want: torch.Tensor, n: int) -> torch.Tensor:
    mg, mw = _rowsets(got, n), _rowsets(want, n)
    return (mg & mw).sum(-1).float() / mw.sum(-1).clamp_min(1).float()


def _structure(tag: str, got: torch.Tensor, want: torch.Tensor, start: int) -> list[str]:
    fails = []
    s = want.shape[0]
    n = start + s
    qpos = torch.arange(start, start + s)[:, None]
    bad = (got < -1) | (got >= n)
    if bad.any():
        return [f"{tag}: {int(bad.sum())} ids outside [-1, {n})"]
    valid = got >= 0
    viol = valid & (got > qpos)
    if viol.any():
        fails.append(
            f"{tag}: {int(viol.sum())} ids after their query position (causality) on {int(viol.any(-1).sum())} rows"
        )
    srt = got.sort(dim=-1).values
    dup = (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)
    if dup.any():
        fails.append(f"{tag}: duplicate ids on {int(dup.any(-1).sum())} rows")
    cnt, wcnt = valid.sum(-1), (want >= 0).sum(-1)
    if not torch.equal(cnt, wcnt):
        d = (cnt != wcnt).nonzero()[0].item()
        fails.append(
            f"{tag}: {int((cnt != wcnt).sum())} rows select another number of ids (row {d}: {int(cnt[d])}, want {int(wcnt[d])})"
        )
    # ids below the query's incomplete pool must form complete pools
    pool_end = (qpos + 1) // KP * KP
    in_pools = valid & (got < pool_end)
    pid = torch.where(in_pools, got // KP, torch.zeros_like(got))
    pc = torch.zeros(s, n // KP + 1, dtype=torch.int64).scatter_add_(1, pid, in_pools.long())
    partial = (pc != 0) & (pc != KP)
    if partial.any():
        fails.append(f"{tag}: incomplete pools selected on {int(partial.any(-1).sum())} rows")
    # rows where every visible pool is selected: the set is exact
    full = ((qpos[:, 0] + 1) // KP) <= SEL_POOLS
    if full.any():
        neq = (_rowsets(got[full], n) != _rowsets(want[full], n)).any(-1)
        if neq.any():
            fails.append(f"{tag}: {int(neq.sum())} of {int(full.sum())} all-visible rows select another set")
    return fails


def _check_keys(tag: str, got_state, want_state, start: int, s: int) -> list[str]:
    if got_state is None or "index_key" not in got_state:
        return [f"{tag}: no dctx.extra['state_out']['index_key'] from the device module (pooled keys unchecked)"]
    p0, p1 = start // KP, (start + s) // KP
    got, want = got_state["index_key"].float(), want_state["index_key"].float()
    got = got.reshape(-1, want.shape[-1]) if got.numel() % want.shape[-1] == 0 else got
    if got.dim() != 2 or got.shape[-1] != want.shape[-1] or got.shape[0] < p1:
        return [f"{tag}: index_key shape {tuple(got_state['index_key'].shape)}, want [>= {p1}, {want.shape[-1]}]"]
    got, want = got[p0:p1], want[p0:p1]
    if not torch.isfinite(got).all():
        return [f"{tag}: index_key not finite"]
    wn = want.norm(dim=-1).clamp_min(1e-12)
    rel = ((got - want).norm() / want.norm()).item()
    row = ((got - want).norm(dim=-1) / wn).max().item()
    ratio = got.norm(dim=-1) / wn
    rmin, rmax = ratio.min().item(), ratio.max().item()
    metrics.record(f"rel_l2_index_key_{tag}_L{LAYER:02d}", rel)
    metrics.record(f"max_row_rel_l2_index_key_{tag}_L{LAYER:02d}", row)
    print(
        f"{tag} index_key rows [{p0}, {p1}): rel_l2={rel:.5f} (<= {KEY_MAX_REL_L2}) worst_row={row:.4f} "
        f"(<= {KEY_MAX_ROW_REL_L2}) ratio=[{rmin:.4f}, {rmax:.4f}] (in {KEY_ROW_NORM_RATIO})"
    )
    fails = []
    if not rel <= KEY_MAX_REL_L2:
        fails.append(f"{tag}: pooled keys rel L2 {rel:.4f} > {KEY_MAX_REL_L2}")
    if not row <= KEY_MAX_ROW_REL_L2:
        fails.append(f"{tag}: pooled keys worst row rel L2 {row:.4f} > {KEY_MAX_ROW_REL_L2}")
    if not (KEY_ROW_NORM_RATIO[0] <= rmin and rmax <= KEY_ROW_NORM_RATIO[1]):
        fails.append(f"{tag}: pooled keys row norm ratio [{rmin:.4f}, {rmax:.4f}] outside {KEY_ROW_NORM_RATIO}")
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
        want = gl[st.output].long()
        start = ch * g.chunk
        rctx, dctx = reference_ctx(ref, LAYER, g, ch), device_ctx(LAYER, g, ch)
        out = fn(rctx, dctx, *inputs)
        if ch == c:
            _, gate_ok = compare(f"pcc_{STEP}_L{LAYER:02d}", out, gl[st.output], COMPARE, thr)
        else:
            compare(f"topk_overlap_{STEP}_L{LAYER:02d}_{tag}", out, gl[st.output], COMPARE, thr)
        if out.numel() != want.numel() or out.is_floating_point():
            fails.append(f"{tag}: output {tuple(out.shape)} {out.dtype}, want {tuple(want.shape)} integer ids")
            continue
        got = out.reshape(want.shape).long()
        fails += _structure(tag, got, want, start)
        ov = _overlap(got, want, start + want.shape[0])
        mean, worst = ov.mean().item(), ov.min().item()
        metrics.record(f"topk_overlap_mean_{STEP}_{tag}_L{LAYER:02d}", mean)
        metrics.record(f"topk_overlap_worst_row_{STEP}_{tag}_L{LAYER:02d}", worst)
        print(f"{tag}: overlap mean {mean:.6f} worst row {worst:.4f}")
        if ch == c:
            if not mean >= MIN_OVERLAP:
                fails.append(f"{tag}: overlap {mean:.5f} < {MIN_OVERLAP}")
            if not worst >= MIN_ROW_OVERLAP:
                fails.append(f"{tag}: worst row overlap {worst:.4f} < {MIN_ROW_OVERLAP}")
        mode = impl_mode()
        if mode == "device":
            fails += _check_keys(tag, dctx.extra.get("state_out"), want_state, start, want.shape[0])
        elif mode == "reference":
            fails += _check_keys(tag, ref.state_tensors(rctx.state, LAYER, g.seq), want_state, start, want.shape[0])
    assert gate_ok, "topk overlap below threshold"
    assert not fails, "; ".join(fails)
