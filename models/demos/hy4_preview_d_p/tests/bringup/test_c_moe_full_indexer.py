# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: indexer of block type moe_full (layer 1) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 1, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_full.indexer.test.1). Same step and checks as test_c_dense_full_indexer.py (layer 0): topk [S, 2048]
= the indexer's top-2048 key positions per query row (absolute, ascending, -1 padded in the reference), from
attn_norm [S, 6144] and q_resid [S, 2048]: q = RoPE(wq_b(q_resid)) [S, 32, 128], k = RoPE(LayerNorm(wk(attn_norm)))
[S, 128] (eps 1e-5, with bias), interleaved RoPE (theta 1e7) on the LAST 64 of the 128 dims, w =
weights_proj(attn_norm) * 32^-0.5 * 128^-0.5, score[s, t] = sum_h w[s, h] relu(q[s, h] . k[t]) for t <= s. Layer 1
has its own indexer weights; its selection is also what the moe_shared layers 2-4 reuse. Stateful: the step writes
this chunk's index keys and reads the prefix [0, 2048) from the golden state (bf16). Golden: s4096 chunk 1 (start
2048, 2048 rows), so every row sees more than 2048 keys and holds exactly 2048 valid positions.

The template's default for an integer output is positional match, which is wrong here: the device returns unsorted
indices (topk_large_indices) and the top-2048 of near-tied scores is not stable, so even the fp32 CPU reference on the
golden (bf16-stored) inputs scores match 0.761 at layer 1. The gated metric pcc_indexer_L01 is therefore the mean
per-row set overlap |got & want| / |want| (order-free, pads dropped), threshold 0.99. Measured on the layer-1 golden
(CPU; mutations of the reference; "self" = fraction of rows that select their own position):

    variant                                   overlap   worst row  valid / row   non-causal  self
    fp32 reference                            0.99908   0.9971     2048          0           1.0
    bf16 x / W / q / k / key cache, fp32 score 0.99872  0.9956     2048          0           1.0
    same + bf16 scores                        0.99568   0.9863     2048          0           1.0
    bf16 + bfp8 q / k / cache (TtIndexer)     0.99554   0.9893     2048          0           1.0
    bf16 + bfp8 + bf16 scores                 0.99395   0.9854     2048          0           1.0
    k_norm eps 1e-6 (harmless)                0.99908   0.9971     2048          0           1.0
    no k_norm bias / no k_norm weight         0.981 / 0.942   0.94 / 0.84
    no k_norm (raw wk)                        0.90924   0.6865
    RoPE positions + 1                        0.98512   0.9692     2048          0           1.0
    RoPE on the first 64 dims / rotate-half   0.773 / 0.811   0.57 / 0.69
    RoPE from position 0 / no q RoPE          0.760 / 0.736
    uniform / |head weights| / no ReLU        0.666 / 0.657 / 0.841
    prefix keys zero / chunk keys zero        0.594 / 0.906
    causal mask t <= s + 1                    0.99880   0.9966     2048          2016        1.0     (passes 0.99)
    causal mask t < s (own key dropped)       0.99880   0.9966     2048          0           0.0     (passes 0.99)
    top-2047 (+ one pad)                      0.99880   0.9966     2047          0           1.0     (passes 0.99)
    last row all pads                         0.99859   0.0        [0, 2048]     0           0.9995  (passes 0.99)
    last tile row all pads                    0.98348   0.0        [0, 2048]
    rows shifted by 1 / SP row halves swapped 0.926 / 0.546   0.79 / 0.49   (halves: 910703 non-causal)
    non-causal                                0.98456   0.9692     2048          64512
    zero stub                                 ~0.0005

The layer-1 margins are a little thinner than layer 0's (bf16 scores 0.99568 vs 0.99699; the layer-0 device module,
fp32 DEST scores, measured 0.99708 there), still well above 0.99. So the test also checks (informational metrics,
asserted): integer output with S x 2048 elements; every entry is a pad (-1, or the 0xFFFFFFFF sentinel of
topk_large_indices) or a position in [0, row position] (causal); no position repeats within a row; each row holds
exactly min(position + 1, 2048) valid positions; the worst row overlap >= 0.97 (device estimates >= 0.985); every
row selects its own position (1.0 in the golden and in every precision variant above). Rows of the golden chunk all
see > 2048 keys, so the short-row rule (a row with position + 1 <= 2048 keeps every key it sees, the rest padded) is
not exercised there; the test runs the module a second time on chunk 0 (start 0, golden chunk-0 inputs, empty
prefix), where every row must hold exactly the positions [0, row position].
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import Ctx
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
STEP = "indexer"
LAYER = 1
COMPARE = "topk_overlap"  # order-free per-row set overlap (the device returns unsorted indices)
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
MIN_ROW_OVERLAP = 0.97  # worst per-row set overlap vs golden (reference 0.9971, bf16 / bfp8 device estimates >= 0.985)
MIN_SELF = 1.0  # fraction of rows that select their own position (golden 1.0)
SENTINEL = 0xFFFFFFFF  # topk_large_indices' pad for rows with fewer valid keys than k


def _normalize(out: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1."""
    assert not out.is_floating_point(), f"indexer output must be integer positions, got {out.dtype}"
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _row_overlap(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per-row |got & want| / |want| over the valid (non -1) positions; 1.0 for a row with none wanted."""
    res = []
    for a, b in zip(got, want):
        b = b[b >= 0]
        res.append(torch.isin(b, a[a >= 0]).float().mean().item() if b.numel() else 1.0)
    return torch.tensor(res)


def _structure(got: torch.Tensor, start: int, k: int) -> list[str]:
    """Causality, uniqueness and per-row valid count of a normalized output at absolute rows [start, start + S)."""
    fails = []
    pos = torch.arange(start, start + got.shape[0])[:, None]
    valid = got >= 0
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt = torch.where(valid, got, torch.full_like(got, -1)).sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    n = valid.sum(-1)
    exp = torch.clamp(pos[:, 0] + 1, max=k)
    bad = (n != exp).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows hold the wrong number of valid positions "
            f"(first: row {r} at position {start + r} has {n[r].item()}, want {exp[r].item()})"
        )
    return fails


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    start = c * g.chunk
    assert start > 0, "component chunk must start after 0 so the indexer reads a real key prefix"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref, LAYER, STEP)
    gl = g.layer(c, LAYER)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output].long()
    k = want.shape[-1]
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), "device_component returned a CPU bridge; indexer is not on the device"
    out = fn(reference_ctx(ref, LAYER, g, c), device_ctx(LAYER, g, c), *inputs)

    got = _normalize(out, want)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", got, want, COMPARE, thr)
    assert ok, "top-k set overlap below threshold"

    # Extra checks (see the module docstring). Informational metrics, not in the runner's threshold list.
    rows = _row_overlap(got, want)
    pos = torch.arange(start, start + got.shape[0])[:, None]
    self_frac = (got == pos).any(-1).float().mean().item()
    metrics.record(f"worst_row_overlap_{STEP}_L{LAYER:02d}", rows.min().item())
    metrics.record(f"self_selected_{STEP}_L{LAYER:02d}", self_frac)
    print(
        f"golden chunk {c}: mean overlap={rows.mean().item():.6f} worst row={rows.min().item():.5f} "
        f"(>= {MIN_ROW_OVERLAP}) rows < 0.99: {(rows < 0.99).sum().item()} self selected={self_frac:.5f} "
        f"(>= {MIN_SELF}) positional match={(got == want).float().mean().item():.4f} (not gated)"
    )
    fails = _structure(got, start, k)
    if rows.min().item() < MIN_ROW_OVERLAP:
        fails.append(f"worst row overlap {rows.min().item():.5f} < {MIN_ROW_OVERLAP} (row {rows.argmin().item()})")
    if self_frac < MIN_SELF:
        fails.append(f"only {self_frac:.5f} of rows select their own position (causal mask off by one?)")
    assert not fails, "; ".join(fails)

    # Chunk 0: every row sees <= 2048 keys and must keep exactly [0, position], the rest padded.
    g0 = g.layer(0, LAYER)
    in0 = [g0[i].float() if g0[i].is_floating_point() else g0[i] for i in st.inputs]
    want0 = g0[st.output].long()
    dctx0 = Ctx(LAYER, 0, g.chunk, None, {"state_prefix": g.state(LAYER), "prefix_len": 0, "max_seq": g.seq})
    out0 = fn(reference_ctx(ref, LAYER, g, 0), dctx0, *in0)
    got0 = _normalize(out0, want0)
    rows0 = _row_overlap(got0, want0)
    metrics.record(f"chunk0_worst_row_overlap_{STEP}_L{LAYER:02d}", rows0.min().item())
    print(f"chunk 0 (start 0): mean overlap={rows0.mean().item():.6f} worst row={rows0.min().item():.5f} (== 1)")
    fails0 = _structure(got0, 0, k)
    if rows0.min().item() < 1.0:
        fails0.append(f"worst row overlap {rows0.min().item():.5f} < 1 (row {rows0.argmin().item()})")
    assert not fails0, "chunk 0: " + "; ".join(fails0)
