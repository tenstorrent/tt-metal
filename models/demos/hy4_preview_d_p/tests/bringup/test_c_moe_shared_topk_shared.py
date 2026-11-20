# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component test: topk_shared of block type moe_shared (layer 2) on device vs golden.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer 2, the component rung's last dumped chunk (see testing/harness.component_golden).

Reviewed (C.moe_shared.topk_shared.test.1). A shared layer has no indexer: topk_shared hands on the top-k of the
latest full layer (cfg.topk_source(2) = 1) for the same chunk, no computation (components.yaml: identity). Its
graph input attn_norm is not used. The reference reads the source top-k from ctx.extra["shared_topk"] (or its
per-chunk cache of layer 1's run, which does not exist here: the rendered test raised a KeyError even with the
reference as the module). This test sets ctx.extra["shared_topk"] = the golden's L1.topk (int64 [2048, 2048],
ascending, -1 padded) in both the reference and the device context, as the moe_shared swap tests do.

Golden facts (s4096, measured): L2.topk == L1.topk exactly on both chunks. Chunk 1 (start 2048, the gated chunk):
every row holds 2048 valid positions, no pads. Chunk 0 (start 0): row p holds exactly [0, p], so 2047 rows carry
-1 pads, always as a contiguous tail.

The template's default for an integer output (positional match) would force an order that the consumer does not
need: in the all-device model the handed-on tensor is layer 1's device indexer output (unsorted
topk_large_indices, 0xFFFFFFFF sentinel tail). The gated metric pcc_topk_shared_L02 is the mean per-row set overlap
(order-free) of the normalized output (pads, -1 or 0xFFFFFFFF, as -1) vs the golden, threshold 0.99. An identity
has no precision loss, so the test also asserts exactness (informational metrics, asserted):
  - integer output with S x 2048 elements;
  - every row's valid set equals the golden row's set (worst row overlap 1.0, same valid count, no extra positions,
    no repeats), and no position after its query row;
  - the pads form a contiguous tail and every row keeps >= 1 valid key (ttnn.transformer.sparse_sdpa's producer
    preconditions, sparse_sdpa.hpp; the attention step consumes this tensor);
  - the module follows its input: a second call on chunk 0 (start 0, golden chunk-0 inputs, shared_topk = the
    golden's chunk-0 L1.topk, 2047 padded rows) must also return exactly the golden sets. This catches a module that
    caches its first result, ignores ctx.extra["shared_topk"], or drops / scatters the pads.
Positional match is printed, not gated. A zero stub scores overlap ~0.0005 on chunk 1.
"""

import torch

from models.demos.common.bringup.core import metrics
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
STEP = "topk_shared"
LAYER = 2
COMPARE = "topk_overlap"  # order-free per-row set overlap (the handed-on device tensor may be unsorted)
THRESHOLD = None  # None = spec thresholds.component (default 0.99)
SENTINEL = 0xFFFFFFFF  # topk_large_indices' / sparse_sdpa's pad


def _normalize(out: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1."""
    assert not out.is_floating_point(), f"{STEP} output must be integer positions, got {out.dtype}"
    assert out.numel() == want.numel(), f"output has {out.numel()} elements, want {tuple(want.shape)}"
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _check(got: torch.Tensor, want: torch.Tensor, start: int, tag: str) -> list[str]:
    """Exact per-row set equality with the golden plus the sparse_sdpa pad layout; returns failure messages."""
    fails = []
    valid = got >= 0
    pos = torch.arange(start, start + got.shape[0])[:, None]
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt = got.sort(dim=-1).values  # pads (-1) first, then the valid positions ascending
    ws = want.sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    bad = (srt != ws).any(-1).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows differ from the golden set (first: row {r} at position {start + r}: "
            f"{valid[r].sum().item()} valid, want {(want[r] >= 0).sum().item()})"
        )
    tail_bad = (valid[:, 1:] & ~valid[:, :-1]).any(-1).sum().item()
    if tail_bad:
        fails.append(f"{tail_bad} rows have a valid position after a pad (sparse_sdpa needs a contiguous pad tail)")
    empty = (valid.sum(-1) == 0).sum().item()
    if empty:
        fails.append(f"{empty} rows have no valid key (sparse_sdpa needs >= 1)")
    rows = torch.tensor(
        [torch.isin(b[b >= 0], a[a >= 0]).float().mean().item() if (b >= 0).any() else 1.0 for a, b in zip(got, want)]
    )
    metrics.record(f"{tag}worst_row_overlap_{STEP}_L{LAYER:02d}", rows.min().item())
    print(
        f"{tag or 'golden chunk: '}mean overlap={rows.mean().item():.6f} worst row={rows.min().item():.5f} (== 1) "
        f"pads={(~valid).sum().item()} (want {(want < 0).sum().item()}) "
        f"positional match={(got == want).float().mean().item():.4f} (not gated)"
    )
    return [f"{tag}{m}" for m in fails]


@mesh_parametrize
def test_component(mesh_device):
    g, c = component_golden(S)
    start = c * g.chunk
    assert start > 0, "component chunk must start after 0"
    ref = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    src = ref.cfg.topk_source(LAYER)
    assert src != LAYER, f"layer {LAYER} is not a shared-index layer"
    st = _step(ref, LAYER, STEP)
    fn = module_under_test(S, ref, mesh_device, LAYER, STEP)
    assert not getattr(fn, "cpu_bridge", False), f"device_component returned a CPU bridge; {STEP} is not on the device"

    def run(chunk):
        gl = g.layer(chunk, LAYER)
        shared = g.layer(chunk, src)["topk"]
        assert shared.shape[0] == g.chunk, f"layer {src} topk {tuple(shared.shape)}"
        rctx, dctx = reference_ctx(ref, LAYER, g, chunk), device_ctx(LAYER, g, chunk)
        rctx.extra["shared_topk"] = shared.clone()
        dctx.extra["shared_topk"] = shared.clone()
        inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
        want = gl[st.output].long()
        return _normalize(fn(rctx, dctx, *inputs), want), want

    got, want = run(c)
    thr = threshold(S, "component") if THRESHOLD is None else THRESHOLD
    _, ok = compare(f"pcc_{STEP}_L{LAYER:02d}", got, want, COMPARE, thr)
    assert ok, "top-k set overlap below threshold"
    fails = _check(got, want, start, "")
    assert not fails, "; ".join(fails)

    # Chunk 0: a different shared input with -1 pads (row p keeps [0, p]); the output must follow it exactly.
    got0, want0 = run(0)
    fails0 = _check(got0, want0, 0, "chunk0_")
    assert not fails0, "; ".join(fails0)
