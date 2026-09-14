# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can the verify bucket be 64 rows instead of 128? That would halve every verify forward.

ANSWER, MEASURED 2026-09-14 (T3K, 8 layers): **no.** ANCHOR stays 128.

    raw model, bucket=64          TtTarget, ANCHOR=64
      chunk_start  64:  0.966630    start  40:  1.000000   (fits one bucket -- proves nothing)
      chunk_start 128:  0.998830    start  64:  0.966630
      chunk_start 192:  0.991629    start 100:  0.984522
                                    start 130:  0.998853

A 128-row bucket is EXACTLY 1.0 at its aligned offsets. No 64-row result is exact, and three of
them fall below 0.99. The gate here is exactness, not a threshold: this target defines which tokens
the model emits, so a target that is merely 0.99-correct emits different text than the real 27B.
That trades the implementation's central claim -- tokens bit-identical to production -- for speed,
which is not a trade this path is allowed to make.

What the numbers DO show is that the spacing argument below was most of the story but not all of
it. At bucket 128 the offending offsets measure 0.16 / 0.27 / 0.25; at bucket 64 the same offsets
are 0.97 / 0.99 / 0.99. Roughly an order of magnitude of the error was the bucket/page spacing
disagreement and it is gone. What remains is smaller and has a shape: offsets that are odd
multiples of 64 (64, 192) are worse than the even one (128), and the deficit shrinks as more real
context precedes the block -- chunk_start=64, the case with the least history, is the worst at
0.9666. Something downstream of the KV write still assumes 128-row granularity. Finding it would
make this lever available; it was not found here, and it is a real investigation rather than a
config change.

Kept as a documented negative result, and opt-in so it does not red the suite: set
``DFLASH_PROBE_ANCHOR64=1``. Re-enable it if the 128-granularity assumption is ever tracked down.

``TtTarget.ANCHOR`` is doing two jobs at once: it is the masked bucket size AND the required
``chunk_start`` alignment. Both are 128 today, and the verify forward therefore always costs 128
rows no matter how few tokens a step actually commits (7.00, measured -- see
test_dflash_block_size_sweep.py). The device profile says ~25 % of that forward is collectives and
~18 % is layout churn, and those are the row-proportional parts, so halving the rows is the largest
lever left that is not a redesign.

The reason 128 was chosen is real, and it is NOT "the bucket must be 128". It is a SPACING
argument: ``paged_fill_cache`` writes a whole padded bucket starting at block ``chunk_start // 64``,
so consecutive segments land spaced by the BUCKET while the page table spaces blocks by 64. At
bucket 128 those disagree at every odd 64-boundary, which is exactly the measured failure --
PCC 1.0 at 128/256 but 0.16 / 0.27 / 0.25 at 64 / 192 / 320
(test_tt_block_forward_needs_bucket_alignment).

If the bucket were 64 the two spacings would be THE SAME NUMBER and the disagreement has nothing to
disagree about. That is the hypothesis this file tests. The other blocker one would expect -- GDN
granularity -- is not one: the bucket list comment calls 128 "the GDN sub-chunk", but the fused op
runs at ``_FUSED_CHUNK_SIZE = 32`` (tt/gdn/fused_chunk.py), so 64 is already a legal multiple.

This is a PROBE, not a product change. It monkeypatches ``TtTarget.ANCHOR`` for the duration of a
test so nothing ships until the PCC says it may. Cheap: 8 layers, not 64.

Run::

    DFLASH_PROBE_ANCHOR64=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_anchor64.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
N_LAYERS = 8

#: Opt-in: this probe records a NEGATIVE result (see the module docstring) and is expected to fail.
pytestmark = pytest.mark.skipif(
    os.environ.get("DFLASH_PROBE_ANCHOR64") != "1",
    reason="ANCHOR=64 probe: answered (no) on 2026-09-14; set DFLASH_PROBE_ANCHOR64=1 to re-measure",
)


def _page_table():
    return torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)


def _build(mesh_device):
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE, n_layers=N_LAYERS
    )
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    return model


def _pcc(a, b):
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(a, b, 0.99)
    return float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc


@torch.no_grad()
@pytest.mark.parametrize("prefix", [64, 128, 192], ids=lambda p: f"offset{p}")
@parametrize_mesh_tp()
def test_bucket64_is_exact_at_64_aligned_offsets(mesh_device, prefix, reset_seeds, ensure_gc):
    """The raw model at bucket=64: are the offsets that break at bucket=128 exact at bucket=64?

    Offsets 64 and 192 are the ones measured at 0.16 / 0.27 with a 128 bucket. If the spacing
    argument above is right they become exact here, because a 64-row bucket advances the KV write
    by exactly one 64-row page per segment.
    """
    model = _build(mesh_device)
    torch.manual_seed(0)
    blk = 16
    total = prefix + blk
    tokens = torch.randint(0, model.args.vocab_size, (1, total), dtype=torch.long)

    golden = model.prefill_block_all_logits(tokens, _page_table(), actual_len=total, chunk_start=0)

    # Feed the prefix as whole 64-row buckets, then the block as a partial 64-row bucket.
    for lo in range(0, prefix, 64):
        model.prefill_block_all_logits(tokens[:, lo : lo + 64], _page_table(), actual_len=64, chunk_start=lo, bucket=64)
    logits = model.prefill_block_all_logits(
        tokens[:, prefix:total], _page_table(), actual_len=blk, chunk_start=prefix, bucket=64
    )

    value = _pcc(golden[:, prefix:], logits)
    logger.info(f"bucket=64 block at chunk_start={prefix}: pcc {value:.6f}")
    assert value > 0.99, (
        f"a 64-row bucket at chunk_start={prefix} diverged from the one-shot prefill (pcc {value:.6f}); "
        "the bucket/page spacing argument does not hold and ANCHOR must stay 128"
    )


@torch.no_grad()
@pytest.mark.parametrize("start", [40, 64, 100, 130], ids=lambda s: f"start{s}")
@parametrize_mesh_tp()
def test_tt_target_is_exact_with_anchor64(mesh_device, start, monkeypatch, reset_seeds, ensure_gc):
    """The same exactness claim ``test_tt_target_forward_matches_one_shot`` makes, at ANCHOR=64.

    If this passes at every start, ANCHOR=64 is a drop-in: the anchoring scheme is unchanged, it
    just re-runs half as many rows per forward.
    """
    monkeypatch.setattr(TtTarget, "ANCHOR", 64)
    model = _build(mesh_device)
    torch.manual_seed(0)
    blk = min(16, TtTarget.ANCHOR - (start % TtTarget.ANCHOR))
    total = start + blk
    tokens = torch.randint(0, model.args.vocab_size, (1, total), dtype=torch.long)

    golden = model.prefill_block_all_logits(tokens, _page_table(), actual_len=total, chunk_start=0)

    target = TtTarget(model, [1, 5, 7], _page_table())
    target.reset()
    target.forward(tokens[:, :start], 0)
    logits, taps = target.forward(tokens[:, start:], start)

    assert logits.shape[1] == blk and taps.shape[1] == blk
    value = _pcc(golden[:, start:], logits)
    logger.info(f"ANCHOR=64 TtTarget block at start={start} (blk={blk}): pcc {value:.6f}")
    assert value > 0.99, f"ANCHOR=64 TtTarget diverged from the one-shot prefill at start={start}: {value:.6f}"
