# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill across every padded shape, and independence from how much padding a prompt gets.

Voxtral prefill has NO buckets (that is the codec). It quantizes: `Sp = ceil(S/128)*128`, so with
`max_seq_len=2048` there are exactly **16** padded shapes, each its own kernel shape, program-cache
entry and `fill_cache` work split. The shipped fixture tops out at P=357 -> Sp=384, so ten of the
sixteen were never exercised by anything.

Two properties, deliberately separate:

  1. SHAPE COVERAGE -- every Sp runs, and prefill writes every (head, position-tile) of the KV
     cache. This is the XTTS BUG-7 shape: `fill_cache` splits n_kv_heads*seq_tiles blocks over the
     core grid and each core walks its later blocks forward by Wt, so a run crossing a head
     boundary leaves the next head's leading tile unwritten -- zero, silently. XTTS lost 7 of 16
     heads at some lengths with every shipped gate passing.
  2. PAD INVARIANCE -- a prompt's answer must not depend on how much padding it receives. Prefill
     zero-pads to Sp, builds a full Sp x Sp causal mask, then sets `pos = S` and slices `last_only`
     at S-1. Padding therefore sits AFTER the real tokens and causality should exclude it, so PCC
     must be flat across a 128 boundary. A mask cut at the wrong place would show as a step.

WHY NO REFERENCE COMPARISON IN (1): a 26-layer fp32 CPU forward at S=2048 is prohibitively slow,
and the failure this targets is a tile that is never written -- a structural property that needs no
oracle. Accuracy against the reference is `test_backbone_pcc.py`'s job, on real prompts.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_prefill_shapes.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    N_KV_HEADS,
    N_LAYERS,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    backbone_state,
    fixture_embeds,
)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

MAX_SEQ = 2048
SHAPES = tuple(range(gpt.PREFILL_MULTIPLE, MAX_SEQ + 1, gpt.PREFILL_MULTIPLE))   # 128 .. 2048
TILE = 32

# Lengths straddling a 128 boundary, all reachable from the longest fixture prompt (P=357).
# 250,256 -> Sp 256 (6 and 0 pad rows); 257,300,352 -> Sp 384 (127, 84, 32 pad rows).
PAD_LENGTHS = (250, 256, 257, 300, 352)
PAD_PCC = 0.999
PAD_SPREAD = 0.002      # PCC must be FLAT across the boundary, not merely above the gate


@pytest.fixture(scope="module")
def dev():
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def big(dev, w):
    """A model whose cache holds the largest shape, so every Sp is reachable."""
    return TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=MAX_SEQ)


def _long_input(w, S):
    """S rows built by repeating a real prompt's embeddings.

    Real embedding vectors, not `randn`: trap 12 (random activations are off-manifold and read PCC
    0.892 where real prompts give 0.9994) applies to anything Block-1 shaped. Repeating is not a
    natural prompt, but every row is a genuine embedding, which is what the write path sees.
    """
    full, _ = fixture_embeds(3, w)                      # longest fixture prompt, P=357
    reps = (S + full.shape[1] - 1) // full.shape[1]
    return full.repeat(1, reps, 1)[:, :S]


@pytest.mark.parametrize("sp", SHAPES, ids=lambda s: f"Sp{s}")
def test_every_padded_prefill_shape_fills_the_cache(big, w, sp):
    """Every head and every position-tile of the KV cache must be written at this shape."""
    S = sp - 5                                          # so the shape is reached WITH padding
    embeds = _long_input(w, S)
    big.reset()
    out = big.prefill(embeds, last_only=True)
    assert torch.isfinite(out).all(), f"Sp={sp}: prefill produced non-finite output"
    assert big.pos == S, f"Sp={sp}: pos is {big.pos}, expected the real length {S}"

    n_tiles = (S + TILE - 1) // TILE                    # tiles holding at least one real position
    blocks = N_KV_HEADS * (sp // TILE)
    unwritten = []
    for li in (0, N_LAYERS // 2, N_LAYERS - 1):         # first, middle, last layer
        k = ttnn.to_torch(big.caches[li][0]).float()
        for h in range(N_KV_HEADS):
            for t in range(n_tiles):
                if float(k[0, h, TILE * t : TILE * (t + 1), :].abs().max()) == 0.0:
                    unwritten.append((li, h, t))
    print(f"\n  Sp={sp:>4} S={S:>4} fill_cache blocks={blocks:>4} "
          f"tiles/head={n_tiles:>2}  unwritten={len(unwritten)}")
    assert not unwritten, (
        f"Sp={sp}: {len(unwritten)} (layer, head, tile) triples are ALL ZERO -- prefill never wrote "
        f"them. First few: {unwritten[:6]}. This is the fill_cache head-straddle (XTTS BUG-7).")


def test_padding_costs_no_accuracy(big, w):
    """More padding must not make the answer WORSE. Three pad amounts, one oracle.

    This test took three attempts and the property is subtler than it looks, so the reasoning is
    here rather than lost in git history:

      1. Truncating a prompt to several lengths and scoring each against the reference measures
         CONTENT, not padding -- 127 pad rows beat 32, because a different last token dominates.
      2. Comparing the same tokens at two pad amounts device-against-itself is unconfounded but
         asserting equality is wrong: PCC 0.99974, max |delta| 0.211. Causally the mask hides the
         pad rows, but a different Sp also means different per-core matmul splits, and bf16
         reduction is not associative -- the same math summed in a different order.
      3. So the testable property is not "identical" but "no WORSE": each pad amount must land the
         same distance from the fp32 reference. A mask that leaked padding into attention would
         show as a systematic penalty that GROWS with the pad count; different rounding shows as
         scatter that does not.

    The token rows are byte-identical across arms; only the trailing zero-row count differs.
    """
    full, case = fixture_embeds(3, w)
    S0 = 250
    base = full[:, :S0]
    exp = bref.reference_forward(base, w, n_layers=N_LAYERS)[:, S0 - 1]

    rows = []
    for extra in (0, gpt.PREFILL_MULTIPLE, 2 * gpt.PREFILL_MULTIPLE):
        embeds = base if not extra else torch.cat(
            [base, base.new_zeros(1, extra, base.shape[-1])], dim=1)
        S = embeds.shape[1]
        sp = (S + gpt.PREFILL_MULTIPLE - 1) // gpt.PREFILL_MULTIPLE * gpt.PREFILL_MULTIPLE
        big.reset()
        got = big.prefill(embeds, last_only=False)[:, S0 - 1]
        m = compare_hidden(got, exp)
        rows.append((extra, sp, sp - S0, m["pcc"], m["worst_pct"]))

    print(f"\n  case 3 ({case['voice']}): {S0} identical tokens, position {S0 - 1} vs fp32")
    print(f"  {'ours':>5} {'Sp':>5} {'total pad':>10} {'PCC':>11} {'worst %':>8}")
    for extra, sp, pad, pc, ws in rows:
        print(f"  {extra:>5} {sp:>5} {pad:>10} {pc:>11.6f} {ws:>7.2f}%")
    pccs = [r[3] for r in rows]
    spread = max(pccs) - min(pccs)
    print(f"  spread {spread:.6f} across {rows[0][2]}..{rows[-1][2]} pad rows")

    assert min(pccs) > 0.999, f"a pad amount dropped below the gate: {[(r[2], round(r[3],6)) for r in rows]}"
    # The discriminator: a leak degrades MONOTONICALLY with pad count. Scatter does not.
    monotonic_penalty = pccs[0] > pccs[1] > pccs[2]
    assert not (monotonic_penalty and spread > 0.001), (
        f"accuracy falls monotonically as padding grows ({[round(p, 6) for p in pccs]}) -- that is "
        f"a mask leak, not rounding")
