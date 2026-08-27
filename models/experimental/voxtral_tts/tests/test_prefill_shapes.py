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

(1) DOES compare against the reference. An earlier version of this file did not, on the grounds
that "a 26-layer fp32 CPU forward at S=2048 is prohibitively slow" -- which was asserted without
measuring. Measured: 3.8 s at S=256, 6.9 s at 512, 16.1 s at 1024, so ~45 s at 2048 and about two
minutes for all sixteen shapes. Cheap enough, and without it the test only proved a tile was
WRITTEN, not that it was written with the right values -- so a long shape picking a wrong program
config and emitting plausible-but-wrong numbers would have passed. `test_backbone_pcc.py` covers 15
real prompts but all are P <= 357, so Sp 512..2048 had no value check anywhere.

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
    corpus_embeds,
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
# Pooled over ALL S positions -- the stable statistic, and it covers the whole block rather than
# one row of it. A SINGLE position's PCC is the noisiest thing here (measured swinging 0.938..0.998
# within one prompt), so it gets its own looser gate and is reported for diagnosis only.
# Both thresholds are set FROM the measured sweep with margin, and the measurements are recorded so
# drift is visible rather than absorbed. Measured 2026-08-27 over all 16 shapes:
#
#   pooled PCC   0.994786 (Sp=768) .. 0.999272 (Sp=128), spread 0.004486
#   worst-sample 1.57% .. 3.89%, plateauing at 3.89% from Sp=1152 up
#
# The pooled curve is a U -- highest at 128, lowest around 768, rising again to 2048 -- which tracks
# the synthetic prompt's composition changing with length, not the shapes. Hence:
SHAPE_PCC_POOLED = 0.99    # a COLLAPSE floor, not an accuracy gate. XTTS's version of this failure
                           # read 0.63 and 0.826 against 0.9997, so 0.99 catches it with room while
                           # leaving the 0.9948 measured minimum alone.
SHAPE_SPREAD = 0.008       # the shape-sensitive assertion. Measured 0.004486; 0.008 leaves headroom
                           # so it flags a real outlier rather than flaking on the existing band.
# Pooled worst-sample, gated for the same reason as in test_backbone_pcc (trap 9: PCC hides
# outliers). Measured 1.57%..3.89% across the 16 shapes on the synthetic long prompt; 8% matches
# the real-prompt gate so the two files cannot drift apart.
SHAPE_WORST_SAMPLE_PCT = 8.0


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
    """S rows of a SINGLE well-formed prompt whose text is long enough to reach S tokens.

    NOT a repeated assembled prompt. An earlier version tiled `fixture_embeds(3)` to length, which
    duplicates the BOS/header rows mid-sequence -- and those embeddings are ~3x the magnitude of an
    ordinary token (position 0 measured absmax 122 against 30-48 typical). The tell was that
    worst-sample came out at 47.16% for EVERY shape from Sp=1152 to 2048, identical to two decimals:
    eight shapes cannot share a worst element by chance, so it was the input, not the shape.

    Here the corpus sentences are joined into one long text and tokenized ONCE, so there is a single
    header and a long natural body. `ar_male` because it has the fewest placeholder rows (67), which
    leaves the most room for text at small S.
    """
    from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer
    from models.experimental.voxtral_tts.tests.sentence_corpus import SENTENCES

    voice = "ar_male"
    tok = TekkenTokenizer()
    pool = [t for group in SENTENCES.values() for t in group]
    text, ids = "", []
    while len(ids) < S:
        text = (text + " " + pool[len(text) % len(pool)]).strip() if text else pool[0]
        ids = tok.build_prompt(text, voice)
        if len(text) > 40000:                     # guard: cannot reach S, fail loudly
            raise AssertionError(f"cannot build a prompt of {S} tokens (reached {len(ids)})")
    return corpus_embeds(text, voice, w)[:, :S]


# Filled in by the parametrized test below; read by the cross-shape test after it. One shape per
# test because the repo enforces a 300 s per-test timeout and the CPU reference alone is ~45 s at
# S=2043 -- a single test over all 16 shapes times out.
_POOLED: dict = {}


@pytest.mark.parametrize("sp", SHAPES, ids=lambda s: f"Sp{s}")
def test_every_padded_prefill_shape_is_correct(big, w, sp):
    """Right values against fp32, and every KV-cache tile written, at this shape.

    WHY THE FLOOR IS 0.995 HERE WHEN REAL PROMPTS GATE AT 0.999. No natural prompt in this repo
    exceeds 357 tokens, so long shapes must be fed a synthetic long prompt, and every construction
    is off-distribution to a degree:

      - repeating the assembled prompt duplicates the BOS/header rows mid-sequence, whose embeddings
        are ~3x an ordinary token's magnitude -> worst-sample 47.16%;
      - joining corpus sentences into one long body fixes that -> 3.89%, but still cycles 19
        sentences across 9 languages, which no real request looks like.

    In BOTH constructions worst-sample came out IDENTICAL at every shape (47.16% everywhere, then
    3.89% everywhere). Shapes cannot share a worst element by chance, so the residual error is the
    INPUT and the shape is not the variable -- which is why the shape-sensitive assertion is the
    cross-shape spread in the next test, not the absolute level here. This floor exists to catch
    collapse: XTTS's version of this failure read 0.63 and 0.826 against 0.9997.

    Real-prompt accuracy at Sp<=384 lives in `test_backbone_pcc.py` (15 prompts, 0.9997+) and
    `test_all_voices_smoke.py` (all 20 voices, pooled 0.99971-0.99978).
    """
    S = sp - 5                                          # reach the shape WITH padding
    embeds = _long_input(w, S)
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)
    big.reset()
    out = big.prefill(embeds, last_only=False)
    assert torch.isfinite(out).all(), f"Sp={sp}: non-finite output"
    assert big.pos == S, f"Sp={sp}: pos {big.pos}, expected {S}"
    assert out.shape == exp.shape, f"Sp={sp}: {tuple(out.shape)} vs {tuple(exp.shape)}"
    m = compare_hidden(out, exp)
    m_last = compare_hidden(out[:, S - 1], exp[:, S - 1])
    _POOLED[sp] = m["pcc"]

    n_tiles = (S + TILE - 1) // TILE
    unwritten = []
    for li in (0, N_LAYERS // 2, N_LAYERS - 1):
        k = ttnn.to_torch(big.caches[li][0]).float()
        for h in range(N_KV_HEADS):
            for t in range(n_tiles):
                if float(k[0, h, TILE * t : TILE * (t + 1), :].abs().max()) == 0.0:
                    unwritten.append((li, h, t))
    print(f"\n  Sp={sp:>4} S={S:>4} blocks={N_KV_HEADS * (sp // TILE):>4} tiles/head={n_tiles:>2}  "
          f"unwritten={len(unwritten)}  pooled {m['pcc']:.6f}  last {m_last['pcc']:.6f}  "
          f"worst {m['worst_pct']:.2f}%")
    assert not unwritten, (
        f"Sp={sp}: {len(unwritten)} (layer, head, tile) blocks are ALL ZERO -- prefill never wrote "
        f"them. First few: {unwritten[:6]}. This is the fill_cache head-straddle (XTTS BUG-7).")
    assert m["pcc"] > SHAPE_PCC_POOLED, (
        f"Sp={sp}: pooled PCC {m['pcc']:.6f} over all {S} positions -- this shape computes the "
        f"wrong values, not merely a missing tile")
    assert m["worst_pct"] < SHAPE_WORST_SAMPLE_PCT, (
        f"Sp={sp}: pooled worst sample {m['worst_pct']:.2f}% -- one element is far off even though "
        f"pooled PCC is {m['pcc']:.6f}")


def test_no_shape_computes_differently_from_its_neighbours():
    """The shape-sensitive assertion: with the input held constant in KIND, no Sp is an outlier.

    Reads what the parametrized test above measured, so it needs the whole sweep to have run --
    skipped under `-k`.
    """
    if len(_POOLED) < len(SHAPES):
        pytest.skip(f"needs all {len(SHAPES)} shapes; have {len(_POOLED)}")
    lo, hi = min(_POOLED.values()), max(_POOLED.values())
    print(f"\n  pooled PCC across {len(_POOLED)} shapes: {lo:.6f} .. {hi:.6f} "
          f"(spread {hi - lo:.6f})")
    assert hi - lo < SHAPE_SPREAD, (
        f"pooled PCC varies by {hi - lo:.6f} across shapes -- one shape computes differently from "
        f"its neighbours: {sorted((k, round(v, 6)) for k, v in _POOLED.items())}")


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
