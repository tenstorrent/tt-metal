# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill at every padded shape, and independence from the amount of padding.

Prefill quantises its sequence to a multiple of `PREFILL_MULTIPLE`, so a cache of `max_seq_len`
rows has a fixed set of reachable shapes, each its own kernel shape and `fill_cache` work split.

  * every shape   -- hidden states and all 26 layers of KV cache against fp32, plus a check that
    every (head, position-tile) of the cache was written.
  * no outlier    -- pooled PCC must not vary across shapes.
  * padding       -- the same tokens at three pad amounts must land the same distance from fp32.
  * over-long     -- a prompt that pads past `max_seq_len` must raise.

Shapes beyond the longest real prompt are driven by the fixture's texts joined and repeated, so
their bar is a collapse floor rather than an accuracy gate; real-prompt accuracy lives in
test_backbone_pcc.py and test_all_voices_smoke.py.

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
    as_device_k_layout,
    backbone_state,
    fixture_embeds,
    long_prompt_embeds,
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
SHAPE_SPREAD = 0.008        # no shape may compute unlike its neighbours
SHAPE_WORST_SAMPLE_PCT = 15.0   # PCC alone hides a single far-off element


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


# Filled in by the parametrized test, read by the cross-shape test that follows.
_POOLED: dict = {}


@pytest.mark.parametrize("sp", SHAPES, ids=lambda s: f"Sp{s}")
def test_every_padded_prefill_shape_is_correct(big, w, sp):
    """Hidden states and all 26 layers of KV cache against fp32, at this shape.

    K needs the head-dim permutation; V does not, being unrotated.
    """
    S = sp - 5                                          # reach the shape WITH padding
    embeds, repeated = long_prompt_embeds(S, w)
    gate = SHAPE_PCC_FLOOR

    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)       # all positions
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    inc.prefill(embeds)                                             # populates the reference cache
    big.reset()
    out = big.prefill(embeds, last_only=False)
    assert torch.isfinite(out).all(), f"Sp={sp}: non-finite output"
    assert big.pos == S, f"Sp={sp}: pos {big.pos}, expected {S}"
    assert out.shape == exp.shape, f"Sp={sp}: {tuple(out.shape)} vs {tuple(exp.shape)}"
    m = compare_hidden(out, exp)
    m_last = compare_hidden(out[:, S - 1], exp[:, S - 1])
    _POOLED[sp] = m["pcc"]

    # every layer, both sides, values -- plus the explicit zero check, which names the tile
    n_tiles = (S + TILE - 1) // TILE
    unwritten, weak = [], []
    for li in range(N_LAYERS):
        k_dev = ttnn.to_torch(big.caches[li][0]).float()[:, :, :S, :]
        v_dev = ttnn.to_torch(big.caches[li][1]).float()[:, :, :S, :]
        k_ref, v_ref = inc.cache[f"layers.{li}."]
        for side, got, ref in (("K", k_dev, as_device_k_layout(k_ref.float())),
                               ("V", v_dev, v_ref.float())):
            c = compare_hidden(got, ref)
            if c["pcc"] <= gate:
                weak.append((li, side, round(c["pcc"], 6)))
        for h in range(N_KV_HEADS):
            for t in range(n_tiles):
                if float(k_dev[0, h, TILE * t : TILE * (t + 1), :].abs().max()) == 0.0:
                    unwritten.append((li, h, t))

    print(f"\n  Sp={sp:>4} S={S:>4} blocks={N_KV_HEADS * (sp // TILE):>4} "
          f"{'repeated' if repeated else 'joined':>8} text  pooled {m['pcc']:.6f}  "
          f"last {m_last['pcc']:.6f}  worst {m['worst_pct']:.2f}%  cache weak {len(weak)}/52  "
          f"unwritten {len(unwritten)}")
    assert not unwritten, (
        f"Sp={sp}: {len(unwritten)} (layer, head, tile) blocks are ALL ZERO -- prefill never wrote "
        f"them. First few: {unwritten[:6]}.")
    assert not weak, f"Sp={sp}: cache entries below {gate}: {weak[:8]}"
    assert m["pcc"] > gate, (
        f"Sp={sp}: pooled PCC {m['pcc']:.6f} over all {S} positions -- below the collapse floor "
        f"{gate}")
    assert m["worst_pct"] < SHAPE_WORST_SAMPLE_PCT, (
        f"Sp={sp}: pooled worst sample {m['worst_pct']:.2f}% even though pooled PCC is {m['pcc']:.6f}")


def test_no_shape_computes_differently_from_its_neighbours():
    """No shape may compute unlike its neighbours. Needs the whole sweep, so skipped under -k."""
    if len(_POOLED) < len(SHAPES):
        pytest.skip(f"needs all {len(SHAPES)} shapes; have {len(_POOLED)}")
    lo, hi = min(_POOLED.values()), max(_POOLED.values())
    print(f"\n  pooled PCC across {len(_POOLED)} shapes: {lo:.6f} .. {hi:.6f} "
          f"(spread {hi - lo:.6f})")
    assert hi - lo < SHAPE_SPREAD, (
        f"pooled PCC varies by {hi - lo:.6f} across shapes -- one shape computes differently from "
        f"its neighbours: {sorted((k, round(v, 6)) for k, v in _POOLED.items())}")


def test_padding_costs_no_accuracy(big, w):
    """The same tokens at three pad amounts must land the same distance from fp32.

    Not an equality check: a different padded length also changes the per-core matmul split, and
    bf16 reduction is not associative. A leak would degrade monotonically with the pad count.
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


def test_prefill_refuses_a_prompt_longer_than_the_cache(big, w):
    """A prompt that pads beyond `max_seq_len` must raise, not overflow the cache."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM

    too_long = big.max_seq_len + 1
    with pytest.raises(ValueError, match="pads to"):
        big.prefill(torch.zeros(1, too_long, DIM), last_only=True)
