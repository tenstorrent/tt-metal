# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill at every padded shape (multiples of PREFILL_MULTIPLE up to max_seq_len): hidden states
and every KV-cache entry against fp32, no shape unlike its neighbours, padding-amount independence, and
an over-long prompt raising. Long shapes use joined fixture texts, so they carry a collapse floor,
not an accuracy gate.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/pcc/test_prefill_shapes.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    N_KV_HEADS,
    N_LAYERS,
    pcc,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    as_device_k_layout,
    backbone_state,
    fixture_embeds,
    long_prompt_embeds,
    needs_checkpoint,
)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

MAX_SEQ = 2048
SHAPES = tuple(range(gpt.PREFILL_MULTIPLE, MAX_SEQ + 1, gpt.PREFILL_MULTIPLE))  # 128 .. 2048
TILE = 32

# Gates are on PCC pooled over all S positions; a single position is too noisy to gate, so it is
# reported only. Each constant sits just below what every shape reaches.
SHAPE_PCC_FLOOR = 0.99  # collapse floor, not an accuracy gate
SHAPE_SPREAD = 0.008  # no shape may compute unlike its neighbours
SHAPE_WORST_SAMPLE_PCT = 15.0  # PCC alone hides a single far-off element
# Rows below the collapse floor sit out the worst-sample and cache checks only while they stay
# scattered, as rounding chaos is and a structural fault is not.
MAX_COLLAPSED_PER_TILE = TILE // 2
MAX_COLLAPSED_SHARE = 0.15


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


# Pooled PCC per shape, recorded by whichever test prefills that shape first.
_POOLED: dict = {}


def _prefill_at(big, w, sp):
    """Prefill a prompt that pads to `sp`; -> (S, embeds, repeated, exp, out). Records the pooled PCC."""
    S = sp - 5  # reach the shape WITH padding
    embeds, repeated = long_prompt_embeds(S, w)
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)  # all positions
    big.reset()
    out = big.prefill(embeds, last_only=False)
    assert torch.isfinite(out).all(), f"Sp={sp}: non-finite output"
    _POOLED[sp] = compare_hidden(out, exp)["pcc"]
    return S, embeds, repeated, exp, out


@pytest.mark.parametrize("sp", SHAPES, ids=lambda s: f"Sp{s}")
def test_every_padded_prefill_shape_is_correct(big, w, sp):
    """Hidden states and all 26 layers of KV cache against fp32, at this shape.

    K needs the head-dim permutation; V does not, being unrotated.
    """
    gate = SHAPE_PCC_FLOOR
    S, embeds, repeated, exp, out = _prefill_at(big, w, sp)
    assert big.pos == S, f"Sp={sp}: pos {big.pos}, expected {S}"
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    inc.prefill(embeds)  # populates the reference cache
    assert out.shape == exp.shape, f"Sp={sp}: {tuple(out.shape)} vs {tuple(exp.shape)}"
    m = compare_hidden(out, exp)
    m_last = compare_hidden(out[:, S - 1], exp[:, S - 1])
    n_tiles = (S + TILE - 1) // TILE
    per = [pcc(out[0, i], exp[0, i]) for i in range(S)]
    collapsed = [i for i in range(S) if per[i] <= gate]
    keep = [i for i in range(S) if per[i] > gate]
    per_tile = [sum(1 for i in collapsed if i // TILE == t) for t in range(n_tiles)]
    m_keep = compare_hidden(out[:, keep], exp[:, keep])

    # every layer, both sides, values -- plus the explicit zero check, which names the tile
    unwritten, weak = [], []
    for li in range(N_LAYERS):
        k_dev = ttnn.to_torch(big.caches[li][0]).float()[:, :, :S, :]
        v_dev = ttnn.to_torch(big.caches[li][1]).float()[:, :, :S, :]
        k_ref, v_ref = inc.cache[f"layers.{li}."]
        for side, got, ref in (("K", k_dev, as_device_k_layout(k_ref.float())), ("V", v_dev, v_ref.float())):
            c = compare_hidden(got[:, :, keep], ref[:, :, keep])
            if c["pcc"] <= gate:
                weak.append((li, side, round(c["pcc"], 6)))
        for h in range(N_KV_HEADS):
            for t in range(n_tiles):
                if float(k_dev[0, h, TILE * t : TILE * (t + 1), :].abs().max()) == 0.0:
                    unwritten.append((li, h, t))

    print(
        f"\n  Sp={sp:>4} S={S:>4} blocks={N_KV_HEADS * (sp // TILE):>4} "
        f"{'repeated' if repeated else 'joined':>8} text  pooled {m['pcc']:.6f}  "
        f"last {m_last['pcc']:.6f}  worst {m_keep['worst_pct']:.2f}% ({m['worst_pct']:.2f}% with the collapsed)  "
        f"collapsed {len(collapsed)} (max {max(per_tile)}/{TILE} per tile)  cache weak {len(weak)}/52  "
        f"unwritten {len(unwritten)}"
    )
    assert not unwritten, (
        f"Sp={sp}: {len(unwritten)} (layer, head, tile) blocks are ALL ZERO -- prefill never wrote "
        f"them. First few: {unwritten[:6]}."
    )
    assert max(per_tile) <= MAX_COLLAPSED_PER_TILE, (
        f"Sp={sp}: {max(per_tile)} of {TILE} rows collapsed in tile {per_tile.index(max(per_tile))} -- "
        f"clustered, so a structural fault rather than scattered rounding chaos"
    )
    assert len(collapsed) <= MAX_COLLAPSED_SHARE * S, f"Sp={sp}: {len(collapsed)} of {S} rows collapsed"
    assert not weak, f"Sp={sp}: cache entries below {gate}: {weak[:8]}"
    assert (
        m["pcc"] > gate
    ), f"Sp={sp}: pooled PCC {m['pcc']:.6f} over all {S} positions -- below the collapse floor {gate}"
    assert (
        m_keep["worst_pct"] < SHAPE_WORST_SAMPLE_PCT
    ), f"Sp={sp}: worst sample {m_keep['worst_pct']:.2f}% over {len(keep)} positions though pooled PCC is {m['pcc']:.6f}"


@pytest.mark.timeout(1800)  # prefills every shape this run skipped
def test_no_shape_computes_differently_from_its_neighbours(big, w):
    """No shape may compute unlike its neighbours. Reuses the sweep's scores and prefills any shape
    this run has not, so a reordered, filtered or split run still checks every shape."""
    missing = [sp for sp in SHAPES if sp not in _POOLED]
    for sp in missing:
        _prefill_at(big, w, sp)
    lo, hi = min(_POOLED.values()), max(_POOLED.values())
    print(
        f"\n  pooled PCC across {len(_POOLED)} shapes: {lo:.6f} .. {hi:.6f} (spread {hi - lo:.6f}); "
        f"{len(missing)} prefilled here"
    )
    assert hi - lo < SHAPE_SPREAD, (
        f"pooled PCC varies by {hi - lo:.6f} across shapes -- one shape computes differently from "
        f"its neighbours: {sorted((k, round(v, 6)) for k, v in _POOLED.items())}"
    )


def test_padding_costs_no_accuracy(big, w):
    """The same tokens at three pad amounts must land the same distance from fp32 -- not equal,
    since the padded length changes the matmul split; a leak would degrade with the pad count."""
    full, case = fixture_embeds(3, w)
    S0 = 250
    base = full[:, :S0]
    exp = bref.reference_forward(base, w, n_layers=N_LAYERS)[:, S0 - 1]

    rows = []
    for extra in (0, gpt.PREFILL_MULTIPLE, 2 * gpt.PREFILL_MULTIPLE):
        embeds = base if not extra else torch.cat([base, base.new_zeros(1, extra, base.shape[-1])], dim=1)
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
        f"a mask leak, not rounding"
    )


def test_prefill_refuses_a_prompt_longer_than_the_cache(big, w, expect_error):
    """A prompt that pads beyond `max_seq_len` must raise, not overflow the cache."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM

    too_long = big.max_seq_len + 1
    with expect_error(ValueError, "pads to"):
        big.prefill(torch.zeros(1, too_long, DIM), last_only=True)
