# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The backbone decode on device against the fp32 reference, teacher-forced on real frames.

Covers the per-prompt horizon, full utterances, determinism, the cache entries decode writes, the
prompt cache staying untouched, tile boundaries, servable and unservable cache lengths, stack depth,
and a full cache raising.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/pcc/test_backbone_decode_pcc.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    DIM,
    N_LAYERS,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    as_device_k_layout,
    backbone_state,
    case_ids,
    fixture_embeds,
    ill_conditioned_frames,
    long_frame_cases,
    needs_checkpoint,
    real_frames_long,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

# Every teacher-forced comparison uses the prompt's OWN recorded trajectory.
PCC_DECODE = 0.999
# Lower: a minimum over every well-conditioned frame of all 15 prompts, each with an isolated hard frame.
PCC_DECODE_HORIZON = 0.997
# Ill-conditioned frames skip the floors above but must each clear this one.
PCC_DECODE_ILL_CONDITIONED = 0.995
CACHE_PCC = 0.998
TILE = 32
MAX_SEQ = 1024
HORIZON = 64  # frames per prompt in the breadth sweep
DEPTHS = (1, 6, 13, 20, N_LAYERS)
# sdpa_decode requires the cache length to be a multiple of its k_chunk_size (512): cache sizes
# a caller may and may not ask for, other than the 1024 and 2048 the suite otherwise uses.
VALID_MAX_SEQ = (512, 1536)
REJECTED_MAX_SEQ = (256, 736, 992)
CACHE_CASE = 0


@pytest.fixture(scope="module")
def dev():
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def gen(dev, w):
    return TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=MAX_SEQ)


def _prefill_both(gen, w, ci, n_layers=N_LAYERS):
    """-> (reference backbone, prompt length). Both sides prefilled on the same real prompt."""
    embeds, _ = fixture_embeds(ci, w)
    inc = bref.IncrementalBackbone(w, n_layers=n_layers)
    inc.prefill(embeds)
    gen.reset()
    gen.prefill(embeds, last_only=True)
    assert gen.pos == inc.pos == embeds.shape[1]
    return inc, embeds.shape[1]


def _steps(gen, inc, w, n, frames=None):
    """-> (per-step pcc, per-step worst-sample). Teacher-forced on real frames."""
    frames = real_frames_long(CACHE_CASE) if frames is None else frames
    pcs, wss = [], []
    for t in range(min(n, frames.shape[0])):
        emb = bref.embed_frame(w, frames[t])
        h_ref = inc.step(emb)
        h_dev = gen.step(emb)
        m = compare_hidden(h_dev, h_ref)
        pcs.append(m["pcc"])
        wss.append(m["worst_pct"])
    return pcs, wss


@pytest.mark.parametrize("ci", case_ids(), ids=lambda c: f"case{c}")
def test_decode_pcc_over_the_horizon(gen, w, ci):
    """Up to HORIZON frames of this prompt's own trajectory, with the per-step trend reported."""
    frames = real_frames_long(ci)[:HORIZON]
    inc, P = _prefill_both(gen, w, ci)
    pcs, wss = _steps(gen, inc, w, frames.shape[0], frames=frames)
    ill = sorted(ill_conditioned_frames(ci) & set(range(len(pcs))))
    gated = [p for t, p in enumerate(pcs) if t not in ill]
    q = max(1, len(pcs) // 4)
    print(
        f"\n  case {ci} P={P}, {len(pcs)} frames: min PCC {min(gated):.6f} over {len(gated)} gated  "
        f"first-quarter mean {sum(pcs[:q])/q:.6f}  last-quarter mean {sum(pcs[-q:])/q:.6f}  "
        f"worst-sample max {max(wss):.2f}% | ill-conditioned " + (", ".join(f"{pcs[t]:.6f}@{t}" for t in ill) or "none")
    )
    assert min(gated) > PCC_DECODE_HORIZON, f"case {ci} decode min PCC {min(gated):.6f} over {len(gated)} frames"
    assert min(pcs) > PCC_DECODE_ILL_CONDITIONED, f"case {ci} decode min PCC {min(pcs):.6f} over all {len(pcs)} frames"


def test_decode_is_bit_deterministic(gen, w):
    """The same config re-run must reproduce bit-identically."""
    frames = real_frames_long(CACHE_CASE)
    embeds, _ = fixture_embeds(CACHE_CASE, w)

    def run():
        gen.reset()
        gen.prefill(embeds, last_only=True)
        return [gen.step(bref.embed_frame(w, frames[t])).clone() for t in range(4)]

    a, b = run(), run()
    for t, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"decode step {t} not reproducible"


def test_decode_writes_the_cache_correctly(gen, w):
    """The entries decode appends at [P, P+k), all 26 layers, against the reference's cache."""
    n = 8
    inc, P = _prefill_both(gen, w, CACHE_CASE)
    _steps(gen, inc, w, n)
    weak = []
    for li in range(N_LAYERS):
        k_ref, v_ref = inc.cache[f"layers.{li}."]
        for side, dev_t, ref_t in (
            ("K", ttnn.to_torch(gen.caches[li][0]).float(), as_device_k_layout(k_ref.float())),
            ("V", ttnn.to_torch(gen.caches[li][1]).float(), v_ref.float()),
        ):
            m = compare_hidden(dev_t[:, :, P : P + n, :], ref_t[:, :, P : P + n, :])
            if m["pcc"] <= CACHE_PCC:
                weak.append((li, side, round(m["pcc"], 6)))
    print(f"\n  {N_LAYERS} layers x (K,V) at decode positions [{P}, {P + n}): {len(weak)} below {CACHE_PCC}")
    assert not weak, f"decode wrote cache entries below {CACHE_PCC}: {weak[:8]}"


def test_decode_does_not_disturb_the_prompt_cache(gen, w):
    """Decode must leave the prompt's positions exactly as prefill wrote them."""
    n = 4
    inc, P = _prefill_both(gen, w, CACHE_CASE)
    before = [(ttnn.to_torch(k).float()[:, :, :P, :], ttnn.to_torch(v).float()[:, :, :P, :]) for k, v in gen.caches]
    _steps(gen, inc, w, n)
    moved = []
    for li, (k, v) in enumerate(gen.caches):
        after = (ttnn.to_torch(k).float()[:, :, :P, :], ttnn.to_torch(v).float()[:, :, :P, :])
        for side, b, a in (("K", before[li][0], after[0]), ("V", before[li][1], after[1])):
            if not torch.equal(b, a):
                moved.append((li, side))
    assert not moved, f"{len(moved)} cache regions inside [0, {P}) changed over {n} steps: {moved[:6]}"


def test_decode_across_a_cache_tile_boundary(gen, w):
    """Stepping past a multiple of the tile height starts a new cache tile."""
    inc, P = _prefill_both(gen, w, CACHE_CASE)
    n = min(real_frames_long(CACHE_CASE).shape[0], (P // TILE + 2) * TILE - P)
    crossings = [t for t in range(n) if (P + t) % TILE == 0]
    pcs, _ = _steps(gen, inc, w, n)
    # The steps walk case 0's own frames, ill-conditioned ones included.
    ill = ill_conditioned_frames(CACHE_CASE)
    gated = [p for t, p in enumerate(pcs) if t not in ill]
    at_crossing = [pcs[t] for t in crossings if t < len(pcs) and t not in ill]
    print(
        f"\n  P={P}, {len(pcs)} steps, crossings at {crossings}: "
        f"min over gated steps {min(gated):.6f}, min at a crossing "
        f"{min(at_crossing) if at_crossing else float('nan'):.6f}, min overall {min(pcs):.6f}"
    )
    assert at_crossing, f"P={P} with {n} steps crosses no well-conditioned tile boundary; pick a different case"
    assert min(at_crossing) > PCC_DECODE, f"decode min PCC {min(at_crossing):.6f} at a tile crossing"
    assert min(gated) > PCC_DECODE_HORIZON, f"decode min PCC {min(gated):.6f} across a tile boundary"
    assert min(pcs) > PCC_DECODE_ILL_CONDITIONED, f"decode min PCC {min(pcs):.6f} across a tile boundary, all steps"


@pytest.mark.parametrize("max_seq", VALID_MAX_SEQ, ids=lambda n: f"maxseq{n}")
def test_decode_at_other_valid_cache_lengths(dev, w, max_seq):
    """Cache sizes other than the two the rest of the suite uses must decode just as well."""
    g = TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=max_seq)
    inc, P = _prefill_both(g, w, CACHE_CASE)
    pcs, _ = _steps(g, inc, w, 8)
    print(f"\n  max_seq_len {max_seq} ({max_seq // TILE} tiles), P={P}: min PCC {min(pcs):.6f}")
    assert min(pcs) > PCC_DECODE, f"decode min PCC {min(pcs):.6f} at max_seq_len {max_seq}"


@pytest.mark.parametrize("max_seq", REJECTED_MAX_SEQ, ids=lambda n: f"maxseq{n}")
def test_a_cache_length_sdpa_cannot_serve_fails_loudly(dev, w, max_seq, expect_error):
    """A cache length that is not a multiple of sdpa's k_chunk_size must raise on first use rather
    than return something wrong (the op enforces it, not the constructor)."""
    g = TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=max_seq)
    embeds, _ = fixture_embeds(CACHE_CASE, w)
    g.reset()
    with expect_error(Exception, "must be multiple of chunk size"):
        g.prefill(embeds, last_only=True)
        g.step(bref.embed_frame(w, real_frames_long(CACHE_CASE)[0]))


@pytest.mark.parametrize("depth", DEPTHS, ids=lambda d: f"depth{d}")
def test_decode_matches_reference_at_each_depth(dev, w, depth):
    """Prefill and decode at a shortened stack, so a failure localises to a depth range."""
    g = TtVoxtralGPT(dev, n_layers=depth, state=w, max_seq_len=MAX_SEQ)
    embeds, _ = fixture_embeds(CACHE_CASE, w)
    exp = bref.reference_forward(embeds, w, n_layers=depth)
    g.reset()
    got = g.prefill(embeds, last_only=False)
    m_pre = compare_hidden(got, exp)
    inc, P = _prefill_both(g, w, CACHE_CASE, n_layers=depth)
    pcs, _ = _steps(g, inc, w, 4)
    print(f"\n  depth {depth}: prefill pooled {m_pre['pcc']:.6f}  decode min {min(pcs):.6f}")
    assert m_pre["pcc"] > PCC_DECODE, f"depth {depth} prefill pooled PCC {m_pre['pcc']:.6f}"
    assert min(pcs) > PCC_DECODE, f"depth {depth} decode min PCC {min(pcs):.6f}"


def test_step_refuses_a_full_cache(dev, w, expect_error):
    """Stepping past max_seq_len must raise, not wrap or overwrite."""
    small = 512  # the smallest cache sdpa_decode will serve
    g = TtVoxtralGPT(dev, n_layers=1, state=w, max_seq_len=small)
    g.reset()
    g.prefill(torch.zeros(1, small, DIM), last_only=True)
    assert g.pos == small
    with expect_error(ValueError, "cache full"):
        g.step(torch.zeros(1, DIM))


# The two longest utterances only; a full solve for every prompt is too slow for the suite.
LONG_CASES = tuple(sorted(long_frame_cases(), key=lambda c: -real_frames_long(c).shape[0])[:2])


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("ci", LONG_CASES, ids=lambda c: f"case{c}")
def test_decode_pcc_over_a_full_utterance(gen, w, ci):
    """A whole utterance, teacher-forced on its own frames, with the trend reported by decile,
    so drift over a real request's length has somewhere to show."""
    frames = real_frames_long(ci)  # this prompt's own trajectory
    inc, P = _prefill_both(gen, w, ci)
    pcs, wss = _steps(gen, inc, w, frames.shape[0], frames=frames)
    d = max(1, len(pcs) // 10)
    deciles = [
        sum(pcs[k * d : (k + 1) * d]) / len(pcs[k * d : (k + 1) * d]) for k in range(10) if pcs[k * d : (k + 1) * d]
    ]
    print(
        f"\n  case {ci} P={P}, {len(pcs)} frames ({len(pcs) / 12.5:.1f}s audio): "
        f"min PCC {min(pcs):.6f}  worst-sample max {max(wss):.2f}%"
    )
    print("    mean PCC by decile: " + " ".join(f"{v:.6f}" for v in deciles))
    assert min(pcs) > PCC_DECODE, f"case {ci} decode min PCC {min(pcs):.6f} over {len(pcs)} frames"
    assert (
        deciles[-1] > deciles[0] - 0.0005
    ), f"decode degrades across the utterance: first decile {deciles[0]:.6f}, last {deciles[-1]:.6f}"


def test_a_mismatched_trajectory_does_not_break_decode(gen, w):
    """Robustness only: a prompt fed another utterance's frames must stay finite and bounded. No
    accuracy is asserted -- nobody sends this request."""
    inc, P = _prefill_both(gen, w, CACHE_CASE)
    other = real_frames_long(LONG_CASES[0])[:16]
    for t in range(other.shape[0]):
        emb = bref.embed_frame(w, other[t])
        h_ref = inc.step(emb)
        h_dev = gen.step(emb)
        assert torch.isfinite(h_dev).all(), f"step {t} produced non-finite values"
        assert (
            h_dev.abs().max() < h_ref.abs().max() * 10
        ), f"step {t} magnitude {h_dev.abs().max():.1f} against reference {h_ref.abs().max():.1f}"
    print(f"\n  {other.shape[0]} off-trajectory steps: finite and bounded")
