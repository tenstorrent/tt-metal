# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 decode on device against the fp32 reference.

Decode advances one position per audio frame off the KV cache. Everything here is teacher-forced on
real frames, so both sides step on the same embedding and each frame is an independent measurement.

  * horizon        -- 64 frames on every prompt, against one shared recording, so most prompts are
    a mismatched pair: a stress case rather than a faithful one.
  * full utterance -- a whole request, each prompt teacher-forced with its own trajectory.
  * determinism    -- a repeat must reproduce bit-identically.
  * cache writes   -- the entries decode appends, all 26 layers, against the reference's own cache.
  * prompt cache   -- decode must leave the prompt's positions exactly as prefill wrote them.
  * tile boundary  -- stepping across a multiple of the tile height starts a new cache tile.
  * cache length   -- sdpa_decode serves only cache lengths that are a multiple of its k_chunk_size;
    valid alternatives must decode correctly and the rest must fail loudly rather than quietly.
  * depth          -- prefill and decode at several stack depths, so a failure localises.
  * full cache     -- stepping past max_seq_len must raise.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_backbone_decode_pcc.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    DIM,
    N_KV_HEADS,
    N_LAYERS,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    as_device_k_layout,
    backbone_state,
    case_ids,
    fixture_embeds,
    long_frame_cases,
    real_frames,
    real_frames_long,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

# Two gates, because the two horizon tests feed different things.
#
# The 64-frame sweep runs every prompt against ONE shared frame recording, so all but one prompt is
# a mismatched pair -- a deliberate stress case, and the looser gate belongs to it.
PCC_DECODE = 0.998
# The full-utterance test gives each prompt its OWN recorded trajectory, which is what a real request
# looks like, so it holds the tighter line.
PCC_DECODE_MATCHED = 0.999
CACHE_PCC = 0.998
TILE = 32
MAX_SEQ = 1024
DEPTHS = (1, 6, 13, 20, N_LAYERS)
# sdpa_decode requires the cache length to be a multiple of its k_chunk_size (512), so these are
# the cache sizes a caller may and may not ask for. The suite otherwise only ever uses 1024 and 2048.
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
    frames = real_frames() if frames is None else frames
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
def test_decode_pcc_over_the_full_horizon(gen, w, ci):
    """Every frame the fixture holds, with the per-step trend reported."""
    inc, P = _prefill_both(gen, w, ci)
    pcs, wss = _steps(gen, inc, w, real_frames().shape[0])
    q = max(1, len(pcs) // 4)
    print(f"\n  case {ci} P={P}, {len(pcs)} frames: min PCC {min(pcs):.6f}  "
          f"first-quarter mean {sum(pcs[:q])/q:.6f}  last-quarter mean {sum(pcs[-q:])/q:.6f}  "
          f"worst-sample max {max(wss):.2f}%")
    assert min(pcs) > PCC_DECODE, f"case {ci} decode min PCC {min(pcs):.6f} over {len(pcs)} frames"


def test_decode_is_bit_deterministic(gen, w):
    """The same config re-run must reproduce bit-identically."""
    frames = real_frames()
    embeds, _ = fixture_embeds(0, w)

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
    print(f"\n  {N_LAYERS} layers x (K,V) at decode positions [{P}, {P + n}): "
          f"{len(weak)} below {CACHE_PCC}")
    assert not weak, f"decode wrote cache entries below {CACHE_PCC}: {weak[:8]}"


def test_decode_does_not_disturb_the_prompt_cache(gen, w):
    """Decode must leave the prompt's positions exactly as prefill wrote them."""
    n = 4
    inc, P = _prefill_both(gen, w, CACHE_CASE)
    before = [(ttnn.to_torch(k).float()[:, :, :P, :], ttnn.to_torch(v).float()[:, :, :P, :])
              for k, v in gen.caches]
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
    n = min(real_frames().shape[0], (P // TILE + 2) * TILE - P)
    crossings = [t for t in range(n) if (P + t) % TILE == 0]
    pcs, _ = _steps(gen, inc, w, n)
    at_crossing = [pcs[t] for t in crossings if t < len(pcs)]
    print(f"\n  P={P}, {len(pcs)} steps, crossings at {crossings}: "
          f"min overall {min(pcs):.6f}, min at a crossing "
          f"{min(at_crossing) if at_crossing else float('nan'):.6f}")
    assert crossings, f"P={P} with {n} steps crosses no tile boundary; pick a different case"
    assert min(pcs) > PCC_DECODE, f"decode min PCC {min(pcs):.6f} across a tile boundary"


@pytest.mark.parametrize("max_seq", VALID_MAX_SEQ, ids=lambda n: f"maxseq{n}")
def test_decode_at_other_valid_cache_lengths(dev, w, max_seq):
    """Cache sizes other than the two the rest of the suite uses must decode just as well."""
    g = TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=max_seq)
    inc, P = _prefill_both(g, w, CACHE_CASE)
    pcs, _ = _steps(g, inc, w, 8)
    print(f"\n  max_seq_len {max_seq} ({max_seq // TILE} tiles), P={P}: min PCC {min(pcs):.6f}")
    assert min(pcs) > PCC_DECODE, f"decode min PCC {min(pcs):.6f} at max_seq_len {max_seq}"


@pytest.mark.parametrize("max_seq", REJECTED_MAX_SEQ, ids=lambda n: f"maxseq{n}")
def test_a_cache_length_sdpa_cannot_serve_fails_loudly(dev, w, max_seq):
    """A cache length sdpa_decode cannot serve must raise, not return something wrong.

    The constraint is a multiple of the program config's k_chunk_size, and it is enforced inside the
    op rather than at construction, so the failure surfaces on the first prefill or step.
    """
    g = TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=max_seq)
    embeds, _ = fixture_embeds(CACHE_CASE, w)
    g.reset()
    with pytest.raises(Exception):
        g.prefill(embeds, last_only=True)
        g.step(bref.embed_frame(w, real_frames()[0]))


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


def test_step_refuses_a_full_cache(dev, w):
    """Stepping past max_seq_len must raise, not wrap or overwrite."""
    small = 512                                   # the smallest cache sdpa_decode will serve
    g = TtVoxtralGPT(dev, n_layers=1, state=w, max_seq_len=small)
    g.reset()
    g.prefill(torch.zeros(1, small, DIM), last_only=True)
    assert g.pos == small
    with pytest.raises(ValueError, match="cache full"):
        g.step(torch.zeros(1, DIM))


LONG_CASES = tuple(c for c in long_frame_cases() if real_frames_long(c).shape[0] > 128)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("ci", LONG_CASES, ids=lambda c: f"case{c}")
def test_decode_pcc_over_a_full_utterance(gen, w, ci):
    """A whole request's worth of frames, teacher-forced, with the trend reported by decile.

    The short fixture covers the first few seconds; this walks the length a real utterance runs, so
    any drift over that span has somewhere to show.
    """
    frames = real_frames_long(ci)          # this prompt's own trajectory
    inc, P = _prefill_both(gen, w, ci)
    pcs, wss = _steps(gen, inc, w, frames.shape[0], frames=frames)
    d = max(1, len(pcs) // 10)
    deciles = [sum(pcs[k * d : (k + 1) * d]) / len(pcs[k * d : (k + 1) * d])
               for k in range(10) if pcs[k * d : (k + 1) * d]]
    print(f"\n  case {ci} P={P}, {len(pcs)} frames ({len(pcs) / 12.5:.1f}s audio): "
          f"min PCC {min(pcs):.6f}  worst-sample max {max(wss):.2f}%")
    print("    mean PCC by decile: " + " ".join(f"{v:.6f}" for v in deciles))
    assert min(pcs) > PCC_DECODE_MATCHED, (
        f"case {ci} decode min PCC {min(pcs):.6f} over {len(pcs)} frames")
    assert deciles[-1] > deciles[0] - 0.0005, (
        f"decode degrades across the utterance: first decile {deciles[0]:.6f}, "
        f"last {deciles[-1]:.6f}")
