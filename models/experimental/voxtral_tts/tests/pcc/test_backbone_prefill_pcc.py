# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The backbone prefill on device against the fp32 reference: one-layer wiring, all 15 fixture
prompts (pooled and last position, each with a worst-sample bound), and every KV-cache entry.
Decode is test_backbone_decode_pcc.py; every padded shape is test_prefill_shapes.py.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/pcc/test_backbone_prefill_pcc.py [-k case0]
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    DIM,
    HEAD_DIM,
    N_KV_HEADS,
    N_LAYERS,
    ROPE_THETA,
    causal_bias,
    pcc,
    rope_cis,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    as_device_k_layout,
    backbone_state,
    case_ids,
    fixture_embeds,
    ill_conditioned_positions,
    needs_checkpoint,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

# Every test here opens a device, the one-layer wiring test included. Module-level, so a new
# test cannot miss the mark.
pytestmark = [pytest.mark.slow, needs_checkpoint]

# Gate constants, each just below what the fixture prompts reach.
PCC_PREFILL = 0.999
CACHE_CASES = (0, 2, 3, 12)  # P = 100..357
CACHE_PCC = 0.998
# A single position's PCC is too noisy to gate, so its minimum is printed; the last position's worst
# sample, the row the flow model consumes, is gated.
MAX_WORST_SAMPLE_PCT = 5.0
MAX_POOLED_WORST_SAMPLE_PCT = 8.0  # over every gated position, so looser than the last position's bound
# The accuracy gates skip the fixture's ill-conditioned positions; the whole prompt, those included,
# still has to clear this collapse floor.
COLLAPSE_FLOOR = 0.99


@pytest.fixture(scope="module")
def dev():
    """The model's own opener, not the repo `device` fixture: this block needs the pipeline's
    l1_small and trace-region settings."""
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def gen(dev, w):
    return TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=1024)


def test_one_layer_wiring_pcc(dev):
    """One layer against the reference, where a rotation-convention error shows.

    Random inputs are fine for this test alone: it checks wiring, not accuracy."""
    S = 128
    one = TtVoxtralGPT(dev, n_layers=1)
    ws = bref.load_backbone_state()
    torch.manual_seed(0)
    x = torch.randn(1, S, DIM) * 0.02
    exp = bref._layer(x, ws, "layers.0.", rope_cis(S, HEAD_DIM, ROPE_THETA), causal_bias(S, torch.float32))
    got = one.prefill(x, apply_final_norm=False)
    got_pcc = compare_hidden(got, exp)["pcc"]
    print(f"\n  [1 layer prefill] PCC {got_pcc:.8f}  maxabs {(got - exp).abs().max():.3e}")
    assert got_pcc > 0.999, f"one-layer wiring PCC {got_pcc:.6f} -- suspect the RoPE convention"


@pytest.mark.parametrize("ci", case_ids(), ids=lambda c: f"case{c}")
def test_prefill_pcc(gen, w, ci):
    """Full 26-layer prefill on a real prompt, pooled over the well-conditioned positions and at the
    last position."""
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)
    got = gen.prefill(embeds)
    ill = sorted(ill_conditioned_positions(ci))
    keep = [i for i in range(P) if i not in ill]
    m_all = compare_hidden(got[:, keep], exp[:, keep])
    all_pcc = m_all["pcc"]
    everywhere = compare_hidden(got, exp)["pcc"]
    m_last = compare_hidden(got[:, -1:], exp[:, -1:])
    last_pcc = m_last["pcc"]
    # The pipeline calls prefill_last, a different op sequence: slice one row then norm it, versus
    # norm every row then index. Asserted equal so the gates above cover the shipped path.
    gen.reset()
    shipped = gen.prefill(embeds, last_only=True).reshape(1, -1)
    per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
    wi = min(keep, key=lambda i: per[i])
    print(
        f"\n  case {ci} ({case['voice']}, P={P}): PCC gated {all_pcc:.6f} ({len(keep)} positions)  "
        f"last {last_pcc:.6f}  worst-sample last {m_last['worst_pct']:.2f}% pooled {m_all['worst_pct']:.2f}%  "
        f"min per-pos {per[wi]:.6f} (@{wi}) | all {P} positions {everywhere:.6f}, ill-conditioned "
        + (", ".join(f"{per[i]:.3f}@{i}" for i in ill) or "none")
    )
    assert last_pcc > PCC_PREFILL, f"case {ci} prefill last-position PCC {last_pcc:.6f}"
    assert all_pcc > PCC_PREFILL, f"case {ci} prefill PCC {all_pcc:.6f} over the {len(keep)} gated positions"
    assert everywhere > COLLAPSE_FLOOR, f"case {ci} prefill PCC {everywhere:.6f} over all {P} positions"
    ws = m_last["worst_pct"]
    assert ws < MAX_WORST_SAMPLE_PCT, f"case {ci} last-position worst sample {ws:.2f}% of reference scale"
    assert m_all["worst_pct"] < MAX_POOLED_WORST_SAMPLE_PCT, (
        f"case {ci} pooled worst sample {m_all['worst_pct']:.2f}% over {len(keep)} positions -- one "
        f"element is far off even though pooled PCC is {all_pcc:.6f}"
    )
    assert torch.equal(shipped, got[:, -1]), (
        f"case {ci}: prefill_last (the call the pipeline makes) differs from prefill(last_only="
        f"False)[:, -1] by max {(shipped - got[:, -1]).abs().max():.3e} -- the two paths have "
        f"diverged, and only the last_only=False one is covered by the gates above"
    )


def _reference_cache(w, embeds):
    """-> {layer_index: (k, v)} after a reference prefill, each [1, N_KV_HEADS, P, HEAD_DIM]."""
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    inc.prefill(embeds)
    return {i: inc.cache[f"layers.{i}."] for i in range(N_LAYERS)}, inc


def _device_cache(gen, P):
    """-> {layer_index: (k, v)} sliced to the prompt's P positions."""
    out = {}
    for i, (kc, vc) in enumerate(gen.caches):
        out[i] = (ttnn.to_torch(kc).float()[:, :, :P, :], ttnn.to_torch(vc).float()[:, :, :P, :])
    return out


@pytest.mark.parametrize("ci", CACHE_CASES, ids=lambda c: f"case{c}")
def test_prefill_kv_cache_matches_reference(gen, w, ci):
    """Every cached K and V entry at the well-conditioned positions, all 26 layers, against the
    reference's own cache."""
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    ref_cache, _ = _reference_cache(w, embeds)
    gen.reset()
    gen.prefill(embeds)
    dev_cache = _device_cache(gen, P)
    # An ill-conditioned position's K/V diverge from the layer where its hidden state does.
    ill = sorted(ill_conditioned_positions(ci))
    keep = torch.tensor([i for i in range(P) if i not in ill])

    rows = []
    for i in range(N_LAYERS):
        for side, j in (("K", 0), ("V", 1)):
            exp, got = ref_cache[i][j].float(), dev_cache[i][j]
            if side == "K":
                exp = as_device_k_layout(exp)  # reference_helpers explains why
            assert (
                exp.shape == got.shape
            ), f"layer {i} {side}: reference {tuple(exp.shape)} vs device {tuple(got.shape)}"
            exp, got = exp[:, :, keep], got[:, :, keep]
            m = compare_hidden(got, exp)
            # worst position, so a failure names one instead of a whole layer
            per_pos = (got - exp).abs().amax(dim=(0, 1, 3))
            rows.append((i, side, m["pcc"], m["worst_pct"], int(keep[per_pos.argmax()])))

    worst_pcc = min(r[2] for r in rows)
    worst_ws = max(r[3] for r in rows)
    print(
        f"\n  case {ci} ({case['voice']}), P={P}, {N_LAYERS} layers x (K,V), "
        f"cache [{1}, {N_KV_HEADS}, {P}, {HEAD_DIM}], gated on {len(keep)} positions (ill-conditioned: {ill})"
    )
    for i, side, pc, ws, pos in sorted(rows, key=lambda r: r[2])[:5]:
        print(f"    weakest: layer {i:>2} {side}  PCC {pc:.6f}  worst-sample {ws:.2f}%  @pos {pos}")
    print(f"  worst PCC {worst_pcc:.6f}, worst-sample {worst_ws:.2f}% across all {len(rows)} (layer, side) pairs")
    bad = [(i, s, pc) for i, s, pc, _, _ in rows if pc <= CACHE_PCC]
    assert not bad, "cache entries below the gate: " + ", ".join(f"layer {i} {s} PCC {pc:.6f}" for i, s, pc in bad)
