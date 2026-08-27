# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 (AR backbone) reference tests.

Two tiers, mirroring what can be checked without an 8 GB non-commercial download:
  * structural + wiring — runs ALWAYS. Every weight the reference asks for must exist in the
    released checkpoint's manifest at the right shape, and the graph must run at real widths
    (random weights, shortened stack) with the KV-cache path reproducing prefill exactly.
  * numerical — runs only when the checkpoint is present (skipped otherwise).

    pytest -svv models/experimental/voxtral_tts/tests/test_backbone_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ATTN_DIM,
    DEFAULT_CKPT,
    DIM,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_DIM,
    N_LAYERS,
    ROPE_THETA,
    apply_rope,
    load_manifest,
    pcc,
    random_state_from_manifest,
    rope_cis,
)

PCC_GATE = 0.999
TEST_LAYERS = 2  # real widths, shortened depth: 26 layers of fp32 weights do not fit in RAM
needs_ckpt = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")


def _layer_keys(n):
    out = ["norm.weight"]
    for i in range(n):
        out += [
            f"layers.{i}.{k}.weight"
            for k in ("attention.wq", "attention.wk", "attention.wv", "attention.wo",
                      "attention_norm", "ffn_norm", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
        ]
    return out


def _short_state(n=TEST_LAYERS, seed=0):
    """Random weights at real shapes, keyed the way the reference expects (no `.weight` suffix
    on layer entries — load_backbone_state strips it)."""
    raw = random_state_from_manifest(keys=_layer_keys(n), seed=seed)
    w = {"norm": raw["norm.weight"]}
    for k, v in raw.items():
        if k.startswith("layers."):
            w[k[: -len(".weight")]] = v
    return w


def test_every_reference_weight_exists_in_checkpoint():
    """The reference must not invent or misname a single tensor."""
    man = load_manifest()
    expect = _layer_keys(N_LAYERS) + [
        "mm_audio_embeddings.tok_embeddings.weight",
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
    ]
    missing = [k for k in expect if k not in man]
    assert not missing, f"reference asks for {len(missing)} tensors not in the checkpoint: {missing[:5]}"
    assert len(expect) == 9 * N_LAYERS + 3, "expected 9 tensors per layer + norm + 2 embedding tables"


@pytest.mark.parametrize(
    "name,shape",
    [
        ("layers.0.attention.wq.weight", (ATTN_DIM, DIM)),
        ("layers.0.attention.wk.weight", (KV_DIM, DIM)),
        ("layers.0.attention.wv.weight", (KV_DIM, DIM)),
        ("layers.0.attention.wo.weight", (DIM, ATTN_DIM)),
        ("layers.0.feed_forward.w1.weight", (HIDDEN_DIM, DIM)),
        ("layers.0.feed_forward.w2.weight", (DIM, HIDDEN_DIM)),
    ],
)
def test_projection_shapes(name, shape):
    """Pins the asymmetry that is easy to get wrong: the attention interior is 4096 wide while
    the residual stream is 3072, so wq/wo are NOT square and NOT transposes of each other."""
    assert tuple(load_manifest()[name]["shape"]) == shape


def test_forward_runs_at_real_widths():
    w = _short_state()
    x = torch.randn(1, 7, DIM) * 0.1
    out = ref.reference_forward(x, w, n_layers=TEST_LAYERS)
    assert out.shape == (1, 7, DIM)
    assert torch.isfinite(out).all()


def test_kv_cache_path_matches_prefill():
    """Incremental decode must equal the full causal forward — the invariant a traced TTNN
    decode step has to preserve, and the one that catches RoPE-offset bugs."""
    w = _short_state()
    x = torch.randn(1, 8, DIM) * 0.1
    full = ref.reference_forward(x, w, n_layers=TEST_LAYERS)
    P = 5
    pre, steps = ref.reference_prefill_then_step(x[:, :P], w, x[:, P:], n_layers=TEST_LAYERS)
    assert pcc(pre, full[:, :P]) > 0.9999, "prefill diverges from the full forward"
    assert pcc(steps, full[:, P:]) > 0.9999, "cached steps diverge from the full forward"
    assert torch.allclose(steps, full[:, P:], atol=2e-4), "cached steps not numerically equal"


def test_causality():
    """Truncating the input must not change earlier positions' outputs."""
    w = _short_state()
    x = torch.randn(1, 9, DIM) * 0.1
    full = ref.reference_forward(x, w, n_layers=TEST_LAYERS)
    short = ref.reference_forward(x[:, :6], w, n_layers=TEST_LAYERS)
    assert torch.allclose(full[:, :6], short, atol=2e-4), "attention is leaking future positions"


def test_rope_is_interleaved_pairs_not_half_split():
    """Guards the convention choice. Mistral-native RoPE rotates ADJACENT pairs (0,1),(2,3),...
    A half-split (HF-style) implementation would rotate (0, d/2) instead, which is a silent,
    accuracy-only failure. Here: rotating position 0 must be identity, and a pure-pair input
    must stay inside its own pair."""
    cis = rope_cis(4, HEAD_DIM, ROPE_THETA)
    x = torch.randn(1, 1, 4, HEAD_DIM)
    out = apply_rope(x, cis)
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6), "position 0 rotation is not identity"
    e = torch.zeros(1, 1, 4, HEAD_DIM)
    e[..., 0] = 1.0  # excite only dim 0; its partner under interleaving is dim 1
    r = apply_rope(e, cis)
    moved = r[0, 0, 1].abs()
    assert moved[1] > 1e-3, "dim 0 did not couple to dim 1 (not interleaved)"
    assert moved[HEAD_DIM // 2] < 1e-6, "dim 0 coupled to dim d/2 (half-split convention)"


def test_frame_embedding_offsets_are_disjoint_and_in_range():
    """The 37 codebooks share one flat table; offsets must not overlap and must fit the table."""
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import codebook_offsets, codebook_sizes

    off, sizes = codebook_offsets().tolist(), codebook_sizes()
    table = load_manifest()["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]["shape"][0]
    for i in range(len(off) - 1):
        assert off[i] + sizes[i] == off[i + 1], f"codebook {i} slice overlaps the next"
    assert off[-1] + sizes[-1] <= table, f"codebooks need {off[-1] + sizes[-1]} rows, table has {table}"


def test_single_and_batched_frame_embedding_agree():
    man = load_manifest()
    w = {"audio_embeddings": random_state_from_manifest(
        keys=["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
    )["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]}
    assert w["audio_embeddings"].shape[0] == man[
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]["shape"][0]
    _, frames = ref.make_synthetic_inputs(n_text=1, n_frames=3)
    one = ref.embed_frame(w, frames[0])
    many = ref.embed_frames(w, frames)
    assert torch.allclose(one, many[:, :1], atol=1e-6)


@needs_ckpt
def test_real_weights_cache_matches_prefill():
    w = ref.load_backbone_state()
    text_ids, frames = ref.make_synthetic_inputs()
    x = torch.cat([ref.embed_text(w, text_ids), ref.embed_frames(w, frames)], dim=1)
    full = ref.reference_forward(x, w)
    P = x.shape[1] - 3
    pre, steps = ref.reference_prefill_then_step(x[:, :P], w, x[:, P:])
    assert pcc(pre, full[:, :P]) > PCC_GATE
    assert pcc(steps, full[:, P:]) > PCC_GATE


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
