# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC for the TTNN codec decoder (Block 3) vs the CPU reference.

The reference is itself validated against upstream (30/30, all 8 decoder stages bit-exact or
PCC ~1.0), so matching it here is a real correctness statement, not a self-comparison.

Skips cleanly without ttnn, a device, or the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py
"""

import os

import pytest
import torch

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, pcc

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable")
# Every test here opens a device, so `slow` joins the checkpoint guard: `-m "not slow"` is
# meant to be the host-only subset, and it used to still run these.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT),
                       reason=f"no checkpoint at {DEFAULT_CKPT}"),
]

WAVE_PCC = 0.999  # the gate XTTS-v2's HiFi-GAN port shipped at (0.99946)
STAGE_PCC = 0.996  # per-stage; the final window-16 stage amplifies inherited error (see below)


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0, l1_small_size=65536)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def pair(device):
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    return TtVoxtralCodecDecoder(device), ref.load_codec_state()


@pytest.mark.parametrize("n_frames", [64, 469])
def test_chunked_matches_unchunked(device, n_frames):
    """Chunking must be EXACT, not an approximation: attention is causal AND windowed, so a slab
    starting `window` positions early has all the context its kept rows need. Compares the two
    paths directly rather than both against the reference, so a shared error cannot hide."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(n_frames)
    un = TtVoxtralCodecDecoder(device, chunk_min=None)(codes)
    ch = TtVoxtralCodecDecoder(device, chunk_min=0, slab=512)(codes)
    assert pcc(ch, un) > 0.9999, f"chunked diverges from unchunked at T={n_frames}"


def test_bias_cache_does_not_grow_with_utterance_length(device):
    """Every chunk is padded to `slab`, so above the threshold the cache holds exactly ONE bias
    per window no matter how many different lengths are decoded. Before this, first/last chunk
    lengths varied (the last with S mod C), so each new length added biases AND a kernel compile.

    Stages whose S <= slab still run unchunked and get an SxS bias, so a few length-specific
    entries remain -- that is the conv-side bucketing work, tracked separately."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    for n in (700, 1000, 1200):  # all well above slab, so every stage chunks
        gen(ref.make_synthetic_codes(n))
    chunked = {(S, w) for (S, w, _) in gen._bias_cache if S == gen.slab}
    assert len(chunked) <= 4, f"expected <=4 slab biases (one per window), got {sorted(chunked)}"
    assert len(gen._bias_cache) <= 6, f"cache grew to {len(gen._bias_cache)} across 3 lengths"


@pytest.mark.parametrize("n_frames", [64, 65, 130, 469])
def test_bucketing_preserves_length_and_accuracy(device, n_frames):
    """Bucketing pads T up to a grid so the 5 convs stop recompiling per length (each distinct T
    otherwise costs 5 new conv programs, measured 1-5 s each). The output must still be trimmed to
    exactly T frames and match the reference."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(n_frames)
    exp = ref.reference_decode(codes, w)
    got = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    assert got.shape == exp.shape == (1, 1, n_frames * 1920), "bucketed output not trimmed back to T"
    assert pcc(got, exp) > WAVE_PCC


def test_bucketing_pads_with_last_frame_not_zeros(device):
    """The pad repeats the final frame rather than zero-filling: zeros are a hard edge to the
    causal convs, and the transposed convs overlap, so a pathological tail could in principle
    reach the kept region. Checks the two agree, i.e. the choice is not load-bearing but is safe."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(70)  # 70 -> bucket 128, so 58 frames of padding
    bucketed = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    plain = TtVoxtralCodecDecoder(device, bucket=None)(codes)
    assert pcc(bucketed, plain) > 0.999, "padding is leaking into the kept region"


def test_prepared_weights_are_deduplicated(device):
    """Prepared conv layouts change at only ONE length threshold per conv (and never for up6), so
    keying the cache by length alone stored up to 12 BYTE-IDENTICAL copies: 730 MB for 8 distinct
    layouts. Content dedup brings it to 98 MB with no accuracy question, since the tensors are
    bit-identical.

    Guards the memory ceiling, which matters once the 3.4B backbone shares the device.

    FOUR convs are prepared, not five: `out` left ttnn.conv1d when its halo_gather kernel turned out
    to hang the card (STATUS.md 6.13), so the expected count went 5x4 = 20 -> 4x4 = 16."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    for b in (128, 256, 512, 1024):  # four different buckets -> 16 (conv, length) pairs
        gen(ref.make_synthetic_codes(b))
    entries, distinct, mb = gen.prepared_weight_stats()
    assert entries == 16, f"expected 16 (conv,length) entries, got {entries}"
    assert distinct <= 8, f"distinct layouts grew to {distinct} (was 8); dedup may have broken"
    assert mb < 150, f"prepared weights hold {mb:.0f} MB; dedup regressed (naive would be ~240 MB)"


def test_slab_is_tile_aligned():
    """TILE_LAYOUT pads every dim to 32. A slab of 272 (= 256 chunk + 16 window) silently becomes
    288, wasting a row and a column of tiles — pick the SLAB aligned and derive the chunk from it."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import SLAB

    assert SLAB % 32 == 0, f"slab {SLAB} is not tile-aligned"


