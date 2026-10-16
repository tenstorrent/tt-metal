# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request-path invariants of the TTNN codec decoder (codec), on device.

Chunked attention == unchunked, bucketing trims back to T and pads with the last frame, the bias
and prepared-weight caches stay bounded, and the slab is tile-aligned.
Skips cleanly without ttnn, a device, or the checkpoint.

    pytest -svv models/experimental/voxtral_tts/tests/test_codec_request_path.py
"""

import os

import pytest

from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, pcc

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable")
# Every test here opens a device, so it is `slow`: `-m "not slow"` is the host-only subset.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}"),
]

WAVE_PCC = 0.999  # same gate as tests/pcc/test_codec_pcc.py
STAGE_PCC = 0.996  # per-stage gate, as in tests/pcc/test_codec_pcc.py


@pytest.fixture(scope="module")
def device():
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import L1_SMALL_SIZE

    d = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def pair(device):
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    return TtVoxtralCodecDecoder(device), ref.load_codec_state()


@pytest.mark.parametrize("n_frames", [64, 469])
def test_chunked_matches_unchunked(device, n_frames):
    """Chunking must be EXACT: attention is causal and windowed, so each slab holds all the context
    its kept rows need. Compares the two paths directly rather than both against the reference."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(n_frames)
    un = TtVoxtralCodecDecoder(device, chunk_min=None)(codes)
    ch = TtVoxtralCodecDecoder(device, chunk_min=0, slab=512)(codes)
    assert pcc(ch, un) > 0.9999, f"chunked diverges from unchunked at T={n_frames}"


def test_bias_cache_does_not_grow_with_utterance_length(device):
    """Every chunk is padded to `slab`, so chunked stages hold ONE bias per window whatever the
    length; stages with S <= slab keep an SxS bias."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    for n in (700, 1000, 1200):  # all well above slab, so every stage chunks
        gen(ref.make_synthetic_codes(n))
    chunked = {(S, w) for (S, w, _) in gen._bias_cache if S == gen.slab}
    assert len(chunked) <= 4, f"expected <=4 slab biases (one per window), got {sorted(chunked)}"
    assert len(gen._bias_cache) <= 6, f"cache grew to {len(gen._bias_cache)} across 3 lengths"


@pytest.mark.parametrize("n_frames", [64, 65, 130, 469])
def test_bucketing_preserves_length_and_accuracy(device, n_frames):
    """Bucketed output is trimmed to exactly T frames and matches the reference."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    w = ref.load_codec_state()
    codes = ref.make_synthetic_codes(n_frames)
    exp = ref.reference_decode(codes, w)
    got = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    assert got.shape == exp.shape == (1, 1, n_frames * 1920), "bucketed output not trimmed back to T"
    assert pcc(got, exp) > WAVE_PCC


def test_bucketing_pads_with_last_frame_not_zeros(device):
    """Bucketed (last-frame padded) and unbucketed decodes agree, so the pad does not leak into
    kept audio."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    codes = ref.make_synthetic_codes(70)  # 70 -> bucket 128, so 58 frames of padding
    bucketed = TtVoxtralCodecDecoder(device, bucket=128)(codes)
    plain = TtVoxtralCodecDecoder(device, bucket=None)(codes)
    assert pcc(bucketed, plain) > 0.999, "padding is leaking into the kept region"


def test_prepared_weights_are_deduplicated(device):
    """Content dedup keeps the prepared-weight cache at <= 8 layouts for 4 convs x 4 buckets."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder

    gen = TtVoxtralCodecDecoder(device)
    for b in (128, 256, 512, 1024):  # four different buckets -> 16 (conv, length) pairs
        gen(ref.make_synthetic_codes(b))
    entries, distinct, mb = gen.prepared_weight_stats()
    assert entries == 16, f"expected 16 (conv,length) entries, got {entries}"
    assert distinct <= 8, f"distinct layouts grew to {distinct} (was 8); dedup may have broken"
    assert mb < 150, f"prepared weights hold {mb:.0f} MB; dedup regressed (naive would be ~240 MB)"


def test_slab_is_tile_aligned():
    """TILE_LAYOUT pads every dim to 32, so an unaligned slab silently wastes tiles."""
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import SLAB

    assert SLAB % 32 == 0, f"slab {SLAB} is not tile-aligned"
