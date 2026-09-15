# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The HiFT length graph, checked against the captured reference.

Assembling this vocoder is mostly a shape problem. There are ~40 convolutions,
two transposed convolutions, a reflection pad, a strided source branch that has
to re-converge with the main path, and an inverse STFT at the end -- and an
off-by-one anywhere surfaces as a broadcasting failure at a residual add, deep in
the graph, long after the mistake.

`shape_trace()` computes the whole chain in pure Python, so these tests run
anywhere and catch that entire class of bug before any device time is spent.
Every expected number here is read from `tests/golden/manifest.json`, i.e. from
tensors the real vocoder actually produced.
"""
from __future__ import annotations

import json
import os

import pytest

from models.demos.cosyvoice.tt.hifigan.generator import shape_trace

MEL_FRAMES = 282  # the captured utterance: 3.3 s of audio


def _manifest():
    from models.demos.cosyvoice.tt.common import GOLDEN_DIR

    path = os.path.join(GOLDEN_DIR, "manifest.json")
    if not os.path.exists(path):
        pytest.skip("run scripts/gen_golden.py in the CosyVoice venv first")
    with open(path) as fh:
        return json.load(fh)["modules"]


def _shape(mods, module, array):
    return mods[module]["arrays"][f"call0.{array}"][0]


# --------------------------------------------------------------------------
# the chain, end to end
# --------------------------------------------------------------------------
def test_total_upsample_closes_the_loop():
    """282 mel frames must produce exactly 282 * 256 audio samples."""
    t = shape_trace(MEL_FRAMES)
    assert t["total_upsample"] == 256
    assert t["audio_length"] == MEL_FRAMES * 256 == 72192
    assert t["waveform_length"] == t["audio_length"], (t["waveform_length"], t["audio_length"])


def test_stage_lengths_match_captured_reference():
    """Every intermediate length against what the reference actually emitted."""
    mods = _manifest()
    t = shape_trace(MEL_FRAMES)
    s0, s1 = t["stages"]

    # conv_pre: 80 -> 512, length unchanged
    assert _shape(mods, "hift.conv_pre", "in_x") == [1, 80, MEL_FRAMES]
    assert _shape(mods, "hift.conv_pre", "out_y") == [1, 512, MEL_FRAMES]

    # ups[0]: 512 -> 256, x8
    assert _shape(mods, "hift.upsample0", "out_y") == [1, s0.out_channels, s0.up_length]
    assert s0.up_length == 2256, s0.up_length

    # ups[1]: 256 -> 128, x8
    assert _shape(mods, "hift.upsample1", "in_x") == [1, s1.in_channels, s1.in_length]
    assert _shape(mods, "hift.upsample1", "out_y") == [1, s1.out_channels, s1.up_length]
    assert s1.up_length == 18048, s1.up_length

    # conv_post sees the REFLECTION-PADDED length, one more than ups[1] emitted
    assert _shape(mods, "hift.conv_post", "in_x") == [1, 128, t["conv_post_length"]]
    assert t["conv_post_length"] == s1.up_length + 1 == 18049


def test_reflection_pad_is_what_aligns_the_source_branch():
    """The +1 is load-bearing: source_downs[1] is a k=1,s=1 conv over the 18049-frame
    excitation spectrogram, so `x = x + si` only aligns because of the pad."""
    t = shape_trace(MEL_FRAMES)
    s1 = t["stages"][-1]
    assert s1.source_length == t["stft_frames"] == 18049
    assert s1.padded_length == s1.source_length, (s1.padded_length, s1.source_length)
    # ... and without the pad it would not
    assert s1.up_length != s1.source_length


def test_source_branch_converges_at_every_stage():
    """Both stages, not just the last: the strided source_downs[0] must land on
    exactly the length ups[0] produces."""
    t = shape_trace(MEL_FRAMES)
    for st in t["stages"]:
        assert st.padded_length == st.source_length, (st.index, st.padded_length, st.source_length)


def test_stft_frames_match_captured_reference():
    mods = _manifest()
    t = shape_trace(MEL_FRAMES)
    assert _shape(mods, "hift.stft", "in_x") == [1, t["audio_length"]]
    assert _shape(mods, "hift.stft", "out_real") == [1, 9, t["stft_frames"]]


def test_istft_output_matches_captured_waveform():
    mods = _manifest()
    t = shape_trace(MEL_FRAMES)
    assert _shape(mods, "hift.istft", "in_magnitude") == [1, 9, t["conv_post_length"]]
    assert _shape(mods, "hift.istft", "out_waveform") == [1, t["waveform_length"]]
    assert _shape(mods, "hift.decode", "out_speech") == [1, t["waveform_length"]]


# --------------------------------------------------------------------------
# generalisation -- the captured utterance must not be a lucky special case
# --------------------------------------------------------------------------
@pytest.mark.parametrize("frames", [32, 100, 282, 500, 1000])
def test_chain_is_consistent_at_other_lengths(frames):
    """If the source branch only converges at 282 frames, the alignment is an
    accident rather than a property."""
    t = shape_trace(frames)
    assert t["waveform_length"] == frames * 256
    for st in t["stages"]:
        assert st.padded_length == st.source_length, (frames, st.index)
