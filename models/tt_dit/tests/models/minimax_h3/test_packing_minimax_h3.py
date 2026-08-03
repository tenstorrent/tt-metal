# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only parity gate for the MiniMax-H3 packed sequence.

Every quantity here is a checkpoint contract: the fp64 rotary grid *is* the
audio/video alignment, and the AdaLN table index decides which modulation a row
receives. Drift is silent -- it produces plausible video that does not match the
reference -- so these assertions are exact rather than PCC-based.

The golden digests were taken from a run verified bit-exact against the
`minimax-h3` diffusers branch. When that branch is installed,
``test_matches_diffusers_reference`` re-checks the whole layout against it
directly; otherwise the digests stand in.
"""

import hashlib

import pytest
import torch

from ....pipelines.minimax_h3 import packing as p

# (label, latent_height, latent_width, num_frames, keyframe_anchors)
BRINGUP = ("bringup", 544 // 16, 960 // 16, 124, ("first", "last"))
CANONICAL = ("canonical", 768 // 16, 1344 // 16, 192, ("first",))

# sequence_length, latent frames, audio latents, sha256[:16] of position_ids and token_tags
GOLDEN = {
    "bringup": (21301, 37, 207, "dcd77044930c7e5f", "320f230cf65b8e0b"),
    "canonical": (60101, 57, 320, "3d2878587ddfdc7d", "bc71672198e60490"),
}

TEXT_LEN = 997
# A keyframe's VLM vision block lands inside the text rows tagged as video, not
# text; covering that here keeps the tag plumbing honest.
VISION_BLOCK = slice(20, 70)


def _text_tags():
    tags = torch.ones(TEXT_LEN, dtype=torch.long)
    tags[VISION_BLOCK] = p.MINIMAX_H3_VIDEO_TAG
    return tags


def _layout(latent_height, latent_width, num_frames, anchors):
    return p.build_packed_sequence(
        _text_tags(),
        p.video_latent_num_frames(num_frames),
        latent_height,
        latent_width,
        p.audio_latent_num_frames(num_frames),
        (1, 2, 2),
        anchors,
    )


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("num_frames,expected", [(1, 5), (5, 5), (6, 22), (22, 22), (120, 124), (192, 192)])
def test_align_num_frames(num_frames, expected):
    assert p.align_num_frames(num_frames) == expected


@pytest.mark.parametrize("num_frames,latents", [(5, 2), (22, 7), (39, 12), (124, 37), (192, 57), (362, 107)])
def test_video_latent_num_frames(num_frames, latents):
    assert p.video_latent_num_frames(num_frames) == latents


def test_video_latent_num_frames_rejects_unaligned(expect_error):
    with expect_error(ValueError, "17n \\+ 5"):
        p.video_latent_num_frames(120)


@pytest.mark.parametrize("num_frames,latents", [(124, 207), (192, 320), (362, 603)])
def test_audio_latent_num_frames(num_frames, latents):
    assert p.audio_latent_num_frames(num_frames) == latents


@pytest.mark.parametrize(
    "aspect,canvas",
    [
        ((16, 9), (768, 1344)),
        ((9, 16), (1344, 768)),
        ((1, 1), (768, 768)),
        ((4, 3), (768, 1024)),
    ],
)
def test_resolve_canvas_size(aspect, canvas):
    assert p.resolve_canvas_size(*aspect) == canvas
    # Both axes must be VAE- and patch-compatible, which is the real constraint.
    assert canvas[0] % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert canvas[1] % p.MINIMAX_H3_CANVAS_MULTIPLE == 0


@pytest.mark.parametrize(
    "aspect,message",
    [
        ((5, 1), "aspect ratios from 1:4 to 4:1"),
        ((1, 5), "aspect ratios from 1:4 to 4:1"),
        ((0, 1), "must be positive"),
        ((-16, 9), "must be positive"),
    ],
)
def test_resolve_canvas_size_rejects_out_of_range(aspect, message, expect_error):
    with expect_error(ValueError, message):
        p.resolve_canvas_size(*aspect)


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_layout_matches_golden(case):
    label, latent_height, latent_width, num_frames, anchors = case
    sequence_length, latent_frames, audio_latents, position_sha, tag_sha = GOLDEN[label]

    assert p.video_latent_num_frames(num_frames) == latent_frames
    assert p.audio_latent_num_frames(num_frames) == audio_latents

    layout = _layout(latent_height, latent_width, num_frames, anchors)
    assert layout.sequence_length == sequence_length
    assert layout.position_ids.dtype == torch.float64
    assert _digest(layout.position_ids) == position_sha
    assert _digest(layout.token_tags) == tag_sha


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_layout_structure(case):
    _, latent_height, latent_width, num_frames, anchors = case
    layout = _layout(latent_height, latent_width, num_frames, anchors)
    rows_per_frame = (latent_height // 2) * (latent_width // 2)

    assert layout.num_condition_video_rows == len(anchors) * rows_per_frame
    assert layout.num_condition_audio_rows == 0

    # Row order is [text | cond | audio | video]: the three index sets must
    # partition the sequence with no overlap and no gap.
    covered = torch.cat([layout.text_indices, layout.video_indices, layout.audio_indices]).sort().values
    assert torch.equal(covered, torch.arange(layout.sequence_length))

    assert torch.equal(layout.token_tags[layout.audio_indices[:1]], torch.tensor([p.MINIMAX_H3_AUDIO_TAG]))
    assert (layout.token_tags[layout.video_indices] == p.MINIMAX_H3_VIDEO_TAG).all()
    # The vision block inside the text rows is video-tagged, everything else text.
    assert (layout.token_tags[VISION_BLOCK] == p.MINIMAX_H3_VIDEO_TAG).all()
    assert (layout.token_tags[VISION_BLOCK.stop : TEXT_LEN] == p.MINIMAX_H3_TEXT_TAG).all()

    # Audio rows carry no height coordinate and pin to the width-grid extremes.
    audio_position_ids = layout.position_ids[layout.audio_indices]
    assert (audio_position_ids[:, 1] == 0).all()
    assert audio_position_ids[:, 2].unique().numel() == 2


def test_keyframe_anchor_times():
    """The `first` anchor coincides with frame 0; the `last` anchor overshoots the end.

    `last` is `origin + span(n) - 5/3`, and 5/3 is the *shortest* per-frame span
    (the `1` of the `(1, 4, 4, 4, 4)` pattern), not the span of the final frame.
    Whenever `(n - 1) % 5 != 0` that final frame is a `4`, so the anchor lands
    past the last target frame rather than on it -- 5.0 units past, for both
    working points. Asserted here because it looks like an off-by-one and is not.
    """
    _, latent_height, latent_width, num_frames, _ = BRINGUP
    latent_frames = p.video_latent_num_frames(num_frames)
    layout = _layout(latent_height, latent_width, num_frames, ("first", "last"))
    rows_per_frame = (latent_height // 2) * (latent_width // 2)

    first_time = layout.position_ids[TEXT_LEN, 0]
    last_time = layout.position_ids[TEXT_LEN + rows_per_frame, 0]
    video_time = layout.position_ids[layout.video_indices[layout.num_condition_video_rows :], 0]

    assert first_time == float(TEXT_LEN)
    assert first_time == video_time.min()
    assert float(last_time) == pytest.approx(
        TEXT_LEN + p._temporal_position_span(latent_frames) - p._ROPE_FRAME_RESCALE
    )
    assert last_time > video_time.max()
    assert float(last_time - video_time.max()) == pytest.approx(5.0)


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_row_timesteps_pin_condition_rows(case):
    _, latent_height, latent_width, num_frames, anchors = case
    layout = _layout(latent_height, latent_width, num_frames, anchors)
    # Row timesteps are fp32, so round the expectations through fp32 before
    # comparing -- 0.7 is not representable and would fail an exact check.
    video_t, audio_t, cond_t = torch.tensor([0.7, 0.5, p.MINIMAX_H3_KEYFRAME_NOISE_AUG], dtype=torch.float32).tolist()
    timesteps, indices = p.build_row_timesteps(layout, video_t, audio_t, cond_t, cond_t)

    assert timesteps.tolist() == sorted({video_t, audio_t, cond_t})
    resolved = timesteps[indices]
    condition_rows = layout.video_indices[: layout.num_condition_video_rows]
    target_rows = layout.video_indices[layout.num_condition_video_rows :]

    assert (resolved[condition_rows] == cond_t).all()
    assert (resolved[target_rows] == video_t).all()
    assert (resolved[layout.audio_indices] == audio_t).all()
    # Text rows never reach an output head and inherit the video timestep.
    assert (resolved[layout.text_indices] == video_t).all()


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_adaln_index_ranges_round_trip(case):
    _, latent_height, latent_width, num_frames, anchors = case
    layout = _layout(latent_height, latent_width, num_frames, anchors)
    _, indices = p.build_row_timesteps(layout, 0.7, 0.5, 0.999, 0.999)
    table_rows = p.adaln_indices(layout.token_tags, indices)

    runs = p.adaln_index_ranges(table_rows)
    rebuilt = torch.empty_like(table_rows)
    for start, stop, value in runs:
        rebuilt[start:stop] = value
    assert torch.equal(rebuilt, table_rows)

    # Contiguous runs are what let the device apply modulation as slice plus
    # broadcast; if this ever grew to O(sequence_length) the design would have
    # to fall back to a gather.
    assert len(runs) <= 8


def test_adaln_index_ranges_are_shard_local():
    """A sequence-parallel shard must see few runs, not just the whole sequence."""
    _, latent_height, latent_width, num_frames, anchors = CANONICAL
    layout = _layout(latent_height, latent_width, num_frames, anchors)
    _, indices = p.build_row_timesteps(layout, 0.7, 0.5, 0.999, 0.999)
    table_rows = p.adaln_indices(layout.token_tags, indices)

    sp_factor = 8
    shard = layout.sequence_length // sp_factor
    counts = [len(p.adaln_index_ranges(table_rows[i * shard : (i + 1) * shard])) for i in range(sp_factor)]

    # Only the shard holding the text/cond/audio prefix is fragmented; the rest
    # are pure video and modulate with a single broadcast.
    assert sum(count == 1 for count in counts) >= sp_factor - 2
    assert max(counts) <= 8


def test_adaln_index_ranges_empty():
    assert p.adaln_index_ranges(torch.empty(0, dtype=torch.long)) == []


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_patchify_round_trip(case):
    _, latent_height, latent_width, num_frames, _ = case
    latent_frames = p.video_latent_num_frames(num_frames)
    latents = torch.randn(1, 24, latent_frames, latent_height, latent_width)

    rows = p.patchify_video_latents(latents, (1, 2, 2))
    assert rows.shape == (latent_frames * (latent_height // 2) * (latent_width // 2), 24 * 4)
    restored = p.unpatchify_video_tokens(rows, latent_frames, latent_height, latent_width, 24, (1, 2, 2))
    assert torch.equal(restored, latents)


def test_unpack_audio_tokens_is_channel_major():
    num_audio_latents, channels = 207, p.MINIMAX_H3_AUDIO_CHANNELS
    rows = torch.randn(num_audio_latents * channels, 32)
    unpacked = p.unpack_audio_tokens(rows, num_audio_latents)

    assert unpacked.shape == (channels, 32, num_audio_latents)
    # Channel-major: the first block of rows is the left channel in full.
    assert torch.equal(unpacked[0], rows[:num_audio_latents].transpose(0, 1))
    assert torch.equal(unpacked[1], rows[num_audio_latents:].transpose(0, 1))


@pytest.mark.parametrize("case", [BRINGUP, CANONICAL], ids=lambda c: c[0])
def test_matches_diffusers_reference(case):
    """Bit-exact cross-check against the `minimax-h3` diffusers branch."""
    reference = pytest.importorskip(
        "diffusers.modular_pipelines.minimax_h3.packing",
        reason="requires the minimax-h3 diffusers branch",
    )
    _, latent_height, latent_width, num_frames, anchors = case
    kwargs = dict(
        text_token_tags=_text_tags(),
        num_latent_frames=p.video_latent_num_frames(num_frames),
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=p.audio_latent_num_frames(num_frames),
        patch_size=(1, 2, 2),
        keyframe_anchors=anchors,
    )
    ours = p.build_packed_sequence(**kwargs)
    theirs = reference.build_packed_sequence(**kwargs)

    assert ours.sequence_length == theirs.sequence_length
    assert torch.equal(ours.position_ids, theirs.position_ids)
    assert torch.equal(ours.token_tags, theirs.token_tags)
    assert torch.equal(ours.video_indices, theirs.video_indices)
    assert torch.equal(ours.audio_indices, theirs.audio_indices)
    assert ours.num_condition_video_rows == theirs.num_condition_video_rows
