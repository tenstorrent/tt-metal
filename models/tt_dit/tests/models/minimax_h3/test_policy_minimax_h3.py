# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the MiniMax-H3 serving policy: the served envelope must resolve to geometry
the pipeline can pack, and the text budget must sit exactly inside the prompt arena cap."""

import pytest
from PIL import Image

from ....models.vae.minimax_h3.vae_minimax_h3 import DEFAULT_TILE_OVERLAP, DEFAULT_TILE_SIZE, split_tiles
from ....pipelines.minimax_h3 import packing as p
from ....pipelines.minimax_h3 import packing_ref2va as rp
from ....pipelines.minimax_h3 import policy
from ....pipelines.minimax_h3.pipeline_minimax_h3 import (
    MINIMAX_H3_BUCKET_LADDER,
    MINIMAX_H3_REF2VA_BUCKET_LADDER,
    MINIMAX_H3_REF2VA_PRESENTATION_RUNGS,
    MINIMAX_H3_VISION_PATCH_LADDER,
    MiniMaxH3ArenaCaps,
    select_bucket,
    validate_bucket_ladder,
)


@pytest.mark.parametrize("aspect", policy.MINIMAX_H3_ASPECT_RATIOS)
def test_served_ratio_resolves_to_aligned_canvas(aspect):
    height, width = p.resolve_canvas_size(*aspect)
    assert height > 0 and width > 0
    assert height % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert width % p.MINIMAX_H3_CANVAS_MULTIPLE == 0


def test_served_canvases_are_unique_and_ordered():
    canvases = policy.served_canvases()
    assert len(canvases) == len(set(canvases))
    assert canvases[0] == p.resolve_canvas_size(*policy.MINIMAX_H3_ASPECT_RATIOS[0])


def test_decodable_canvases_cover_every_resolvable_ratio():
    canvases = policy.decodable_canvases()
    assert len(canvases) == 95
    multiple = p.MINIMAX_H3_CANVAS_MULTIPLE
    swept = {
        p.resolve_canvas_size(width, height)
        for height in range(multiple, 4096 + 1, multiple)
        for width in range(multiple, 4096 + 1, multiple)
        if p.MINIMAX_H3_MIN_ASPECT_RATIO <= width / height <= p.MINIMAX_H3_MAX_ASPECT_RATIO
    }
    assert swept == set(canvases)
    assert set(policy.served_canvases()) <= swept


def _tower_program_key(n_keyframes: int, canvas: tuple[int, int], alignment: int) -> tuple[int, bool]:
    total = n_keyframes * 4 * (canvas[0] // p.MINIMAX_H3_CANVAS_MULTIPLE) * (canvas[1] // p.MINIMAX_H3_CANVAS_MULTIPLE)
    padded = -(-total // alignment) * alignment
    return padded, n_keyframes == 1 and padded == total


def test_served_envelope_covers_every_fl2va_tower_program():
    alignment = 32 * 32
    units = list(policy.served_envelope("t2va", patch_alignment=alignment))
    assert units[0] == (0, None)
    assert len(units) == len(set(units))

    step = 16
    sides = range(policy.MINIMAX_H3_KEYFRAME_MIN_SIDE, policy.MINIMAX_H3_KEYFRAME_MAX_SIDE + 1, step)
    reachable = {
        p.resolve_canvas_size(width, height)
        for width in sides
        for height in sides
        if max(width, height) <= 4 * min(width, height)
    }
    swept_keys = {_tower_program_key(n, canvas, alignment) for n in (1, 2) for canvas in reachable}
    unit_keys = [_tower_program_key(n, canvas, alignment) for n, canvas in units[1:]]
    assert len(unit_keys) == len(set(unit_keys))
    assert set(unit_keys) == swept_keys
    assert all(canvas in reachable for _n, canvas in units[1:])


def test_served_envelope_t2va_needs_patch_alignment(expect_error):
    with expect_error(ValueError, "patch_alignment"):
        list(policy.served_envelope("t2va"))


def test_served_envelope_covers_ref2va_image_sizes():
    units = list(policy.served_envelope("ref2va"))
    assert len(units) == len(set(units))
    for canvas in policy.served_reference_canvases():
        sizes = policy.served_reference_image_sizes(*canvas)
        assert sizes
        for size in sizes:
            assert (canvas, size) in units


def test_served_reference_canvases_are_one_per_area():
    canvases = policy.served_reference_canvases()
    areas = [height * width for height, width in canvases]
    assert len(areas) == len(set(areas))
    served_areas = {height * width for height, width in policy.served_canvases()}
    assert set(areas) == served_areas


def test_served_reference_image_sizes_are_one_per_token_count():
    for canvas in policy.served_reference_canvases():
        sizes = policy.served_reference_image_sizes(*canvas)
        tokens = [(h // p.MINIMAX_H3_CANVAS_MULTIPLE) * (w // p.MINIMAX_H3_CANVAS_MULTIPLE) for h, w in sizes]
        assert len(tokens) == len(set(tokens))


def test_swept_token_counts_close_over_continuous_aspects():
    """The sweeps sample the ratio space on a finite grid; this probe checks the grid is dense
    enough that no continuous input aspect reaches a token count the grid missed."""
    import random

    multiple = p.MINIMAX_H3_CANVAS_MULTIPLE
    image_tokens_by_canvas = {
        canvas: {(h // multiple) * (w // multiple) for h, w in policy.served_reference_image_sizes(*canvas)}
        for canvas in policy.served_reference_canvases()
    }

    rng = random.Random(0)
    for _ in range(50_000):
        ratio = rng.uniform(0.25, 4.0)
        for target, image_tokens in image_tokens_by_canvas.items():
            size = rp.resolve_reference_image_size(
                round(1000 * ratio), 1000, mode="match", target_width=target[1], target_height=target[0]
            )
            assert (size[0] // multiple) * (size[1] // multiple) in image_tokens


def test_served_envelope_rejects_unknown_task(expect_error):
    with expect_error(NotImplementedError, "is not defined for task"):
        list(policy.served_envelope("fl2va"))


def test_served_reference_image_sizes_deduped_and_stable():
    for canvas in policy.served_canvases():
        sizes = policy.served_reference_image_sizes(*canvas)
        assert sizes == policy.served_reference_image_sizes(*canvas)
        assert len(sizes) == len(set(sizes))
        for height, width in sizes:
            assert height % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
            assert width % p.MINIMAX_H3_CANVAS_MULTIPLE == 0


def test_ref2va_presentation_ladder_pad_to_rung(expect_error):
    caps = MiniMaxH3ArenaCaps.for_task("ref2va")
    ladder = tuple(rung for rung in MINIMAX_H3_REF2VA_PRESENTATION_RUNGS if rung < caps.prompt) + (caps.prompt,)
    assert ladder[-1] == caps.prompt
    assert all(rung % 1024 == 0 for rung in ladder)
    assert list(ladder) == sorted(set(ladder))
    for seq_len in range(1, ladder[-1] + 1):
        assert select_bucket(seq_len, ladder) == min(rung for rung in ladder if rung >= seq_len)
    with expect_error(ValueError, "exceeds the top bucket"):
        select_bucket(ladder[-1] + 1, ladder)


@pytest.mark.parametrize("num_frames,expected", [(1, 5), (5, 5), (6, 22), (22, 22), (120, 124), (192, 192)])
def test_align_num_frames(num_frames, expected):
    assert policy.align_num_frames(num_frames) == expected


@pytest.mark.parametrize("duration_s,expected", [(4, 107), (4.5, 124), (5, 124), (10, 243), (15, 362)])
def test_get_num_frames(duration_s, expected):
    assert policy.get_num_frames(duration_s) == expected


@pytest.mark.parametrize("duration_s", policy.MINIMAX_H3_DURATIONS_S)
def test_served_duration_aligns_to_encodable_frame_count(duration_s):
    frames = policy.align_num_frames(duration_s * p.MINIMAX_H3_FPS)
    assert frames % p.MINIMAX_H3_FRAMES_PER_CHUNK == p.MINIMAX_H3_LATENTS_PER_CHUNK


def test_text_budget_fills_prompt_cap():
    caps = MiniMaxH3ArenaCaps()
    assert policy.MINIMAX_H3_MAX_TEXT_TOKENS + policy.MINIMAX_H3_MAX_KEYFRAME_TOKENS == caps.prompt
    caps.validate()


def test_prompt_pad_warm_lengths_are_every_padded_prompt_below_the_cap():
    caps = MiniMaxH3ArenaCaps()
    alignment = 1024
    padded = {-(-length // alignment) * alignment for length in range(1, caps.prompt + 1)}
    assert padded - {caps.prompt} == set(range(alignment, caps.prompt, alignment))


def test_a_32_tile_canvas_fills_the_4x32_decode_wave():
    wave_size = 128

    def tiles(length: int) -> int:
        return len(split_tiles(length, DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP, p.MINIMAX_H3_CANVAS_MULTIPLE)[1])

    counts = [tiles(height) * tiles(width) for height, width in policy.decodable_canvases()]
    assert max(count for count in counts if wave_size % count == 0) == 32


def test_caps_sum_fits_top_rung():
    caps = MiniMaxH3ArenaCaps()
    admissible = caps.prompt + caps.condition_video_rows + caps.audio_rows + caps.video_rows
    assert admissible <= MINIMAX_H3_BUCKET_LADDER[-1]


def test_ref2va_caps_sum_fits_top_ref2va_rung():
    caps = MiniMaxH3ArenaCaps.for_task("ref2va")
    admissible = caps.prompt + caps.condition_video_rows + caps.audio_rows + caps.video_rows + caps.condition_audio_rows
    assert admissible <= MINIMAX_H3_REF2VA_BUCKET_LADDER[-1]
    caps.validate()


def test_reference_patch_cap_is_nine_images_and_three_longest_videos_at_the_area_cap():
    assert policy.MINIMAX_H3_MAX_REFERENCE_BLOCK_PATCHES == 4032
    assert policy.MINIMAX_H3_MAX_REFERENCE_PATCHES == 4032 * (9 + 3 * 16) == 229824


def test_vision_patch_ladder_covers_every_ref2va_request():
    validate_bucket_ladder(MINIMAX_H3_VISION_PATCH_LADDER, 32 * 32)
    caps = MiniMaxH3ArenaCaps.for_task("ref2va")
    assert MINIMAX_H3_VISION_PATCH_LADDER[-1] >= min(4 * caps.prompt, policy.MINIMAX_H3_MAX_REFERENCE_PATCHES)
    smallest_block = min(
        4 * (h // p.MINIMAX_H3_CANVAS_MULTIPLE) * (w // p.MINIMAX_H3_CANVAS_MULTIPLE)
        for canvas in policy.served_canvases()
        for h, w in policy.served_reference_image_sizes(*canvas)
    )
    assert MINIMAX_H3_VISION_PATCH_LADDER[0] - 1024 < smallest_block <= MINIMAX_H3_VISION_PATCH_LADDER[0]


@pytest.mark.parametrize("canvas", policy.served_canvases())
def test_match_never_exceeds_the_target_canvas_area(canvas):
    height, width = canvas
    for source_height in range(256, 1025, 8):
        for source_width in range(max(256, source_height // 4), min(4096, 4 * source_height) + 1, 8):
            size = rp.resolve_reference_image_size(
                source_width, source_height, mode="match", target_width=width, target_height=height
            )
            assert size[0] * size[1] <= height * width, (source_width, source_height)
            if source_width == source_height:
                assert size[0] == size[1]


def test_allow_align_above_area_restores_rounding_up():
    kwargs = dict(mode="match", target_width=1344, target_height=768)
    assert rp.resolve_reference_image_size(560, 1843, **kwargs) == (1824, 544)
    assert rp.resolve_reference_image_size(560, 1843, allow_align_above_area=True, **kwargs) == (1856, 576)


@pytest.mark.parametrize("text,expected", [("16:9", (16, 9)), ("9:16", (9, 16)), ("1:1", (1, 1))])
def test_parse_aspect_ratio_published(text, expected):
    assert policy.minimax_h3_parse_aspect_ratio(text) == expected


def test_parse_aspect_ratio_accepts_x_and_slash():
    assert policy.minimax_h3_parse_aspect_ratio("16x9") == (16, 9)
    assert policy.minimax_h3_parse_aspect_ratio("16/9") == (16, 9)


def test_parse_aspect_ratio_rejects_unpublished_and_names_the_set(expect_error):
    with expect_error(ValueError, "is not served") as exc_info:
        policy.minimax_h3_parse_aspect_ratio("2:1")
    for width, height in policy.MINIMAX_H3_ASPECT_RATIOS:
        assert f"{width}:{height}" in str(exc_info.value)


def test_parse_aspect_ratio_rejects_malformed(expect_error):
    with expect_error(ValueError, "must look like"):
        policy.minimax_h3_parse_aspect_ratio("widescreen")


def test_num_inference_steps_fixed_at_fifty():
    assert policy.MINIMAX_H3_NUM_INFERENCE_STEPS == 50


def test_frames_are_aligned_tracks_encodable_counts():
    frames = policy.get_num_frames(policy.MINIMAX_H3_DEFAULT_DURATION_S)
    assert frames == 124
    assert policy.minimax_h3_frames_are_aligned(frames)
    assert not policy.minimax_h3_frames_are_aligned(120)


@pytest.mark.parametrize("num_frames", [96, 107, 124, 360, 362])
def test_validate_num_frames_accepts_served_lengths(num_frames):
    policy.validate_num_frames(num_frames)


@pytest.mark.parametrize("num_frames", [1, 90, 363, 720])
def test_validate_num_frames_rejects_unserved_lengths(num_frames, expect_error):
    with expect_error(ValueError, "served lengths are 107 to 362 frames"):
        policy.validate_num_frames(num_frames)


_KEYFRAME = Image.new("RGB", (1344, 768))
_TASK_INPUTS = {
    "t2va": {"image": None, "last_image": None},
    "fl2va": {"image": _KEYFRAME, "last_image": None},
    "fl2va_last_frame": {"image": None, "last_image": _KEYFRAME},
}
_REFERENCES = [rp.MiniMaxH3Reference(image=_KEYFRAME)]


def _request(**overrides):
    request = dict(
        image=None, last_image=None, references=None, aspect_ratio=(16, 9), height=None, width=None, num_frames=124
    )
    request.update(overrides)
    return request


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_frames": 96},
        {"num_frames": 362},
        {"image": _KEYFRAME},
        {"references": _REFERENCES},
        {"references": _REFERENCES, "num_frames": None},
    ],
)
def test_validate_request_accepts_served_requests(overrides):
    policy.validate_request(**_request(**overrides))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"num_frames": 90}, "served lengths are"),
        ({"references": _REFERENCES, "num_frames": 363}, "served lengths are"),
        ({"references": _REFERENCES, "aspect_ratio": (2, 1)}, "is not served for Ref2VA"),
        ({"references": _REFERENCES, "height": 770, "width": 1344}, "multiple of 32"),
        ({"references": _REFERENCES, "image": _KEYFRAME}, "different tasks"),
    ],
)
def test_validate_request_rejects_unserved_requests(overrides, match, expect_error):
    with expect_error(ValueError, match):
        policy.validate_request(**_request(**overrides))


@pytest.mark.parametrize("aspect", policy.MINIMAX_H3_ASPECT_RATIOS)
def test_validate_t2va_accepts_served_ratios(aspect):
    policy.validate_input(**_TASK_INPUTS["t2va"], aspect_ratio=aspect, height=None, width=None)


@pytest.mark.parametrize("aspect", [(3, 2), (5, 2)])
def test_validate_t2va_rejects_unlisted_ratio(aspect, expect_error):
    with expect_error(ValueError, "is not served"):
        policy.validate_input(**_TASK_INPUTS["t2va"], aspect_ratio=aspect, height=None, width=None)


@pytest.mark.parametrize("aspect", policy.MINIMAX_H3_ASPECT_RATIOS)
def test_validate_ref2va_accepts_served_ratios(aspect):
    policy.validate_ref2va(aspect_ratio=aspect, height=None, width=None)


@pytest.mark.parametrize("aspect", [(3, 2), (2, 1)])
def test_validate_ref2va_rejects_unlisted_ratio(aspect, expect_error):
    with expect_error(ValueError, "is not served for Ref2VA"):
        policy.validate_ref2va(aspect_ratio=aspect, height=None, width=None)


def test_validate_ref2va_ignores_aspect_ratio_with_explicit_canvas():
    policy.validate_ref2va(aspect_ratio=(2, 1), height=768, width=1344)


@pytest.mark.parametrize(
    "height, width, match",
    [
        (770, 1344, "multiple of 32"),
        (2048, 2048, "area cap"),
        (384, 1600, "1:4 to 4:1"),
    ],
)
def test_validate_ref2va_rejects_bad_explicit_canvas(height, width, match, expect_error):
    with expect_error(ValueError, match):
        policy.validate_ref2va(aspect_ratio=(16, 9), height=height, width=width)


@pytest.mark.parametrize("task", ["fl2va", "fl2va_last_frame"])
def test_validate_fl2va_ignores_aspect_ratio(task):
    policy.validate_input(**_TASK_INPUTS[task], aspect_ratio=(3, 2), height=None, width=None)


@pytest.mark.parametrize("task", list(_TASK_INPUTS))
def test_validate_accepts_explicit_canvas(task):
    policy.validate_input(**_TASK_INPUTS[task], aspect_ratio=(3, 2), height=768, width=1344)


@pytest.mark.parametrize("task", list(_TASK_INPUTS))
@pytest.mark.parametrize(
    "height, width, match",
    [
        (768, None, "both height and width"),
        (None, 1344, "both height and width"),
        (770, 1344, "multiple of 32"),
        (2048, 2048, "area cap"),
        (384, 1600, "1:4 to 4:1"),
    ],
)
def test_validate_rejects_bad_explicit_canvas(task, height, width, match, expect_error):
    with expect_error(ValueError, match):
        policy.validate_input(**_TASK_INPUTS[task], aspect_ratio=(16, 9), height=height, width=width)


@pytest.mark.parametrize("slot", ["image", "last_image"])
@pytest.mark.parametrize("size", [(64, 64), (8000, 3000), (1024, 256), (256, 1024)])
def test_validate_fl2va_accepts_keyframe_in_ratio_range(slot, size):
    inputs = {"image": _KEYFRAME, "last_image": None, slot: Image.new("L", size)}
    policy.validate_input(**inputs, aspect_ratio=(16, 9), height=None, width=None)


@pytest.mark.parametrize("slot", ["image", "last_image"])
@pytest.mark.parametrize("size", [(1025, 256), (256, 1025)])
def test_validate_fl2va_rejects_keyframe_outside_ratio_range(slot, size, expect_error):
    inputs = {"image": _KEYFRAME, "last_image": None, slot: Image.new("L", size)}
    with expect_error(ValueError, f"{slot} is .*aspect ratio must be"):
        policy.validate_input(**inputs, aspect_ratio=(16, 9), height=None, width=None)


def _admission(**overrides):
    request = dict(aspect_ratio=None, duration_s=None, height=None, width=None)
    request.update(overrides)
    return request


def _ref(kind, name="references[0]", **fields):
    return policy.MiniMaxH3ReferenceInfo(name=name, kind=kind, **fields)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"aspect_ratio": "9:16", "duration_s": 15},
        {"duration_s": 4.0},
        {"height": 768, "width": 1344, "aspect_ratio": "16:9"},
        {"height": 384, "width": 1536},
        {"keyframe_sizes": {"image_prompts[0]": (1024, 256)}},
        {"references": [_ref("image", size=(4000, 1000))]},
        {"references": [_ref("video", size=(1280, 720), fps=30.0, audio_channels=2), _ref("audio", audio_channels=1)]},
        {"references": [_ref("image")] * 9 + [_ref("video", fps=24.0)] * 3},
    ],
)
def test_admit_request_accepts_served_requests(overrides):
    policy.admit_request(**_admission(**overrides))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"aspect_ratio": "2:1"}, "is not served"),
        ({"aspect_ratio": "wide"}, "must look like"),
        ({"aspect_ratio": "2:1", "height": 768, "width": 1344}, "is not served"),
        ({"duration_s": 3}, "whole number of seconds from 4 to 15"),
        ({"duration_s": 16}, "whole number"),
        ({"duration_s": 5.5}, "whole number"),
        ({"height": 768}, "both height and width"),
        ({"width": 1344}, "both height and width"),
        ({"height": 770, "width": 1344}, "multiple of 32"),
        ({"height": 2048, "width": 2048}, "area cap"),
        ({"height": 384, "width": 1600}, "1:4 to 4:1"),
        ({"keyframe_sizes": {"image_prompts[1]": (1025, 256)}}, r"image_prompts\[1\] is .*aspect ratio"),
        ({"references": [_ref("image", "references.images[2]", size=(4001, 1000))]}, r"references.images\[2\] must be"),
        ({"references": [_ref("video", size=(100, 401), fps=24.0)]}, "within 1:4 and 4:1"),
        ({"references": [_ref("video")]}, "reports no frame rate"),
        ({"references": [_ref("video", fps=0.0)]}, "reports no frame rate"),
        ({"references": [_ref("image"), _ref("audio")]}, "has no audio stream"),
        ({"references": [_ref("image"), _ref("audio", audio_channels=6)]}, "6 audio channels"),
        ({"references": [_ref("video", fps=24.0, audio_channels=0)]}, "mono or stereo"),
    ],
)
def test_admit_request_rejects_unserved_requests(overrides, match, expect_error):
    with expect_error(ValueError, match):
        policy.admit_request(**_admission(**overrides))
