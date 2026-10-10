# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the opt-in warm scope (`warm_rungs`, `warm_canvases`)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from ....pipelines.minimax_h3 import pipeline_minimax_h3 as pm
from ....pipelines.minimax_h3 import policy
from ....pipelines.minimax_h3.packing import resolve_canvas_size

ALIGNMENTS = (32, 64, 128, 256, 512, 1024)


def _old_served_keyframe_layouts(patch_alignment):
    """`served_keyframe_layouts` as it was written before `keyframe_layout_key` was split out."""
    canvases = policy.decodable_canvases()
    multiple = policy.MINIMAX_H3_CANVAS_MULTIPLE

    def patches(canvas):
        return 4 * (canvas[0] // multiple) * (canvas[1] // multiple)

    by_key = {}
    for n_keyframes in (1, 2):
        for canvas in sorted(canvases, key=lambda canvas: (patches(canvas), canvas)):
            total = n_keyframes * patches(canvas)
            padded = -(-total // patch_alignment) * patch_alignment
            by_key.setdefault((padded, n_keyframes == 1 and padded == total), (n_keyframes, canvas))
    return tuple(by_key[key] for key in sorted(by_key)), by_key


@pytest.mark.parametrize("alignment", ALIGNMENTS)
def test_keyframe_layouts_match_old_code(alignment):
    old_layouts, old_keys = _old_served_keyframe_layouts(alignment)
    assert policy.served_keyframe_layouts(alignment) == old_layouts
    for key, (n_keyframes, canvas) in old_keys.items():
        assert policy.keyframe_layout_key(n_keyframes, canvas, alignment) == key
    for n_keyframes in (1, 2):
        for canvas in policy.decodable_canvases():
            assert policy.keyframe_layout_key(n_keyframes, canvas, alignment) in old_keys


@pytest.mark.parametrize("alignment", ALIGNMENTS)
def test_filter_warm_layouts(alignment):
    layouts = list(policy.served_envelope("t2va", patch_alignment=alignment))
    canvas = resolve_canvas_size(9, 16)
    canvases = {canvas, resolve_canvas_size(16, 9)}
    kept = policy.filter_warm_layouts(layouts, canvases, alignment)

    assert kept[0] == (0, None)
    assert set(kept) <= set(layouts)
    kept_keys = {policy.keyframe_layout_key(n, c, alignment) for n, c in kept if c is not None}
    assert kept_keys == {policy.keyframe_layout_key(n, c, alignment) for n in (1, 2) for c in canvases}
    dropped = [layout for layout in layouts if layout not in kept]
    for n_keyframes, c in dropped:
        assert policy.keyframe_layout_key(n_keyframes, c, alignment) not in kept_keys


def test_resolve_warm_rungs(expect_error):
    ladder = (1024, 2048, 4096)
    assert policy.resolve_warm_rungs(None, ladder) is None
    assert policy.resolve_warm_rungs([1024], ladder) == {1024, 4096}
    assert policy.resolve_warm_rungs([], ladder) == {4096}
    assert policy.resolve_warm_rungs([4096], ladder) == {4096}
    with expect_error(ValueError, r"\[3000\]"):
        policy.resolve_warm_rungs([1024, 3000], ladder)


def _fake_pipeline(*, warm_canvases, trace_denoise, warming=False):
    fake = SimpleNamespace(
        warm_canvases=None if warm_canvases is None else frozenset(warm_canvases),
        trace_denoise=trace_denoise,
        _warming=warming,
        sp_factor=4,
        logs=[],
    )
    fake._host_log = fake.logs.append
    fake._warm_layout_canvases = lambda: pm.MiniMaxH3Pipeline._warm_layout_canvases(fake)
    return fake


def _check(fake, n_keyframes, canvas):
    pm.MiniMaxH3Pipeline._check_warm_keyframe_layout(fake, n_keyframes, canvas)


def _outside_canvas(alignment, canvases):
    wanted = policy.warm_keyframe_layout_keys(canvases, alignment)
    for canvas in policy.decodable_canvases():
        if policy.keyframe_layout_key(1, canvas, alignment) not in wanted:
            return canvas
    raise AssertionError("every canvas shares a layout with the warm set")


def test_trace_guard_raises_outside_warm_canvases(expect_error):
    warm = {resolve_canvas_size(16, 9)}
    fake = _fake_pipeline(warm_canvases=warm, trace_denoise=True)
    outside = _outside_canvas(fake.sp_factor * 32, warm)
    _check(fake, 1, resolve_canvas_size(16, 9))
    _check(fake, 2, resolve_canvas_size(16, 9))
    with expect_error(ValueError, "outside warm_canvases"):
        _check(fake, 1, outside)


def test_trace_guard_logs_when_untraced_and_is_off_by_default():
    warm = {resolve_canvas_size(16, 9)}
    untraced = _fake_pipeline(warm_canvases=warm, trace_denoise=False)
    outside = _outside_canvas(untraced.sp_factor * 32, warm)
    _check(untraced, 1, outside)
    assert len(untraced.logs) == 1 and "compiles" in untraced.logs[0]

    for fake in (
        _fake_pipeline(warm_canvases=None, trace_denoise=True),
        _fake_pipeline(warm_canvases=warm, trace_denoise=True, warming=True),
    ):
        _check(fake, 1, outside)
        assert fake.logs == []


def _canvas_without_16x9_layouts(alignment):
    landscape = policy.warm_keyframe_layout_keys({resolve_canvas_size(16, 9)}, alignment)
    for canvas in policy.decodable_canvases():
        if not policy.warm_keyframe_layout_keys({canvas}, alignment) & landscape:
            return canvas
    raise AssertionError("every canvas shares a layout with 16:9")


def test_trace_guard_always_admits_the_16x9_warmup_canvas():
    fake = _fake_pipeline(warm_canvases=set(), trace_denoise=True)
    fake.warm_canvases = frozenset({_canvas_without_16x9_layouts(fake.sp_factor * 32)})
    # `_warmup_on_init` warms 16:9 with one and two keyframes whatever `warm_canvases` holds.
    _check(fake, 1, resolve_canvas_size(16, 9))
    _check(fake, 2, resolve_canvas_size(16, 9))
    assert fake.logs == []


class _Stop(Exception):
    pass


def _call_until_encode(expect_error, fake, **kwargs):
    fake.vae_config = SimpleNamespace(spatial_compression_ratio=16)
    fake._force_bucket = None
    fake._log = lambda message: None
    fake._track_cache_misses = lambda *args: nullcontext()

    def encode_prompt(prompt, keyframes):
        raise _Stop("encode_prompt")

    fake.encode_prompt = encode_prompt
    fake._check_warm_keyframe_layout = lambda n, canvas: pm.MiniMaxH3Pipeline._check_warm_keyframe_layout(
        fake, n, canvas
    )
    with expect_error(_Stop, "encode_prompt"):
        pm.MiniMaxH3Pipeline.__call__(fake, "a prompt", num_frames=124, **kwargs)


def test_call_runs_the_trace_guard_before_encoding(expect_error):
    warm = {resolve_canvas_size(16, 9)}
    fake = _fake_pipeline(warm_canvases=warm, trace_denoise=True)
    outside = _outside_canvas(fake.sp_factor * 32, warm)
    landscape = Image.new("RGB", resolve_canvas_size(16, 9)[::-1])
    _call_until_encode(expect_error, fake, image=landscape, last_image=landscape)
    _call_until_encode(expect_error, fake)  # t2va has no keyframe layout to check.

    image = Image.new("RGB", outside[::-1])
    with expect_error(ValueError, "outside warm_canvases"):
        pm.MiniMaxH3Pipeline.__call__(fake, "a prompt", num_frames=124, image=image)
