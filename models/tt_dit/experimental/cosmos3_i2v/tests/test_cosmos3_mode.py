# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the unified-pipeline mode dispatch (`cosmos3_mode`).

Pure classification, defaults-merge, and gating logic; no device. Covers the
`resolve_mode` truth table, `load_modality_defaults`, and the `run_cosmos3`
forwarding / negative-prompt-fallback / signature-filter contract.
"""

from __future__ import annotations

import pytest

from models.tt_dit.experimental.cosmos3_i2v.pipelines.cosmos3_mode import (
    ModelMode,
    load_modality_defaults,
    resolve_mode,
    run_cosmos3,
)

_IMG = object()  # opaque non-None stand-in for a conditioning image


@pytest.mark.parametrize(
    "image, num_frames, enable_sound, expected",
    [
        (None, None, False, ModelMode.TEXT2VIDEO),
        (None, 189, False, ModelMode.TEXT2VIDEO),
        (None, 1, False, ModelMode.TEXT2IMAGE),
        (_IMG, 189, False, ModelMode.IMAGE2VIDEO),
        (_IMG, None, False, ModelMode.IMAGE2VIDEO),
        (_IMG, 1, False, ModelMode.TEXT2IMAGE),  # image dropped for single frame
        (_IMG, 189, True, ModelMode.AUDIO_IMAGE2VIDEO),
        (None, 189, True, ModelMode.AUDIO_IMAGE2VIDEO),
    ],
)
def test_resolve_mode_truth_table(image, num_frames, enable_sound, expected):
    assert resolve_mode(image=image, num_frames=num_frames, enable_sound=enable_sound) == expected


def test_resolve_mode_action_unsupported(expect_error):
    with expect_error(NotImplementedError, "action-conditioned"):
        resolve_mode(action=object())


# --- per-mode defaults ------------------------------------------------------


@pytest.mark.parametrize("mode", [ModelMode.TEXT2VIDEO, ModelMode.IMAGE2VIDEO, ModelMode.AUDIO_IMAGE2VIDEO])
def test_load_defaults_video_modes(mode):
    d = load_modality_defaults(mode)
    args = d["sample_args"]
    assert args["num_frames"] == 189
    assert args["flow_shift"] == 6.0  # unified preset; build-time, filtered out before forward
    assert args["height"] == 720 and args["width"] == 1280
    # recommended NVIDIA quality-control negative prompt (JSON free-text in Phase 2)
    assert d["negative_prompt"].startswith('{"subjects"')


def test_load_defaults_text2image():
    d = load_modality_defaults(ModelMode.TEXT2IMAGE)
    assert d["sample_args"]["num_frames"] == 1
    assert d["sample_args"]["flow_shift"] == 6.0
    assert d["negative_prompt"] == ""  # T2I ships no default negative


# --- forwarding / merge / filter --------------------------------------------


class _RefLikePipe:
    """Concrete `__call__` mirroring the Cosmos3OmniPipeline params we filter to."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        image=None,
        num_frames=None,
        height=None,
        width=None,
        fps=24.0,
        num_inference_steps=35,
        guidance_scale=6.0,
        enable_sound=False,
        generator=None,
        output_type="pil",
    ):
        self.calls.append(
            dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                enable_sound=enable_sound,
            )
        )
        return "result"


class _CfgWrapperPipe(_RefLikePipe):
    """Mirrors the cfg subclass: `(*args, **kwargs)` forwarder over the real sig."""

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def test_run_cosmos3_t2v_forwards():
    pipe = _RefLikePipe()
    out = run_cosmos3(pipe, prompt="p", image=None, num_frames=189, height=720, width=1280)
    assert out == "result"
    (call,) = pipe.calls
    assert call["prompt"] == "p"
    assert call["image"] is None
    assert call["num_frames"] == 189
    assert call["height"] == 720 and call["width"] == 1280
    assert call["enable_sound"] is False


def test_run_cosmos3_i2v_forwards():
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189)
    assert pipe.calls[0]["image"] is _IMG


def test_run_cosmos3_merges_defaults_caller_wins():
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189, num_inference_steps=12)
    (call,) = pipe.calls
    assert call["num_inference_steps"] == 12  # caller override
    assert call["guidance_scale"] == 6.0  # from mode defaults
    assert call["fps"] == 24 and call["height"] == 720 and call["width"] == 1280


def test_run_cosmos3_drops_build_time_flow_shift():
    """flow_shift lives in sample_args but is not a __call__ param — must be filtered."""
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189)
    assert "flow_shift" not in pipe.calls[0]


def test_run_cosmos3_injects_default_negative_prompt():
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189)
    assert pipe.calls[0]["negative_prompt"] == load_modality_defaults(ModelMode.IMAGE2VIDEO)["negative_prompt"]


def test_run_cosmos3_caller_negative_prompt_wins():
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189, negative_prompt="my neg")
    assert pipe.calls[0]["negative_prompt"] == "my neg"


def test_run_cosmos3_cfg_wrapper_signature_resolved():
    """MRO walk skips the (*args, **kwargs) forwarder and filters against the real sig."""
    pipe = _CfgWrapperPipe()
    run_cosmos3(pipe, prompt="p", image=_IMG, num_frames=189)
    (call,) = pipe.calls
    assert "flow_shift" not in call
    assert call["num_inference_steps"] == 35


def test_run_cosmos3_t2i_forwards():
    pipe = _RefLikePipe()
    run_cosmos3(pipe, prompt="p", image=None, num_frames=1)
    (call,) = pipe.calls
    assert call["num_frames"] == 1
    assert call["image"] is None
    assert call["num_inference_steps"] == 50  # T2I mode default
    assert call["guidance_scale"] == 4.0
    assert call["height"] == 1024 and call["width"] == 1024
    assert call["negative_prompt"] == ""  # T2I ships no default negative


def test_run_cosmos3_rejects_audio_mode(expect_error):
    pipe = _RefLikePipe()
    with expect_error(NotImplementedError, "not wired"):
        run_cosmos3(pipe, prompt="p", image=None, num_frames=189, enable_sound=True)
    assert pipe.calls == []  # gated before dispatch


def test_run_cosmos3_mode_mismatch_raises(expect_error):
    pipe = _RefLikePipe()
    with expect_error(ValueError, "contradicts"):
        run_cosmos3(pipe, prompt="p", image=None, num_frames=189, mode=ModelMode.IMAGE2VIDEO)
