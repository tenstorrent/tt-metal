# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for two-time (interval) conditioning: the adapter's sampling contract, and the
one invariant whose failure is silent.

That invariant is the **level match**. The AdaLN table is addressed by looking a row's ``(t, r)``
level up *by value* among the levels the table was built for. Every other wiring mistake in this
path raises -- a missing level raises ``IndexError`` in the lookup, a wrong step count raises in
``assert_covers`` -- but two float32 values that agree to within a rounding do not raise; they
either find the row or they do not. So the loop's levels and the table's levels are asserted to be
bit-equal here, on the schedulers a request actually runs.
"""

import json

import pytest
import torch

from ....pipelines.minimax_h3 import adaln_precompute as ap
from ....pipelines.minimax_h3 import packing as p
from ....pipelines.minimax_h3.hyperflow_minimax_h3 import MARKER_KEY, MiniMaxH3HyperFlow, validate_sigmas
from ....pipelines.minimax_h3.pipeline_minimax_h3 import AUDIO_SHIFT, MINIMAX_H3_AUDIO_CONDITION_TIMESTEP, VIDEO_SHIFT
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler, shift_sigmas

# The published 8-forward grid: 9 points, and the only HyperFlow-specific numbers in this file.
GRID_8STEP = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
GATE = 0.75

TEXT_LEN = 997


def _metadata(**overrides) -> dict[str, str]:
    metadata = {
        MARKER_KEY: "true",
        "hyperflow_version": "1.0.0",
        "hyperflow_gate": str(GATE),
        "hyperflow_sigmas": json.dumps(list(GRID_8STEP)),
        "hyperflow_video_shift": str(VIDEO_SHIFT),
        "hyperflow_audio_shift": str(AUDIO_SHIFT),
    }
    metadata.update({key: value for key, value in overrides.items() if value is not None})
    for key, value in overrides.items():
        if value is None:
            metadata.pop(key, None)
    return metadata


def _contract(**overrides) -> MiniMaxH3HyperFlow:
    contract = MiniMaxH3HyperFlow.from_adapter_metadata(
        _metadata(**overrides), video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
    )
    assert contract is not None
    return contract


def _schedulers(contract: MiniMaxH3HyperFlow | None, steps: int = 50):
    video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
    if contract is None:
        video.set_timesteps(steps)
        audio.set_timesteps(steps)
    else:
        video.set_timesteps(sigmas=contract.modality_sigmas(VIDEO_SHIFT))
        audio.set_timesteps(sigmas=contract.modality_sigmas(AUDIO_SHIFT))
    return video, audio


def _assert_every_asked_level_is_held(asked: torch.Tensor, held: torch.Tensor) -> None:
    """The lookup `_denoise` performs, asserted to resolve exactly and injectively.

    One direction only. The table's levels come from the *task's* pin rules while the asked ones
    come from the *layout*, so the table is a superset whenever a pin applies to no row -- t2va
    carries the keyframe floor it has no condition rows for. An extra row costs a few KB; a missing
    one is the failure this guards.
    """
    rows = [(held == level).all(dim=-1).nonzero() for level in asked]
    assert all(row.shape[0] == 1 for row in rows), "a level the loop asks for is absent or duplicated"
    positions = [int(row[0, 0]) for row in rows]
    assert len(set(positions)) == len(positions), "two distinct levels resolved to one table row"


def _layout(anchors=("first",)):
    tags = torch.ones(TEXT_LEN, dtype=torch.long)
    tags[20:70] = p.MINIMAX_H3_VIDEO_TAG
    return p.build_packed_sequence(tags, 37, 544 // 16, 960 // 16, 207, (1, 2, 2), anchors)


# ---------------------------------------------------------------- the contract


def test_a_plain_adapter_publishes_no_contract():
    """FastH3 and friends: no marker, so the pipeline keeps taking its step count from the caller."""
    assert MiniMaxH3HyperFlow.from_adapter_metadata({}, video_shift=12.0, audio_shift=3.0) is None
    assert MiniMaxH3HyperFlow.from_adapter_metadata(None, video_shift=12.0, audio_shift=3.0) is None


@pytest.mark.parametrize("dropped", ["hyperflow_version", "hyperflow_gate", "hyperflow_sigmas"])
def test_a_half_written_contract_raises(dropped, expect_error):
    """The alternative is a two-time adapter sampled on a single-time grid, which produces video."""
    with expect_error(ValueError, "missing"):
        MiniMaxH3HyperFlow.from_adapter_metadata(
            _metadata(**{dropped: None}), video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
        )


def test_a_contract_trained_at_other_shifts_raises(expect_error):
    """There is no shift override here to reconcile them with, so this is the wrong file."""
    with expect_error(ValueError, "cannot be reproduced"):
        MiniMaxH3HyperFlow.from_adapter_metadata(
            _metadata(hyperflow_video_shift="7.0"), video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT
        )


def test_forwards_are_one_fewer_than_the_grid_points(expect_error):
    contract = _contract()
    assert contract.num_grid_points == 9
    assert contract.num_forwards == 8

    contract.assert_forwards(None)
    contract.assert_forwards(9)
    with expect_error(ValueError, "cannot be honoured"):
        contract.assert_forwards(50)


def test_tasks_default_to_every_task(expect_error):
    _contract().assert_supports_task("ref2va")
    _contract(tasks=json.dumps(["t2va", "fl2va"])).assert_supports_task("fl2va")
    with expect_error(ValueError, "ref2va"):
        _contract(tasks=json.dumps(["t2va", "fl2va"])).assert_supports_task("ref2va")


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_the_grid_is_shifted_with_the_schedulers_own_formula(shift):
    contract = _contract()
    assert torch.equal(contract.modality_sigmas(shift), shift_sigmas(validate_sigmas(GRID_8STEP), shift))
    # A valid grid stays valid: the shift maps 0 to 0 and 1 to 1.
    validate_sigmas(contract.modality_sigmas(shift))


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_the_endpoint_of_a_step_is_the_next_steps_timestep(shift):
    """`r_i = 1 - sigma_{i+1}`, so the last step aims at a clean sample."""
    contract = _contract()
    sigmas = contract.modality_sigmas(shift)
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(sigmas=sigmas)

    endpoints = contract.endpoints(sigmas)
    assert endpoints.numel() == contract.num_forwards
    assert torch.equal(endpoints[:-1], scheduler.timesteps[1:])
    assert float(endpoints[-1]) == 1.0
    assert bool((endpoints > scheduler.timesteps).all())


def test_identity_separates_every_term_that_moves_a_row():
    base = _contract().identity()
    assert _contract(hyperflow_gate="0.5").identity() != base
    assert _contract(hyperflow_version="1.0.1").identity() != base
    assert _contract(hyperflow_sigmas=json.dumps([1.0, 0.5, 0.0])).identity() != base
    assert _contract().identity() == base


# ------------------------------------------------------------- the level match


@pytest.mark.parametrize("task", ["t2va", "ref2va"])
def test_the_loops_levels_are_bit_equal_to_the_tables(task):
    """The silent one. See the module docstring."""
    contract = _contract()
    video, audio = _schedulers(contract)
    layout = _layout(anchors=() if task == "t2va" else ("first",))

    video_endpoints = contract.endpoints(video.sigmas)
    audio_endpoints = contract.endpoints(audio.sigmas)
    step_levels = ap.request_step_levels(
        video.sigmas,
        audio.sigmas,
        p.MINIMAX_H3_KEYFRAME_NOISE_AUG,
        audio_condition_timestep=MINIMAX_H3_AUDIO_CONDITION_TIMESTEP if task == "ref2va" else None,
        video_endpoints=video_endpoints,
        audio_endpoints=audio_endpoints,
    )
    assert len(step_levels) == contract.num_forwards

    for step, t in enumerate(video.timesteps):
        asked, row_index = p.build_row_levels(
            layout,
            float(t),
            float(audio.timesteps[step]),
            max(float(t), p.MINIMAX_H3_KEYFRAME_NOISE_AUG),
            MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
            video_endpoint=float(video_endpoints[step]),
            audio_endpoint=float(audio_endpoints[step]),
        )
        _assert_every_asked_level_is_held(asked, step_levels[step])
        assert int(row_index.max()) < asked.shape[0]


def test_a_single_time_schedule_still_matches_after_the_pair_change():
    """The base model runs the same code with `r == t`; the match must not depend on a contract."""
    video, audio = _schedulers(None, steps=50)
    layout = _layout()
    step_levels = ap.request_step_levels(video.sigmas, audio.sigmas, p.MINIMAX_H3_KEYFRAME_NOISE_AUG)

    for step, t in enumerate(video.timesteps):
        asked, _ = p.build_row_levels(
            layout,
            float(t),
            float(audio.timesteps[step]),
            max(float(t), p.MINIMAX_H3_KEYFRAME_NOISE_AUG),
            MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
        )
        _assert_every_asked_level_is_held(asked, step_levels[step])


def test_the_endpoint_axis_is_what_makes_the_levels_distinct():
    """Without the pair, the two modalities collapse onto one level wherever their `t` coincides.

    Constructed rather than sampled: the shifted grids only meet at the endpoints of a real request,
    so the collision this guards against needs a schedule that forces it.
    """
    layout = _layout(anchors=())
    levels, _ = p.build_row_levels(layout, 0.5, 0.5, 0.999, 1.0, video_endpoint=0.75, audio_endpoint=0.625)

    video_and_audio = levels[levels[:, 0] == 0.5]
    assert video_and_audio.shape[0] == 2
    assert torch.equal(video_and_audio[:, 1], torch.tensor([0.625, 0.75]))
