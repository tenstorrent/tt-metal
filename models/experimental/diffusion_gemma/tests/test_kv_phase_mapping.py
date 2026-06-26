# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import models.experimental.diffusion_gemma.kv_phase as kv_phase
from models.experimental.diffusion_gemma.kv_phase import KVPhaseMapping


def test_mapping_module_is_reference_spec_not_runtime_path():
    assert "Reference/spec" in kv_phase.__doc__
    assert "Runtime Gemma4 cache updates" in kv_phase.__doc__
    assert "Reference logical-to-physical" in KVPhaseMapping.__doc__


def test_commit_positions_append_after_prompt():
    mapping = KVPhaseMapping(prompt_len=32, canvas_len=8, sliding_window=16)

    assert mapping.commit_positions == tuple(range(32, 40))
    assert mapping.canvas_scratch_positions == tuple(range(8))


def test_full_attention_frozen_cache_keeps_absolute_prompt_positions():
    mapping = KVPhaseMapping(prompt_len=12, canvas_len=4, sliding_window=8)

    assert mapping.full_attention_frozen_positions == tuple(range(12))


def test_sliding_frozen_cache_keeps_only_live_window():
    mapping = KVPhaseMapping(prompt_len=20, canvas_len=4, sliding_window=8)

    assert mapping.sliding_frozen_positions == tuple(range(12, 20))
    assert mapping.sliding_frozen_slots == (4, 5, 6, 7, 0, 1, 2, 3)


def test_sliding_commit_slots_wrap_after_window_boundary():
    mapping = KVPhaseMapping(prompt_len=14, canvas_len=6, sliding_window=16)

    assert mapping.commit_positions == (14, 15, 16, 17, 18, 19)
    assert mapping.sliding_commit_slots == (14, 15, 0, 1, 2, 3)


def test_qb2_default_commit_slots_for_256_canvas_after_2k_prompt():
    mapping = KVPhaseMapping(prompt_len=2048)

    assert len(mapping.commit_positions) == 256
    assert mapping.commit_positions[0] == 2048
    assert mapping.commit_positions[-1] == 2303
    assert mapping.sliding_commit_slots == tuple(range(256))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_len": -1},
        {"prompt_len": 0, "canvas_len": 0},
        {"prompt_len": 0, "sliding_window": 0},
    ],
)
def test_mapping_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        KVPhaseMapping(**kwargs)
