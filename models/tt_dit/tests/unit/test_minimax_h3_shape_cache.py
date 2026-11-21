# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU contract for the H3 denoise preamble's per-shape reuse: what counts as the same shape and when a
resident tensor is trusted to still hold its value. No device."""

from __future__ import annotations

import pytest
import torch

from ...pipelines.minimax_h3 import packing
from ...pipelines.minimax_h3 import pipeline_minimax_h3 as pipeline
from ...utils import tracing

KEY = ("shape",)
RUNG_TENSORS = ("rope_cos", "rope_sin", "adaln", "tsi", "assembly_idx")


@pytest.fixture
def tracers(monkeypatch):
    """The live trace count per device, replaced by a dict the test fills."""
    live = {}
    monkeypatch.setattr(tracing.Tracer, "_traces_live", live)
    return live


def _layout(text_len=40, num_frames=124):
    return packing.build_packed_sequence(
        torch.ones(text_len, dtype=torch.long),
        packing.video_latent_num_frames(num_frames),
        768 // 16,
        1344 // 16,
        packing.audio_latent_num_frames(num_frames),
        (1, 2, 2),
        (),
    )


def _bound_state(names=RUNG_TENSORS):
    state = pipeline._BucketState(shape_key=KEY)
    for name in names:
        getattr(state, name).update(object(), traced=False)
    return state


def test_layout_digest_separates_shapes():
    assert _layout().digest() == _layout().digest()
    assert _layout(text_len=41).digest() != _layout().digest()
    assert _layout(num_frames=73).digest() != _layout().digest()
    retagged = _layout()
    retagged.token_tags = retagged.token_tags.clone()
    retagged.token_tags[0] += 1
    assert retagged.digest() != _layout().digest()


def test_state_tensor_keeps_value_only_when_bound_before_any_trace(tracers):
    state = tracing.StateTensor()
    assert not state.keeps_value

    tracers[0] = 0  # a tracer exists but has captured nothing
    state.update(object(), traced=False)
    assert state.keeps_value
    # A trace captured later sees the buffer allocated, so its replay leaves it alone.
    tracers[0] = 1
    assert state.keeps_value

    state.update(object(), traced=False)  # rebound under a live trace: its replay may overwrite it
    assert not state.keeps_value

    tracers[0] = 0  # the trace was released
    state.update(object(), traced=False)
    assert state.keeps_value


def test_bucket_state_holds_only_its_own_shape(tracers):
    assert not pipeline._BucketState().holds(KEY)

    state = _bound_state()
    assert state.holds(KEY)
    assert not state.holds(("other",))

    state.tile_rows = True  # the tile-row pair is in use but was never bound
    assert not state.holds(KEY)
    assert _bound_state((*RUNG_TENSORS, "adaln_tile_map", "adaln_expanded")).holds(KEY)
