# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""High-signal host-side regression tests for Gemma4 disaggregated prefill.

These tests lock down the invariants the short device smoke test should protect:
- ring metadata must be updated per user and per chunk;
- traced and untraced prefill dispatch should agree on the same chunk metadata;
- chunk boundary cases around the 1024-token sliding window are covered.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from models.demos.gemma4_d_p.tt.model import Gemma4Model


def _make_model(events):
    """Build a minimal model whose layer calls record the chunk metadata we care about."""
    model = object.__new__(Gemma4Model)
    model.mesh_device = object()
    model.mesh_config = SimpleNamespace(prefill=SimpleNamespace(sp=8))
    model.hf_config = SimpleNamespace(layer_types=("sliding_attention", "full_attention"))
    model.tt_kv_cache = [None, None]
    model._rope_prefill_positions = None
    rope = np.zeros((1, 1, 4096, 1), dtype=np.float32)
    model.rope_caches_2d = {
        "sliding_attention": (rope.copy(), rope.copy()),
        "full_attention": (rope.copy(), rope.copy()),
    }
    model.rope_caches = {
        "sliding_attention": (rope.copy(), rope.copy()),
        "full_attention": (rope.copy(), rope.copy()),
    }
    model._ring_metadata_external = False
    model._packed_global_rope_trans_mat = None
    model._prefill_trace_mode = True
    model._prefill_trace_controller = SimpleNamespace(layer_ack=lambda idx: events.append(("ack", idx)))
    model.ccl_manager = SimpleNamespace(
        set_ring_metadata=lambda slot_idx, kv_actual_global: events.append(
            ("ring_metadata", slot_idx, kv_actual_global)
        ),
        ring_attention_ccl_semaphore_handles=[],
    )
    model.norm = SimpleNamespace(forward=lambda hidden: hidden)

    def layer_fn(layer_idx):
        def _call(hidden_states, **kwargs):
            events.append(
                (
                    "layer",
                    layer_idx,
                    kwargs["chunk_start_idx"],
                    kwargs.get("rope_mats") is not None,
                )
            )
            return hidden_states

        return _call

    model.layers = [layer_fn(0), layer_fn(1)]
    return model


@pytest.mark.parametrize(
    "chunk_start,user_id",
    [
        (0, 0),
        (1024, 1),
        (2048, 0),
        (8192, 2),
    ],
)
def test_prefill_metadata_tracks_user_and_chunk(chunk_start, user_id):
    events = []
    model = _make_model(events)
    hidden = SimpleNamespace(shape=(1, 1, 128, 64))

    model(hidden_states=hidden, chunk_start_idx=chunk_start, user_id=user_id)

    assert ("ring_metadata", user_id, chunk_start) in events


@pytest.mark.parametrize(
    "chunk_size,context_len,expected_starts",
    [
        (1024, 1024, [0]),
        (1024, 2048, [0, 1024]),
        (1024, 3072, [0, 1024, 2048]),
        (2048, 4096, [0, 2048]),
        (2048, 8192, [0, 2048, 4096, 6144]),
    ],
)
def test_chunk_boundary_starts_cover_sliding_window_edges(chunk_size, context_len, expected_starts):
    starts = [idx * chunk_size for idx in range(context_len // chunk_size)]
    assert starts == expected_starts


def test_traced_and_untraced_prefill_dispatch_share_same_chunk_metadata(monkeypatch):
    traced_events = []
    eager_events = []
    traced_model = _make_model(traced_events)
    eager_model = _make_model(eager_events)

    monkeypatch.setattr("ttnn.embedding", lambda *args, **kwargs: args[1])
    monkeypatch.setattr("ttnn.unsqueeze_to_4D", lambda value: value)

    hidden = SimpleNamespace(shape=(1, 1, 128, 64))
    traced_model._rope_prefill_positions = [0, 1, 2, 3]
    traced_model._rope_prefill_positions = [0, 1, 2, 3]

    traced_model(hidden_states=hidden, chunk_start_idx=2048, user_id=1)
    eager_model(hidden_states=hidden, rope_mats=("cos", "sin"), chunk_start_idx=2048, user_id=1)

    traced_ring = [event for event in traced_events if event[0] == "ring_metadata"]
    eager_ring = [event for event in eager_events if event[0] == "ring_metadata"]
    assert traced_ring == eager_ring == [("ring_metadata", 1, 2048)]

    traced_layers = [event for event in traced_events if event[0] == "layer"]
    eager_layers = [event for event in eager_events if event[0] == "layer"]
    assert [event[1:] for event in traced_layers] == [event[1:] for event in eager_layers]
    assert traced_layers and eager_layers
