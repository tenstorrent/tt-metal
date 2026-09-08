# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for prefill ownership and migration acknowledgement ordering."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ttnn
from models.demos.gemma4_d_p.tt.model import Gemma4Model


@pytest.mark.parametrize("ack_mode", ["callback", "segmented_trace", "socket"])
def test_migration_ack_follows_each_layer_write(monkeypatch, ack_mode):
    events = []
    hidden = SimpleNamespace(shape=(1, 1, 1024, 64))
    model = object.__new__(Gemma4Model)
    model.mesh_device = object()
    model.hf_config = SimpleNamespace(layer_types=("sliding_attention", "full_attention"))
    model.tt_kv_cache = [None, None]
    model._rope_prefill_positions = None
    model.rope_caches_2d = {}
    model._ring_metadata_external = True
    model._packed_global_rope_trans_mat = None
    model._prefill_trace_mode = True
    model._prefill_trace_controller = (
        SimpleNamespace(layer_ack=lambda idx: events.append(("ack", idx))) if ack_mode == "segmented_trace" else None
    )
    model.norm = SimpleNamespace(forward=lambda x: events.append(("norm", None)) or x)

    def layer(idx):
        def forward(x, **kwargs):
            assert kwargs["chunk_start_idx"] == 8192
            assert kwargs["rope_mats"] == (idx, idx)
            events.append(("write", idx))
            return x

        return forward

    model.layers = [layer(0), layer(1)]
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _: events.append(("sync", None)))
    service, metadata = object(), object()

    def socket_ack(actual_service, *, metadata):
        assert actual_service is service
        assert metadata is metadata_msg
        events.append(("ack", len([e for e in events if e[0] == "write"]) - 1))

    metadata_msg = metadata
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "outbound_socket_service_sync", socket_ack)
    output = model(
        hidden,
        rope_mats={"sliding_attention": (0, 0), "full_attention": (1, 1)},
        chunk_start_idx=8192,
        on_layer_complete=lambda idx: events.append(("ack", idx)),
        d2h_service=service if ack_mode == "socket" else None,
        metadata_msg=metadata if ack_mode == "socket" else None,
    )
    assert output is hidden
    expected = []
    for idx in range(2):
        expected.append(("write", idx))
        if ack_mode == "callback":
            expected.append(("sync", None))
        expected.append(("ack", idx))
    assert events == expected + [("norm", None)]


@pytest.mark.parametrize("is_global", [False, True])
def test_attention_reuses_external_ring_cache_without_auxiliary_allocations(monkeypatch, is_global):
    from models.demos.gemma4_d_p.config import MeshConfig
    from models.demos.gemma4_d_p.tt import attention

    cache = object()
    monkeypatch.setattr(attention, "load_attention_weights", lambda **_: SimpleNamespace(is_global=is_global))
    allocate = Mock(side_effect=AssertionError("external caches must not allocate replacements or tail pools"))
    monkeypatch.setattr(attention, "init_ring_kv_cache", allocate)
    monkeypatch.setattr(attention, "init_packed_ring_kv_cache", allocate)
    monkeypatch.setattr(ttnn, "zeros", allocate)
    result = attention.Gemma4Attention(
        mesh_device=object(),
        config=SimpleNamespace(is_sliding=not is_global, sliding_window=1024),
        state_dict={},
        ccl_manager=object(),
        mesh_config=MeshConfig((8, 4)),
        layer_idx=0,
        ring_kv_cache=cache,
    )
    assert result.ring_kv_cache is cache
    allocate.assert_not_called()


@pytest.mark.parametrize("cached_qk", [False, True])
def test_global_projection_loads_only_selected_weight(monkeypatch, cached_qk):
    from models.demos.gemma4_d_p.tt.attention import weights

    monkeypatch.setenv("GEMMA4_TIED_QKV", "1")
    monkeypatch.setattr(weights, "_cached_tensor_exists", lambda _: cached_qk)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda _: None)
    loaded = []

    def as_tensor(_host, **kwargs):
        name = str(kwargs["cache_file_name"])
        loaded.append(name)
        return name

    monkeypatch.setattr(ttnn, "as_tensor", as_tensor)
    mesh_config = SimpleNamespace(
        tp=4,
        prefill=SimpleNamespace(sp=8),
        column_parallel=lambda _: None,
        row_parallel=lambda _: None,
    )
    config = SimpleNamespace(
        use_kv_tying=True, num_attention_heads=32, num_key_value_heads=4, head_dim=512, hidden_size=5376
    )
    result = weights.load_attention_weights(object(), config, {}, mesh_config, tensor_cache_path="/tmp/weights")
    assert (result.wqk is not None) == cached_qk
    assert (result.wqkv is not None) != cached_qk
    assert len([name for name in loaded if "/wqk" in name]) == 1


def test_last_token_projection_gathers_cp_before_slicing(monkeypatch):
    events = []
    hidden = SimpleNamespace(shape=(1, 1, 1024, 64), deallocate=Mock())
    gathered = SimpleNamespace(shape=(1, 1, 8192, 64), deallocate=Mock())
    token_tile = object()
    logits = object()
    model = object.__new__(Gemma4Model)

    def gather(actual):
        assert actual is hidden
        events.append("gather")
        return gathered

    def slice_tile(actual, start, end):
        assert actual is gathered
        assert start == (0, 0, 8160, 0)
        assert end == (1, 1, 8192, 64)
        events.append("slice")
        return token_tile

    def project(actual):
        assert actual is token_tile
        events.append("project")
        return logits

    model._cp_gather_prefill_sequence = gather
    model._apply_lm_head = project
    monkeypatch.setattr(ttnn, "slice", slice_tile)
    assert model.process_logits_after_prefill_trace(hidden, 8191) is logits
    assert events == ["gather", "slice", "project"]
    hidden.deallocate.assert_not_called()
    gathered.deallocate.assert_called_once_with(True)
