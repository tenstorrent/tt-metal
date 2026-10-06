# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing/precision guards for opt-in shared decode batching."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder, _SharedMLP
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


@pytest.mark.parametrize("batch", [1, 2, 3, 7, 8, 16, 31, 32, 33])
@pytest.mark.parametrize("enabled", [False, True])
def test_only_measured_batch_range_selects_new_path(monkeypatch, batch, enabled):
    decoder = MultichipDecoder()
    shared = _SharedMLP.__new__(_SharedMLP)
    shared.decode_weights = (object(), object())
    decoder.layer = SimpleNamespace(shared_mlp=shared)
    decoder.batched_shared_decode = enabled
    decoder.topology = ttnn.Topology.Linear
    decoder.sharded_residual = False
    decoder.grouped_moe_reduce = decoder.fused_tail = True
    decoder._decode_shared_batch = Mock(return_value="batched")
    fallback = Mock(return_value="serialized")
    monkeypatch.setattr(OptimizedDecoder, "decode_forward", fallback)
    result = decoder.decode_forward(
        SimpleNamespace(shape=(1, 1, batch, 2816)),
        rope_mats=None,
        current_pos=None,
        cache_pos=None,
        page_table=None,
        kv_cache=None,
    )
    assert result == ("batched" if enabled and 8 <= batch <= 32 else "serialized")


@pytest.mark.parametrize("unsupported", ["sharded", "unpaired", "unfused", "no_weights", "other_shared", "ring"])
def test_incompatible_configurations_keep_existing_path(monkeypatch, unsupported):
    decoder = MultichipDecoder()
    shared = _SharedMLP.__new__(_SharedMLP)
    shared.decode_weights = None if unsupported == "no_weights" else (object(), object())
    decoder.layer = SimpleNamespace(shared_mlp=object() if unsupported == "other_shared" else shared)
    decoder.batched_shared_decode = True
    decoder.topology = ttnn.Topology.Ring if unsupported == "ring" else ttnn.Topology.Linear
    decoder.sharded_residual = unsupported == "sharded"
    decoder.grouped_moe_reduce = unsupported != "unpaired"
    decoder.fused_tail = unsupported != "unfused"
    decoder._decode_shared_batch = Mock(side_effect=AssertionError("unexpected new path"))
    monkeypatch.setattr(OptimizedDecoder, "decode_forward", Mock(return_value="serialized"))
    assert (
        decoder.decode_forward(
            SimpleNamespace(shape=(1, 1, 32, 2816)),
            rope_mats=None,
            current_pos=None,
            cache_pos=None,
            page_table=None,
            kv_cache=None,
        )
        == "serialized"
    )


def test_explicit_batched_decode_keeps_decode_weights_without_changing_prefill():
    shared = _SharedMLP.__new__(_SharedMLP)
    shared.decode_weights = (object(), object())
    shared._forward = Mock(return_value=object())
    value = SimpleNamespace(shape=(1, 1, 8, 2816))
    shared(value, reduce_output=False)
    shared._forward.assert_called_once_with(value, decode=False, reduce_output=False)
    shared._forward.reset_mock()
    shared.decode_batch(value, reduce_output=False)
    shared._forward.assert_called_once_with(value, decode=True, reduce_output=False)


@pytest.mark.parametrize("batch", [0, 33])
def test_invalid_shared_batch_is_rejected(batch, expect_error):
    shared = _SharedMLP.__new__(_SharedMLP)
    shared.decode_weights = (object(), object())
    with expect_error(ValueError, "at most 32 rows"):
        shared.decode_batch(SimpleNamespace(shape=(1, 1, batch, 2816)))
