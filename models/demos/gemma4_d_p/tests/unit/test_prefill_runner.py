# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4PrefillAdapter, validate_params
from models.demos.gemma4_d_p.tt.runners.prefill_producer import iter_chunks, wait_for_layers
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def service_params():
    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=262144,
        chunk_size=8192,
        num_users=6,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
    )


def test_adapter_uses_canonical_weight_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    adapter = get_adapter("gemma4_d_p")
    assert isinstance(adapter, Gemma4PrefillAdapter)
    assert adapter.weight_cache_path((8, 4)) == tmp_path / "tensor_cache_bf16_mesh8x4"
    validate_params(service_params())


@pytest.mark.parametrize(
    "override", [{"num_users": 7}, {"mesh_shape": (4, 8)}, {"num_layers": 30}, {"chunk_size": 4096}]
)
def test_unsupported_service_shapes(override, expect_error):
    with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
        validate_params(replace(service_params(), **override))


def test_slot_offsets_and_partial_final_chunk(expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.slot_ends = [8192, 16384, 0, 0, 0, 253952]
    runtime.validate_chunk(0, 8192, 16384)
    runtime.validate_chunk(1, 16384, 16385)
    runtime.validate_chunk(5, 253952, 262144)
    runtime.validate_chunk(5, 0, 8192)
    for chunk in [(6, 0, 8192), (0, 16384, 24576), (0, 1, 8193), (5, 253952, 262145)]:
        with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
            runtime.validate_chunk(*chunk)


def test_producer_interleaves_slots_and_pads_only_the_tail():
    prompts = [[11] * 8193, [22] * 8193]
    chunks = list(iter_chunks(prompts, pad_token_id=0))
    assert [chunk[:3] for chunk in chunks] == [(0, 0, 8192), (1, 0, 8192), (0, 8192, 8193), (1, 8192, 8193)]
    final_chunk = chunks[-1][3]
    assert final_chunk.shape == (8, 1, 1024)
    assert final_chunk.dtype == np.uint32
    assert final_chunk[0, 0, 0] == 22
    assert np.count_nonzero(final_chunk) == 1


def test_completion_wait_requires_exactly_sixty_layers(expect_error):
    counts = iter([20, 40])
    wait_for_layers(SimpleNamespace(try_consume_all=lambda: next(counts)), timeout_s=1)
    with expect_error(RuntimeError, "61 layer acknowledgments"):
        wait_for_layers(SimpleNamespace(try_consume_all=lambda: 61), timeout_s=1)
    with expect_error(TimeoutError, "0/60 layer acknowledgments"):
        wait_for_layers(SimpleNamespace(try_consume_all=lambda: 0), timeout_s=0)
