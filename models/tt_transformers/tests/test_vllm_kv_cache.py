# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vLLM-side KV cache allocator helpers in
``generator_vllm.py``.

Verifies the new per-layer entry point (``allocate_vllm_kv_cache_per_layer``)
and that the legacy uniform-shape entry point (``allocate_vllm_kv_cache``)
still delegates to it bit-for-bit.

Real ttnn allocation requires a mesh device, so this test mocks
``allocate_replicated_zeros`` and the ``dp_model``
handles. We verify call structure and shape routing, not the resulting
tensor contents.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def dp_model():
    """One submesh handle whose optimizations return None (so the allocator
    falls back to the bfloat8_b default — keeps the test independent of
    the model's optimization config table)."""
    submesh = MagicMock()
    args = MagicMock()
    args.optimizations = None  # Force the bfloat8_b fallback path.
    model = MagicMock()
    model.mesh_device = submesh
    model.args = args
    return [model]


def _make_ttnn_mock():
    ttnn_mock = MagicMock()
    ttnn_mock.zeros.side_effect = lambda *a, **kw: ("tt-tensor", kw.get("dtype"), tuple(a[0]))
    ttnn_mock.bfloat8_b = "bfloat8_b-sentinel"
    ttnn_mock.bfloat16 = "bfloat16-sentinel"
    return ttnn_mock


def test_per_layer_allocates_one_kv_pair_per_unique_tensor(dp_model):
    """Each unique ``tensor_idx`` allocates one (k, v) pair; layers that
    share a ``tensor_idx`` reuse the same handles."""
    from models.tt_transformers.tt import generator_vllm

    # Layers 0, 1, 2 all use tensor_idx=0,1,2 respectively → three buffers.
    per_layer = [
        ((4, 2, 32, 64), torch.bfloat16, 0),
        ((4, 2, 32, 64), torch.bfloat16, 1),
        ((4, 2, 32, 64), torch.bfloat16, 2),
    ]

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock, patch.object(
        generator_vllm, "allocate_replicated_zeros", side_effect=lambda *a, **kw: ttnn_mock.zeros(*a, **kw)
    ):
        kv_cache = generator_vllm.allocate_vllm_kv_cache_per_layer(
            per_layer, dp_model=dp_model, tt_cache_path=Path("/tmp/tt-test-cache")
        )

    # One submesh, three layers, two tensors per layer (k, v) = 6 calls.
    assert ttnn_mock.zeros.call_count == 6
    assert len(kv_cache) == 1  # one submesh
    assert len(kv_cache[0]) == 3  # three layers
    assert all(len(layer) == 2 for layer in kv_cache[0])  # k, v


def test_shared_tensor_idx_reuses_one_buffer(dp_model):
    """Layers sharing a ``tensor_idx`` (HMA tensor sharing) point at the
    same underlying ttnn handles and only one allocation runs per
    ``tensor_idx``."""
    from models.tt_transformers.tt import generator_vllm

    # Layers 0 and 2 share tensor 0; layer 1 has its own tensor 1.
    per_layer = [
        ((4, 2, 32, 64), torch.bfloat16, 0),
        ((4, 2, 32, 64), torch.bfloat16, 1),
        ((4, 2, 32, 64), torch.bfloat16, 0),
    ]

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock, patch.object(
        generator_vllm, "allocate_replicated_zeros", side_effect=lambda *a, **kw: ttnn_mock.zeros(*a, **kw)
    ):
        kv_cache = generator_vllm.allocate_vllm_kv_cache_per_layer(
            per_layer, dp_model=dp_model, tt_cache_path=Path("/tmp/tt-test-cache")
        )

    # 2 unique tensor_idx values × 2 (k, v) = 4 allocations.
    assert ttnn_mock.zeros.call_count == 4
    # Layers 0 and 2 must reference the *same* handle list.
    assert kv_cache[0][0] is kv_cache[0][2]
    assert kv_cache[0][0] is not kv_cache[0][1]


def test_per_layer_uses_shape_without_host_zeros_or_cache_files(dp_model):
    from models.tt_transformers.tt import generator_vllm

    per_layer = [
        ((4, 2, 32, 64), torch.bfloat16, 0),
        ((8, 2, 32, 64), torch.bfloat16, 1),
        ((6, 2, 32, 64), torch.bfloat16, 0),
    ]

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock, patch.object(
        generator_vllm, "allocate_replicated_zeros", side_effect=lambda *a, **kw: ttnn_mock.zeros(*a, **kw)
    ), patch.object(torch, "zeros", side_effect=AssertionError("host zero allocation")):
        cache = generator_vllm.allocate_vllm_kv_cache_per_layer(
            per_layer, dp_model=dp_model, tt_cache_path=Path("/tmp/tt-test-cache")
        )

    calls = ttnn_mock.zeros.call_args_list
    assert [tuple(call.args[0]) for call in calls] == [(6, 2, 32, 64)] * 2 + [(8, 2, 32, 64)] * 2
    assert all("cache_file_name" not in call.kwargs for call in calls)
    assert cache[0][0] is cache[0][2]
    ttnn_mock.as_tensor.assert_not_called()


def test_legacy_uniform_shape_delegates_to_per_layer(dp_model):
    """The legacy ``allocate_vllm_kv_cache`` must produce identical output to
    calling ``allocate_vllm_kv_cache_per_layer`` with a per-layer triple
    list (each layer its own ``tensor_idx``), so existing single-group
    callers keep working unchanged."""
    from models.tt_transformers.tt import generator_vllm

    shape = (4, 2, 32, 64)
    dtype = torch.bfloat16
    num_layers = 3

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock, patch.object(
        generator_vllm, "allocate_replicated_zeros", side_effect=lambda *a, **kw: ttnn_mock.zeros(*a, **kw)
    ):
        legacy = generator_vllm.allocate_vllm_kv_cache(
            shape, dtype, num_layers, dp_model=dp_model, tt_cache_path=Path("/tmp/c")
        )
    legacy_call_count = ttnn_mock.zeros.call_count

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock, patch.object(
        generator_vllm, "allocate_replicated_zeros", side_effect=lambda *a, **kw: ttnn_mock.zeros(*a, **kw)
    ):
        per_layer = generator_vllm.allocate_vllm_kv_cache_per_layer(
            [(shape, dtype, i) for i in range(num_layers)],
            dp_model=dp_model,
            tt_cache_path=Path("/tmp/c"),
        )
    per_layer_call_count = ttnn_mock.zeros.call_count

    assert legacy_call_count == per_layer_call_count
    assert len(legacy[0]) == len(per_layer[0]) == num_layers
