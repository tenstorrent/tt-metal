# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vLLM-side KV cache allocator helpers in
``generator_vllm.py``.

Verifies the new per-layer entry point (``allocate_vllm_kv_cache_per_layer``)
and that the legacy uniform-shape entry point (``allocate_vllm_kv_cache``)
still delegates to it bit-for-bit.

Real ttnn allocation requires a mesh device, so this test mocks
``ttnn.as_tensor`` / ``ttnn.ReplicateTensorToMesh`` and the ``dp_model``
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
    ttnn_mock.as_tensor.side_effect = lambda *a, **kw: ("tt-tensor", kw.get("dtype"), kw.get("cache_file_name"))
    ttnn_mock.bfloat8_b = "bfloat8_b-sentinel"
    ttnn_mock.bfloat16 = "bfloat16-sentinel"
    return ttnn_mock


def test_per_layer_allocates_one_kv_pair_per_spec(dp_model):
    from models.tt_transformers.tt import generator_vllm

    per_layer = [
        ((4, 2, 32, 64), torch.bfloat16),  # layer 0
        ((4, 1, 32, 64), torch.bfloat16),  # layer 1 (smaller — sliding)
        ((4, 2, 32, 64), torch.bfloat16),  # layer 2
    ]

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock:
        kv_cache = generator_vllm.allocate_vllm_kv_cache_per_layer(
            per_layer, dp_model=dp_model, tt_cache_path=Path("/tmp/tt-test-cache")
        )

    # One submesh, three layers, two tensors per layer (k, v) = 6 calls.
    assert ttnn_mock.as_tensor.call_count == 6
    assert len(kv_cache) == 1  # one submesh
    assert len(kv_cache[0]) == 3  # three layers
    assert all(len(layer) == 2 for layer in kv_cache[0])  # k, v


def test_per_layer_passes_each_shape_to_cache_filename(dp_model):
    """The cache filename is keyed on the per-layer shape; verify a layer
    with a smaller shape gets a different cache file from a larger one."""
    from models.tt_transformers.tt import generator_vllm

    per_layer = [
        ((4, 2, 32, 64), torch.bfloat16),
        ((4, 1, 32, 64), torch.bfloat16),  # half-sized
    ]

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock:
        generator_vllm.allocate_vllm_kv_cache_per_layer(
            per_layer, dp_model=dp_model, tt_cache_path=Path("/tmp/tt-test-cache")
        )

    cache_filenames = [str(call.kwargs["cache_file_name"]) for call in ttnn_mock.as_tensor.call_args_list]
    # Layer 0's shape (4, 2, 32, 64) must show up in two filenames (k+v),
    # and so must layer 1's (4, 1, 32, 64). Different shapes → different
    # filenames so caches don't collide.
    assert sum("(4, 2, 32, 64)" in f for f in cache_filenames) == 2
    assert sum("(4, 1, 32, 64)" in f for f in cache_filenames) == 2


def test_check_per_group_kwargs_strips_single_group_silently():
    """A single-element list carries no extra info beyond the legacy
    page_table arg; the wrapper drops it without complaint."""
    from models.tt_transformers.tt.generator_vllm import _check_per_group_kwargs

    kwargs = {"page_tables_per_group": ["t0"], "tokens": "tok"}
    _check_per_group_kwargs(kwargs, "FakeModel")

    assert "page_tables_per_group" not in kwargs
    assert kwargs == {"tokens": "tok"}


def test_check_per_group_kwargs_passes_when_kwarg_absent():
    """Legacy callers that don't set page_tables_per_group are unaffected."""
    from models.tt_transformers.tt.generator_vllm import _check_per_group_kwargs

    kwargs = {"tokens": "tok"}
    _check_per_group_kwargs(kwargs, "FakeModel")

    assert kwargs == {"tokens": "tok"}


def test_check_per_group_kwargs_raises_for_multi_group():
    """Hybrid input with multiple groups must error loudly: silently using
    only group 0 would corrupt KV state for the other groups."""
    from models.tt_transformers.tt.generator_vllm import _check_per_group_kwargs

    kwargs = {"page_tables_per_group": ["t0", "t1"], "tokens": "tok"}
    with pytest.raises(NotImplementedError, match="FakeModel"):
        _check_per_group_kwargs(kwargs, "FakeModel")


def test_legacy_uniform_shape_delegates_to_per_layer(dp_model):
    """The legacy ``allocate_vllm_kv_cache`` must produce identical output to
    calling ``allocate_vllm_kv_cache_per_layer`` with a uniform-spec list,
    so existing single-group callers keep working unchanged."""
    from models.tt_transformers.tt import generator_vllm

    shape = (4, 2, 32, 64)
    dtype = torch.bfloat16
    num_layers = 3

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock:
        legacy = generator_vllm.allocate_vllm_kv_cache(
            shape, dtype, num_layers, dp_model=dp_model, tt_cache_path=Path("/tmp/c")
        )
    legacy_call_count = ttnn_mock.as_tensor.call_count

    with patch.object(generator_vllm, "ttnn", new=_make_ttnn_mock()) as ttnn_mock:
        per_layer = generator_vllm.allocate_vllm_kv_cache_per_layer(
            [(shape, dtype)] * num_layers,
            dp_model=dp_model,
            tt_cache_path=Path("/tmp/c"),
        )
    per_layer_call_count = ttnn_mock.as_tensor.call_count

    assert legacy_call_count == per_layer_call_count
    assert len(legacy[0]) == len(per_layer[0]) == num_layers
