# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model-owned KV cache allocation (``Transformer.allocate_kv_cache``).

The model now owns its KV cache: ``allocate_kv_cache(per_layer_specs)`` builds one
``[k, v]`` pair per unique ``tensor_idx`` (vLLM HMA buffer sharing) and installs it
into each attention layer's ``layer_past``. Layers sharing a ``tensor_idx`` get the
SAME tensor object.

Real ttnn allocation requires a mesh device, so we stub each attention's
``_build_kv_pair`` (the single tensor-construction primitive) and verify call
structure + sharing, not tensor contents. The model instance is built via
``Transformer.__new__`` (skipping ``__init__``) with fake layers, so we exercise the
real ``allocate_kv_cache`` / ``kv_cache_per_layer`` / ``_layer_attentions`` logic.
"""

from types import SimpleNamespace

import torch

from models.tt_transformers.tt.model import Transformer


class _FakeAttention:
    """Stands in for a decoder layer's attention: records build calls and
    returns a unique ``[k, v]`` object per call so sharing can be checked by
    identity."""

    def __init__(self):
        self.layer_past = None
        self.build_calls = []

    def _build_kv_pair(self, num_blocks, block_size, dtype=None, weight_cache_path=None, dummy_weights=False):
        self.build_calls.append((num_blocks, block_size))
        return [object(), object()]  # fresh (k, v) per call


def _make_model(num_layers):
    model = Transformer.__new__(Transformer)  # skip __init__ (no device needed)
    model.layers = [SimpleNamespace(attention=_FakeAttention()) for _ in range(num_layers)]
    return model


def test_allocates_one_pair_per_unique_tensor_idx():
    """Each unique ``tensor_idx`` builds exactly one (k, v) pair."""
    model = _make_model(3)
    specs = [
        ((4, 2, 32, 64), torch.bfloat16, 0),
        ((4, 2, 32, 64), torch.bfloat16, 1),
        ((4, 2, 32, 64), torch.bfloat16, 2),
    ]
    per_layer = model.allocate_kv_cache(specs)

    # One build per layer (all unique idx); num_blocks/block_size routed from shape.
    assert [a.build_calls for a in model._layer_attentions()] == [[(4, 32)], [(4, 32)], [(4, 32)]]
    assert len(per_layer) == 3
    assert all(len(layer) == 2 for layer in per_layer)
    assert model.kv_cache_allocated


def test_shared_tensor_idx_reuses_one_buffer():
    """Layers sharing a ``tensor_idx`` reference the same handle; build runs once
    per unique idx."""
    model = _make_model(3)
    specs = [
        ((4, 2, 32, 64), torch.bfloat16, 0),
        ((4, 2, 32, 64), torch.bfloat16, 1),
        ((4, 2, 32, 64), torch.bfloat16, 0),  # shares buffer with layer 0
    ]
    per_layer = model.allocate_kv_cache(specs)

    attns = model._layer_attentions()
    # Only tensor_idx 0 and 1 built → layer 0 built once, layer 1 once, layer 2 reused.
    assert attns[0].build_calls == [(4, 32)]
    assert attns[1].build_calls == [(4, 32)]
    assert attns[2].build_calls == []  # reused layer 0's buffer, no new build
    # Layers 0 and 2 are the SAME object; layer 1 is distinct.
    assert per_layer[0] is per_layer[2]
    assert per_layer[0] is not per_layer[1]


def test_uniform_path_gives_unique_buffer_per_layer():
    """The legacy uniform entry point (Generator.allocate_kv_cache → unique
    tensor_idx per layer) yields one independent buffer per layer."""
    model = _make_model(3)
    specs = [((8, 2, 32, 64), torch.bfloat16, i) for i in range(3)]
    per_layer = model.allocate_kv_cache(specs)

    assert all(a.build_calls == [(8, 32)] for a in model._layer_attentions())
    # All distinct objects.
    assert len({id(p) for p in per_layer}) == 3
