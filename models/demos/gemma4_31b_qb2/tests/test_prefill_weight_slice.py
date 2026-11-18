# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The interleaved prefill weights sliced straight from the sharded decode weight equal the copy-then-slice rebuild bit for bit."""

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator import build_generator


def host(mesh, tensor):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def copy_then_slice(layer, name):
    """The earlier rebuild: copy the whole fused decode weight to interleaved DRAM, then slice the copy."""
    full = ttnn.to_memory_config(layer.decode_weights["qkv" if name == "qk" else "gate_up"], ttnn.DRAM_MEMORY_CONFIG)
    shape = list(full.shape)
    zeros = [0] * len(shape)
    n = shape[-1]
    if name == "qk":
        width = n - layer.kv_heads * layer.head_dim
        return (ttnn.slice(full, zeros, shape[:-1] + [width]),)
    half = n // 2
    return ttnn.slice(full, zeros, shape[:-1] + [half]), ttnn.slice(full, zeros[:-1] + [half], shape[:-1] + [n])


@pytest.fixture
def generator(qb2_mesh):
    gen = build_generator(qb2_mesh, layer_indices=[0, 5], max_seq_len=8192)
    yield gen


@pytest.mark.parametrize("layer_index, name", [(0, "gate_up"), (1, "gate_up"), (1, "qk")])
def test_sliced_prefill_weights_equal_the_copied_ones(generator, qb2_mesh, layer_index, name):
    layer = generator.model.layers[layer_index]
    assert layer.sliding == (layer_index == 0)
    new = layer._prefill_weight(name)
    new = new if isinstance(new, tuple) else (new,)
    old = copy_then_slice(layer, name)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert a.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(a.shape) == tuple(b.shape)
        assert torch.equal(host(qb2_mesh, a), host(qb2_mesh, b))
