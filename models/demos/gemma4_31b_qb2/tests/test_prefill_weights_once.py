# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A prefill of several chunks rebuilds the layer's interleaved weights once; the outputs equal the per-chunk rebuild bit for bit."""

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator import build_generator


@pytest.fixture
def generator(qb2_mesh):
    gen = build_generator(qb2_mesh, layer_indices=[0, 5], max_seq_len=16384)
    yield gen
    gen.close()


def run(layer, qb2_mesh, rows, once):
    """The layer over `rows` fixed random rows on its own cache pages: (output, names of the weight rebuilds)."""
    layer._prepare_prefill_norm_stats()
    pages = (rows // 128 + 2 + 7) // 8 * 8
    table = (torch.randperm(pages, generator=torch.Generator().manual_seed(3)) + 1).reshape(1, pages).int()
    cache = layer.allocate_cache(physical_pages=pages + 2)
    replicate = ttnn.ReplicateTensorToMesh(qb2_mesh)
    pt = ttnn.from_torch(table, device=qb2_mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate)
    sharded = layer.prefill_input_width(rows) != layer.hidden_size
    x = torch.randn(1, 1, rows, layer.hidden_size, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x,
        device=qb2_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(qb2_mesh, dim=3) if sharded else replicate,
    )
    rebuilds = []
    original = layer._prefill_weight
    layer._prefill_weight = lambda name: rebuilds.append(name) or original(name)
    if not once:
        layer._rebuild_weights_once = lambda chunks: {}
    try:
        out = layer.prefill_forward(tx, page_table=pt, kv_cache=cache, start_pos=0)
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    finally:
        del layer._prefill_weight
        if not once:
            del layer._rebuild_weights_once
    assert layer.pref_weights == {}
    return (torch.cat(parts, dim=-1) if sharded else parts[0]), rebuilds


@pytest.mark.parametrize(
    "layer_index, rows", [(0, 3 * 1024), (0, 7168), (1, 7168)], ids=["sliding-3", "sliding-7", "full-2"]
)  # the loaded layers are checkpoint layers 0 (sliding, 1024-row chunks) and 5 (full, 6656-row chunks)
def test_chunks_share_one_weight_rebuild_with_identical_output(generator, qb2_mesh, layer_index, rows):
    layer = generator.model.layers[layer_index]
    chunks = -(-rows // layer.PREFILL_CHUNK)
    shared, rebuilt_once = run(layer, qb2_mesh, rows, once=True)
    separate, rebuilt_per_chunk = run(layer, qb2_mesh, rows, once=False)
    assert torch.equal(shared, separate)
    names = {"qkv" if layer.sliding else "qk", "o", "gate_up", "down"}
    assert sorted(rebuilt_once) == sorted(names)  # one rebuild of each weight for the whole span
    assert len(rebuilt_per_chunk) == len(names) * chunks and chunks > 1


def test_a_single_chunk_rebuilds_each_weight_once_as_before(generator, qb2_mesh):
    layer = generator.model.layers[0]
    _, rebuilds = run(layer, qb2_mesh, 1024, once=True)
    assert len(rebuilds) == 4
