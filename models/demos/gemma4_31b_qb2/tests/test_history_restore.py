# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Restoring a cached continuation's sliding history from KV pages: same values, and new block ids compile no programs.

The page id used to be part of a Python slice (cache[p : p + 1]) and so of the program hash: every request with new block
ids compiled new programs, and a prefill that compiles programs retires the decode traces.
"""

import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import build_generator


def _reference(generator, state, tables, slot, start):
    """The previous implementation: Python slices by page id."""
    histories = []
    for i, layer in enumerate(generator.model.layers):
        if not layer.sliding:
            histories.append(None)
            continue
        first = max(0, start - layer.window) // Decoder.PAGE_SIZE
        pages = tables[i][slot, first : start // Decoder.PAGE_SIZE].tolist()
        pair = []
        for cache in state.kv[i]:
            parts = [cache[p : p + 1, :, :, :] for p in pages]
            tail = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            pair.append(ttnn.typecast(tail, ttnn.bfloat16))
        histories.append(tuple(pair))
    return histories


def _host(tensor, mesh):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def test_restore_reads_the_same_pages_and_new_block_ids_add_no_programs(qb2_mesh):
    generator = build_generator(qb2_mesh, layer_indices=[0], max_seq_len=4096)
    state = generator.allocate_cache(batch=1, capacity=4096)
    # A toy pool with room for the block ids below (the serving pool is much larger than the default test cache).
    state.kv[0] = generator.model.layers[0].allocate_cache(physical_pages=40)
    rng = torch.Generator().manual_seed(5)
    pages_total = state.kv[0][0].shape[0]
    for tensor in state.kv[0]:
        host = torch.randn(tuple(tensor.shape), generator=rng).bfloat16()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host, dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh)
            ),
            tensor,
        )
    start = 1152
    counts = []
    for ids in ([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 12, 14, 16, 18, 20, 22, 24, 26], [3, 5, 7, 9, 11, 13, 15, 17, 19]):
        assert max(ids) < pages_total
        tables = {0: torch.tensor([ids + [0] * 4], dtype=torch.int32)}
        before = qb2_mesh.num_program_cache_entries()
        restored = generator._restore_prefill_history(state, tables, 0, start)
        counts.append(qb2_mesh.num_program_cache_entries() - before)
        expected = _reference(generator, state, tables, 0, start)
        for got, want in zip(restored[0], expected[0]):
            assert torch.equal(_host(got, qb2_mesh), _host(want, qb2_mesh))
    # The first restore compiles the slice (and concat/typecast) once; later restores with other block ids add nothing.
    assert counts[1] == 0 and counts[2] == 0, counts
