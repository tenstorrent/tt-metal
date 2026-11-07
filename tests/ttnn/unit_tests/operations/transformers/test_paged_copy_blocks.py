# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.paged_copy_blocks: copy or zero whole blocks of several paged caches in one dispatch.

A serving engine sends block copies (copy-on-write and prefix-cache reuse) and block zeroing (state of a new request)
as lists of block ids; the op applies one list to every cache of the model.
"""

import pytest
import torch

import ttnn

ZERO = 0xFFFFFFFF
NUM_BLOCKS = 16


def _caches(device, n, shape, dtype, mesh_mapper=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    hosts = [torch.randn(NUM_BLOCKS, *shape, generator=gen).to(torch.bfloat16) for _ in range(n)]
    devs = [
        ttnn.from_torch(h, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper) for h in hosts
    ]
    # bfloat8_b rounds on the way in; the reference is what the device holds before the op.
    return [ttnn.to_torch(d, mesh_composer=_composer(device, mesh_mapper)) for d in devs], devs


def _composer(device, mesh_mapper):
    return ttnn.ConcatMeshToTensor(device, dim=1) if mesh_mapper is not None else None


def _table(pairs, device, mesh_mapper=None):
    # int32 -1 becomes 0xFFFFFFFF in the uint32 device tensor.
    rows = [[-1 if v == ZERO else v for v in pair] for pair in pairs]
    return ttnn.from_torch(
        torch.tensor(rows, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if mesh_mapper is not None else None,
    )


def _expected(before, pairs):
    after = before.clone()
    for src, dst in pairs:
        if src == dst:
            continue
        after[dst] = 0 if src == ZERO else before[src]
    return after


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("shape", [(1, 64, 256), (1, 1024, 256), (8, 128, 128)], ids=["block64", "block1024", "heads8"])
def test_paged_copy_blocks(device, shape, dtype):
    pairs = [(3, 7), (ZERO, 2), (5, 5), (11, 0), (ZERO, 14)]
    before, caches = _caches(device, 4, shape, dtype)
    out = ttnn.experimental.paged_copy_blocks(caches, _table(pairs, device))
    assert [o.buffer_address() for o in out] == [c.buffer_address() for c in caches], "the op must work in place"
    for i, cache in enumerate(caches):
        assert torch.equal(ttnn.to_torch(cache), _expected(before[i], pairs)), f"cache {i}"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1024 * 1024}], indirect=True)
def test_paged_copy_blocks_trace_follows_table(device):
    """Capture once, then apply a different list of pairs by rewriting only the table."""
    shape = (1, 256, 256)
    before, caches = _caches(device, 2, shape, ttnn.bfloat16, seed=1)
    first = [(1, 2), (ZERO, 3)]
    second = [(4, 5), (6, 6)]
    table = _table(first, device)

    ttnn.experimental.paged_copy_blocks(caches, table)  # compile
    after_first = [ttnn.to_torch(c) for c in caches]
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.experimental.paged_copy_blocks(caches, table)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    ttnn.copy_host_to_device_tensor(_table(second, None), table)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
    for i, cache in enumerate(caches):
        assert torch.equal(after_first[i], _expected(before[i], first))
        assert torch.equal(ttnn.to_torch(cache), _expected(after_first[i], second)), f"cache {i}"


def test_paged_copy_blocks_on_mesh(mesh_device):
    n = mesh_device.get_num_devices()
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    pairs = [(2, 9), (ZERO, 4)]
    before, caches = _caches(mesh_device, 2, (n, 128, 256), ttnn.bfloat16, mesh_mapper=shard, seed=2)
    ttnn.experimental.paged_copy_blocks(caches, _table(pairs, mesh_device, mesh_mapper=shard))
    for i, cache in enumerate(caches):
        got = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        assert torch.equal(got, _expected(before[i], pairs)), f"cache {i}"


def test_paged_copy_blocks_rejects_mismatched_caches(device, expect_error):
    _, a = _caches(device, 1, (1, 64, 256), ttnn.bfloat16)
    _, b = _caches(device, 1, (1, 128, 256), ttnn.bfloat16)
    with expect_error(RuntimeError, "device of the first"):
        ttnn.experimental.paged_copy_blocks(a + b, _table([(0, 1)], device))
    with expect_error(RuntimeError, "num_pairs, 2"):
        bad = ttnn.from_torch(
            torch.zeros(1, 3, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        ttnn.experimental.paged_copy_blocks(a, bad)
