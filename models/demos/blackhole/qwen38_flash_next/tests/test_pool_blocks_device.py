# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``pool_blocks`` on the held one-by-eight mesh: copy_blocks with mixed copy / zero / skip pairs (both pools bitwise
== the torch reference, untouched blocks unchanged), and the state <-> block byte copies (an fp32 TILE batch row and a
bf16 ROW_MAJOR tensor at a byte offset; the pool bytes == the reference, the round trip == the source bitwise)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.pool_blocks import (
    ZERO_SOURCE,
    block_to_state,
    copy_blocks,
    state_to_block,
    zero_blocks,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
BLOCKS = 8


def up(mesh, tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    first = replicas[0].contiguous().view(torch.uint8)
    assert all(torch.equal(first, other.contiguous().view(torch.uint8)) for other in replicas[1:])
    return replicas[0]


def pools(mesh, block_size, generator):
    kv = torch.randn(1, 1, BLOCKS * block_size, 512, generator=generator).to(torch.bfloat16)
    index = torch.randn(BLOCKS, 1, block_size // 4, 128, generator=generator).to(torch.bfloat16)
    return kv, index, up(mesh, kv, ttnn.bfloat16), up(mesh, index, ttnn.bfloat16, ttnn.TILE_LAYOUT)


@MESH
@PARAMS
@pytest.mark.parametrize("block_size", [128, 1024])
def test_copy_and_zero_blocks_match_the_reference(mesh_device, block_size):
    generator = torch.Generator().manual_seed(7)
    kv, index, kv_dev, index_dev = pools(mesh_device, block_size, generator)
    pairs = [(1, 2), (ZERO_SOURCE, 4), (5, 5), (6, 3), (0, 7)]
    copy_blocks(kv_dev, index_dev, pairs, block_size=block_size)
    want_kv, want_index = ref.pool_after_copy_blocks(kv, index, pairs, block_size)
    assert torch.equal(host(kv_dev), want_kv)
    assert torch.equal(host(index_dev), want_index)
    for untouched in (0, 1, 5, 6):
        window = slice(untouched * block_size, (untouched + 1) * block_size)
        assert torch.equal(host(kv_dev)[0, 0, window], kv[0, 0, window])
        assert torch.equal(host(index_dev)[untouched], index[untouched])

    kv2, index2 = want_kv, want_index
    zero_blocks(kv_dev, index_dev, [1, 2], block_size=block_size)
    want_kv, want_index = ref.pool_after_copy_blocks(kv2, index2, [(ZERO_SOURCE, 1), (ZERO_SOURCE, 2)], block_size)
    assert torch.equal(host(kv_dev), want_kv)
    assert torch.equal(host(index_dev), want_index)


@MESH
@PARAMS
def test_copy_blocks_rejects_overlapping_pairs(mesh_device, expect_error):
    generator = torch.Generator().manual_seed(1)
    _, _, kv_dev, index_dev = pools(mesh_device, 128, generator)
    with expect_error(ValueError, "both a dst and a src"):
        copy_blocks(kv_dev, index_dev, [(1, 2), (2, 3)], block_size=128)
    with expect_error(ValueError, "must be distinct"):
        copy_blocks(kv_dev, index_dev, [(1, 2), (3, 2)], block_size=128)
    with expect_error(ValueError, "is outside the"):
        copy_blocks(kv_dev, index_dev, [(1, BLOCKS)], block_size=128)


@MESH
@PARAMS
def test_state_block_round_trip(mesh_device):
    block_size = 1024
    generator = torch.Generator().manual_seed(11)
    kv, _, kv_dev, _ = pools(mesh_device, block_size, generator)
    kv_bytes = kv.reshape(-1).contiguous().view(torch.uint8)

    recurrent = torch.randn(2, 12, 128, 128, generator=generator)
    ring = torch.randn(1, 1, 4, 2560, generator=generator).to(torch.bfloat16)
    recurrent_dev = up(mesh_device, recurrent, ttnn.float32, ttnn.TILE_LAYOUT)
    ring_dev = up(mesh_device, ring, ttnn.bfloat16)
    block, ring_offset = 3, 12 * 128 * 128 * 4

    state_to_block(recurrent_dev, kv_dev, block, block_size=block_size, src_batch_index=1)
    state_to_block(ring_dev, kv_dev, block, block_size=block_size, byte_offset=ring_offset)
    want = ref.pool_after_state_to_block(kv_bytes, ref.tile_order_bytes(recurrent[1]), block, block_size, 0)
    want = ref.pool_after_state_to_block(
        want.view(torch.uint8), ring.reshape(-1).contiguous().view(torch.uint8), block, block_size, ring_offset
    )
    got = host(kv_dev).reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(got, want)

    recurrent_back = up(mesh_device, torch.zeros(2, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT)
    ring_back = up(mesh_device, torch.zeros(1, 1, 4, 2560, dtype=torch.bfloat16), ttnn.bfloat16)
    block_to_state(kv_dev, recurrent_back, block, block_size=block_size, dst_batch_index=1)
    block_to_state(kv_dev, ring_back, block, block_size=block_size, byte_offset=ring_offset)
    back = host(recurrent_back)
    assert torch.equal(back[1].contiguous().view(torch.uint8), recurrent[1].contiguous().view(torch.uint8))
    assert not back[0].any()
    assert torch.equal(host(ring_back), ring)
    assert torch.equal(host(kv_dev).reshape(-1).contiguous().view(torch.uint8), want)
