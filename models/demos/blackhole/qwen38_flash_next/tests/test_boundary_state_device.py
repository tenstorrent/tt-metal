# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Boundary state save / restore on the held one-by-eight mesh.

Real GDN and PLE states (the layer classes' own allocators) and a real device position stand in for layers 0-2 of the
chain; layer 3's QSA staging and raw-key ring are plain BF16 tiles zeroed through the ``zero_qsa`` hook.  The states
get random per-device contents, are saved into K/V pool blocks, reset, restored, and compared bitwise.  Pool bytes
outside the state ranges and the other blocks must stay untouched."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    Qwen38MeshContract,
    Qwen38TTNNDevicePosition,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDNState
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38TTNNPLEState

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
BLOCK_SIZE = 1024
BLOCKS = 8
TP = MESH_SHAPE[1]


def per_device(tensor):
    return [ttnn.to_torch(local).contiguous().view(torch.uint8) for local in ttnn.get_device_tensors(tensor)]


def fill_random(mesh, tensor, shard_dim, generator):
    """Different random contents on every device, written in place at the tensor's address."""

    local = [int(extent) for extent in tensor.shape]
    global_shape = list(local)
    global_shape[shard_dim] *= TP
    torch_dtype = torch.float32 if tensor.dtype == ttnn.float32 else torch.bfloat16
    host = ttnn.from_torch(
        torch.randn(*global_shape, generator=generator).to(torch_dtype),
        dtype=tensor.dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, shard_dim)),
    )
    ttnn.copy_host_to_device_tensor(host, tensor)


def layer_state(mesh, contract, index, generator):
    gdn = Qwen38TTNNGDNState.allocate(mesh, contract, layer_index=index)
    fill_random(mesh, gdn.recurrent, 1, generator)
    for tensor in gdn.conv:
        fill_random(mesh, tensor, 3, generator)
    ple = None
    if index == model_module.PLE_CHECKPOINT_LAYER:
        ple = Qwen38TTNNPLEState.allocate(mesh, contract)
        for tensor in ple.conv:
            fill_random(mesh, tensor, 3, generator)
        ple.token_context = torch.tensor([[11, 12]], dtype=torch.long)
    return SimpleNamespace(attention=gdn, ple=ple)


def pool(mesh, generator):
    host = torch.randn(1, 1, BLOCKS * BLOCK_SIZE, 512, generator=generator).to(torch.bfloat16)
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


@MESH
@PARAMS
def test_boundary_state_round_trip(mesh_device, expect_error):
    generator = torch.Generator().manual_seed(3)
    contract = Qwen38MeshContract(tuple(int(device_id) for device_id in mesh_device.get_device_ids()))
    layers = [layer_state(mesh_device, contract, index, generator) for index in range(3)]
    qsa = SimpleNamespace(
        kv_staging=ttnn.from_torch(
            torch.randn(1, 1, 32, 512).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        raw_key_ring=ttnn.from_torch(
            torch.randn(1, 1, 32, 128).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    )
    layers.append(SimpleNamespace(attention=qsa, ple=None))
    position = Qwen38TTNNDevicePosition.allocate(mesh_device, contract, position=7)
    state = SimpleNamespace(layers=tuple(layers), position=position)

    pool_a, pool_b = pool(mesh_device, generator), pool(mesh_device, generator)
    pool_a_before, pool_b_before = per_device(pool_a), per_device(pool_b)
    layer_blocks = {0: (pool_a, 3), 1: (pool_b, 2), 2: (pool_a, 5)}

    pieces = {i: model_module.boundary_state_pieces(state, i, block_size=BLOCK_SIZE) for i in range(3)}
    assert [offset for _, offset in pieces[0]] == [0] + [393_216 + i * 81_920 for i in range(4)]
    assert len(pieces[1]) == 5 + 9 and pieces[1][-1][1] == 720_896 + 8 * 20_480
    want = {i: [per_device(tensor) for tensor, _ in pieces[i]] for i in range(3)}

    model_module.save_boundary_state(state, layer_blocks, block_size=BLOCK_SIZE, position=2 * BLOCK_SIZE)

    # Pool bytes: the state ranges changed, every other block and the tail of each used block did not.
    after_a, after_b = per_device(pool_a), per_device(pool_b)
    row_bytes = 512 * 2
    for device in range(TP):
        for after, before, used in ((after_a, pool_a_before, (3, 5)), (after_b, pool_b_before, (2,))):
            a = after[device].reshape(BLOCKS, BLOCK_SIZE * row_bytes)
            b = before[device].reshape(BLOCKS, BLOCK_SIZE * row_bytes)
            for block in range(BLOCKS):
                if block not in used:
                    assert torch.equal(a[block], b[block]), f"block {block} changed"
        tail = 905_216
        for block in (3, 5):
            assert torch.equal(
                after_a[device].reshape(BLOCKS, -1)[block, 720_896:],
                pool_a_before[device].reshape(BLOCKS, -1)[block, 720_896:],
            )
        assert torch.equal(
            after_b[device].reshape(BLOCKS, -1)[2, tail:], pool_b_before[device].reshape(BLOCKS, -1)[2, tail:]
        )

    for layer in layers[:3]:
        layer.attention.reset_inplace()
        if layer.ple is not None:
            layer.ple.reset_inplace()
    for i in range(3):
        assert all(not torch.equal(a[0], b[0]) for a, b in zip(want[i], [per_device(t) for t, _ in pieces[i]]))

    def zero_qsa(index, qsa_state):
        for tensor in (qsa_state.kv_staging, qsa_state.raw_key_ring):
            ttnn.fill(tensor, 0.0, output_tensor=tensor)

    for layer in layers[:3]:
        layer.attention.conv_phase = 2
    model_module.restore_boundary_state(
        state, layer_blocks, block_size=BLOCK_SIZE, position=2 * BLOCK_SIZE, zero_qsa=zero_qsa
    )

    for i in range(3):
        got = [per_device(tensor) for tensor, _ in pieces[i]]
        for piece, (g, w) in enumerate(zip(got, want[i])):
            for device in range(TP):
                assert torch.equal(g[device], w[device]), f"layer {i} piece {piece} device {device}"
        assert layers[i].attention.conv_phase == 0
    assert layers[1].ple.token_context is None
    for tensor in (qsa.kv_staging, qsa.raw_key_ring):
        assert all(not bool(shard.any()) for shard in per_device(tensor))
    assert int(ttnn.to_torch(ttnn.get_device_tensors(position.scalar)[0]).flatten()[0]) == 2 * BLOCK_SIZE

    # Preconditions fail loudly.
    with expect_error(ValueError, "not a multiple of 4"):
        model_module.save_boundary_state(state, layer_blocks, block_size=BLOCK_SIZE, position=BLOCK_SIZE + 4)
