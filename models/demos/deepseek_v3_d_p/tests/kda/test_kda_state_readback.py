# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: golden state -> per-chip slab bytes -> published table -> ``kda_state_readback.read_layer`` -> golden.

The per-chip DRAM is built from the storage rule alone (SP row ``r`` of column ``c`` stores local heads
``[r * H_store, (r + 1) * H_store)``), laid out as the slabs lay
it out (ND round-robin ``[D, 32]`` shards in tile-face order; interleaved 384-byte convolution pages). The
table is a real ``KvChunkAddressTable`` filled by ``populate_kv_chunk_address_table_kda``; only
``read_dram_umd`` is replaced by a lookup into that DRAM.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry
from models.demos.deepseek_v3_d_p.tt.runners import kda_state_readback
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    kda_chunk_n_tokens,
    kda_segment_bytes,
    kda_window,
    populate_kv_chunk_address_table_kda,
)

NUM_BANKS, BASE, NUM_LAYERS, NUM_USERS = 8, 0x40_0000, 2, 2
MESH_ID = 0
disagg = ttnn.experimental.disaggregation


def _geometry(sp, tp, heads):
    return KdaContractGeometry(
        num_heads=heads, head_dim=128, conv_history=3, sequence_parallel_size=sp, tensor_parallel_size=tp
    )


def _band_bytes(band: torch.Tensor) -> bytes:
    """``[D, 32]`` FP32 -> the tile-face byte order of one ND shard (inverse of ``recurrent_segment_to_torch``)."""
    tiles = band.shape[0] // 32
    faces = band.reshape(tiles, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).contiguous()
    return faces.numpy().tobytes()


def _chip_dram(geometry, kind, golden, sp_row, tp_col):
    """``{(bank, address): bytes}`` of one chip's slab, from the storage rule and the slab layout alone."""
    dram = {}
    stored, local = geometry.stored_heads, geometry.local_heads
    first_head = tp_col * local + sp_row * stored
    segment_bytes = kda_segment_bytes(geometry, kind)
    shard = 0  # shards / pages in slab order: batch-major, then the store's heads
    for slot in range(NUM_USERS):
        for layer in range(NUM_LAYERS):
            state = golden[(slot, layer)]
            if kind == "kda_recurrent":
                pieces = [
                    _band_bytes(state[first_head + j, :, 32 * band : 32 * band + 32])
                    for j in range(stored)
                    for band in range(geometry.bands)
                ]
            else:
                pieces = [
                    state[:, (branch * geometry.num_heads + first_head + j) * 128 + 64 * half :][:, :64]
                    .contiguous()
                    .view(torch.int16)
                    .numpy()
                    .tobytes()
                    for branch in range(3)
                    for j in range(stored)
                    for half in range(geometry.halves)
                ]
            for piece in pieces:
                assert len(piece) == segment_bytes
                dram[(shard % NUM_BANKS, BASE + (shard // NUM_BANKS) * segment_bytes)] = piece
                shard += 1
    return dram


def _chip(sp_row, tp_col, tp):
    return sp_row * tp + tp_col


@pytest.mark.parametrize("kind", ["kda_recurrent", "kda_convolution"])
@pytest.mark.parametrize(
    "sp,tp,heads",
    [(8, 4, 96), (8, 1, 24), (4, 1, 24), (2, 2, 48)],
    ids=["galaxy-SP8xTP4", "loudbox-SP8xTP1", "quietbox-SP4xTP1", "SP2xTP2"],
)
def test_published_table_reads_back_the_golden_state(sp, tp, heads, kind, monkeypatch):
    geometry = _geometry(sp, tp, heads)
    torch.manual_seed(7)
    golden = {}
    for key in [(slot, layer) for slot in range(NUM_USERS) for layer in range(NUM_LAYERS)]:
        if kind == "kda_recurrent":
            golden[key] = torch.randn(heads, 128, 128)
        else:
            golden[key] = torch.randn(3, 3 * heads * 128).to(torch.bfloat16)

    dram = {}  # unique id -> chip DRAM
    for sp_row in range(sp):
        for tp_col in range(tp):
            dram[1000 + _chip(sp_row, tp_col, tp)] = _chip_dram(geometry, kind, golden, sp_row, tp_col)

    config = disagg.KvChunkAddressTableConfig()
    config.num_layers = NUM_LAYERS
    config.max_sequence_length = kda_window(geometry)
    config.num_slots = NUM_USERS
    config.chunk_n_tokens = kda_chunk_n_tokens(geometry, kind)
    config.chunk_size_bytes = kda_segment_bytes(geometry, kind)
    config.tensor_type = disagg.KvChunkTensorType.Snapshot
    table = disagg.KvChunkAddressTable(config)
    layout = [
        {
            "rank": 0,
            "first_layer": 0,
            "count": NUM_LAYERS,
            "base_addr": BASE,
            "num_banks": NUM_BANKS,
            "host_tag": 5,
            "fnids": [
                [ttnn.FabricNodeId(ttnn.MeshId(MESH_ID), _chip(row, col, tp)) for col in range(tp)] for row in range(sp)
            ],
        }
    ]
    populate_kv_chunk_address_table_kda(
        table, config, (sp, tp), 0, 1, geometry, kind, num_users=NUM_USERS, config_id=0, stage_layout=layout
    )
    holders = {
        len(table.get_device_group(disagg.DeviceGroupIndex(i)).fabric_node_ids)
        for i in range(table.num_device_groups())
    }
    assert holders == {1}

    reads = []

    def read_dram_umd(unique_id, noc_addr, size):
        reads.append(unique_id)
        return dram[unique_id][(noc_addr >> 32, noc_addr & 0xFFFFFFFF)][:size]

    monkeypatch.setattr(disagg, "read_dram_umd", read_dram_umd, raising=False)
    device_map = {(MESH_ID, chip): 1000 + chip for chip in range(sp * tp)}
    for slot in range(NUM_USERS):
        for layer in range(NUM_LAYERS):
            state, status = kda_state_readback.read_layer(
                table, 0, kind, layer, slot, geometry, kda_state_readback.umd_chunk_reader(device_map)
            )
            assert status == "ok"
            assert torch.equal(state, golden[(slot, layer)]), (slot, layer)
    assert set(reads) == set(device_map.values())  # every chip serves its own heads


def test_readback_refuses_a_table_published_with_replicated_groups(monkeypatch, expect_error):
    """A table from before the SP sharding (one group per TP column spanning its SP rows) names a primary chip that
    holds other heads at other addresses: the read-back must fail loudly, not read the wrong bytes."""
    geometry = _geometry(2, 2, 48)
    config = disagg.KvChunkAddressTableConfig()
    config.num_layers = 1
    config.max_sequence_length = kda_window(geometry)
    config.num_slots = 1
    config.chunk_n_tokens = kda_chunk_n_tokens(geometry, "kda_recurrent")
    config.chunk_size_bytes = kda_segment_bytes(geometry, "kda_recurrent")
    config.tensor_type = disagg.KvChunkTensorType.Snapshot
    table = disagg.KvChunkAddressTable(config)
    column = [ttnn.FabricNodeId(ttnn.MeshId(MESH_ID), _chip(row, 0, 2)) for row in range(2)]
    location = disagg.KvCacheLocation()
    location.noc_addr = BASE
    location.size_bytes = kda_segment_bytes(geometry, "kda_recurrent")
    location.device_group_index = table.add_device_group(column)
    table.set(0, 0, 0, location)
    monkeypatch.setattr(disagg, "read_dram_umd", lambda *_: pytest.fail("must not read"), raising=False)
    device_map = {(MESH_ID, chip): chip for chip in range(4)}
    with expect_error(ValueError, "device group of 2 chips, but every segment is stored on one chip"):
        kda_state_readback.read_layer(
            table, 0, "kda_recurrent", 0, 0, geometry, kda_state_readback.umd_chunk_reader(device_map)
        )
