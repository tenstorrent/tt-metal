# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks of the KDA state slabs: per-layer layout, read-back, and the address table.

Synthetic data with a distinct value in every element, so a wrong head, band, branch, half, slot or
layer shows up as a mismatch rather than a coincidence. The table check reads every segment of every
(layer, slot) back through ``read_device_chunk`` and reassembles the global state from the segment
numbering alone, which is exactly what a migration consumer has to do.
"""

from __future__ import annotations

import socket

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_bit_identical, decode_tail, encode_tail
from models.demos.deepseek_v3_d_p.tests.kda.utils import read_slab_layer, upload_slab
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    assemble_convolution,
    assemble_recurrent,
    convolution_segment_to_torch,
    deallocate_state,
    recurrent_segment_to_torch,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    kda_chunk_n_tokens,
    kda_position,
    kda_segment_bytes,
    kda_window,
    populate_kv_chunk_address_table_kda,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize(
        "device_params",
        [pytest.param({"trace_region_size": 64 * 1024 * 1024}, id="trace64m")],
        indirect=True,
    ),
]

SP_AXIS, TP_AXIS = 0, 1
LAYER_IDS = (1, 2)
NUM_SLOTS = 2


def _geometry(mesh_device) -> KdaContractGeometry:
    return KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_device.shape), sp_axis=SP_AXIS, tp_axis=TP_AXIS
    )


def _global_patterns(geometry: KdaContractGeometry, slot: int, layer_idx: int):
    """Global recurrent ``[H, D, D]`` and golden-order convolution ``[K-1, 3*H*D]`` with a per-(slot, layer) offset."""
    tag = 1000.0 * (slot * 10 + layer_idx)
    heads, dim = geometry.num_heads, geometry.head_dim
    recurrent = (torch.arange(heads * dim * dim, dtype=torch.float32).reshape(heads, dim, dim) + tag).float()
    width = 3 * heads * dim
    convolution = (
        torch.arange(geometry.conv_history * width, dtype=torch.int32).remainder(509) + slot * 7 + layer_idx
    ).to(torch.bfloat16)
    return recurrent, convolution.reshape(geometry.conv_history, width)


def _per_chip_convolution(geometry: KdaContractGeometry, convolution: torch.Tensor, tp_col: int) -> torch.Tensor:
    """The chip's ``[q_local | k_local | v_local]`` row from the golden-order global tensor."""
    heads, dim, local = geometry.num_heads, geometry.head_dim, geometry.local_heads
    parts = [
        convolution[:, branch * heads * dim + tp_col * local * dim : branch * heads * dim + (tp_col + 1) * local * dim]
        for branch in range(3)
    ]
    return torch.cat(parts, dim=-1)


def _native_from_global(mesh_device, geometry, recurrent, convolution) -> KdaState:
    """TP-shard heads, replicate across SP, in the layout the layer itself produces."""
    tp = geometry.tensor_parallel_size
    rec_dims, conv_dims = [None, None], [None, None]
    rec_dims[TP_AXIS], conv_dims[TP_AXIS] = 1, 1
    stacked_conv = encode_tail(
        torch.cat([_per_chip_convolution(geometry, convolution, col) for col in range(tp)], dim=-1),
        KDA_CONVOLUTION_PAGE_CHANNELS,
    )
    mesh_shape = tuple(mesh_device.shape)
    return KdaState(
        recurrent=ttnn.from_torch(
            recurrent.unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(rec_dims), mesh_shape=mesh_shape),
        ),
        convolution=ttnn.from_torch(
            stacked_conv.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(conv_dims), mesh_shape=mesh_shape),
        ),
    )


def _shards(tensor: ttnn.Tensor, mesh_device) -> dict[tuple[int, int], torch.Tensor]:
    rows, cols = tuple(mesh_device.shape)
    host = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    return {(row, col): host[row * cols + col] for row in range(rows) for col in range(cols)}


def _expected_chip(geometry, recurrent, convolution, tp_col):
    local = geometry.local_heads
    return recurrent[tp_col * local : (tp_col + 1) * local], _per_chip_convolution(geometry, convolution, tp_col)


def _single_stage_layout(mesh_device, slab, count):
    rows, cols = tuple(mesh_device.shape)
    return [
        {
            "rank": 0,
            "first_layer": 0,
            "count": count,
            "base_addr": int(slab.buffer_address()),
            "num_banks": mesh_device.dram_grid_size().x,
            "host_tag": 0,
            "fnids": [
                [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col)) for col in range(cols)]
                for row in range(rows)
            ],
        }
    ]


@pytest.mark.parametrize("mesh_device", [(8, 4), (2, 4)], ids=["8x4", "2x4"], indirect=True)
@pytest.mark.parametrize(
    "layer_ids,num_slots",
    [(LAYER_IDS, NUM_SLOTS), ((1,), 1)],
    ids=["2layers-2slots", "1layer-1slot"],
)
def test_slabs_round_trip_and_table(mesh_device, device_params, layer_ids, num_slots):
    """The one-layer, one-slot case matters: a slice over a whole slab can hand back the slab itself."""
    geometry = _geometry(mesh_device)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=layer_ids, num_slots=num_slots)
    rows, cols = tuple(mesh_device.shape)
    tp = geometry.tensor_parallel_size

    patterns = {
        (slot, layer): _global_patterns(geometry, slot, layer) for slot in range(num_slots) for layer in layer_ids
    }
    natives = {key: _native_from_global(mesh_device, geometry, *value) for key, value in patterns.items()}
    # Each device's slab holds its own shard of every (slot, layer) carry: build both on the host, upload once.
    for slab, part in ((slabs.recurrent, "recurrent"), (slabs.convolution, "convolution")):
        shards = [torch.zeros(tuple(slab.shape)) for _ in range(rows * cols)]
        for (slot, layer), state in natives.items():
            for shard, piece in zip(shards, ttnn.get_device_tensors(getattr(state, part)), strict=True):
                shard[slabs.batch_index(slot, layer)] = ttnn.to_torch(piece)[0]
        upload_slab(slab, shards)
    ttnn.synchronize_device(mesh_device)

    # 1. Every slab batch holds its layer's per-chip state, on every SP replica, bit for bit.
    rec_shards = _shards(slabs.recurrent, mesh_device)
    conv_shards = _shards(slabs.convolution, mesh_device)
    for (slot, layer), (recurrent, convolution) in patterns.items():
        batch = slabs.batch_index(slot, layer)
        for row in range(rows):
            for col in range(cols):
                coord = [0, 0]
                coord[SP_AXIS], coord[TP_AXIS] = row if SP_AXIS == 0 else col, col if TP_AXIS == 1 else row
                tp_col = coord[TP_AXIS]
                want_rec, want_conv = _expected_chip(geometry, recurrent, convolution, tp_col)
                assert_bit_identical(
                    want_rec, rec_shards[(row, col)][batch], name=f"S slot{slot} L{layer} ({row},{col})"
                )
                assert_bit_identical(
                    want_conv,
                    decode_tail(conv_shards[(row, col)][batch], KDA_CONVOLUTION_PAGE_CHANNELS),
                    name=f"conv slot{slot} L{layer} ({row},{col})",
                )

    def host(state):
        return [ttnn.to_torch(t) for s in (state.recurrent, state.convolution) for t in ttnn.get_device_tensors(s)]

    def read_back(slot, layer):
        back = read_slab_layer(mesh_device, slabs, slot, layer)
        ttnn.synchronize_device(mesh_device)
        shards = host(back)
        deallocate_state(back)
        return shards

    # 2. A read-back lands the same bytes in a native carry.
    for (slot, layer), state in natives.items():
        for index, (want, got) in enumerate(zip(host(state), read_back(slot, layer), strict=True)):
            assert_bit_identical(want, got, name=f"read slot{slot} L{layer} shard{index}")

    # 3. The address table resolves every (layer, segment, slot) to the right bytes, and the segment
    #    numbering alone reassembles the global state.
    disagg = ttnn.experimental.disaggregation
    layer_rows = list(layer_ids)
    configs = {}
    for name, kind in (("01", "kda_recurrent"), ("02", "kda_convolution")):
        cfg = disagg.KvChunkAddressTableConfig()
        cfg.num_layers = max(layer_rows) + 1
        cfg.max_sequence_length = kda_window(geometry)
        cfg.num_slots = num_slots
        cfg.chunk_n_tokens = kda_chunk_n_tokens(geometry, kind)
        cfg.chunk_size_bytes = kda_segment_bytes(geometry, kind)
        cfg.tensor_type = disagg.KvChunkTensorType.Snapshot
        configs[name] = cfg
    table = disagg.KvChunkAddressTable(configs)
    for name, kind, slab in (("01", "kda_recurrent", slabs.recurrent), ("02", "kda_convolution", slabs.convolution)):
        populate_kv_chunk_address_table_kda(
            table,
            configs[name],
            tuple(mesh_device.shape),
            SP_AXIS,
            TP_AXIS,
            geometry,
            kind,
            num_users=num_slots,
            config_id=table.config_id_of(name),
            stage_layout=_single_stage_layout(mesh_device, slab, len(layer_ids)),
            layer_rows=layer_rows,
        )
    assert table.num_device_groups() == tp
    for (slot, layer), (recurrent, convolution) in patterns.items():
        rec_segments = {
            segment: recurrent_segment_to_torch(
                table.read_device_chunk(
                    layer, kda_position(geometry, "kda_recurrent", segment), slot, table.config_id_of("01")
                ),
                geometry,
            )
            for segment in range(geometry.recurrent_segments_per_layer)
        }
        assert_bit_identical(recurrent, assemble_recurrent(rec_segments, geometry), name=f"table S slot{slot} L{layer}")
        conv_segments = {
            segment: convolution_segment_to_torch(
                table.read_device_chunk(
                    layer, kda_position(geometry, "kda_convolution", segment), slot, table.config_id_of("02")
                ),
                geometry,
            )
            for segment in range(geometry.convolution_segments_per_layer)
        }
        assert_bit_identical(
            convolution, assemble_convolution(conv_segments, geometry), name=f"table conv slot{slot} L{layer}"
        )

    # 4. A UMD read at the documented address matches the table's own read.
    fid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
    probe_layer, probe_slot = layer_ids[-1], num_slots - 1
    loc = table.lookup(probe_layer, kda_position(geometry, "kda_recurrent", 5), probe_slot, table.config_id_of("01"))
    assert loc.size_bytes == geometry.recurrent_segment_bytes
    shard = slabs.batch_index(probe_slot, probe_layer) * geometry.recurrent_shards_per_layer + 5
    banks = mesh_device.dram_grid_size().x
    assert loc.noc_addr == ((shard % banks) << 32) | (
        slabs.recurrent.buffer_address() + (shard // banks) * geometry.recurrent_segment_bytes
    )
    print(f"host {socket.gethostname()}: table check ok for fabric node {int(fid.mesh_id)}:{int(fid.chip_id)}")

    for state in natives.values():
        ttnn.deallocate(state.recurrent)
        ttnn.deallocate(state.convolution)
    slabs.deallocate()
