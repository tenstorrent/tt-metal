# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks of the KDA state slabs: the per-chip layout, the read-back, and the address table.

Each chip's slabs hold only its SP row's ``H_store`` heads of its TP column. A native carry stored per chip
(``write_slab_layer``) and assembled back from every chip (``read_slab_layer``) must be bit-identical. (The layer's
ops read and write these slabs themselves; ``test_relay_chunk_scan.py`` and ``test_qkv_convolution_exchange.py`` cover
them, slot replay included.)

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
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_bit_identical, decode_tail
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    kda_device_params,
    kda_state_patterns,
    native_state_from_global,
    per_chip_convolution,
    read_slab_layer,
    write_slab_layer,
)
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

pytestmark = run_for_blackhole()

LAYER_IDS = (1, 2)
NUM_SLOTS = 2


def _device_params(shape: tuple[int, int]) -> dict:
    return kda_device_params(shape, trace_region_size=64 * 1024 * 1024)


def _geometry(mesh_device, sp_axis: int) -> KdaContractGeometry:
    return KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_device.shape), sp_axis=sp_axis, tp_axis=1 - sp_axis
    )


def _shards(tensor: ttnn.Tensor, mesh_device) -> dict[tuple[int, int], torch.Tensor]:
    rows, cols = tuple(mesh_device.shape)
    host = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    return {(row, col): host[row * cols + col] for row in range(rows) for col in range(cols)}


def _expected_chip(geometry, recurrent, convolution, tp_col, sp_row):
    """What chip (sp_row, tp_col) stores: its SP row's heads of its column, the convolution branch-major over them."""
    local, stored, dim = geometry.local_heads, geometry.stored_heads, geometry.head_dim
    first = tp_col * local + sp_row * stored
    column = per_chip_convolution(geometry, convolution, tp_col)
    branches = [
        column[:, (branch * local + sp_row * stored) * dim : (branch * local + (sp_row + 1) * stored) * dim]
        for branch in range(3)
    ]
    return recurrent[first : first + stored], torch.cat(branches, dim=-1)


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


# The meshes K3 runs on: SP on axis 0 of the 8x4 Galaxy and of a 2x4 LoudBox, SP on the axis-1 ring of a 4x4 stage (a
# Galaxy tray half, run_4x4_stage_tests.sh).
@pytest.mark.parametrize(
    "mesh_device,device_params,sp_axis",
    [
        pytest.param(
            shape,
            _device_params(shape),
            sp_axis,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=f"mesh-{shape[0]}x{shape[1]}"),
            id=case_id,
        )
        for shape, sp_axis, case_id in (((8, 4), 0, "8x4"), ((2, 4), 0, "2x4"), ((4, 4), 1, "4x4-sp1"))
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_ids,num_slots",
    [(LAYER_IDS, NUM_SLOTS), ((1,), 1)],
    ids=["2layers-2slots", "1layer-1slot"],
)
def test_slabs_round_trip_and_table(mesh_device, device_params, sp_axis, layer_ids, num_slots):
    """The one-layer, one-slot case matters: a slice over a whole slab can hand back the slab itself."""
    tp_axis = 1 - sp_axis
    geometry = _geometry(mesh_device, sp_axis)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=layer_ids, num_slots=num_slots)
    rows, cols = tuple(mesh_device.shape)

    def stored_by(row, col):
        """The chip at mesh (row, col) as (sp_row, tp_col)."""
        return {"sp_row": (row, col)[sp_axis], "tp_col": (row, col)[tp_axis]}

    patterns = {
        (slot, layer): kda_state_patterns(geometry, slot, layer) for slot in range(num_slots) for layer in layer_ids
    }
    natives = {key: native_state_from_global(mesh_device, geometry, *value, tp_axis) for key, value in patterns.items()}
    for (slot, layer), state in natives.items():
        write_slab_layer(mesh_device, slabs, state, slot, layer)

    # 1. Every slab batch holds its layer's state for the chip's own heads only, bit for bit.
    rec_shards = _shards(slabs.recurrent, mesh_device)
    conv_shards = _shards(slabs.convolution, mesh_device)
    for (slot, layer), (recurrent, convolution) in patterns.items():
        batch = slabs.batch_index(slot, layer)
        for row in range(rows):
            for col in range(cols):
                want_rec, want_conv = _expected_chip(geometry, recurrent, convolution, **stored_by(row, col))
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

    def imported(slot, layer):
        back = read_slab_layer(mesh_device, slabs, slot, layer)
        ttnn.synchronize_device(mesh_device)
        shards = host(back)
        deallocate_state(back)
        return shards

    # 2. The chips' stored heads assemble back into every chip's native carry.
    for (slot, layer), state in natives.items():
        for index, (want, got) in enumerate(zip(host(state), imported(slot, layer), strict=True)):
            assert_bit_identical(want, got, name=f"read back slot{slot} L{layer} shard{index}")

    # 4. The address table resolves every (layer, segment, slot) to the right bytes, and the segment
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
            sp_axis,
            tp_axis,
            geometry,
            kind,
            num_users=num_slots,
            config_id=table.config_id_of(name),
            stage_layout=_single_stage_layout(mesh_device, slab, len(layer_ids)),
            layer_rows=layer_rows,
        )
    assert table.num_device_groups() == rows * cols  # one chip per group
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

    # 5. A UMD read at the documented address matches the table's own read.
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
