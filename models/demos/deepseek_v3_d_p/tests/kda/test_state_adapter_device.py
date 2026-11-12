# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks of the KDA state slabs: per-layer export/import, trace replay, and the address table.

Each chip's slabs hold only its SP row's ``H_store`` heads of its TP column, so the round trip is
native -> export (the chip's heads, selected on device by its SP coordinate) -> import -> all-gather -> native, and
must be bit-identical.

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
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params, kda_mesh_param, read_slab_layer
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    assemble_convolution,
    assemble_recurrent,
    convolution_segment_to_torch,
    deallocate_state,
    device_scalar,
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
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    all_zero_bits,
    assert_bit_identical,
    decode_tail,
    encode_tail,
)

pytestmark = run_for_blackhole()

SP_AXIS, TP_AXIS = 0, 1
LAYER_IDS = (1, 2)
NUM_SLOTS = 2


def _device_params(shape: tuple[int, int]) -> dict:
    return kda_device_params(shape, trace_region_size=64 * 1024 * 1024)


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


def _expected_chip(geometry, recurrent, convolution, tp_col, sp_row):
    """What chip (sp_row, tp_col) stores: its SP row's heads of its column, the convolution branch-major over them."""
    local, stored, dim = geometry.local_heads, geometry.stored_heads, geometry.head_dim
    first = tp_col * local + sp_row * stored
    column = _per_chip_convolution(geometry, convolution, tp_col)
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


# The import all-gathers over SP, so the mesh opens the fabric K3 runs on.
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        kda_mesh_param(shape, f"mesh-{shape[0]}x{shape[1]}", f"{shape[0]}x{shape[1]}", _device_params(shape))
        for shape in ((8, 4), (2, 4))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "layer_ids,num_slots",
    [(LAYER_IDS, NUM_SLOTS), ((1,), 1)],
    ids=["2layers-2slots", "1layer-1slot"],
)
def test_slabs_round_trip_replay_and_table(mesh_device, device_params, layer_ids, num_slots):
    """The one-layer, one-slot case matters: a slice over a whole slab can hand back the slab itself."""
    geometry = _geometry(mesh_device)
    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=SP_AXIS, layer_ids=layer_ids, num_slots=num_slots)
    rows, cols = tuple(mesh_device.shape)
    tp = geometry.tensor_parallel_size

    patterns = {
        (slot, layer): _global_patterns(geometry, slot, layer) for slot in range(num_slots) for layer in layer_ids
    }
    natives = {key: _native_from_global(mesh_device, geometry, *value) for key, value in patterns.items()}
    for (slot, layer), state in natives.items():
        index = device_scalar(mesh_device, slot)
        slabs.export_layer(state, index, layer)
        ttnn.deallocate(index)
    ttnn.synchronize_device(mesh_device)

    # 1. Every slab batch holds its layer's state for the chip's own heads only, bit for bit.
    rec_shards = _shards(slabs.recurrent, mesh_device)
    conv_shards = _shards(slabs.convolution, mesh_device)
    for (slot, layer), (recurrent, convolution) in patterns.items():
        batch = slabs.batch_index(slot, layer)
        for row in range(rows):
            for col in range(cols):
                want_rec, want_conv = _expected_chip(geometry, recurrent, convolution, tp_col=col, sp_row=row)
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

    # 2. Import gathers the same bytes back into a native carry on every chip, through the same device-read slot,
    #    and a repeated import (the staging buffer reused) lands them again.
    for repeat in range(2):
        for (slot, layer), state in natives.items():
            for index, (want, got) in enumerate(zip(host(state), imported(slot, layer), strict=True)):
                assert_bit_identical(want, got, name=f"import {repeat} slot{slot} L{layer} shard{index}")

    # 2b. A captured import follows the device slot on replay: captured on slot 0, replayed on the last slot.
    carry = read_slab_layer(mesh_device, slabs, 0, layer_ids[-1])
    slot_tt = device_scalar(mesh_device, 0)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    slabs.import_layer(carry, slot_tt, layer_ids[-1])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    staged_slot = device_scalar(mesh_device, num_slots - 1)
    ttnn.copy(staged_slot, slot_tt)
    for replay in range(2):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        want_state = natives[(num_slots - 1, layer_ids[-1])]
        for index, (want, got) in enumerate(zip(host(want_state), host(carry), strict=True)):
            assert_bit_identical(want, got, name=f"traced import replay {replay} shard{index}")
    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (slot_tt, staged_slot):
        ttnn.deallocate(tensor)
    deallocate_state(carry)

    # 3. A captured export follows the carry's contents and the device slot on replay, and touches
    #    nothing else: captured on slot 0, replayed with the last slot written into the slot tensor.
    victim = natives[(0, layer_ids[0])]
    slot_tt = device_scalar(mesh_device, 0)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    slabs.export_layer(victim, slot_tt, layer_ids[0])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    new_rec, new_conv = _global_patterns(geometry, 5, 9)
    staged = _native_from_global(mesh_device, geometry, new_rec, new_conv)
    ttnn.copy(staged.recurrent, victim.recurrent)
    ttnn.copy(staged.convolution, victim.convolution)
    replay_slot = num_slots - 1
    staged_slot = device_scalar(mesh_device, replay_slot)
    ttnn.copy(staged_slot, slot_tt)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)
    for tensor in (slot_tt, staged_slot):
        ttnn.deallocate(tensor)
    patterns[(replay_slot, layer_ids[0])] = (new_rec, new_conv)
    rec_shards = _shards(slabs.recurrent, mesh_device)
    conv_shards = _shards(slabs.convolution, mesh_device)
    for (slot, layer), (recurrent, convolution) in patterns.items():
        batch = slabs.batch_index(slot, layer)
        for row in range(rows):
            for col in range(cols):
                want_rec, want_conv = _expected_chip(geometry, recurrent, convolution, tp_col=col, sp_row=row)
                assert_bit_identical(
                    want_rec, rec_shards[(row, col)][batch], name=f"replay S slot{slot} L{layer} ({row},{col})"
                )
                assert_bit_identical(
                    want_conv,
                    decode_tail(conv_shards[(row, col)][batch], KDA_CONVOLUTION_PAGE_CHANNELS),
                    name=f"replay conv slot{slot} L{layer} ({row},{col})",
                )

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
            SP_AXIS,
            TP_AXIS,
            geometry,
            kind,
            num_users=num_slots,
            config_id=table.config_id_of(name),
            stage_layout=_single_stage_layout(mesh_device, slab, len(layer_ids)),
            layer_rows=layer_rows,
        )
    assert table.num_device_groups() == rows * tp  # one chip per group
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

    # 6. A request-head import (zero_fill_at_zero reading 0, as a chunk start of 0 does) loads zeros into both
    #    carries of every layer, over a carry that held a pattern, and reads the slot without writing any slab byte.
    slabs_before = [*_shards(slabs.recurrent, mesh_device).values(), *_shards(slabs.convolution, mesh_device).values()]
    slot_tt, head_tt = device_scalar(mesh_device, 0), device_scalar(mesh_device, 0)
    for layer in layer_ids:
        carry = natives[(0, layer)]
        slabs.import_layer(carry, slot_tt, layer, zero_fill_at_zero=head_tt)
        for index, got in enumerate(host(carry)):
            assert all_zero_bits(got), f"zero_fill_at_zero import left bytes in L{layer} dev{index}"
    slabs_after = [*_shards(slabs.recurrent, mesh_device).values(), *_shards(slabs.convolution, mesh_device).values()]
    for index, (want, got) in enumerate(zip(slabs_before, slabs_after, strict=True)):
        assert_bit_identical(want, got, name=f"zero_fill_at_zero import wrote slab shard {index}")
    for tensor in (slot_tt, head_tt):
        ttnn.deallocate(tensor)

    for state in natives.values():
        ttnn.deallocate(state.recurrent)
        ttnn.deallocate(state.convolution)
    ttnn.deallocate(staged.recurrent)
    ttnn.deallocate(staged.convolution)
    slabs.deallocate()
