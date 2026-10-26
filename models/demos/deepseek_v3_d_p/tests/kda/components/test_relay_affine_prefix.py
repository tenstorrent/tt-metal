# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SP relay must leave every rank the entry state the gather + chain gives it, and the final state on the first
and last chronological ranks; with an output gate beside it, and call after call."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    RELAY_HEADS,
    RELAY_KEY,
    RELAY_ROWS,
    RelayTransforms,
    make_relay,
    relay_worker_columns,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import block_projection_program_config, dram_width_sharded
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _RelayFinalCarry
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import RelayAffinePrefix, RelayOutputGate
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = RELAY_ROWS, RELAY_HEADS, RELAY_KEY

_MESH = pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)

# The prefill runner's fabric: a 7168-byte payload carries one FP32 tile per packet, so a tile column needs four packet
# headers and the sender buffers fewer columns.
_MESH_ONE_TILE_PACKETS = pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=7168),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-one-tile-packets",
        )
    ],
    indirect=True,
)


def landed_final_states(relay: RelayAffinePrefix, landing_device_tensor) -> torch.Tensor:
    """The [BH, K, K] states in one device's final landing, head h's taken from the relay's h-th head core (the
    landing stores each state by tile column)."""
    landing = ttnn.to_torch(landing_device_tensor)
    grid = relay.final_landing.memory_config().shard_spec.grid
    shard_of = {(c.x, c.y): i for i, c in enumerate(ttnn.corerange_to_cores(grid, row_wise=True))}
    kt = _KEY // 32
    states = []
    for core in relay.head_cores:
        shard = shard_of[(core.x, core.y)]
        stored = landing[shard * _KEY : (shard + 1) * _KEY].reshape(kt, 32, kt, 32)
        # The stored tile (i, j) is the state's tile (j, i).
        states.append(stored.permute(2, 1, 0, 3).reshape(_KEY, _KEY))
    return torch.stack(states)


@_MESH
# Only the rank holding the first token orders the relay; "split" checks that an offset within that rank is ignored.
@pytest.mark.parametrize(
    "start", [0, 640, 672, 3232, 4480], ids=["aligned", "rotated", "split", "late-split", "last-rank"]
)
def test_relay_affine_prefix_matches_gather_and_chain(mesh_device, start):
    transforms = RelayTransforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    entry, final, gate = make_relay(mesh_device)(transforms.packed, transforms.initial_tt, actual_start)
    assert gate is None
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)


@_MESH_ONE_TILE_PACKETS
def test_relay_affine_prefix_one_tile_packets(mesh_device):
    start = 672
    transforms = RelayTransforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    entry, final, _ = make_relay(mesh_device)(transforms.packed, transforms.initial_tt, actual_start)
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)


@_MESH
@pytest.mark.parametrize("weight_layout", ["interleaved", "dram_width_sharded"])
def test_relay_affine_prefix_with_output_gate(mesh_device, weight_layout):
    """The output gate beside the relay, as the model lays it out: the relay on the right-hand worker columns, the
    gate's matmul on the block from (0, 0), its first columns copied in by the relay columns' spare cores. The gate's
    weight is interleaved or, as the model keeps it, DRAM width-sharded over the banks; either way the gate equals
    ttnn.linear with the interleaved weight bit for bit."""
    torch.manual_seed(0)
    start = 640
    transforms = RelayTransforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    grid = mesh_device.compute_with_storage_grid_size()
    relay_columns = relay_worker_columns(mesh_device)
    relay = make_relay(mesh_device, cores=relay_columns)
    copy_cores = [core for core in relay_columns if core not in relay.cores]
    replicate = lambda host: ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x, w = replicate(torch.randn(1, _ROWS, 256)), replicate(torch.randn(256, 512) * 0.05)
    copy_source, copy_source_column, copied_columns = replicate(torch.randn(1, _ROWS, 256)), 3, 2
    # The gate's worker columns: those left of the relay's three, within the model's 110 matmul cores shared with the
    # relay's head cores.
    columns = min(grid.x - 3, (110 - _HEADS) // grid.y)
    config = block_projection_program_config(columns, grid.y, _ROWS // 32, 512 // 32, 4)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
    )
    gate_weight = dram_width_sharded(mesh_device, ttnn.clone(w)) if weight_layout == "dram_width_sharded" else w
    output_gate = RelayOutputGate(
        input=x,
        weight=gate_weight,
        program_config=config,
        compute_config=compute_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        copy_source=copy_source,
        copy_source_column=copy_source_column,
        copied_columns=copied_columns,
        copy_cores=copy_cores,
    )
    entry, final, gate = relay(transforms.packed, transforms.initial_tt, actual_start, output_gate=output_gate)
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    expected_projection = ttnn.to_torch(
        ttnn.linear(x, w, program_config=config, compute_kernel_config=compute_config), mesh_composer=composer
    )
    copied = ttnn.to_torch(copy_source, mesh_composer=composer)[
        ..., copy_source_column * 32 : (copy_source_column + copied_columns) * 32
    ]
    got = ttnn.to_torch(gate, mesh_composer=composer)
    assert got.shape[-1] == copied_columns * 32 + 512
    assert torch.equal(got[..., : copied_columns * 32], copied), "the copied columns differ from the copy source"
    assert torch.equal(got[..., copied_columns * 32 :], expected_projection), "the gate differs from ttnn.linear"


@_MESH
def test_relay_affine_prefix_back_to_back(mesh_device):
    """Two relay calls separated by the SP-line barrier the model runs between layers (the relay's final carry): the
    second call must be as exact as the first, from the landing buffers and arrival counters the first left."""
    # Rank-aligned starts, the second rotated: without a split tail the carry is the prefix's final state.
    starts = (0, 640)
    transforms = RelayTransforms(mesh_device, seed=starts[0])
    relay = make_relay(mesh_device)
    final_carry = _RelayFinalCarry.shared(mesh_device, 0, relay.num_links)
    results = []
    for start in starts:
        actual_start = make_actual_start(mesh_device, start)
        entry, final, _ = relay(transforms.packed, transforms.initial_tt, actual_start)
        # Every rank enters the carry only after its relay call: a line barrier before the next call. Without a split
        # tail the carry takes the prefix's final state; rank_final only has to be a distinct tensor.
        rank_final = ttnn.clone(final)
        carry, _ = final_carry(rank_final, final, actual_start, None, _ROWS)
        ttnn.deallocate(rank_final)
        results.append((start, actual_start, entry, final, carry))
    for start, actual_start, entry, final, carry in results:
        transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)
        # The carry leaves every rank the final state of the chronologically last rank.
        _, expected_carry, _ = transforms.reference(start)
        for index, local_carry in enumerate(ttnn.get_device_tensors(carry)):
            rank, tp = divmod(index, transforms.tp_size)
            assert_accurate(
                expected_carry,
                ttnn.to_torch(local_carry),
                name=f"carry start={start} rank={rank} tp={tp}",
                nrmse_threshold=0.03,
            )


@_MESH
@pytest.mark.parametrize("start", [640, 672], ids=["rotated", "split"])
def test_relay_affine_prefix_leaves_final_in_landing(mesh_device, start):
    """With leave_final_in_landing the first rank does not take the last rank's final state: it stays in the first
    rank's final landing, head h's on the relay's h-th head core, and the final-state output is valid on the last
    rank only. The head cores go column by column, so their order differs from the landing's shard order."""
    transforms = RelayTransforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    relay = make_relay(mesh_device, cores=relay_worker_columns(mesh_device))
    entry, final, _ = relay(transforms.packed, transforms.initial_tt, actual_start, leave_final_in_landing=True)
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start, final_in_landing=True)
    _, carry, _ = transforms.reference(start)
    first_rank = (start // _ROWS) % transforms.sp_size
    landings = ttnn.get_device_tensors(relay.final_landing)
    for tp in range(transforms.tp_size):
        assert_accurate(
            carry,
            landed_final_states(relay, landings[first_rank * transforms.tp_size + tp]),
            name=f"landed final start={start} tp={tp}",
            nrmse_threshold=0.03,
        )
