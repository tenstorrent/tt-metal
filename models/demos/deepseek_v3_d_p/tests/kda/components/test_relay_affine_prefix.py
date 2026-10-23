# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SP relay must leave every rank the entry state the gather + chain gives it, and the final state on the first
and last chronological ranks; with an output gate beside it, and call after call."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.config import block_projection_program_config
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import (
    AffineTransform,
    _RelayFinalCarry,
    _sp_links,
    distributed_prefix,
    pack_transform,
)
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import RelayAffinePrefix, RelayOutputGate
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = 640, 24, 128

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


class _Transforms:
    """One transform per rank, its packed device copy, and the FP64 reference of the relay around the ring."""

    def __init__(self, mesh_device, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.sp_size, self.tp_size = tuple(mesh_device.shape)
        # BF16-representable transforms make the BF16 transport exact for both paths.
        self.a = (0.1 * torch.randn(self.sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
        self.b = (0.1 * torch.randn(self.sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
        self.initial = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)

        def per_rank(host):
            return ttnn.from_torch(
                host.reshape((-1,) + host.shape[2:]),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=(0, None), mesh_shape=(self.sp_size, self.tp_size)
                ),
            )

        self.packed = pack_transform(AffineTransform(per_rank(self.a), per_rank(self.b)))
        self.initial_tt = ttnn.from_torch(
            self.initial,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def reference(self, start: int):
        """Each rank's entry state, the final carry and the ranks the relay leaves the carry on."""
        first_rank = (start // _ROWS) % self.sp_size
        carry = self.initial.double()
        entries = {}
        for step in range(self.sp_size):
            rank = (first_rank + step) % self.sp_size
            entries[rank] = carry.float()
            carry = self.a[rank].double() @ carry + self.b[rank].double()
        # The final state is valid where the relay leaves it: the last chronological rank and the first.
        return entries, carry.float(), {first_rank, (first_rank + self.sp_size - 1) % self.sp_size}

    def assert_matches_gather_and_chain(self, entry, final, actual_start, start: int):
        chained_entry, chained_final = distributed_prefix(
            self.packed,
            self.initial_tt,
            sequence_parallel_axis=0,
            compute_config=self.compute_config,
            actual_start=actual_start,
            local_rows=_ROWS,
        )
        entries, carry, final_ranks = self.reference(start)

        def assert_matches_chain(got, chained, expected, name):
            # The same products at the same fidelity in another operand and accumulation order (the relay accumulates
            # the K-sum on top of B in DST, the chain adds B after the matmul): the relay must be no further from the
            # FP64 reference than the gather + chain.
            expected = expected.double()
            relay_error = torch.linalg.vector_norm(got.double() - expected).item()
            chain_error = torch.linalg.vector_norm(chained.double() - expected).item()
            assert relay_error <= 1.01 * chain_error, f"{name}: relay error {relay_error} > chain error {chain_error}"

        devices = zip(
            ttnn.get_device_tensors(entry),
            ttnn.get_device_tensors(final),
            ttnn.get_device_tensors(chained_entry),
            ttnn.get_device_tensors(chained_final),
            strict=True,
        )
        for index, (local_entry, local_final, local_chained_entry, local_chained_final) in enumerate(devices):
            rank, tp = divmod(index, self.tp_size)
            name = f"start={start} rank={rank} tp={tp}"
            got_entry = ttnn.to_torch(local_entry)
            assert_accurate(entries[rank], got_entry, name=f"entry {name}", nrmse_threshold=0.025)
            assert_matches_chain(got_entry, ttnn.to_torch(local_chained_entry), entries[rank], f"entry {name}")
            if rank in final_ranks:
                got_final = ttnn.to_torch(local_final)
                assert_accurate(carry, got_final, name=f"final {name}", nrmse_threshold=0.03)
                assert_matches_chain(got_final, ttnn.to_torch(local_chained_final), carry, f"final {name}")


def _relay(mesh_device, cores=None) -> RelayAffinePrefix:
    return RelayAffinePrefix(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        key_dim=_KEY,
        value_dim=_KEY,
        num_links=_sp_links(mesh_device, 0),
        local_rows=_ROWS,
        cores=cores,
    )


@_MESH
# Only the rank holding the first token orders the relay; "split" checks that an offset within that rank is ignored.
@pytest.mark.parametrize(
    "start", [0, 640, 672, 3232, 4480], ids=["aligned", "rotated", "split", "late-split", "last-rank"]
)
def test_relay_affine_prefix_matches_gather_and_chain(mesh_device, start):
    transforms = _Transforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    entry, final, gate = _relay(mesh_device)(transforms.packed, transforms.initial_tt, actual_start)
    assert gate is None
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)


@_MESH_ONE_TILE_PACKETS
def test_relay_affine_prefix_one_tile_packets(mesh_device):
    start = 672
    transforms = _Transforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    entry, final, _ = _relay(mesh_device)(transforms.packed, transforms.initial_tt, actual_start)
    transforms.assert_matches_gather_and_chain(entry, final, actual_start, start)


@_MESH
def test_relay_affine_prefix_with_output_gate(mesh_device):
    """The output gate beside the relay, as the model lays it out: the relay on the right-hand worker columns, the
    gate's matmul on the block from (0, 0), its first columns copied in by the relay columns' spare cores."""
    torch.manual_seed(0)
    start = 640
    transforms = _Transforms(mesh_device, seed=start)
    actual_start = make_actual_start(mesh_device, start)
    grid = mesh_device.compute_with_storage_grid_size()
    relay_columns = [ttnn.CoreCoord(x, y) for x in range(grid.x - 1, grid.x - 4, -1) for y in range(grid.y)]
    relay = _relay(mesh_device, cores=relay_columns)
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
    output_gate = RelayOutputGate(
        input=x,
        weight=w,
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
    transforms = _Transforms(mesh_device, seed=starts[0])
    relay = _relay(mesh_device)
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
