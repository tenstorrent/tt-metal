# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-tensor logical lengths for the ring / exp ring recipes.

logical_n (and, for ring with a sharded joint, logical_l) given as single-element device tensors must give
the same bits as the host scalars over every valid output row. The tensor path compiles once (the scalar
attributes become worst-case placeholders) and reads the live lengths on device, so one captured trace is
replayed with only the length tensors refreshed in place, in an order that makes a stale read visible.
Garbage keys/values past the logical lengths make any leaked padding visible. Mirrors
models/tt_dit/tests/unit/test_ring_joint_attention.py::test_ring_joint_sdpa_logical_tensor_trace_replay.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import VARIANTS, digest, prepare
from .test_sdpa_recipe_ring_geometry import RING, chips, close_subdevice_mesh, open_subdevice_mesh, precision_of, randn


def length_tensor(mesh, value, *, on_device=True):
    """Single-valued uint32 length, replicated; on_device=False gives the host twin for an in-place refresh."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        **({"device": mesh} if on_device else {}),
    )


def refresh(mesh, tensor, value):
    ttnn.copy_host_to_device_tensor(length_tensor(mesh, value, on_device=False), tensor)


def check_lengths(mesh, call, pairs, valid, record_property):
    """call(n, l) runs the op; valid(outputs, n, l) returns the valid output rows per chip. Scalar results
    first, then the tensor path eagerly per pair, then one trace replayed in a non-monotonic order."""
    references = []
    for n, l in pairs:
        outputs = call(n, l)
        ttnn.synchronize_device(mesh)
        references.append(valid(outputs, n, l))
    for i in range(1, len(pairs)):
        # A stale length read must be visible: each pair's spatial rows differ from the first pair's.
        shared = min(pairs[i][0], pairs[0][0])
        assert not torch.equal(references[i][0][..., :shared, :], references[0][0][..., :shared, :]), pairs[i]

    n_tensor = length_tensor(mesh, pairs[0][0])
    l_tensor = length_tensor(mesh, pairs[0][1]) if pairs[0][1] is not None else None

    def load(n, l):
        refresh(mesh, n_tensor, n)
        if l_tensor is not None:
            refresh(mesh, l_tensor, l)

    for i, (n, l) in enumerate(pairs):
        load(n, l)
        outputs = call(n_tensor, l_tensor if l_tensor is not None else l)
        ttnn.synchronize_device(mesh)
        got = valid(outputs, n, l)
        assert [digest(x) for x in got] == [digest(x) for x in references[i]], f"eager tensor pair {pairs[i]}"
    record_property("eager_tensor_equal", True)

    load(*pairs[0])
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    traced = call(n_tensor, l_tensor if l_tensor is not None else pairs[0][1])
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.synchronize_device(mesh)
    try:
        order = list(range(len(pairs))) + list(reversed(range(len(pairs)))) + [0, len(pairs) - 1, 1]
        for step, i in enumerate(order):
            load(*pairs[i])
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            got = valid(traced, *pairs[i])
            assert [digest(x) for x in got] == [
                digest(x) for x in references[i]
            ], f"replay {step} of {order}: pair {pairs[i]} differs from the host-scalar path"
    finally:
        ttnn.release_trace(mesh, trace)
    record_property("trace_replays_equal", len(order))


RING_GEOMETRIES = [(256, 512, 128), (96, 160, 96)]
GEOMETRY_ID = lambda g: f"q{g[0]}k{g[1]}d{g[2]}"


class TestRingLengths:
    @pytest.fixture(scope="class")
    def ring_mesh(self):
        if not is_blackhole() or ttnn.GetNumAvailableDevices() != RING:
            pytest.skip("This ring suite requires exactly two connected Blackholes")
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )
        mesh = manager = None
        try:
            mesh, subdevice, cores, manager, hardware = open_subdevice_mesh(dict(trace_region_size=33554432))
            semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
            yield mesh, subdevice, semaphores, hardware.x - 1
        finally:
            if mesh is not None:
                close_subdevice_mesh(mesh, manager)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("q_chunk,k_chunk,head_dim", RING_GEOMETRIES, ids=[GEOMETRY_ID(g) for g in RING_GEOMETRIES])
    @pytest.mark.parametrize("joint", ["sharded", "replicated"])
    def test_ring_logical_tensors(self, ring_mesh, variant, q_chunk, k_chunk, head_dim, joint, record_property):
        mesh, subdevice, semaphores, ccl_column = ring_mesh
        local, joint_local = 1024, 256
        # logical_n: full, a partial second shard, an emptied second shard (inactive ring step), a sub-tile
        # tail. A sharded joint also moves logical_l (kept above one joint shard so the scalar path shards too).
        n_values = [2 * local, 1500, 777, 2 * local - 49]
        l_values = [2 * joint_local, 300, 257, 450] if joint == "sharded" else [None] * 4
        pairs = list(zip(n_values, l_values))
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)
        joint_mapper = shard if joint == "sharded" else ttnn.ReplicateTensorToMesh(mesh)
        joint_rows = RING * joint_local if joint == "sharded" else joint_local
        host = [randn((1, 1, RING * local, head_dim), 20260926 + i, scale=8.0) for i in range(3)]
        joint_host = [randn((1, 1, joint_rows, head_dim), 20261026 + i, scale=8.0) for i in range(3)]

        def upload(values, mapper):
            return prepare(
                [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values], variant
            )

        inputs = upload(host, shard)
        joints = upload(joint_host, joint_mapper)

        def allocate(rows, like):
            return ttnn.allocate_tensor_on_device(
                [1, 1, rows, head_dim], like.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )

        backing = [allocate(RING * local, x) for x in inputs[1:]]
        joint_backing = (
            dict(
                persistent_output_buffer_joint_k=allocate(RING * joint_local, joints[1]),
                persistent_output_buffer_joint_v=allocate(RING * joint_local, joints[2]),
            )
            if joint == "sharded"
            else {}
        )
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(2, 1), q_chunk_size=q_chunk, k_chunk_size=k_chunk
        )

        def call(logical_n, logical_l):
            return ttnn.transformer.ring_joint_scaled_dot_product_attention(
                *inputs,
                *joints,
                persistent_output_buffer_k=backing[0],
                persistent_output_buffer_v=backing[1],
                **joint_backing,
                joint_strategy="rear",
                logical_n=logical_n,
                logical_l=logical_l if joint == "sharded" else joint_local,
                is_causal=False,
                program_config=program_config,
                precision=precision_of(variant),
                inputs_prepared=variant.startswith("E_"),
                dim=2,
                multi_device_global_semaphore=semaphores,
                num_links=1,
                cluster_axis=1,
                mesh_device=mesh,
                topology=ttnn.Topology.Linear,
                subdevice_id=subdevice,
                ccl_core_grid_offset=(ccl_column, 0),
                use_column_major_ccl=True,
            )

        def valid(outputs, n, l):
            # Rows past the logical lengths are padding queries; only valid rows are part of the contract.
            spatial = torch.cat(chips(outputs[0]), dim=2)[..., :n, :]
            joint_out = chips(outputs[1])
            if joint == "sharded":
                return [spatial, torch.cat(joint_out, dim=2)[..., :l, :]]
            return [spatial, *joint_out]

        check_lengths(mesh, call, pairs, valid, record_property)


# (q_chunk, k_chunk, head_dim, local rows, heads, grid, joint rows): the exp ring work split needs the Q
# chunks per head to be a multiple of the SDPA columns (grid.x - 1) and 1-3 head-segments per row.
EXP_CASES = {
    "q256k512d128": (256, 512, 128, 1024, 4, (5, 4), 0),
    "q256k512d128-joint": (256, 512, 128, 1024, 4, (4, 4), 512),
    "q96k160d96": (96, 160, 96, 960, 4, (6, 4), 0),
}


class TestExpRingLengths:
    @pytest.fixture(scope="class")
    def exp_mesh(self):
        if not is_blackhole() or ttnn.GetNumAvailableDevices() != RING:
            pytest.skip("This exp ring suite requires exactly two connected Blackholes")
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = 8192
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            router,
        )
        mesh = manager = None
        try:
            mesh, subdevice, cores, manager, _ = open_subdevice_mesh(
                dict(worker_l1_size=int(os.getenv("TEST_EXP_RING_WORKER_L1", "1344544")), trace_region_size=33554432)
            )
            semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
            yield mesh, subdevice, semaphores
        finally:
            if mesh is not None:
                close_subdevice_mesh(mesh, manager)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("case", list(EXP_CASES))
    def test_exp_ring_logical_tensor(self, exp_mesh, variant, case, record_property):
        mesh, subdevice, semaphores = exp_mesh
        q_chunk, k_chunk, head_dim, local, heads, grid, joint_rows = EXP_CASES[case]
        n_values = [RING * local, 1500, 777, RING * local - 49]
        pairs = [(n, None) for n in n_values]
        host = [randn((1, heads, RING * local, head_dim), 20260927 + i, scale=8.0) for i in range(3)]

        def upload(values, mapper):
            return prepare(
                [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values], variant
            )

        inputs = upload(host, ttnn.ShardTensorToMesh(mesh, dim=2))
        joints = [None] * 3
        if joint_rows:
            joint_host = [randn((1, heads, joint_rows, head_dim), 20261027 + i) for i in range(3)]
            joints = upload(joint_host, ttnn.ReplicateTensorToMesh(mesh))
        backing = [
            ttnn.allocate_tensor_on_device(
                [1, heads, RING * local, head_dim], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for x in inputs[1:]
        ]

        def call(logical_n, _):
            return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                *inputs,
                *joints,
                persistent_output_buffer_k=backing[0],
                persistent_output_buffer_v=backing[1],
                joint_strategy="rear",
                logical_n=logical_n,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
                ),
                dim=2,
                multi_device_global_semaphore=semaphores,
                num_links=2,
                cluster_axis=1,
                mesh_device=mesh,
                topology=ttnn.Topology.Ring,
                subdevice_id=subdevice,
                num_workers_per_link=grid[1] // 2,
                num_buffers_per_channel=16,
                precision=precision_of(variant),
                inputs_prepared=variant.startswith("E_"),
            )

        def valid(outputs, n, _):
            spatial = torch.cat(chips(outputs[0]), dim=2)[..., :n, :]
            return [spatial, *(chips(outputs[1]) if joint_rows else [])]

        check_lengths(mesh, call, pairs, valid, record_property)
