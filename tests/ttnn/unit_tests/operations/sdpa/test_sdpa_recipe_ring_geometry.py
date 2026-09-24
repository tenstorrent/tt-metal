# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic ring / exp ring recipe geometry: any tile-aligned Q chunk, K chunk and head dim that fits L1.

The numerical implementation of each recipe is the contract; geometry only changes blocking. Ring and
dense recipes share numerics, so for B-E each chip's ring output must equal the dense recipe (same
Q/K chunking) on that chip's KV concatenated in visiting order whenever every visited segment except the
last is a whole number of K chunks (the ring restarts K chunking at each shard); other chips are gated on
FP64 L2 against that dense result. Every case also checks determinism (program-cache rerun). A (FAST) keeps
the legacy ring / exp ring compute: inside its qualified geometries it is gated on FP64 L2 against the dense
FAST recipe (ring) or the legacy default call (exp ring); outside them the op must reject it before dispatch.
Two connected Blackholes; each class opens its own fabric.
"""

import math
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, metrics, prepare, reference

RING = 2

# (q_chunk, k_chunk, head_dim): Q, K and D swept one at a time around the qualified Q256/K512/D128,
# then combined odd geometries. The qualified geometry anchors the sweep.
GEOMETRIES = [
    (256, 512, 128),
    *[(q, 512, 128) for q in (32, 64, 96, 160, 384, 512, 1024)],
    *[(256, k, 128) for k in (32, 64, 96, 128, 160, 640, 1024)],
    *[(256, 256, d) for d in (32, 96, 160, 192)],
    (32, 32, 32),
    (96, 160, 96),
    (96, 96, 96),
    (160, 160, 160),
    (64, 1024, 64),
    (512, 128, 96),
    (1024, 128, 64),
]
JOINT_GEOMETRIES = [(256, 512, 128), (32, 32, 32), (96, 160, 96), (160, 96, 160)]
GEOMETRY_ID = lambda g: f"q{g[0]}k{g[1]}d{g[2]}"
SELECTED = os.getenv("SDPA_RING_GEOMETRY")  # e.g. "q96k160d96": run one geometry per process


def precision_of(variant):
    return getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION"))


def randn(shape, seed, scale=1.0):
    return (scale * torch.randn(shape, generator=torch.Generator().manual_seed(seed))).bfloat16()


def l1_rejection(error):
    return "L1" in str(error)


def open_subdevice_mesh(options):
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, RING), **options)
    mesh.enable_program_cache()
    hardware = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hardware.x - 1, hardware.y - 1))})
    subdevice = ttnn.SubDeviceId(0)
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
    mesh.load_sub_device_manager(manager)
    mesh.set_sub_device_stall_group([subdevice])
    return mesh, subdevice, cores, manager, hardware


def close_subdevice_mesh(mesh, manager):
    if manager is not None:
        mesh.reset_sub_device_stall_group()
        mesh.clear_loaded_sub_device_manager()
        mesh.remove_sub_device_manager(manager)
    ttnn.close_mesh_device(mesh)


def chips(tensor):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tensor)]


def dense_recipe(mesh, queries, keys, values, variant, q_chunk, k_chunk, grid):
    """Dense recipe per chip (chips on the mesh batch axis) with the ring's Q/K blocking."""
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
    inputs = prepare(
        [
            ttnn.from_torch(torch.cat(x, dim=0), device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for x in (queries, keys, values)
        ],
        variant,
    )
    output = ttnn.transformer.scaled_dot_product_attention(
        *inputs,
        is_causal=False,
        precision=precision_of(variant),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        ),
    )
    return chips(output)


def maybe_select(geometry):
    if SELECTED and GEOMETRY_ID(geometry) != SELECTED:
        pytest.skip(f"SDPA_RING_GEOMETRY={SELECTED}")


def fast_geometry(exp, q_chunk, k_chunk, head_dim):
    """FAST keeps the legacy ring / exp ring kernels and stays limited to their qualified geometries."""
    ring = head_dim in (64, 128, 256) and 128 <= q_chunk <= 320 and k_chunk in (256, 384, 512)
    return ring and (not exp or (k_chunk == 512 and head_dim == 128))


def expect_fast_rejection(invoke, exp, q_chunk, k_chunk, head_dim):
    """Outside the FAST set the op must reject before dispatch; returns True when it did."""
    if fast_geometry(exp, q_chunk, k_chunk, head_dim):
        return False
    with pytest.raises(RuntimeError, match="do not support"):
        invoke()
    return True


def gate(record_property, variant, chip, got, dense, expected, bitwise):
    """Ring output vs the dense recipe at the same blocking (bitwise when the chunking coincides)."""
    observed = metrics(got, expected)
    baseline = metrics(dense, expected)
    equal = digest(got) == digest(dense)
    for key, value in observed.items():
        record_property(f"chip{chip}_{key}", value)
    record_property(f"chip{chip}_dense_l2_pct", baseline["l2_pct"])
    record_property(f"chip{chip}_ordered_dense_equal", equal)
    record_property(f"chip{chip}_chunk_aligned", bitwise)
    assert observed["l2_pct"] is not None
    if variant in ["C", "D"]:
        assert observed["l2_pct"] < {"C": 2, "D": 0.4}[variant], observed
    if variant == "A":
        assert observed["l2_pct"] <= baseline["l2_pct"] * 1.05 + 1e-6, (observed, baseline)
    elif bitwise:
        assert equal, f"chip {chip}: ring recipe must equal the dense recipe in visiting order"
    else:
        # Different K chunk boundaries than dense: the same recipe error level, not the same bits.
        assert observed["l2_pct"] <= baseline["l2_pct"] * 1.25 + 1e-6, (observed, baseline)


class TestRingGeometry:
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
            mesh, subdevice, cores, manager, hardware = open_subdevice_mesh(dict(trace_region_size=16777216))
            semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
            yield mesh, subdevice, semaphores, hardware.x - 1
        finally:
            if mesh is not None:
                close_subdevice_mesh(mesh, manager)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @staticmethod
    def run_case(ring_mesh, variant, q_chunk, k_chunk, head_dim, joint, record_property):
        mesh, subdevice, semaphores, ccl_column = ring_mesh
        # Three Q chunks per chip (the last one tile tall) on two workers: one worker checkpoints two Q blocks
        # through DRAM between ring steps, the other keeps its single state resident. K shards are whole K
        # chunks of at least 1024 rows. A sharded joint needs Q and K shards of one length (no is_cross), at
        # least 512 rows, with joint shards of at least 256 rows.
        if joint:
            k_local = q_local = k_chunk * math.ceil(512 / k_chunk)
        else:
            q_local = 2 * q_chunk + 32
            k_local = k_chunk * math.ceil(1024 / k_chunk)
        joint_local = k_chunk * math.ceil(256 / k_chunk) if joint else 0
        # Sub-tile tails in the second primary shard (no joint) or in the second joint shard.
        valid_n = 2 * k_local if joint else 2 * k_local - 33
        valid_l = 2 * joint_local - 33 if joint else 0
        host = [randn((1, 1, 2 * n, head_dim), 20260924 + i) for i, n in enumerate((q_local, k_local, k_local))]
        for x in host[1:]:
            x[..., valid_n:, :] = randn(x[..., valid_n:, :].shape, 7, scale=8.0)
        joint_host = None
        if joint:
            joint_host = [randn((1, 1, 2 * joint_local, head_dim), 20261024 + i) for i in range(3)]
            for x in joint_host[1:]:
                x[..., valid_l:, :] = randn(x[..., valid_l:, :].shape, 9, scale=8.0)
        shard = ttnn.ShardTensorToMesh(mesh, dim=2)

        def upload(values):
            return prepare(
                [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard) for x in values], variant
            )

        inputs = upload(host)
        joints = upload(joint_host) if joint else [None] * 3
        backing = [
            ttnn.allocate_tensor_on_device(
                [1, 1, 2 * k_local, head_dim], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for x in inputs[1:]
        ]
        joint_backing = {}
        if joint:
            joint_backing = dict(
                zip(
                    ("persistent_output_buffer_joint_k", "persistent_output_buffer_joint_v"),
                    [
                        ttnn.allocate_tensor_on_device(
                            [1, 1, 2 * joint_local, head_dim], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
                        )
                        for x in joints[1:]
                    ],
                )
            )

        def invoke():
            return ttnn.transformer.ring_joint_scaled_dot_product_attention(
                *inputs,
                *joints,
                persistent_output_buffer_k=backing[0],
                persistent_output_buffer_v=backing[1],
                **joint_backing,
                joint_strategy="rear",
                logical_n=valid_n,
                logical_l=valid_l,
                is_causal=False,
                is_cross=q_local != k_local,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(2, 1), q_chunk_size=q_chunk, k_chunk_size=k_chunk
                ),
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

        if variant == "A" and expect_fast_rejection(invoke, False, q_chunk, k_chunk, head_dim):
            record_property("fast_rejected", True)
            return
        try:
            outputs = invoke()
        except RuntimeError as error:
            if not l1_rejection(error):
                raise
            record_property("rejected_l1", True)
            pytest.skip(f"{variant} ring Q{q_chunk}/K{k_chunk}/D{head_dim} exceeds Blackhole L1")
        actual = chips(outputs[0])
        joint_actual = chips(outputs[1]) if joint else []
        again = invoke()
        assert [digest(x) for x in chips(again[0])] == [digest(x) for x in actual], "ring recipe is not deterministic"
        if joint:
            assert [digest(x) for x in chips(again[1])] == [digest(x) for x in joint_actual]

        # Prepared values as the device holds them, so the FP64 reference and dense see the ring's inputs.
        prepared = [ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)) for x in inputs]
        prepared_joint = (
            [ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)) for x in joints] if joint else None
        )
        kv_true = [x[..., :valid_n, :] for x in prepared[1:]]
        if joint:
            kv_true = [torch.cat([x, j[..., :valid_l, :]], dim=2) for x, j in zip(kv_true, prepared_joint[1:])]
        # Dense inputs come from the original host values and are prepared on upload exactly like the ring's.
        query_segments, ordered, aligned = [[], []], [[], []], []
        for chip in range(RING):
            query_segments[0].append(host[0][..., chip * q_local : (chip + 1) * q_local, :])
            if joint:
                query_segments[1].append(joint_host[0][..., chip * joint_local : (chip + 1) * joint_local, :])
            segment_rows = []
            for index in (1, 2):
                pieces = []
                for rank in (chip, 1 - chip):
                    start, end = rank * k_local, min((rank + 1) * k_local, valid_n)
                    if end > start:
                        pieces.append(host[index][..., start:end, :])
                        segment_rows.append(end - start) if index == 1 else None
                    if joint:
                        start, end = rank * joint_local, min((rank + 1) * joint_local, valid_l)
                        if end > start:
                            pieces.append(joint_host[index][..., start:end, :])
                            segment_rows.append(end - start) if index == 1 else None
                ordered[index - 1].append(torch.cat(pieces, dim=2))
            aligned.append(all(rows % k_chunk == 0 for rows in segment_rows[:-1]))
        try:
            dense_segments = [
                dense_recipe(mesh, queries, *ordered, variant, q_chunk, k_chunk, (1, 1))
                for queries in query_segments
                if queries
            ]
        except RuntimeError as error:
            if not l1_rejection(error):
                raise
            # The ring's single-slot Q fallback can fit where the double-buffered dense Q does not.
            record_property("dense_reference_rejected_l1", True)
            pytest.skip(f"dense {variant} Q{q_chunk}/K{k_chunk}/D{head_dim} reference exceeds Blackhole L1")
        for chip in range(RING):
            got = torch.cat([actual[chip], joint_actual[chip]], dim=2) if joint else actual[chip]
            dense = torch.cat([segment[chip] for segment in dense_segments], dim=2)
            queries = [prepared[0][..., chip * q_local : (chip + 1) * q_local, :]]
            if joint:
                queries.append(prepared_joint[0][..., chip * joint_local : (chip + 1) * joint_local, :])
            gate(record_property, variant, chip, got, dense, reference(torch.cat(queries, dim=2), *kv_true), aligned[chip])

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("q_chunk,k_chunk,head_dim", GEOMETRIES, ids=[GEOMETRY_ID(g) for g in GEOMETRIES])
    def test_ring_geometry(self, ring_mesh, variant, q_chunk, k_chunk, head_dim, record_property):
        maybe_select((q_chunk, k_chunk, head_dim))
        self.run_case(ring_mesh, variant, q_chunk, k_chunk, head_dim, False, record_property)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("q_chunk,k_chunk,head_dim", JOINT_GEOMETRIES, ids=[GEOMETRY_ID(g) for g in JOINT_GEOMETRIES])
    def test_ring_geometry_sharded_joint(self, ring_mesh, variant, q_chunk, k_chunk, head_dim, record_property):
        maybe_select((q_chunk, k_chunk, head_dim))
        self.run_case(ring_mesh, variant, q_chunk, k_chunk, head_dim, True, record_property)


def exp_layout(q_chunk, k_chunk, joint):
    """Local rows, joint rows, heads and grid satisfying the exp ring work split: Q chunks per head a multiple
    of the SDPA columns and 1-3 head-segments per each of the four rows. Local shards are whole K chunks."""
    joint_rows = math.lcm(q_chunk, k_chunk) if joint else 0
    best = None
    for local in range(k_chunk, 4096 + 1, k_chunk):
        chunks = -(-local // q_chunk) + joint_rows // q_chunk
        for columns in range(3, 8):  # at least one pure SDPA column plus the two MUX-writer columns
            if chunks % columns:
                continue
            for heads in range(1, 9):
                if 4 <= heads * chunks // columns <= 12:
                    cost = heads * (local + joint_rows)
                    if best is None or cost < best[0]:
                        best = (cost, local, joint_rows, heads, (columns + 1, 4))
                    break
    assert best is not None, (q_chunk, k_chunk)
    return best[1:]


class TestExpRingGeometry:
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
                dict(worker_l1_size=int(os.getenv("TEST_EXP_RING_WORKER_L1", "1344544")), trace_region_size=16777216)
            )
            semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
            yield mesh, subdevice, semaphores
        finally:
            if mesh is not None:
                close_subdevice_mesh(mesh, manager)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @staticmethod
    def run_case(exp_mesh, variant, q_chunk, k_chunk, head_dim, joint, record_property):
        mesh, subdevice, semaphores = exp_mesh
        local, joint_rows, heads, grid = exp_layout(q_chunk, k_chunk, joint)
        # Without joint KV, a sub-tile tail in the second shard; with it, full shards (joint is visited
        # after shard ring_size-1, so a primary tail would never be the last segment).
        logical_n = 2 * local if joint else 2 * local - 33
        record_property("layout", dict(local=local, joint=joint_rows, heads=heads, grid=grid))
        host = [randn((1, heads, 2 * local, head_dim), 20260925 + i) for i in range(3)]
        for x in host[1:]:
            x[..., logical_n:, :] = randn(x[..., logical_n:, :].shape, 5, scale=8.0)
        joint_host = [randn((1, heads, joint_rows, head_dim), 20261025 + i) for i in range(3)] if joint else None

        def upload(values, mapper):
            return prepare(
                [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values], variant
            )

        inputs = upload(host, ttnn.ShardTensorToMesh(mesh, dim=2))
        joints = upload(joint_host, ttnn.ReplicateTensorToMesh(mesh)) if joint else [None] * 3
        backing = [
            ttnn.allocate_tensor_on_device(
                [1, heads, 2 * local, head_dim], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for x in inputs[1:]
        ]

        def invoke(**options):
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
                **options,
            )

        recipe = dict(precision=precision_of(variant), inputs_prepared=variant.startswith("E_"))
        if variant == "A" and expect_fast_rejection(lambda: invoke(**recipe), True, q_chunk, k_chunk, head_dim):
            record_property("fast_rejected", True)
            return
        try:
            outputs = invoke(**recipe)
        except RuntimeError as error:
            if not l1_rejection(error):
                raise
            record_property("rejected_l1", True)
            pytest.skip(f"{variant} exp ring Q{q_chunk}/K{k_chunk}/D{head_dim} exceeds the pipeline L1")
        actual = chips(outputs[0])
        joint_actual = chips(outputs[1]) if joint else []
        again = invoke(**recipe)
        assert [digest(x) for x in chips(again[0])] == [digest(x) for x in actual], "exp ring recipe not deterministic"

        prepared = [ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)) for x in inputs]
        prepared_joint = [chips(x)[0] for x in joints] if joint else None
        kv_true = [x[..., :logical_n, :] for x in prepared[1:]]
        if joint:
            kv_true = [torch.cat([x, j], dim=2) for x, j in zip(kv_true, prepared_joint[1:])]
        # FP64 reference on the prepared (device) values; dense on the original host values, prepared on upload
        # exactly like the ring's.
        # The exp ring chunks primary and joint queries as separate Q segments, so dense does too.
        queries, dense_queries, gots, ordered, aligned = [], [[], []], [], [[], []], []
        for chip in range(RING):
            query = prepared[0][..., chip * local : (chip + 1) * local, :]
            dense_queries[0].append(host[0][..., chip * local : (chip + 1) * local, :])
            got = actual[chip]
            if joint:
                query = torch.cat([query, prepared_joint[0]], dim=2)
                dense_queries[1].append(joint_host[0])
                got = torch.cat([got, joint_actual[chip]], dim=2)
            queries.append(query)
            gots.append(got)
            segments = []
            for t in range(RING):
                ring_id = (chip + t) % RING
                rows = min(max(logical_n - ring_id * local, 0), local)
                if rows:
                    segments.append((host, ring_id * local, rows))
                if ring_id == RING - 1 and joint:
                    segments.append((joint_host, 0, joint_rows))
            for index in (1, 2):
                ordered[index - 1].append(
                    torch.cat([source[index][..., start : start + rows, :] for source, start, rows in segments], dim=2)
                )
            aligned.append(all(rows % k_chunk == 0 for _, _, rows in segments[:-1]))
        references = [reference(query, *kv_true) for query in queries]

        if variant == "A":
            legacy = invoke()
            legacy_out = chips(legacy[0])
            legacy_joint = chips(legacy[1]) if joint else []
            for chip in range(RING):
                base = torch.cat([legacy_out[chip], legacy_joint[chip]], dim=2) if joint else legacy_out[chip]
                gate(record_property, variant, chip, gots[chip], base, references[chip], False)
            return
        try:
            segments = [
                dense_recipe(mesh, q, *ordered, variant, q_chunk, k_chunk, (min(heads, 8), -(-heads // 8)))
                for q in dense_queries
                if q
            ]
            dense = [torch.cat([segment[chip] for segment in segments], dim=2) for chip in range(RING)]
        except RuntimeError as error:
            if not l1_rejection(error):
                raise
            record_property("dense_reference_rejected_l1", True)
            pytest.skip(f"dense {variant} Q{q_chunk}/K{k_chunk}/D{head_dim} reference exceeds Blackhole L1")
        for chip in range(RING):
            gate(record_property, variant, chip, gots[chip], dense[chip], references[chip], aligned[chip])

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("q_chunk,k_chunk,head_dim", GEOMETRIES, ids=[GEOMETRY_ID(g) for g in GEOMETRIES])
    def test_exp_ring_geometry(self, exp_mesh, variant, q_chunk, k_chunk, head_dim, record_property):
        maybe_select((q_chunk, k_chunk, head_dim))
        self.run_case(exp_mesh, variant, q_chunk, k_chunk, head_dim, False, record_property)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("q_chunk,k_chunk,head_dim", JOINT_GEOMETRIES, ids=[GEOMETRY_ID(g) for g in JOINT_GEOMETRIES])
    def test_exp_ring_geometry_joint(self, exp_mesh, variant, q_chunk, k_chunk, head_dim, record_property):
        maybe_select((q_chunk, k_chunk, head_dim))
        self.run_case(exp_mesh, variant, q_chunk, k_chunk, head_dim, True, record_property)
