# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import statistics
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare, reference


@pytest.fixture(scope="module")
def recipe_ring_device():
    if not is_blackhole() or ttnn.GetNumAvailableDevices() < 2:
        pytest.skip("Ring recipes require two connected Blackholes")
    mesh = None
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    try:
        options = dict(mesh_shape=ttnn.MeshShape(1, 2), trace_region_size=16777216)
        if os.getenv("TT_METAL_LLK_ASSERTS"):
            options["worker_l1_size"] = 1444992
        mesh = ttnn.open_mesh_device(**options)
        mesh.enable_program_cache()
        hardware = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hardware.x - 1, hardware.y - 1))}
        )
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        yield mesh, subdevice, semaphores, hardware.x - 1
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("distribution", ["normal", "uniform", "changed_max"])
@pytest.mark.parametrize(
    "q_local,k_local,joint_kind,grid,valid_n,batch,heads,kv_heads",
    [
        pytest.param(256, 512, None, (1, 1), None, 1, 1, 1, id="single"),
        pytest.param(768, 1536, None, (1, 1), None, 1, 1, 1, id="staged-odd"),
        pytest.param(768, 1536, None, (2, 1), None, 1, 1, 1, id="unequal-chain"),
        pytest.param(1024, 1024, "replicated", (1, 1), None, 1, 1, 1, id="replicated-joint"),
        pytest.param(1024, 1024, "sharded", (1, 1), None, 1, 1, 1, id="sharded-joint"),
        pytest.param(512, 1024, None, (1, 1), 777, 1, 1, 1, id="skipped-shard-tail"),
        pytest.param(768, 1536, None, (4, 2), None, 1, 4, 2, id="gqa-chain"),
        pytest.param(512, 1024, None, (4, 1), None, 2, 2, 1, id="batched-gqa"),
        *[
            pytest.param(
                4096,
                4096,
                None,
                grid,
                None,
                1,
                4,
                4,
                id=f"perf-{name}",
                marks=pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_PERF") != "1", reason="Opt-in ring timing"),
            )
            for name, grid in [("resident-state", (8, 4)), ("staged-state", (8, 1))]
        ],
    ],
)
def test_recipe_ring(
    recipe_ring_device,
    variant,
    distribution,
    q_local,
    k_local,
    joint_kind,
    grid,
    valid_n,
    batch,
    heads,
    kv_heads,
    record_property,
):
    mesh, subdevice, semaphores, ccl_column = recipe_ring_device

    def generate(q_length, k_length, seed=20260919):
        values = make_inputs(k_length, distribution, q_length=q_length, heads=batch * heads, seed=seed)
        values = [x.reshape(batch, heads, x.shape[2], 128) for x in values]
        values[1:] = [x[:, :kv_heads].contiguous() for x in values[1:]]
        return values

    host = generate(2 * q_local, 2 * k_local)
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh)

    def upload(values, mapper):
        return prepare(
            [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values], variant
        )

    inputs = upload(host, shard)
    backing = [
        ttnn.allocate_tensor_on_device(
            [batch, kv_heads, 2 * k_local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        for x in inputs[1:]
    ]
    joint_host = None
    joints = [None] * 3
    joint_length = 0
    if joint_kind:
        joint_length = 1024 if joint_kind == "sharded" else 512
        joint_host = generate(joint_length, joint_length, seed=20261001)
        joints = upload(joint_host, shard if joint_kind == "sharded" else replicate)

    sources = [*inputs, *(x for x in joints if x is not None)]

    def source_digests():
        return [digest(ttnn.to_torch(chip)) for tensor in sources for chip in ttnn.get_device_tensors(tensor)]

    before = source_digests()

    def invoke():
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            *inputs,
            *joints,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=valid_n or 2 * k_local,
            logical_l=joint_length,
            is_causal=False,
            is_cross=q_local != k_local,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512
            ),
            precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
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

    outputs = invoke()
    actual = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(outputs[0])]
    joint_actual = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(outputs[1])] if joint_kind else []
    kv = [x[..., : valid_n or 2 * k_local, :] for x in host[1:]]
    if joint_host:
        kv = [torch.cat([x, j], dim=2) for x, j in zip(kv, joint_host[1:])]
    kv = [x.repeat_interleave(heads // kv_heads, dim=1) for x in kv]
    observations, references = [], []
    for chip in range(2):
        queries = host[0].chunk(2, dim=2)[chip]
        got = actual[chip]
        if joint_host:
            jq = joint_host[0].chunk(2, dim=2)[chip] if joint_kind == "sharded" else joint_host[0]
            queries = torch.cat([queries, jq], dim=2)
            got = torch.cat([got, joint_actual[chip]], dim=2)
        expected = reference(queries, *kv)
        references.append(expected)
        observed = metrics(got, expected)
        observations.append(observed)
        if variant in ["C", "D"]:
            assert observed["l2_pct"] < {"C": 2, "D": 0.4}[variant]
        for key, value in observed.items():
            record_property(f"chip{chip}_{key}", value)
    ordered = [[], [], []]
    for chip in range(2):
        query = host[0].chunk(2, dim=2)[chip]
        if joint_host:
            jq = joint_host[0].chunk(2, dim=2)[chip] if joint_kind == "sharded" else joint_host[0]
            query = torch.cat([query, jq], dim=2)
        ordered[0].append(query)
        for index in [1, 2]:
            chunks = []
            for rank in [chip, 1 - chip]:
                start, end = rank * k_local, min((rank + 1) * k_local, valid_n or 2 * k_local)
                if end > start:
                    chunks.append(host[index][..., start:end, :])
                if joint_kind == "sharded":
                    chunks.append(joint_host[index].chunk(2, dim=2)[rank])
                elif joint_kind == "replicated" and rank == 1:
                    chunks.append(joint_host[index])
            ordered[index].append(torch.cat(chunks, dim=2))
    dense_inputs = upload([torch.cat(x, dim=0) for x in ordered], ttnn.ShardTensorToMesh(mesh, dim=0))
    dense_output = ttnn.transformer.scaled_dot_product_attention(
        *dense_inputs,
        is_causal=False,
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(batch * heads, 1), q_chunk_size=256, k_chunk_size=512
        ),
    )
    dense = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(dense_output)]
    for chip in range(2):
        got = torch.cat([actual[chip], joint_actual[chip]], dim=2) if joint_kind else actual[chip]
        equal = digest(got) == digest(dense[chip])
        record_property(f"chip{chip}_ordered_dense_equal", equal)
        baseline = metrics(dense[chip], references[chip])
        record_property(f"chip{chip}_dense_l2_pct", baseline["l2_pct"])
        assert observations[chip]["l2_pct"] <= baseline["l2_pct"] * 1.05 + 1e-6
        if variant != "A":
            assert equal, "Ring continuation must preserve the monolithic recipe in the same KV order"
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for index, expected in [(0, actual), (1, joint_actual)]:
                if expected:
                    got = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced[index])]
                    assert [digest(x) for x in got] == [digest(x) for x in expected]
        if os.getenv("TEST_SDPA_RECIPE_PERF") == "1":
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                samples.append((time.perf_counter() - start) * 1000)
            record_property("trace_wall_ms_median", statistics.median(samples))
            record_property("trace_wall_ms_min", min(samples))
            record_property("trace_wall_ms_max", max(samples))
            record_property("preparation_in_timing", False)
    finally:
        ttnn.release_trace(mesh, trace)
    assert source_digests() == before


@pytest.mark.parametrize(
    "case", ["causal", "balanced", "window", "cache", "scale", "compute", "exp", "prepared", "unprepared", "float32_kv"]
)
def test_recipe_ring_rejection(recipe_ring_device, case):
    mesh, subdevice, semaphores, ccl_column = recipe_ring_device
    host = make_inputs(1024, "normal", q_length=1024)
    if case == "float32_kv":
        host[1:] = [x.float() for x in host[1:]]
    inputs = [
        ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        for x in host
    ]
    backing = [
        ttnn.allocate_tensor_on_device([1, 1, 1024, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    options = dict(
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=1024,
        precision=ttnn.SDPAPrecision.COMPENSATED,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(1, 1), q_chunk_size=256, k_chunk_size=512
        ),
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
    changes = {
        "causal": dict(is_causal=True),
        "balanced": dict(is_balanced=True),
        "window": dict(sliding_window_size=128),
        "cache": dict(kv_cache_batch_idx=0),
        "scale": dict(scale=0.125),
        "compute": dict(compute_kernel_config=ttnn.BlackholeComputeKernelConfig()),
        "exp": dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(1, 1), q_chunk_size=256, k_chunk_size=512, exp_approx_mode=False
            )
        ),
        "prepared": dict(inputs_prepared=True),
        "unprepared": dict(precision=ttnn.SDPAPrecision.LOW_PRECISION),
    }
    options.update(changes.get(case, {}))
    with pytest.raises(RuntimeError):
        ttnn.transformer.ring_joint_scaled_dot_product_attention(*inputs, None, None, None, **options)
