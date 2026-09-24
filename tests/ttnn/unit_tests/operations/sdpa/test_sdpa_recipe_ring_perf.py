# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in (TEST_SDPA_RECIPE_RING_PERF=1) trace-wall timing: ring_joint and exp_ring recipes vs legacy.

Two connected Blackholes as a 1x2 FABRIC_1D_RING mesh. Legacy uses the DiT configuration (HiFi2, BF16
dest, exp_approx_mode=False). Each case records the median/min trace wall time over 9 replays; a
recipe the host rejects (e.g. L1) is recorded as rejected instead of failing.
"""

import math
import os
import statistics
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, prepare

pytestmark = pytest.mark.skipif(os.getenv("TEST_SDPA_RECIPE_RING_PERF") != "1", reason="Opt-in ring perf")
ALL = ("legacy", *VARIANTS)
HEADS = int(os.getenv("PERF_HEADS", "10"))


@pytest.fixture(scope="module")
def perf_mesh():
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("requires two connected Blackholes")
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
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), worker_l1_size=1344544, trace_region_size=33554432)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        yield mesh, subdevice, semaphores, grid
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def legacy_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )


def upload(mesh, heads, local, variant):
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    gen = torch.Generator().manual_seed(7)
    host = [torch.randn(1, heads, 2 * local, 128, generator=gen).bfloat16() for _ in range(3)]
    inputs = [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard) for x in host]
    inputs = prepare(inputs, variant) if variant != "legacy" else inputs
    backing = [
        ttnn.allocate_tensor_on_device([1, heads, 2 * local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    return inputs, backing


def time_trace(mesh, invoke):
    invoke()
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    invoke()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        samples = []
        for _ in range(9):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - start) * 1000)
    finally:
        ttnn.release_trace(mesh, trace)
    return statistics.median(samples), min(samples)


def recipe_options(variant, grid, q, k):
    if variant == "legacy":
        return dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False
            ),
        )
    return dict(
        program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k),
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
    )


def record(record_property, variant, fn):
    try:
        median, minimum = fn()
    except RuntimeError as error:
        record_property("rejected", str(error).split("\n")[0][:200] + " | " + str(error).split("info:")[-1][:300])
        pytest.skip(f"{variant} rejected")
    record_property("trace_wall_ms_median", round(median, 4))
    record_property("trace_wall_ms_min", round(minimum, 4))


@pytest.mark.parametrize("variant", ALL)
@pytest.mark.parametrize("q, k", [(128, 512), (256, 512), (256, 256), (320, 384)], ids=lambda v: str(v))
def test_ring_joint_perf(perf_mesh, q, k, variant, record_property):
    mesh, subdevice, semaphores, full = perf_mesh
    local = int(os.getenv("PERF_RING_LOCAL", "4096"))
    inputs, backing = upload(mesh, HEADS, local, variant)
    grid = (full.x - 1, full.y)
    options = recipe_options(variant, grid, q, k)
    if variant == "legacy":
        options["compute_kernel_config"] = legacy_config(mesh)

    def invoke():
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            *inputs,
            None,
            None,
            None,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            logical_l=0,
            is_causal=False,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            subdevice_id=subdevice,
            ccl_core_grid_offset=(full.x - 1, 0),
            use_column_major_ccl=True,
            **options,
        )

    record_property("grid", str(grid))
    record(record_property, variant, lambda: time_trace(mesh, invoke))


@pytest.mark.parametrize("variant", ALL)
@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("q", [224, 256])
def test_exp_ring_perf(perf_mesh, q, passes, variant, record_property):
    mesh, subdevice, semaphores, full = perf_mesh
    chunks = int(os.getenv("PERF_EXP_CHUNKS", "20"))  # local Q chunks per head
    local = q * chunks
    cols = max(c for c in range(3, full.x) if chunks % c == 0)
    rows = full.y - full.y % 2
    segs = chunks // cols  # head-segments per head (rows of the grid one head fills)
    heads = rows * passes // segs
    inputs, backing = upload(mesh, heads, local, variant)
    grid = (cols + 1, rows)
    options = recipe_options(variant, grid, q, 512)
    if variant == "legacy":
        options["compute_kernel_config"] = legacy_config(mesh)

    def invoke():
        return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            *inputs,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            dim=2,
            multi_device_global_semaphore=semaphores[:2],
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=subdevice,
            num_workers_per_link=rows // 2,
            num_buffers_per_channel=16,
            **options,
        )

    record_property("grid", str(grid))
    record_property("local", local)
    record_property("heads", heads)
    record(record_property, variant, lambda: time_trace(mesh, invoke))


@pytest.mark.parametrize("variant", ALL)
@pytest.mark.parametrize("q, k", [(256, 512), (224, 512)], ids=lambda v: str(v))
def test_dense_perf(perf_mesh, q, k, variant, record_property):
    """Same per-core Q x K work as the exp ring cases, on one chip's KV (no ring): isolates compute."""
    mesh, subdevice, semaphores, full = perf_mesh
    heads, sq, sk = 10, 20 * q, 40 * q
    gen = torch.Generator().manual_seed(7)
    host = [torch.randn(1, heads, n, 128, generator=gen).bfloat16() for n in (sq, sk, sk)]
    inputs = [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for x in host]
    inputs = prepare(inputs, variant) if variant != "legacy" else inputs
    options = recipe_options(variant, (10, 10), q, k)
    if variant == "legacy":
        options["compute_kernel_config"] = legacy_config(mesh)

    def invoke():
        return ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **options)

    record(record_property, variant, lambda: time_trace(mesh, invoke))
