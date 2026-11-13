# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test (8 chips): DecodeResidualAllReduce (tt/decode_allreduce.py), the replicated decode residual.

For R = 1, 4, 8, 16, 32 rows, per-device bf16 row-parallel partials and a replicated residual:
* enter: the gathered RES40 residual equals the fractured input, element for element.
* all_reduce (one_shot, residual fused) equals the default all_reduce_async followed by ttnn.add(residual, sum), byte
  for byte on every device, eagerly and under a traced replay.
* Against the fractured path it replaces (tt_all_reduce, i.e. a reduce-scatter, then the residual add): the summation
  order differs, so the test logs how many elements differ and by how much, and bounds the difference.

Run:
    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_decode_allreduce.py
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.decode_allreduce import SUPPORTED_DIM, DecodeResidualAllReduce

DIM = SUPPORTED_DIM


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _assert_same(got, ref, what):
    for dev, (g, r) in enumerate(zip(got, ref)):
        assert torch.equal(g, r), (
            f"{what}, device {dev}: max |diff| {(g.float() - r.float()).abs().max().item():.3e}, "
            f"{(g != r).sum().item()} elements differ"
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 8 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8")], indirect=True)
def test_decode_allreduce_against_the_unfused_and_the_fractured_path(mesh_device, reset_seeds):
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    tp = list(mesh_device.shape)[1]
    tt_ccl = TT_CCL(mesh_device)
    topology = ttnn.Topology.Ring
    dar = DecodeResidualAllReduce(mesh_device, SimpleNamespace(dim=DIM, ccl_topology=lambda: topology), tt_ccl)
    shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)

    for rows in (1, 4, 8, 16, 32):
        g = torch.Generator().manual_seed(rows)
        parts = (torch.randn(tp, 1, 1, rows, DIM, generator=g) * 0.5).to(torch.bfloat16)
        res = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)

        def partial():
            return ttnn.from_torch(
                torch.cat(list(parts), dim=3),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard_mapper,
            )

        fractured_res = ttnn.from_torch(
            res,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_mapper,
        )
        residual = dar.enter(fractured_res)
        for dev, r in enumerate(_per_device(residual)):
            assert torch.equal(r, res), f"rows={rows} device {dev}: enter changed the residual"

        packed = ttnn.to_memory_config(partial(), dar.res_mem)
        plain = ttnn.experimental.all_reduce_async(
            packed,
            dar.buffer,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=topology,
            num_links=dar.num_links,
            memory_config=dar.res_mem,
        )
        unfused = ttnn.add(residual, plain, memory_config=dar.res_mem)
        ref = _per_device(unfused)
        for t in (packed, plain, unfused):
            ttnn.deallocate(t)

        for it in range(2):
            out = dar.with_residual(residual).all_reduce(partial())
            _assert_same(_per_device(out), ref, f"rows={rows} call {it}: fused one_shot vs unfused default")
            ttnn.deallocate(out)

        rs = tt_all_reduce(
            partial(),
            mesh_device,
            tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        old = ttnn.to_torch(ttnn.add(fractured_res, rs), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        new = ref[0]
        diff = (new.float() - old.float()).abs()
        exact = (parts.float().sum(0) + res.float()).abs().max().item()
        logger.info(
            f"rows={rows}: vs the reduce-scatter path {(diff > 0).sum().item()} of {diff.numel()} elements differ, "
            f"max |diff| {diff.max().item():.3e} (max |sum| {exact:.2f})"
        )
        assert (
            diff.max().item() <= 2.0**-5 * exact
        ), f"rows={rows}: the two summation orders differ by more than 4 ulp of the largest sum"
        ttnn.deallocate(residual)
        ttnn.deallocate(fractured_res)

    # Traced: one fused call per replay reads the residual and the partial in place.
    rows = 32
    g = torch.Generator().manual_seed(99)
    parts = (torch.randn(tp, 1, 1, rows, DIM, generator=g) * 0.5).to(torch.bfloat16)
    res = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)
    part = ttnn.from_torch(
        torch.cat(list(parts), dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard_mapper,
    )
    residual = ttnn.from_torch(
        res,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dar.res_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    eager = dar.with_residual(residual).all_reduce(ttnn.clone(part))
    ref = _per_device(eager)
    ttnn.deallocate(eager)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = dar.with_residual(residual).all_reduce(ttnn.clone(part))
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for rep in range(2):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            _assert_same(_per_device(traced), ref, f"traced replay {rep}")
    finally:
        ttnn.release_trace(mesh_device, tid)
