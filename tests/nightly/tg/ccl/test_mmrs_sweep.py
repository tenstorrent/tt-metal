# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""In-process config sweep for MinimalMatmulStridedReduceScatterAsync.

Driven by tools/mmrs_sweep/orchestrate.py, which hands over a batch of configs as JSON and joins
device times from the raw profiler CSV afterwards. One pytest process runs a whole batch over a
single held-open mesh, so the 32-device open is paid once per batch rather than once per config.

Every config appends a manifest record the moment it finishes, before the next one starts: that is
what lets the orchestrator salvage a batch that ends in a device hang.
"""

import json
import os

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tools.mmrs_sweep.space import TILE

CONFIGS = json.load(open(os.environ["MMRS_CONFIGS"]))
MANIFEST = os.environ["MMRS_MANIFEST"]
REPS = int(os.environ.get("MMRS_REPS", "5"))
WARMUP = int(os.environ.get("MMRS_WARMUP", "2"))
# Above this many MACs the per-device torch golden costs more than the config it validates, so those
# configs run timing-only. Correctness for the big shapes is the nightly test's job, not the sweep's.
PCC_BUDGET = float(os.environ.get("MMRS_PCC_BUDGET", "5e9"))
PCC_MIN = 0.99
# All configs in a batch share one fabric packet payload: it is set through device_params at mesh
# open, so it cannot vary without reopening the mesh.
PACKET = int(os.environ.get("MMRS_PACKET", "8192"))
CLUSTER_AXIS = int(os.environ.get("MMRS_CLUSTER_AXIS", "0"))


def _fabric_router_config(payload_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = payload_bytes
    return config


def _device_params():
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}
    if PACKET != 4096:
        params["fabric_router_config"] = _fabric_router_config(PACKET)
    return params


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True, ids=[f"packet{PACKET}"])
@pytest.mark.timeout(36000)
def test_mmrs_sweep(mesh_device):
    assert all(c["packet"] == PACKET for c in CONFIGS), "batch mixes packet sizes; orchestrator must split"
    ring_size = mesh_device.shape[CLUSTER_AXIS]
    if ring_size == 1:
        pytest.skip(f"cluster_axis={CLUSTER_AXIS} has 1 device; reduce-scatter needs a ring > 1")

    mesh_device.enable_program_cache()
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    semaphores = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    manifest = []
    tensors = {}  # shape -> (input, weight, mm_goldens, rs_goldens); hoisted out of the config loop

    def build_tensors(M, K, N):
        torch.manual_seed(0)
        shard_dims = [None, None]
        shard_dims[CLUSTER_AXIS] = 0
        torch_input = torch.randn([1, 1, M, K], dtype=torch.float32)
        torch_weight = torch.randn([ring_size, 1, K, N], dtype=torch.float32)
        a = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=dram,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        b = ttnn.from_torch(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=dram,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        mm_goldens = rs_goldens = None
        if M * K * N <= PCC_BUDGET:
            chunks = torch.chunk(torch_weight, ring_size, dim=0)
            mm_goldens = [torch.matmul(torch_input, chunks[d]) for d in range(ring_size)]
            reduced = torch.sum(torch.stack(mm_goldens), dim=0)
            rs_goldens = torch.chunk(reduced, ring_size, dim=3)
        return a, b, mm_goldens, rs_goldens

    def composer(concat_dims):
        shape = list(mesh_device.shape)
        shape[1 - CLUSTER_AXIS] = 1
        return ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(shape)))

    def check_pcc(mm_out, rs_out, mm_goldens, rs_goldens):
        """Worst PCC over devices for both outputs. The MM output is only an intermediate, but a
        blocking config that corrupts it while the reduce still sums to something plausible is
        exactly the failure worth catching."""
        worst = 1.0
        dims = [0, 0]
        dims[1 - CLUSTER_AXIS] = 1
        mm_torch = ttnn.to_torch(mm_out, mesh_composer=composer(dims))
        for d in range(ring_size):
            _, out = comp_pcc(mm_torch[d : d + 1, :, :, :], mm_goldens[d], PCC_MIN)
            worst = min(worst, float(out.split("PCC: ")[-1]))
        dims = [3, 3]
        dims[1 - CLUSTER_AXIS] = 0
        rs_torch = ttnn.to_torch(rs_out, mesh_composer=composer(dims))
        for d, chunk in enumerate(torch.chunk(rs_torch, ring_size, dim=3)):
            _, out = comp_pcc(chunk, rs_goldens[d], PCC_MIN)
            worst = min(worst, float(out.split("PCC: ")[-1]))
        return round(worst, 5)

    def run_config(cfg, rec):
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        if (M, K, N) not in tensors:
            tensors[(M, K, N)] = build_tensors(M, K, N)
        a, b, mm_goldens, rs_goldens = tensors[(M, K, N)]

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=cfg["mb"],
            K_block_size=cfg["kb"],
            N_block_size=cfg["nb"],
            subblock_h=cfg["sbh"],
            subblock_w=cfg["sbw"],
            compute_with_storage_grid_size=ttnn.CoreCoord(cfg["gx"], cfg["gy"]),
        )
        rs_offset = ttnn.CoreCoord(0, cfg["gy"])

        def invoke():
            if cfg["mode"] == "fused":
                return ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                    a,
                    b,
                    3,
                    semaphores,
                    rs_offset,
                    num_links=cfg["links"],
                    memory_config_mm=dram,
                    rs_output_mem_config=dram,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=CLUSTER_AXIS,
                    config=matmul_config,
                    compute_kernel_config=compute_config,
                    barrier_semaphore=barrier_semaphore,
                    num_workers_per_link=cfg["workers"],
                    chunk_width_in_mm_blocks=cfg["chunk"],
                )
            mm_out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=compute_config, config=matmul_config)
            rs_out = ttnn.experimental.strided_reduce_scatter_async(
                mm_out,
                None,
                3,
                semaphores,
                barrier_semaphore=barrier_semaphore,
                num_links=cfg["links"],
                memory_config=dram,
                topology=ttnn.Topology.Ring,
                cluster_axis=CLUSTER_AXIS,
                num_workers_per_link=cfg["workers"],
                mm_cores_y=cfg["gy"],
                mm_block_ht=cfg["mb"],
                mm_block_wt=cfg["nb"],
                mm_N_full_block_wt=N // TILE // cfg["gx"],
                chunk_width_in_mm_blocks=cfg["chunk"],
            )
            return mm_out, rs_out

        mesh_device.clear_program_cache()
        mm_out, rs_out = invoke()
        ttnn.synchronize_device(mesh_device)
        if mm_goldens is not None:
            rec["pcc"] = check_pcc(mm_out, rs_out, mm_goldens, rs_goldens)
        mm_out.deallocate()
        rs_out.deallocate()

        for _ in range(WARMUP):
            for t in invoke():
                t.deallocate()
        ttnn.synchronize_device(mesh_device)
        for _ in range(REPS):
            for t in invoke():
                t.deallocate()
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        rec["ok"] = True

    for i, cfg in enumerate(CONFIGS):
        # nops lets the orchestrator slice the ordered CSV durations: the fused op is one device op,
        # the unfused path is a matmul followed by a reduce-scatter.
        rec = dict(cfg, idx=i, nops=1 if cfg["mode"] == "fused" else 2, ok=False, pcc=None)
        hang = False
        try:
            run_config(cfg, rec)
        except Exception as e:
            msg = str(e).replace("\n", " ")
            rec["err"] = msg[:200]
            hang = any(k in msg for k in ("TIMEOUT:", "potential hang", "unrecoverable"))
        manifest.append(rec)
        json.dump(manifest, open(MANIFEST, "w"))
        logger.info(f"[{i+1}/{len(CONFIGS)}] ok={rec['ok']} pcc={rec['pcc']} {cfg}")
        # A dispatch timeout can leave the device unrecoverable, and then every later config fails
        # fast at roughly the timeout each. Stop now that this config is recorded; the orchestrator
        # resets the device, keeps the completed prefix and requeues the rest.
        if hang:
            raise RuntimeError(f"ABORT_ON_DISPATCH_TIMEOUT at config {i}: {cfg}")

    print("SWEEP_DONE", flush=True)
