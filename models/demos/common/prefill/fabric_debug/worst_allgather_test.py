#!/usr/bin/env python3
"""Standalone traced reproducer of the WORST AllGather from Kimi-K2.6 prefill.

Isolates the single collective that dominated the connected-mesh slowdown so we can
A/B it across fabric setups:

  worst op  = AllGatherDeviceOperation (plain ttnn.all_gather, NOT async)
  tensor    = [1,1,640,1792] bf16 TILE DRAM-INTERLEAVED (per-device input [1,1,640,448])
  kwargs    = dim=3, cluster_axis=1, num_links=2, topology=Linear, ring_size=4
  measured  = ~43us (1-galaxy 1D) / ~55us (1-galaxy 2D) / ... (connected 2-galaxy 2D)

Captures ONE all_gather into a ttnn trace, then execute_trace()es it N_REPLAYS (default 100)
times and reports per-replay e2e; with TT_METAL_DEVICE_PROFILER=1 it also flushes the device
profiler so the per-op DEVICE KERNEL DURATION lands in the ops CSV. A tracy signpost brackets
the replay region ("worst_ag_replay") so it is trivially isolable.

Runs three ways (fabric selected by PREFILL_FABRIC_MODE, mesh by the launcher):
  1) single galaxy FABRIC_1D   : PREFILL_FABRIC_MODE=1d,  direct python
  2) single galaxy FABRIC_2D   : PREFILL_FABRIC_MODE=2d,  direct python
  3) connected 2-galaxy 2D     : PREFILL_FABRIC_MODE=2d,  via ttrun (rank0 runs it, rank1 idles)

The AllGather is IDENTICAL in all three (rank0's 8x4 mesh) — only the fabric bring-up differs,
which is exactly the variable under test.
"""
import os
import time

# --- Channel-trimming env setup (must run BEFORE ttnn/metal reads rtoptions) ---
# Keyed off WORST_AG_CT_MODE (capture|apply) + this rank, so the connected run needs no special
# per-rank wrapper file (avoids the mpirun/NFS wrapper-not-found gremlin on the peer). Each rank
# writes/reads its OWN profile under CTDIR/rank<N>.
_CT_MODE = os.environ.get("WORST_AG_CT_MODE", "")
_CT_RANK = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMIX_RANK", "0"))
_CT_DIR = os.environ.get("WORST_AG_CT_DIR", "/data/ppopovic/prof_out/ct_connected")
if _CT_MODE == "capture":
    os.environ["TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE"] = "1"
    os.environ["TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING"] = "1"
    os.environ["TT_METAL_LOGS_PATH"] = f"{_CT_DIR}/rank{_CT_RANK}"
    os.makedirs(f"{_CT_DIR}/rank{_CT_RANK}", exist_ok=True)
elif _CT_MODE == "apply":
    os.environ[
        "TT_METAL_FABRIC_TRIMMING_PROFILE"
    ] = f"{_CT_DIR}/rank{_CT_RANK}/generated/reports/channel_trimming_capture.yaml"

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import open_mesh_device
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config

# --- worst-AllGather parameters (from the pipe5L_r0 profiler capture, row 25333: 279us) ---
# Profiler INPUT_0_X=1792 is the PER-DEVICE input width; gathered over the 4-device axis on dim 3
# => output width 1792*4 = 7168. So the full (sharded-source == gathered-output) tensor is X=7168,
# each device holds [1,1,640,1792], and all_gather(dim=3, cluster_axis=1) -> [1,1,640,7168].
OUT_SHAPE = [1, 1, 640, 7168]  # gathered/output shape (per-device input = 7168/4 = 1792)
GATHER_DIM = 3  # tensor dim gathered
CLUSTER_AXIS = 1  # mesh axis of size 4 (ring_size 4)
MESH_SHAPE = (8, 4)
NUM_LINKS = int(os.environ.get("WORST_AG_NUM_LINKS", "2"))
N_REPLAYS = int(os.environ.get("WORST_AG_REPLAYS", "100"))
TRACE_REGION = 32 * 1024 * 1024  # generous; this single op needs << 1MB
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _signpost(name):
    try:
        from tracy import signpost

        signpost(name)
    except Exception:
        pass


def _run_allgather(tt_input):
    return ttnn.all_gather(
        tt_input,
        dim=GATHER_DIM,
        cluster_axis=CLUSTER_AXIS,
        num_links=NUM_LINKS,
        topology=ttnn.Topology.Linear,
        memory_config=DRAM,
    )


def run_on_rank0(mesh_device):
    logger.info(
        f"[worst-ag] mesh={tuple(mesh_device.shape)} out_shape={OUT_SHAPE} num_links={NUM_LINKS} replays={N_REPLAYS}"
    )

    # Sharded input: replicate over mesh axis 0 (8), shard tensor dim 3 over mesh axis 1 (4)
    # -> each device holds [1,1,640,448]; all_gather(dim=3, cluster_axis=1) -> [1,1,640,1792].
    shard_dims = (None, GATHER_DIM)  # cluster_axis==1 -> (None, dim)
    torch_input = torch.rand(OUT_SHAPE, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=MESH_SHAPE),
        device=mesh_device,
    )

    # 1) eager compile pass (populate program cache) — trace records dispatch only.
    out = _run_allgather(tt_input)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    # 2) capture ONE all_gather into a trace.
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = _run_allgather(tt_input)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # 3) replay it N_REPLAYS times, ISOLATED per-op: sync before (device idle) -> execute -> sync
    # after (op complete) -> record. This times ONE AllGather per iteration (latency, not pipelined
    # throughput). Bracketed by a signpost so the device-profiler region is isolable.
    import statistics

    _signpost("worst_ag_replay")
    times_us = []
    for _ in range(N_REPLAYS):
        ttnn.synchronize_device(mesh_device)  # device idle BEFORE the op
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)  # wait for the op to COMPLETE
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
    _signpost("worst_ag_replay_end")

    mean, mn, mx = statistics.mean(times_us), min(times_us), max(times_us)
    med = statistics.median(times_us)
    logger.info(
        f"[worst-ag] per-op host latency (sync before+after, N={N_REPLAYS}): "
        f"mean={mean:.1f}us min={mn:.1f}us max={mx:.1f}us median={med:.1f}us"
    )
    logger.info(
        "[worst-ag] (device KERNEL duration mean/min/max is in the ops CSV; host latency above includes ~2 syncs of dispatch)"
    )

    # flush device profiler so per-op DEVICE KERNEL DURATION lands in the ops CSV
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)

    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    return mean


def main():
    # ttrun / connected: init the distributed context so each rank binds its mesh_id.
    multi = False
    try:
        if not ttnn.distributed_context_is_initialized():
            ttnn.init_distributed_context()
        rank = int(ttnn.distributed_context_get_rank())
        size = int(ttnn.distributed_context_get_size())
        multi = size > 1
    except Exception:
        rank, size = 0, 1

    logger.info(f"[worst-ag] rank {rank}/{size} (multi={multi})")
    mesh_device = open_mesh_device(MESH_SHAPE, KimiK26Config, l1_small_size=0, trace_region_size=TRACE_REGION)
    try:
        if multi:
            ttnn.distributed_context_barrier()  # all ranks up before rank0 runs

        if rank == 0:
            run_on_rank0(mesh_device)
        else:
            # Non-rank0 (connected case): just hold the fabric up; do no collective work.
            logger.info(f"[worst-ag] rank {rank} idle — holding fabric up")

        if multi:
            ttnn.distributed_context_barrier()  # rank0 done before anyone tears down
    finally:
        ttnn.synchronize_device(mesh_device)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        logger.info(f"[worst-ag] rank {rank} shutdown complete")


if __name__ == "__main__":
    main()
