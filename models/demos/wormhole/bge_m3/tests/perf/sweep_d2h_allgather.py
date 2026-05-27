# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
D2H all-gather sweep -- saturate device, eliminate host concat.

Source state matches the actual BGE-M3 trace output:
    [B, 1, S, D]  bf8b TILE  DRAM_INTERLEAVED  sharded(dim=0) across all chips

Two configurations:
    dp8_global256   = 1x8 mesh,  global_batch=256  (dev box, 32 per-chip)
    dp32_global1024 = 4x8 mesh,  global_batch=1024 (Blackhole Galaxy, 32 per-chip)

The per-chip payload (32 MB bf16) is identical across both, so we exercise the same
shard layout while stressing 4x more chips at DP=32. To select a specific config,
use `pytest ... -k "dp32"` or `-k "dp8"`.

Hypothesis: most of S0's 485 ms is host concat + bf8b unpack work that runs single-threaded.
If we instead use all device cores to (a) untilize to bf16 row-major and (b) all-gather to
make the result replicated on a single chip (or a smaller submesh), the host only has to do
a contiguous PCIe read with no unpack work.

Variants:
    A0 baseline                  - S0 from previous sweep (to_torch + ConcatMesh)
    A1 untilize_only             - device untilize_with_unpadding + to_torch    (== S4 from earlier sweep)
    A2 untilize + ag_full_ring   - untilize + ring all_gather to ALL 8 chips, D2H from chip0 only
    A3 untilize + ag_full_linear - untilize + linear all_gather to ALL 8 chips, D2H from chip0 only
    A4 untilize + ag_4chip       - untilize + all_gather over a 1x4 submesh (cluster_axis along 4 chips)
                                   D2H from 2 chips in parallel (each chip holds 128 rows)
    A5 untilize + ag_2chip       - untilize + all_gather over a 1x2 pair (cluster_axis along 2 chips)
                                   D2H from 4 chips in parallel (each chip holds 64 rows)

In A4 and A5 the all_gather is INTRA-cluster (along the cluster axis), so the data
becomes replicated only within each row of the cluster. We then D2H from one chip
per cluster row, in parallel across cluster rows.

Each variant reports:
    untilize_ms        - device-side untilize_with_unpadding
    allgather_ms       - device-side all-gather (or 0 if none)
    d2h_ms             - PCIe + host wrap
    cat_ms             - host-side torch.cat (if multiple shards survived)
    total_ms           - wall of entire pipeline
    pcc_vs_A0          - bf16 PCC against A0 reference

Run:
    cd /home/gtobar/new_test && source local_env.sh && cd tt-metal
    python -m pytest models/demos/wormhole/bge_m3/tests/perf/sweep_d2h_allgather.py -sv
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger

import ttnn

SEQ_LEN = 512
HIDDEN = 1024
WARMUP_ITERS = 2
MEASURED_ITERS = 5


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().to(torch.float32)
    b_f = b.flatten().to(torch.float32)
    if (a_f == b_f).all():
        return 1.0
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    denom = (a_c.norm() * b_c.norm()).item()
    if denom == 0:
        return 1.0
    pcc = (a_c @ b_c).item() / denom
    return max(-1.0, min(1.0, pcc))


def _allocate_source(global_batch: int, mesh_device, mapper) -> ttnn.Tensor:
    """Same as the model's trace_output state: bf8b TILE DRAM, sharded along dim=0."""
    src_torch = torch.randn((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(
        src_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# Variants
# ============================================================================


def _A0_baseline(out_dev, composer, mesh_device):
    t = time.perf_counter()
    out_torch = ttnn.to_torch(out_dev, mesh_composer=composer)
    ttnn.synchronize_device(mesh_device)
    d2h = (time.perf_counter() - t) * 1000.0
    return {"untilize_ms": 0.0, "allgather_ms": 0.0, "d2h_ms": d2h, "cat_ms": 0.0, "total_ms": d2h, "out": out_torch}


def _do_untilize(out_dev, mesh_device):
    """Untilize bf8b TILE -> bf16 ROW_MAJOR (use all cores per chip)."""
    b, _, s, d = out_dev.shape
    t0 = time.perf_counter()
    out_rm = ttnn.untilize_with_unpadding(
        out_dev,
        output_tensor_end=(b - 1, 0, s - 1, d - 1),
        use_multicore=True,
    )
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1000.0, out_rm


def _A1_untilize_only(out_dev, composer, mesh_device):
    t_tot = time.perf_counter()
    untilize_ms, out_rm = _do_untilize(out_dev, mesh_device)

    t = time.perf_counter()
    out_torch = ttnn.to_torch(out_rm, mesh_composer=composer)
    ttnn.synchronize_device(mesh_device)
    d2h_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    ttnn.deallocate(out_rm)
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": out_torch,
    }


def _A_untilize_then_full_allgather(out_dev, mesh_device, topology):
    """Untilize, then all_gather across ALL 8 chips (no cluster_axis = full mesh).
    Result is replicated on every chip; we read from chip 0 only.
    """
    t_tot = time.perf_counter()
    untilize_ms, out_rm = _do_untilize(out_dev, mesh_device)

    t = time.perf_counter()
    out_ag = ttnn.all_gather(out_rm, dim=0, topology=topology)
    ttnn.synchronize_device(mesh_device)
    allgather_ms = (time.perf_counter() - t) * 1000.0

    # D2H from chip 0 only (every chip has the full tensor now)
    t = time.perf_counter()
    shard0 = ttnn.get_device_tensors(out_ag)[0]
    out_torch = shard0.to_torch()
    ttnn.synchronize_device(mesh_device)
    d2h_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    ttnn.deallocate(out_rm)
    ttnn.deallocate(out_ag)
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": allgather_ms,
        "d2h_ms": d2h_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": out_torch,
    }


def _A2_untilize_ag_ring(out_dev, mesh_device):
    return _A_untilize_then_full_allgather(out_dev, mesh_device, ttnn.Topology.Ring)


def _A3_untilize_ag_linear(out_dev, mesh_device):
    return _A_untilize_then_full_allgather(out_dev, mesh_device, ttnn.Topology.Linear)


def _A_partial_allgather_then_parallel_d2h(out_dev, mesh_device, cluster_size, executor):
    """Untilize, then all_gather over a sub-cluster, then parallel D2H.

    The mesh is 1x8. We use cluster_axis=1 (the size-8 axis). With cluster_size 4,
    all_gather replicates within each group of 4 chips along the cluster axis -- but
    cluster_axis is applied to the whole row, not partial. So for a 1x8 mesh, the
    only valid cluster_axis options gather across all 8 chips.

    Instead, we emulate "partial all_gather" by reshaping the input layout so that
    the relevant tensor dim is sharded across a smaller submesh slice -- but this
    requires re-mapping the source tensor, which is non-trivial without copying.

    Simpler approach: skip cluster_axis and instead do the full all_gather + D2H
    from N chips in parallel via threadpool. After full all_gather, all chips have
    identical data, so we can pull from N chips in parallel. The PCIe traffic is the
    same total bytes (N * full_size), but it's spread over N parallel PCIe links.

    Wait -- that's worse, not better: we'd send 8x256MB = 2GB total via PCIe.

    Best variant: do the FULL all_gather, but then D2H from one chip ONLY.
    That's already A2/A3.

    Skipping A4/A5 as designed; instead introducing parallel-from-original (no all_gather)
    using direct from_device + threaded shard.to_torch -- this is U1 from sweep_d2h_host_unpack.
    """
    raise NotImplementedError("Partial all_gather requires re-mapping source; using U1-style threading instead.")


def _A4_no_allgather_threaded_to_torch(out_dev, mesh_device, executor):
    """No all_gather. Untilize on device, then 8 parallel shard.to_torch + torch.cat on host.

    This is essentially S4 from the earlier sweep but with the host-side per-shard work
    parallelized across N threads. Tests whether device-side untilize + host threading
    composes the way we hypothesized.
    """
    t_tot = time.perf_counter()
    untilize_ms, out_rm = _do_untilize(out_dev, mesh_device)

    t = time.perf_counter()
    # Get the host ttnn tensor (PCIe-only)
    host_ttnn = ttnn.from_device(out_rm)
    ttnn.synchronize_device(mesh_device)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    # Parallel per-shard to_torch
    t = time.perf_counter()
    shards = ttnn.get_device_tensors(host_ttnn)
    futs = [executor.submit(lambda s=s: s.to_torch()) for s in shards]
    parts = [f.result() for f in futs]
    unpack_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    out_torch = torch.cat(parts, dim=0)
    cat_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    ttnn.deallocate(out_rm)
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + unpack_ms,
        "cat_ms": cat_ms,
        "total_ms": total_ms,
        "out": out_torch,
    }


def _A6_preallocated_dest_parallel_copy(out_dev, mesh_device, executor, dest_buf, global_batch):
    """A4 + pre-allocated destination + parallel slice copies (no torch.cat).

    Each of 8 threads does:  dest_buf[start:end].copy_(shard.to_torch())
    Threads write into NON-OVERLAPPING slices of one big pre-allocated tensor.
    The torch.cat step is eliminated; the parallel slice copies replace it.
    """
    t_tot = time.perf_counter()
    untilize_ms, out_rm = _do_untilize(out_dev, mesh_device)

    t = time.perf_counter()
    host_ttnn = ttnn.from_device(out_rm)
    ttnn.synchronize_device(mesh_device)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    shards = ttnn.get_device_tensors(host_ttnn)
    num_shards = len(shards)
    batch_per_shard = global_batch // num_shards

    def _copy_into_slice(idx, shard):
        start = idx * batch_per_shard
        end = start + batch_per_shard
        # Each thread independently: to_torch() then write into its slice.
        local = shard.to_torch()
        dest_buf[start:end].copy_(local)

    futs = [executor.submit(_copy_into_slice, i, s) for i, s in enumerate(shards)]
    for f in futs:
        f.result()
    unpack_and_copy_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    ttnn.deallocate(out_rm)
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + unpack_and_copy_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": dest_buf,
    }


def _setup_A8_staging(out_dev, mesh_device):
    """One-time setup for A8: untilize once to get a sample, allocate a DRAM staging
    buffer and a matching pre-allocated host ttnn tensor.

    Returns: (dram_staging, host_staging) — both reused across iters.

    Note: the staging is allocated for the *untilized* shape (bf16 RM), matching
    what we'll D2H every iter.
    """
    b, _, s, d = out_dev.shape
    sample_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    dram_staging = ttnn.clone(sample_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host_staging = ttnn.allocate_tensor_on_host(dram_staging.spec, mesh_device)
    ttnn.deallocate(sample_rm)
    return dram_staging, host_staging


def _A8_prealloc_host_staging(out_dev, mesh_device, executor, dram_staging, host_staging, global_batch):
    """Yolo-style: pre-allocated DRAM staging + pre-allocated host ttnn tensor,
    use copy_device_to_host_tensor (not from_device) so the host ttnn tensor is
    REUSED across iters — no per-iter ttnn host allocation.

    Then ThreadPool(8) per-shard .to_torch() + torch.cat (same host-side path as A4).
    """
    t_tot = time.perf_counter()

    # Step 1: untilize on device, copy into the persistent DRAM staging slot.
    b, _, s, d = out_dev.shape
    t = time.perf_counter()
    out_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    # Persist into the staging slot via on-device copy (cheap, all on chip)
    ttnn.copy(out_rm, dram_staging)
    ttnn.synchronize_device(mesh_device)
    untilize_ms = (time.perf_counter() - t) * 1000.0
    ttnn.deallocate(out_rm)

    # Step 2: PCIe D2H from dram_staging into pre-allocated host_staging
    # No new host allocation — the existing host_staging buffers get overwritten.
    t = time.perf_counter()
    ttnn.copy_device_to_host_tensor(dram_staging, host_staging, blocking=True, cq_id=0)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    # Step 3: ThreadPool(8) per-shard .to_torch() + torch.cat (identical to A4)
    t = time.perf_counter()
    shards = ttnn.get_device_tensors(host_staging)
    futs = [executor.submit(lambda s=s: s.to_torch()) for s in shards]
    parts = [f.result() for f in futs]
    unpack_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    out_torch = torch.cat(parts, dim=0)
    cat_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + unpack_ms,
        "cat_ms": cat_ms,
        "total_ms": total_ms,
        "out": out_torch,
    }


def _A10_prealloc_batch_to_torch_threaded(out_dev, mesh_device, dram_staging, host_staging, dest_buf, n_threads):
    """A9 + multi-threaded batch_to_torch.

    Uses the new n_threads kwarg on batch_to_torch to split the memcpy work
    across multiple std::thread workers. Aims to saturate the dual-socket EPYC's
    aggregate memory bandwidth (~100 GB/s) instead of one core's ~12 GB/s.
    """
    t_tot = time.perf_counter()

    b, _, s, d = out_dev.shape
    t = time.perf_counter()
    out_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    ttnn.copy(out_rm, dram_staging)
    ttnn.synchronize_device(mesh_device)
    untilize_ms = (time.perf_counter() - t) * 1000.0
    ttnn.deallocate(out_rm)

    t = time.perf_counter()
    ttnn.copy_device_to_host_tensor(dram_staging, host_staging, blocking=True, cq_id=0)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    host_staging.batch_to_torch(dest_buf, physical=True, n_threads=n_threads)
    compose_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + compose_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": dest_buf,
    }


def _A12_copy_device_to_torch(out_dev, mesh_device, dram_staging, dest_buf):
    """A8/A10 stack but using the new ttnn.copy_device_to_torch fast path.

    Eliminates the host_staging tensor entirely: the device DMA writes directly
    into the slices of dest_buf. One memcpy in the path instead of two.
    """
    t_tot = time.perf_counter()

    b, _, s, d = out_dev.shape
    t = time.perf_counter()
    out_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    ttnn.copy(out_rm, dram_staging)
    ttnn.synchronize_device(mesh_device)
    untilize_ms = (time.perf_counter() - t) * 1000.0
    ttnn.deallocate(out_rm)

    t = time.perf_counter()
    ttnn.copy_device_to_torch(dram_staging, dest_buf)
    d2h_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": dest_buf,
    }


def _A9_prealloc_host_staging_batch_to_torch(out_dev, mesh_device, dram_staging, host_staging, dest_buf, global_batch):
    """A8 stack + yolo's batch_to_torch instead of threaded shard.to_torch + torch.cat.

    All host work collapses into a single C++ call that memcpys all shards
    contiguously into a pre-allocated torch tensor (no torch.cat, no per-shard
    Python to_torch overhead, GIL released for the duration).
    """
    t_tot = time.perf_counter()

    b, _, s, d = out_dev.shape
    t = time.perf_counter()
    out_rm = ttnn.untilize_with_unpadding(out_dev, output_tensor_end=(b - 1, 0, s - 1, d - 1), use_multicore=True)
    ttnn.copy(out_rm, dram_staging)
    ttnn.synchronize_device(mesh_device)
    untilize_ms = (time.perf_counter() - t) * 1000.0
    ttnn.deallocate(out_rm)

    t = time.perf_counter()
    ttnn.copy_device_to_host_tensor(dram_staging, host_staging, blocking=True, cq_id=0)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    host_staging.batch_to_torch(dest_buf, physical=True)
    compose_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + compose_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": dest_buf,
    }


def _A7_preallocated_dest_single_thread(out_dev, mesh_device, dest_buf, global_batch):
    """Sanity check: same pre-allocated dest, but single-threaded copies.

    Tells us how much of A6's win comes from parallelism vs from eliminating torch.cat.
    """
    t_tot = time.perf_counter()
    untilize_ms, out_rm = _do_untilize(out_dev, mesh_device)

    t = time.perf_counter()
    host_ttnn = ttnn.from_device(out_rm)
    ttnn.synchronize_device(mesh_device)
    d2h_pcie_ms = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    shards = ttnn.get_device_tensors(host_ttnn)
    num_shards = len(shards)
    batch_per_shard = global_batch // num_shards
    for i, shard in enumerate(shards):
        start = i * batch_per_shard
        end = start + batch_per_shard
        dest_buf[start:end].copy_(shard.to_torch())
    seq_copy_ms = (time.perf_counter() - t) * 1000.0

    total_ms = (time.perf_counter() - t_tot) * 1000.0
    ttnn.deallocate(out_rm)
    return {
        "untilize_ms": untilize_ms,
        "allgather_ms": 0.0,
        "d2h_ms": d2h_pcie_ms + seq_copy_ms,
        "cat_ms": 0.0,
        "total_ms": total_ms,
        "out": dest_buf,
    }


# ============================================================================
# Harness
# ============================================================================


# Mesh / batch configurations:
#   - dp8 / global256  : dev-box default (1x8 mesh, 256 global batch, 32 per chip)
#   - dp32 / global1024: Blackhole Galaxy DP=32 (4x8 mesh, 1024 global batch, 32 per chip)
#                        Run with: pytest ... -k "dp32"
# The per-chip payload (32 MB bf16) is identical across the two configs, so we exercise
# the same shard layout but stress 4x more chips (and 4x the host concat work).
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, global_batch",
    [
        ((1, 8), 256),
        ((4, 8), 1024),
    ],
    indirect=["mesh_device"],
    ids=["dp8_global256", "dp32_global1024"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 1_000_000, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_d2h_allgather(mesh_device, global_batch):
    num_devices = mesh_device.get_num_devices()
    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0)
    composer = ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)

    payload_bf8b_mb = (global_batch * SEQ_LEN * HIDDEN * 1.0625) / (1024 * 1024)
    payload_bf16_mb = (global_batch * SEQ_LEN * HIDDEN * 2.0) / (1024 * 1024)
    per_chip_bf8b_mb = payload_bf8b_mb / num_devices

    logger.info("=" * 110)
    logger.info(f"D2H all-gather sweep  |  devices={num_devices}  global_batch={global_batch}")
    logger.info(
        f"Payload: bf8b={payload_bf8b_mb:.1f} MB / bf16={payload_bf16_mb:.1f} MB / per chip bf8b={per_chip_bf8b_mb:.1f} MB"
    )
    logger.info("=" * 110)

    # Persistent source state
    out_dev = _allocate_source(global_batch, mesh_device, mapper)
    executor = ThreadPoolExecutor(max_workers=8)

    # Reference output for PCC
    ref = _A0_baseline(out_dev, composer, mesh_device)
    ref_out = ref["out"]
    logger.info(f"Reference: shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")

    # Pre-allocated destination buffer for A6 / A7. Reused across iters and warmups.
    dest_buf_A6 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    dest_buf_A7 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)

    # A8/A9 setup: persistent DRAM staging + pinned host ttnn tensor.
    logger.info("  Setting up A8/A9 staging buffers (one-time) ...")
    A8_dram_staging, A8_host_staging = _setup_A8_staging(out_dev, mesh_device)

    # A9 also needs a pre-allocated torch dest buffer.
    # batch_to_torch(physical=True) writes contiguous physical bytes; for bf16
    # row-major output the physical shape equals the logical shape, so dest is
    # plain [B, 1, S, D] bf16.
    dest_buf_A9 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    # A10 thread-count sweep buffers (one per variant so they don't collide)
    dest_buf_A10_8 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    dest_buf_A10_16 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    dest_buf_A10_32 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    dest_buf_A10_64 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    dest_buf_A10_auto = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)
    # A12 reuses the dram_staging slot but writes directly to its own dest torch tensor.
    dest_buf_A12 = torch.empty((global_batch, 1, SEQ_LEN, HIDDEN), dtype=torch.bfloat16)

    variants = [
        ("A0_baseline", lambda: _A0_baseline(out_dev, composer, mesh_device)),
        ("A1_untilize_only", lambda: _A1_untilize_only(out_dev, composer, mesh_device)),
        ("A2_untilize_ag_ring", lambda: _A2_untilize_ag_ring(out_dev, mesh_device)),
        ("A3_untilize_ag_linear", lambda: _A3_untilize_ag_linear(out_dev, mesh_device)),
        ("A4_untilize_threaded_to_torch", lambda: _A4_no_allgather_threaded_to_torch(out_dev, mesh_device, executor)),
        (
            "A6_prealloc_dest_parallel",
            lambda: _A6_preallocated_dest_parallel_copy(out_dev, mesh_device, executor, dest_buf_A6, global_batch),
        ),
        (
            "A7_prealloc_dest_seq",
            lambda: _A7_preallocated_dest_single_thread(out_dev, mesh_device, dest_buf_A7, global_batch),
        ),
        (
            "A8_prealloc_host_staging",
            lambda: _A8_prealloc_host_staging(
                out_dev, mesh_device, executor, A8_dram_staging, A8_host_staging, global_batch
            ),
        ),
        (
            "A9_prealloc_batch_to_torch",
            lambda: _A9_prealloc_host_staging_batch_to_torch(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A9, global_batch
            ),
        ),
        (
            "A10_batch_to_torch_8t",
            lambda: _A10_prealloc_batch_to_torch_threaded(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A10_8, 8
            ),
        ),
        (
            "A10_batch_to_torch_16t",
            lambda: _A10_prealloc_batch_to_torch_threaded(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A10_16, 16
            ),
        ),
        (
            "A10_batch_to_torch_32t",
            lambda: _A10_prealloc_batch_to_torch_threaded(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A10_32, 32
            ),
        ),
        (
            "A10_batch_to_torch_64t",
            lambda: _A10_prealloc_batch_to_torch_threaded(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A10_64, 64
            ),
        ),
        (
            "A10_batch_to_torch_auto",
            lambda: _A10_prealloc_batch_to_torch_threaded(
                out_dev, mesh_device, A8_dram_staging, A8_host_staging, dest_buf_A10_auto, 0
            ),
        ),
        (
            "A12_copy_device_to_torch",
            lambda: _A12_copy_device_to_torch(out_dev, mesh_device, A8_dram_staging, dest_buf_A12),
        ),
    ]

    results = {}
    for name, fn in variants:
        logger.info(f"  Warming up {name} ...")
        try:
            for _ in range(WARMUP_ITERS):
                _ = fn()
        except Exception as e:
            logger.error(f"  {name} CRASHED in warmup: {type(e).__name__}: {e}")
            results[name] = {
                "untilize_ms": float("nan"),
                "allgather_ms": float("nan"),
                "d2h_ms": float("nan"),
                "cat_ms": float("nan"),
                "total_ms": float("nan"),
                "pcc": float("nan"),
                "shape": "crash",
            }
            continue

        samples = {"untilize_ms": [], "allgather_ms": [], "d2h_ms": [], "cat_ms": [], "total_ms": []}
        last_out = None
        logger.info(f"  Measuring {name}: {MEASURED_ITERS} iters")
        for _ in range(MEASURED_ITERS):
            r = fn()
            for k in samples:
                samples[k].append(r[k])
            last_out = r["out"]
        try:
            if last_out.dtype != torch.bfloat16:
                last_out_cmp = last_out.to(torch.bfloat16)
            else:
                last_out_cmp = last_out
            pcc = _pcc(ref_out, last_out_cmp)
            shape = tuple(last_out.shape)
        except Exception as e:
            pcc = float("nan")
            shape = ("err", str(e))
        results[name] = {k: sum(v) / len(v) for k, v in samples.items()}
        results[name]["pcc"] = pcc
        results[name]["shape"] = shape

    # ---- Reporting ----
    logger.info("")
    logger.info("=" * 130)
    logger.info(
        f"  D2H all-gather results  |  global_batch={global_batch}  |  bf8b={payload_bf8b_mb:.1f} MB / bf16={payload_bf16_mb:.1f} MB"
    )
    logger.info("=" * 130)
    name_w = 32
    col_w = 11
    headers = ["untilize", "allgather", "d2h", "cat", "TOTAL", "pcc"]
    hdr_line = f"{'Variant':<{name_w}}" + "".join(f"| {h:>{col_w}}" for h in headers) + " | shape"
    logger.info(hdr_line)
    logger.info("-" * 130)
    a0_total = results["A0_baseline"]["total_ms"]
    for name, _ in variants:
        r = results[name]
        spd = a0_total / r["total_ms"] if r["total_ms"] > 0 else float("inf")
        logger.info(
            f"{name:<{name_w}}"
            f"| {r['untilize_ms']:>{col_w}.2f}"
            f"| {r['allgather_ms']:>{col_w}.2f}"
            f"| {r['d2h_ms']:>{col_w}.2f}"
            f"| {r['cat_ms']:>{col_w}.2f}"
            f"| {r['total_ms']:>{col_w}.2f}"
            f"| {r['pcc']:>{col_w}.4f}"
            f" | {r['shape']}     ({spd:.2f}x vs A0)"
        )
    logger.info("=" * 130)

    executor.shutdown(wait=False)
