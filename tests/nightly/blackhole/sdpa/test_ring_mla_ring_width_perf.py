# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Paired cost of ringing ring_mla over the whole 32-device mesh instead of the SP axis.

Every arm runs the same kimi50k final chunk -- 5120 Q rows against a 56320 prefix, Q sharded over
SP alone -- on the same (sp, tp) mesh. Only the KV sharding, the ring width and k_chunk_size move:

  arm                 ring  kv/device  region  k_chunk   what it isolates
  sp_ring_8-k640         8       7040     640      640   baseline, region and k_chunk coincide
  mesh_ring_32-k640     32       1760     160      640   the full-mesh ring as proposed
  mesh_ring_32-k160     32       1760     160      160   ring width alone, k_chunk matched to region

Measured 2026-09-15, 8x4 Blackhole Galaxy, wall clock, 10 iters after 2 warmups (min / median ms):

  sp_ring_8-k640     7.296 / 7.935          mesh_ring_32-k640   9.846 / 10.115   1.35x / 1.27x
  mesh_ring_32-k160  9.703 / 10.284         i.e. 1.33x / 1.30x

The two 32-ring arms agree within the 4-11% run-to-run spread, so the cost tracks ring width, not
the 4:1 k_chunk padding. One depth only; a second point is needed before reading this as a curve.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from ttnn import Topology

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_CHUNK_SIZE,
    CHUNKED_PREFILL_HEADS_PER_RING,
    CHUNKED_PREFILL_N_CHUNKS,
    CHUNKED_PREFILL_PER_DEVICE_CHUNK,
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
)

WARMUP = 2
ITERS = 10
D_Q = D_K = 576
D_V = 512
Q_CHUNK = 32


def _time_ring_mla(runtime, tt_q, tt_kv, gathered_kv, *, logical_n, cluster_axis, k_chunk):
    mesh_device = runtime.mesh_device
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=runtime.sdpa_compute_grid,
        q_chunk_size=Q_CHUNK,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )

    def one():
        out, _ = ttnn.transformer.ring_mla(
            tt_q,
            tt_kv,
            persistent_output_buffer_kv=gathered_kv,
            head_dim_v=D_V,
            logical_n=logical_n,
            program_config=program_config,
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=Topology.Ring,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
        )
        return out

    for _ in range(WARMUP):
        ttnn.deallocate(one())
    ttnn.synchronize_device(mesh_device)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = one()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    samples.sort()
    return samples


def _report(tag, samples, region_rows, k_chunk):
    mean = sum(samples) / len(samples)
    logger.info(
        f"[ring-width] {tag}: min {samples[0]:.3f} ms  median {samples[len(samples)//2]:.3f} ms  "
        f"mean {mean:.3f} ms  max {samples[-1]:.3f} ms  (n={len(samples)}, "
        f"region {region_rows} rows vs k_chunk {k_chunk} -> pad {max(k_chunk / region_rows, 1.0):.2f}x)"
    )
    return mean


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "arm,k_chunk",
    [
        ("sp_ring_8", 640),  # baseline: region 640, k_chunk 640 -> exact fit
        ("mesh_ring_32", 640),  # as proposed: region 160, k_chunk 640 -> 4:1 pad
        ("mesh_ring_32_k160", 160),  # region-matched: isolates ring width from the padding
    ],
    ids=["sp_ring_8-k640", "mesh_ring_32-k640", "mesh_ring_32-k160"],
)
def test_ring_mla_ring_width_cost(arm, k_chunk):
    if not MESH_CONFIG.is_galaxy:
        pytest.skip("needs the 8x4 Galaxy")
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    chunk_global = CHUNKED_PREFILL_CHUNK_SIZE  # 5120
    logical_n = chunk_global * CHUNKED_PREFILL_N_CHUNKS  # 56320
    nhq, nhk = CHUNKED_PREFILL_HEADS_PER_RING, 1
    b = 1

    full_mesh = arm.startswith("mesh_ring_32")
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG, full_mesh=True, sp_outer=True)
    try:
        mesh_device = runtime.mesh_device
        assert tuple(mesh_device.shape) == (sp, tp), tuple(mesh_device.shape)

        torch.manual_seed(2026)
        # Q is the final chunk, sharded over SP alone in BOTH arms.
        q_global = fa_rand(b, nhq, chunk_global, D_Q)
        tt_q = ttnn.from_torch(
            q_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None]),
        )
        assert tt_q.shape[2] == chunk_global // sp == CHUNKED_PREFILL_PER_DEVICE_CHUNK

        kv_global = fa_rand(b, nhk, logical_n, D_K)
        if full_mesh:
            ranks, region = sp * tp, chunk_global // (sp * tp)  # 32, 160
            kv_local = logical_n // ranks  # 1760
            # Block-cyclic: position (chunk, dev, off) -> row chunk*region + off on device dev.
            n_chunks = logical_n // chunk_global
            kv_host = (
                kv_global.reshape(b, nhk, n_chunks, ranks, region, D_K)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, nhk, ranks * kv_local, D_K)
                .contiguous()
            )
            kv_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            cluster_axis = None
        else:
            ranks, region = sp, chunk_global // sp  # 8, 640
            kv_local = logical_n // ranks  # 7040
            kv_host = kv_global
            kv_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])
            cluster_axis = 0  # sp_outer puts SP on axis 0

        tt_kv = ttnn.from_torch(
            kv_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=kv_mapper
        )
        assert tt_kv.shape[2] == kv_local, f"{tt_kv.shape[2]} != {kv_local}"

        gathered_kv = ttnn.from_torch(
            torch.zeros(b, nhk, ranks * kv_local, D_K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        logger.info(
            f"[ring-width] {arm}: mesh={sp}x{tp} ring={ranks} kv_local={kv_local} region={region} "
            f"q_local={tt_q.shape[2]} logical_n={logical_n} k_chunk={k_chunk}"
        )
        samples = _time_ring_mla(
            runtime, tt_q, tt_kv, gathered_kv, logical_n=logical_n, cluster_axis=cluster_axis, k_chunk=k_chunk
        )
        _report(arm, samples, region, k_chunk)
    finally:
        close_ring_joint_sdpa_runtime(runtime)
