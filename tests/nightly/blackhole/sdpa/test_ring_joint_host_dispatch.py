# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cache-hit host dispatch time of ring joint SDPA (untraced), the ring counterpart of
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_host_dispatch.py.

Each case calls the op once to build and cache its program, then times calls that hit the program cache. The mesh is
synchronized before every timed call, so the time is the host's work to dispatch one call (argument checks, the cached
full-mesh route, program-cache lookup, runtime-argument and semaphore patching, output allocation). Results are logged
as "SDPA host dispatch <case>: ..." lines; the test only asserts that the op runs.
"""

import statistics
import time

import pytest
import torch
import ttnn
from loguru import logger

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    fence_persistent_buffers,
    open_ring_joint_sdpa_runtime,
)

ITERS = 50
BURST, BURSTS = 16, 20

# name, heads per TP rank, local sequence, head dim, causal, q chunk, k chunk (0: chosen by the op)
CASES = [
    ("ring_joint_wan_like_s2368_nh10", 10, 2368, 128, False, 288, 512),
    ("ring_joint_causal_s1024_nh4", 4, 1024, 128, True, 128, 256),
    ("ring_joint_wan_like_s2368_nh10_op_chosen", 10, 2368, 128, False, 0, 0),
    # The blocking the op chooses for the case above, given explicitly (separates the resolve from the program).
    ("ring_joint_wan_like_s2368_nh10_q224", 10, 2368, 128, False, 224, 512),
]


def _deallocate(out):
    for t in out if isinstance(out, (tuple, list)) else [out]:
        t.deallocate(True)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_ring_joint_host_dispatch(case):
    name, nh_local, s_local, d, causal, q_chunk, k_chunk = case
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    mesh_device = runtime.mesh_device
    try:
        sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
        nh, sq = nh_local * tp, s_local * sp
        sp_dims = [None, None]
        sp_dims[runtime.sp_axis] = 2
        sp_dims[runtime.tp_axis] = 1
        buf_dims = [None, None]
        buf_dims[runtime.tp_axis] = 1

        def to_mesh(t, dims):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            )

        tt_q, tt_k, tt_v = (to_mesh(fa_rand(1, nh, sq, d), sp_dims) for _ in range(3))
        p_buf_k = to_mesh(torch.zeros(1, nh, sq, d), buf_dims)
        p_buf_v = to_mesh(torch.zeros(1, nh, sq, d), buf_dims)
        fence_persistent_buffers(mesh_device)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk
        )

        def call():
            return call_sdpa(
                tt_q,
                tt_k,
                tt_v,
                sq,
                is_causal=causal,
                is_balanced=False,
                p_buf_k=p_buf_k,
                p_buf_v=p_buf_v,
                program_config=program_config,
                compute_kernel_config=None,
                ccl_semaphore_handles=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                sp_axis=runtime.sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                worker_sub_device_id=runtime.worker_sub_device_id,
                ccl_column=runtime.ccl_column,
            )

        out = call()  # program-cache miss
        ttnn.synchronize_device(mesh_device)
        _deallocate(out)
        times = []
        for _ in range(ITERS):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = call()
            times.append((time.perf_counter() - t0) * 1e6)
            ttnn.synchronize_device(mesh_device)
            _deallocate(out)
        bursts = []
        for _ in range(BURSTS):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            outs = [call() for _ in range(BURST)]
            bursts.append((time.perf_counter() - t0) * 1e6 / BURST)
            ttnn.synchronize_device(mesh_device)
            for out in outs:
                _deallocate(out)
        logger.info(
            f"SDPA host dispatch {name}: median {statistics.median(times):.1f} us, min {min(times):.1f} us, "
            f"burst {statistics.median(bursts):.1f} us per call"
        )
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)
