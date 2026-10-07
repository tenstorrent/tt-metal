# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-card moe_compute timing control (bfp4 weights, the only dtype this build supports; upstream harness at MiMo dims K=4096 N=2048, 8 experts, T=32,
top-8, compute_only): runs the harness once for correctness (PCC printed by it), then times the op in a TRACE of 10 back-to-back calls replayed 30x
(device-side compute time of the experts stage on ONE chip; compare with the unified op's 'experts' stage and the 214 MB / 450 GB/s bfp8 weight-read floor).
Lower bound on what the ring moe_compute path buys for the expert stage; bfp8 weights would read 2x the bytes."""
import time

import pytest
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card as up
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 50000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens", [32, 16, 8])
def test_moe_compute_timing(mesh_device, mesh_shape, tokens, monkeypatch):
    orig = ttnn.experimental.moe_compute
    state = {}

    def shim(*a, **kw):
        outs = orig(*a, **kw)
        if "done" in state:
            return outs
        state["done"] = True
        ttnn.synchronize_device(mesh_device)

        def free(o):
            for i in (0, 1, 2, 4):
                ttnn.deallocate(o[i])

        free(outs)
        for _ in range(3):
            free(orig(*a, **kw))
        ttnn.synchronize_device(mesh_device)
        n = 10
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(n):
            free(orig(*a, **kw))
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        reps = 30
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.time() - t0) / reps / n * 1e3
        print(
            f"RESULT moe_compute single card bfp4 compute_only T={tokens} K=4096 N=2048 8 experts: {dt:.3f} ms per call (traced, host-timed replay of {n} calls)",
            flush=True,
        )
        ttnn.release_trace(mesh_device, tid)
        return orig(*a, **kw)

    monkeypatch.setattr(ttnn.experimental, "moe_compute", shim)
    ring_n = effective_matmul_ring_size(mesh_device)
    up._run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=8,
        tokens_per_device=tokens,
        selected_experts_k=8,
        N=2048,
        hidden_size=4096,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(4096, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=True,
    )
