# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Dense matmuls of the V4.1 attention on ``tt/v41/attention.dense_program_config`` (bead 8y7.9.7, G2 gap #10).

Per chip on LoudBox 2x4 (rows = chunk / sp; tiles [M, K] x [K, N], M = 80 at chunk 5120, 32 at 2048):
wq_a [M, 40] x [40, 40], wkv [M, 40] x [40, 16], wq_b [M, 40] x [40, 256], wo_a batched 2 x [M, 128] x [128, 32]; bf16
activations, bfp8 weights, HiFi2 (``DENSE_COMPUTE_CONFIG`` = ttnn's default for these inputs, checked bit-exact).
Each K block of ``IN0_BLOCK_W`` is timed by trace replay (``ITERS`` back-to-back calls per trace, min over replays of
the device-synchronized wall per call) and compared with the fp32 torch product of the same device inputs: bf16
partial sums round more with wider K blocks. Contract: the attention's chosen K block (``DENSE_IN0_BLOCK_W``) is
faster than ttnn's default config and within ``ACCURACY_SLACK`` of its PCC. Rejected alternative (measured): q heads
straight from wq_b as a batched in0-reuse 1D matmul (16 x [80, 40] x [40, 16]) took 880-1600 us vs 382 + 198 us for
wq_b + nlp_create_qkv_heads.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41.attention import DENSE_COMPUTE_CONFIG, DENSE_IN0_BLOCK_W, dense_program_config
from tests.ttnn.utils_for_testing import comp_pcc

MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(trace_region_size=64 << 20),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]
ITERS = 10
M = 2560


# name -> (in0 shape per M, in1 shape): per chip, 2x4 mesh
CASES = {
    "wq_a": (lambda m: (1, 1, m, 1280), (1, 1, 1280, 1280)),
    "wkv": (lambda m: (1, 1, m, 1280), (1, 1, 1280, 512)),
    "wq_b": (lambda m: (1, 1, m, 1280), (1, 1, 1280, 8192)),
    "wo_a": (lambda m: (1, 2, m, 4096), (1, 2, 4096, 1024)),
}
ROWS = {"chunk5120": 2560, "chunk2048": 1024}  # rows per chip (chunk / sp)
IN0_BLOCK_W = (2, 4, 5, 8, 16)
ACCURACY_SLACK = 2e-5  # PCC vs fp32 the chosen config may lose against the default


def _timed_us(mesh_device, fn, replays: int = 5) -> float:
    """Min over ``replays`` trace replays of the per-call wall (``ITERS`` calls per trace)."""
    fn()  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(ITERS):
        fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    best = float("inf")
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        best = min(best, (time.perf_counter() - t0) * 1e6 / ITERS)
    ttnn.release_trace(mesh_device, tid)
    return best


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("rows", list(ROWS))
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_attention_matmul_configs(mesh_device, device_params, case, rows):
    a_shape, b_shape = CASES[case]
    a_shape = a_shape(ROWS[rows])
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    a = ttnn.from_torch(
        torch.randn(a_shape).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    b = ttnn.from_torch(
        (torch.randn(b_shape) / b_shape[-2] ** 0.5).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    exact = torch.matmul(first(a), first(b))
    default = first(ttnn.linear(a, b))
    hifi2 = DENSE_COMPUTE_CONFIG
    assert torch.equal(default, first(ttnn.linear(a, b, compute_kernel_config=hifi2))), "default is not HiFi2"
    pcc = lambda x, y: comp_pcc(x, y, 0.0)[1]
    results = {"default": (_timed_us(mesh_device, lambda: ttnn.linear(a, b)), pcc(exact, default))}
    m, k, n = a_shape[-2], b_shape[-2], b_shape[-1]
    for w in IN0_BLOCK_W:
        cfg = dense_program_config(m, k, n, w)
        if cfg is None:
            continue
        run = lambda cfg=cfg: ttnn.linear(a, b, program_config=cfg, compute_kernel_config=hifi2)
        results[f"k{w}"] = (_timed_us(mesh_device, run), pcc(exact, first(run())))
    base, base_pcc = results["default"]
    for name, (us, vs_exact) in sorted(results.items(), key=lambda kv: kv[1][0]):
        logger.info(f"MATMUL {case} {rows} {name}: {us:.1f} us ({base / us:.2f}x) pcc vs fp32 {vs_exact:.6f}")
    chosen = f"k{DENSE_IN0_BLOCK_W[case]}"
    us, vs_exact = results[chosen]
    assert vs_exact >= base_pcc - ACCURACY_SLACK, (chosen, results)
    assert us < base, (chosen, results)
