# SPDX-License-Identifier: Apache-2.0
"""Sequential-fusion CALIBRATION at BGE B6/S8192/D1024 (JIT Priority 1), TRACED.

Go/no-go gate for descriptor fusion: does removing ONE program boundary between
two LayerNorms save repeatable TRACED wall (identical kernels + DRAM traffic)?
  A = two normal ttnn.layer_norm calls              (2 programs)
  C = one Sequential(ln1, ln2) fused, built once     (1 program)
Both trace-captured and replayed; report traced-replay wall + PCC.
"""
import os

os.environ["TT_METAL_ENABLE_PARALLEL_SEQUENTIAL"] = "1"

import time

import pytest
import torch
from loguru import logger

import ttnn

B, S, D = 6, 8192, 1024
N_ITERS = 30


def _torch_ln(x, w, eps=1e-5):
    return torch.nn.functional.layer_norm(x.float(), (D,), w.float(), None, eps)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000, "num_command_queues": 1}], indirect=True)
def test_seq_fusion_calib(mesh_device):
    from models.experimental.ops.descriptors.fusion import Sequential
    from models.experimental.ops.descriptors.normalization import layer_norm

    torch.manual_seed(0)
    x = torch.randn(B, 1, S, D, dtype=torch.bfloat16)
    w0 = torch.ones(D, dtype=torch.bfloat16)
    w1 = torch.ones(D, dtype=torch.bfloat16)

    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tw0 = ttnn.from_torch(
        w0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tw1 = ttnn.from_torch(
        w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    grid = mesh_device.compute_with_storage_grid_size()
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    ref = _torch_ln(_torch_ln(x, w0).to(torch.bfloat16), w1)

    def pcc(t):
        got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
        return torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    def trace_bench(name, capture_body):
        # warmup
        out = capture_body()
        ttnn.synchronize_device(mesh_device)
        p = pcc(out)
        # capture trace
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = capture_body()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ts = []
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(mesh_device, tid)
        ts.sort()
        logger.info(f"{name:<28} TRACED wall min={ts[0]:.4f} med={ts[len(ts)//2]:.4f} ms  pcc={p:.5f}")

    # A: two stock ttnn.layer_norm (reference only; conflates op-kind w/ boundary)
    def body_a():
        g = ttnn.layer_norm(tx, weight=tw0, epsilon=1e-5)
        o = ttnn.layer_norm(g, weight=tw1, epsilon=1e-5)
        return o

    trace_bench("A_two_ttnn_layernorm", body_a)

    # B: two SEPARATE descriptor launches (same kernels as C, but 2 programs).
    #    This is the correct control: B-vs-C isolates PURE boundary removal.
    b_op1 = layer_norm.layer_norm(tx, core_range_set=cr, weight=tw0, epsilon=1e-5)
    b_ln1 = Sequential(b_op1).build(mesh_device)
    b_op2 = layer_norm.layer_norm(b_ln1.output_tensors[0], core_range_set=cr, weight=tw1, epsilon=1e-5)
    b_ln2 = Sequential(b_op2).build(mesh_device)

    def body_b():
        b_ln1.launch()
        b_ln2.launch()
        return b_ln2.output_tensors[0]

    try:
        trace_bench("B_two_descriptor_launches", body_b)
    except Exception as e:
        logger.error(f"B_two_descriptor_launches FAILED: {str(e)[:250]}")

    # C: Sequential fused, built ONCE outside the timed region (1 program).
    ln1 = layer_norm.layer_norm(tx, core_range_set=cr, weight=tw0, epsilon=1e-5)
    ln2 = layer_norm.layer_norm(ln1.output_tensors[0], core_range_set=cr, weight=tw1, epsilon=1e-5)
    fused = Sequential(ln1, ln2).build(mesh_device)

    def body_c():
        fused.launch()
        return fused.output_tensors[0]

    try:
        trace_bench("C_sequential_fused", body_c)
    except Exception as e:
        logger.error(f"C_sequential_fused FAILED: {str(e)[:250]}")

    # Program-count proof: B kernels vs C kernels
    try:
        logger.info(
            f"PROGRAM-COUNT: B_ln1 kernels={len(b_ln1.descriptor.kernels)} "
            f"B_ln2 kernels={len(b_ln2.descriptor.kernels)} "
            f"C_fused kernels={len(fused.descriptor.kernels)}"
        )
    except Exception as e:
        logger.error(f"program-count probe failed: {str(e)[:150]}")
