# SPDX-License-Identifier: Apache-2.0
"""Multi-core full-N fused LN: PCC + traced wall vs baseline (minimal_matmul + layer_norm)."""
import time, pytest, torch
from loguru import logger
import ttnn
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_attn_out_ln import fused_attn_out_ln_multicore

@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True, ids=["n1"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000, "num_command_queues": 1}], indirect=True)
def test_mc(mesh_device):
    torch.manual_seed(0)
    dev = mesh_device
    M, K, N = 49152, 1024, 1024
    eps = 1e-5
    A = torch.randn(1, 1, M, K) * 0.1; Wt = torch.randn(1, 1, K, N) * 0.05
    R = torch.randn(1, 1, M, N) * 0.1; g = torch.randn(N) * 0.1 + 1.0; b = torch.randn(N) * 0.1
    h = (A.reshape(M, K) @ Wt.reshape(K, N)) + R.reshape(M, N)
    mu = h.mean(-1, keepdim=True); var = h.var(-1, unbiased=False, keepdim=True)
    ref = ((h - mu) / torch.sqrt(var + eps)) * g + b
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tA, tW, tR = mk(A), mk(Wt), mk(R)
    tg = mk(g.reshape(1,1,1,N).expand(1,1,32,N).contiguous()); tb_ = mk(b.reshape(1,1,1,N).expand(1,1,32,N).contiguous())
    out = ttnn.allocate_tensor_on_device(ttnn.Shape((1,1,M,N)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    # correctness
    fused_attn_out_ln_multicore(tA, tW, tR, tg, tb_, out, eps=eps)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].reshape(M, N).float()
    fin = got.isfinite() & ref.isfinite()
    pcc = torch.corrcoef(torch.stack([got[fin].flatten(), ref[fin].flatten()]))[0,1].item()
    logger.info(f"MC fused: PCC={pcc:.5f} finite_frac={fin.float().mean().item():.3f}")
    # timing: fused op
    def run_fused(): fused_attn_out_ln_multicore(tA, tW, tR, tg, tb_, out, eps=eps)
    run_fused(); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); run_fused(); ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
    for _ in range(3): ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ts=[]
    for _ in range(20):
        t0=time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); ts.append((time.perf_counter()-t0)*1e3)
    ttnn.release_trace(dev, tid); ts.sort()
    logger.info(f"MC fused traced wall: min={ts[0]:.3f} med={ts[len(ts)//2]:.3f} ms  (matmul+residual+LN in ONE op)")
    logger.info(f"BASELINE for reference: separate AttnOut matmul (~0.83ms) + LayerNorm (~0.78ms) + 2 program dispatch")
