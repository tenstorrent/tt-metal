# SPDX-License-Identifier: Apache-2.0
"""Production-shape fast N-split fused matmul+residual+cross-core-LN.
AttnOut per-DP-shard shape M=49152,K=1024,N=1024 on the 8x8 grid. PCC (bf16) +
traced wall, compared against a ttnn matmul+add+layer_norm baseline."""
import time
import pytest
import torch
from loguru import logger
import ttnn
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_attn_out_ln import fused_attn_out_ln_split


def _wall(fn, dev, iters=20):
    fn(); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); fn(); ttnn.end_trace_capture(dev, tid, cq_id=0); ttnn.synchronize_device(dev)
    for _ in range(3): ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(dev, tid); ts.sort()
    return ts[0], ts[len(ts) // 2]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True, ids=["n1"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 90_000_000, "num_command_queues": 1}], indirect=True)
def test_prod(mesh_device):
    torch.manual_seed(0)
    dev = mesh_device
    M, K, N = 49152, 1024, 1024
    eps = 1e-5
    A = torch.randn(1, 1, M, K) * 0.1
    Wt = torch.randn(1, 1, K, N) * 0.05
    R = torch.randn(1, 1, M, N) * 0.1
    g = torch.randn(N) * 0.1 + 1.0
    b = torch.randn(N) * 0.1
    h = (A.reshape(M, K) @ Wt.reshape(K, N)) + R.reshape(M, N)
    mu = h.mean(-1, keepdim=True); var = h.var(-1, unbiased=False, keepdim=True)
    ref = ((h - mu) / torch.sqrt(var + eps)) * g + b

    def mk(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = mk(g.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    tb_ = mk(b.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())

    # ---- correctness (bf16) ----
    tA, tW, tR = mk(A), mk(Wt), mk(R)
    out16 = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, M, N)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    fused_attn_out_ln_split(tA, tW, tR, tg, tb_, out16, eps=eps, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out16, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].reshape(M, N).float()
    fin = got.isfinite() & ref.isfinite()
    pcc = torch.corrcoef(torch.stack([got[fin].flatten(), ref[fin].flatten()]))[0, 1].item()
    logger.info(f"PROD fused: PCC={pcc:.6f} finite={fin.float().mean().item():.3f}")

    # ---- fused traced wall (bf8) ----
    tA8, tW8, tR8 = mk(A, ttnn.bfloat8_b), mk(Wt, ttnn.bfloat8_b), mk(R, ttnn.bfloat8_b)
    tg8, tb8 = mk(g.reshape(1,1,1,N).expand(1,1,32,N).contiguous(), ttnn.bfloat8_b), mk(b.reshape(1,1,1,N).expand(1,1,32,N).contiguous(), ttnn.bfloat8_b)
    out8 = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, M, N)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    fmn, fmd = _wall(lambda: fused_attn_out_ln_split(tA8, tW8, tR8, tg8, tb8, out8, eps=eps, dtype=ttnn.bfloat8_b), dev)
    logger.info(f"PROD fused traced wall (bf8): min={fmn:.3f} med={fmd:.3f} ms")

    # ---- baseline: matmul + add + layer_norm (bf8) ----
    try:
        ln_prog = None
        def base():
            mm = ttnn.matmul(tA8, tW8, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s = ttnn.add(mm, tR8, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mm)
            o = ttnn.layer_norm(s, epsilon=eps, weight=tg8, bias=tb8, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s)
            return o
        bmn, bmd = _wall(base, dev)
        logger.info(f"BASELINE matmul+add+layer_norm traced wall (bf8): min={bmn:.3f} med={bmd:.3f} ms")
        logger.info(f"==> fused {fmn:.3f} vs baseline {bmn:.3f} ms  (delta {fmn-bmn:+.3f} ms, {(1-fmn/bmn)*100:+.1f}%)")
    except Exception as e:
        logger.warning(f"baseline failed ({e}); reporting fused only")
    assert pcc > 0.90, f"prod PCC too low: {pcc}"
