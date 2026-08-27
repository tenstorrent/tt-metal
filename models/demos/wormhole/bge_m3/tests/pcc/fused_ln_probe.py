# SPDX-License-Identifier: Apache-2.0
"""Single-core PCC validation of the fused AttnOut+residual+LN compute kernel."""
import pytest, torch
from loguru import logger
import ttnn
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_attn_out_ln import fused_attn_out_ln_singlecore

@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True, ids=["n1"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
def test_fused_ln_singlecore(mesh_device):
    torch.manual_seed(0)
    dev = mesh_device
    M, K, N = 64, 128, 128  # small single-core case; LN over N=128
    eps = 1e-5
    A = torch.randn(1, 1, M, K) * 0.1
    Wt = torch.randn(1, 1, K, N) * 0.05
    R = torch.randn(1, 1, M, N) * 0.1
    g = torch.randn(N) * 0.1 + 1.0
    b = torch.randn(N) * 0.1
    # torch reference
    h = (A.reshape(M, K) @ Wt.reshape(K, N)) + R.reshape(M, N)
    mu = h.mean(-1, keepdim=True); var = h.var(-1, unbiased=False, keepdim=True)
    ref = ((h - mu) / torch.sqrt(var + eps)) * g + b
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tA, tW, tR = mk(A), mk(Wt), mk(R)
    tg = mk(g.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    tb_ = mk(b.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    import os
    bypass = os.environ.get("BYPASS_LN", "0") == "1"
    if bypass:
        ref = h  # matmul + residual only
    out = fused_attn_out_ln_singlecore(tA, tW, tR, tg, tb_, eps=eps, M_block=1, bypass_ln=bypass)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].reshape(M, N).float()
    logger.info(f"got: shape={tuple(got.shape)} nan={got.isnan().any().item()} min={got.min().item():.4f} max={got.max().item():.4f} std={got.std().item():.4f}")
    logger.info(f"ref: min={ref.min().item():.4f} max={ref.max().item():.4f} std={ref.std().item():.4f}")
    g=got
    for r in [0,1,31,32,33,63]:
        row=g[r]; logger.info(f"row{r}: finite={row.isfinite().float().mean().item():.2f} absmax={row.abs().max().item():.3e} mean={row[row.isfinite()].mean().item():.3f}")
    logger.info(f"got[0,:6]={got[0,:6].tolist()}")
    logger.info(f"ref[0,:6]={ref[0,:6].tolist()}")
    fin = got.isfinite() & ref.isfinite()
    pcc = torch.corrcoef(torch.stack([got[fin].flatten(), ref[fin].flatten()]))[0, 1].item() if fin.any() else float("nan")
    logger.info(f"FUSED-LN single-core PCC = {pcc:.5f}")
    assert pcc > 0.99, f"PCC {pcc} too low"
