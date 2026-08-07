"""Is emb in {6144,7168} a KERNEL limit or just declared SUPPORTED scope?"""
import sys, torch, ttnn
import torch.nn.functional as F
import ttnn.operations.moe_fused_swiglu.moe_fused_swiglu  # noqa: F401

M = sys.modules["ttnn.operations.moe_fused_swiglu.moe_fused_swiglu"]
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

HIDDEN, CAP, COUNT = 2048, 1024, 256
NG, NL, LID, GID = 256, 8, 3, 137
GRID = (11, 8)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    print("declared SUPPORTED[emb] =", M.SUPPORTED["emb"], flush=True)
    for emb in (4096, 5120, 8192, 3584):
        M.SUPPORTED["emb"] = sorted(set(M.SUPPORTED["emb"]) | {emb})  # widen the DECLARATION only
        torch.manual_seed(0)
        wg = torch.randn(emb, HIDDEN, dtype=torch.bfloat16) * 0.02
        wu = torch.randn(emb, HIDDEN, dtype=torch.bfloat16) * 0.02
        wd = torch.randn(HIDDEN, emb, dtype=torch.bfloat16) * 0.02
        xt = torch.zeros(CAP, emb, dtype=torch.bfloat16)
        xt[:COUNT] = torch.randn(COUNT, emb, dtype=torch.bfloat16)
        ref = (F.silu(xt[:COUNT].float() @ wg.float()) * (xt[:COUNT].float() @ wu.float())) @ wd.float()
        try:
            x = ttnn.from_torch(
                xt.reshape(1, 1, CAP, emb),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gu, dm = weight_memory_configs(device, emb, HIDDEN, core_grid=GRID)
            w = [
                ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                for t, mc in ((wg, gu), (wu, gu), (wd, dm))
            ]
            c = torch.zeros(NG, dtype=torch.int32)
            c[GID] = COUNT
            tc = ttnn.from_torch(c, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ix = torch.tensor([(11 + 37 * i) % NG for i in range(NL)], dtype=torch.int32)
            ix[LID] = GID
            ti = ttnn.from_torch(ix, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            out = M.moe_fused_swiglu(x, w[0], w[1], w[2], tc, ti, LID, core_grid=GRID)
            got = ttnn.to_torch(out)[0, 0, :COUNT]
            print(f"emb={emb:5d}  RAN   PCC={pcc(ref, got):.6f}", flush=True)
            for t in (x, *w, tc, ti, out):
                ttnn.deallocate(t)
        except Exception as e:
            print(f"emb={emb:5d}  REFUSED  {type(e).__name__}: {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(device)
