"""ttnn.topk with k not a multiple of 32: count the device ops per call.
Env: TK_N (row width, default 4096), TK_KS (comma list of k, default 32,40,64,100), TK_ROWS (default 32), TK_ITERS (default 3).
Run under tracy with -r so the ops report lists every device op:
  python -m tracy -r -m pytest analysis/topk_slice_probe.py -s
"""
import os
import torch
import ttnn


def test_topk_slice_probe(device):
    n = int(os.environ.get("TK_N", "4096"))
    ks = [int(k) for k in os.environ.get("TK_KS", "32,40,64,100").split(",")]
    rows = int(os.environ.get("TK_ROWS", "32"))
    iters = int(os.environ.get("TK_ITERS", "3"))
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, n, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref_v, ref_i = None, None
    for k in ks:
        for it in range(iters):
            ttnn.synchronize_device(device)
            ttnn.tracy.signpost(f"topk_k{k}_it{it}") if hasattr(ttnn, "tracy") else None
            v, i = ttnn.topk(tx, k, dim=-1, largest=True, sorted=True)
            ttnn.synchronize_device(device)
        tv = ttnn.to_torch(v)
        ti = ttnn.to_torch(i)
        exp_v, exp_i = torch.topk(x.float(), k, dim=-1, largest=True, sorted=True)
        assert tv.shape[-1] == k, (k, tv.shape)
        assert ti.shape[-1] == k, (k, ti.shape)
        assert torch.allclose(tv.float(), exp_v, atol=1e-2, rtol=1e-2), k
        print(f"[topk_slice_probe] k={k} shapes {tuple(tv.shape)} {tuple(ti.shape)} ok", flush=True)
