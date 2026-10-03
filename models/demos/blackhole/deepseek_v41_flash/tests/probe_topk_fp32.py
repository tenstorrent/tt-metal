"""Single device: can ttnn.topk rank fp32 values exactly (k=6 of 384), and what does it cost?"""
import time

import torch

import ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
# values that differ below bf16 resolution: base ~10 + tiny offsets
x = (10.0 + torch.randn(32, 384) * 0.5).float()
x[:, 5] = x[:, 7] + 1e-3  # near-ties
ref_idx = x.topk(6, dim=-1).indices
for dtype, name in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    try:
        t = ttnn.from_torch(x.reshape(1, 1, 32, 384), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        vals, idx = ttnn.topk(t, k=6, dim=-1, largest=True, sorted=True)
        idx = ttnn.to_torch(idx).reshape(32, 6).long()
        ok = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(idx, ref_idx))
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.topk(t, k=6, dim=-1, largest=True, sorted=True)
        ttnn.synchronize_device(dev)
        print(
            f"RESULT topk {name}: exact set match {ok}/32 rows, {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms/call (eager)"
        )
    except Exception as e:
        print(f"RESULT topk {name}: FAILED {type(e).__name__}: {str(e)[:200]}")
ttnn.close_device(dev)
