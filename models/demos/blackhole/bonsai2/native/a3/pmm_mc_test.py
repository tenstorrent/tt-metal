"""Small pmm multicast check: gdn_out L0 at S=32, mc vs non-mc vs float64. Run under `timeout`."""
import os, numpy as np, torch, ttnn
from models.demos.blackhole.bonsai2.native import cpu_ref
from models.demos.blackhole.bonsai2.native.a3 import linear as al
from models.demos.blackhole.bonsai2.native.a3.pmm import pmm
from models.demos.blackhole.bonsai2.native.a3.mlp import Shared
from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    sh = Shared(dev); go = al.TernaryLinear(sh, "blk.0.ssm_out.weight", os.path.join(os.environ["TT_CACHE_PATH"], "a3_linear"))
    t = read_ternary(sh.gguf, "blk.0.ssm_out.weight")
    w = t.codes.astype(np.float64) * np.repeat(t.scales.astype(np.float64), 128, axis=1)
    s = int(os.environ.get("S", "32"))
    xh = torch.randn(1, 1, s, 6144).bfloat16()
    x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ref = xh.float().numpy().reshape(s, -1) @ w.T
    for mc in ("0", "1"):
        os.environ["BONSAI_PMM_MC"] = mc
        y = ttnn.to_torch(pmm(x, go.w, go.s, 6144, 5120)).float().numpy().reshape(s, -1)
        ttnn.synchronize_device(dev)
        print(f"MC={mc} finite={np.isfinite(y).all()} pcc={cpu_ref.pcc(y, ref):.6f}", flush=True)
finally:
    ttnn.close_device(dev)
