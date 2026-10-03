"""Single device: does a bfloat4_b matmul have a systematic gain at LoFi vs HiFi2/HiFi4?  Real expert weights + real tokens;
oracle = float64 matmul with the bit-exact emulated bfp4 weights."""
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.bfp_emulation import bfp_roundtrip
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _fp4_expert, _Shards

dev = ttnn.open_device(device_id=0)
sh = _Shards()
d = torch.load("/mnt/tt-data/ssinghal/dsv4-chain-e/ffn_inputs_1.pt")
x = d["x"][:16].float()
e = int(d["idx"][0, 0])
W = _fp4_expert(sh, f"layers.1.ffn.experts.{e}.w1", torch.float32)  # [5120, 2304]
Wq = bfp_roundtrip(W, 3)
oracle = x.double() @ Wq.double()
xt = ttnn.from_torch(x.bfloat16().reshape(1, 1, 16, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for wdt, nm in ((ttnn.bfloat4_b, "bfp4"), (ttnn.bfloat8_b, "bfp8")):
    wt = ttnn.from_torch(W.reshape(1, 1, 5120, 2304), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
        for acc in (False, True):
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=acc, packer_l1_acc=False
            )
            y = ttnn.to_torch(ttnn.matmul(xt, wt, compute_kernel_config=ckc)).reshape(16, 2304).double()
            ref = oracle if wdt == ttnn.bfloat4_b else (x.double() @ bfp_roundtrip(W, 7).double())
            gain = float((y * ref).sum() / (ref * ref).sum())
            r = float((y - ref).norm() / ref.norm())
            print(
                f"RESULT {nm} weights  {str(fid).split('.')[-1]:6s} fp32acc={acc!s:5s} gain {gain:.4f}  rel err vs emulated-weight oracle {r:.4f}"
            )
ttnn.close_device(dev)
