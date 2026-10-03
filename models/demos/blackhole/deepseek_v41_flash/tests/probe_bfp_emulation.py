"""Validate reference/bfp_emulation.py against the device's own bfp4_b / bfp8_b roundtrip (single device)."""
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.bfp_emulation import bfp_roundtrip
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _fp4_expert, _Shards

dev = ttnn.open_device(device_id=0)
rt = (
    lambda w, dt: ttnn.to_torch(
        ttnn.from_torch(w.float().reshape(1, 1, *w.shape[-2:]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    )
    .reshape(w.shape)
    .float()
)
sh = _Shards()
tests = {
    "randn": torch.randn(64, 256),
    "randn-scaled-rows": torch.randn(64, 256) * torch.exp2(torch.randint(-6, 6, (64, 1)).float()),
    "real expert w1 (layer 2, e0) [5120,2304]": _fp4_expert(sh, "layers.2.ffn.experts.0.w1", torch.bfloat16)[:512],
    "real expert w2 (layer 2, e0) [2304,5120]": _fp4_expert(sh, "layers.2.ffn.experts.0.w2", torch.bfloat16)[:256],
}
for name, w in tests.items():
    for dt, mb in ((ttnn.bfloat4_b, 3), (ttnn.bfloat8_b, 7)):
        ref = bfp_roundtrip(w.bfloat16(), mb)
        got = rt(w.bfloat16(), dt)
        diff = (ref - got).abs()
        print(
            f"RESULT {name:44s} {str(dt).split('.')[-1]:10s} max|emu-dev| {diff.max():.3e}  mismatching elems {(diff > 0).sum().item()}/{diff.numel()}"
        )
ttnn.close_device(dev)
