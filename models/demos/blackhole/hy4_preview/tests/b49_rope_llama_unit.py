"""B49 unit (1x8 slot): RopeDev.apply with HY4_B49_ROPE_LLAMA=1 (rotary_embedding_llama) vs the stock matmul + multiply + add rope, x [1,1,512,64] and [1,8,512,64] bf16, real rope tables
(theta 1e7, adjacent pairs) at starts 0 / 65536 / 1M-512, amplitude 0.1..30. PCC, max abs, bf16 ulp stats, repeat equality. RES lines."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_common import RopeDev, compute_cfg, upload


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def snap(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main():
    full, md = open_row()
    try:
        rope = RopeDev(md, 64, 1.0e7, True)
        ckc = compute_cfg(md, "hifi4", True)
        g = torch.Generator().manual_seed(1)
        for start in (0, 65536, 1048576 - 512):
            cos, sin = rope.tables(list(range(start, start + 512)))
            for H, amp in ((1, 0.1), (8, 1.0), (8, 30.0)):
                xh = (torch.randn(1, H, 512, 64, generator=g) * amp).bfloat16()
                x = upload(md, xh, ttnn.bfloat16)
                os.environ["HY4_B49_ROPE_LLAMA"] = "0"
                ref = snap(rope.apply(x, cos, sin, ckc))
                os.environ["HY4_B49_ROPE_LLAMA"] = "1"
                o1 = snap(rope.apply(x, cos, sin, ckc))
                o2 = snap(rope.apply(x, cos, sin, ckc))
                ulp = (o1.bfloat16().view(torch.int16).int() - ref.bfloat16().view(torch.int16).int()).abs()
                res(
                    event="rope",
                    start=start,
                    H=H,
                    amp=amp,
                    pcc=pcc(o1, ref),
                    maxabs=float((o1 - ref).abs().max()),
                    refmax=float(ref.abs().max()),
                    frac_ulp_ge1=float((ulp >= 1).float().mean()),
                    frac_ulp_ge2=float((ulp >= 2).float().mean()),
                    repeat_equal=bool(torch.equal(o1, o2)),
                )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
