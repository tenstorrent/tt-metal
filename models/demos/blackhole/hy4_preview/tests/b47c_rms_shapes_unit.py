"""B47c shape guard unit test (1x8 slot, MESH_DEVICE=1x8): rms_any / rms_cat_any for T in {1,2,4,16,17,32,512} vs the stock chain (no assertion; T > 16 takes the stock fallback), the kernel itself rejects T > 16 with ValueError,
and Hy4MTP-style fp32/bf16 + residual + out_x variants. RES lines: ne = differing bf16 elements, max_ulp."""
import json

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_rms_fused as RF
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

HID, EPS = 6144, 1e-5


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def ref6(x32, wsc):
    return RF._stock(x32, wsc, EPS)


def cmp(name, a, b, T):
    a, b = a[..., :T, :].float(), b[..., :T, :].float()
    ulp = (a - b).abs() / (b.abs().clamp_min(1e-30) * 2.0**-8)
    res(case=name, T=T, n=a.numel(), ne=int((a != b).sum()), max_ulp=float(ulp.max()))


def main():
    full, md = open_row()
    try:
        g = torch.Generator().manual_seed(11)
        w = torch.randn(1, 1, 1, HID, generator=g) * 0.5 + 1.0
        for T in (1, 2, 4, 16, 17, 32, 512):
            wsc = upload(md, (w * HID**0.5).expand(1, 1, T, HID).contiguous(), ttnn.float32)
            xh = torch.randn(1, 1, T, HID, generator=g)
            rh = torch.randn(1, 1, T, HID, generator=g) * 0.7
            xb, rb = upload(md, xh.bfloat16(), ttnn.bfloat16), upload(md, rh.bfloat16(), ttnn.bfloat16)
            x32 = upload(md, xh.float(), ttnn.float32)
            ref = RF._stock(ttnn.typecast(xb, ttnn.float32), wsc, EPS)
            cmp("A", d0(RF.rms_any(xb, wsc, EPS)), d0(ref), T)
            sref = ttnn.add(x32, ttnn.typecast(rb, ttnn.float32))
            out, xn = RF.rms_any(x32, wsc, EPS, res=rb, out_x=True)
            cmp("B", d0(out), d0(RF._stock(sref, wsc, EPS)), T)
            res(case="B_xnew", T=T, ne=int((d0(xn)[..., :T, :] != d0(sref)[..., :T, :]).sum()))
            cat = RF.rms_cat_any(xb, wsc, rb, wsc, EPS)
            rc = ttnn.concat([RF._stock(xb, wsc, EPS), RF._stock(rb, wsc, EPS)], dim=3)
            cmp("cat", d0(cat), d0(rc), T)
            if T > 16:
                try:
                    RF.rms_fused(xb, wsc, EPS)
                    res(case="reject", T=T, ok=False)
                except ValueError as ex:
                    res(case="reject", T=T, ok=True, msg=str(ex)[:60])
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
