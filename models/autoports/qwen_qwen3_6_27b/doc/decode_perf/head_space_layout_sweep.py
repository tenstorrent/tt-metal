"""Go/no-go for doing gated-delta head-space work in a packed layout.

Head space is currently [batch, heads, 1, dim] = [32, 12, 1, 128], which tile
pads M from 1 to 32: every elementwise op touches 1.57 M elements to use 49 K.
[1, 1, 384, 128] (slot-major, head-minor) is tile aligned with no padding and
keeps per-(slot, head) reductions on the last axis, so the same arithmetic fits.

But the state matmul needs one M tile per (slot, head), so the packed form has to
be converted around it. This measures both sides: what the elementwise ops cost
in each layout, and what a conversion costs. If a conversion costs more than the
ops it saves, the repack is a wash.
"""

import time

import torch

import ttnn

B, H, D = 32, 12, 128
G = B * H


def timed(fn, device, iters=30):
    fn()
    ttnn.synchronize_device(device)
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return 1e6 * (time.perf_counter() - started) / iters


def main():
    d = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.DRAM_MEMORY_CONFIG
        mk = dict(device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc)
        padded = ttnn.from_torch(torch.randn(B, H, 1, D, dtype=torch.bfloat16), **mk)
        padded2 = ttnn.from_torch(torch.randn(B, H, 1, D, dtype=torch.bfloat16), **mk)
        pack = ttnn.from_torch(torch.randn(1, 1, G, D, dtype=torch.bfloat16), **mk)
        pack2 = ttnn.from_torch(torch.randn(1, 1, G, D, dtype=torch.bfloat16), **mk)
        scal_pad = ttnn.from_torch(torch.randn(B, H, 1, 1, dtype=torch.bfloat16), **mk)
        scal_pack = ttnn.from_torch(torch.randn(1, 1, G, 1, dtype=torch.bfloat16), **mk)
        gnorm = ttnn.from_torch(torch.randn(1, 1, 1, D, dtype=torch.bfloat16), **mk)
        wide = ttnn.from_torch(torch.randn(1, 1, B, H * D, dtype=torch.bfloat16), **mk)

        def bench(label, fn):
            try:
                us = timed(fn, d)
                print(f"  {label:52s} {us:8.1f} us", flush=True)
                return us
            except Exception as exc:
                print(f"  {label:52s} FAILED {type(exc).__name__}: {str(exc)[:110]}", flush=True)
                return None

        print("elementwise, per-head padded [32,12,1,128] (1.57 M elements):")
        a = bench("multiply(x, y)", lambda: ttnn.deallocate(ttnn.multiply(padded, padded2, memory_config=mc)))
        b = bench(
            "multiply(x, per-head scalar)", lambda: ttnn.deallocate(ttnn.multiply(padded, scal_pad, memory_config=mc))
        )
        c = bench("sum(x, dim=-1, keepdim)", lambda: ttnn.deallocate(ttnn.sum(padded, dim=-1, keepdim=True)))
        e = bench(
            "rms_norm(x, weight)",
            lambda: ttnn.deallocate(ttnn.rms_norm(padded, epsilon=1e-6, weight=gnorm, memory_config=mc)),
        )
        f = bench("silu(x)", lambda: ttnn.deallocate(ttnn.silu(padded, memory_config=mc)))
        padded_total = sum(v for v in (a, b, c, e, f) if v)

        print("elementwise, packed [1,1,384,128] (49 K elements):")
        a2 = bench("multiply(x, y)", lambda: ttnn.deallocate(ttnn.multiply(pack, pack2, memory_config=mc)))
        b2 = bench(
            "multiply(x, per-row scalar)", lambda: ttnn.deallocate(ttnn.multiply(pack, scal_pack, memory_config=mc))
        )
        c2 = bench("sum(x, dim=-1, keepdim)", lambda: ttnn.deallocate(ttnn.sum(pack, dim=-1, keepdim=True)))
        e2 = bench(
            "rms_norm(x, weight)",
            lambda: ttnn.deallocate(ttnn.rms_norm(pack, epsilon=1e-6, weight=gnorm, memory_config=mc)),
        )
        f2 = bench("silu(x)", lambda: ttnn.deallocate(ttnn.silu(pack, memory_config=mc)))
        pack_total = sum(v for v in (a2, b2, c2, e2, f2) if v)

        print("conversions between the two:")
        bench("reshape packed -> per-head", lambda: ttnn.deallocate(ttnn.reshape(pack, (B, H, 1, D))))
        bench("reshape per-head -> packed", lambda: ttnn.deallocate(ttnn.reshape(padded, (1, 1, G, D))))
        print("what the shipped code already pays to reach per-head form:")
        bench("reshape [1,1,32,1536] -> [32,12,1,128]", lambda: ttnn.deallocate(ttnn.reshape(wide, (B, H, 1, D))))
        bench("reshape [1,1,32,1536] -> [1,1,384,128]", lambda: ttnn.deallocate(ttnn.reshape(wide, (1, 1, G, D))))
        bench("permute per-head (2,0,1,3)", lambda: ttnn.deallocate(ttnn.permute(padded, (2, 0, 1, 3))))

        print(f"\nfive elementwise ops: padded {padded_total:.1f} us, packed {pack_total:.1f} us")
        print(f"headroom per layer if the repack were free: {padded_total - pack_total:.1f} us")
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
