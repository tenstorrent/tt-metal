# Round 3 decisions: outputs of the generic reduce's SCALAR path (a one-tile tensor reduced over H and W runs the
# single-core REDUCE_SCALAR kernel), MAX and MIN on bf16, fp32 and bfp8 inputs (the gated unpack), SUM and MEAN (main's
# unpack), fixed seeds. usage: python rc_scalar.py <out.pt>
import sys
import torch, ttnn
dev = ttnn.open_device(device_id=0)
out = {}
for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32"), (ttnn.bfloat8_b, "bfp8")):
    for seed, shape, neg in ((1, (1, 1, 32, 32), False), (2, (1, 1, 32, 32), True), (3, (1, 1, 17, 29), False)):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(shape, generator=g) * 4
        if neg:
            x = -x.abs() - 0.5
        t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        for op in ("max", "min", "sum", "mean"):
            k = f"{op}_{name}_s{seed}_{'x'.join(map(str, shape))}"
            try:
                r = getattr(ttnn, op)(t, dim=[-2, -1], keepdim=True)
                out[k] = ttnn.to_torch(r).float()
                print(f"CASE {k} ok", flush=True)
            except Exception as e:
                print(f"CASE {k} FAIL {str(e)[:200]}", flush=True)
torch.save(out, sys.argv[1])
ttnn.close_device(dev)
