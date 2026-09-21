import torch, ttnn, glob, os
SD=os.environ["SD"]
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    def ulp(x):  # bf16 ulp at |x|
        e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
        return torch.pow(2.0, e - 7)
    for f in sorted(glob.glob(f"{SD}/phndump/phn_*.pt")):
        d = torch.load(f); x_t, w_t, eps, tag = d["x"], d["w"], d["eps"], d["tag"]
        xd = x_t.double()
        ref = xd / torch.sqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
        if w_t is not None: ref = ref * w_t.double().reshape(1, 1, 1, -1)
        x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = None if w_t is None else ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        outs = {}
        for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
            kw = {"epsilon": eps, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
            if w is not None: kw["weight"] = w
            outs[side] = ttnn.to_torch(fn(x, **kw)).double()
        g, n = outs["gen"], outs["nat"]
        torch.save({"gen": g, "nat": n, "ref": ref}, f.replace(".pt", "_out.pt"))
        D = (g - n).squeeze()                     # gen - nat
        R = ref.squeeze(); u = ulp(R)
        Du = D / u                                # in ulps of the reference value
        Eg, En = ((g - ref) / ulp(ref)).squeeze(), ((n - ref) / ulp(ref)).squeeze()
        diff = D != 0
        rows, cols = D.shape
        # position structure
        col_frac = diff.double().mean(0)          # fraction of rows differing, per column
        row_frac = diff.double().mean(1)
        face_col = diff.double().reshape(rows, cols // 16, 16).mean((0, 1))   # by column-within-face
        face_row = diff.double().reshape(rows // 16, 16, cols).mean((0, 2))   # by row-within-face
        name = os.path.basename(f)
        print(f"== {name} shape={tuple(D.shape)} gamma={w_t is not None}")
        print(f"   differing elems: {diff.double().mean().item()*100:.1f}%  |gen-nat| in ulps: mean {Du.abs().mean().item():.3f} max {Du.abs().max().item():.1f}  "
              f"sign: +{(Du>0).sum().item()} -{(Du<0).sum().item()}")
        print(f"   |err| vs fp64 in ulps: gen mean {Eg.abs().mean().item():.3f} max {Eg.abs().max().item():.1f} | nat mean {En.abs().mean().item():.3f} max {En.abs().max().item():.1f}")
        print(f"   elems where gen err > 2 ulp: {(Eg.abs()>2).sum().item()}  nat: {(En.abs()>2).sum().item()}  | gen>4ulp: {(Eg.abs()>4).sum().item()} nat>4ulp: {(En.abs()>4).sum().item()}")
        print(f"   diff-frac per column: min {col_frac.min().item():.2f} max {col_frac.max().item():.2f} | per row: min {row_frac.min().item():.2f} max {row_frac.max().item():.2f}")
        print(f"   by col-in-face(16): {[f'{v:.2f}' for v in face_col.tolist()]}")
        print(f"   by row-in-face(16): {[f'{v:.2f}' for v in face_row.tolist()]}")
        big = (Eg.abs() > 2).nonzero()
        if len(big):
            r0, c0 = big[0].tolist()
            print(f"   example gen>2ulp at (row {r0}, col {c0}): ref={R[r0,c0].item():+.6f} gen={g.squeeze()[r0,c0].item():+.6f} nat={n.squeeze()[r0,c0].item():+.6f} x={xd.squeeze()[r0,c0].item():+.6f}")
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
