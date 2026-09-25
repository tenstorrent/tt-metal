# Legacy SDPA config matrix driver for BASE/HEAD ELF comparison on Wormhole.
# Uses only APIs present on BASE (explicit chunk sizes, no precision=).
import sys, math, torch, ttnn

def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

def ckc(fid, fp32, approx=True):
    if fid is None:
        return None
    return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=approx,
                                           fp32_dest_acc_en=fp32, packer_l1_acc=False)

def tt(x, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

def ref(q, k, v, causal, mask=None):
    nh, nkv = q.shape[1], k.shape[1]
    k = k.repeat_interleave(nh // nkv, 1); v = v.repeat_interleave(nh // nkv, 1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, attn_mask=mask)

CONFIGS = []
for causal in (False, True):
    for fid, fp32 in ((None, None), ("HiFi2", False), ("HiFi4", False), ("LoFi", False), ("HiFi2", True), ("HiFi4", True)):
        CONFIGS.append(dict(kind="dense", causal=causal, fid=fid, fp32=fp32, s=1024, d=128, nh=4, nkv=4, q=128, k=128, dtype="bf16"))
CONFIGS += [
    dict(kind="dense", causal=False, fid=None, fp32=None, s=2048, d=64, nh=8, nkv=1, q=256, k=256, dtype="bf8"),
    dict(kind="dense", causal=True, fid=None, fp32=None, s=2048, d=64, nh=8, nkv=2, q=64, k=512, dtype="bf8"),
    dict(kind="dense", causal=False, fid="HiFi2", fp32=False, s=4096, d=128, nh=2, nkv=2, q=256, k=512, dtype="bf16"),
    dict(kind="mask", causal=False, fid=None, fp32=None, s=1024, d=128, nh=4, nkv=4, q=128, k=128, dtype="bf16"),
    dict(kind="mask", causal=False, fid="HiFi4", fp32=True, s=1024, d=128, nh=4, nkv=4, q=128, k=128, dtype="bf16"),
    dict(kind="joint", causal=False, fid=None, fp32=None, s=1024, d=128, nh=4, nkv=4, q=128, k=128, dtype="bf16"),
    dict(kind="joint", causal=False, fid="HiFi4", fp32=True, s=1024, d=128, nh=4, nkv=4, q=128, k=128, dtype="bf16"),
    dict(kind="joint", causal=False, fid="LoFi", fp32=False, s=2048, d=64, nh=4, nkv=4, q=256, k=256, dtype="bf16"),
]

def run(dev, c):
    torch.manual_seed(0)
    b = 1
    dt = ttnn.bfloat16 if c["dtype"] == "bf16" else ttnn.bfloat8_b
    q = torch.randn(b, c["nh"], c["s"], c["d"]); k = torch.randn(b, c["nkv"], c["s"], c["d"]); v = torch.randn(b, c["nkv"], c["s"], c["d"])
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
                                q_chunk_size=c["q"], k_chunk_size=c["k"], exp_approx_mode=True)
    kc = ckc(c["fid"], c["fp32"])
    if c["kind"] == "dense":
        o = ttnn.transformer.scaled_dot_product_attention(tt(q, dev, dt), tt(k, dev, dt), tt(v, dev, dt), is_causal=c["causal"],
                                                          program_config=pc, compute_kernel_config=kc)
        r = ref(q, k, v, c["causal"])
    elif c["kind"] == "mask":
        m = torch.zeros(b, 1, c["s"], c["s"]); m[..., c["s"] // 2 :] = -float("inf"); m[..., :64] = torch.randn(b, 1, c["s"], 64)
        o = ttnn.transformer.scaled_dot_product_attention(tt(q, dev, dt), tt(k, dev, dt), tt(v, dev, dt), attn_mask=tt(m, dev, ttnn.bfloat8_b),
                                                          is_causal=False, program_config=pc, compute_kernel_config=kc)
        r = ref(q, k, v, False, m)
    else:
        js = 128
        jq = torch.randn(b, c["nh"], js, c["d"]); jk = torch.randn(b, c["nh"], js, c["d"]); jv = torch.randn(b, c["nh"], js, c["d"])
        o, jo = ttnn.transformer.joint_scaled_dot_product_attention(tt(q, dev), tt(k, dev), tt(v, dev), tt(jq, dev), tt(jk, dev), tt(jv, dev),
                                                                    joint_strategy="rear", program_config=pc, compute_kernel_config=kc)
        rr = ref(torch.cat([q, jq], 2), torch.cat([k, jk], 2), torch.cat([v, jv], 2), False)
        r = rr[:, :, : c["s"]]
        print("   joint pcc", pcc(rr[:, :, c["s"]:], ttnn.to_torch(jo)[:, :, :js]))
    o = ttnn.to_torch(o)[:, :, : c["s"], : c["d"]]
    import os
    out = os.environ.get("WHCHECK_OUT")
    if out:
        os.makedirs(out, exist_ok=True); torch.save(o.contiguous(), f"{out}/matrix_{run.idx}.pt")
    return pcc(r, o)

if __name__ == "__main__":
    sel = sys.argv[1:]
    dev = ttnn.open_device(device_id=0)
    fails = 0
    for i, c in enumerate(CONFIGS):
        if sel and str(i) not in sel:
            continue
        run.idx = i
        p = run(dev, c)
        ok = p > 0.99
        fails += not ok
        print(f"CFG {i} {c} pcc={p:.6f} {'OK' if ok else 'LOWPCC'}", flush=True)
    ttnn.close_device(dev)
    print("MATRIX_DONE fails=", fails)
