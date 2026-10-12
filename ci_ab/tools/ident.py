# Round 3 sfpu_reduce rework: bit identity of the SFPU reduce results between trees. Runs the ttnn reductions that reach
# the SFPU reduce on fixed inputs that hold every bf16 bit pattern, the fp32 special values and the Int32 extremes, and
# saves the raw result bits; compare two trees' files with ident_cmp.py.
# usage: python ident.py <out.pt>
import os
import sys

import torch
import ttnn

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")

OUT = sys.argv[1]
MIN, MAX = -(2**31), 2**31 - 1


def bf16_all(seed):
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    g = torch.Generator().manual_seed(seed)
    return bits[torch.randperm(bits.numel(), generator=g)].view(torch.bfloat16)


def fp32_special(seed):
    g = torch.Generator().manual_seed(seed)
    hi = torch.arange(-(2**15), 2**15, dtype=torch.int32) << 16
    sp = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
          0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF,
          0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF]
    sp = torch.tensor([v - (1 << 32) if v >= 2**31 else v for v in sp], dtype=torch.int32).repeat(64)
    rnd = torch.randint(MIN, MAX, (2**16 - sp.numel(),), generator=g, dtype=torch.int64).to(torch.int32)
    x = torch.cat([hi, sp, rnd])
    return x[torch.randperm(x.numel(), generator=g)].view(torch.float32)


def int32_special(seed):
    g = torch.Generator().manual_seed(seed)
    sp = [MIN, MIN + 1, MIN + 2, -2, -1, 0, 1, 2, MAX - 2, MAX - 1, MAX]
    sp += [s * ((1 << k) + d) for k in range(31) for d in (-1, 0, 1) for s in (1, -1) if -(2**31) <= s * ((1 << k) + d) <= MAX]
    sp = torch.tensor(sp, dtype=torch.int64).repeat(64)
    rnd = torch.randint(MIN, MAX, (2**16 - sp.numel(),), generator=g, dtype=torch.int64)
    x = torch.cat([sp, rnd]).to(torch.int32)
    return x[torch.randperm(x.numel(), generator=g)]


# (dtype, op, dim, 32-bit DEST): the ops that reach the SFPU reduce, with ttnn's default 32-bit DEST, plus bf16 MIN in a
# 16-bit DEST (compute_kernel_config), which reaches the Float16_b kernels.
CASES = [("bfloat16", "min", d, dest32) for dest32 in (True, False) for d in (-1, -2)]
CASES += [("float32", op, d, True) for op in ("max", "min", "sum", "mean") for d in (-1, -2)]
CASES += [("int32", op, d, True) for op in ("max", "min", "sum") for d in (-1, -2)]


def main():
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        for dtype, opname, dim, dest32 in CASES:
            kw = {} if dest32 else {"compute_kernel_config": ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=False)}
            for si in range(3):
                for seed in (0, 1):
                    src = {"bfloat16": bf16_all, "float32": fp32_special, "int32": int32_special}[dtype](seed)
                    n = src.numel()
                    shape = [(1, 1, n // 32, 32), (1, 1, 32, n // 32), (2, 1, n // 256, 128)][si]
                    x = src.reshape(shape)
                    tdt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32, "int32": ttnn.int32}[dtype]
                    t = ttnn.from_torch(x, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)
                    y = ttnn.to_torch(getattr(ttnn, opname)(t, dim=dim, keepdim=True, **kw))
                    ttnn.deallocate(t)
                    raw = y.contiguous().view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32).clone()
                    res[f"{dtype}-{opname}-{dim}-{'dest32' if dest32 else 'dest16'}-{'x'.join(map(str, shape))}-{seed}"] = raw
                    print(f"{dtype} {opname} dim {dim} dest32 {dest32} {shape} seed {seed}: {tuple(raw.shape)}", flush=True)
    finally:
        ttnn.close_device(dev)
        torch.save(res, OUT)
    print(f"IDENT {len(res)} cases saved to {OUT}", flush=True)


main()
