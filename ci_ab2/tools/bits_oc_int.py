# llk_analysis one-call check (#58743, ops/blackhole/sfpu_int_typecast): raw output words of the ttnn ops that reach the
# float unary_max_tile, unary_min_tile, signbit_tile, clamp_tile_int32 and the int32 division, remainder, fmod, gcd
# and lcm entry points #58825 switches to one 32-row call (the plain ops, ttnn.unary_chain, the binary_ng activation
# path of ttnn.multiply and ttnn.clamped_silu_glu, and mhc_split_sinkhorn), on the tree under test; two trees are compared word for word offline ($R3_BITS_OUT/<case>.npz).
import os
import sys

import numpy as np
import pytest
import torch
import ttnn

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("R3_BITS_OUT", os.path.join(HERE, "bits_out"))
MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


@pytest.fixture(scope="module")
def dev():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def words(t):
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).numpy()
    if t.dtype == torch.float32:
        return t.view(torch.int32).numpy()
    return t.numpy()


def tt(x, d, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dt, layout=layout, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


SHAPE = (1, 1, 1024, 1024)
SPECIAL = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, -1.5, 1.5, float("inf"), float("-inf"), float("nan"), 1e-40, -1e-40, 3.0e38, -3.0e38, 2.0**-126, -(2.0**-126)]


def mixed(shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * 3
    flat = x.view(-1)
    sp = torch.tensor(SPECIAL)
    n = min(flat.numel() // 2, sp.numel() * 997)
    flat[:n] = sp.repeat(997)[:n]
    return x


def i32(shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int64)
    flat = x.view(-1)
    sp = torch.tensor([-(2**31), 2**31 - 1, 0, -1, 1, -5, 5, -6, 6, 4, -4, 1000, 1001, 999, -1000, -1001, -999], dtype=torch.int64)
    flat[: sp.numel() * 997] = sp.repeat(997)
    return x.to(torch.int32)


CASES = {}
DTS = (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32))
for dt_name, dt in DTS:
    for s in (0.0, 0.5, -1.5, 1e30):
        CASES[f"maximum_{dt_name}_{s}"] = lambda d, dt=dt, s=s: [ttnn.maximum(tt(mixed(SHAPE, 1), d, dt), s)]
        CASES[f"minimum_{dt_name}_{s}"] = lambda d, dt=dt, s=s: [ttnn.minimum(tt(mixed(SHAPE, 2), d, dt), s)]
        CASES[f"chain_max_{dt_name}_{s}"] = lambda d, dt=dt, s=s: [
            ttnn.unary_chain(tt(mixed(SHAPE, 1), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.MAXIMUM, s)])
        ]
        CASES[f"chain_min_{dt_name}_{s}"] = lambda d, dt=dt, s=s: [
            ttnn.unary_chain(tt(mixed(SHAPE, 2), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, s)])
        ]
    CASES[f"signbit_{dt_name}"] = lambda d, dt=dt: [ttnn.signbit(tt(mixed(SHAPE, 3), d, dt))]
    CASES[f"chain_signbit_{dt_name}"] = lambda d, dt=dt: [
        ttnn.unary_chain(tt(mixed(SHAPE, 3), d, dt), [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGNBIT)])
    ]
    # binary_ng with a MAXIMUM / MINIMUM activation on the inputs (unary_max_tile / unary_min_tile in the binary kernel)
    CASES[f"multiply_lhs_max_{dt_name}"] = lambda d, dt=dt: [
        ttnn.multiply(
            tt(mixed(SHAPE, 4), d, dt),
            tt(mixed(SHAPE, 5), d, dt),
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MAXIMUM, 0.5)],
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, -1.5)],
        )
    ]


def _csg(d):
    g = torch.Generator().manual_seed(6)
    gate = (torch.randn((1, 1, 512, 1024), generator=g) * 8).to(torch.bfloat16)
    up = (torch.randn((1, 1, 512, 1024), generator=g) * 8).to(torch.bfloat16)
    return [ttnn.clamped_silu_glu(tt(gate, d), tt(up, d), 10.0)]


CASES["clamped_silu_glu_composite"] = _csg

for lo, hi in ((-5, 1000), (-1000, -5), (-5, 5), (5, 1000), (-(2**31), 2**31 - 1), (0, 0)):
    CASES[f"clamp_i32_{lo}_{hi}"] = lambda d, lo=lo, hi=hi: [ttnn.clamp(tt(i32(SHAPE, 7), d, ttnn.int32), min=lo, max=hi)]
CASES["clamp_i32_rm"] = lambda d: [ttnn.clamp(tt(i32((1, 1, 64, 1000), 8), d, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), min=-5, max=1000)]


def ints(shape, dt, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(lo, hi, shape, dtype=torch.int64, generator=g)
    return (x & 0xFFFFFFFF) if dt == ttnn.uint32 else x.to(torch.int32)


def div_inputs(fn, dt, shape, d):
    lo = 0 if dt == ttnn.uint32 else -(2**31)
    hi = 2**31 - 1
    if fn == "lcm":
        lo, hi = -(2**15) + 1, 2**15
    elif fn == "gcd":
        lo, hi = -(2**31) + 1, 2**31 - 1
    a = ints(shape, dt, lo, hi, 1)
    b = ints(shape, dt, 1, hi if fn != "lcm" else 2**15, 2)
    if dt != ttnn.uint32:
        b = b * torch.where(ints(shape, dt, 0, 2, 3) == 0, -1, 1).to(torch.int32)
    return tt(a, d, dt), tt(b, d, dt)


DIV_FNS = {
    "remainder": lambda a, b: ttnn.remainder(a, b),
    "fmod": lambda a, b: ttnn.fmod(a, b),
    "gcd": lambda a, b: ttnn.gcd(a, b),
    "lcm": lambda a, b: ttnn.lcm(a, b),
    "div": lambda a, b: ttnn.divide(a, b),
    "div_floor": lambda a, b: ttnn.div(a, b, rounding_mode="floor"),
    "div_trunc": lambda a, b: ttnn.div(a, b, rounding_mode="trunc"),
}
for fn in DIV_FNS:
    for dt, dn in ((ttnn.int32, "i32"), (ttnn.uint32, "u32")):
        if dt == ttnn.uint32 and fn != "remainder":
            continue
        CASES[f"{fn}_{dn}_1024x1024"] = lambda d, fn=fn, dt=dt: [DIV_FNS[fn](*div_inputs(fn, dt, SHAPE, d))]


# Exhaustive inputs where the input space allows it (rule 3 of 07_ISSUE_RESOLUTION.md): every bfloat16 pattern (NaNs,
# infinities, zeros and denormals included) and every uint16 value, 65536 words in one 256x256 tensor.
A16 = (1, 1, 256, 256)


def all_bf16():
    w = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.int16)
    return torch.from_numpy(w.copy()).view(torch.bfloat16).reshape(A16)


def all_u16():
    return torch.arange(65536, dtype=torch.int32).reshape(A16)


for s in (0.0, -0.0, 0.5, -1.5, 1e30, float("inf"), float("-inf")):
    CASES[f"maximum_bf16_all_{s}"] = lambda d, s=s: [ttnn.maximum(tt(all_bf16(), d), s)]
    CASES[f"minimum_bf16_all_{s}"] = lambda d, s=s: [ttnn.minimum(tt(all_bf16(), d), s)]
CASES["signbit_bf16_all"] = lambda d: [ttnn.signbit(tt(all_bf16(), d))]
CASES["signbit_bf16_all_chain"] = lambda d: [ttnn.unary_chain(tt(all_bf16(), d), [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGNBIT)])]
CASES["multiply_lhs_max_rhs_min_bf16_all"] = lambda d: [
    ttnn.multiply(
        tt(all_bf16(), d),
        tt(all_bf16().reshape(-1).flip(0).reshape(A16), d),
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MAXIMUM, 0.5)],
        input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MINIMUM, -1.5)],
    )
]
# The typecast pairs #58825 rewrites, in the dest mode ttnn picks for each: bfloat16 and uint16 inputs exhaustive, the
# Float32 inputs on special values and random words.
for dst_name, dst in (("int32", ttnn.int32), ("uint16", ttnn.uint16), ("uint32", ttnn.uint32), ("float32", ttnn.float32)):
    CASES[f"typecast_bf16_all_to_{dst_name}"] = lambda d, dst=dst: [ttnn.typecast(tt(all_bf16(), d), dst)]
for dst_name, dst in (("float32", ttnn.float32), ("uint32", ttnn.uint32), ("int32", ttnn.int32), ("bfloat16", ttnn.bfloat16)):
    CASES[f"typecast_u16_all_to_{dst_name}"] = lambda d, dst=dst: [ttnn.typecast(tt(all_u16(), d, ttnn.uint16), dst)]


def f32_words():
    g = torch.Generator().manual_seed(9)
    x = torch.randint(-(2**31), 2**31 - 1, (65536,), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    sp = torch.tensor(SPECIAL + [65535.0, 65535.5, 65536.0, 2.0**31, -(2.0**31), 2.0**31 - 128, 2.0**24 + 1, 1.5, 2.5, -2.5, 0.49999997])
    x[: sp.numel()] = sp
    return x.reshape(A16)


for dst_name, dst in (("uint16", ttnn.uint16), ("bfloat16", ttnn.bfloat16), ("int32", ttnn.int32), ("uint32", ttnn.uint32)):
    CASES[f"typecast_f32_to_{dst_name}"] = lambda d, dst=dst: [ttnn.typecast(tt(f32_words(), d, ttnn.float32), dst)]


# Integer binary entry points (#58743 for #58825) and the ordered int32 scalar compares (#59107): random words with the
# special values (INT_MIN, INT_MAX, 0, +-1, the shift counts 0 to 40) on every lane position; uint16 operands exhaustive.
INT_SPECIAL = [-(2**31), 2**31 - 1, 0, 1, -1, 2, -2, 31, 32, 33, 40, 65535, 65536, -65536, 2**30, -(2**30), 12345, -12345]


def int_words(shape, dt, seed, lo=None, hi=None):
    g = torch.Generator().manual_seed(seed)
    n = 1
    for s in shape:
        n *= s
    lo = -(2**31) if lo is None else lo
    hi = 2**31 - 1 if hi is None else hi
    x = torch.randint(lo, hi, (n,), generator=g, dtype=torch.int64)
    sp = torch.tensor(INT_SPECIAL, dtype=torch.int64)
    k = min(n // 4, sp.numel() * 1021)
    x[:k] = sp.repeat(1021)[:k]
    x = x[torch.randperm(n, generator=g)]
    if dt == ttnn.uint32:
        x = x & 0xFFFFFFFF
    elif dt == ttnn.uint16:
        x = x & 0xFFFF
    else:
        x = x.to(torch.int32)
    return x.reshape(shape)


def u16_all():
    return torch.arange(65536, dtype=torch.int32).reshape(A16)


IB = {
    "add": lambda a, b: ttnn.add(a, b),
    "sub": lambda a, b: ttnn.subtract(a, b),
    "rsub": lambda a, b: ttnn.rsub(a, b),
    "mul": lambda a, b: ttnn.multiply(a, b),
    "and": lambda a, b: ttnn.bitwise_and(a, b),
    "or": lambda a, b: ttnn.bitwise_or(a, b),
    "xor": lambda a, b: ttnn.bitwise_xor(a, b),
    "lshift": lambda a, b: ttnn.bitwise_left_shift(a, b),
    "rshift": lambda a, b: ttnn.bitwise_right_shift(a, b),
    "lrshift": lambda a, b: ttnn.logical_right_shift(a, b),
    "lt": lambda a, b: ttnn.lt(a, b),
    "gt": lambda a, b: ttnn.gt(a, b),
    "le": lambda a, b: ttnn.le(a, b),
    "ge": lambda a, b: ttnn.ge(a, b),
}
for fn in IB:
    for dt_name, dt in (("i32", ttnn.int32), ("u32", ttnn.uint32), ("u16", ttnn.uint16)):
        def _ib(d, fn=fn, dt=dt, dt_name=dt_name):
            shift = fn in ("lshift", "rshift", "lrshift")
            if dt_name == "u16":
                a = u16_all()
                b = int_words(A16, dt, 52, 0, 41) if shift else int_words(A16, dt, 52)
            else:
                a = int_words(SHAPE, dt, 50)
                b = int_words(SHAPE, dt, 51, 0, 41) if shift else int_words(SHAPE, dt, 51)
            return [IB[fn](tt(a, d, dt), tt(b, d, dt))]

        if not (fn == "lrshift" and dt_name == "u16"):  # ttnn has no uint16 logical right shift
            CASES[f"ib_{fn}_{dt_name}"] = _ib

# The division, remainder, fmod, gcd and lcm entry points on the same special values (zero divisors included) on every lane
# position; lcm's random words in its documented 15-bit range.
for fn in DIV_FNS:
    for dt, dn in ((ttnn.int32, "i32"), (ttnn.uint32, "u32")):
        if dt == ttnn.uint32 and fn != "remainder":
            continue
        rng = (-(2**15) + 1, 2**15) if fn == "lcm" else ()
        CASES[f"{fn}_{dn}_special"] = lambda d, fn=fn, dt=dt, rng=rng: [
            DIV_FNS[fn](tt(int_words(SHAPE, dt, 56, *rng), d, dt), tt(int_words(SHAPE, dt, 57, *rng), d, dt))
        ]

for cmp_name in ("lt", "gt", "le", "ge"):
    for s in (0, 1, -1, 5, -5, 2**31 - 1, -(2**31), 2**31 - 2, -(2**31) + 1, 123456789, -987654321):
        CASES[f"cmp_scalar_{cmp_name}_i32_{s}"] = lambda d, cmp_name=cmp_name, s=s: [
            getattr(ttnn, cmp_name)(tt(int_words(SHAPE, ttnn.int32, 53), d, ttnn.int32), s)
        ]


def _sinkhorn(d, scale_val):
    p = os.path.join(os.environ.get("TT_METAL_HOME", "."), "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill")
    sys.path.insert(0, p)
    import test_mhc_split_sinkhorn as M

    cfg = M.MHCConfig(dim=64, n=4)
    mixes, scale, base = M._params(cfg, scale_val, M.TOKENS)
    return list(M._run(d, mixes, scale, base, cfg))


CASES["mhc_split_sinkhorn_s0.01"] = lambda d: _sinkhorn(d, 0.01)
CASES["mhc_split_sinkhorn_s1.0"] = lambda d: _sinkhorn(d, 1.0)


@pytest.mark.parametrize("name", list(CASES))
def test_bits(dev, name):
    outs = CASES[name](dev)
    if MOCK:
        return
    os.makedirs(OUT, exist_ok=True)
    arrs = {f"o{i}": words(o if isinstance(o, torch.Tensor) else ttnn.to_torch(o)) for i, o in enumerate(outs)}
    np.savez(os.path.join(OUT, f"{name}.npz"), **arrs)
