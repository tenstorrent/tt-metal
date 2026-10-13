# llk_analysis one-call check (#58743, ops/blackhole/sfpu_int_typecast): raw output words of ttnn.typecast over every
# pair with an SFPU body on Blackhole and of ttnn.quantize, ttnn.dequantize and ttnn.requantize over their int32, int8
# and uint8 forms, on the tree under test, for the typecast_tile and quantization one-call change of #58825; two trees
# are compared word for word offline ($R3_BITS_OUT/<case>.npz). bfloat16, uint16, uint8 and int8 inputs are exhaustive;
# float32, int32 and uint32 inputs are random words with the special values on every lane position.
import os
import sys

import numpy as np
import pytest
import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bits_oc_int as B  # noqa: E402
from bits_oc_int import dev  # noqa: E402,F401

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

OUT = os.environ.get("R3_BITS_OUT", os.path.join(HERE, "bits_out"))
A16 = B.A16
CASES = {}


def tt(x, d, dt):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def u8_all():
    return (torch.arange(65536, dtype=torch.int32) % 256).reshape(A16)


def i8_all():
    return ((torch.arange(65536, dtype=torch.int32) % 256) - 128).to(torch.int8).reshape(A16)


def u32_words():
    x = B.int_words(A16, ttnn.uint32, 71)
    return x


DT = {
    "bf16": ttnn.bfloat16,
    "fp32": ttnn.float32,
    "bfp8": ttnn.bfloat8_b,
    "i32": ttnn.int32,
    "u32": ttnn.uint32,
    "u16": ttnn.uint16,
    "u8": ttnn.uint8,
    "i8": ttnn.int8,
}
SRC = {
    "bf16": lambda: B.all_bf16(),
    "bfp8": lambda: B.all_bf16().to(torch.float32),
    "fp32": lambda: B.f32_words(),
    "i32": lambda: B.int_words(A16, ttnn.int32, 70),
    "u32": u32_words,
    "u16": lambda: B.all_u16(),
    "u8": u8_all,
    "i8": i8_all,
}
PAIRS = {
    "bf16": ["i32", "u16", "u32", "u8", "i8"],
    "bfp8": ["i32", "u16", "u32", "u8"],
    "fp32": ["bf16", "u16", "i32", "u32", "u8", "i8"],
    "i32": ["bf16", "fp32", "u16", "u8", "i8", "bfp8"],
    "u32": ["bf16", "fp32", "u16", "u8", "bfp8"],
    "u16": ["bf16", "fp32", "u32", "i32", "u8", "bfp8"],
    "u8": ["fp32", "bf16", "u16"],
    "i8": ["i32", "u32", "u16", "fp32", "bf16"],
}
for src, dsts in PAIRS.items():
    for dst in dsts:
        CASES[f"typecast_{src}_to_{dst}"] = lambda d, src=src, dst=dst: [ttnn.typecast(tt(SRC[src](), d, DT[src]), DT[dst])]


# The unary int32 scalar compares, the body of #59107 (`unary_lt_tile_int32` and its siblings), as ttnn reaches them: a
# unary chain op (also a fused activation) on an int32 tensor, whose float parameter arrives as the int32 scalar
# std::bit_cast<uint32_t>(param) (unary_op_utils.cpp). The parameters below give scalars on both sides of the sign:
# 0, INT_MIN (-0.0), 1 and 2 (denormals), 0x40A00000 (5.0), 0xC0A00000 (-5.0), 0x7F7FFFFF (the largest finite) and
# 0xFF7FFFFF; random words with the special values on every lane position.
_UCMP_OPS = {"lt": "UNARY_LT", "gt": "UNARY_GT", "le": "UNARY_LE", "ge": "UNARY_GE"}
for _cmp, _op in _UCMP_OPS.items():
    for _name, _p in (("0", 0.0), ("intmin", -0.0), ("1", 1e-45), ("2", 3e-45), ("5f", 5.0), ("m5f", -5.0), ("max", 3.4028234663852886e38), ("mmax", -3.4028234663852886e38)):
        CASES[f"unary_{_cmp}_i32_{_name}"] = lambda d, op=_op, p=_p: [
            ttnn.unary_chain(tt(B.int_words(B.SHAPE, ttnn.int32, 58), d, ttnn.int32), [ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, op), p)])
        ]


def qin(seed):
    # Values around the quantization range: the rounding ties of scale 0.05, both saturation ends, zero, specials.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(A16, generator=g) * 4
    flat = x.view(-1)
    ties = torch.arange(-300, 300, dtype=torch.float32) * 0.025
    sp = torch.cat([ties, torch.tensor([0.0, -0.0, 1e6, -1e6, 6.35, -6.4, 6.375, -6.425, 12.75, 1e-40])])
    flat[: sp.numel()] = sp
    return x


QZ = [(0.05, 0), (0.05, 3), (0.1, -7)]
for in_name in ("fp32", "bf16"):
    for scale, zp in QZ:
        for out_name in ("i32", "u8", "i8"):
            CASES[f"quantize_{in_name}_{out_name}_{scale}_{zp}"] = lambda d, in_name=in_name, out_name=out_name, scale=scale, zp=zp: [
                ttnn.quantize(tt(qin(80).to(torch.float32 if in_name == "fp32" else torch.bfloat16), d, DT[in_name]), scale, zp, dtype=DT[out_name])
            ]


def qwords(lo, hi, seed, dt=torch.int32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(lo, hi, A16, generator=g, dtype=torch.int64)
    flat = x.view(-1)
    sp = torch.tensor([lo, hi - 1, 0, 1, -1 if lo < 0 else 2, (lo + hi) // 2])
    flat[: sp.numel()] = sp
    return x.to(dt)


for scale, zp in QZ:
    for out_name in ("fp32", "bf16"):
        CASES[f"dequantize_i32_{out_name}_{scale}_{zp}"] = lambda d, out_name=out_name, scale=scale, zp=zp: [
            ttnn.dequantize(tt(qwords(-(2**20), 2**20, 81), d, ttnn.int32), scale, zp, dtype=DT[out_name])
        ]
        CASES[f"dequantize_i8_{out_name}_{scale}_{zp}"] = lambda d, out_name=out_name, scale=scale, zp=zp: [
            ttnn.dequantize(tt(i8_all(), d, ttnn.int8), scale, zp, dtype=DT[out_name])
        ]
    for in_name, out_name in (("i32", "i32"), ("i32", "i8"), ("i32", "u8"), ("i8", "i8"), ("i8", "u8"), ("i8", "i32")):
        CASES[f"requantize_{in_name}_{out_name}_{scale}_{zp}"] = lambda d, in_name=in_name, out_name=out_name, scale=scale, zp=zp: [
            ttnn.requantize(
                tt(qwords(-2000, 2000, 82) if in_name == "i32" else i8_all(), d, DT[in_name]), scale, zp, 0.07, 5, dtype=DT[out_name]
            )
        ]


def _per_channel(d, op):
    x = qin(83).reshape(1, 1, 256, 256)
    sc = torch.rand(256, generator=torch.Generator().manual_seed(84)) * 0.1 + 0.01
    zp = torch.randint(-10, 10, (256,), generator=torch.Generator().manual_seed(85), dtype=torch.int32)
    tsc = ttnn.from_torch(sc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
    tzp = ttnn.from_torch(zp, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=d)
    q = ttnn.quantize(tt(x, d, ttnn.float32), tsc, tzp, axis=3)
    if op == "q":
        return [q]
    return [ttnn.dequantize(q, tsc, tzp, axis=3)]


CASES["quantize_per_channel_fp32"] = lambda d: _per_channel(d, "q")
CASES["dequantize_per_channel_fp32"] = lambda d: _per_channel(d, "dq")


@pytest.mark.parametrize("name", list(CASES))
def test_bits(dev, name):  # noqa: F811
    outs = CASES[name](dev)
    if B.MOCK:
        return
    os.makedirs(OUT, exist_ok=True)
    arrs = {f"o{i}": B.words(o if isinstance(o, torch.Tensor) else ttnn.to_torch(o)) for i, o in enumerate(outs)}
    np.savez(os.path.join(OUT, f"{name}.npz"), **arrs)
