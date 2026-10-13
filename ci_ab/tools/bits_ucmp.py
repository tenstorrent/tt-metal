# llk_analysis #59107 check (ops/blackhole/sfpu_int_typecast): raw output words of the unary int32 scalar compares
# (unary_lt_tile_int32, unary_gt_tile_int32, unary_le_tile_int32, unary_ge_tile_int32) as ttnn reaches them with an
# integer scalar: ttnn.unary_chain with ttnn._ttnn.activation.EltwiseUnaryWithParam(op, int), whose int32_t (or uint32_t)
# parameter reaches the kernel unchanged (unary_op_utils.cpp, std::bit_cast<uint32_t>(param0_raw)). A float parameter cannot
# carry INT_MAX, and -0.0 (INT_MIN) shares the program cache entry of 0.0. Inputs: random words, with s - 1, s, s + 1,
# INT_MIN, INT_MIN + 1, INT_MAX, INT_MAX - 1, 0, 1 and -1 on every lane position of the first four tile rows.
# Two trees are compared word for word offline ($R3_BITS_OUT/<case>.npz).
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

EUP = ttnn._ttnn.activation.EltwiseUnaryWithParam
OUT = os.environ.get("R3_BITS_OUT", os.path.join(HERE, "bits_out"))
IMIN, IMAX = -(2**31), 2**31 - 1
SCALARS = [IMIN, IMIN + 1, -987654321, -65536, -5, -1, 0, 1, 5, 65535, 123456789, IMAX - 1, IMAX]
OPS = {"lt": "UNARY_LT", "gt": "UNARY_GT", "le": "UNARY_LE", "ge": "UNARY_GE"}


def wrap(v):
    return ((v + 2**31) % 2**32) - 2**31


def inputs(s, seed, unsigned=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(IMIN, IMAX, (1024, 1024), generator=g, dtype=torch.int64)
    sp = sorted({wrap(s - 1), wrap(s), wrap(s + 1), IMIN, IMIN + 1, IMAX, IMAX - 1, 0, 1, -1})
    n = len(sp)
    r = torch.arange(128).view(-1, 1)
    c = torch.arange(1024).view(1, -1)
    idx = (r // 32 + c // 32 + (r % 32) * 32 + c % 32) % n
    x[:128] = torch.tensor(sp, dtype=torch.int64)[idx]
    if unsigned:
        return (x & 0xFFFFFFFF).reshape(1, 1, 1024, 1024)
    return x.to(torch.int32).reshape(1, 1, 1024, 1024)


def tt(x, d, dt):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


CASES = {}
for name, op in OPS.items():
    for s in SCALARS:
        CASES[f"ucmp_{name}_i32_{s}"] = lambda d, op=op, s=s: [
            ttnn.unary_chain(
                tt(inputs(s, 59107), d, ttnn.int32), [EUP(getattr(ttnn.UnaryOpType, op), s)]
            )
        ]
    # uint32 tensors take the same body; the parameter is the uint32 word.
    for s in (0, 1, 2**31 - 1, 2**31, 2**32 - 1):
        CASES[f"ucmp_{name}_u32_{s}"] = lambda d, op=op, s=s: [
            ttnn.unary_chain(
                tt(inputs(wrap(s), 59108, unsigned=True), d, ttnn.uint32),
                [EUP(getattr(ttnn.UnaryOpType, op), s)],
            )
        ]


@pytest.mark.parametrize("name", list(CASES))
def test_bits(dev, name):  # noqa: F811
    outs = CASES[name](dev)
    if B.MOCK:
        return
    os.makedirs(OUT, exist_ok=True)
    arrs = {f"o{i}": B.words(o if isinstance(o, torch.Tensor) else ttnn.to_torch(o)) for i, o in enumerate(outs)}
    np.savez(os.path.join(OUT, f"{name}.npz"), **arrs)
