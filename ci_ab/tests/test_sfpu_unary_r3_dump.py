# SPDX-License-Identifier: Apache-2.0
#
# llk_analysis, ops/blackhole/sfpu_unary round 3: raw output bits of the ttnn ops whose compute API entry points, LLK
# bodies or unary compute kernel round 3 changes, so that current main and the branch can be compared bit for bit. Float
# ops: every bf16 bit pattern as bfloat16 (16-bit DEST), the same patterns widened to float32 and 65536 seeded float32
# values in [-16, 16] (32-bit DEST), 256 x 256 each (64 tiles). Integer ops: 65536 seeded int32 values, and uint32 or
# uint16 where named. Output directory from SFPU_FIX_DUMP_DIR; R3_DUMP_OPS and R3_DUMP_INPUTS select.
import os
import pathlib

import pytest
import torch
import ttnn
import sys

if not os.environ.get("HWLOCK_HELD") and not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
    sys.exit("not under hwlock")

R3_LAYOUT = os.environ.get("R3_DUMP_LAYOUT", "dram")


def _r3_put(x, dtype, device):
    # dram: DRAM interleaved, a 256 x 256 input is one tile per core; sh4: height-sharded on 4 cores, 16 tiles per core;
    # dram16: the input repeated 16 times along the height, DRAM interleaved, 9 or 10 tiles per core
    if R3_LAYOUT == "sh4":
        mem = ttnn.create_sharded_memory_config(
            shape=(x.shape[-2] // 4, x.shape[-1]),
            core_grid=ttnn.CoreGrid(y=1, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
    if R3_LAYOUT == "dram16":
        x = x.repeat(1, 1, 16, 1)
    return ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _SM():
    m = getattr(ttnn, "SigmoidMode", None)
    return m if m is not None else ttnn.unary.SigmoidMode

pytestmark = pytest.mark.use_module_device

OUT = pathlib.Path(os.environ.get("SFPU_FIX_DUMP_DIR", "/tmp/sfpu_fix_dump_r3"))


def _snake_beta(t):
    # alpha and beta as in test_snake_beta.py's ranges, same shape, dtype and memory as x (no broadcast), fixed seed
    g = torch.Generator().manual_seed(7)
    shape = tuple(t.shape)
    tdt = torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16
    a = (torch.rand(shape, generator=g) * 0.44 + 0.96).to(tdt)
    b = (torch.rand(shape, generator=g) * 0.64 + 0.97).to(tdt)
    mk = lambda v: ttnn.from_torch(v, dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=t.device(), memory_config=t.memory_config())
    return ttnn.snake_beta(t, mk(a), mk(b))

F = {
    "reciprocal": lambda t: ttnn.reciprocal(t),
    "hardsigmoid": lambda t: ttnn.hardsigmoid(t),
    "celu": lambda t: ttnn.celu(t, alpha=1.5),
    "hardshrink": lambda t: ttnn.hardshrink(t, lambd=0.5),
    "cbrt": lambda t: ttnn.cbrt(t),
    "digamma": lambda t: ttnn.digamma(t),
    "elu": lambda t: ttnn.elu(t, alpha=1.0),
    "erf": lambda t: ttnn.erf(t),
    "erf_accurate": lambda t: ttnn.erf(t, fast_and_approximate_mode=False),
    "erfinv": lambda t: ttnn.erfinv(t),
    "gelu": lambda t: ttnn.gelu(t),
    "gelu_approx": lambda t: ttnn.gelu(t, fast_and_approximate_mode=True),
    "hardmish": lambda t: ttnn.hardmish(t),
    "hardtanh": lambda t: ttnn.hardtanh(t, min_val=-1.0, max_val=1.0),
    "i1": lambda t: ttnn.i1(t),
    "log1p": lambda t: ttnn.log1p(t),
    "mish": lambda t: ttnn.mish(t),
    "neg": lambda t: ttnn.neg(t),
    "polygamma": lambda t: ttnn.polygamma(t, 2),
    "logsigmoid": lambda t: ttnn.log_sigmoid(t),
    "lgamma": lambda t: ttnn.lgamma(t),
    "snake_beta": lambda t: _snake_beta(t),
    "rdiv": lambda t: ttnn.rdiv(t, 3.0),
    "leaky_relu": lambda t: ttnn.leaky_relu(t, negative_slope=0.1),
    "remainder": lambda t: ttnn.remainder(t, 3.0),
    "fmod": lambda t: ttnn.fmod(t, 3.0),
    "round": lambda t: ttnn.round(t, decimals=0),
    "rpow": lambda t: ttnn.rpow(t, 2.0),
    "rsqrt": lambda t: ttnn.rsqrt(t),
    "selu": lambda t: ttnn.selu(t),
    "softplus": lambda t: ttnn.softplus(t, beta=1.0, threshold=20.0),
    "sqrt": lambda t: ttnn.sqrt(t),
    "tanhshrink": lambda t: ttnn.tanhshrink(t),
    "threshold": lambda t: ttnn.threshold(t, 1.0, 10.0),
    "sin": lambda t: ttnn.sin(t),
    "cos": lambda t: ttnn.cos(t),
    "tan": lambda t: ttnn.tan(t),
    "asin": lambda t: ttnn.asin(t),
    "acos": lambda t: ttnn.acos(t),
    "atan": lambda t: ttnn.atan(t),
    "asinh": lambda t: ttnn.asinh(t),
    "acosh": lambda t: ttnn.acosh(t),
    "atanh": lambda t: ttnn.atanh(t),
    "sinh": lambda t: ttnn.sinh(t),
    "xielu": lambda t: ttnn.xielu(t),
    "sigmoid": lambda t: ttnn.sigmoid(t),
    "sigmoid_approx": lambda t: ttnn.sigmoid(t, mode=_SM().FastApproximate),
    "sigmoid_fastexp": lambda t: ttnn.sigmoid(t, mode=_SM().AccurateWithFastExp),
    "silu": lambda t: ttnn.silu(t),
    "tanh": lambda t: ttnn.tanh(t),
    "tanh_approx": lambda t: ttnn.tanh(t, fast_and_approximate_mode=True),
    "square": lambda t: ttnn.square(t),
    "log": lambda t: ttnn.log(t),
    "log_approx": lambda t: ttnn.log(t, fast_and_approximate_mode=True),
    "log2": lambda t: ttnn.log2(t),
    "log10": lambda t: ttnn.log10(t),
    "signbit": lambda t: ttnn.signbit(t),
    "pow3": lambda t: ttnn.pow(t, 3.0),
    "pow2_5": lambda t: ttnn.pow(t, 2.5),
    "pow_int3": lambda t: ttnn.pow(t, 3),
    "gelu_tanh": lambda t: ttnn.gelu(t, variant=ttnn.GeluVariant.Tanh),
    "gelu_bw_poly": lambda t: ttnn.gelu_bw(t, t)[0],
    "alt_complex_rotate90": lambda t: ttnn.alt_complex_rotate90(t),
    "exp2": lambda t: ttnn.exp2(t),
    "expm1": lambda t: ttnn.expm1(t),
    "maximum": lambda t: ttnn.maximum(t, 0.5),
    "minimum": lambda t: ttnn.minimum(t, 0.5),
    "isinf": lambda t: ttnn.isinf(t),
    "isnan": lambda t: ttnn.isnan(t),
    "isposinf": lambda t: ttnn.isposinf(t),
    "isneginf": lambda t: ttnn.isneginf(t),
    "isfinite": lambda t: ttnn.isfinite(t),
    "softcap": lambda t: ttnn.softcap(t, 10.0),
    "relu": lambda t: ttnn.relu(t),
    "exp": lambda t: ttnn.exp(t),
    "exp_approx": lambda t: ttnn.exp(t, fast_and_approximate_mode=True),
    "abs": lambda t: ttnn.abs(t),
    "gtz": lambda t: ttnn.gtz(t),
    "eqz": lambda t: ttnn.eqz(t),
    "lez": lambda t: ttnn.lez(t),
    "gt_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_GT, 0.5)]),
    "relu6": lambda t: ttnn.relu6(t),
    "relu_min": lambda t: ttnn.relu_min(t, 1.0),
    "relu_max": lambda t: ttnn.relu_max(t, 2.0),
    "clamp": lambda t: ttnn.clamp(t, -1.0, 1.0),
    "sign": lambda t: ttnn.sign(t),
    "heaviside": lambda t: ttnn.heaviside(t, 0.5),
    "floor": lambda t: ttnn.floor(t),
    "ceil": lambda t: ttnn.ceil(t),
    "trunc": lambda t: ttnn.trunc(t),
    "frac": lambda t: ttnn.frac(t),
    "prelu": lambda t: ttnn.prelu(t, 0.25),
    "i0": lambda t: ttnn.i0(t),
    "erfc": lambda t: ttnn.erfc(t),
    "cosh": lambda t: ttnn.cosh(t),
    "softshrink": lambda t: ttnn.softshrink(t, lambd=0.5),
    "softsign": lambda t: ttnn.softsign(t),
    "logical_not": lambda t: ttnn.logical_not(t),
    "rsub": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RSUB, 2.0)]),
    "fill_c": lambda t: ttnn.fill(t, 2.5),
    "lt_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_LT, 0.5)]),
    "ne_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_NE, 0.5)]),
    "gez": lambda t: ttnn.gez(t),
    "ltz": lambda t: ttnn.ltz(t),
    "nez": lambda t: ttnn.nez(t),
    "eq_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_EQ, 0.5)]),
    "ge_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_GE, 0.5)]),
    "le_scalar": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_LE, 0.5)]),
    "add_unary_c": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 2.5)]),
    "sub_unary_c": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.SUB_UNARY_SFPU, 2.5)]),
    "mul_unary_c": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.5)]),
    "div_unary_c": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.DIV_UNARY_SFPU, 2.5)]),
}
I = {
    "abs_i32": lambda t: ttnn.abs(t),
    "neg_i32": lambda t: ttnn.neg(t),
    "relu_i32": lambda t: ttnn.relu(t),
    "relu_min_i32": lambda t: ttnn.relu_min(t, 5),
    "relu_max_i32": lambda t: ttnn.relu_max(t, 5),
    "signbit_i32": lambda t: ttnn.signbit(t),
    "maximum_i32": lambda t: ttnn.maximum(t, 7),
    "minimum_i32": lambda t: ttnn.minimum(t, 7),
    "clamp_i32": lambda t: ttnn.clamp(t, -1000, 1000),
    "gtz_i32": lambda t: ttnn.gtz(t),
    "eqz_i32": lambda t: ttnn.eqz(t),
    "nez_i32": lambda t: ttnn.nez(t),
    "bitwise_and_i32": lambda t: ttnn.bitwise_and(t, 0x0F0F),
    "bitwise_or_i32": lambda t: ttnn.bitwise_or(t, 0x0F0F),
    "bitwise_xor_i32": lambda t: ttnn.bitwise_xor(t, 0x0F0F),
    "bitwise_not_i32": lambda t: ttnn.bitwise_not(t),
    "left_shift_i32": lambda t: ttnn.bitwise_left_shift(t, 3),
    "right_shift_i32": lambda t: ttnn.bitwise_right_shift(t, 3),
    "square_i32": lambda t: ttnn.square(t),
    "eq_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_EQ, 5.0)]),
    "gt_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_GT, 5.0)]),
    "relu6_i32": lambda t: ttnn.relu6(t),
    "relu_min_u32": lambda t: ttnn.relu_min(t, 5),
    "relu_max_u32": lambda t: ttnn.relu_max(t, 5),
    "relu_min_u16": lambda t: ttnn.relu_min(t, 5),
    "relu_max_u16": lambda t: ttnn.relu_max(t, 5),
    "ne_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_NE, 5.0)]),
    "eqz_u32": lambda t: ttnn.eqz(t),
    "nez_u32": lambda t: ttnn.nez(t),
    "eqz_u16": lambda t: ttnn.eqz(t),
    "nez_u16": lambda t: ttnn.nez(t),
    "bitwise_not_u32": lambda t: ttnn.bitwise_not(t),
    "bitwise_not_u16": lambda t: ttnn.bitwise_not(t),
    "left_shift_u32": lambda t: ttnn.bitwise_left_shift(t, 3),
    "right_shift_u32": lambda t: ttnn.bitwise_right_shift(t, 3),
    "left_shift_u16": lambda t: ttnn.bitwise_left_shift(t, 3),
    "right_shift_u16": lambda t: ttnn.bitwise_right_shift(t, 3),
    "ge_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_GE, 5.0)]),
    "lt_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_LT, 5.0)]),
    "le_scalar_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.UNARY_LE, 5.0)]),
    "rsub_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RSUB, 5.0)]),
    "fill_i32": lambda t: ttnn.fill(t, 7),
    "fill_u32": lambda t: ttnn.fill(t, 7),
    "fill_u16": lambda t: ttnn.fill(t, 7),
    "remainder_u32": lambda t: ttnn.remainder(t, 7),
    "lez_i32": lambda t: ttnn.lez(t),
    "gez_i32": lambda t: ttnn.gez(t),
    "ltz_i32": lambda t: ttnn.ltz(t),
    "add_unary_c_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 5.0)]),
    "sub_unary_c_i32": lambda t: ttnn.unary_chain(t, [ttnn.UnaryWithParam(ttnn.UnaryOpType.SUB_UNARY_SFPU, 5.0)]),
}


def _float_inputs():
    bits16 = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16)
    bf16 = bits16.view(torch.bfloat16).reshape(1, 1, 256, 256)
    g = torch.Generator().manual_seed(1234)
    rnd = torch.empty(1, 1, 256, 256, dtype=torch.float32).uniform_(-16.0, 16.0, generator=g)
    return {
        "bf16": (bf16, ttnn.bfloat16, torch.int16),
        "fp32widened": (bf16.to(torch.float32), ttnn.float32, torch.int32),
        "fp32random": (rnd, ttnn.float32, torch.int32),
    }


def _int_inputs(op=""):
    g = torch.Generator().manual_seed(4321)
    if op.endswith("_u32"):
        u = torch.randint(0, 2**31 - 1, (1, 1, 256, 256), dtype=torch.int32, generator=g)
        u[0, 0, 0, :8] = torch.tensor([0, 1, 4, 5, 6, 7, 1000, 2**31 - 1], dtype=torch.int32)
        return {"uint32": (u, ttnn.uint32, torch.int32)}
    if op.endswith("_u16"):
        u = torch.arange(0, 2**16, dtype=torch.int32).reshape(1, 1, 256, 256)
        return {"uint16": (u, ttnn.uint16, torch.int32)}
    i32 = torch.randint(-(2**31), 2**31 - 1, (1, 1, 256, 256), dtype=torch.int32, generator=g)
    i32[0, 0, 0, :64] = torch.tensor([0, 1, -1, 5, -5, 7, -7, 1000, -1000, 2**31 - 1, -(2**31)] + [0] * 53, dtype=torch.int32)
    return {"int32": (i32, ttnn.int32, torch.int32)}


SEL_OPS = [o for o in os.environ.get("R3_DUMP_OPS", ",".join(list(F) + list(I))).split(",") if o]
SEL_INPUTS = [i for i in os.environ.get("R3_DUMP_INPUTS", "bf16,fp32widened,fp32random,int32").split(",") if i] + ["uint32", "uint16"]


@pytest.mark.parametrize("op", SEL_OPS)
def test_dump_r3(device, op):
    OUT.mkdir(parents=True, exist_ok=True)
    fn = F.get(op) or I[op]
    inputs = _float_inputs() if op in F else _int_inputs(op)
    for name, (x, dtype, view) in inputs.items():
        if name not in SEL_INPUTS:
            continue
        tt_x = _r3_put(x, dtype, device)
        try:
            y = ttnn.to_torch(fn(tt_x))
        except Exception as e:  # an op that refuses an input dtype is recorded, not failed
            (OUT / f"{op}_{name}.refused").write_text(str(e)[:2000])
            continue
        torch.save(y.contiguous().view(torch.int32 if y.dtype in (torch.float32, torch.int32, torch.uint32) else torch.int16 if y.dtype in (torch.bfloat16, torch.int16) else y.dtype), OUT / f"{op}_{name}.pt")
