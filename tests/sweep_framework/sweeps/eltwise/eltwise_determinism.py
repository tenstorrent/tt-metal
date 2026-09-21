# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Every eltwise op, run N times on identical inputs, graded on two properties.

Each run must be bit-identical to the first, and the first must match the golden. Either alone
passes for the wrong reason: an op returning garbage returns the same garbage every time, so
determinism says nothing about correctness, and an op checked once cannot see a result that
moves between dispatches. Run 1 populates the program cache and runs 2..N hit it, so a
cache-path divergence shows up here rather than as an intermittent failure downstream.

The run-to-run comparison is on raw bytes rather than values, so a NaN payload change or a
signed-zero flip counts -- `==` would call those equal, or for NaN never equal. Correctness is
PCC, matching how the rest of the eltwise sweeps grade.

Suites are the configuration surface. `default` is one vector per op at bfloat16, a single
32x32 tile, DRAM in and out -- small enough to run daily. The wider suites exist for manual
dispatch, via --suite-name, when a dtype, shape or memory-config axis is worth sweeping.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Hang detection. A vector runs the op RUNS times, so it needs more headroom than a single call.
TIMEOUT = 60

DEFAULT_SHAPE = [1, 1, 32, 32]
# The GLU family splits the last dim in half and each half has to be a full tile.
GLU_SHAPE = [1, 1, 32, 64]


def _make_torch(shape, torch_dtype, values, seed):
    """Operand generator. `values` picks a range that keeps the op inside its domain."""
    gen = torch.Generator().manual_seed(seed)
    if torch_dtype == torch.int32:
        lo, hi = {"positive": (1, 100), "small": (-10, 10), "unit": (0, 2)}.get(values, (-1000, 1000))
        return torch.randint(lo, hi, tuple(shape), dtype=torch_dtype, generator=gen)
    raw = torch.rand(tuple(shape), dtype=torch.float32, generator=gen)
    if values == "positive":  # (0.1, 100]  -- log, sqrt, reciprocal, divisors
        out = raw * 99.9 + 0.1
    elif values == "unit":  # (-1, 1)       -- asin, acos, atanh, erfinv
        out = raw * 1.98 - 0.99
    elif values == "small":  # (-1, 1)      -- exp-family, avoids overflow
        out = raw * 2.0 - 1.0
    elif values == "gamma":  # [1, 100]     -- digamma, lgamma, multigammaln
        out = raw * 99.0 + 1.0
    elif values == "above_one":  # (1, 100] -- acosh
        out = raw * 99.0 + 1.0001
    elif values == "well_above_one":  # [2, 100] -- acosh_bw, whose derivative blows up at 1
        out = raw * 98.0 + 2.0
    elif values == "mid":  # [3, 10]        -- multigammaln_bw
        out = raw * 7.0 + 3.0
    else:  # "range": (-100, 100)
        out = raw * 200.0 - 100.0
    return out.to(torch_dtype)


@dataclass(frozen=True)
class OpSpec:
    name: str
    ttnn_fn: Callable
    # None means "ask ttnn for the registered golden", which is what the backward ops need.
    golden: Optional[Callable] = None
    arity: int = 1
    values: Any = "range"  # str, or a per-operand tuple
    dtype: Any = None  # pins the dtype when the op only accepts one (bitwise ops are int32)
    shape: Optional[Sequence[int]] = None  # pins the shape when the op constrains it (GLU)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    # Backward ops return a tuple of gradients; compare this many of them.
    outputs: int = 1
    pcc: float = 0.99
    # Backward ops take (grad, *inputs). Their goldens run a real autograd backward pass, so
    # every operand after the gradient has to be a leaf with requires_grad set.
    backward: bool = False

    def operand_values(self, i):
        return self.values[i] if isinstance(self.values, (tuple, list)) else self.values


def _relu6(x):
    return torch.nn.functional.relu6(x)


UNARY_OPS = [
    OpSpec("abs", ttnn.abs, torch.abs),
    OpSpec("acos", ttnn.acos, torch.acos, values="unit"),
    OpSpec("acosh", ttnn.acosh, torch.acosh, values="above_one"),
    OpSpec("asin", ttnn.asin, torch.asin, values="unit"),
    OpSpec("asinh", ttnn.asinh, torch.asinh),
    OpSpec("atan", ttnn.atan, torch.atan),
    OpSpec("atanh", ttnn.atanh, torch.atanh, values="unit"),
    OpSpec("cbrt", ttnn.cbrt, lambda x: torch.sign(x) * torch.abs(x).pow(1 / 3)),
    OpSpec("ceil", ttnn.ceil, torch.ceil),
    OpSpec("celu", ttnn.celu, torch.nn.functional.celu),
    OpSpec("cos", ttnn.cos, torch.cos),
    OpSpec("cosh", ttnn.cosh, torch.cosh, values="small"),
    OpSpec("deg2rad", ttnn.deg2rad, torch.deg2rad),
    OpSpec("digamma", ttnn.digamma, torch.digamma, values="gamma"),
    OpSpec("elu", ttnn.elu, lambda x: torch.nn.functional.elu(x, alpha=1.0), kwargs={"alpha": 1.0}),
    OpSpec("eqz", ttnn.eqz, lambda x: x == 0),
    OpSpec("erf", ttnn.erf, torch.erf),
    OpSpec("erfc", ttnn.erfc, torch.erfc),
    OpSpec("erfinv", ttnn.erfinv, torch.erfinv, values="unit"),
    OpSpec("exp", ttnn.exp, torch.exp, values="small"),
    OpSpec("exp2", ttnn.exp2, torch.exp2, values="small"),
    OpSpec("expm1", ttnn.expm1, torch.expm1, values="small"),
    OpSpec("floor", ttnn.floor, torch.floor),
    OpSpec("frac", ttnn.frac, torch.frac),
    OpSpec("gelu", ttnn.gelu, torch.nn.functional.gelu),
    OpSpec("gez", ttnn.gez, lambda x: x >= 0),
    OpSpec("gtz", ttnn.gtz, lambda x: x > 0),
    OpSpec("hardsigmoid", ttnn.hardsigmoid, torch.nn.functional.hardsigmoid),
    OpSpec("hardswish", ttnn.hardswish, torch.nn.functional.hardswish),
    OpSpec("hardtanh", ttnn.hardtanh, torch.nn.functional.hardtanh),
    OpSpec("i0", ttnn.i0, torch.i0, values="small"),
    OpSpec("identity", ttnn.identity, lambda x: x),
    OpSpec("isfinite", ttnn.isfinite, torch.isfinite),
    OpSpec("isinf", ttnn.isinf, torch.isinf),
    OpSpec("isnan", ttnn.isnan, torch.isnan),
    OpSpec("isneginf", ttnn.isneginf, torch.isneginf),
    OpSpec("isposinf", ttnn.isposinf, torch.isposinf),
    OpSpec("lez", ttnn.lez, lambda x: x <= 0),
    OpSpec("lgamma", ttnn.lgamma, torch.lgamma, values="positive"),
    OpSpec("log", ttnn.log, torch.log, values="positive"),
    OpSpec("log10", ttnn.log10, torch.log10, values="positive"),
    OpSpec("log1p", ttnn.log1p, torch.log1p, values="small"),
    OpSpec("log2", ttnn.log2, torch.log2, values="positive"),
    OpSpec("log_sigmoid", ttnn.log_sigmoid, torch.nn.functional.logsigmoid, values="small"),
    OpSpec("logical_not", ttnn.logical_not, torch.logical_not),
    OpSpec("ltz", ttnn.ltz, lambda x: x < 0),
    OpSpec("mish", ttnn.mish, lambda x: x * torch.tanh(torch.nn.functional.softplus(x))),
    OpSpec("multigammaln", ttnn.multigammaln, lambda x: torch.mvlgamma(x, 2), values="gamma"),
    OpSpec("neg", ttnn.neg, torch.neg),
    OpSpec("nez", ttnn.nez, lambda x: x != 0),
    OpSpec("rad2deg", ttnn.rad2deg, torch.rad2deg),
    OpSpec("reciprocal", ttnn.reciprocal, torch.reciprocal, values="positive"),
    OpSpec("relu", ttnn.relu, torch.relu),
    OpSpec("relu6", ttnn.relu6, _relu6),
    OpSpec("round", ttnn.round, torch.round),
    OpSpec("rsqrt", ttnn.rsqrt, torch.rsqrt, values="positive"),
    OpSpec("selu", ttnn.selu, torch.nn.functional.selu),
    OpSpec("sigmoid", ttnn.sigmoid, torch.sigmoid),
    OpSpec("sign", ttnn.sign, torch.sign),
    OpSpec("signbit", ttnn.signbit, torch.signbit),
    OpSpec("silu", ttnn.silu, torch.nn.functional.silu),
    OpSpec("sin", ttnn.sin, torch.sin),
    OpSpec("sinh", ttnn.sinh, torch.sinh, values="small"),
    OpSpec("softplus", ttnn.softplus, torch.nn.functional.softplus),
    OpSpec("softsign", ttnn.softsign, torch.nn.functional.softsign),
    OpSpec("sqrt", ttnn.sqrt, torch.sqrt, values="positive"),
    OpSpec("square", ttnn.square, torch.square),
    OpSpec("tan", ttnn.tan, torch.tan, values="small"),
    OpSpec("tanh", ttnn.tanh, torch.tanh),
    OpSpec("tanhshrink", ttnn.tanhshrink, torch.nn.functional.tanhshrink),
    OpSpec("tril", ttnn.tril, torch.tril),
    OpSpec("triu", ttnn.triu, torch.triu),
    OpSpec("trunc", ttnn.trunc, torch.trunc),
    # parameterised unary
    OpSpec("clamp", ttnn.clamp, lambda x: torch.clamp(x, -1.0, 1.0), args=(-1.0, 1.0)),
    OpSpec("hardshrink", ttnn.hardshrink, lambda x: torch.nn.functional.hardshrink(x, 0.5), kwargs={"lambd": 0.5}),
    OpSpec("heaviside", ttnn.heaviside, lambda x: torch.heaviside(x, torch.zeros_like(x)), args=(0.0,)),
    OpSpec("leaky_relu", ttnn.leaky_relu, lambda x: torch.nn.functional.leaky_relu(x, 0.01), args=(0.01,)),
    OpSpec("relu_max", ttnn.relu_max, lambda x: torch.clamp(torch.relu(x), max=6.0), args=(6.0,)),
    OpSpec("relu_min", ttnn.relu_min, lambda x: torch.clamp(x, min=0.1), args=(0.1,)),
    OpSpec("softshrink", ttnn.softshrink, lambda x: torch.nn.functional.softshrink(x, 0.5), kwargs={"lambd": 0.5}),
    OpSpec("threshold", ttnn.threshold, lambda x: torch.threshold(x, 0.1, 0.0), args=(0.1, 0.0)),
    # GLU family: last dim is split in half, so each half must still be a full tile
    OpSpec("glu", ttnn.glu, lambda x: torch.nn.functional.glu(x, -1), args=(-1,), shape=GLU_SHAPE),
    OpSpec(
        "geglu",
        ttnn.geglu,
        lambda x: x.chunk(2, -1)[0] * torch.nn.functional.gelu(x.chunk(2, -1)[1]),
        shape=GLU_SHAPE,
    ),
    OpSpec("reglu", ttnn.reglu, lambda x: x.chunk(2, -1)[0] * torch.relu(x.chunk(2, -1)[1]), shape=GLU_SHAPE),
    OpSpec(
        "swiglu",
        ttnn.swiglu,
        lambda x: x.chunk(2, -1)[0] * torch.nn.functional.silu(x.chunk(2, -1)[1]),
        shape=GLU_SHAPE,
    ),
    # integer unary
    OpSpec("bitwise_not", ttnn.bitwise_not, torch.bitwise_not, dtype=ttnn.int32),
]

BINARY_OPS = [
    OpSpec("add", ttnn.add, torch.add, arity=2),
    OpSpec("atan2", ttnn.atan2, torch.atan2, arity=2),
    OpSpec("bias_gelu", ttnn.bias_gelu, lambda a, b: torch.nn.functional.gelu(a + b), arity=2),
    OpSpec("divide", ttnn.divide, torch.divide, arity=2, values=("range", "positive")),
    OpSpec("eq", ttnn.eq, torch.eq, arity=2),
    OpSpec("fmod", ttnn.fmod, torch.fmod, arity=2, values=("range", "positive")),
    OpSpec("ge", ttnn.ge, torch.ge, arity=2),
    OpSpec("gt", ttnn.gt, torch.gt, arity=2),
    OpSpec("hypot", ttnn.hypot, torch.hypot, arity=2),
    OpSpec("ldexp", ttnn.ldexp, torch.ldexp, arity=2, values="small"),
    OpSpec("le", ttnn.le, torch.le, arity=2),
    OpSpec("logaddexp", ttnn.logaddexp, torch.logaddexp, arity=2, values="small"),
    OpSpec("logaddexp2", ttnn.logaddexp2, torch.logaddexp2, arity=2, values="small"),
    OpSpec("logical_and", ttnn.logical_and, torch.logical_and, arity=2),
    OpSpec("logical_or", ttnn.logical_or, torch.logical_or, arity=2),
    OpSpec("logical_xor", ttnn.logical_xor, torch.logical_xor, arity=2),
    OpSpec("lt", ttnn.lt, torch.lt, arity=2),
    OpSpec("maximum", ttnn.maximum, torch.maximum, arity=2),
    OpSpec("minimum", ttnn.minimum, torch.minimum, arity=2),
    OpSpec("multiply", ttnn.multiply, torch.multiply, arity=2),
    OpSpec("ne", ttnn.ne, torch.ne, arity=2),
    OpSpec("nextafter", ttnn.nextafter, torch.nextafter, arity=2),
    OpSpec("pow", ttnn.pow, torch.pow, arity=2, values=("positive", "small")),
    OpSpec("remainder", ttnn.remainder, torch.remainder, arity=2, values=("range", "positive")),
    OpSpec("squared_difference", ttnn.squared_difference, lambda a, b: (a - b) ** 2, arity=2),
    OpSpec("subtract", ttnn.subtract, torch.subtract, arity=2),
    OpSpec("xlogy", ttnn.xlogy, torch.xlogy, arity=2, values=("range", "positive")),
    OpSpec("addalpha", ttnn.addalpha, lambda a, b: a + 2.0 * b, arity=2, args=(2.0,)),
    OpSpec("subalpha", ttnn.subalpha, lambda a, b: a - 2.0 * b, arity=2, args=(2.0,)),
    OpSpec(
        "isclose",
        ttnn.isclose,
        lambda a, b: torch.isclose(a, b, rtol=1e-5, atol=1e-8),
        arity=2,
        kwargs={"rtol": 1e-5, "atol": 1e-8},
    ),
    # integer binary
    OpSpec("bitwise_and", ttnn.bitwise_and, torch.bitwise_and, arity=2, dtype=ttnn.int32),
    OpSpec("bitwise_or", ttnn.bitwise_or, torch.bitwise_or, arity=2, dtype=ttnn.int32),
    OpSpec("bitwise_xor", ttnn.bitwise_xor, torch.bitwise_xor, arity=2, dtype=ttnn.int32),
]

TERNARY_OPS = [
    OpSpec("addcmul", ttnn.addcmul, lambda a, b, c: torch.addcmul(a, b, c, value=1.0), arity=3, kwargs={"value": 1.0}),
    OpSpec(
        "addcdiv",
        ttnn.addcdiv,
        lambda a, b, c: torch.addcdiv(a, b, c, value=1.0),
        arity=3,
        values=("range", "range", "positive"),
        kwargs={"value": 1.0},
    ),
    OpSpec("lerp", ttnn.lerp, torch.lerp, arity=3, values=("range", "range", "small")),
    OpSpec("mac", ttnn.mac, lambda a, b, c: a * b + c, arity=3),
    OpSpec("where", ttnn.where, lambda c, a, b: torch.where(c != 0, a, b), arity=3),
]

# Backward ops take (grad, *inputs) and return a tuple of gradients. Their goldens are the ones
# registered with ttnn, so a golden that disagrees with the op is caught here rather than hidden.
UNARY_BW = [
    ("abs_bw", "range"),
    ("acos_bw", "unit"),
    ("acosh_bw", "well_above_one"),
    ("asin_bw", "unit"),
    ("asinh_bw", "range"),
    ("atan_bw", "range"),
    ("atanh_bw", "unit"),
    ("ceil_bw", "range"),
    ("cos_bw", "range"),
    ("cosh_bw", "small"),
    ("deg2rad_bw", "range"),
    ("digamma_bw", "positive"),
    ("erf_bw", "range"),
    ("erfc_bw", "range"),
    ("erfinv_bw", "unit"),
    ("exp_bw", "small"),
    ("exp2_bw", "small"),
    ("expm1_bw", "small"),
    ("floor_bw", "range"),
    ("frac_bw", "range"),
    ("gelu_bw", "range"),
    ("hardsigmoid_bw", "range"),
    ("hardswish_bw", "range"),
    ("lgamma_bw", "gamma"),
    ("log_bw", "positive"),
    ("log_sigmoid_bw", "range"),
    ("log1p_bw", "positive"),
    ("log10_bw", "positive"),
    ("log2_bw", "positive"),
    ("multigammaln_bw", "mid"),
    ("neg_bw", "range"),
    ("rad2deg_bw", "range"),
    ("reciprocal_bw", "positive"),
    ("relu_bw", "range"),
    ("relu6_bw", "range"),
    ("round_bw", "range"),
    ("rsqrt_bw", "positive"),
    ("selu_bw", "range"),
    ("sigmoid_bw", "range"),
    ("sign_bw", "range"),
    ("silu_bw", "range"),
    ("sin_bw", "range"),
    ("sinh_bw", "small"),
    ("softsign_bw", "range"),
    ("sqrt_bw", "positive"),
    ("square_bw", "range"),
    ("tan_bw", "small"),
    ("tanh_bw", "unit"),
    ("tanhshrink_bw", "range"),
    ("trunc_bw", "range"),
    ("fill_bw", "range"),
    ("fill_zero_bw", "range"),
]

BINARY_BW = [
    ("add_bw", "range"),
    ("atan2_bw", "range"),
    ("bias_gelu_bw", "range"),
    ("fmod_bw", "range"),
    ("hypot_bw", "range"),
    ("ldexp_bw", "small"),
    ("logaddexp_bw", "small"),
    ("logaddexp2_bw", "small"),
    ("max_bw", "range"),
    ("min_bw", "range"),
    ("mul_bw", "range"),
    ("remainder_bw", "range"),
    ("rsub_bw", "range"),
    ("squared_difference_bw", "range"),
    ("sub_bw", "range"),
    ("xlogy_bw", "range"),
]

BACKWARD_OPS = [
    OpSpec(name, getattr(ttnn, name), None, arity=2, values=values, outputs=1, backward=True)
    for name, values in UNARY_BW
    if hasattr(ttnn, name)
] + [
    OpSpec(name, getattr(ttnn, name), None, arity=3, values=values, outputs=2, backward=True)
    for name, values in BINARY_BW
    if hasattr(ttnn, name)
]

ALL_OPS = UNARY_OPS + BINARY_OPS + TERNARY_OPS + BACKWARD_OPS
OPS = {spec.name: spec for spec in ALL_OPS}
OP_NAMES = sorted(OPS)

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.int32: torch.int32}


# Vectors carry only serialisable values, so the op travels as its name and is looked up here.
parameters = {
    # One vector per op at the smallest useful configuration. This is what runs on a schedule.
    "default": {
        "op_name": OP_NAMES,
        "input_shape": [DEFAULT_SHAPE],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "runs": [3],
    },
    # The axes below are for manual dispatch (--suite-name), not the daily run.
    "dtypes": {
        "op_name": OP_NAMES,
        "input_shape": [DEFAULT_SHAPE],
        "input_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "runs": [3],
    },
    "memory_configs": {
        "op_name": OP_NAMES,
        "input_shape": [DEFAULT_SHAPE],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "runs": [3],
    },
    "shapes": {
        "op_name": OP_NAMES,
        # A sub-tile shape, the single tile, a multi-tile row and a batched case. The GLU family
        # overrides this anyway, since it needs an even last dim of at least two tiles.
        "input_shape": [[1, 1, 32, 32], [1, 1, 64, 128], [2, 3, 96, 96], [1, 1, 320, 384]],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "runs": [3],
    },
}


def _build_operands(spec, shape, ttnn_dtype, memory_config, device):
    ttnn_dtype = spec.dtype or ttnn_dtype
    torch_dtype = _TORCH_DTYPE[ttnn_dtype]
    shape = list(spec.shape or shape)
    torch_operands, ttnn_operands = [], []
    for i in range(spec.arity):
        t = _make_torch(shape, torch_dtype, spec.operand_values(i), seed=1000 + i)
        # The ttnn tensor is built before requires_grad is set: from_torch wants a plain leaf.
        ttnn_operands.append(
            ttnn.from_torch(t, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        )
        if spec.backward and i > 0 and t.is_floating_point():
            t.requires_grad_(True)
        torch_operands.append(t)
    return torch_operands, ttnn_operands


def _as_bytes(tensor):
    """Raw bytes of an output, so NaN payloads and signed zeros take part in the comparison."""
    t = ttnn.to_torch(tensor).contiguous()
    return t.view(torch.uint8).clone() if t.dtype != torch.bool else t.to(torch.uint8).clone()


def _outputs_of(result, count):
    return [result[i] for i in range(count)] if isinstance(result, (list, tuple)) else [result]


def _golden_for(spec, torch_operands, device):
    """Either the torch expression in the table, or the golden ttnn registers for the op.

    A table golden already closes over the op's constants, so it takes the operands only. The
    registered golden is the op's own signature and does take them.
    """
    if spec.golden is not None:
        return [spec.golden(*torch_operands)]
    golden_fn = ttnn.get_golden_function(spec.ttnn_fn)
    try:
        result = golden_fn(*torch_operands, *spec.args, **spec.kwargs)
    except TypeError:
        # A few goldens build an intermediate tensor and need somewhere to put it.
        result = golden_fn(*torch_operands, *spec.args, **spec.kwargs, device=device)
    return list(result)[: spec.outputs] if isinstance(result, (list, tuple)) else [result]


def run(
    op_name,
    input_shape,
    input_dtype,
    input_memory_config,
    output_memory_config,
    runs,
    *,
    device,
) -> list:
    spec = OPS[op_name]
    torch_operands, ttnn_operands = _build_operands(spec, input_shape, input_dtype, input_memory_config, device)

    kwargs = dict(spec.kwargs)
    if output_memory_config is not None:
        kwargs["memory_config"] = output_memory_config

    start_time = start_measuring_time()
    captured = []
    for _ in range(runs):
        try:
            result = spec.ttnn_fn(*ttnn_operands, *spec.args, **kwargs)
        except TypeError:
            # Not every op takes memory_config; fall back to the op's own signature.
            kwargs.pop("memory_config", None)
            result = spec.ttnn_fn(*ttnn_operands, *spec.args, **kwargs)
        captured.append([_as_bytes(t) for t in _outputs_of(result, spec.outputs)])
        last = result
    e2e_perf = stop_measuring_time(start_time)

    # Determinism. Compared as bytes, so a NaN payload or signed-zero change is a difference.
    for run_index in range(1, runs):
        for out_index, (first, later) in enumerate(zip(captured[0], captured[run_index])):
            if not torch.equal(first, later):
                differing = int((first != later).sum())
                return [
                    (
                        False,
                        f"{op_name} is not deterministic: run {run_index + 1} of {runs} differs from "
                        f"run 1 in output {out_index} at {differing} bytes, on identical inputs",
                    ),
                    e2e_perf,
                ]

    # Bring the outputs to host, then free every device tensor before grading. Vectors run back
    # to back in one process, so an L1 residency left alive here is still held while the next
    # vector allocates its own -- which showed up as neighbouring vectors failing in the L1
    # suites while passing in isolation. Outputs are read first because an in-place op aliases
    # its input, and deallocating the operand would take the result with it.
    actual = [ttnn.to_torch(t) for t in _outputs_of(last, spec.outputs)]
    for tensor in list(_outputs_of(last, spec.outputs)) + ttnn_operands:
        try:
            ttnn.deallocate(tensor)
        except RuntimeError:
            pass  # already freed, e.g. an in-place result aliasing an operand

    # Correctness. Determinism alone would also hold for an op that is consistently wrong.
    expected = _golden_for(spec, torch_operands, device)
    for out_index, (want, got_torch) in enumerate(zip(expected, actual)):
        if want.dtype == torch.bool or got_torch.dtype == torch.bool:
            if not torch.equal(got_torch.to(torch.float32), want.to(torch.float32)):
                return [(False, f"{op_name} output {out_index} does not match the golden"), e2e_perf]
        else:
            passed, message = check_with_pcc(want, got_torch, spec.pcc)
            if not passed:
                return [(False, f"{op_name} output {out_index}: {message}"), e2e_perf]

    return [(True, f"{op_name}: {runs} runs bit-identical, output matches golden"), e2e_perf]
