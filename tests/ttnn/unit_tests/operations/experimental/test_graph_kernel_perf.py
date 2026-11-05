# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
graph_kernel vs. the equivalent chain of ttnn binary ops, on large tensors.

Run with `-s` to see the table. Both sides are measured the same way: programs are compiled and
cached by a warm-up call, then `iters` iterations are enqueued back to back and the device is
synchronized once at the end, so host launch overhead overlaps device work for both and the
number reported is throughput per evaluation of the whole expression.

    pytest tests/ttnn/unit_tests/operations/experimental/test_graph_kernel_perf.py -s
"""

import time

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

ITERS = 20


def ttnn_chain(expression, tensors):
    """Evaluate the expression with ttnn.add / sub / mul, one launch per operator, like a user would.

    The grammar is the op's: single letters, + - *, parentheses. Intermediates are freed as we go.
    """
    pos = [0]

    def peek():
        return expression[pos[0]] if pos[0] < len(expression) else ""

    def factor():
        if peek() == "(":
            pos[0] += 1
            v, owned = expr()
            pos[0] += 1  # ')'
            return v, owned
        letter = expression[pos[0]]
        pos[0] += 1
        return tensors[letter], False  # inputs are not ours to free

    def apply(op, lhs, lhs_owned, rhs, rhs_owned):
        out = op(lhs, rhs)
        if lhs_owned:
            ttnn.deallocate(lhs)
        if rhs_owned:
            ttnn.deallocate(rhs)
        return out, True

    def term():
        lhs, owned = factor()
        while peek() == "*":
            pos[0] += 1
            rhs, rhs_owned = factor()
            lhs, owned = apply(ttnn.mul, lhs, owned, rhs, rhs_owned)
        return lhs, owned

    def expr():
        lhs, owned = term()
        while peek() in ("+", "-"):
            op = ttnn.add if expression[pos[0]] == "+" else ttnn.sub
            pos[0] += 1
            rhs, rhs_owned = term()
            lhs, owned = apply(op, lhs, owned, rhs, rhs_owned)
        return lhs, owned

    out, _ = expr()
    return out


def time_per_call(device, fn, iters):
    out = fn()  # warm-up: JIT + program cache
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / iters


def bytes_moved(expression, shape, dtype_bytes):
    """DRAM bytes a fused kernel must touch: every input once, the output once."""
    n_inputs = len({c for c in expression if c.isalpha()})
    numel = 1
    for d in shape:
        numel *= d
    return (n_inputs + 1) * numel * dtype_bytes


def chain_bytes_moved(expression, shape, dtype_bytes):
    """DRAM bytes the unfused chain touches: each binary op reads two operands and writes one."""
    n_ops = sum(expression.count(c) for c in "+-*")
    numel = 1
    for d in shape:
        numel *= d
    return n_ops * 3 * numel * dtype_bytes


CASES = [
    # (expression, shape, dtype)
    ("a*b+c", (8, 2048, 2048), ttnn.bfloat16),  # fused multiply-add, 64 MB per tensor
    ("a*b+c", (8, 2048, 2048), ttnn.float32),  # same, fp32 (128 MB per tensor)
    ("(a+b)*(c-d)", (8, 2048, 2048), ttnn.bfloat16),  # two independent subtrees
    ("a+b+c+d+e+f+g+h", (4, 2048, 2048), ttnn.bfloat16),  # 8-way sum: 7 launches vs 1
    ("a*b+c*d+e*f", (4, 2048, 2048), ttnn.bfloat16),  # three products summed
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "expression, shape, dtype", CASES, ids=[f"{e}-{s}-{str(d).split('.')[-1]}" for e, s, d in CASES]
)
def test_graph_kernel_perf(device, expression, shape, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    dtype_bytes = 2 if dtype == ttnn.bfloat16 else 4
    letters = sorted({c for c in expression if c.isalpha()})
    torch_inputs = {letter: torch.randn(shape, dtype=torch_dtype) for letter in letters}
    tensors = {
        letter: ttnn.from_torch(torch_inputs[letter], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for letter in letters
    }
    inputs = [tensors[letter] for letter in letters]

    # Correctness of both paths against torch before timing anything.
    expected = eval(expression, {"__builtins__": {}}, torch_inputs)
    fused = ttnn.graph_kernel(inputs, expression)
    chain = ttnn_chain(expression, tensors)
    assert_with_pcc(expected, ttnn.to_torch(fused), 0.999)
    assert_with_pcc(expected, ttnn.to_torch(chain), 0.999)
    ttnn.deallocate(fused)
    ttnn.deallocate(chain)

    t_fused = time_per_call(device, lambda: ttnn.graph_kernel(inputs, expression), ITERS)
    t_chain = time_per_call(device, lambda: ttnn_chain(expression, tensors), ITERS)

    fused_gb = bytes_moved(expression, shape, dtype_bytes) / 1e9
    chain_gb = chain_bytes_moved(expression, shape, dtype_bytes) / 1e9
    print(
        f"\n{expression:>18s} {str(shape):>16s} {str(dtype).split('.')[-1]:>8s} | "
        f"graph_kernel {t_fused * 1e3:7.3f} ms ({fused_gb / t_fused:6.1f} GB/s over {fused_gb * 1e3:6.0f} MB) | "
        f"ttnn chain {t_chain * 1e3:7.3f} ms ({chain_gb / t_chain:6.1f} GB/s over {chain_gb * 1e3:6.0f} MB) | "
        f"speedup {t_chain / t_fused:5.2f}x"
    )

    for t in inputs:
        ttnn.deallocate(t)
