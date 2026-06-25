# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Up-front precompile on a small two-layer MLP (matmuls + weights).

Builds a tiny MLP from plain ttnn ops -- linear -> gelu -> linear -- and drives it
through collect -> parallel compile -> warm forward, exercising matmul programs and
the ttnn.from_torch weight tensors the collector handles.

Run on a FRESH cache so the parallel compile does real work, e.g.:
    TT_METAL_CACHE=/tmp/upfront_mlp_$$ scripts/run_safe_pytest.sh \
        tests/ttnn/unit_tests/test_up_front_compile_mlp.py
"""

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

IN_DIM = 256
HIDDEN_DIM = 512
SEQ = 128


def _build_mlp(device):
    """Return (tt_x, tt_w1, tt_w2, torch_reference) for a random two-layer MLP.

    Weights are stored [in, out], the layout ttnn.linear expects.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, SEQ, IN_DIM, dtype=torch.float32) * 0.5
    w1 = torch.randn(IN_DIM, HIDDEN_DIM, dtype=torch.float32) * (IN_DIM**-0.5)
    w2 = torch.randn(HIDDEN_DIM, IN_DIM, dtype=torch.float32) * (HIDDEN_DIM**-0.5)
    torch_out = torch.nn.functional.gelu(x @ w1) @ w2

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return to_tt(x), to_tt(w1), to_tt(w2), torch_out


def _mlp(x, w1, w2):
    """linear -> gelu -> linear."""
    h = ttnn.gelu(ttnn.linear(x, w1))
    return ttnn.linear(h, w2)


def test_mlp_up_front_compile(device):
    tt_x, tt_w1, tt_w2, torch_out = _build_mlp(device)

    # Collect (NO_DISPATCH): capture the MLP's programs without running anything.
    ttnn.graph.up_front_clear()
    ttnn.graph.up_front_begin_collect()
    try:
        _mlp(tt_x, tt_w1, tt_w2)
    finally:
        ttnn.graph.up_front_end_collect()
    n_unique = ttnn.graph.up_front_num_unique()
    assert ttnn.graph.up_front_num_collected() >= 2, "expected at least the two matmuls to be captured"
    assert n_unique >= 1

    # Parallel compile warms the kernel cache and must build exactly the unique set, error-free.
    num_programs, num_errors, _, _ = ttnn.graph.up_front_compile(device, 4)
    assert num_errors == 0, "parallel compile reported errors"
    assert num_programs == n_unique, f"compiled {num_programs} programs, expected the {n_unique} unique collected"

    # Warm forward must match the torch reference.
    out = _mlp(tt_x, tt_w1, tt_w2)
    passed, pcc = assert_with_pcc(torch_out, ttnn.to_torch(out).to(torch_out.dtype), 0.99)
    assert passed, f"warm MLP forward PCC too low: {pcc}"
