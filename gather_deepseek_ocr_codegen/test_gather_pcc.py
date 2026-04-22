"""
CPU sanity + TT PCC comparison for the gather codegen.

CPU side: deterministic inputs from _build_inputs(seed=42), golden via torch.gather.
TT side: inputs loaded via load_inputs_for__main, run through codegen _main.
Compare with assert_with_pcc.

Usage:
    pytest -svv gather_deepseek_ocr_codegen/test_gather_pcc.py
"""

import pytest
import torch
import ttnn

from gather_deepseek_ocr_codegen.main import _main, load_inputs_for__main
from tests.ttnn.utils_for_testing import assert_with_pcc

S = 913
D = 1280
N = 903


@pytest.fixture
def cpu_expected():
    """CPU golden: deterministic inputs with seed=42, torch.gather on CPU."""
    torch.manual_seed(42)

    source = torch.randn(N, D, dtype=torch.bfloat16)

    mask_1d = torch.zeros(S, dtype=torch.bool)
    mask_1d[:N] = True
    mask_1d = mask_1d[torch.randperm(S)]

    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, N - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand(S, D)

    return torch.gather(source, 0, source_idx_2d)


@pytest.fixture
def tt_result():
    """TT result: load inputs from tensorbin, run codegen ttnn graph."""
    tt_inputs = load_inputs_for__main()
    tt_outputs = _main(tt_inputs)
    return ttnn.to_torch(ttnn.from_device(tt_outputs[0]))


def test_gather_pcc(cpu_expected, tt_result):
    passed, msg = assert_with_pcc(cpu_expected, tt_result, pcc=0.99)
    assert passed, msg

