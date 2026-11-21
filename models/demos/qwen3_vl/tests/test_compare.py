# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from models.demos.multimodal.siglip.compare import comp_pcc as siglip_comp_pcc
from models.demos.qwen3_vl.compare import comp_pcc as qwen3_vl_comp_pcc
from models.demos.qwen25_vl.compare import comp_pcc as qwen25_vl_comp_pcc
from models.experimental.ops.quasar.qwen3_vl.compare import comp_pcc as quasar_qwen3_vl_comp_pcc

ALL_COMP_PCC = [
    ("qwen3_vl", qwen3_vl_comp_pcc),
    ("qwen25_vl", qwen25_vl_comp_pcc),
    ("siglip", siglip_comp_pcc),
    ("quasar_qwen3_vl", quasar_qwen3_vl_comp_pcc),
]


@pytest.mark.parametrize("name, comp_pcc", ALL_COMP_PCC)
def test_constant_2s_vs_3s_fails(name, comp_pcc):
    """Golden all 2s vs output all 3s must FAIL, not false-pass with PCC 1.0."""
    golden = torch.full((16, 16), 2.0)
    calculated = torch.full((16, 16), 3.0)
    passing, pcc = comp_pcc(golden, calculated)
    assert not passing, f"[{name}] Expected failure for 2s vs 3s, but got passing={passing}, pcc={pcc}"
    assert pcc != 1.0, f"[{name}] pcc should not be 1.0 for mismatched constants"
    assert pcc == 0.0, f"[{name}] Expected pcc=0.0 on constant mismatch fallback, got {pcc}"


@pytest.mark.parametrize("name, comp_pcc", ALL_COMP_PCC)
def test_equal_tensors_pass(name, comp_pcc):
    """Equal tensors (both constant or both variable) must PASS with PCC 1.0."""
    # Both identical constants
    golden_const = torch.full((16, 16), 2.0)
    calculated_const = torch.full((16, 16), 2.0)
    passing, pcc = comp_pcc(golden_const, calculated_const)
    assert passing, f"[{name}] Expected pass for identical constants, got passing={passing}, pcc={pcc}"
    assert pcc == 1.0, f"[{name}] Expected pcc=1.0 for identical constants, got {pcc}"

    # Both identical variable tensors
    golden_var = torch.randn(16, 16)
    calculated_var = golden_var.clone()
    passing, pcc = comp_pcc(golden_var, calculated_var)
    assert passing, f"[{name}] Expected pass for identical variable tensors, got passing={passing}, pcc={pcc}"
    assert pcc == 1.0, f"[{name}] Expected pcc=1.0 for identical variable tensors, got {pcc}"


@pytest.mark.parametrize("name, comp_pcc", ALL_COMP_PCC)
def test_normal_close_pair_passes(name, comp_pcc):
    """A normal close pair of non-constant tensors must PASS with high PCC."""
    golden = torch.linspace(0.0, 10.0, 1000)
    calculated = golden + torch.randn(1000) * 1e-4
    passing, pcc = comp_pcc(golden, calculated, pcc=0.99)
    assert passing, f"[{name}] Expected pass for close tensors, got passing={passing}, pcc={pcc}"
    assert pcc > 0.99, f"[{name}] Expected pcc > 0.99, got {pcc}"


@pytest.mark.parametrize("name, comp_pcc", ALL_COMP_PCC)
def test_one_constant_vs_random_fails(name, comp_pcc):
    """One constant vs random tensor must FAIL, not false-pass from corrcoef diagonal."""
    golden = torch.full((32, 32), 2.0)
    calculated = torch.randn(32, 32)
    passing, pcc = comp_pcc(golden, calculated)
    assert not passing, f"[{name}] Expected failure for constant vs random, got passing={passing}, pcc={pcc}"
    assert pcc != 1.0, f"[{name}] pcc should not be 1.0 for constant vs random"

    # Reverse: random golden vs constant calculated
    passing_rev, pcc_rev = comp_pcc(calculated, golden)
    assert (
        not passing_rev
    ), f"[{name}] Expected failure for random vs constant, got passing={passing_rev}, pcc={pcc_rev}"
    assert pcc_rev != 1.0, f"[{name}] pcc should not be 1.0 for random vs constant"


@pytest.mark.parametrize("name, comp_pcc", ALL_COMP_PCC)
def test_return_shape_and_type(name, comp_pcc):
    """comp_pcc return shape must be (passed, pcc) with float pcc, not MaskedConstant."""
    golden = torch.full((8, 8), 2.0)
    calculated = torch.full((8, 8), 3.0)
    ret = comp_pcc(golden, calculated)
    assert isinstance(ret, tuple), f"[{name}] Expected tuple return, got {type(ret)}"
    assert len(ret) == 2, f"[{name}] Expected 2 elements (passed, pcc), got {len(ret)}"
    passing, pcc = ret
    assert isinstance(passing, (bool, np.bool_)), f"[{name}] Expected bool for passed, got {type(passing)}"
    assert isinstance(pcc, (float, np.floating)), f"[{name}] Expected float for pcc, got {type(pcc)}"
    assert not isinstance(pcc, np.ma.core.MaskedConstant), f"[{name}] pcc must not be MaskedConstant"
