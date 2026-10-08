# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, sixth pass (#58725): the Python-scalar kernel's single section (rsub and an add with relu on a, 4
and 8 tiles per core, bf16 and bfp8) with and without the operand pass, each test run in a process of its own; rows at 32
tiles per core (the same program on both sides) as the control. CI only."""
import pytest

import test_eb_r3_mp as m
from test_eb_r3_mp import device  # noqa: F401

ISO = [(op, mem, d) for op in ("rsub_s", "add_arelu_s") for mem in ("ws32_t4", "hs8_t8", "hs8_t32") for d in ("bf16", "bfp8")]


@pytest.mark.parametrize("op, mem, d", ISO, ids=["-".join(c) for c in ISO])
def test_iso6(device, op, mem, d):  # noqa: F811
    m.test_mp(device, op, mem, d)
