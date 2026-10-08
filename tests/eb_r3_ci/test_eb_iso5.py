# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fourth pass: the r4b cases that read above main (non-native block or width sharded broadcasts
with activations, a Float32 column multiply), in a module by themselves. CI only."""
import pytest

import test_eb_blk4 as b
import test_eb_r3_mp as m
from test_eb_r3_mp import device  # noqa: F401

NAT2 = [("add", "bs16_t64", "col", "gelu"), ("add", "ws16_t32", "scalar", "gelu"), ("add", "ws32_t4", "col", "silu"),
        ("add", "ws32_t4", "scalar", "silu"), ("add_arelu", "bs16_t64", "scalar", None), ("add_arelu", "ws16_t32", "col", None),
        ("add_arelu", "ws32_t4", "scalar", None), ("logical_and", "bs16_n2_t128", "col", None),
        ("logical_and", "bs16_n2_t128", "scalar", None), ("logical_and", "bs16_t64", "col", None), ("mul", "bs16_n2_t128", "col", "relu"),
        ("mul", "ws32_t4", "scalar", "silu"), ("rsub", "bs16_n2_t128", "scalar", None), ("rsub", "ws32_t4", "col", None),
        ("rsub", "ws32_t4", "scalar", None)]


@pytest.mark.parametrize("op, mem, kind, act", NAT2, ids=["-".join(str(x) for x in c) for c in NAT2])
def test_iso_nat2(device, op, mem, kind, act):  # noqa: F811
    m.test_nat2(device, op, mem, kind, act)


@pytest.mark.parametrize("op, mem, kind, d, do, act", [("mul", "bs64_t80", "col", "bf16", "fp32", None)], ids=["mul-bs64_t80-col-bf16-fp32"])
def test_iso_bcast(device, op, mem, kind, d, do, act):  # noqa: F811
    b.test_blk4_bcast(device, op, mem, kind, d, do, act)
