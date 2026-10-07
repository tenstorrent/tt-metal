# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fourth pass: the cases of r4a that read above main, each the same program on both sides, in a
module by themselves (no changed program runs before them). CI only."""
import pytest

import test_eb_r3_mp as m
from test_eb_r3_mp import device  # noqa: F401

MP = [("add_arelu_s", "hs8_t8", "bf16"), ("add_arelu_s", "hs8_t8", "bfp8"), ("rsub", "hs8_t8", "bf16"), ("rsub", "ws32_t4", "bfp8"),
      ("rsub_s", "hs8_t8", "bf16"), ("rsub_s", "hs8_t8", "bfp8"), ("rsub_s", "ws32_t4", "bf16"), ("rsub_s", "ws32_t4", "bfp8"),
      ("ldexp", "ws32_t4", "bf16")]
NAT = [("add", "bs16_n2_t128", "row", "bfp8", None), ("add", "ws16_t32", "row", "bf16", None), ("add", "ws16_t32", "row", "bf16", "relu"),
       ("add", "ws16_t32", "scalar", "bf16", "relu"), ("add", "ws32_t4", "row", "bfp8", None), ("add", "ws32_t4", "scalar", "bfp8", None),
       ("mul", "ws16_t32", "col", "bfp8", None), ("mul", "ws32_t4", "scalar", "bfp8", None)]
NAT3 = [("add", "bs4_t64", "col", "bfp4-bfp4-bfp4"), ("add", "bs4_t64", "scalar", "bf16-bfp8-bf16"), ("add", "ws2_t64", "scalar", "bfp4-bfp4-bfp4"),
        ("add", "ws4_t64", "col", "bf16-bfp8-bf16"), ("add", "ws4_t64", "scalar", "bfp4-bfp4-bfp4"), ("add", "ws4_t64", "scalar", "bf16-bfp8-bf16"),
        ("mul", "bs4_t64", "scalar", "bfp8-bfp8-bfp8"), ("mul", "ws2_t64", "scalar", "bfp8-bfp8-bfp8"), ("mul", "ws4_t64", "col", "bf16-bfp8-bf16"),
        ("mul", "ws4_t64", "col", "bfp8-bfp8-bfp8"), ("mul", "ws4_t64", "scalar", "bfp4-bfp4-bfp4")]


@pytest.mark.parametrize("op, mem, d", MP, ids=["-".join(c) for c in MP])
def test_iso_mp(device, op, mem, d):  # noqa: F811
    m.test_mp(device, op, mem, d)


@pytest.mark.parametrize("op, mem, kind, d, act", NAT, ids=["-".join(str(x) for x in c) for c in NAT])
def test_iso_nat(device, op, mem, kind, d, act):  # noqa: F811
    m.test_nat(device, op, mem, kind, d, act)


@pytest.mark.parametrize("op, mem, kind, dts", NAT3, ids=["-".join(c) for c in NAT3])
def test_iso_nat3(device, op, mem, kind, dts):  # noqa: F811
    m.test_nat3(device, op, mem, kind, dts)
