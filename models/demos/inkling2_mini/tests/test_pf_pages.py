# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""tt/fused/pf_pages.py: a plain copy of a bank-streamed weight is the weight's tiles, in plain order. Needs the card."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.device import open_device
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.fused.pf_pages import PfPages


@pytest.mark.parametrize(
    "k, n, kw",
    [
        (2304, 8672, dict(trim=False, split=1, pass_cols=None, spread=False)),  # global w_all
        (3072, 2304, dict(trim=True, split=2, pass_cols=3, spread=True)),  # w_o (alt-bank blocks)
    ],
)
def test_plain_copy(k, n, kw):
    torch.manual_seed(0)
    w = (torch.randn(k, n) * 0.02).bfloat16().float()  # BankLinear packs bf16 values
    dev = open_device()
    try:
        nb = dev.dram_grid_size().x
        bank_w = -(-n // (32 * nb)) * 32
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )  # fmt: skip
        trim = kw.pop("trim")
        bl = BankLinear(dev, w, ttnn.bfloat8_b, bank_w * nb, kb=2, cores_per_bank=2, compute_kernel_config=ckc,
                        out_cols=n if trim else bank_w * nb, out_dtype=ttnn.bfloat16, **kw)  # fmt: skip
        ref = ttnn.to_torch(ttnn.from_torch(w.view(1, 1, k, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        pages = PfPages(dev)
        cols = [(0, 1024), (n - 64, n - 32)]
        got = ttnn.to_torch(pages.plain(bl, cols)).float()
        assert torch.equal(got, torch.cat([ref[..., a:b] for a, b in cols], dim=-1))
        got = ttnn.to_torch(pages.plain(bl, [(0, n)])).float()
        assert torch.equal(got, ref)
    finally:
        ttnn.close_device(dev)
