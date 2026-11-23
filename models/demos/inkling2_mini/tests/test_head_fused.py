# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""LM head with the logits tail inside the bank unembed (fused/head.py FUSED: softcap in Dest, argmax on the writers)
vs the unembed + tail programs: logits bitwise equal (all vocab_pad columns) and the same token, 1 row and R rows.
Cases: an exact tie planted across workers (identical weight columns, the lower must win), a saturated softcap (many
equal +-30 values: lowest column wins), random. A 2-chunk vocab with the model's chunk, kb and cores per bank."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import head as H
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.fused.head import FusedLogitsTail
from models.demos.inkling2_mini.tt.model import HIDDEN, SOFTCAP, UNEMBED_CHUNK


def _host(lg, tok):
    return ttnn.to_torch(lg).float().reshape(1, -1).clone(), int(ttnn.to_torch(tok).reshape(-1)[0])


@pytest.mark.parametrize("n_rows", [1, 3])
def test_head_fused(device, n_rows):
    torch.manual_seed(3)
    vocab = 2 * UNEMBED_CHUNK
    x = torch.randn(HIDDEN)
    w = torch.randn(HIDDEN, vocab) * 0.02
    lo, hi = 32 * 60, 32 * 700  # identical 32-column blocks on different workers, column 7 dominant
    w[:, lo : lo + 32] = w[:, hi : hi + 32]
    w[:, lo + 7] = w[:, hi + 7] = x / x.norm() * 2.0
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )  # fmt: skip
    bl = BankLinear(device, w, ttnn.bfloat8_b, UNEMBED_CHUNK, kb=3, cores_per_bank=3, compute_kernel_config=ckc)
    tail = FusedLogitsTail(device, bl, vocab, vocab - 40, SOFTCAP)
    cases = {"tie": x, "saturated": x * 400.0, "random": torch.randn(HIDDEN)}
    for name, v in cases.items():
        xs = torch.stack([v] + [torch.randn(HIDDEN) * (10.0 if r % 2 else 1.0) for r in range(1, n_rows)])
        xt = ttnn.from_torch(xs.reshape(1, 1, n_rows, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device)  # fmt: skip
        res = {}
        for fused in (False, True):
            H.FUSED = fused
            if n_rows == 1:
                res[fused] = [_host(*tail(xt))]
            else:
                res[fused] = [_host(*o) for o in tail.rows(xt)]
        H.FUSED = True
        for r in range(n_rows):
            (a, ta), (b, tb) = res[False][r], res[True][r]
            assert torch.equal(a, b), (name, r, float((a - b).abs().max()), int((a != b).sum()))
            assert ta == tb, (name, r, ta, tb)
        if name == "tie":
            assert res[True][0][1] == lo + 7, res[True][0][1]
