# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch

from scripts.tt_hw_planner.host_op_observer import observe_forward, verdict


def test_torch_compute_is_seen_and_named():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)

    def forward():
        return torch.where(a < 0, a * 2, a) @ b

    v = verdict(observe_forward(forward))
    assert v["on_device"] is False
    joined = " ".join(v["host_ops"])
    assert "where" in joined or "mul" in joined or "mm" in joined


def test_host_embedding_is_seen():
    table = torch.randn(1026, 16)
    tok = torch.tensor([100])

    def forward():
        return torch.nn.functional.embedding(tok, table)

    v = verdict(observe_forward(forward))
    assert v["on_device"] is False
    assert "embedding" in " ".join(v["host_ops"])


def test_pure_python_is_not_flagged():
    def forward():
        s = 0
        for i in range(1000):
            s += i
        return s

    v = verdict(observe_forward(forward))
    assert v["on_device"] is True


def test_benign_bookkeeping_not_flagged():
    x = torch.randn(2, 3)

    def forward():
        return x.view(3, 2).contiguous().size()

    v = verdict(observe_forward(forward))
    assert v["on_device"] is True
