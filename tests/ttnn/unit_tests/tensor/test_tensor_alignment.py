# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def test_tensor_alignment(device):
    a = torch.arange(1022).to(torch.bfloat16).reshape(1, 1, 1, 1022)
    b = torch.arange(1024).to(torch.bfloat16).reshape(1, 1, 32, 32)

    t0 = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    t1 = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    t0_d = t0.to(device)

    t1_d = t1.to(device)

    t0_h = t0_d.cpu()

    eq = torch.equal(a, t0_h.to_torch())
    assert eq

    t1_h = t1_d.cpu()

    eq = torch.equal(b, t1_h.to_torch())
    assert eq
