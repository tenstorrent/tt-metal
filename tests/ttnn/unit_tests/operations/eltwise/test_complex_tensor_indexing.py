# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _complex(device):
    def t(v):
        return ttnn.from_torch(
            torch.full(SHAPE, v, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    return ttnn.complex_tensor(t(1.0), t(2.0))


def test_complex_tensor_index_zero_and_one_are_real_and_imag(device):
    c = _complex(device)
    assert ttnn.to_torch(c[0]).float().mean().item() == ttnn.to_torch(c.real).float().mean().item() == 1.0
    assert ttnn.to_torch(c[1]).float().mean().item() == ttnn.to_torch(c.imag).float().mean().item() == 2.0
    assert ttnn.to_torch(c[-1]).float().mean().item() == 2.0


def test_complex_tensor_index_out_of_range_raises_index_error(device, expect_error):
    c = _complex(device)
    for index in (2, 3, 1000, -3):
        with expect_error(IndexError, "out of range"):
            c[index]


def test_complex_tensor_has_two_parts(device):
    c = _complex(device)
    assert len(c) == 2
    parts = list(c)
    assert len(parts) == 2
    assert ttnn.to_torch(parts[1]).float().mean().item() == 2.0
