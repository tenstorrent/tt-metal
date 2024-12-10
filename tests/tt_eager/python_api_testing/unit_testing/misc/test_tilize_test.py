# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.utility_functions import tilize


@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        (5, 2, 4, 8),
        (5, 2, 4, 7),
        ## resnet shapes
        (1, 1, 784, 2),
        (8, 1, 2, 64),
        (1, 1, 1, 64),
    ),
)
@pytest.mark.parametrize(
    "multicore",
    (
        False,
        True,
    ),
)
def test_run_tilize_test(nb, nc, nh, nw, multicore, device):
    shape = [nb, nc, nh * 32, nw * 32]

    inp = torch.rand(*shape).bfloat16()

    a = ttnn.Tensor(
        inp,
        ttnn.bfloat16,
    ).to(device)
    b = ttnn.tilize(a, use_multicore=multicore)
    c = b.cpu().to_torch()

    tilized_inp = tilize(inp)
    passing = torch.equal(tilized_inp, c)
    assert passing


@pytest.mark.parametrize(
    "shape",
    (
        [1, 1, 1, 5, 1],
        [1, 1, 1, 4, 2],
        [1, 1, 1, 3, 3],
        [1, 1, 1, 2, 4],
        [1, 1, 1, 1, 5],
        [1, 2, 3, 2, 1],
    ),
)
@pytest.mark.parametrize(
    "multicore",
    (
        False,
        True,
    ),
)
def test_tilize_5d(shape, multicore, device):
    # tests that host -> device -> tilize -> untilize -> host is a no-op
    shape[-1] *= 32
    shape[-2] *= 32

    inp = torch.rand(*shape).bfloat16()
    a = ttnn.Tensor(
        inp,
        ttnn.bfloat16,
    ).to(device)
    b = ttnn.tilize(a, use_multicore=multicore)
    c = ttnn.untilize(b)
    d = c.cpu().to_torch()
    assert torch.equal(inp, d)
