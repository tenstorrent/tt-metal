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
    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

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
