# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
import numpy as np
from models.utility_functions import (
    skip_for_grayskull,
)


@skip_for_grayskull()
def test_dopout(device):
    # t = torch.ones((4, 5))
    t = torch.ones(
        (
            32,
            32,
        )
    )
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_ratios = []
    s = 124
    prob = 0.2
    # prob = 0.0001
    iter_num = 1
    # for _ in range(1000):
    torch.set_printoptions(linewidth=2000)
    for _ in range(iter_num):
        output = ttnn.dropout(t_tt, seed=s, probability=prob, scale=1.0 / (1.0 - prob))
        output_torch = ttnn.to_torch(output)
        print(output_torch)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        tt_ratios.append(round(r, 2))

    print("Test ttnn dropout")
    print("Zero prob: ", tt_ratios)
    print(f"Mean: {sum(tt_ratios) / iter_num}, Min: {min(tt_ratios)}, Max: {max(tt_ratios)}")

    # mean = np.mean(tt_ratios)
    # std = np.std(tt_ratios)
    # # current dropout has pretty high variance so we just checking with some reasonable nubmers
    # assert np.allclose(mean, prob, rtol=0.02)
    # assert std < prob
