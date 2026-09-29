# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


# These zero arguments used to reach an integer division on the host, which killed the process
# with SIGFPE instead of raising.


def test_moreh_group_norm_num_groups_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(2, 32, 32, 32).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.operations.moreh.group_norm(x, 0, 1e-5)


def test_moreh_fold_stride_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 36, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "stride must be greater than 0"):
        ttnn.operations.moreh.fold(
            x, None, output_size=[8, 8], kernel_size=[3, 3], dilation=[1, 1], padding=[1, 1], stride=[0, 1]
        )
