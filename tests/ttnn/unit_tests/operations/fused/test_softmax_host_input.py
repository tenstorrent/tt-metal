# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


ENTRIES = {
    "softmax": lambda t: ttnn.softmax(t, dim=-1),
    "scale_mask_softmax": lambda t: ttnn.scale_mask_softmax(t),
    "softmax_in_place": lambda t: ttnn.softmax_in_place(t),
    "scale_mask_softmax_in_place": lambda t: ttnn.scale_mask_softmax_in_place(t),
    "scale_causal_mask_hw_dims_softmax_in_place": lambda t: ttnn.scale_causal_mask_hw_dims_softmax_in_place(t),
}


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_softmax_rejects_host_tensor(entry, expect_error):
    host = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "move input_tensor to device"):
        ENTRIES[entry](host)
