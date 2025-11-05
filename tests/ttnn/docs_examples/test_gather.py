# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


def test_gather_example(device):
    ttnn_input = ttnn.rand([4, 4], ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.rand([4, 2], ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn.gather(ttnn_input, 1, index=ttnn_index)
