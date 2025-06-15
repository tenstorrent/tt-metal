# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def test_ttnn_where():
    C = torch.ones(4, 4, dtype=torch.float32)
    T = torch.randn(4, 4, dtype=torch.float32)
    F = torch.ones(4, 4, dtype=torch.float32) * 10
    golden = torch.where(C != 0, T, F)
    with ttnn.manage_device(0) as dev:
        ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
        result = ttnn.to_torch(ttnn_result)
        print(result)
    print(golden)


if __name__ == "__main__":
    test_ttnn_where()
