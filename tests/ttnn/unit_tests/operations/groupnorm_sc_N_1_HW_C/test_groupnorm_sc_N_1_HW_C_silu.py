# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""activation="silu": output must equal silu(groupnorm(x)) on the SDXL full-grid shards (TILE and RM)."""
import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C_sharded import (
    block_shard_config,
    torch_groupnorm_n_1_hw_c,
    compute_pcc,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "shape,shard,grid",
    [
        ((1, 1, 16384, 320), [1664, 32], (10, 10)),
        ((1, 1, 4096, 640), [512, 64], (10, 8)),
        ((1, 1, 1024, 1280), [128, 128], (10, 8)),
    ],
    ids=["16384x320", "4096x640", "1024x1280"],
)
def test_fused_silu(device, shape, shard, grid, layout):
    torch.manual_seed(0)
    N, _, HW, C = shape
    x = torch.randn(shape).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    mc = block_shard_config(shard, grid)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = groupnorm_sc_N_1_HW_C(
        tt_x, 32, gamma=ttnn.from_torch(gamma, **kw), beta=ttnn.from_torch(beta, **kw), in_place=True, activation="silu"
    )
    expected = torch.nn.functional.silu(torch_groupnorm_n_1_hw_c(x, 32, gamma=gamma, beta=beta).float())
    actual = ttnn.to_torch(y).float()
    pcc = compute_pcc(actual, expected)
    rms = float(((actual - expected) ** 2).mean().sqrt())
    assert torch.isfinite(actual).all()
    assert pcc >= 0.9995 and rms <= 0.02, f"pcc={pcc} rms={rms}"
