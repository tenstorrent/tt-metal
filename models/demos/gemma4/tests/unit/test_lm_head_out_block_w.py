# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""decode_matmul.lm_head_out_block_w_config: an explicit out_block_w on the full-vocab (262144) decode lm_head.

The config restates ttnn's automatic one except for out_block_w, which only splits N into output passes, so the
logits must be bit-identical to the automatic config. Single chip, no checkpoint (random 12B-shaped weights)."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.decode_matmul import lm_head_out_block_w_config

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
VOCAB = 262144


def test_out_block_w_must_divide_per_core_n(expect_error):
    grid = SimpleNamespace(x=11, y=10)  # 110 cores: per_core_N = ceil(ceil(262144 / 110) / 32) = 75
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: grid)
    with expect_error(ValueError, "must divide per_core_N=75"):
        lm_head_out_block_w_config(device, 1024, VOCAB, 4, ttnn.bfloat16, ttnn.bfloat16)


@pytest.mark.parametrize("k", [1024, 3840], ids=["assistant12b", "target12b"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_out_block_w_matches_automatic_config(mesh_device, device_params, k):
    grid = mesh_device.compute_with_storage_grid_size()
    per_core_n = -(-(-(-VOCAB // (grid.x * grid.y))) // ttnn.TILE_SIZE)
    widths = [d for d in (1, 3, 5, per_core_n) if per_core_n % d == 0]
    g = torch.Generator().manual_seed(k)
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, k, generator=g).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    w = ttnn.from_torch(
        (torch.randn(1, 1, k, VOCAB, generator=g) * 0.02).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    auto = ttnn.to_torch(ttnn.linear(x, w))
    for out_block_w in widths:
        pc, ckc = lm_head_out_block_w_config(mesh_device, k, VOCAB, out_block_w, ttnn.bfloat16, ttnn.bfloat16)
        got = ttnn.to_torch(ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc))
        assert torch.equal(got, auto), f"out_block_w={out_block_w}: logits differ from the automatic config"
