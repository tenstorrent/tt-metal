# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ConcatHeadsDecode (custom op) vs the shipped reshard + nlp_concat_heads_decode + sharded_to_interleaved."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.concat_heads_decode import ConcatHeadsDecode
from models.demos.gemma4.tt.attention.operations import concat_heads

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


@pytest.mark.parametrize("hg", [None, 1, 2, 99], ids=["hg_auto", "hg1", "hg2", "hg_all"])
@pytest.mark.parametrize(
    "B, H, D",
    [
        (32, 4, 256),
        (1, 4, 256),
        (8, 16, 128),
        (32, 2, 256),
        (32, 8, 256),
        (32, 1, 512),
        (32, 4, 512),
        # dense per-chip head counts: 31B tp4 (8 heads, D 256 / 512), 12B tp8 (2 heads), batch 1 and 8 rows
        (32, 8, 512),
        (1, 8, 256),
        (1, 8, 512),
        (8, 8, 256),
        (32, 2, 512),
        (1, 2, 256),
    ],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_concat_heads_decode_matches_composite(mesh_device, device_params, B, H, D, hg):
    torch.manual_seed(0)
    x = torch.randn(1, B, 32, D).to(torch.bfloat16)  # heads padded to 32 rows (rows >= H: padding)
    golden = ConcatHeadsDecode.golden(x.float(), H)
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ref = ttnn.to_torch(concat_heads(tx, is_decode_mode=True, num_heads=H, head_dim=D, mesh_device=mesh_device)).float()
    out = ttnn.to_torch(ConcatHeadsDecode.op(tx, H, heads_per_unit=hg)).float()
    assert list(out.shape) == [1, 1, B, H * D]
    assert torch.equal(out, golden), f"max|diff| vs golden {(out - golden).abs().max()}"
    assert torch.equal(out, ref[:, :, :B, :]), f"max|diff| vs composite {(out - ref[:, :, :B, :]).abs().max()}"
    tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    assert torch.equal(ttnn.to_torch(ConcatHeadsDecode.op(tx2, H, heads_per_unit=hg)).float(), out)
