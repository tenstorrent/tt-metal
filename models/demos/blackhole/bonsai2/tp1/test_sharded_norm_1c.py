# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_SHARDED_NORM: the sharded single-chip decode norm matches the interleaved one, layout included.
    pytest models/demos/blackhole/bonsai2/tp1/test_sharded_norm_1c.py -s
"""
import pytest
import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.layer import sharded_decode_norm
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.common import Mode

MIN_PCC = 0.9999


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sharded_norm_1c(mesh_device):
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    torch.manual_seed(0)
    norm = RMSNorm(
        device=mesh_device,
        dim=args.dim,
        state_dict={"layers.0.input_layernorm.weight": torch.randn(args.dim) * 0.1},
        weight_key="input_layernorm",
        state_dict_prefix="layers.0.",
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=True,
        eps=args.norm_eps,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, args.dim) * 4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ref = norm(x, mode=Mode.DECODE, norm_config={"output_mem_config": ttnn.L1_MEMORY_CONFIG})
    out = sharded_decode_norm(norm, x, args, ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config() == ref.memory_config()
    assert list(out.shape) == list(ref.shape)
    a, b = ttnn.to_torch(ref).float().flatten(), ttnn.to_torch(out).float().flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"sharded vs interleaved decode norm: pcc {pcc:.6f}, max abs diff {(a - b).abs().max().item():.4g}")
    assert pcc > MIN_PCC
