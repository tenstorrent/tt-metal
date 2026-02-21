# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Non-fused LN → RMS → LN for Tracy profiling comparison.

Run with:
    python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/fused/test_unfused_ln_rms_ln.py -svv
"""

import torch
import ttnn
import pytest

from models.common.utility_functions import comp_pcc


def torch_layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def torch_rms_norm(x, weight, eps=1e-5):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    if weight is not None:
        x_norm = x_norm * weight
    return x_norm


def test_unfused_sharded_ln_rms_ln(device):
    """Block-sharded LN → RMS → LN as 3 separate ttnn ops.

    Same setup as TestMatmulFusionChains::test_sharded_ln_rms_ln but
    without fusion, for direct Tracy comparison.
    """
    torch.manual_seed(42)

    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

    # Block-sharded: 4x4 grid, shard (32, 128) = 1 M-tile per core
    shard_spec = ttnn.ShardSpec(cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    torch_input = torch.randn(1, 1, 128, 512, dtype=torch.bfloat16)
    torch_gamma = torch.ones(1, 1, 1, 512, dtype=torch.bfloat16)
    torch_beta = torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem,
    )
    tt_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Sharded program config matching fused test:
    #   shard (32, 128) → block_h=1, block_w=4, subblock_w=4
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4),
        subblock_w=4,  # 128/32 = 4 tiles wide
        block_h=1,  # 32/32  = 1 tile high
        block_w=4,
        inplace=False,
    )
    compute_config = ttnn.layernorm_default_compute_config(device.arch())

    # --- 3 separate ops (non-fused) ---
    out1 = ttnn.layer_norm(
        tt_input,
        weight=tt_gamma,
        bias=tt_beta,
        epsilon=1e-5,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config=sharded_mem,
    )

    out2 = ttnn.rms_norm(
        out1,
        weight=tt_gamma,
        epsilon=1e-5,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config=sharded_mem,
    )

    out3 = ttnn.layer_norm(
        out2,
        weight=tt_gamma,
        bias=tt_beta,
        epsilon=1e-5,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config=sharded_mem,
    )

    result = ttnn.to_torch(out3)

    # Gold
    temp1 = torch_layer_norm(torch_input.float(), torch_gamma.float(), torch_beta.float())
    temp2 = torch_rms_norm(temp1, torch_gamma.float())
    golden = torch_layer_norm(temp2, torch_gamma.float(), torch_beta.float())

    passing, pcc = comp_pcc(golden, result, pcc=0.98)
    assert passing, f"Unfused sharded LN→RMS→LN PCC: {pcc}"
