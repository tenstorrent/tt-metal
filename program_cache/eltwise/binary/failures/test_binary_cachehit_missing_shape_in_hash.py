# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc


@pytest.mark.timeout(30)
def test_eltwise_binary_program_cache_missing_shape_in_hash(device):
    """
    Two-run cache test targeting eltwise binary equal-shape path.

    Intent: Show that the program hash omits input/output shapes while the compiled program encodes
    TensorAccessorArgs as compile-time args. On cache-hit with a different shape but the same volume,
    stale compile-time args are reused, causing incorrect addressing.

    Affected code:
    - Hash (shapes omitted):
      - Binary: compute_program_hash() doesn't include tensor shapes
    - Compiled reader/writer use TensorAccessorArgs at creation, not updated on override.
    """

    torch.manual_seed(0)

    dtype = ttnn.bfloat16
    memcfg = ttnn.DRAM_MEMORY_CONFIG

    # Shapes with same volume but different geometry (Ht,Wt swap)
    shape_run1 = [1, 1, 32, 64]  # H=32, W=64 => Ht=1, Wt=2
    shape_run2 = [1, 1, 64, 32]  # H=64, W=32 => Ht=2, Wt=1

    logger.debug("Executing first run (compile & seed cache)")
    a1 = torch.randn(shape_run1).bfloat16()
    b1 = torch.randn(shape_run1).bfloat16()
    tt_a1 = ttnn.Tensor(a1, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg)
    tt_b1 = ttnn.Tensor(b1, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg)

    num_cache_start = device.num_program_cache_entries()
    out1_dev = ttnn.add(tt_a1, tt_b1)
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    out1 = out1_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden1 = (a1 + b1).to(torch.bfloat16)
    ok1, pcc1 = comp_pcc(out1, golden1)
    logger.debug(f"First run PCC: ok={ok1}, pcc={pcc1}")
    assert ok1, f"First run PCC failed: {pcc1}"

    logger.debug("Executing second run (cache-hit expected with different shape, same volume)")
    a2 = torch.randn(shape_run2).bfloat16()
    b2 = torch.randn(shape_run2).bfloat16()
    tt_a2 = ttnn.Tensor(a2, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg)
    tt_b2 = ttnn.Tensor(b2, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg)

    out2_dev = ttnn.add(tt_a2, tt_b2)
    out2 = out2_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden2 = (a2 + b2).to(torch.bfloat16)
    ok2, pcc2 = comp_pcc(out2, golden2)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")

    # Expect PCC mismatch on cache-hit due to stale compile-time TensorAccessorArgs
    assert ok2, "PCC mismatch on cache-hit path: shapes not hashed but used in compile-time args"
