# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


@pytest.mark.timeout(30)
def test_attn_matmul_program_cache_override_rtargs(device):
    torch.manual_seed(0)

    # Shapes mirror existing unit test coverage to ensure the same compiled program
    q_len = 1
    q_heads = 10
    batch = 64
    K = 96
    kv_heads = 1
    seq_len = 32

    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.bfloat16
    out_dtype = ttnn.bfloat16

    a_shape = [q_len, q_heads, batch, K]
    b_shape = [batch, kv_heads, K, seq_len]

    compute_grid_size = device.compute_with_storage_grid_size()

    # 1) First run compiles and seeds the cache
    logger.debug("Executing first run")
    a1 = torch.randn(a_shape).bfloat16()
    b1 = torch.randn(b_shape).bfloat16()
    tt_a1 = ttnn.Tensor(a1, in0_dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_b1 = ttnn.Tensor(b1, in1_dtype).to(ttnn.TILE_LAYOUT).to(device)

    logger.debug("Launching OP for first run")
    out1 = ttnn.experimental.attn_matmul(
        tt_a1,
        tt_b1,
        compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=out_dtype,
    )
    logger.debug("Finished OP for first run")

    # Validate correctness
    out1_host = out1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden1 = (a1.transpose(0, 2) @ b1).transpose(0, 2)
    ok1, pcc1 = comp_pcc(out1_host, golden1)
    logger.debug(f"First run PCC: ok={ok1}, pcc={pcc1}")
    assert ok1, f"First run PCC failed: {pcc1}"

    # 2) Second run hits cache and triggers override path with new buffers
    logger.debug("Executing second run (cache-hit expected)")
    a2 = torch.randn(a_shape).bfloat16()
    b2 = torch.randn(b_shape).bfloat16()
    tt_a2 = ttnn.Tensor(a2, in0_dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_b2 = ttnn.Tensor(b2, in1_dtype).to(ttnn.TILE_LAYOUT).to(device)

    logger.debug("Launching OP for second run (cache-hit expected)")
    out2 = ttnn.experimental.attn_matmul(
        tt_a2,
        tt_b2,
        compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=out_dtype,
    )
    logger.debug("Finished OP for second run")

    # Expect a failure on cache-hit due to incorrect override of rt args (addresses, arg order, CB size)
    out2_host = out2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden2 = (a2.transpose(0, 2) @ b2).transpose(0, 2)
    ok2, pcc2 = comp_pcc(out2_host, golden2)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")
    # Let this assertion FAIL on cache-hit path (or test may timeout if the kernel hangs)
    assert ok2, "PCC mismatch on cache-hit path (expected failure if override runtime args are incorrect)"
