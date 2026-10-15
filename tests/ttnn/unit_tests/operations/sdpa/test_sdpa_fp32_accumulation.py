# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPAProgramConfig(fp32_accumulation=True) on SDPA decode, post-commit: short contexts that also fit the simulators.

Long-context accuracy (8K-131K, where the default path drifts most) is covered by
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_fp32_accumulation_long.py.
"""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.fp32_accumulation_test_utils import (
    compute_config,
    program_config,
    run_paged_decode,
    run_unpaged_decode,
)

# Relative L2 vs FP64 on these shapes (Blackhole): 2.0-2.2% by default vs 0.26% with the option. Each check requires
# the option under BOUND and better than the default by the factor below.
BOUND = 5e-3


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype",
    [
        (1, 8, 2, 4096, 32, 128, 16, ttnn.bfloat8_b),
        (4, 16, 4, 2048, 64, 128, 8, ttnn.bfloat16),
    ],
    ids=["b1_4k", "b4_2k_block64"],
)
def test_paged_decode_fp32_accumulation(
    device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype
):
    run_paged_decode(
        device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype, BOUND, 4.0
    )


@pytest.mark.timeout(300)
def test_unpaged_decode_fp32_accumulation(device):
    run_unpaged_decode(device, 2, 8, 2, 2048, 128, BOUND, 4.0)


@pytest.mark.timeout(120)
def test_fp32_accumulation_rejects_unsupported_configs(device, expect_error):
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(1, 1, 8, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kv = ttnn.from_torch(torch.randn(1, 2, 1024, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos = ttnn.from_torch(torch.tensor([1000], dtype=torch.int32), dtype=ttnn.int32, device=device)
    config = program_config(device, 0, 128, True)

    with expect_error(RuntimeError, "fp32_accumulation requires fp32_dest_acc_en"):
        ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv,
            kv,
            cur_pos_tensor=pos,
            compute_kernel_config=compute_config(device, fp32_dest_acc_en=False),
            program_config=config,
        )
    with expect_error(RuntimeError, "fp32_accumulation supports causal decode only"):
        ttnn.transformer.scaled_dot_product_attention_decode(
            q, kv, kv, is_causal=False, compute_kernel_config=compute_config(device), program_config=config
        )

    approx = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=True,
        fp32_accumulation=True,
    )
    with expect_error(RuntimeError, "fp32_accumulation uses the accurate exponential"):
        ttnn.transformer.scaled_dot_product_attention_decode(
            q, kv, kv, cur_pos_tensor=pos, compute_kernel_config=compute_config(device), program_config=approx
        )

    prefill_q = ttnn.from_torch(
        torch.randn(1, 8, 1024, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "fp32_accumulation is supported by the SDPA decode ops only"):
        ttnn.transformer.scaled_dot_product_attention(
            prefill_q,
            kv,
            kv,
            is_causal=True,
            compute_kernel_config=compute_config(device),
            program_config=program_config(device, 128, 128, True),
        )
