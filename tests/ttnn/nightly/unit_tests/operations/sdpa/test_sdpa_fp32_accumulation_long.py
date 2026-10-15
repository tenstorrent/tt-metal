# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPAProgramConfig(fp32_accumulation=True) at long contexts, where the default recurrence drifts most.

Measured on Blackhole (relative L2 vs FP64): default decode 2-2.7% and chunked prefill 4.6-59% between 64K and
512K, against 0.18-0.33% with the option. Short-context and validation checks live in the post-commit suite
(tests/ttnn/unit_tests/operations/sdpa/test_sdpa_fp32_accumulation.py).
"""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.fp32_accumulation_test_utils import (
    run_chunked_prefill,
    run_paged_decode,
    run_prefill,
    run_unpaged_decode,
)

BOUND = 5e-3
FACTOR = 4.0  # the option must cut the default error at least fourfold at these lengths


@pytest.mark.parametrize(
    "batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype",
    [
        (1, 8, 2, 131072, 32, 256, 64, ttnn.bfloat8_b),  # long context, all cores on one head
        (8, 8, 2, 16384, 32, 128, 16, ttnn.bfloat16),  # batched, ragged positions
        (4, 16, 4, 8192, 64, 128, 8, ttnn.bfloat8_b),  # 64-token pages, more Q heads
    ],
    ids=["b1_131k", "b8_16k", "b4_8k_block64"],
)
def test_paged_decode_fp32_accumulation_long(
    device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype
):
    run_paged_decode(
        device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype, BOUND, FACTOR
    )


def test_unpaged_decode_fp32_accumulation_long(device):
    run_unpaged_decode(device, 2, 8, 2, 8192, 256, BOUND, FACTOR)


@pytest.mark.parametrize("start_as_tensor", [False, True], ids=["scalar_start", "tensor_start"])
@pytest.mark.parametrize(
    "start, chunk, q_chunk_size, k_chunk_size",
    [(61440, 2048, 64, 512), (126976, 4096, 128, 256)],
    ids=["60k", "124k"],
)
def test_chunked_prefill_fp32_accumulation_long(device, start, chunk, q_chunk_size, k_chunk_size, start_as_tensor):
    run_chunked_prefill(device, start, chunk, q_chunk_size, k_chunk_size, start_as_tensor, BOUND, FACTOR)


@pytest.mark.parametrize("use_sink", [False, True], ids=["no_sink", "sink"])
def test_prefill_fp32_accumulation_long(device, use_sink):
    # Standard prefill gains less at 8K than chunked prefill at 60K+; require only an improvement.
    run_prefill(device, 8192, 128, 256, use_sink, BOUND, 1.0)
