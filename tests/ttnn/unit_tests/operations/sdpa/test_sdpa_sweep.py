# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SDPA shape-sweep harness.

Covers decode + prefill across real model presets (Whisper / Llama2/3 / Mistral /
Qwen / Gemma) plus non-tile-aligned and long-context edge cases. Designed to be
runnable in parallel with `pytest -n N --dist worksteal` — every parametrize
point creates its own device, so workers are independent.

Filter examples:
  pytest -k "decode and llama3"
  pytest -k "prefill and (long or mistral)"
  pytest -m "not slow"            # skip 64k+ context entries

See scripts/run_sdpa_sweep.sh for the recommended invocation.
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    run_test_sdpa_decode_single_iter,
)
from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill import run_test_sdpa_tt


# ---------------------------------------------------------------------------
# Decode presets
# ---------------------------------------------------------------------------
# (b, nh, nkv, s, d, grid_size)
DECODE_CASES = [
    pytest.param(1, 12, 12, 1024, 64, (8, 4), id="gpt2_medium"),
    pytest.param(2, 20, 20, 512, 64, (8, 8), id="whisper_large__nh20"),
    pytest.param(1, 32, 32, 4096, 128, (8, 8), id="llama2_7b_mha"),
    pytest.param(32, 8, 1, 32768, 128, (8, 6), id="llama2_70b_mqa_32k"),
    pytest.param(1, 32, 8, 8192, 128, (8, 8), id="llama3_8b_gqa4"),
    pytest.param(8, 16, 4, 4096, 128, (8, 2), id="llama31_8b_n300"),
    pytest.param(1, 32, 8, 32768, 128, (8, 8), id="mistral_7b_32k"),
    pytest.param(1, 28, 4, 32768, 128, (8, 8), id="qwen2_7b_gqa7"),
    pytest.param(1, 16, 16, 8192, 256, (8, 8), id="gemma_7b_d256"),
    pytest.param(1, 32, 8, 131072, 128, (8, 8), id="llama31_8b_long131k", marks=pytest.mark.slow),
    # GQA-ratio sweep at a common base shape
    pytest.param(1, 8, 1, 2048, 128, (8, 4), id="gqa_ratio8_mqa"),
    pytest.param(1, 8, 2, 2048, 128, (8, 4), id="gqa_ratio4"),
    pytest.param(1, 8, 4, 2048, 128, (8, 4), id="gqa_ratio2"),
    pytest.param(1, 8, 8, 2048, 128, (8, 4), id="gqa_ratio1_mha"),
]


# ---------------------------------------------------------------------------
# Prefill presets
# ---------------------------------------------------------------------------
# (b, nh, nkv, s, d)
PREFILL_CASES = [
    pytest.param(1, 8, 1, 2048, 128, id="llama2_7b_mqa_2k"),
    pytest.param(1, 32, 8, 8192, 128, id="llama3_8b_8k"),
    pytest.param(1, 32, 8, 16384, 128, id="mistral_7b_16k"),
    pytest.param(1, 32, 8, 32768, 128, id="mistral_7b_32k"),
    pytest.param(1, 16, 16, 1536, 64, id="whisper_enc_1536"),
    pytest.param(1, 16, 16, 8192, 256, id="gemma_7b_d256"),
    pytest.param(1, 32, 8, 65536, 128, id="long_64k", marks=pytest.mark.slow),
    pytest.param(1, 32, 8, 131072, 128, id="long_131k", marks=pytest.mark.slow),
]

# (q_chunk_size, k_chunk_size)
PREFILL_CHUNK_CASES = [
    pytest.param(128, 128, id="q128_k128"),
    pytest.param(256, 256, id="q256_k256"),
    pytest.param(32, 128, id="q32_k128"),
]


# ---------------------------------------------------------------------------
# Decode sweep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kv_dtype, q_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat8_b, ttnn.bfloat16),
    ],
    ids=["kv_bf16", "kv_bfp8"],
)
@pytest.mark.parametrize("b, nh, nkv, s, d, grid_size", DECODE_CASES)
@pytest.mark.timeout(900)
def test_decode_sweep(device, b, nh, nkv, s, d, grid_size, kv_dtype, q_dtype):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 (GQA/MHA) requires q_dtype=bfloat16")
    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        kv_dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor=True,
    )


# ---------------------------------------------------------------------------
# Prefill sweep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bfp8"],
)
@pytest.mark.parametrize("q_chunk_size, k_chunk_size", PREFILL_CHUNK_CASES)
@pytest.mark.parametrize("b, nh, nkv, s, d", PREFILL_CASES)
@pytest.mark.timeout(1800)
def test_prefill_sweep(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if s % q_chunk_size != 0 or s % k_chunk_size != 0:
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    rmse_threshold = 0.0092 if dtype == ttnn.bfloat8_b else 0.0093
    run_test_sdpa_tt(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        rmse_threshold=rmse_threshold,
    )
