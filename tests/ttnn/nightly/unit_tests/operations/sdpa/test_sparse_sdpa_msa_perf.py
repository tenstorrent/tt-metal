# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Performance smoke for the production-scale sparse_sdpa_msa shapes.

Profile with:
    python -m tracy -p -r -v -m pytest tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa_perf.py
then read "DEVICE KERNEL DURATION [ns]" for SparseSDPAMsaOperation.

Per-chip TP-shard shape: H=16, n_kv=1, S=640, T=56320 (nblk=440), topk=16, d=v_dim=128,
block_size=128.
Single-chip native GQA shape: H=64, n_kv=4, S=640, T=56320, topk=16, d=v_dim=128, block_size=128.
"""

import os

import pytest

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import make_msa_inputs, run_op_msa_native

pytestmark = [
    pytest.mark.skipif(os.getenv("CI") == "true", reason="sparse_sdpa_msa perf smoke is skipped on CI for now"),
    pytest.mark.use_module_device,
]


@run_for_blackhole()
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
def test_msa_perf_prod(device, kv_dtype):
    d, H, S, T, topk = 128, 16, 640, 56320, 16
    # Non-causal selection keeps topk=16 valid blocks for every query.
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=7)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype)
    assert tuple(out.shape) == (1, H, S, d)


@run_for_blackhole()
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
def test_msa_perf_prod_single_chip_gqa(device, kv_dtype):
    d, H, n_kv, S, T, topk = 128, 64, 4, 640, 56320, 16
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=7)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype)
    assert tuple(out.shape) == (1, H, S, d)
