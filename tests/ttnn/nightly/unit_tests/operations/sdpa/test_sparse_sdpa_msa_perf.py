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
from loguru import logger

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import make_msa_inputs, run_op_msa_native


def _profile_msa_native_duration_ns(device, run_fn):
    """Run the msa op under the real-time device program profiler and return its device kernel duration (ns).
    SparseSDPAMsaOperation dominates the prod shape by >100x; input tilize / output typecast records are
    negligible, so the max device duration across the collected records is the msa-attention kernel time."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail(
            "Real-time profiler must be active for sparse_sdpa_msa perf checks (run with TT_METAL_DEVICE_PROFILER=1)"
        )
    _, records = profile_realtime_program(device, run_fn, collect_all=True)
    return max(record["duration_ns"] for record in records), len(records)


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

    result = {}

    def run():
        result["out"] = run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype)

    duration_ns, n_records = _profile_msa_native_duration_ns(device, run)
    logger.info(
        f"sparse_sdpa_msa perf prod kv={kv_dtype}: device_kernel_duration={duration_ns / 1e3:.3f} us (records={n_records})"
    )
    assert tuple(result["out"].shape) == (1, H, S, d)


@run_for_blackhole()
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
def test_msa_perf_prod_single_chip_gqa(device, kv_dtype):
    d, H, n_kv, S, T, topk = 128, 64, 4, 640, 56320, 16
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=7)

    result = {}

    def run():
        result["out"] = run_op_msa_native(q, k, v, indices, device, kv_dtype=kv_dtype)

    duration_ns, n_records = _profile_msa_native_duration_ns(device, run)
    logger.info(
        f"sparse_sdpa_msa perf prod_single_chip_gqa kv={kv_dtype}: "
        f"device_kernel_duration={duration_ns / 1e3:.3f} us (records={n_records})"
    )
    assert tuple(result["out"].shape) == (1, H, S, d)
