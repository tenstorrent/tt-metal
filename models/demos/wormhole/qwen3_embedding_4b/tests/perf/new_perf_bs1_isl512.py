# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly Qwen3-Embedding-4B perf capture: bs=1 / ISL=512 / DP=1.

ONE measured iteration with `tracy.signpost("start"/"stop")` markers around the
trace-replay forward. The signposted zone is what you filter on in Tracy/the CSV
post-processor to get a clean device-time number.

Uses the same all-optimizations-on environment as `demo_bs1_isl512.py`,
including head-split NlpCreateHeads/NLPConcatHeads (~128 cores), BFP4 weights,
BFP8 residual stream, and RoPE tables in L1.

Usage (Tracy device profile):
    MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/wormhole/qwen3_embedding_4b/tests/perf/new_perf_bs1_isl512.py -sv

Filter the resulting `ops_perf_results_*.csv` between `start`/`stop` signposts.
"""

import pytest

from models.demos.wormhole.qwen3_embedding_4b.demo._common import apply_recommended_env, run_perf

BATCH_SIZE = 1
SEQ_LEN = 512

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs1_isl512_tracy(mesh_device):
    run_perf(
        mesh_device,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_iterations=1,
        emit_signposts=True,
        is_ci_env=False,
    )
