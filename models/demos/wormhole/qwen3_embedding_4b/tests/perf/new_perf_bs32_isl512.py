# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly Qwen3-Embedding-4B perf capture: bs=32 / ISL=512 / DP=1.

ONE measured iteration with signpost markers around the trace-replay forward.
bs=32 keeps activations DRAM-resident (80 MB activation exceeds L1 budget) and
uses `QWEN_MM_GRID=13,10` to push MinimalMatmuls onto the full (13,10)=130-core
grid on Blackhole (18% faster than 80-core). Same config as `demo_bs32_isl512.py`.

Usage:
    MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/wormhole/qwen3_embedding_4b/tests/perf/new_perf_bs32_isl512.py -sv

Filter the resulting CSV between the `start`/`stop` signposts.
"""


import pytest

from models.demos.wormhole.qwen3_embedding_4b.demo._common import apply_recommended_env, run_perf

BATCH_SIZE = 32
SEQ_LEN = 512

apply_recommended_env(batched_l1=False)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs32_isl512_tracy(mesh_device):
    run_perf(
        mesh_device,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_iterations=1,
        emit_signposts=True,
        is_ci_env=False,
    )
