# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly Qwen3-Embedding-0.6B perf capture: bs=32 / ISL=512 / DP=1.

ONE measured iteration with signpost markers around the trace-replay forward.
bs=32 keeps activations DRAM-resident (L1 budget would be exceeded) and turns
on `QWEN_MM_BIG_GRID_BH=1` to push MinimalMatmuls onto the (8,10)=80-core grid
on Blackhole. Same config as `demo_bs32_isl512.py`, including the head-split
NlpCreateHeads/NLPConcatHeads paths.

Usage:
    HF_MODEL=Qwen/Qwen3-Embedding-0.6B MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs32_isl512.py -sv

Filter the resulting CSV between the `start`/`stop` signposts.
"""

import os

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf

BATCH_SIZE = 32
SEQ_LEN = 512

apply_recommended_env(batched_l1=False)
os.environ.setdefault("QWEN_MM_BIG_GRID_BH", "1")


@pytest.mark.parametrize(
    "device_params",
    # Same device_params shape as the standalone demo (see new_perf_bs1 note).
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
