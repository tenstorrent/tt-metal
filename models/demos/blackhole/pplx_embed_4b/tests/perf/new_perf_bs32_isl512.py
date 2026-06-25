# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly pplx-embed-v1-4B perf capture: bs=32 / ISL=512 / DP=1.

ONE measured iteration with signpost markers around the trace-replay forward.
bs=32 ISL=512 (80 MB activation) is DRAM-resident and runs MinimalMatmuls on the
full 130-core (13x10) Blackhole grid via QWEN_MM_GRID=13,10 — the dominant
DRAM-path speedup for the 4B model.

Usage:
    MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/blackhole/pplx_embed_4b/tests/perf/new_perf_bs32_isl512.py -sv

Filter the resulting CSV between the ``start``/``stop`` signposts.
"""

import pytest

from models.demos.blackhole.pplx_embed_4b.demo._common import apply_workload_env, run_perf

BATCH_SIZE = 32
SEQ_LEN = 512

apply_workload_env(BATCH_SIZE, SEQ_LEN)


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
