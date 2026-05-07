# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly Qwen3-Embedding-0.6B perf capture: bs=1 / ISL=512 / DP=1.

Designed to mirror BGE-M3's `tests/perf/new_perf.py`: ONE measured iteration
with `tracy.signpost("start"/"stop")` markers around the trace-replay forward.
The signposted zone is what you filter on in Tracy/the CSV post-processor to
get a clean device-time number — uncluttered by the compile pass.

Uses the same all-optimizations-on environment as `demo_bs1_isl512.py`,
including `QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1` and
`QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1`, so Tracy should show both TM ops at ~128
cores rather than the generic 16-core sequence-only split.

Usage (Tracy device profile):
    HF_MODEL=Qwen/Qwen3-Embedding-0.6B MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs1_isl512.py -sv

Then filter the resulting `ops_perf_results_*.csv` to ops between the `start`
and `stop` signposts (Tracy GUI: zone view; CLI: process_ops_perf with
--signpost-start start --signpost-stop stop). The summed device kernel time
inside that window should match the demo's reported best/avg prefill time.
"""

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf

BATCH_SIZE = 1
SEQ_LEN = 512

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
    # Match demo.py / demo_bs1_isl512.py exactly: `fabric_config: True` is
    # required (without it batched-prefill trace capture clashes with L1
    # activation buffers on the (8,10) core range), 200 MB trace region matches
    # the demo path so swapping between this and the standalone demo doesn't
    # change device topology.
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs1_isl512_tracy(mesh_device):
    # Single iteration: signposts wrap exactly one forward, so the captured
    # zone is one trace replay's worth of device kernels. Don't bump this above
    # 1 unless you want to cross-check against the demo.py num_iterations=3
    # path — extra iters dilute the zone with extra forwards.
    run_perf(
        mesh_device,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_iterations=1,
        emit_signposts=True,
        is_ci_env=False,
    )
