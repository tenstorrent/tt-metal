# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tracy-friendly Qwen3-Embedding-0.6B perf capture: bs=8 / ISL=512 / DP=1.

ONE measured iteration with signpost markers, exactly like the BGE-M3
`new_perf.py` pattern. `TT_BATCHED_L1_PREFILL=1` is set by default so
activations stay L1-resident through the prefill — same config as the
matching standalone `demo_bs8_isl512.py`, including the head-split
NlpCreateHeads/NLPConcatHeads paths.

Usage:
    HF_MODEL=Qwen/Qwen3-Embedding-0.6B MESH_DEVICE=P150 \\
      TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/wormhole/qwen3_embedding_0_6b/tests/perf/new_perf_bs8_isl512.py -sv

Filter the resulting CSV between the `start`/`stop` signposts.
"""

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf

BATCH_SIZE = 8
SEQ_LEN = 512

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
    # Same device_params shape as the standalone demo (see new_perf_bs1 note).
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs8_isl512_tracy(mesh_device):
    run_perf(
        mesh_device,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_iterations=1,
        emit_signposts=True,
        is_ci_env=False,
    )
