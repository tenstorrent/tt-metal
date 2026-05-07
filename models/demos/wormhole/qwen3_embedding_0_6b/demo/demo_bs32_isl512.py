# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B perf demo, fixed at batch=32 / ISL=512 / DP=1 / single device.

bs=32 ISL=512 has a 16384-token activation (32 MB at bf16) which exceeds the
P150 L1 budget for `TT_BATCHED_L1_PREFILL=1`, so this entry point intentionally
leaves activations DRAM-resident and instead leans on the BFP4/BFP8/RoPE-in-L1
+ MinimalMatmul-on-(8,10)-grid path (`QWEN_MM_BIG_GRID_BH=1`). The shared
optimization setup also enables head-split NlpCreateHeads/NLPConcatHeads; bs=32
already has many sequence/batch work units, so the gain is small but
non-negative.

LN block-sharding auto-disables for this shape (per-core block_h=64 > 16) so
QWEN_LN_BLOCK_SHARDED=1 is set but inert.

Usage:
    pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs32_isl512.py -sv
    python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs32_isl512.py
"""

import os

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf, standalone_main

BATCH_SIZE = 32
SEQ_LEN = 512
NUM_ITERATIONS = 10

# bs=32 explicitly does NOT use TT_BATCHED_L1_PREFILL — activations stay in
# DRAM. The other knobs all apply.
apply_recommended_env(batched_l1=False)
# DRAM-resident activations free up the per-core L1 budget, so we can push the
# MinimalMatmul grid from (8,8)=64 cores to (8,10)=80 cores on Blackhole. This
# is the only knob in the cohort that's bs=32-specific (auto-disabled for L1
# batches via the same code path that gates LN sharding).
os.environ.setdefault("QWEN_MM_BIG_GRID_BH", "1")


@pytest.mark.parametrize(
    "device_params",
    # See note in demo_bs1_isl512.py for why `fabric_config: True` is required.
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs32_isl512(mesh_device, is_ci_env):
    run_perf(
        mesh_device,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_iterations=NUM_ITERATIONS,
        emit_signposts=False,
        is_ci_env=is_ci_env,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B bs=32 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id)
