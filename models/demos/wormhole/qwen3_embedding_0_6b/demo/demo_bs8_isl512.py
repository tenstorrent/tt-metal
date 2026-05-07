# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B perf demo, fixed at batch=8 / ISL=512 / DP=1 / single device.

`TT_BATCHED_L1_PREFILL=1` is set by default — the 4096-token activation
(8 * 512 * 1024 * 2 B = 8 MB) fits in P150 L1 with the 8 MiB cap and yields
a ~21% speedup vs. DRAM-resident activations. All other recommended knobs are
also defaulted on (BFP4 weights, BFP8 outputs, RoPE in L1, LN block-shard, and
head-split NlpCreateHeads/NLPConcatHeads). Head-split is roughly neutral at bs=8
because the generic path already has enough sequence/batch blocks, but keeping
it on makes the workload match the bs=1/bs=32 optimized path.

Usage:
    pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs8_isl512.py -sv
    python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs8_isl512.py
"""

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf, standalone_main

BATCH_SIZE = 8
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
    # See note in demo_bs1_isl512.py: `fabric_config: True` matches demo.py and
    # is required to keep batched-prefill trace capture from over-expanding the
    # matmul grid into L1-clashing core ranges.
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs8_isl512(mesh_device, is_ci_env):
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

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B bs=8 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id)
