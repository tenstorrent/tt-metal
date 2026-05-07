# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-0.6B perf demo, fixed at batch=1 / ISL=512 / DP=1 / single device.

All recommended optimizations are turned on by default (FF1/FF3 BFP4 baseline +
QKV/WO BFP4 + FF1/FF3 output BFP8 + FFNORM input BFP8 + RoPE tables in L1 + LN
block-sharding + head-split NlpCreateHeads/NLPConcatHeads). The head-split TM
paths are bs=1-critical: each changes the per-layer sequence-only split from 16
work units to 128 head-grouped work units. Each env var is `setdefault`'d so you
can still override one from the shell for A/B knob comparisons.

Usage:
    pytest models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs1_isl512.py -sv
    python models/demos/wormhole/qwen3_embedding_0_6b/demo/demo_bs1_isl512.py
"""

import pytest

from models.demos.wormhole.qwen3_embedding_0_6b.demo._common import apply_recommended_env, run_perf, standalone_main

BATCH_SIZE = 1
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
    # 200MB trace region matches demo.py's pytest fixture; 50MB is fine for
    # bs=1 ISL=512 specifically but 200MB lets users bump iterations or seq_len
    # without re-tuning. l1_small_size and num_command_queues match the
    # standalone path so pytest and `python ...` runs are identical.
    # `fabric_config: True` matches demo.py — without it the device set-up
    # picks different NoC routing and ends up trying to allocate matmul programs
    # on the (8,10) BH grid even when all our knobs target (8,8). That triggers
    # `Statically allocated CBs ... clash with L1 buffers on core range
    # [(0,0)-(7,9)]` during batched-prefill trace capture. Keeping the same
    # device_params shape as demo.py so the trace topology is bit-identical.
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs1_isl512(mesh_device, is_ci_env):
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

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B bs=1 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id)
