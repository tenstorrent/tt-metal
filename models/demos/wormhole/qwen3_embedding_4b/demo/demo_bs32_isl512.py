# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-4B perf demo, fixed at batch=32 / ISL=512 / DP=1 / single device.

bs=32 ISL=512 has a 16384-token activation (32*512*2560*2 = 80 MB at bf16) which
far exceeds the P150 L1 budget, so this entry point intentionally keeps
activations DRAM-resident and leans on:
  - BFP4/BFP8 weight and activation precision reductions
  - MinimalMatmul on (13,10)=130-core grid (`QWEN_MM_GRID=13,10`), an 18%
    speedup over the default (8,10)=80-core grid
  - Head-split NlpCreateHeads/NLPConcatHeads (helps marginally at high batch but
    non-negative)
  - RoPE cos/sin tables in L1

LN block-sharding auto-disables for this shape (dim=2560 already exceeds the
per-core budget, and bs=32 pushes it even further).

Usage:
    pytest models/demos/wormhole/qwen3_embedding_4b/demo/demo_bs32_isl512.py -sv
    python models/demos/wormhole/qwen3_embedding_4b/demo/demo_bs32_isl512.py
"""


import pytest

from models.demos.wormhole.qwen3_embedding_4b.demo._common import apply_recommended_env, run_perf, standalone_main

BATCH_SIZE = 32
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_recommended_env(batched_l1=False)


@pytest.mark.parametrize(
    "device_params",
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

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-4B bs=32 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id)
