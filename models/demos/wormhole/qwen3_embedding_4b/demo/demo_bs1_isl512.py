# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-4B perf demo, fixed at batch=1 / ISL=512 / DP=1 / single device.

All recommended optimizations are turned on by default:
  - Weight precision:  QKV/WO -> BFP4, FF1/FF3 -> BFP4 (baseline), FF2 -> BFP8
  - Activation precision: FF1/FF3 output -> BFP8, FFNORM input -> BFP8,
                           post-FFN residual -> BFP8 (full BFP8 residual stream)
  - Head-split NlpCreateHeads/NLPConcatHeads (16 -> 128 work units for bs=1)
  - RoPE cos/sin tables in L1 (128 KB, well within budget)
  - TT_SKIP_KV_CACHE_FILL=1

Note: LN block-sharding auto-disables for 4B (dim=2560 gives block_h*block_w=20
      which exceeds the 16-tile per-core cap). The env var is set but inert.

bs=1 ISL=512 activation = 512 * 2560 * 2 = 2.5 MB -> fits in L1 single-user path
(no TT_BATCHED_L1_PREFILL needed for bs=1).

Usage:
    pytest models/demos/wormhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py -sv
    python models/demos/wormhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py
"""

import pytest

from models.demos.wormhole.qwen3_embedding_4b.demo._common import apply_recommended_env, run_perf, standalone_main

BATCH_SIZE = 1
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_recommended_env(batched_l1=BATCH_SIZE > 1)


@pytest.mark.parametrize(
    "device_params",
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

    parser = argparse.ArgumentParser(description="Qwen3-Embedding-4B bs=1 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id)
