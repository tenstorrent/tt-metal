# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
pplx-embed-v1-0.6B perf demo, fixed at batch=1 / ISL=512.

All recommended optimizations are turned on by default (same Qwen3 architecture).

Usage:
    # Single device (P150)
    HF_MODEL=perplexity-ai/pplx-embed-v1-0.6b MESH_DEVICE=P150 pytest \
      models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py -sv

    # Standalone (no pytest)
    python models/demos/blackhole/pplx_embed_0_6b/demo/demo_bs1_isl512.py
"""

import pytest

from models.demos.blackhole.pplx_embed_0_6b.demo._common import apply_recommended_env, run_perf, standalone_main

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

    parser = argparse.ArgumentParser(description="pplx-embed-v1-0.6B bs=1 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Measure full generator pipeline latency instead of direct trace replay",
    )
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id, full_pipeline=args.full_pipeline)
