# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Embedding-4B perf demo, batch=1 / ISL=512 / single device, through the optimized pplx-embed-4B
stack (see ``_common.py``). Times the extended trace (forward + pooling + I/O in one replay) by default,
which is the mode the README numbers use.

Usage:
    pytest models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py -sv
    python models/demos/blackhole/qwen3_embedding_4b/demo/demo_bs1_isl512.py [--iterations N] [--device-id D]
"""

import pytest

from models.demos.blackhole.qwen3_embedding_4b.demo._common import apply_workload_env, run_perf, standalone_main

BATCH_SIZE = 1
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_workload_env(BATCH_SIZE, SEQ_LEN)


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
    parser.add_argument(
        "--full-pipeline",
        dest="full_pipeline",
        action="store_true",
        default=True,
        help="Time forward + pooling + I/O in one traced replay (default; the mode the README numbers use)",
    )
    parser.add_argument(
        "--no-full-pipeline",
        dest="full_pipeline",
        action="store_false",
        help="Time only the bare forward trace replay",
    )
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id, full_pipeline=args.full_pipeline)
