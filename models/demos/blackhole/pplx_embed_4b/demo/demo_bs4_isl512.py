# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
pplx-embed-v1-4B perf demo, fixed at batch=4 / ISL=512.

This is the throughput-optimal config.  Activation = 4 * 512 * 2560 * 2 = 10.5 MB,
which fits the P150 batched-L1 prefill cap (12 MiB), so activations stay L1-resident
instead of round-tripping through DRAM.  Measured on this P150: 74.8 ms vs 90.8 ms
on the DRAM path (+21%, 27.4k tok/s — higher than bs8/bs32).  The per-op memory
snapshot showed the DRAM path leaving ~1.45 MB/core of L1 idle; this config uses it.

Usage:
    pytest models/demos/blackhole/pplx_embed_4b/demo/demo_bs4_isl512.py -sv
    python models/demos/blackhole/pplx_embed_4b/demo/demo_bs4_isl512.py
"""

import pytest

from models.demos.blackhole.pplx_embed_4b.demo._common import apply_workload_env, run_perf, standalone_main

BATCH_SIZE = 4
SEQ_LEN = 512
NUM_ITERATIONS = 10

apply_workload_env(BATCH_SIZE, SEQ_LEN)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_bs4_isl512(mesh_device, is_ci_env):
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

    parser = argparse.ArgumentParser(description="pplx-embed-v1-4B bs=4 ISL=512 perf demo")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Measure full generator pipeline latency instead of direct trace replay",
    )
    args = parser.parse_args()
    standalone_main(BATCH_SIZE, SEQ_LEN, args.iterations, args.device_id, full_pipeline=args.full_pipeline)
