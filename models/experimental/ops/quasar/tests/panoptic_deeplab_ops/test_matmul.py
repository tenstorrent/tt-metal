# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/panoptic_deeplab_20_cores_oct14_1821/graph_capture.python_io.json --out models/experimental/ops/quasar/tests/panoptic_deeplab_ops --max-cases 64
# Source capture: generated/ttnn/reports/panoptic_deeplab_20_cores_oct14_1821/graph_capture.python_io.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.matmul`` — every distinct call the model made, as captured.

Captured 6 call(s) to this op; 2 distinct signature(s) covering 6 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)
  * the output has more elements than all inputs combined (a batch-padded decode tensor), so the op cannot write all of it; finiteness is asserted over the portion the inputs can account for

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.matmul

CASES = [
    {
        "id": "00_32x512x128_bf8_int-dram",
        "op": "ttnn.matmul",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 512, 128],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 32, 128, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [5, 4],
                    "in0_block_w": 2,
                    "out_subblock_h": 4,
                    "out_subblock_w": 2,
                    "out_block_h": 16,
                    "out_block_w": 2,
                    "per_core_M": 16,
                    "per_core_N": 2,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 0,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 512, 1024],
            },
        ],
    },
    {
        "id": "01_32x128x256_bf8_int-l1",
        "op": "ttnn.matmul",
        "count": 3,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 256],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 256, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [5, 4],
                    "in0_block_w": 2,
                    "out_subblock_h": 4,
                    "out_subblock_w": 2,
                    "out_block_h": 4,
                    "out_block_w": 2,
                    "per_core_M": 4,
                    "per_core_N": 2,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 0,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 1024],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matmul(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
