#!/bin/bash

set -euo pipefail

test_file="tests/ttnn/unit_tests/operations/test_noc_atomic_barriers_single_device.py"
test_cases=(
    "${test_file}::test_block_sharded_2d_matmul_flushes_receiver_atomics"
    "${test_file}::test_sharded_norm_flushes_receiver_atomics[operation_name=layer_norm]"
    "${test_file}::test_sharded_norm_flushes_receiver_atomics[operation_name=rms_norm]"
    "${test_file}::test_noncausal_sdpa_flushes_receiver_atomics"
    "${test_file}::test_overlapping_move_flushes_worker_atomics[tile_layout]"
    "${test_file}::test_overlapping_move_flushes_worker_atomics[row_major_layout]"
    "${test_file}::test_conv2d_flushes_receiver_atomics[height_sharded]"
    "${test_file}::test_conv2d_flushes_receiver_atomics[width_sharded]"
    "${test_file}::test_conv2d_flushes_receiver_atomics[block_sharded]"
    "${test_file}::test_interleaved_group_norm_flushes_receiver_atomics[tile_reduction]"
    "${test_file}::test_interleaved_group_norm_flushes_receiver_atomics[two_pass]"
    "${test_file}::test_sharded_group_norm_flushes_receiver_atomics[tile_reduction]"
    "${test_file}::test_sharded_group_norm_flushes_receiver_atomics[two_pass]"
)

for test_case in "${test_cases[@]}"; do
    TT_METAL_NOC_DEBUG_DUMP=1 pytest -xv "${test_case}"
done
