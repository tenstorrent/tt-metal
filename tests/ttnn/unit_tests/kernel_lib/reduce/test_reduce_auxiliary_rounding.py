# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host recipe precision, including normalization rewritten by sequence planning."""

import struct

import pytest
import torch

import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scenario", ["explicit_scale", "partial_average", "sequence_average"])
def test_reduce_auxiliary_host_rounding(dtype, scenario):
    planner = ttnn.reduce_planner
    widths = [32, 64] if scenario == "sequence_average" else [41 if scenario == "partial_average" else 32]
    scalar = 1 / 41 if scenario == "explicit_scale" else None
    sequence = planner.make_reduce_sequence_plan(
        reductions=[
            (
                0,
                planner.ReduceCallConfig(
                    block=planner.ReduceBlockSpec(32, width, dtype, dtype),
                    reduce_math=planner.ReduceMath.SUM if scalar is not None else planner.ReduceMath.AVG,
                    reduce_dim=planner.ReduceDimension.ROW,
                    scalar=scalar,
                    fp32_mode=planner.ReduceFp32Mode.FAST,
                    input_policy=planner.ReduceInputPolicy.NO_WAIT_NO_POP,
                ),
            )
            for width in widths
        ],
        cb_ids=planner.ReduceSequenceCbIds(1, 2, 16),
        hardware=planner.ReduceHardwareConfig(
            arch=ttnn.device.Arch.BLACKHOLE,
            fp32_dest_acc_en=dtype == ttnn.float32,
            dst_full_sync_en=False,
        ),
    )
    value = torch.tensor(scalar if scalar is not None else 1 / sum(widths), dtype=torch.float32)
    expected = value.to(torch.bfloat16).float().item() if dtype == ttnn.bfloat16 else value.item()
    assert sequence.auxiliary.tiles
    assert all(call.plan.algorithm == planner.ReduceAlgorithm.REDUCE_TILE for call in sequence.calls)
    for tile in sequence.auxiliary.tiles:
        assert tile.value == expected
    # The existing device-side truncation must preserve the host-rounded value;
    # FP32 recipes must instead retain the original float's lower mantissa bits.
    encoded_values = sequence.auxiliary_compile_time_args[2::2]
    expected_bits = struct.unpack("I", struct.pack("f", expected))[0]
    assert all(bits == expected_bits for bits in encoded_values)
    if dtype == ttnn.bfloat16:
        assert expected_bits & 0xFFFF == 0
    else:
        assert expected_bits & 0xFFFF != 0
