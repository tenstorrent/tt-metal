# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    UnarySFPUGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DstSync,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test

# -----------------------------------------------------------------------------
# 1. Parameter space definition
# -----------------------------------------------------------------------------

# Reuse pack/untilize-supported formats
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Bfp8_b,  # allowed only as input
    # DataFormat.Float32, TODO: Model fp32 -> tf32 truncation when doing eltwise binary on them
]

# Binary FPU operations under test
binary_ops = [
    MathOperation.Elwadd,
    MathOperation.Elwsub,
    MathOperation.Elwmul,
]

# Unary SFPU operations to apply after binary
unary_ops = [
    # MathOperation.Reciprocal,
    # MathOperation.Silu,
    # MathOperation.Celu,
    # MathOperation.Cos,
    # MathOperation.Sin,
    # MathOperation.Elu,
    # MathOperation.Exp,
    # MathOperation.Exp2,
    # MathOperation.Log,
    MathOperation.Square,
    MathOperation.Abs,
    MathOperation.Gelu,
    MathOperation.Hardsigmoid,
    MathOperation.Neg,
    MathOperation.Sqrt,
]

dst_sync_options = [DstSync.SyncHalf, DstSync.SyncFull]

# Generate format combinations (same input/output) and add a few mixed cases
_test_formats = input_output_formats(supported_formats, same=False)

all_params = [
    {
        "testname": "eltwise_binary_sfpu_pack_untilize",
        "formats": fmt,
        "dest_acc": dest_acc,
        "mathop": bin_op,
        "unary_op": un_op,
        "dst_sync": dst_sync,
        "math_fidelity": fidelity,
    }
    for fmt in _test_formats
    for dest_acc in [DestAccumulation.Yes, DestAccumulation.No]
    for bin_op in binary_ops
    for un_op in unary_ops
    for dst_sync in dst_sync_options
    for fidelity in [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]
]

param_ids = [
    f"bin={p['mathop'].name}|un={p['unary_op'].name}|fmt={p['formats'].input_format.name}->{p['formats'].output_format.name}|acc={p['dest_acc'].name}|sync={p['dst_sync'].name}|fid={p['math_fidelity'].name}"
    for p in all_params
]


# -----------------------------------------------------------------------------
# 2. Test implementation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", all_params, ids=param_ids)
def test_sweep_test(config):
    """Runs the C++ eltwise_binary_pack_untilize.cpp kernel for all parameter combinations.
    Single 32×32 tile only.
    """

    formats = config["formats"]
    mathop = config["mathop"]  # binary op
    unary_op = config["unary_op"]
    dest_acc = config["dest_acc"]
    dst_sync = config["dst_sync"]
    math_fidelity = config["math_fidelity"]

    # Skip unsupported output Bfp8_b
    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack untilize does not support Bfp8_b as output format")

    if mathop == MathOperation.Elwmul and unary_op == MathOperation.Square:
        pytest.skip("Elwmul and Square are not supported together")

    # Fidelity irrelevant for add/sub – mirror behaviour of other tests
    if (
        mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        and math_fidelity != MathFidelity.LoFi
    ):
        pytest.skip("Math fidelity does not affect Elwadd/Elwsub")

    input_dimensions = [32, 32]

    # Stimuli generation
    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    # If unary op is Sqrt, ensure binary result will be non-negative to avoid NaNs/​Infs
    if unary_op == MathOperation.Sqrt:
        if mathop == MathOperation.Elwsub:
            # Make src_A >= src_B element-wise
            src_B = src_B.abs()
            src_A = src_B + src_A.abs()
        elif mathop == MathOperation.Elwadd:
            # Ensure positive by taking abs
            src_A = src_A.abs()
            src_B = src_B.abs()
        elif mathop == MathOperation.Elwmul:
            # Multiplication of any signed numbers could be negative; force operands positive
            src_A = src_A.abs()
            src_B = src_B.abs()

    # Golden pipeline: binary FPU → unary SFPU → untilize

    bin_golden_fn = get_golden_generator(EltwiseBinaryGolden)
    binary_out = bin_golden_fn(
        mathop, src_A, src_B, formats.output_format, math_fidelity
    )

    unary_golden_fn = get_golden_generator(UnarySFPUGolden)
    unary_out = unary_golden_fn(
        unary_op, binary_out, formats.output_format, dest_acc, formats.output_format
    )

    untilize_golden_fn = get_golden_generator(UntilizeGolden)
    golden_tensor = untilize_golden_fn(unary_out, formats.output_format, [32, 32])

    unpack_to_dest = formats.input_format.is_32_bit()

    # Load stimuli into device
    test_config = {
        "formats": formats,
        "testname": config["testname"],
        "dest_acc": dest_acc,
        "dst_sync": dst_sync,
        "mathop": mathop,
        "unary_op": unary_op,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "unpack_to_dest": unpack_to_dest,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    # Build & run on device
    run_test(test_config)

    # Fetch results
    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
