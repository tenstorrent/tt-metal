# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TilizeGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
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

# Focus on commonly used formats in production ML workloads
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
]

# No binary operations needed for this simplified test

# SFPU unary operations to test after datacopy
unary_ops = [
    MathOperation.Square,
    MathOperation.Abs,
    MathOperation.Celu,
    MathOperation.Cos,
    MathOperation.Sin,
    MathOperation.Elu,
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Gelu,
    MathOperation.Hardsigmoid,
    MathOperation.Log,
    MathOperation.Neg,
    MathOperation.Reciprocal,
    MathOperation.Silu,
    MathOperation.Sqrt,
]

# Generate format combinations focusing on same input/output for simplicity
test_formats = input_output_formats(supported_formats, same=True)

# Parameter sweep covering the most important combinations
all_params = [
    {
        "testname": "unpack_tilize_sfpu_pack",
        "formats": fmt,
        "dest_acc": dest_acc,
        "approx_mode": approx_mode,
        "unary_op": un_op,
        "dst_sync": dst_sync,
        "math_fidelity": fidelity,
    }
    for fmt in test_formats
    for dest_acc in [DestAccumulation.Yes, DestAccumulation.No]
    for approx_mode in [ApproximationMode.No, ApproximationMode.Yes]
    for un_op in unary_ops
    for dst_sync in [DstSync.SyncHalf, DstSync.SyncFull]
    for fidelity in [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]
]

# Generate descriptive parameter IDs for test identification
param_ids = [
    f"un={p['unary_op'].name}|fmt={p['formats'].input_format.name}->{p['formats'].output_format.name}|acc={p['dest_acc'].name}|approx={p['approx_mode'].name}|sync={p['dst_sync'].name}|fid={p['math_fidelity'].name}"
    for p in all_params
]

# -----------------------------------------------------------------------------
# 2. Test implementation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", all_params, ids=param_ids)
def test_fused_tilize_sfpu_pack(config):
    """
    Tests the simplified pipeline: unpack_tilize → datacopy → sfpu_unary → pack

    This represents a basic ML pattern where:
    1. Row-major input tensors are tilized for efficient computation
    2. Data is copied from src to dest register
    3. Unary activations are applied (e.g., GELU, SiLU, Abs, etc.)
    4. Results are packed back to L1 in tile format

    Key validation points:
    - Tilization preserves data correctness
    - Datacopy operation works correctly
    - SFPU operations handle edge cases (NaN, Inf)
    - Dest accumulation behavior is correct
    - All format conversions maintain precision
    """

    # Extract test parameters
    formats = config["formats"]
    unary_op = config["unary_op"]
    dest_acc = config["dest_acc"]
    approx_mode = config["approx_mode"]
    dst_sync = config["dst_sync"]
    math_fidelity = config["math_fidelity"]

    # ---------------------------------------------------------------------
    # Skip conditions for known hardware limitations or unsupported combos
    # ---------------------------------------------------------------------

    # Skip approximation mode combinations that cause numerical instability
    # if (
    #     approx_mode == ApproximationMode.Yes
    #     and unary_op in [MathOperation.Gelu, MathOperation.Silu]
    #     and formats.input_format == DataFormat.Float16
    # ):
    #     pytest.skip("FP16 approximation mode unstable for complex activations")

    # ---------------------------------------------------------------------
    # Generate input stimuli
    # ---------------------------------------------------------------------

    # Use single tile: 32x32 input dimensions (standard tile size)
    input_dimensions = [32, 32]
    src_A, _, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )
    # We only use src_A for this simplified test

    # Handle domain restrictions for operations that require specific input ranges
    if unary_op == MathOperation.Sqrt:
        # Ensure input will be non-negative
        src_A = src_A.abs()
    elif unary_op == MathOperation.Log:
        # Ensure input is positive for log operation
        src_A = src_A.abs() + 1e-6
    elif unary_op == MathOperation.Reciprocal:
        # Avoid division by zero for reciprocal operation
        src_A = torch.where(src_A.abs() < 1e-6, torch.sign(src_A) * 1e-6, src_A)

    # ---------------------------------------------------------------------
    # Generate golden reference
    # ---------------------------------------------------------------------

    # Step 1: Tilize input (convert row-major to tiled format)
    tilize_golden = get_golden_generator(TilizeGolden)
    tilized_A = tilize_golden(src_A, input_dimensions, formats.input_format)

    # Step 2: Apply datacopy (src_A -> dest, which is just identity)
    datacopy_result = tilized_A

    # Step 3: Apply unary SFPU operation
    unary_golden = get_golden_generator(UnarySFPUGolden)
    unary_result = unary_golden(
        unary_op,
        datacopy_result,
        formats.output_format,
        dest_acc,
        formats.output_format,
    )

    # Final result remains in tile format (no untilize)
    golden_tensor = unary_result

    # ---------------------------------------------------------------------
    # Execute hardware test
    # ---------------------------------------------------------------------

    # Configure test parameters for kernel
    test_config = {
        "testname": config["testname"],
        "formats": formats,
        "dest_acc": dest_acc,
        "approx_mode": approx_mode,
        "unary_op": unary_op,  # SFPU operation
        "dst_sync": dst_sync,
        "math_fidelity": math_fidelity,
        "tile_cnt": tile_cnt,
        "input_dimensions": input_dimensions,
        "unpack_to_dest": formats.input_format.is_32_bit(),
    }

    # Write raw stimuli to L1 (will be tilized by unpack)
    # Only need src_A for this simplified test
    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_A,  # Dummy value for src_B parameter
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    # Build and execute kernel
    run_test(test_config)

    # ---------------------------------------------------------------------
    # Collect and validate results
    # ---------------------------------------------------------------------

    # Read results from L1
    res_from_L1 = collect_results(
        formats,
        tile_count=tile_cnt,
        address=res_address,
    )

    # Verify result length
    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result length mismatch: got {len(res_from_L1)}, expected {len(golden_tensor)}"

    # Convert to tensor for comparison
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Validate against golden reference
    assert passed_test(golden_tensor, res_tensor, formats.output_format), (
        f"Fused tilize->datacopy->sfpu({unary_op.name})->pack failed "
        f"for format {formats.input_format.name}->{formats.output_format.name}, "
        f"dest_acc={dest_acc.name}, fidelity={math_fidelity.name}"
    )
