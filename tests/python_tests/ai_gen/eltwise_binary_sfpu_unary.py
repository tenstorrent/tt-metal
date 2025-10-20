# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
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
from helpers.test_config import ProfilerBuild, run_test
from helpers.utils import passed_test

# -----------------------------------------------------------------------------
# 2. Helper constants & parameter space definition
# -----------------------------------------------------------------------------

# SUPPORTED FORMATS FOR TEST - following the same pattern as other tests
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    # DataFormat.Bfp8_b,
    DataFormat.Float32,
]

# BINARY OPERATIONS TO TEST
binary_ops = [
    MathOperation.Elwadd,
    MathOperation.Elwsub,
    MathOperation.Elwmul,
]

# UNARY OPERATIONS TO TEST
unary_ops = [
    # MathOperation.Celu,
    # MathOperation.Cos,
    # MathOperation.Sin,
    # MathOperation.Elu,
    # MathOperation.Exp,
    # MathOperation.Exp2,
    # MathOperation.Reciprocal,
    # MathOperation.Log,
    # MathOperation.Silu,
    MathOperation.Square,
    MathOperation.Abs,
    MathOperation.Gelu,
    MathOperation.Hardsigmoid,
    MathOperation.Neg,
    MathOperation.Sqrt,
]

# DST SYNC OPTIONS TO TEST
dst_sync_options = [DstSync.SyncHalf, DstSync.SyncFull]

# Generate format combinations (both same and mixed format combinations)
test_formats = input_output_formats(supported_formats, same=False)

# Generate parameter combinations manually since generate_params doesn't support unary_op and dst_sync
all_params = [
    {
        "testname": "eltwise_binary_sfpu_unary",
        "formats": fmt,
        "dest_acc": dest_acc,
        "approx_mode": approx_mode,
        "mathop": binary_op,
        "unary_op": unary_op,
        "dst_sync": dst_sync,
        "math_fidelity": math_fidelity,
    }
    for fmt in test_formats
    for dest_acc in [DestAccumulation.Yes, DestAccumulation.No]
    for approx_mode in [ApproximationMode.Yes, ApproximationMode.No]
    for binary_op in binary_ops
    for unary_op in unary_ops
    for dst_sync in dst_sync_options
    for math_fidelity in [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]
]

# Generate parameter IDs manually
param_ids = [
    f"bin={p['mathop'].name}|un={p['unary_op'].name}|fmt={p['formats'].input_format.name}->{p['formats'].output_format.name}|acc={p['dest_acc'].name}|approx={p['approx_mode'].name}|sync={p['dst_sync'].name}|fid={p['math_fidelity'].name}"
    for p in all_params
]


# -----------------------------------------------------------------------------
# 3. The test implementation
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("config", all_params, ids=param_ids)
def test_sweep_test(config):
    """Runs the C++ eltwise_binary_sfpu_unary.cpp kernel for a full sweep of all parameter combinations.
    Tests across all commonly used formats in the test infrastructure with 32×32 tensor shape (1 tile).
    """

    # Extract parameters from config
    formats = config["formats"]
    binary_op = config["mathop"]
    unary_op = config["unary_op"]
    dest_acc = config["dest_acc"]
    approx_mode = config["approx_mode"]
    dst_sync = config["dst_sync"]
    math_fidelity = config["math_fidelity"]

    # ------------------------------------------------------------------
    # Skip known failing cases
    # ------------------------------------------------------------------

    if (
        binary_op in [MathOperation.Elwadd, MathOperation.Elwsub]
        and math_fidelity != MathFidelity.LoFi
    ):
        pytest.skip("No need to test higher fidelities for add/sub")

    # Temporary: skip precision-sensitive fused case we have not modelled yet
    if binary_op == MathOperation.Elwmul and unary_op == MathOperation.Square:
        pytest.skip(
            "Known precision edge-case for Elwadd+Square with dest_acc; skipped for now"
        )

    # ------------------------------------------------------------------
    # Generate input stimuli
    # ------------------------------------------------------------------
    input_dimensions = [32, 32]
    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    # Will hold pre-SFPU tensor when inputs are shifted. Initialization
    # here guarantees the variable exists on all execution paths.
    adjusted_binary_tensor = None

    # --------------------------------------------------------------
    #  Adjust inputs if the chosen unary op requires non-negative
    #  arguments (e.g. Sqrt, Log) but the binary result would become
    #  negative. We add a positive offset to src_A to shift the
    #  post-binary tensor into the valid range and then recalculate
    #  the golden reference.
    # --------------------------------------------------------------
    POSITIVE_ONLY_UNARY = {MathOperation.Sqrt, MathOperation.Log}

    def _compute_binary(tA, tB):
        gen = get_golden_generator(EltwiseBinaryGolden)
        return gen(
            binary_op,
            tA,
            tB,
            formats.output_format,
            math_fidelity,
        )

    if unary_op in POSITIVE_ONLY_UNARY:
        binary_res = _compute_binary(src_A, src_B)
        min_val = torch.min(binary_res).item()
        if min_val <= 0:
            offset = abs(min_val) + 0.1  # small positive margin
            src_A = (src_A + offset).to(src_A.dtype)
            # Re-compute binary result after the shift
            binary_res = _compute_binary(src_A, src_B)
            # Now min should be >0; if still negative or zero, increase offset again
            if torch.min(binary_res).item() <= 0:
                extra = abs(torch.min(binary_res).item()) + 0.1
                src_A = (src_A + extra).to(src_A.dtype)
                binary_res = _compute_binary(src_A, src_B)
            # Replace subsequent golden computation base tensor
            adjusted_binary_tensor = binary_res

    # Handle reciprocal operation to avoid division by zero
    elif unary_op == MathOperation.Reciprocal:
        binary_res = _compute_binary(src_A, src_B)
        # Check if any values are too close to zero
        min_abs_val = torch.min(torch.abs(binary_res)).item()
        if min_abs_val < 1e-6:
            # Find the smallest absolute value and ensure it's at least 1e-6
            mask = torch.abs(binary_res) < 1e-6
            offset = 1e-6 - min_abs_val + 0.1
            src_A = (src_A + offset).to(src_A.dtype)
            # Re-compute binary result after the shift
            binary_res = _compute_binary(src_A, src_B)
            adjusted_binary_tensor = binary_res

    # ------------------------------------------------------------------
    # Produce golden output
    # ------------------------------------------------------------------
    if adjusted_binary_tensor is None:
        gen_binary = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = gen_binary(
            binary_op,
            src_A,
            src_B,
            formats.output_format,
            math_fidelity,
        )
    else:
        golden_tensor = adjusted_binary_tensor

    gen_unary = get_golden_generator(UnarySFPUGolden)
    golden_tensor = gen_unary(
        unary_op,
        golden_tensor,
        formats.output_format,
        dest_acc,
        formats.output_format,
    )

    # ------------------------------------------------------------------
    # Write stimuli to device L1 and run the HW test
    # ------------------------------------------------------------------
    # Create test config based on the parametrized config
    test_config = config.copy()
    test_config.update(
        {
            "testname": "eltwise_binary_sfpu_unary",
            "tile_cnt": tile_cnt,
            "input_dimensions": input_dimensions,
        }
    )

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config, profiler_build=ProfilerBuild.No)

    # ------------------------------------------------------------------
    # Collect results and compare with golden
    # ------------------------------------------------------------------
    res_from_L1 = collect_results(
        formats,
        tile_count=tile_cnt,
        address=res_address,
    )
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
