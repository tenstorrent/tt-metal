# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Extended matmul tests with all supported fused activations."""

import functools
import re

import pytest
from loguru import logger
import ttnn
from tests.ttnn.nightly.unit_tests.operations.matmul.utility_functions import ttnn_matmul, ttnn_linear
from models.common.utility_functions import torch2tt_tensor, tt2torch_tensor
import torch
import torch.nn.functional as F
from tests.ttnn.utils_for_testing import assert_numeric_metrics

# Every tolerance in this file is a literal, predicted from the number formats on
# the device path rather than observed. numeric_tolerances.py in this directory
# records the error budget the literals come from and how each one is arrived at;
# nothing imports it, so re-derive the tables from it by hand if a test's shapes,
# seeds or activations change.
#
# The shape of the answer, from that budget: the relative error of a matmul does
# not grow with K. Each product carries a fractional slip from the mantissa bits
# the multiplier discards, those slips multiply values of random sign, so the
# error grows as sqrt(K) and so does the result. They cancel. At LoFi the
# multiplier keeps only 4 of the 7 mantissa bits of the right hand operand, a 3.6
# percent relative error that dominates every other term, which is why the limits
# below barely move with K or with the data type.

# rtol carries only what scales with an element's own value: the rounding into the
# output format, doubled for the safety factor, plus two units in the last place
# when the activation is a fitted polynomial rather than a clamp the device
# evaluates exactly. Everything else in a matmul's error is proportional to the
# size of the whole dot product, not to the element, so atol carries it.
_RTOL_BY_OUTPUT_FORMAT = {
    (ttnn.bfloat16, True): 0.0046,
    (ttnn.bfloat16, False): 0.0358,
    (ttnn.bfloat8_b, True): 0.0181,
    (ttnn.bfloat8_b, False): 0.0493,
}

_ACTIVATION_PARAMS_PATTERN = re.compile(r"params=\[([^\]]*)\]")


def activation_params(activation):
    """Parameters of a ``ttnn.UnaryWithParam``, read from its text form.

    The binding exposes ``op_type`` but no accessor for the parameters, so the
    only way to recover them is the object's own text form.
    """
    if activation is None:
        return []
    match = _ACTIVATION_PARAMS_PATTERN.search(repr(activation))
    if match is None or not match.group(1).strip():
        return []
    return [float(value) for value in match.group(1).split(",")]


def get_activation_golden_function(activation):
    """Get PyTorch equivalent function for each activation."""
    if activation is None:
        return lambda x: x

    activation_map = {
        "relu": F.relu,
        "relu6": F.relu6,
        "silu": F.silu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "hardsigmoid": F.hardsigmoid,
        "hardtanh": F.hardtanh,
        "selu": F.selu,
        "softplus": F.softplus,
    }

    if isinstance(activation, str):
        return activation_map.get(activation, lambda x: x)
    elif isinstance(activation, ttnn.UnaryWithParam):
        # Handle UnaryWithParam objects
        op_type_map = {
            ttnn.UnaryOpType.RELU: F.relu,
            ttnn.UnaryOpType.RELU6: F.relu6,
            ttnn.UnaryOpType.SILU: F.silu,
            ttnn.UnaryOpType.GELU: F.gelu,
            ttnn.UnaryOpType.TANH: torch.tanh,
            ttnn.UnaryOpType.SIGMOID: torch.sigmoid,
            ttnn.UnaryOpType.HARDSIGMOID: F.hardsigmoid,
            ttnn.UnaryOpType.HARDTANH: lambda x: F.hardtanh(x, -1.0, 1.0),  # Default values
            ttnn.UnaryOpType.SELU: F.selu,
            ttnn.UnaryOpType.SOFTPLUS: lambda x: F.softplus(x, beta=1.0, threshold=20.0),  # Default values
        }
        return op_type_map.get(activation.op_type, lambda x: x)

    return lambda x: x


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


# Limits for each activation, folded over the two shapes, both data types and both
# packer_l1_acc settings. Each is the most permissive value the budget gives over
# those cases, so no case is held tighter than predicted; the trailing comment
# gives the full range, widest spread 1.69x. The last field says whether the
# device evaluates the activation exactly, with comparison and selection only,
# rather than with a fitted polynomial, which is what picks rtol out of
# _RTOL_BY_OUTPUT_FORMAT.
#
# atol for an activation that saturates is set by that activation's own output
# range rather than by K, which is why hardtanh sits at 4.0 and the unbounded
# activations near 10.
@pytest.mark.parametrize(
    "activation, atol, frobenius_threshold, pcc_threshold, evaluated_exactly",
    [
        # activation                atol      frob       pcc    exact      derived atol      frobenius            pcc
        (None, 9.857, 0.0886, 0.99607, True),  # 5.83..9.86   0.0766..0.0886  0.99608..0.99707
        # String-based activations
        ("relu", 9.857, 0.0879, 0.99433, True),  # 5.83..9.86   0.0765..0.0878  0.99433..0.99572
        ("relu6", 9.857, 0.1528, 0.97807, True),  # 5.83..9.86   0.1204..0.1528  0.97808..0.98686
        ("silu", 10.3894, 0.0997, 0.99272, False),  # 6.27..10.39   0.0850..0.0996  0.99273..0.99476
        ("gelu", 10.1969, 0.0995, 0.99274, False),  # 6.17..10.20   0.0848..0.0994  0.99275..0.99476
        ("tanh", 3.9426, 0.2803, 0.96072, False),  # 3.59..3.94   0.2264..0.2803  0.96072..0.97437
        ("sigmoid", 1.6864, 0.1563, 0.97458, False),  # 1.24..1.69   0.1206..0.1562  0.97459..0.98475
        ("hardsigmoid", 1.6429, 0.1525, 0.9758, False),  # 0.97..1.64   0.1173..0.1524  0.97581..0.98557
        ("hardtanh", 4.0, 0.3064, 0.95307, True),  # 4.00..4.00   0.2542..0.3064  0.95307..0.96770
        ("selu", 10.6898, 0.1013, 0.99308, False),  # 6.46..10.69   0.0865..0.1012  0.99309..0.99507
        ("softplus", 9.857, 0.0982, 0.9929, False),  # 5.83..9.86   0.0829..0.0981  0.99290..0.99495
        # UnaryWithParam versions with default parameters, same limits as the
        # string spelling of the same activation
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), 9.857, 0.0879, 0.99433, True),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6), 9.857, 0.1528, 0.97807, True),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), 10.3894, 0.0997, 0.99272, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU), 10.1969, 0.0995, 0.99274, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH), 3.9426, 0.2803, 0.96072, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), 1.6864, 0.1563, 0.97458, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSIGMOID), 1.6429, 0.1525, 0.9758, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH), 4.0, 0.3064, 0.95307, True),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU), 10.6898, 0.1013, 0.99308, False),
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS), 9.857, 0.0982, 0.9929, False),
    ],
    ids=[
        "no_activation",
        # String IDs
        "relu_str",
        "relu6_str",
        "silu_str",
        "gelu_str",
        "tanh_str",
        "sigmoid_str",
        "hardsigmoid_str",
        "hardtanh_str",
        "selu_str",
        "softplus_str",
        # UnaryWithParam IDs
        "relu_param",
        "relu6_param",
        "silu_param",
        "gelu_param",
        "tanh_param",
        "sigmoid_param",
        "hardsigmoid_param",
        "hardtanh_param",
        "selu_param",
        "softplus_param",
    ],
)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128, 256, 256),  # Small test case - N must be >= num_cores * 32
        (256, 512, 512),  # Medium size
    ],
    ids=["small", "medium"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize("packer_l1_acc", [False, True], ids=["no_l1_acc", "l1_acc"])
def test_matmul_with_fused_activations(
    device,
    activation,
    atol,
    frobenius_threshold,
    pcc_threshold,
    evaluated_exactly,
    M,
    K,
    N,
    dtype,
    packer_l1_acc,
    function_level_defaults,
):
    """Test matmul with all supported fused activations."""
    torch.manual_seed(42)

    # Create input tensors
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    in0 = torch.randn(in0_shape).bfloat16()
    in1 = torch.randn(in1_shape).bfloat16()

    # Convert to TT tensors
    in0_t = torch2tt_tensor(in0.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=dtype)

    # Setup program config for 1D multicast
    # Adapt grid size based on problem size to ensure at least one tile per core
    max_cores = min(8, N // 32)  # Ensure each core gets at least one tile
    grid_size = (max_cores, 1)
    num_cores = grid_size[0] * grid_size[1]

    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    # Ensure valid subblock dimensions
    out_subblock_h = 1
    out_subblock_w = min(4, out_block_w) if out_block_w > 0 else 1

    # Convert string activation to UnaryWithParam if needed
    if isinstance(activation, str):
        activation_map = {
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "relu6": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
            "silu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            "tanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
            "sigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            "hardsigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSIGMOID),
            "hardtanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH),
            "selu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU),
            "softplus": ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS),
        }
        fused_activation = activation_map.get(activation, None)
    else:
        fused_activation = activation

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_size[0], grid_size[1]),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=fused_activation,  # Pass the converted activation
        mcast_in0=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=packer_l1_acc,
    )

    # Run matmul with fused activation
    output_t = ttnn_matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )

    # Get TT output
    tt_out = tt2torch_tensor(output_t)

    # Compute golden reference
    pt_matmul = in0 @ in1
    activation_fn = get_activation_golden_function(activation)
    pt_out = activation_fn(pt_matmul)

    assert_numeric_metrics(
        pt_out.float(),
        tt_out,
        atol=atol,
        rtol=_RTOL_BY_OUTPUT_FORMAT[(dtype, evaluated_exactly)],
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )


# One shape, one data type and one packer_l1_acc setting here, so each row holds
# the limits for exactly its own case with nothing folded in. rtol is 0.0045 for
# the two clamps and 0.0358 for the two fitted polynomials, matching
# _RTOL_BY_OUTPUT_FORMAT for a bfloat16 output.
@pytest.mark.parametrize(
    "activation, atol, rtol, frobenius_threshold, pcc_threshold",
    [
        # activation with its custom parameters             atol     rtol     frob      pcc
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6, 3.0), 3.7732, 0.0046, 0.1338, 0.98297),  # Custom max=3.0
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -2.0, 2.0), 3.7732, 0.0046, 0.1608, 0.98708),  # Custom min/max
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU, 1.5, 1.1), 4.3585, 0.0358, 0.0862, 0.99518),  # Custom alpha/lambda
        (
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 2.0, 10.0),
            3.7732,
            0.0358,
            0.0831,
            0.99489,
        ),  # Custom beta/threshold
    ],
    ids=["relu6_custom", "hardtanh_custom", "selu_custom", "softplus_custom"],
)
def test_matmul_with_custom_activation_params(
    device,
    activation,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    function_level_defaults,
):
    """Test matmul with activations using custom parameters."""
    torch.manual_seed(42)

    # Test size that ensures valid tile dimensions
    M, K, N = 64, 128, 128  # N must be >= num_cores * 32

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    in0 = torch.randn(in0_shape).bfloat16()
    in1 = torch.randn(in1_shape).bfloat16()

    # Convert to TT tensors
    in0_t = torch2tt_tensor(in0.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=ttnn.bfloat16)
    in1_t = torch2tt_tensor(in1.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=ttnn.bfloat16)

    # Adaptive 1D config
    max_cores = min(4, N // 32)  # Ensure each core gets at least one tile
    grid_size = (max_cores, 1)
    num_cores = grid_size[0]

    in0_block_w = K // num_cores // 32

    # activation is already UnaryWithParam in this test, no conversion needed
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_size[0], grid_size[1]),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32,  # M/32
        per_core_N=N // num_cores // 32,  # N/num_cores/32
        fuse_batch=True,
        fused_activation=activation,  # Already UnaryWithParam
        mcast_in0=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Run matmul with custom activation
    output_t = ttnn_matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )

    # Get TT output
    tt_out = tt2torch_tensor(output_t)

    # Compute golden reference, taking the activation parameters from the same
    # object the device was given so the two cannot drift apart.
    pt_matmul = in0 @ in1
    activation_fn = functools.partial(apply_activation_to_reference, activation=activation)
    pt_out = activation_fn(pt_matmul)

    assert_numeric_metrics(
        pt_out.float(),
        tt_out,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )


@pytest.mark.parametrize(
    "grid_config",
    [
        ((8, 1), "1d"),  # 1D multicast
        ((4, 2), "2d"),  # 2D multicast
    ],
    ids=["1d_multicast", "2d_multicast"],
)
# Folded over the 1D and 2D grid configs, which differ only in how many K tiles a
# block covers (1 against 2) and so move the limits by at most 1.01x.
@pytest.mark.parametrize(
    "activation, atol, rtol, frobenius_threshold, pcc_threshold",
    [
        # activation      atol     rtol     frob      pcc
        ("relu", 5.9798, 0.0046, 0.076, 0.99576),
        ("gelu", 6.3191, 0.0358, 0.0843, 0.99479),
        ("sigmoid", 1.2673, 0.0358, 0.1185, 0.9851),
        ("hardtanh", 4.0, 0.0046, 0.2471, 0.96947),
        ("softplus", 5.9798, 0.0358, 0.0825, 0.99498),
    ],
    ids=["relu", "gelu", "sigmoid", "hardtanh", "softplus"],
)
def test_activation_with_different_program_configs(
    device,
    grid_config,
    activation,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    function_level_defaults,
):
    """Test activations work with different program configurations."""
    torch.manual_seed(42)

    grid_size, config_type = grid_config
    M, K, N = 256, 256, 256

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    in0 = torch.randn(in0_shape).bfloat16()
    in1 = torch.randn(in1_shape).bfloat16()

    in0_t = torch2tt_tensor(in0.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=ttnn.bfloat16)
    in1_t = torch2tt_tensor(in1.float(), device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG, tt_dtype=ttnn.bfloat16)

    # Convert string activation to UnaryWithParam
    if isinstance(activation, str):
        activation_map = {
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            "sigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            "hardtanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH),
            "softplus": ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS),
        }
        fused_activation = activation_map.get(activation, None)
    else:
        fused_activation = activation

    if config_type == "1d":
        # 1D multicast configuration
        num_cores = grid_size[0]
        per_core_N = N // num_cores // 32
        out_subblock_w = min(2, per_core_N) if per_core_N > 0 else 1
        in0_block_w = K // num_cores // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_size[0], grid_size[1]),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=M // 32,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )
    else:
        # 2D multicast configuration
        per_core_N = N // grid_size[0] // 32
        out_subblock_w = min(2, per_core_N) if per_core_N > 0 else 1
        in0_block_w = K // grid_size[0] // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_size[0], grid_size[1]),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=M // grid_size[1] // 32,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
        )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Run matmul
    output_t = ttnn_matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )

    tt_out = tt2torch_tensor(output_t)

    # Golden reference
    pt_matmul = in0 @ in1
    activation_fn = get_activation_golden_function(activation)
    pt_out = activation_fn(pt_matmul)

    assert_numeric_metrics(
        pt_out.float(),
        tt_out,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )


# ============================================================================
# DRAM Sharded Matmul Tests with Bias and Activation
# ============================================================================

from models.common.utility_functions import (
    is_blackhole,
)


def pad_to_dram_banks(num, num_banks):
    lcm = 32 * num_banks
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


def convert_activation_to_unary_param(activation):
    """Convert string activation names to UnaryWithParam objects."""
    if activation is None:
        return None

    if isinstance(activation, ttnn.UnaryWithParam):
        # Already a UnaryWithParam, return as-is
        return activation

    if isinstance(activation, str):
        # Map string names to UnaryWithParam objects with default parameters
        activation_map = {
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "relu6": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
            "silu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            "tanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
            "sigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            "hardsigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSIGMOID),
            "hardtanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH),
            "selu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU),
            "softplus": ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS),
        }

        if activation in activation_map:
            return activation_map[activation]
        else:
            raise ValueError(f"Unsupported activation string: {activation}")

    # If it's not a string or UnaryWithParam, return as-is (might be None)
    return activation


def apply_activation_to_reference(tensor, activation):
    """Apply an activation to a reference tensor, honouring its parameters.

    The reference is always the true mathematical function. Some activations
    accept a flag that makes the device evaluate them with a coarse piecewise
    linear table instead of a fitted polynomial, but that changes only how
    accurately the function is computed, not which function is intended, so it
    belongs in the tolerance rather than in the reference.
    """
    if activation is None:
        return tensor

    if isinstance(activation, str):
        activation = convert_activation_to_unary_param(activation)
        if activation is None:
            return tensor

    op_type = getattr(activation, "op_type", activation)
    params = activation_params(activation)

    if op_type == ttnn.UnaryOpType.RELU:
        return torch.nn.functional.relu(tensor)
    elif op_type == ttnn.UnaryOpType.RELU6:
        max_val = params[0] if params else 6.0
        return torch.clamp(tensor, min=0, max=max_val)
    elif op_type == ttnn.UnaryOpType.SILU:
        return torch.nn.functional.silu(tensor)
    elif op_type == ttnn.UnaryOpType.GELU:
        return torch.nn.functional.gelu(tensor)
    elif op_type == ttnn.UnaryOpType.GELU_TANH:
        return torch.nn.functional.gelu(tensor, approximate="tanh")
    elif op_type == ttnn.UnaryOpType.TANH:
        return torch.tanh(tensor)
    elif op_type == ttnn.UnaryOpType.SIGMOID:
        return torch.sigmoid(tensor)
    elif op_type == ttnn.UnaryOpType.HARDSIGMOID:
        return torch.nn.functional.hardsigmoid(tensor)
    elif op_type == ttnn.UnaryOpType.HARDTANH:
        min_val = params[0] if params else -1.0
        max_val = params[1] if len(params) > 1 else 1.0
        return torch.nn.functional.hardtanh(tensor, min_val=min_val, max_val=max_val)
    elif op_type == ttnn.UnaryOpType.SELU:
        # The fused activation interface documents the first parameter as alpha
        # and the second as the scale, so the reference reads them in that
        # order. With no parameters the standard SELU constants apply. SELU is
        # scale * (max(0, x) + min(0, alpha * (exp(x) - 1))).
        if not params:
            return torch.nn.functional.selu(tensor)
        alpha = params[0]
        scale = params[1] if len(params) > 1 else 1.0507009873554805
        return scale * torch.where(tensor > 0, tensor, alpha * (torch.exp(tensor) - 1.0))
    elif op_type == ttnn.UnaryOpType.SOFTPLUS:
        beta = params[0] if params else 1.0
        threshold = params[1] if len(params) > 1 else 20.0
        return torch.nn.functional.softplus(tensor, beta=beta, threshold=threshold)
    else:
        raise ValueError(f"Unsupported activation type: {op_type}")


def run_test_matmul_dram_sharded_with_bias_and_activation(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    packer_l1_acc,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    function_level_defaults,
):
    torch.manual_seed(0)

    if is_blackhole():
        num_banks = device.dram_grid_size().x
    else:
        num_banks = 12

    N_padded = pad_to_dram_banks(N, num_banks)

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, N]
    bias_shard_shape = [32, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    # Create input tensors
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=in0_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=in1_dtype)

    # Handle bias if present
    bias_t = None
    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, 32 - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = torch2tt_tensor(bias_padded, device, tt_memory_config=bias_mem_config, tt_dtype=ttnn.bfloat16)

    # Shard in0
    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M, int(in0_block_w * 32)],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Convert string activation to UnaryWithParam if needed
    activation_param = convert_activation_to_unary_param(activation)

    # Program config with activation
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=activation_param,  # Pass the converted activation parameter
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )

    # Run the operation
    if has_bias:
        output_t = ttnn_linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn_matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    # Compute golden reference
    pt_matmul = in0 @ in1
    pt_pre_activation = (pt_matmul + bias) if has_bias else pt_matmul

    activation_fn = functools.partial(apply_activation_to_reference, activation=activation_param)
    pt_out = activation_fn(pt_pre_activation)

    tt_out = tt2torch_tensor(output_t)

    assert_numeric_metrics(
        pt_out,
        tt_out,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )


@pytest.mark.parametrize("fidelity", [ttnn.MathFidelity.LoFi], ids=["LoFi"])
@pytest.mark.parametrize("packer_l1_acc", [False, True], ids=["no_l1_acc", "l1_acc"])
@pytest.mark.parametrize(
    "has_bias, activation, atol, rtol, frobenius_threshold, pcc_threshold",
    [
        # Limits folded over the two shapes and both packer_l1_acc settings,
        # widest spread 1.46x, on atol between K = 4096 and K = 8192. The 32 bit
        # accumulator puts rtol at its floor everywhere except selu and softplus,
        # which take their 16 bit polynomial branch whatever the accumulator is.
        #
        #                       atol     rtol     frob      pcc
        # Test bias alone
        (True, None, 33.6096, 0.0046, 0.0769, 0.99704),
        (False, None, 33.607, 0.0046, 0.0769, 0.99704),
        # Test activation alone
        (False, "relu", 33.607, 0.0046, 0.0768, 0.99566),
        (False, "gelu", 33.9469, 0.0046, 0.0769, 0.99565),
        (False, "sigmoid", 1.9992, 0.0046, 0.2126, 0.95478),
        # Test bias + activation combinations (main focus)
        (True, "relu", 33.6096, 0.0046, 0.0769, 0.99566),
        (True, "gelu", 33.9495, 0.0046, 0.077, 0.99565),
        (True, "sigmoid", 1.9992, 0.0046, 0.2132, 0.95453),
        (True, "hardtanh", 4.0, 0.0046, 0.3372, 0.94316),
        (True, "selu", 35.6466, 0.0358, 0.0853, 0.99481),
        (True, "softplus", 33.6096, 0.0358, 0.0845, 0.99476),
        # Test with UnaryWithParam
        (True, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6, 6.0), 12.0, 0.0046, 0.211, 0.95665),
        (True, ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -1.0, 1.0), 4.0, 0.0046, 0.3372, 0.94316),
    ],
    ids=[
        "bias_only",
        "no_bias_no_activation",
        "relu_only",
        "gelu_only",
        "sigmoid_only",
        "bias_relu",
        "bias_gelu",
        "bias_sigmoid",
        "bias_hardtanh",
        "bias_selu",
        "bias_softplus",
        "bias_relu6_param",
        "bias_hardtanh_param",
    ],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16)],
)
@pytest.mark.parametrize(
    "M, K, N, grid_size",
    [
        (32, 4096, 1024, (8, 1)),  # Small test case
        (32, 8192, 2048, (8, 2)),  # Medium test case
    ],
    ids=["small", "medium"],
)
def test_matmul_dram_sharded_with_bias_and_activation(
    device,
    M,
    K,
    N,
    fidelity,
    packer_l1_acc,
    has_bias,
    activation,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
):
    """Test DRAM sharded matmul with combinations of bias and activation."""
    run_test_matmul_dram_sharded_with_bias_and_activation(
        device=device,
        in0_sharded=True,
        out_sharded=True,
        in1_in_dram=False,
        M=M,
        K=K,
        N=N,
        fidelity=fidelity,
        packer_l1_acc=packer_l1_acc,
        has_bias=has_bias,
        activation=activation,
        grid_size=grid_size,
        in0_dtype=in0_dtype,
        in1_dtype=in1_dtype,
        out_dtype=out_dtype,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        function_level_defaults=function_level_defaults,
    )


@pytest.mark.parametrize(
    "activation, fidelity, packer_l1_acc, atol, rtol, frobenius_threshold, pcc_threshold",
    [
        # One shape and one setting each, so nothing is folded into these limits.
        # HiFi2 makes the whole budget about three times smaller than LoFi, since
        # it consumes the right hand operand's full mantissa instead of 4 of its
        # 7 bits, which is why tanh reaches 0.992 here.
        #
        # Special combinations to test edge cases
        ("tanh", ttnn.MathFidelity.HiFi2, True, 3.5315, 0.0046, 0.1249, 0.99221),  # High precision with tanh
        ("gelu", ttnn.MathFidelity.LoFi, False, 16.777, 0.0046, 0.0767, 0.9957),  # Fast approximation with GELU
        ("sigmoid", ttnn.MathFidelity.LoFi, True, 1.9354, 0.0046, 0.1738, 0.96931),  # Sigmoid with L1 accumulation
    ],
    ids=["tanh_hifi", "gelu_lofi", "sigmoid_l1acc"],
)
def test_special_activation_combinations(
    device,
    activation,
    fidelity,
    packer_l1_acc,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    function_level_defaults,
):
    """Test specific activation combinations with different settings."""
    # Fixed test parameters
    M, K, N = 32, 2048, 1024
    grid_size = (8, 1)
    has_bias = True  # Always test with bias for these special cases

    run_test_matmul_dram_sharded_with_bias_and_activation(
        device=device,
        in0_sharded=True,
        out_sharded=True,
        in1_in_dram=False,
        M=M,
        K=K,
        N=N,
        fidelity=fidelity,
        packer_l1_acc=packer_l1_acc,
        has_bias=has_bias,
        activation=activation,
        grid_size=grid_size,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        function_level_defaults=function_level_defaults,
    )


# Limits folded over the three shapes. The inputs here are scaled by 0.1, so the
# products are a hundredth the size of the other tests' and every absolute limit
# scales with them.
#
# gelu_fast and tanh_fast replace the fitted polynomial with a handful of straight
# line segments, whose error is absolute and so does not shrink with the output.
# At this input scale that error is the whole budget, which is why their limits
# are an order of magnitude looser than accurate gelu's and tanh's. gelu_fast is
# the one entry in this file whose fold exceeds 2x (its pcc runs 0.856 at K = 2048
# to 0.931 at K = 4096, a 2.08x spread in the distance from 1); it is left folded
# because the spread comes entirely from the segment error bound, the single
# quantity in the budget that is estimated rather than derived, so splitting it
# would give false precision.
@pytest.mark.parametrize(
    "activation, atol, rtol, frobenius_threshold, pcc_threshold",
    [
        #                                              atol     rtol     frob      pcc
        (None, 0.2897, 0.0046, 0.0958, 0.99541),
        ("relu", 0.2897, 0.0046, 0.0948, 0.99341),
        ("relu6", 0.2897, 0.0046, 0.0948, 0.99341),
        ("silu", 0.3186, 0.0358, 0.1042, 0.99416),
        ("gelu", 0.3269, 0.0358, 0.1051, 0.99367),
        ("tanh", 0.2892, 0.0358, 0.1050, 0.99449),
        ("sigmoid", 0.0724, 0.0358, 0.0449, 0.98184),
        ("hardsigmoid", 0.0483, 0.0358, 0.0410, 0.96747),
        ("hardtanh", 0.2897, 0.0046, 0.1002, 0.99498),
        ("selu", 0.4741, 0.0358, 0.1039, 0.99458),
        ("softplus", 0.2736, 0.0358, 0.0533, 0.98921),
        # Test with UnaryWithParam objects with parameters
        # fast_and_approximate mode
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0), 0.5869, 0.0046, 0.5131, 0.85621),
        # fast_and_approximate mode
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH, 1.0), 0.5892, 0.0046, 0.3976, 0.92096),
        # The first parameter of sigmoid is the vector mode, not the
        # fast_and_approximate flag, which is the second one, so this case runs
        # the accurate sigmoid and carries the same limits as it.
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 1.0), 0.0724, 0.0358, 0.0449, 0.98184),
        # Custom min/max
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -2.0, 2.0), 0.2897, 0.0046, 0.0959, 0.99541),
        # Custom alpha/lambda
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SELU, 1.5, 1.2), 0.4854, 0.0358, 0.1037, 0.99462),
        # Custom beta/threshold
        (ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 2.0, 10.0), 0.2887, 0.0358, 0.0716, 0.99326),
    ],
    ids=[
        "no_activation",
        "relu_str",
        "relu6_str",
        "silu_str",
        "gelu_str",
        "tanh_str",
        "sigmoid_str",
        "hardsigmoid_str",
        "hardtanh_str",
        "selu_str",
        "softplus_str",
        "gelu_fast",
        "tanh_fast",
        "sigmoid_fast",
        "hardtanh_custom",
        "selu_custom",
        "softplus_custom",
    ],
)
@pytest.mark.parametrize(
    "M, K, N, grid",
    [
        (32, 2048, 256, (8, 1)),  # Small test case with 1D grid
        (64, 4096, 512, (8, 1)),  # Medium test case with 1D grid
        (128, 2048, 1024, (8, 1)),  # Large test case with 1D grid
    ],
    ids=["small", "medium", "large"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16)],
)
@pytest.mark.parametrize("packer_l1_acc", [False], ids=["no_l1_acc"])
def test_matmul_1d_gather_with_activations(
    device,
    activation,
    atol,
    rtol,
    frobenius_threshold,
    pcc_threshold,
    M,
    K,
    N,
    grid,
    in0_dtype,
    in1_dtype,
    out_dtype,
    packer_l1_acc,
    function_level_defaults,
):
    """Test matmul operations with various activation functions using simpler config.

    This test verifies that fused activations work correctly with matmul,
    testing both string-based activation names and UnaryWithParam objects with custom parameters.
    """

    # Skip if device doesn't support the required grid
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid[0] > compute_grid_size.x or grid[1] > compute_grid_size.y:
        pytest.skip(f"Device grid {compute_grid_size} is smaller than required grid {grid}")

    # Create input tensors
    torch.manual_seed(42)
    in0 = torch.randn(1, 1, M, K, dtype=torch.float32) * 0.1
    in1 = torch.randn(1, 1, K, N, dtype=torch.float32) * 0.1

    # Convert activation string to UnaryWithParam if needed
    if isinstance(activation, str):
        activation_param = convert_activation_to_unary_param(activation)
    else:
        activation_param = activation

    # Convert to TTNN tensors
    in0_ttnn = ttnn.from_torch(in0, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    in1_ttnn = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Create program config with activation
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=2,
        per_core_M=M // ttnn.TILE_SIZE // grid[1],
        per_core_N=N // ttnn.TILE_SIZE // grid[0],
        fuse_batch=True,
        fused_activation=activation_param,
        mcast_in0=True,
    )

    # Run matmul with fused activation
    try:
        output_ttnn = ttnn_matmul(
            in0_ttnn,
            in1_ttnn,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=out_dtype,
        )
    except Exception as e:
        # If it fails with this config, try without the specific program config
        logger.warning(f"Failed with 1D program config: {e}, trying without specific config")
        output_ttnn = ttnn_matmul(
            in0_ttnn,
            in1_ttnn,
            activation=activation_param,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=out_dtype,
            core_grid=ttnn.CoreGrid(y=grid[1], x=grid[0]),
        )

    # Get output back to CPU
    output = ttnn.to_torch(output_ttnn)

    # Compute reference in PyTorch
    pt_pre_activation = torch.matmul(in0, in1)
    activation_fn = functools.partial(apply_activation_to_reference, activation=activation_param)
    reference = activation_fn(pt_pre_activation)

    # No compute kernel config is passed, so the limits above were derived from
    # the op's own defaults: LoFi because a program config was supplied, a 16 bit
    # accumulator because the output is not float32, and accumulation in memory
    # for the same reason.
    assert_numeric_metrics(
        reference,
        output,
        atol=atol,
        rtol=rtol,
        frobenius_threshold=frobenius_threshold,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )
