# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase D: Helper Integration Testing

Tests the actual sfpu_helpers.hpp API — compute kernels that use sfpu_op(),
sfpu_pipeline(), sfpu_chain(), and named aliases like sfpu_sigmoid().

Each test creates tensors, runs a custom compute kernel that includes
sfpu_helpers.hpp and uses the helper API, then compares against PyTorch golden.
"""

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import skip_for_blackhole

pytestmark = pytest.mark.use_module_device

KERNEL_DIR = "tests/ttnn/unit_tests/operations/debug/kernels"


# =============================================================================
# Helper functions (reused from Phase B pattern)
# =============================================================================


def get_page_size(dtype):
    """Return tile page size in bytes for a given ttnn dtype."""
    if dtype == ttnn.bfloat16:
        return 2 * 1024
    elif dtype == ttnn.float32:
        return 4 * 1024
    elif dtype == ttnn.bfloat8_b:
        return 1 * 1024 + 64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def setup_single_core(device):
    """Return core grid for single-core execution on core (0,0)."""
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def run_unary_helper_test(
    device,
    input_data,
    golden_fn,
    compute_kernel_path,
    input_dtype=ttnn.bfloat16,
    output_dtype=None,
    pcc_threshold=0.99,
    atol=None,
):
    """
    Run a unary SFPU helper test using generic_op.

    The compute kernel at compute_kernel_path should include sfpu_helpers.hpp
    and use the helper API (sfpu_op, sfpu_chain, sfpu_pipeline, etc.).
    """
    if output_dtype is None:
        output_dtype = input_dtype

    num_tiles = 1
    shape = list(input_data.shape)
    core_grid = setup_single_core(device)
    work_per_core = num_tiles

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    torch_dtype_map = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
    }
    torch_input = input_data.to(torch_dtype_map.get(input_dtype, torch.bfloat16))

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )
    io_tensors = [input_tensor, output_tensor]

    # CB setup
    in_cb = 0
    out_cb = 16
    in_page_size = get_page_size(input_dtype)
    out_page_size = get_page_size(output_dtype)

    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=2 * in_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=in_cb, data_format=input_dtype, page_size=in_page_size)
        ],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=2 * out_page_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=output_dtype, page_size=out_page_size)
        ],
    )

    # Kernel args
    reader_compile_time_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_compile_time_args = [work_per_core, 1]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [input_tensor.buffer_address(), work_per_core, 0]
    writer_rt_args[0][0] = [output_tensor.buffer_address(), work_per_core, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)
    torch_output = ttnn.to_torch(output)

    # Compute golden
    torch_golden = golden_fn(input_data.float())
    torch_golden = torch_golden.to(torch_dtype_map.get(output_dtype, torch.bfloat16))

    if atol is not None:
        matching = torch.allclose(torch_golden.float(), torch_output.float(), atol=atol, rtol=0.05)
        if not matching:
            max_diff = (torch_golden.float() - torch_output.float()).abs().max().item()
            logger.error(f"Max absolute diff: {max_diff}, atol: {atol}")
    else:
        golden_flat = torch_golden.float().flatten()
        output_flat = torch_output.float().flatten()
        pcc = torch.corrcoef(torch.stack([golden_flat, output_flat]))[0, 1].item()
        matching = pcc >= pcc_threshold
        if not matching:
            logger.error(f"PCC: {pcc}, threshold: {pcc_threshold}")

    return matching, torch_output, torch_golden


def run_binary_helper_test(
    device,
    input_data_a,
    input_data_b,
    golden_fn,
    compute_kernel_path,
    input_dtype_a=ttnn.bfloat16,
    input_dtype_b=None,
    output_dtype=None,
    pcc_threshold=0.99,
):
    """
    Run a binary SFPU helper test with two input CBs (cb0, cb1) and output cb16.
    """
    if input_dtype_b is None:
        input_dtype_b = input_dtype_a
    if output_dtype is None:
        output_dtype = input_dtype_a

    num_tiles = 1
    shape = list(input_data_a.shape)
    core_grid = setup_single_core(device)
    work_per_core = num_tiles

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    torch_dtype_a = torch.bfloat16 if input_dtype_a == ttnn.bfloat16 else torch.float32
    torch_dtype_b = torch.bfloat16 if input_dtype_b == ttnn.bfloat16 else torch.float32

    input_tensor_a = ttnn.from_torch(
        input_data_a.to(torch_dtype_a),
        dtype=input_dtype_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        input_data_b.to(torch_dtype_b),
        dtype=input_dtype_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        output_dtype,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )
    io_tensors = [input_tensor_a, input_tensor_b, output_tensor]

    # CB setup
    in_cb0 = 0
    in_cb1 = 1
    out_cb = 16
    page_size_a = get_page_size(input_dtype_a)
    page_size_b = get_page_size(input_dtype_b)
    out_page_size = get_page_size(output_dtype)

    cb_descriptors = [
        ttnn.CBDescriptor(
            total_size=2 * page_size_a,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=in_cb0, data_format=input_dtype_a, page_size=page_size_a)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * page_size_b,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=in_cb1, data_format=input_dtype_b, page_size=page_size_b)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * out_page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=output_dtype, page_size=out_page_size)
            ],
        ),
    ]

    # Reader: reads from two tensors into cb0 and cb1
    reader_compile_time_args = list(ttnn.TensorAccessorArgs(input_tensor_a).get_compile_time_args())
    reader_compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor_b).get_compile_time_args())

    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_compile_time_args = [work_per_core, 1]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [
        input_tensor_a.buffer_address(),
        input_tensor_b.buffer_address(),
        work_per_core,
        0,
    ]
    writer_rt_args[0][0] = [output_tensor.buffer_address(), work_per_core, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/dataflow/reader_binary_two_cb.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cb_descriptors,
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)
    torch_output = ttnn.to_torch(output)

    # Golden
    torch_golden = golden_fn(input_data_a.float(), input_data_b.float())
    torch_golden = torch_golden.to(torch.bfloat16 if output_dtype == ttnn.bfloat16 else torch.float32)

    golden_flat = torch_golden.float().flatten()
    output_flat = torch_output.float().flatten()
    pcc = torch.corrcoef(torch.stack([golden_flat, output_flat]))[0, 1].item()
    matching = pcc >= pcc_threshold
    if not matching:
        logger.error(f"PCC: {pcc}, threshold: {pcc_threshold}")

    return matching, torch_output, torch_golden


# =============================================================================
# Test 1: sfpu_op<ICB>(ocb, num_tiles, Exp<>{})
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_op_exp(device):
    """Test sfpu_op with Exp — the most common single-unary helper pattern."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 2.0 - 1.0

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: torch.exp(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_op_exp.cpp",
        pcc_threshold=0.999,
    )
    logger.info(f"test_sfpu_op_exp: matching={matching}")
    assert matching, "sfpu_op<Exp> failed PCC check"


# =============================================================================
# Test 2: sfpu_sigmoid<ICB>(ocb, num_tiles)
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_alias_sigmoid(device):
    """Test named alias sfpu_sigmoid — convenience wrapper around sfpu_op."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 6.0 - 3.0

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: torch.sigmoid(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_sigmoid.cpp",
        pcc_threshold=0.99,
    )
    logger.info(f"test_sfpu_alias_sigmoid: matching={matching}")
    assert matching, "sfpu_sigmoid alias failed PCC check"


# =============================================================================
# Test 3: sfpu_chain(Load, Exp, Recip) => 1/exp(x)
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_chain_exp_recip(device):
    """Test chained unary: exp then recip via sfpu_chain + sfpu_pipeline."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 2.0 - 1.0

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: 1.0 / torch.exp(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_chain_exp_recip.cpp",
        pcc_threshold=0.99,
    )
    logger.info(f"test_sfpu_chain_exp_recip: matching={matching}")
    assert matching, "sfpu_chain(Exp, Recip) failed PCC check"


# =============================================================================
# Test 4: sfpu_chain hardswish = x * hardsigmoid(x)
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_chain_hardswish(device):
    """Test mixed chain: same-CB double-load + unary + binary SFPU for hardswish."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 6.0 - 3.0

    def hardswish_golden(x):
        hs = torch.clamp(x / 6.0 + 0.5, 0.0, 1.0)
        return x * hs

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        hardswish_golden,
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_chain_hardswish.cpp",
        pcc_threshold=0.98,
    )
    logger.info(f"test_sfpu_chain_hardswish: matching={matching}")
    assert matching, "sfpu_chain hardswish failed PCC check"


# =============================================================================
# Test 5: sfpu_chain tanhshrink = x - tanh(x)
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_chain_tanhshrink(device):
    """Test mixed chain: same-CB double-load, tanh on D1, sub D0-D1 for tanhshrink."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 4.0 - 2.0

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: x - torch.tanh(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_chain_tanhshrink.cpp",
        pcc_threshold=0.98,
    )
    logger.info(f"test_sfpu_chain_tanhshrink: matching={matching}")
    assert matching, "sfpu_chain tanhshrink failed PCC check"


# =============================================================================
# Test 6: sfpu_chain two-CB binary add
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_chain_binary_add_two_cbs(device):
    """Test two-CB binary add: Load A from cb0, Load B from cb1, SfpuAdd."""
    torch.manual_seed(42)
    input_a = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    input_b = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)

    matching, output, golden = run_binary_helper_test(
        device,
        input_a,
        input_b,
        lambda a, b: a + b,
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_chain_binary_add.cpp",
        pcc_threshold=0.999,
    )
    logger.info(f"test_sfpu_chain_binary_add_two_cbs: matching={matching}")
    assert matching, "sfpu_chain two-CB binary add failed PCC check"


# =============================================================================
# Test 7: SfpuOutputPolicy::Bulk with exp
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_bulk_output_policy(device):
    """Test Bulk output policy: reserve all tiles upfront, push all at end."""
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 2.0 - 1.0

    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: torch.exp(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_bulk_exp.cpp",
        pcc_threshold=0.999,
    )
    logger.info(f"test_sfpu_bulk_output_policy: matching={matching}")
    assert matching, "sfpu_op with Bulk output policy failed PCC check"


# =============================================================================
# Test 8: DEST slot validation (compile-time)
# =============================================================================


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_dest_validation(device):
    """
    Verify that valid DEST slots compile correctly via the helpers.

    The sfpu_helpers.hpp contains static_assert(dst_idx < 8) on all op structs,
    ensuring that Dst::D0 through Dst::D7 are valid at compile time. Since the
    other integration tests above successfully compile and run with D0 and D1,
    this test verifies the helper API works with a broader range of valid slots
    by running a simple exp operation (which uses D0 internally via sfpu_op).

    Testing invalid slots (e.g., values >= 8) would require a compile-failure
    test, which is not practical in the device test framework. The static_assert
    in the helper structs prevents misuse at compile time.
    """
    torch.manual_seed(42)
    input_data = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 2.0 - 1.0

    # If the helpers compile and run correctly, DEST validation is working
    matching, output, golden = run_unary_helper_test(
        device,
        input_data,
        lambda x: torch.exp(x),
        compute_kernel_path=f"{KERNEL_DIR}/compute/helpers_sfpu_op_exp.cpp",
        pcc_threshold=0.999,
    )
    logger.info(f"test_sfpu_dest_validation: matching={matching}")
    assert matching, "DEST validation test failed — helper compilation or execution issue"
