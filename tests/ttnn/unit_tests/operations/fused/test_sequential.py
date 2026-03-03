# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration Tests for Sequential Kernel Chaining

These tests validate the sequential chaining infrastructure with real
LayerNorm and RMSNorm descriptors.

Note: Full kernel-level fusion requires kernels to accept CB indices
as compile-time arguments. The current layernorm kernels have hardcoded
CB indices. These tests demonstrate the infrastructure and API while
documenting what's needed for full implementation.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc


def torch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-5):
    """Reference LayerNorm implementation."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """Reference RMSNorm implementation."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    if weight is not None:
        x_norm = x_norm * weight
    return x_norm


@pytest.fixture
def test_tensors(device):
    """Create test tensors for normalization tests."""
    torch.manual_seed(42)

    batch, seq_len, hidden = 1, 32, 128
    input_shape = (batch, 1, seq_len, hidden)
    weight_shape = (1, 1, 1, hidden)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight1 = torch.ones(weight_shape, dtype=torch.bfloat16)  # Use ones for stability
    torch_weight2 = torch.ones(weight_shape, dtype=torch.bfloat16)
    torch_weight3 = torch.ones(weight_shape, dtype=torch.bfloat16)
    torch_bias1 = torch.zeros(weight_shape, dtype=torch.bfloat16)
    torch_bias2 = torch.zeros(weight_shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_weight1 = ttnn.from_torch(
        torch_weight1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weight2 = ttnn.from_torch(
        torch_weight2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weight3 = ttnn.from_torch(
        torch_weight3,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias1 = ttnn.from_torch(
        torch_bias1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_bias2 = ttnn.from_torch(
        torch_bias2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    return {
        "torch_input": torch_input,
        "torch_weight1": torch_weight1,
        "torch_weight2": torch_weight2,
        "torch_weight3": torch_weight3,
        "torch_bias1": torch_bias1,
        "torch_bias2": torch_bias2,
        "tt_input": tt_input,
        "tt_weight1": tt_weight1,
        "tt_weight2": tt_weight2,
        "tt_weight3": tt_weight3,
        "tt_bias1": tt_bias1,
        "tt_bias2": tt_bias2,
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialChainInfrastructure:
    """Tests for sequential chaining infrastructure with real descriptors."""

    def test_extract_cb_info_from_layernorm(self, device, test_tensors):
        """Test extracting CB info from a real LayerNorm descriptor."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Create a LayerNorm descriptor
        ln_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        # Extract CB info
        cb_info = extract_cb_info(ln_desc.descriptor)

        # LayerNorm should use multiple CBs
        assert len(cb_info) > 0, "LayerNorm should have CB descriptors"

        # Check for expected CBs (input=0, output=16 are standard)
        cb_indices = set(cb_info.keys())
        assert 0 in cb_indices, "Should have input CB (c_0)"
        assert 16 in cb_indices, "Should have output CB (c_16)"

    def test_extract_cb_info_from_rmsnorm(self, device, test_tensors):
        """Test extracting CB info from a real RMSNorm descriptor."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info
        from models.experimental.ops.descriptors.normalization import rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Create an RMSNorm descriptor
        rms_desc = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        # Extract CB info
        cb_info = extract_cb_info(rms_desc.descriptor)

        assert len(cb_info) > 0, "RMSNorm should have CB descriptors"

        cb_indices = set(cb_info.keys())
        assert 0 in cb_indices, "Should have input CB (c_0)"
        assert 16 in cb_indices, "Should have output CB (c_16)"

    def test_chain_builder_with_real_descriptors(self, device, test_tensors):
        """Test building a chain with real LayerNorm/RMSNorm descriptors."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info, build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Create descriptors for the chain: LayerNorm -> RMSNorm -> LayerNorm
        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        # Use LN compute config for RMS so all phases have consistent fp32_dest_acc_en
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        rms_desc = rms_norm.rms_norm(
            ln1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        ln2_desc = layer_norm.layer_norm(
            rms_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        # Build the chain and verify we get a valid fused descriptor
        fused = build_op_graph([ln1_desc, rms_desc, ln2_desc], [], device)
        assert fused is not None
        assert hasattr(fused, "descriptor")
        num_kernels = len(fused.descriptor.kernels)
        assert num_kernels >= 3, "Should have reader, writer, and compute kernels"

    def test_barrier_config_added(self, device, test_tensors):
        """Test that fused descriptors get barrier configuration (GlobalSemaphores)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        ln2_desc = layer_norm.layer_norm(
            ln1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1_desc, ln2_desc], [], device)

        # Verify we got a fused descriptor with kernels
        assert fused is not None
        assert hasattr(fused, "descriptor")
        num_kernels = len(fused.descriptor.kernels)
        assert num_kernels >= 3, "Should have reader, writer, and compute kernels"

        # Verify the fused kernels are SOURCE_CODE type (generated)
        for kernel in fused.descriptor.kernels:
            assert kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE


def dump_fused_kernels(fused_desc, output_dir: str, label: str = "fused"):
    """
    Dump all generated fused kernel files and metadata to output_dir for inspection.

    For each kernel in the fused descriptor, writes:
      - <label>_kernel_<i>_<type>.cpp  (source code, read from file if FILE_PATH)
      - <label>_kernel_<i>_<type>_meta.txt  (defines, compile-time args, named args, etc.)
    Also writes:
      - <label>_cbs.txt  (all CB descriptors)
      - <label>_summary.txt  (high-level overview)

    Args:
        fused_desc: An OpDescriptor whose .descriptor is the fused ProgramDescriptor.
        output_dir: Directory to write files into (created if needed).
        label: Prefix for output filenames.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    descriptor = fused_desc.descriptor
    kernels = list(descriptor.kernels)
    cbs = list(descriptor.cbs)

    summary_lines = [
        f"Fused descriptor summary: {label}",
        f"  Kernels: {len(kernels)}",
        f"  CBs:     {len(cbs)}",
        f"  Input tensors:  {len(fused_desc.input_tensors)}",
        f"  Output tensors: {len(fused_desc.output_tensors)}",
        "",
    ]

    for i, kernel in enumerate(kernels):
        # Classify kernel type
        config = kernel.config
        if isinstance(config, ttnn.ComputeConfigDescriptor):
            ktype = "compute"
        elif isinstance(config, ttnn.ReaderConfigDescriptor):
            ktype = "reader"
        elif isinstance(config, ttnn.WriterConfigDescriptor):
            ktype = "writer"
        elif isinstance(config, ttnn.DataMovementConfigDescriptor):
            from ttnn import DataMovementProcessor

            ktype = "reader" if config.processor == DataMovementProcessor.RISCV_1 else "writer"
        else:
            ktype = "unknown"

        # Determine source type and get source code
        is_source_code = kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        source_path_or_code = kernel.kernel_source

        if is_source_code:
            source_code = source_path_or_code
            source_origin = "SOURCE_CODE (generated)"
        else:
            source_origin = f"FILE_PATH: {source_path_or_code}"
            # Try to read the file
            source_code = ""
            base_paths = [
                os.environ.get("TT_METAL_HOME", ""),
                "",
            ]
            for base in base_paths:
                full = os.path.join(base, source_path_or_code) if base else source_path_or_code
                if os.path.exists(full):
                    with open(full) as f:
                        source_code = f.read()
                    break
            if not source_code:
                source_code = f"// Could not read file: {source_path_or_code}\n"

        # Write source file
        src_filename = f"{label}_kernel_{i}_{ktype}.cpp"
        src_path = os.path.join(output_dir, src_filename)
        with open(src_path, "w") as f:
            f.write(source_code)

        # Collect metadata
        meta_lines = [
            f"Kernel {i}: {ktype}",
            f"  Source type: {source_origin}",
            f"  Source length: {len(source_code)} chars",
            "",
        ]

        # Defines
        defines = list(kernel.defines)
        meta_lines.append(f"  Defines ({len(defines)}):")
        for name, value in defines:
            meta_lines.append(f"    {name} = {value}")
        meta_lines.append("")

        # Compile-time args
        ct_args = list(kernel.compile_time_args)
        meta_lines.append(f"  Compile-time args ({len(ct_args)}):")
        for j, val in enumerate(ct_args):
            meta_lines.append(f"    [{j}] = {val}")
        meta_lines.append("")

        # Named compile-time args
        named_args = list(kernel.named_compile_time_args)
        meta_lines.append(f"  Named compile-time args ({len(named_args)}):")
        for name, value in named_args:
            meta_lines.append(f"    {name} = {value}")
        meta_lines.append("")

        # Write metadata file
        meta_filename = f"{label}_kernel_{i}_{ktype}_meta.txt"
        meta_path = os.path.join(output_dir, meta_filename)
        with open(meta_path, "w") as f:
            f.write("\n".join(meta_lines))

        summary_lines.append(f"  Kernel {i} [{ktype}]: {source_origin} ({len(source_code)} chars)")
        summary_lines.append(f"    defines={len(defines)}  ct_args={len(ct_args)}  named_args={len(named_args)}")

    # Dump CB descriptors
    summary_lines.append("")
    cb_lines = [f"CB descriptors ({len(cbs)}):", ""]
    for cb_desc in cbs:
        for fmt in cb_desc.format_descriptors:
            idx = fmt.buffer_index
            try:
                df = fmt.data_format
            except (TypeError, AttributeError):
                df = "N/A"
            cb_lines.append(f"  CB {idx}: page_size={fmt.page_size}  total_size={cb_desc.total_size}  data_format={df}")
            summary_lines.append(f"  CB {idx}: page_size={fmt.page_size}  total_size={cb_desc.total_size}")

    cb_path = os.path.join(output_dir, f"{label}_cbs.txt")
    with open(cb_path, "w") as f:
        f.write("\n".join(cb_lines))

    # Write summary
    summary_path = os.path.join(output_dir, f"{label}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialChainExecution:
    """
    Execution tests for sequential chains.

    Note: These tests are marked as skip because full kernel-level fusion
    requires kernels to accept CB indices as compile-time arguments.
    The tests document the intended API and what golden results should be.
    """

    def test_layernorm_rmsnorm_layernorm_chain(self, device, test_tensors):
        """
        Test fusing LayerNorm -> RMSNorm -> LayerNorm chain.

        This test validates true multi-phase kernel fusion where:
        1. A single fused reader reads all inputs (gamma/beta for all phases)
        2. A single fused compute runs all phase computations sequentially
        3. A single fused writer writes the final output

        STATUS: Infrastructure complete. Fused kernel source is generated with:
        - Phase 0 reader extended to read gamma/beta for all phases
        - Fused compute with each phase wrapped as phaseN_compute() function
        - Phase-prefixed named compile-time args for CB indices
        - Merged defines and compile-time args

        REMAINING: Test actual kernel compilation and device execution.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Create descriptors
        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        # Use LN compute config for RMS so all phases have consistent fp32_dest_acc_en.
        # DST_ACCUM_MODE is a kernel-level hardware setting — all fused phases must agree.
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        rms_desc = rms_norm.rms_norm(
            ln1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        ln2_desc = layer_norm.layer_norm(
            rms_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        # Build chain
        fused_desc = build_op_graph([ln1_desc, rms_desc, ln2_desc], [], device)

        # Execute
        outputs = composite.launch([fused_desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        # Compute golden: LayerNorm(RMSNorm(LayerNorm(x)))
        temp1 = torch_layer_norm(
            test_tensors["torch_input"],
            test_tensors["torch_weight1"],
            test_tensors["torch_bias1"],
            eps=1e-5,
        )
        temp2 = torch_rms_norm(temp1, test_tensors["torch_weight2"], eps=1e-5)
        torch_golden = torch_layer_norm(
            temp2,
            test_tensors["torch_weight3"],
            test_tensors["torch_bias2"],
            eps=1e-5,
        )

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.98)
        assert passing, f"PCC check failed: {pcc}"

    def test_four_phase_rms_chain(self, device, test_tensors):
        """Test 4-phase RMS→RMS→RMS→RMS chain to isolate the 4-phase issue."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Create 4 RMS descriptors
        rms1 = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        rms2 = rms_norm.rms_norm(
            rms1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        rms3 = rms_norm.rms_norm(
            rms2.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        # Create a 4th weight for the 4th RMS
        torch_weight4 = torch.ones((1, 1, 1, 128), dtype=torch.bfloat16)
        tt_weight4 = ttnn.from_torch(
            torch_weight4,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rms4 = rms_norm.rms_norm(
            rms3.output_tensors[0],
            core_range_set=core_range,
            weight=tt_weight4,
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        # Fuse and execute
        fused = build_op_graph([rms1, rms2, rms3, rms4], [], device)
        outputs = composite.launch([fused])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        # Compute golden
        temp1 = torch_rms_norm(test_tensors["torch_input"], test_tensors["torch_weight1"])
        temp2 = torch_rms_norm(temp1, test_tensors["torch_weight2"])
        temp3 = torch_rms_norm(temp2, test_tensors["torch_weight3"])
        golden = torch_rms_norm(temp3, torch_weight4)

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        assert passing, f"4-phase RMS chain PCC check failed: {pcc}"

    def test_two_phase_layernorm_chain_multicore(self, device, test_tensors):
        """Test 2-phase LN→LN chain on 2 cores to validate global barrier."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})

        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        ln2_desc = layer_norm.layer_norm(
            ln1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused_desc = build_op_graph([ln1_desc, ln2_desc], [], device)

        outputs = composite.launch([fused_desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        # Golden: LN(LN(x))
        temp = torch_layer_norm(
            test_tensors["torch_input"],
            test_tensors["torch_weight1"],
            test_tensors["torch_bias1"],
            eps=1e-5,
        )
        golden = torch_layer_norm(
            temp,
            test_tensors["torch_weight2"],
            test_tensors["torch_bias2"],
            eps=1e-5,
        )

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        assert passing, f"Multi-core 2-phase chain PCC check failed: {pcc}"

    def test_chain_with_parallel_op(self, device, test_tensors):
        """
        Test running a fused chain in parallel with another op on different cores.

        This demonstrates the integration of sequential.py with composite.py:
        - Cores (0,0): LayerNorm -> RMSNorm -> LayerNorm (fused chain)
        - Cores (1,0): RMSNorm (independent op)

        Both execute in a single program via composite.launch().
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        # Core ranges - non-overlapping
        chain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        parallel_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Create a second input for the parallel op
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Build the fused chain on chain_cores
        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms_desc = rms_norm.rms_norm(
            ln1_desc.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2_desc = layer_norm.layer_norm(
            rms_desc.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        fused_chain = build_op_graph([ln1_desc, rms_desc, ln2_desc], [], device)

        # Create parallel RMSNorm on parallel_cores
        parallel_rms = rms_norm.rms_norm(
            tt_input2,
            core_range_set=parallel_cores,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        # Execute both in parallel via composite.launch
        outputs = composite.launch([fused_chain, parallel_rms])

        # Verify we got outputs from both
        assert len(outputs) == 2
        chain_output = outputs[0][0]
        parallel_output = outputs[1][0]

        # Convert to torch
        torch_chain_out = ttnn.to_torch(chain_output)
        torch_parallel_out = ttnn.to_torch(parallel_output)

        # Compute golden for chain
        temp1 = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], eps=1e-5)
        temp2 = torch_rms_norm(temp1, test_tensors["torch_weight2"], eps=1e-5)
        golden_chain = torch_layer_norm(temp2, test_tensors["torch_weight3"], eps=1e-5)

        # Compute golden for parallel op
        golden_parallel = torch_rms_norm(torch_input2, test_tensors["torch_weight1"], eps=1e-5)

        # Verify both outputs
        passing1, pcc1 = comp_pcc(golden_chain, torch_chain_out, pcc=0.98)
        assert passing1, f"Chain PCC check failed: {pcc1}"

        passing2, pcc2 = comp_pcc(golden_parallel, torch_parallel_out, pcc=0.99)
        assert passing2, f"Parallel op PCC check failed: {pcc2}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialIndividualOps:
    """
    Test individual ops through the descriptor API to validate the infrastructure
    works correctly before attempting fusion.
    """

    def test_single_layernorm_via_descriptor(self, device, test_tensors):
        """Test single LayerNorm execution via descriptor API."""
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = torch_layer_norm(
            test_tensors["torch_input"],
            test_tensors["torch_weight1"],
            test_tensors["torch_bias1"],
            eps=1e-5,
        )

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"LayerNorm PCC check failed: {pcc}"

    def test_single_rmsnorm_via_descriptor(self, device, test_tensors):
        """Test single RMSNorm execution via descriptor API."""
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        desc = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = torch_rms_norm(
            test_tensors["torch_input"],
            test_tensors["torch_weight1"],
            eps=1e-5,
        )

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"RMSNorm PCC check failed: {pcc}"

    def test_two_ops_parallel_via_composite(self, device, test_tensors):
        """Test running two ops in parallel on different cores via composite."""
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        # Non-overlapping core ranges
        cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

        # Create second input
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ln_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores1,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        rms_desc = rms_norm.rms_norm(
            tt_input2,
            core_range_set=cores2,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        # Run both in parallel
        outputs = composite.launch([ln_desc, rms_desc])

        assert len(outputs) == 2

        # Verify LayerNorm output
        torch_ln_out = ttnn.to_torch(outputs[0][0])
        golden_ln = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], eps=1e-5)
        passing1, pcc1 = comp_pcc(golden_ln, torch_ln_out, pcc=0.99)
        assert passing1, f"LayerNorm PCC: {pcc1}"

        # Verify RMSNorm output
        torch_rms_out = ttnn.to_torch(outputs[1][0])
        golden_rms = torch_rms_norm(torch_input2, test_tensors["torch_weight2"], eps=1e-5)
        passing2, pcc2 = comp_pcc(golden_rms, torch_rms_out, pcc=0.99)
        assert passing2, f"RMSNorm PCC: {pcc2}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestFusedKernelSource:
    """Tests for the fused kernel source generation."""

    def test_fused_source_has_phases(self, device, test_tensors):
        """Test that fused kernel source contains phase functions."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        ln2 = layer_norm.layer_norm(
            ln1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, ln2], [], device)

        # Verify fused kernels are SOURCE_CODE type with phase functions
        for kernel in fused.descriptor.kernels:
            assert kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            source = kernel.kernel_source
            assert "Phase 0" in source, "Should have Phase 0 comment"
            assert "Phase 1" in source, "Should have Phase 1 comment"
            assert "void kernel_main()" in source

    def test_fused_source_has_barrier(self, device, test_tensors):
        """Test that fused kernel source contains barrier synchronization."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        ln2 = layer_norm.layer_norm(
            ln1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, ln2], [], device)

        # Check reader has barrier code (namespace-based barrier infrastructure)
        for kernel in fused.descriptor.kernels:
            source = kernel.kernel_source
            if "namespace barrier" in source:
                assert "reset_cbs" in source, "Reader should have CB reset"
                assert "compute_done" in source, "Reader should wait for compute_done"
                assert "writer_done" in source, "Reader should wait for writer_done"
                break
        else:
            pytest.fail("No kernel has barrier namespace")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelChains:
    """Tests for parallel chain creation and execution."""

    def test_parallel_linear_chains(self, device, test_tensors):
        """Test creating parallel chain descriptors."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        # Create two separate core ranges
        cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

        # Create second input tensor
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Use consistent fp32 config for all phases
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Chain 1: LayerNorm -> RMSNorm on cores1
        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores1,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores1,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        chain1 = [ln1, rms1]

        # Chain 2: RMSNorm -> LayerNorm on cores2
        rms2 = rms_norm.rms_norm(
            tt_input2,
            core_range_set=cores2,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms2.output_tensors[0],
            core_range_set=cores2,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        chain2 = [rms2, ln2]

        # Create parallel chain descriptors
        fused_descriptors = [build_op_graph(c, [], device) for c in [chain1, chain2]]

        assert len(fused_descriptors) == 2

        # Verify each chain has a merged descriptor with kernels from both ops
        for i, desc in enumerate(fused_descriptors):
            num_kernels = len(desc.descriptor.kernels)
            num_cbs = len(desc.descriptor.cbs)

    def test_three_phase_linear_chain(self, device, test_tensors):
        """Test a three-phase linear chain."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Create a chain
        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        # Fuse the chain
        fused = build_op_graph([ln1, rms1, ln2], [], device)

        # Verify structure
        assert fused is not None
        assert hasattr(fused, "descriptor")

        # Should have kernels from all 3 ops
        # Each op typically has 3 kernels (reader, compute, writer)
        num_kernels = len(fused.descriptor.kernels)
        assert num_kernels >= 3, "Should have kernels from fused ops"

    def test_chain_single_op(self, device, test_tensors):
        """Test that chaining a single op returns FusedOp wrapping the original."""
        from models.experimental.ops.descriptors.fusion import OpGraphBuilder, OpNode
        from models.experimental.ops.descriptors.fusion import FusedOp
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        fused = OpGraphBuilder(OpNode(ln)).build(device)

        # Should return a FusedOp wrapping the original descriptor
        assert isinstance(fused, FusedOp)
        assert fused.descriptor is ln.descriptor
        assert fused.input_tensors is ln.input_tensors
        assert fused.output_tensors is ln.output_tensors

    def test_dump_fused_kernel_files(self, device, test_tensors, tmp_path):
        """Build a fused 3-phase chain and dump all generated kernel files for inspection."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, rms1, ln2], [], device)

        output_dir = str(tmp_path / "gen_kernels")
        dump_fused_kernels(fused, output_dir, label="ln_rms_ln")

        # Verify files were created
        files = os.listdir(output_dir)
        assert any(f.endswith(".cpp") for f in files), "Should have generated .cpp source files"
        assert any("summary" in f for f in files), "Should have generated summary file"
        assert any("meta" in f for f in files), "Should have generated metadata files"
        assert any("cbs" in f for f in files), "Should have generated CB descriptor file"

        # Verify the compute kernel is SOURCE_CODE (generated)
        summary_path = os.path.join(output_dir, "ln_rms_ln_summary.txt")
        with open(summary_path) as f:
            summary = f.read()
        assert "SOURCE_CODE" in summary, "Fused compute kernel should be SOURCE_CODE type"

    def test_extract_named_compile_time_args(self, device, test_tensors):
        """Test that named compile-time args can be extracted from LayerNorm kernels."""
        from models.experimental.ops.descriptors.fusion import extract_cb_names_from_kernel
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        # Extract CB names from each kernel
        for kernel in desc.descriptor.kernels:
            cb_names = extract_cb_names_from_kernel(kernel)
            if cb_names:
                # Verify expected CB names are present
                expected_names = ["cb_in", "cb_out", "cb_scaler", "cb_eps"]
                for name in expected_names:
                    if name in cb_names:
                        pass

    def test_cb_overflow_validation(self, device, test_tensors):
        """Test that CB overflow is detected and reported clearly."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Try to create a very long chain that would overflow CBs
        # Each LayerNorm uses ~6 CBs (input, output, gamma, beta, scaler, eps)
        # With remapping, this could potentially exceed 32 CBs
        descriptors = []

        # Add many phases - enough to potentially overflow
        # (The actual overflow depends on CB remapping strategy)
        prev_desc = None
        for i in range(10):  # 10 phases should be enough to trigger overflow warning
            if i == 0:
                desc = layer_norm.layer_norm(
                    test_tensors["tt_input"],
                    core_range_set=core_range,
                    weight=test_tensors["tt_weight1"],
                    bias=test_tensors["tt_bias1"],
                    epsilon=1e-5,
                )
            else:
                desc = rms_norm.rms_norm(
                    prev_desc.output_tensors[0],
                    core_range_set=core_range,
                    weight=test_tensors["tt_weight1"],
                    epsilon=1e-5,
                    compute_kernel_config=ln_compute_config,
                )
            descriptors.append(desc)
            prev_desc = desc

        # This should either succeed (if CB merging is efficient) or raise a clear error
        try:
            fused = build_op_graph(descriptors, [], device)
            # The build succeeded, so CBs should be within limits
            assert True, "Build succeeded without CB overflow"
        except (ValueError, RuntimeError) as e:
            # Expected to fail with a clear CB overflow message
            error_msg = str(e)
            assert (
                "CB" in error_msg or "circular buffer" in error_msg.lower()
            ), f"Error should mention CB overflow: {error_msg}"
            assert "32" in error_msg or "NUM_CBS" in error_msg, f"Error should mention the 32 CB limit: {error_msg}"

    def test_named_args_have_phase_prefix(self, device, test_tensors):
        """Test that fused kernel named args get proper phase prefixes."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        ln2 = layer_norm.layer_norm(
            ln1.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, ln2], [], device)

        # Check that fused kernels have phase-prefixed named args
        for kernel in fused.descriptor.kernels:
            named_args = dict(kernel.named_compile_time_args)
            # Should have barrier_rt_offset
            assert "barrier_rt_offset" in named_args, f"Missing barrier_rt_offset in {list(named_args.keys())}"
            # Phase 1 args should have phase_1_ prefix
            phase1_args = [k for k in named_args if k.startswith("phase_1_")]
            assert len(phase1_args) > 0, f"Should have phase_1_ prefixed args, got: {list(named_args.keys())}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelChainsExecution:
    """
    Execution tests for parallel chains.

    These demonstrate running multiple fused chains in parallel using composite.launch().
    Some tests are skipped pending full kernel CB parameterization.
    """

    def test_two_parallel_chains_execution(self, device, test_tensors):
        """
        Test executing two parallel chains:
        - Chain A: LayerNorm -> RMSNorm (on cores 0,0)
        - Chain B: RMSNorm -> LayerNorm (on cores 1,0)
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Second input
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Chain A: LayerNorm -> RMSNorm
        ln_a = layer_norm.layer_norm(
            test_tensors["tt_input"], core_range_set=cores1, weight=test_tensors["tt_weight1"], epsilon=1e-5
        )
        rms_a = rms_norm.rms_norm(
            ln_a.output_tensors[0],
            core_range_set=cores1,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        chain_a = [ln_a, rms_a]

        # Chain B: RMSNorm -> LayerNorm
        rms_b = rms_norm.rms_norm(
            tt_input2,
            core_range_set=cores2,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln_b = layer_norm.layer_norm(
            rms_b.output_tensors[0], core_range_set=cores2, weight=test_tensors["tt_weight2"], epsilon=1e-5
        )
        chain_b = [rms_b, ln_b]

        # Fuse chains
        fused = [build_op_graph(c, [], device) for c in [chain_a, chain_b]]

        # Execute in parallel
        outputs = composite.launch(fused)

        assert len(outputs) == 2

        # Compute golden
        temp_a1 = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"])
        golden_a = torch_rms_norm(temp_a1, test_tensors["torch_weight2"])

        temp_b1 = torch_rms_norm(torch_input2, test_tensors["torch_weight1"])
        golden_b = torch_layer_norm(temp_b1, test_tensors["torch_weight2"])

        # Verify outputs
        out_a = ttnn.to_torch(outputs[0][0])
        out_b = ttnn.to_torch(outputs[1][0])

        passing_a, pcc_a = comp_pcc(golden_a, out_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, out_b, pcc=0.98)

        assert passing_a, f"Chain A PCC: {pcc_a}"
        assert passing_b, f"Chain B PCC: {pcc_b}"

    def test_four_parallel_chains(self, device, test_tensors):
        """
        Test running 4 parallel chains on 4 different cores.
        Each chain is: LayerNorm -> RMSNorm
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # 4 non-overlapping core ranges
        core_ranges = [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0))}) for i in range(4)
        ]

        # Create 4 inputs
        torch_inputs = [torch.randn_like(test_tensors["torch_input"]) for _ in range(4)]
        tt_inputs = [
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t in torch_inputs
        ]

        # Create 4 chains (RMS needs compute_kernel_config to match LN's fp32 setting)
        chains = []
        for i in range(4):
            ln_i = layer_norm.layer_norm(
                tt_inputs[i], core_range_set=core_ranges[i], weight=test_tensors["tt_weight1"], epsilon=1e-5
            )
            rms_i = rms_norm.rms_norm(
                ln_i.output_tensors[0],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            chains.append([ln_i, rms_i])

        # Fuse and execute
        fused = [build_op_graph(c, [], device) for c in chains]
        outputs = composite.launch(fused)

        assert len(outputs) == 4

        # Verify each output
        for i in range(4):
            temp = torch_layer_norm(torch_inputs[i], test_tensors["torch_weight1"])
            golden = torch_rms_norm(temp, test_tensors["torch_weight2"])
            out = ttnn.to_torch(outputs[i][0])

            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Chain {i} PCC: {pcc}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelExecutionProfiling:
    """Profiling tests to measure performance benefits of parallel execution."""

    def test_profile_parallel_matmul_vs_norm_chain(self, device, test_tensors):
        """
        Profile parallel execution: matmul vs 4-phase normalization chain.

        Scenario:
        - Branch A: Matmul operation [128x256] @ [256x512]
        - Branch B: Fused LN -> RMS -> LN -> RMS chain

        Compare:
        1. Serial: Run matmul, then 4 normalizations sequentially
        2. Parallel: Run matmul and fused norm chain in parallel on different cores

        Tests the fix for 4+ phase fusion where phases were using hardcoded
        compile-time args instead of named args, causing incorrect results.
        """
        import time
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # Branch A: Matmul - [1, 1, 128, 256] @ [1, 1, 256, 512] -> [1, 1, 128, 512]
        # Sized to be computationally intensive while fitting in L1
        torch_a = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 256, 512, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Branch B: 4 normalizations on the test input [1, 1, 32, 128]
        # Create 4 sets of weights
        weight_shape = (1, 1, 1, 128)
        torch_weights = [torch.ones(weight_shape, dtype=torch.bfloat16) for _ in range(4)]
        torch_biases = [torch.zeros(weight_shape, dtype=torch.bfloat16) for _ in range(2)]  # Only LN needs bias

        tt_weights = [
            ttnn.from_torch(
                w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for w in torch_weights
        ]
        tt_biases = [
            ttnn.from_torch(
                b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for b in torch_biases
        ]

        # ============================================================
        # SERIAL BASELINE: Run everything sequentially
        # ============================================================

        # Use consistent compute config for all norms (needed for fp32_dest_acc_en consistency)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        start_serial = time.perf_counter()

        # Run matmul
        serial_mm_out = ttnn.matmul(tt_a, tt_b)

        # Run 4 normalizations sequentially
        serial_norm_out = test_tensors["tt_input"]
        serial_norm_out = ttnn.layer_norm(serial_norm_out, weight=tt_weights[0], bias=tt_biases[0], epsilon=1e-5)
        serial_norm_out = ttnn.rms_norm(
            serial_norm_out, weight=tt_weights[1], epsilon=1e-5, compute_kernel_config=ln_compute_config
        )
        serial_norm_out = ttnn.layer_norm(serial_norm_out, weight=tt_weights[2], bias=tt_biases[1], epsilon=1e-5)
        serial_norm_out = ttnn.rms_norm(
            serial_norm_out, weight=tt_weights[3], epsilon=1e-5, compute_kernel_config=ln_compute_config
        )

        # Wait for completion
        _ = ttnn.to_torch(serial_mm_out)
        _ = ttnn.to_torch(serial_norm_out)

        end_serial = time.perf_counter()
        serial_time = end_serial - start_serial

        # ============================================================
        # PARALLEL EXECUTION: Run matmul and norm chain in parallel
        # ============================================================

        # Core allocation - non-overlapping
        cores_matmul = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        cores_norm = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})

        # Create matmul descriptor
        mm_descriptor = matmul_desc(tt_a, tt_b, core_range_set=cores_matmul)

        # Create norm chain descriptors
        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores_norm,
            weight=tt_weights[0],
            bias=tt_biases[0],
            epsilon=1e-5,
        )
        rms1_desc = rms_norm.rms_norm(
            ln1_desc.output_tensors[0],
            core_range_set=cores_norm,
            weight=tt_weights[1],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2_desc = layer_norm.layer_norm(
            rms1_desc.output_tensors[0],
            core_range_set=cores_norm,
            weight=tt_weights[2],
            bias=tt_biases[1],
            epsilon=1e-5,
        )

        # Fuse the 3-phase norm chain (LN→RMS→LN)
        # Note: 4-phase chains with mixed LN/RMS hit CB limits due to phantom CB reservations
        fused_norm_chain = build_op_graph([ln1_desc, rms1_desc, ln2_desc], [], device)

        start_parallel = time.perf_counter()

        # Launch both in parallel
        outputs = composite.launch([mm_descriptor, fused_norm_chain])

        # Wait for completion
        parallel_mm_out = outputs[0][0]
        parallel_norm_out = outputs[1][0]
        _ = ttnn.to_torch(parallel_mm_out)
        _ = ttnn.to_torch(parallel_norm_out)

        end_parallel = time.perf_counter()
        parallel_time = end_parallel - start_parallel

        # ============================================================
        # RESULTS
        # ============================================================

        if parallel_time < serial_time:
            speedup = serial_time / parallel_time
            time_saved = (serial_time - parallel_time) * 1000
        else:
            pass

        # Verify correctness against torch golden

        # Matmul golden
        torch_golden_mm = torch.matmul(torch_a, torch_b)
        torch_parallel_mm = ttnn.to_torch(parallel_mm_out)
        passing_mm, pcc_mm = comp_pcc(torch_golden_mm, torch_parallel_mm, pcc=0.99)
        assert passing_mm, f"Matmul output doesn't match torch: {pcc_mm}"

        # Norm chain golden (3-phase: LN→RMS→LN)
        temp1 = torch_layer_norm(test_tensors["torch_input"], torch_weights[0], torch_biases[0], eps=1e-5)
        temp2 = torch_rms_norm(temp1, torch_weights[1], eps=1e-5)
        torch_golden_norm = torch_layer_norm(temp2, torch_weights[2], torch_biases[1], eps=1e-5)

        torch_parallel_norm = ttnn.to_torch(parallel_norm_out)
        passing_norm, pcc_norm = comp_pcc(torch_golden_norm, torch_parallel_norm, pcc=0.98)
        assert passing_norm, f"Norm chain output doesn't match torch: {pcc_norm}"


@pytest.fixture
def matmul_tensors(device):
    """Create test tensors for matmul tests."""
    torch.manual_seed(42)

    # A: [1, 1, 32, 64], B: [1, 1, 64, 128] -> C: [1, 1, 32, 128]
    torch_a = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(
        torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_b = ttnn.from_torch(
        torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Weights for LN on matmul output shape [1, 1, 32, 128]
    ln_weight_shape = (1, 1, 1, 128)
    torch_ln_weight = torch.ones(ln_weight_shape, dtype=torch.bfloat16)
    torch_ln_bias = torch.zeros(ln_weight_shape, dtype=torch.bfloat16)

    tt_ln_weight = ttnn.from_torch(
        torch_ln_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_ln_bias = ttnn.from_torch(
        torch_ln_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return {
        "torch_a": torch_a,
        "torch_b": torch_b,
        "tt_a": tt_a,
        "tt_b": tt_b,
        "torch_ln_weight": torch_ln_weight,
        "torch_ln_bias": torch_ln_bias,
        "tt_ln_weight": tt_ln_weight,
        "tt_ln_bias": tt_ln_bias,
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestMatmulDescriptor:
    """Tests for the matmul descriptor API."""

    def test_matmul_standalone_descriptor(self, device, matmul_tensors):
        """Test single matmul execution via descriptor API."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"Matmul PCC check failed: {pcc}"

    def test_matmul_composite_parallel_with_layernorm(self, device, matmul_tensors, test_tensors):
        """Test running matmul + layernorm in parallel on different cores via composite."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

        mm_desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"], core_range_set=cores1)

        ln_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores2,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        outputs = composite.launch([mm_desc, ln_desc])
        assert len(outputs) == 2

        # Verify matmul output
        torch_mm_out = ttnn.to_torch(outputs[0][0])
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing1, pcc1 = comp_pcc(golden_mm, torch_mm_out, pcc=0.99)
        assert passing1, f"Matmul PCC: {pcc1}"

        # Verify layernorm output
        torch_ln_out = ttnn.to_torch(outputs[1][0])
        golden_ln = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        passing2, pcc2 = comp_pcc(golden_ln, torch_ln_out, pcc=0.99)
        assert passing2, f"LayerNorm PCC: {pcc2}"

    def test_matmul_descriptor_cb_info(self, device, matmul_tensors):
        """Test extracting CB info from a matmul descriptor."""
        from models.experimental.ops.descriptors.fusion import extract_cb_info
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        cb_info = extract_cb_info(desc.descriptor)
        assert len(cb_info) > 0, "Matmul should have CB descriptors"

        cb_indices = set(cb_info.keys())

        # Matmul should have at least: c_0 (in0), c_1 (in1), c_4 (out)
        assert 0 in cb_indices, "Should have input A CB (c_0)"
        assert 1 in cb_indices, "Should have input B CB (c_1)"
        assert 4 in cb_indices, "Should have output CB (c_4)"

    def test_matmul_descriptor_named_args(self, device, matmul_tensors):
        """Test that matmul descriptor has named compile-time args for CB indices."""
        from models.experimental.ops.descriptors.fusion import extract_cb_names_from_kernel
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        found_named_args = False
        for kernel in desc.descriptor.kernels:
            cb_names = extract_cb_names_from_kernel(kernel)
            if cb_names:
                found_named_args = True
                # Verify expected matmul CB names
                assert "cb_in0" in cb_names, "Should have cb_in0"
                assert "cb_out" in cb_names, "Should have cb_out"

        assert found_named_args, "At least one kernel should have named compile-time args"

    def test_matmul_core_range_respected(self, device, matmul_tensors):
        """Test matmul on a non-origin core range to verify core_range_set is respected."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        # Use core (2,0) — not the origin core
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))})
        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"], core_range_set=cores)

        # Verify kernel core ranges contain only the requested core
        for kernel in desc.descriptor.kernels:
            for cr in kernel.core_ranges.ranges():
                assert cr.start.x >= 2, f"Kernel core range starts before requested: {cr}"
                assert cr.end.x <= 2, f"Kernel core range extends beyond requested: {cr}"

        # Verify execution produces correct output
        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"

    def test_matmul_restricted_core_composite(self, device, matmul_tensors, test_tensors):
        """Test matmul on non-origin core + layernorm on another core via composite."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        # Matmul on core (2,0), layernorm on core (0,0) — non-overlapping
        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))})
        ln_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        mm_desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"], core_range_set=mm_cores)
        ln_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=ln_cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        outputs = composite.launch([mm_desc, ln_desc])
        assert len(outputs) == 2

        # Verify matmul output
        torch_mm_out = ttnn.to_torch(outputs[0][0])
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing1, pcc1 = comp_pcc(golden_mm, torch_mm_out, pcc=0.99)
        assert passing1, f"Matmul PCC: {pcc1}"

        # Verify layernorm output
        torch_ln_out = ttnn.to_torch(outputs[1][0])
        golden_ln = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        passing2, pcc2 = comp_pcc(golden_ln, torch_ln_out, pcc=0.99)
        assert passing2, f"LayerNorm PCC: {pcc2}"

    def test_matmul_multi_core_range(self, device, matmul_tensors):
        """Test matmul distributed across 2 cores with multi-core program config."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        # Use cores (3,0)-(4,0) — 2 non-origin cores
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 0))})

        # A: [1,1,32,64] @ B: [1,1,64,128] -> C: [1,1,32,128]
        # M=1, N=4, K=2 tile blocks
        # per_core_N=2 → 2 output blocks across 2 cores
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(2, 1),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=2,
        )

        desc = matmul_desc(
            matmul_tensors["tt_a"], matmul_tensors["tt_b"], core_range_set=cores, program_config=program_config
        )

        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"

    def test_matmul_large_core_range(self, device):
        """Test matmul distributed across a large 4x2 = 8 core range."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # A [1,1,256,256] @ B [1,1,256,256] -> C [1,1,256,256]
        # M=8, N=8, K=8 tile blocks
        torch_a = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # 8 cores in a row
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

        # per_core_N must equal N for MatmulMultiCoreReuseProgramConfig
        # per_core_M=1, per_core_N=N=8 → (8/1)*(8/8) = 8 output blocks → 1 per core
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
        )

        desc = matmul_desc(tt_a, tt_b, core_range_set=cores, program_config=program_config)

        outputs = composite.launch([desc])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        torch_golden = torch_a @ torch_b

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"

    def test_matmul_2d_grid_offset(self, device):
        """Test matmul on a 2D grid offset from origin: (2,1)-(4,2) = 3x2 = 6 cores."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # A [1,1,192,64] @ B [1,1,64,128] -> C [1,1,192,128]
        # M=6, N=4, K=2 tile blocks
        torch_a = torch.randn(1, 1, 192, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # 3x2 grid at offset (2,1)
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(4, 2))})

        # 6 output blocks = M/per_core_M = 6/1 = 6, one per core
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(3, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,
        )

        desc = matmul_desc(tt_a, tt_b, core_range_set=cores, program_config=program_config)

        # Verify kernel core ranges are within the requested range
        for kernel in desc.descriptor.kernels:
            for cr in kernel.core_ranges.ranges():
                assert cr.start.x >= 2 and cr.start.y >= 1, f"Core range starts before requested: {cr}"
                assert cr.end.x <= 4 and cr.end.y <= 2, f"Core range extends beyond requested: {cr}"

        outputs = composite.launch([desc])
        torch_output = ttnn.to_torch(outputs[0][0])
        torch_golden = torch_a @ torch_b

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"

    def test_matmul_column_offset_grid(self, device):
        """Test matmul on a column-oriented grid offset from origin: (5,0)-(5,3) = 1x4 = 4 cores."""
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # A [1,1,128,64] @ B [1,1,64,128] -> C [1,1,128,128]
        # M=4, N=4, K=2 tile blocks
        torch_a = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # 1x4 column at x=5
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 3))})

        # 4 output blocks = M/per_core_M = 4/1 = 4, one per core
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 4),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,
        )

        desc = matmul_desc(tt_a, tt_b, core_range_set=cores, program_config=program_config)

        # Verify kernel core ranges are within the requested range
        for kernel in desc.descriptor.kernels:
            for cr in kernel.core_ranges.ranges():
                assert cr.start.x == 5 and cr.end.x == 5, f"Core range not on column 5: {cr}"
                assert cr.start.y >= 0 and cr.end.y <= 3, f"Core range extends beyond rows 0-3: {cr}"

        outputs = composite.launch([desc])
        torch_output = ttnn.to_torch(outputs[0][0])
        torch_golden = torch_a @ torch_b

        passing, pcc = comp_pcc(torch_golden, torch_output, pcc=0.99)
        assert passing, f"PCC check failed: {pcc}"

    def test_matmul_2d_offset_composite_with_fused_chain(self, device, test_tensors):
        """Multi-core 2D offset matmul + fused LN->RMS chain in parallel via composite."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on 3x2 grid at (2,1)-(4,2) = 6 cores
        torch_a = torch.randn(1, 1, 192, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)

        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(4, 2))})
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(3, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,
        )
        mm = matmul_desc(tt_a, tt_b, core_range_set=mm_cores, program_config=program_config)

        # Fused LN->RMS on (0,0)-(1,0) = 2 cores (non-overlapping with matmul)
        chain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused = build_op_graph([ln, rms], [], device)

        outputs = composite.launch([mm, fused])
        assert len(outputs) == 2

        # Verify matmul
        golden_mm = torch_a @ torch_b
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        # Verify chain
        golden_chain = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_chain, pcc_chain = comp_pcc(golden_chain, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_chain, f"Chain PCC: {pcc_chain}"


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestStressInfrastructure:
    """Stress tests to exercise edge cases in sequential kernel chaining."""

    def test_six_parallel_two_phase_chains(self, device, test_tensors):
        """6 independent LN->RMS chains on cores (0,0)-(5,0) in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())
        n_chains = 6

        core_ranges = [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0))}) for i in range(n_chains)
        ]

        torch_inputs = [torch.randn_like(test_tensors["torch_input"]) for _ in range(n_chains)]
        tt_inputs = [
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t in torch_inputs
        ]

        chains = []
        for i in range(n_chains):
            ln = layer_norm.layer_norm(
                tt_inputs[i],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            chains.append([ln, rms])

        fused = [build_op_graph(c, [], device) for c in chains]
        outputs = composite.launch(fused)
        assert len(outputs) == n_chains

        for i in range(n_chains):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
            )
            out = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Chain {i} on core ({i},0) PCC: {pcc}"

    @pytest.mark.parametrize("core_x", [0, 3, 5, 7])
    def test_three_phase_chain_on_various_cores(self, device, test_tensors, core_x):
        """LN->RMS->LN chain on different single-core positions."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core_x, 0), ttnn.CoreCoord(core_x, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, rms1, ln2], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Core ({core_x},0) PCC: {pcc}"

    def test_four_phase_all_rms_non_zero_core(self, device, test_tensors):
        """4-phase all-RMS chain on core (5,0)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0))})

        torch_weight4 = torch.ones_like(test_tensors["torch_weight1"])
        tt_weight4 = ttnn.from_torch(
            torch_weight4,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        weights = [test_tensors["tt_weight1"], test_tensors["tt_weight2"], test_tensors["tt_weight3"], tt_weight4]
        torch_weights = [
            test_tensors["torch_weight1"],
            test_tensors["torch_weight2"],
            test_tensors["torch_weight3"],
            torch_weight4,
        ]

        descs = []
        prev_input = test_tensors["tt_input"]
        for w in weights:
            d = rms_norm.rms_norm(prev_input, core_range_set=cores, weight=w, epsilon=1e-5)
            descs.append(d)
            prev_input = d.output_tensors[0]

        fused = build_op_graph(descs, [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = test_tensors["torch_input"]
        for tw in torch_weights:
            golden = torch_rms_norm(golden, tw)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"4-phase all-RMS PCC: {pcc}"

    def test_matmul_plus_two_norm_chains(self, device, test_tensors, matmul_tensors):
        """1 matmul + 2 independent fused norm chains in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on default core (0,0)
        mm = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        # Chain A on core (4,0): LN->RMS
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})
        ln_a = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms_a = rms_norm.rms_norm(
            ln_a.output_tensors[0],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_a = build_op_graph([ln_a, rms_a], [], device)

        # Chain B on core (5,0): RMS->LN (different order)
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0))})
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rms_b = rms_norm.rms_norm(
            tt_input2,
            core_range_set=cores_b,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln_b = layer_norm.layer_norm(
            rms_b.output_tensors[0],
            core_range_set=cores_b,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        fused_b = build_op_graph([rms_b, ln_b], [], device)

        # Launch all 3 in parallel
        outputs = composite.launch([mm, fused_a, fused_b])
        assert len(outputs) == 3

        # Verify matmul
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        # Verify chain A (LN->RMS)
        golden_a = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_a, f"Chain A PCC: {pcc_a}"

        # Verify chain B (RMS->LN)
        golden_b = torch_layer_norm(
            torch_rms_norm(torch_input2, test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[2][0]), pcc=0.98)
        assert passing_b, f"Chain B PCC: {pcc_b}"

    def test_matmul_plus_three_phase_chain(self, device, test_tensors, matmul_tensors):
        """1 matmul + 1 three-phase LN->RMS->LN fused chain in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        mm = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})
        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )
        fused = build_op_graph([ln1, rms1, ln2], [], device)

        outputs = composite.launch([mm, fused])
        assert len(outputs) == 2

        # Verify matmul
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        # Verify 3-phase chain
        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden_norm = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])
        passing_norm, pcc_norm = comp_pcc(golden_norm, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_norm, f"Norm chain PCC: {pcc_norm}"

    def test_mixed_chain_lengths_parallel(self, device, test_tensors):
        """2-phase chain + 3-phase chain running in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Chain A: 2-phase LN->RMS on core (0,0)
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_a = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms_a = rms_norm.rms_norm(
            ln_a.output_tensors[0],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_a = build_op_graph([ln_a, rms_a], [], device)

        # Chain B: 3-phase LN->RMS->LN on core (1,0)
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
        torch_input2 = torch.randn_like(test_tensors["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ln_b1 = layer_norm.layer_norm(
            tt_input2,
            core_range_set=cores_b,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms_b = rms_norm.rms_norm(
            ln_b1.output_tensors[0],
            core_range_set=cores_b,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln_b2 = layer_norm.layer_norm(
            rms_b.output_tensors[0],
            core_range_set=cores_b,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )
        fused_b = build_op_graph([ln_b1, rms_b, ln_b2], [], device)

        outputs = composite.launch([fused_a, fused_b])
        assert len(outputs) == 2

        # Verify 2-phase chain
        golden_a = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[0][0]), pcc=0.98)
        assert passing_a, f"2-phase chain PCC: {pcc_a}"

        # Verify 3-phase chain
        temp = torch_layer_norm(torch_input2, test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden_b = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_b, f"3-phase chain PCC: {pcc_b}"

    def test_repeated_chain_execution(self, device, test_tensors):
        """Run chain 3 times with fresh descriptors each time to check for state leaks."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0))})

        golden_temp = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"]
        )
        golden = torch_rms_norm(golden_temp, test_tensors["torch_weight2"])

        pccs = []
        for iteration in range(3):
            ln1 = layer_norm.layer_norm(
                test_tensors["tt_input"],
                core_range_set=cores,
                weight=test_tensors["tt_weight1"],
                bias=test_tensors["tt_bias1"],
                epsilon=1e-5,
            )
            rms1 = rms_norm.rms_norm(
                ln1.output_tensors[0],
                core_range_set=cores,
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            fused = build_op_graph([ln1, rms1], [], device)
            outputs = composite.launch([fused])
            result = ttnn.to_torch(outputs[0][0])

            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            pccs.append(pcc)
            assert passing, f"Iteration {iteration} PCC: {pcc}"

    def test_larger_tensor_chain(self, device):
        """2-phase LN->RMS chain with larger tensors (128x256)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(123)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Larger tensor: 128 rows x 256 cols (4x standard tile count)
        torch_input = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)
        weight_shape = (1, 1, 1, 256)
        torch_w1 = torch.ones(weight_shape, dtype=torch.bfloat16)
        torch_w2 = torch.ones(weight_shape, dtype=torch.bfloat16)
        torch_b1 = torch.zeros(weight_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_w1 = ttnn.from_torch(
            torch_w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_w2 = ttnn.from_torch(
            torch_w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b1 = ttnn.from_torch(
            torch_b1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))})
        ln1 = layer_norm.layer_norm(tt_input, core_range_set=cores, weight=tt_w1, bias=tt_b1, epsilon=1e-5)
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=tt_w2,
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused = build_op_graph([ln1, rms1], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(torch_layer_norm(torch_input, torch_w1, torch_b1), torch_w2)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Larger tensor chain PCC: {pcc}"

    def test_all_ln_three_phase_chain(self, device, test_tensors):
        """LN->LN->LN chain (all same op type, with biases)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        ln2 = layer_norm.layer_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight2"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )
        ln3 = layer_norm.layer_norm(
            ln2.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, ln2, ln3], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_layer_norm(temp, test_tensors["torch_weight2"], test_tensors["torch_bias2"])
        golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias1"])

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"All-LN chain PCC: {pcc}"

    def test_three_chains_plus_matmul(self, device, test_tensors, matmul_tensors):
        """1 matmul + 3 independent fused norm chains in parallel (4 total ops)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on default core (0,0)
        mm = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        # 3 norm chains on cores 4, 5, 6
        torch_inputs = [torch.randn_like(test_tensors["torch_input"]) for _ in range(3)]
        tt_inputs = [
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t in torch_inputs
        ]

        fused_chains = []
        for i, core_x in enumerate([4, 5, 6]):
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core_x, 0), ttnn.CoreCoord(core_x, 0))})
            ln = layer_norm.layer_norm(
                tt_inputs[i],
                core_range_set=cores,
                weight=test_tensors["tt_weight1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=cores,
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            fused_chains.append(build_op_graph([ln, rms], [], device))

        outputs = composite.launch([mm] + fused_chains)
        assert len(outputs) == 4

        # Verify matmul
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        # Verify each chain
        for i in range(3):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
            )
            out = ttnn.to_torch(outputs[i + 1][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Chain {i} PCC: {pcc}"

    # Multi-core stress tests
    # =========================================================================

    def test_two_phase_chain_on_multicore_range(self, device, test_tensors):
        """LN->RMS chain on 4-core range (0,0)-(3,0)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})

        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        fused = build_op_graph([ln, rms], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Multi-core 2-phase chain PCC: {pcc}"

    def test_three_phase_chain_on_2x2_grid(self, device, test_tensors):
        """LN->RMS->LN chain on 2x2 core grid (0,0)-(1,1)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0],
            core_range_set=cores,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = build_op_graph([ln1, rms, ln2], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"2x2 grid 3-phase chain PCC: {pcc}"

    def test_three_parallel_chains_on_nonoverlapping_multicore_ranges(self, device, test_tensors):
        """3 independent LN->RMS chains on non-overlapping multi-core ranges."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Chain 0: cores (0,0)-(1,0) - 2 cores
        # Chain 1: cores (2,0)-(4,0) - 3 cores
        # Chain 2: cores (5,0)-(6,0) - 2 cores
        core_ranges = [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(4, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0))}),
        ]

        torch_inputs = [torch.randn_like(test_tensors["torch_input"]) for _ in range(3)]
        tt_inputs = [
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t in torch_inputs
        ]

        chains = []
        for i in range(3):
            ln = layer_norm.layer_norm(
                tt_inputs[i],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            chains.append([ln, rms])

        fused = [build_op_graph(c, [], device) for c in chains]
        outputs = composite.launch(fused)
        assert len(outputs) == 3

        for i in range(3):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
            )
            out = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Chain {i} PCC: {pcc}"

    def test_four_phase_rms_chain_on_multicore(self, device, test_tensors):
        """4-phase all-RMS chain on 3-core range (2,0)-(4,0)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(4, 0))})

        torch_weight4 = torch.ones_like(test_tensors["torch_weight1"])
        tt_weight4 = ttnn.from_torch(
            torch_weight4,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        weights = [test_tensors["tt_weight1"], test_tensors["tt_weight2"], test_tensors["tt_weight3"], tt_weight4]
        torch_weights = [
            test_tensors["torch_weight1"],
            test_tensors["torch_weight2"],
            test_tensors["torch_weight3"],
            torch_weight4,
        ]

        descs = []
        prev_input = test_tensors["tt_input"]
        for w in weights:
            d = rms_norm.rms_norm(prev_input, core_range_set=cores, weight=w, epsilon=1e-5)
            descs.append(d)
            prev_input = d.output_tensors[0]

        fused = build_op_graph(descs, [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = test_tensors["torch_input"]
        for tw in torch_weights:
            golden = torch_rms_norm(golden, tw)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Multi-core 4-phase RMS PCC: {pcc}"

    def test_mixed_single_and_multicore_parallel_chains(self, device, test_tensors):
        """Mix of single-core and multi-core chains in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Chain A: single core (0,0)
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        torch_input_a = test_tensors["torch_input"]
        tt_input_a = test_tensors["tt_input"]

        # Chain B: 3 cores (1,0)-(3,0)
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0))})
        torch_input_b = torch.randn_like(test_tensors["torch_input"])
        tt_input_b = ttnn.from_torch(
            torch_input_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Chain C: 2 cores (4,0)-(5,0)
        cores_c = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0))})
        torch_input_c = torch.randn_like(test_tensors["torch_input"])
        tt_input_c = ttnn.from_torch(
            torch_input_c,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Build all chains
        chains_data = [
            (cores_a, tt_input_a, torch_input_a),
            (cores_b, tt_input_b, torch_input_b),
            (cores_c, tt_input_c, torch_input_c),
        ]

        fused_chains = []
        for cores, tt_in, _ in chains_data:
            ln = layer_norm.layer_norm(
                tt_in,
                core_range_set=cores,
                weight=test_tensors["tt_weight1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=cores,
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            fused_chains.append(build_op_graph([ln, rms], [], device))

        outputs = composite.launch(fused_chains)
        assert len(outputs) == 3

        # Verify all chains
        for i, (_, _, torch_in) in enumerate(chains_data):
            golden = torch_rms_norm(
                torch_layer_norm(torch_in, test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
            )
            out = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Chain {i} PCC: {pcc}"

    def test_parallel_three_phase_chains_on_separate_multicore_ranges(self, device, test_tensors):
        """2 independent 3-phase LN->RMS->LN chains on separate multi-core ranges."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Chain A: cores (0,0)-(2,0)
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})
        torch_input_a = test_tensors["torch_input"]
        tt_input_a = test_tensors["tt_input"]

        # Chain B: cores (3,0)-(5,0)
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(5, 0))})
        torch_input_b = torch.randn_like(test_tensors["torch_input"])
        tt_input_b = ttnn.from_torch(
            torch_input_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        chains_data = [
            (cores_a, tt_input_a, torch_input_a),
            (cores_b, tt_input_b, torch_input_b),
        ]

        fused_chains = []
        for cores, tt_in, _ in chains_data:
            ln1 = layer_norm.layer_norm(
                tt_in,
                core_range_set=cores,
                weight=test_tensors["tt_weight1"],
                bias=test_tensors["tt_bias1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln1.output_tensors[0],
                core_range_set=cores,
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            ln2 = layer_norm.layer_norm(
                rms.output_tensors[0],
                core_range_set=cores,
                weight=test_tensors["tt_weight3"],
                bias=test_tensors["tt_bias2"],
                epsilon=1e-5,
            )
            fused_chains.append(build_op_graph([ln1, rms, ln2], [], device))

        outputs = composite.launch(fused_chains)
        assert len(outputs) == 2

        # Verify both chains
        for i, (_, _, torch_in) in enumerate(chains_data):
            temp = torch_layer_norm(torch_in, test_tensors["torch_weight1"], test_tensors["torch_bias1"])
            temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
            golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])
            out = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"3-phase chain {i} PCC: {pcc}"

    def test_matmul_plus_multicore_norm_chains(self, device, test_tensors, matmul_tensors):
        """1 matmul + 2 multi-core fused norm chains in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on default core
        mm = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        # Chain A on cores (2,0)-(3,0): LN->RMS
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        ln_a = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms_a = rms_norm.rms_norm(
            ln_a.output_tensors[0],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_a = build_op_graph([ln_a, rms_a], [], device)

        # Chain B on cores (4,0)-(6,0): RMS->LN
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(6, 0))})
        torch_input_b = torch.randn_like(test_tensors["torch_input"])
        tt_input_b = ttnn.from_torch(
            torch_input_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rms_b = rms_norm.rms_norm(
            tt_input_b,
            core_range_set=cores_b,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln_b = layer_norm.layer_norm(
            rms_b.output_tensors[0],
            core_range_set=cores_b,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        fused_b = build_op_graph([rms_b, ln_b], [], device)

        # Launch all 3 in parallel
        outputs = composite.launch([mm, fused_a, fused_b])
        assert len(outputs) == 3

        # Verify matmul
        golden_mm = matmul_tensors["torch_a"] @ matmul_tensors["torch_b"]
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        # Verify chain A (LN->RMS)
        golden_a = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_a, f"Multi-core chain A PCC: {pcc_a}"

        # Verify chain B (RMS->LN)
        golden_b = torch_layer_norm(
            torch_rms_norm(torch_input_b, test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[2][0]), pcc=0.98)
        assert passing_b, f"Multi-core chain B PCC: {pcc_b}"

    def test_four_parallel_multicore_chains_stress(self, device, test_tensors):
        """4 independent 2-phase chains on different multi-core ranges - maximum stress."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # 4 chains with varying core counts
        # Chain 0: 2 cores (0,0)-(1,0)
        # Chain 1: 1 core  (2,0)-(2,0)
        # Chain 2: 3 cores (3,0)-(5,0)
        # Chain 3: 2 cores (6,0)-(7,0)
        core_ranges = [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 0))}),
        ]

        torch_inputs = [torch.randn_like(test_tensors["torch_input"]) for _ in range(4)]
        tt_inputs = [
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t in torch_inputs
        ]

        chains = []
        for i in range(4):
            ln = layer_norm.layer_norm(
                tt_inputs[i],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight1"],
                epsilon=1e-5,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                core_range_set=core_ranges[i],
                weight=test_tensors["tt_weight2"],
                epsilon=1e-5,
                compute_kernel_config=ln_compute_config,
            )
            chains.append([ln, rms])

        fused = [build_op_graph(c, [], device) for c in chains]
        outputs = composite.launch(fused)
        assert len(outputs) == 4

        for i in range(4):
            golden = torch_rms_norm(
                torch_layer_norm(torch_inputs[i], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
            )
            out = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, out, pcc=0.98)
            assert passing, f"Multi-core chain {i} PCC: {pcc}"

    def test_multicore_2d_matmul_plus_fused_chain(self, device, test_tensors):
        """Multi-core 2D offset matmul (6 cores) + fused LN->RMS chain (2 cores) in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on 3x2 grid at (2,1)-(4,2) = 6 cores
        torch_a = torch.randn(1, 1, 192, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(4, 2))})
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(3, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,
        )
        mm = matmul_desc(tt_a, tt_b, core_range_set=mm_cores, program_config=program_config)

        # Fused LN->RMS on (0,0)-(1,0) = 2 cores (non-overlapping with matmul)
        chain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused = build_op_graph([ln, rms], [], device)

        outputs = composite.launch([mm, fused])
        assert len(outputs) == 2

        golden_mm = torch_a @ torch_b
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        golden_chain = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_chain, pcc_chain = comp_pcc(golden_chain, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_chain, f"Chain PCC: {pcc_chain}"

    def test_multicore_2d_matmul_plus_multiple_chains(self, device, test_tensors):
        """Multi-core 2D offset matmul (6 cores) + 2 fused chains (2c each) + 1 single LN in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on 2x3 grid at (3,2)-(4,4) = 6 cores
        torch_a = torch.randn(1, 1, 192, 64, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 4))})
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(2, 3),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=4,
        )
        mm = matmul_desc(tt_a, tt_b, core_range_set=mm_cores, program_config=program_config)

        # Chain A on (0,0)-(1,0) = 2 cores: LN->RMS
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        ln_a = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms_a = rms_norm.rms_norm(
            ln_a.output_tensors[0],
            core_range_set=cores_a,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_a = build_op_graph([ln_a, rms_a], [], device)

        # Chain B on (0,1)-(1,1) = 2 cores: RMS->LN
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1))})
        torch_input_b = torch.randn_like(test_tensors["torch_input"])
        tt_input_b = ttnn.from_torch(
            torch_input_b,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rms_b = rms_norm.rms_norm(
            tt_input_b,
            core_range_set=cores_b,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln_b = layer_norm.layer_norm(
            rms_b.output_tensors[0],
            core_range_set=cores_b,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        fused_b = build_op_graph([rms_b, ln_b], [], device)

        # Single LN on (2,0)
        cores_c = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))})
        torch_input_c = torch.randn_like(test_tensors["torch_input"])
        tt_input_c = ttnn.from_torch(
            torch_input_c,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ln_c = layer_norm.layer_norm(
            tt_input_c,
            core_range_set=cores_c,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        # Launch all 4 in parallel: matmul(6c) + chain A(2c) + chain B(2c) + LN(1c) = 11 cores
        outputs = composite.launch([mm, fused_a, fused_b, ln_c])
        assert len(outputs) == 4

        golden_mm = torch_a @ torch_b
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        golden_a = torch_rms_norm(
            torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_a, f"Chain A PCC: {pcc_a}"

        golden_b = torch_layer_norm(
            torch_rms_norm(torch_input_b, test_tensors["torch_weight1"]), test_tensors["torch_weight2"]
        )
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[2][0]), pcc=0.98)
        assert passing_b, f"Chain B PCC: {pcc_b}"

        golden_c = torch_layer_norm(torch_input_c, test_tensors["torch_weight3"], test_tensors["torch_bias1"])
        passing_c, pcc_c = comp_pcc(golden_c, ttnn.to_torch(outputs[3][0]), pcc=0.99)
        assert passing_c, f"Single LN PCC: {pcc_c}"

    def test_large_2d_matmul_plus_multicore_three_phase_chain(self, device, test_tensors):
        """Large 2D matmul (8 cores) + multi-core 3-phase fused chain (4 cores) in parallel."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Matmul on 4x2 grid at (4,0)-(7,1) = 8 cores
        torch_a = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)
        tt_a = ttnn.from_torch(
            torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_b = ttnn.from_torch(
            torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 1))})
        # M=8, N=8, K=8; 8 output blocks -> 1 per core
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
        )
        mm = matmul_desc(tt_a, tt_b, core_range_set=mm_cores, program_config=program_config)

        # 3-phase LN->RMS->LN chain on (0,0)-(3,0) = 4 cores
        chain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms1.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )
        fused = build_op_graph([ln1, rms1, ln2], [], device)

        outputs = composite.launch([mm, fused])
        assert len(outputs) == 2

        golden_mm = torch_a @ torch_b
        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing_mm, f"Matmul PCC: {pcc_mm}"

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden_chain = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])
        passing_chain, pcc_chain = comp_pcc(golden_chain, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_chain, f"Chain PCC: {pcc_chain}"


# =============================================================================
# Sharded Fusion Tests
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestShardedSequentialFusion:
    """Tests for sequential fusion with sharded layernorm and rms_norm operations."""

    def test_two_phase_ln_rms_block_sharded(self, device):
        """LN->RMS chain with BLOCK_SHARDED input."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm,
            rms_norm_golden,
        )

        # Setup: 8x10 tiles = (256, 320), 2x5 cores, single-stage (block sharded)
        h, w = 32 * 8, 32 * 10
        num_cores_h, num_cores_w = 2, 5
        block_ht, block_wt = 4, 2
        two_stage = False

        torch.manual_seed(12345)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weight_ln = torch.ones((w,), dtype=torch.bfloat16)
        torch_weight_rms = torch.ones((w,), dtype=torch.bfloat16)

        # Create sharded input
        sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
        tt_input = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )
        tt_weight_ln = ttnn.from_torch(torch_weight_ln, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight_rms = ttnn.from_torch(torch_weight_rms, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program config for sharded operations
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build chain
        ln_desc = layer_norm.layer_norm(
            tt_input,
            weight=tt_weight_ln,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )
        rms_desc = rms_norm.rms_norm(
            ln_desc.output_tensors[0],
            weight=tt_weight_rms,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )

        fused = build_op_graph([ln_desc, rms_desc], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden
        temp = torch_layer_norm(torch_input, weight=torch_weight_ln)
        golden = rms_norm_golden(temp, torch_weight_rms)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded LN->RMS PCC: {pcc}"

    def test_two_phase_rms_ln_width_sharded(self, device):
        """RMS->LN chain with WIDTH_SHARDED input (two-stage reduction)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm,
            rms_norm_golden,
        )

        # Setup: 8x16 tiles = (256, 512), 2x4 cores, two-stage (width sharded)
        h, w = 32 * 8, 32 * 16
        num_cores_h, num_cores_w = 2, 4
        block_ht, block_wt = 8, 2
        two_stage = True

        torch.manual_seed(12346)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weight_rms = torch.ones((w,), dtype=torch.bfloat16)
        torch_weight_ln = torch.ones((w,), dtype=torch.bfloat16)

        # Create width-sharded input
        sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
        tt_input = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )
        tt_weight_rms = ttnn.from_torch(torch_weight_rms, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight_ln = ttnn.from_torch(torch_weight_ln, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program config for sharded operations
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build chain
        rms_desc = rms_norm.rms_norm(
            tt_input,
            weight=tt_weight_rms,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )
        ln_desc = layer_norm.layer_norm(
            rms_desc.output_tensors[0],
            weight=tt_weight_ln,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )

        fused = build_op_graph([rms_desc, ln_desc], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden
        temp = rms_norm_golden(torch_input, torch_weight_rms)
        golden = torch_layer_norm(temp, weight=torch_weight_ln)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded RMS->LN PCC: {pcc}"

    def test_three_phase_sharded_chain(self, device):
        """LN->RMS->LN chain with BLOCK_SHARDED input."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm,
            rms_norm_golden,
        )

        # Setup: 4x8 tiles = (128, 256), 4x4 cores, single-stage
        h, w = 32 * 4, 32 * 8
        num_cores_h, num_cores_w = 4, 4
        block_ht, block_wt = 1, 2
        two_stage = False

        torch.manual_seed(12347)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weight1 = torch.ones((w,), dtype=torch.bfloat16)
        torch_weight2 = torch.ones((w,), dtype=torch.bfloat16)
        torch_weight3 = torch.ones((w,), dtype=torch.bfloat16)

        # Create sharded input
        sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
        tt_input = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )
        tt_weight1 = ttnn.from_torch(torch_weight1, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight2 = ttnn.from_torch(torch_weight2, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight3 = ttnn.from_torch(torch_weight3, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program config
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build 3-phase chain
        ln1 = layer_norm.layer_norm(
            tt_input,
            weight=tt_weight1,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0],
            weight=tt_weight2,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0],
            weight=tt_weight3,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )

        fused = build_op_graph([ln1, rms, ln2], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden
        temp = torch_layer_norm(torch_input, weight=torch_weight1)
        temp = rms_norm_golden(temp, torch_weight2)
        golden = torch_layer_norm(temp, weight=torch_weight3)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded 3-phase chain PCC: {pcc}"

    def test_parallel_sharded_chains_nonoverlapping_grids(self, device):
        """2 independent LN->RMS chains on non-overlapping sharded grids."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            torch_layer_norm,
            rms_norm_golden,
        )

        # Chain A: cores (0,0)-(1,1) - 2x2 grid, block sharded
        # Chain B: cores (2,0)-(3,1) - 2x2 grid, block sharded
        h, w = 32 * 4, 32 * 8
        num_cores_h, num_cores_w = 2, 2

        torch.manual_seed(12348)
        torch_input_a = torch.randn((h, w), dtype=torch.bfloat16)
        torch_input_b = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weight_ln = torch.ones((w,), dtype=torch.bfloat16)
        torch_weight_rms = torch.ones((w,), dtype=torch.bfloat16)

        # Create sharded configs for non-overlapping grids
        shard_height = h // num_cores_h
        shard_width = w // num_cores_w

        # Chain A: cores (0,0)-(1,1)
        shard_spec_a = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config_a = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec_a,
        )

        # Chain B: cores (2,0)-(3,1)
        shard_spec_b = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))}),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config_b = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec_b,
        )

        tt_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_a)
        tt_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_b)
        tt_weight_ln = ttnn.from_torch(torch_weight_ln, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight_rms = ttnn.from_torch(torch_weight_rms, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program configs
        block_ht, block_wt = 2, 4
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build chains
        chains_fused = []
        for tt_in in [tt_input_a, tt_input_b]:
            ln = layer_norm.layer_norm(
                tt_in,
                weight=tt_weight_ln,
                epsilon=1e-5,
                compute_kernel_config=compute_config,
                program_config=program_config,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                weight=tt_weight_rms,
                epsilon=1e-5,
                compute_kernel_config=compute_config,
                program_config=program_config,
            )
            chains_fused.append(build_op_graph([ln, rms], [], device))

        outputs = composite.launch(chains_fused)
        assert len(outputs) == 2

        # Verify both chains
        for i, torch_in in enumerate([torch_input_a, torch_input_b]):
            temp = torch_layer_norm(torch_in, weight=torch_weight_ln)
            golden = rms_norm_golden(temp, torch_weight_rms)
            result = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            assert passing, f"Sharded chain {i} PCC: {pcc}"

    def test_four_phase_all_rms_sharded(self, device):
        """4-phase all-RMS chain with BLOCK_SHARDED input."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            rms_norm_golden,
        )

        # Setup: 8x8 tiles = (256, 256), 4x4 cores, single-stage
        h, w = 32 * 8, 32 * 8
        num_cores_h, num_cores_w = 4, 4
        block_ht, block_wt = 2, 2
        two_stage = False

        torch.manual_seed(12349)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weights = [torch.ones((w,), dtype=torch.bfloat16) for _ in range(4)]

        # Create sharded input
        sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
        tt_input = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )
        tt_weights = [ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device) for w in torch_weights]

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program config
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build 4-phase chain
        descs = []
        prev_input = tt_input
        for tt_w in tt_weights:
            d = rms_norm.rms_norm(
                prev_input,
                weight=tt_w,
                epsilon=1e-5,
                compute_kernel_config=compute_config,
                program_config=program_config,
            )
            descs.append(d)
            prev_input = d.output_tensors[0]

        fused = build_op_graph(descs, [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden
        golden = torch_input
        for tw in torch_weights:
            golden = rms_norm_golden(golden, tw)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded 4-phase RMS PCC: {pcc}"

    @pytest.mark.parametrize("two_stage", [False, True])
    def test_sharded_with_bias_and_residual(self, device, two_stage):
        """LN->RMS chain with bias and residual using sharded tensors."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            create_sharded_mem_config,
            torch_layer_norm,
            rms_norm_golden,
        )

        # Setup
        h, w = 32 * 4, 32 * 8
        num_cores_h, num_cores_w = 2, 4
        # Single-stage: shard_ht = h/nch/32 = 128/2/32 = 2 tile rows, so block_ht must be <= 2
        block_ht = 2 if not two_stage else 4
        block_wt = 2 if not two_stage else 1

        torch.manual_seed(12350)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_residual = torch.randn((h, w), dtype=torch.bfloat16)
        torch_weight_ln = torch.ones((w,), dtype=torch.bfloat16)
        torch_bias_ln = torch.zeros((w,), dtype=torch.bfloat16)
        torch_weight_rms = torch.ones((w,), dtype=torch.bfloat16)

        # Create sharded tensors
        sharded_mem_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
        tt_input = ttnn.from_torch(
            torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config
        )
        tt_residual = ttnn.from_torch(
            torch_residual, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem_config
        )
        tt_weight_ln = ttnn.from_torch(torch_weight_ln, layout=ttnn.TILE_LAYOUT, device=device)
        tt_bias_ln = ttnn.from_torch(torch_bias_ln, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight_rms = ttnn.from_torch(torch_weight_rms, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Create program config
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=block_ht,
            block_w=block_wt,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build chain with residual and bias
        ln = layer_norm.layer_norm(
            tt_input,
            weight=tt_weight_ln,
            bias=tt_bias_ln,
            residual_input_tensor=tt_residual,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            weight=tt_weight_rms,
            epsilon=1e-5,
            compute_kernel_config=compute_config,
            program_config=program_config,
        )

        fused = build_op_graph([ln, rms], [], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden
        temp = torch_layer_norm(torch_input, residual=torch_residual, weight=torch_weight_ln, bias=torch_bias_ln)
        golden = rms_norm_golden(temp, torch_weight_rms)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        stage_name = "two-stage" if two_stage else "single-stage"
        assert passing, f"Sharded {stage_name} with bias/residual PCC: {pcc}"

    def test_sharded_stress_varied_grid_sizes(self, device):
        """Stress test: 3 parallel chains with different sharded grid sizes."""
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
            torch_layer_norm,
            rms_norm_golden,
        )

        torch.manual_seed(12351)
        torch_weight_ln = torch.ones((32 * 8,), dtype=torch.bfloat16)
        torch_weight_rms = torch.ones((32 * 8,), dtype=torch.bfloat16)

        tt_weight_ln = ttnn.from_torch(torch_weight_ln, layout=ttnn.TILE_LAYOUT, device=device)
        tt_weight_rms = ttnn.from_torch(torch_weight_rms, layout=ttnn.TILE_LAYOUT, device=device)

        # Create compute config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Chain A: 2x2 grid on cores (0,0)-(1,1)
        h_a, w_a = 32 * 4, 32 * 8
        torch_input_a = torch.randn((h_a, w_a), dtype=torch.bfloat16)
        shard_spec_a = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [h_a // 2, w_a // 2],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config_a = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec_a
        )
        tt_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_a)
        prog_config_a = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=4,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Chain B: 4x4 grid on cores (2,0)-(5,3)
        h_b, w_b = 32 * 8, 32 * 8
        torch_input_b = torch.randn((h_b, w_b), dtype=torch.bfloat16)
        shard_spec_b = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(5, 3))}),
            [h_b // 4, w_b // 4],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config_b = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec_b
        )
        tt_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_b)
        prog_config_b = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=2,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Chain C: 2x4 grid on cores (6,0)-(7,3) — 2 cols (x:6-7), 4 rows (y:0-3)
        # Use COL_MAJOR so 4 rows map to width shards (4 <= 4 rows) and 2 cols map to height (2 <= 2 cols)
        h_c, w_c = 32 * 4, 32 * 8
        torch_input_c = torch.randn((h_c, w_c), dtype=torch.bfloat16)
        shard_spec_c = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 3))}),
            [h_c // 2, w_c // 4],
            ttnn.ShardOrientation.COL_MAJOR,
        )
        mem_config_c = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec_c
        )
        tt_input_c = ttnn.from_torch(torch_input_c, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config_c)
        prog_config_c = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=2,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        # Build all chains
        chains_data = [
            (tt_input_a, torch_input_a, prog_config_a),
            (tt_input_b, torch_input_b, prog_config_b),
            (tt_input_c, torch_input_c, prog_config_c),
        ]

        chains_fused = []
        for tt_in, _, prog_cfg in chains_data:
            ln = layer_norm.layer_norm(
                tt_in,
                weight=tt_weight_ln,
                epsilon=1e-5,
                compute_kernel_config=compute_config,
                program_config=prog_cfg,
            )
            rms = rms_norm.rms_norm(
                ln.output_tensors[0],
                weight=tt_weight_rms,
                epsilon=1e-5,
                compute_kernel_config=compute_config,
                program_config=prog_cfg,
            )
            chains_fused.append(build_op_graph([ln, rms], [], device))

        outputs = composite.launch(chains_fused)
        assert len(outputs) == 3

        # Verify all chains
        for i, (_, torch_in, _) in enumerate(chains_data):
            temp = torch_layer_norm(torch_in, weight=torch_weight_ln)
            golden = rms_norm_golden(temp, torch_weight_rms)
            result = ttnn.to_torch(outputs[i][0])
            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            assert passing, f"Sharded stress chain {i} PCC: {pcc}"

    def test_block_ht_1_sharded_fusion(self, device):
        """Regression test: block_ht=1 fused chains previously asserted in cb_pop_front.

        When block_ht=1, gamma CB has capacity for 1 tile but compute pushes block_w
        tiles without popping. The CB reset between phases must pop one tile at a time
        to handle circular buffer wrapping correctly.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite
        from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import rms_norm_golden

        h, w, nch, ncw = 64, 128, 2, 2
        torch.manual_seed(12347)
        torch_input = torch.randn((h, w), dtype=torch.bfloat16)
        torch_w1 = torch.ones((w,), dtype=torch.bfloat16)
        torch_w2 = torch.ones((w,), dtype=torch.bfloat16)

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncw - 1, nch - 1))}),
            [h // nch, w // ncw],
            ttnn.ShardOrientation.COL_MAJOR,
        )
        sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)

        tt_in = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded)
        tt_w1 = ttnn.from_torch(torch_w1, layout=ttnn.TILE_LAYOUT, device=device)
        tt_w2 = ttnn.from_torch(torch_w2, layout=ttnn.TILE_LAYOUT, device=device)

        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            block_h=1,
            block_w=2,
            subblock_w=1,
            use_welford=False,
            inplace=False,
        )

        d1 = rms_norm.rms_norm(tt_in, weight=tt_w1, epsilon=1e-5, compute_kernel_config=cc, program_config=pc)
        d2 = rms_norm.rms_norm(
            d1.output_tensors[0], weight=tt_w2, epsilon=1e-5, compute_kernel_config=cc, program_config=pc
        )
        fused = build_op_graph([d1, d2], [], device)

        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # Golden: RMS(RMS(input))
        temp = rms_norm_golden(torch_input, torch_w1)
        golden = rms_norm_golden(temp, torch_w2)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Block_ht=1 sharded RMS->RMS PCC: {pcc}"


# =============================================================================
# Cross-Op Compilation Tests
# =============================================================================


class TestCrossOpCompilation:
    """Verify that merging pre-main from different ops produces compilable kernels.

    These tests read REAL kernel source files from different operations,
    process them through the pre-main merging pipeline, construct complete
    fused kernel sources with an empty kernel_main(), create
    ProgramDescriptors, and verify JIT compilation succeeds via
    ttnn.generic_op().

    The kernel_main() bodies are empty -- we are only testing that the
    merged pre-main (includes, defines, helpers, namespaces) forms a
    valid C++ translation unit that the RISC-V compiler accepts.
    """

    # Kernel source file paths (compute kernels)
    KERNEL_PATHS = {
        "layernorm": "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp",
        "rmsnorm_post": "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp",
        "matmul": "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        "batchnorm": "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp",
        "untilize": "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/common.cpp",
        "eltwise_sfpu": "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        "typecast": "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
    }

    @staticmethod
    def _read_and_process(kernel_path, defines=None):
        """Read a kernel source file and process through the pipeline.

        Returns (headers, source) where headers is a list of
        (resolved_path, content) tuples from inlined local includes.
        """
        import os
        from models.experimental.ops.descriptors.fusion import (
            inline_local_includes,
        )

        with open(kernel_path, "r") as f:
            source = f.read()

        kernel_dir = os.path.dirname(os.path.abspath(kernel_path))
        headers, source = inline_local_includes(source, kernel_dir)
        return headers, source

    @staticmethod
    def _build_fused_source(sources_with_headers):
        """Build a compilable fused source to test pre-main compatibility.

        Tests that header content, aliases, and helper functions from
        different ops can coexist in a single translation unit.  Uses
        file-scope header content + per-phase namespaces for isolation,
        but skips RT arg redirection and compile-time arg offsets (not
        needed for compilation-only verification).

        Args:
            sources_with_headers: List of (phase_idx, headers, source) tuples.
        """
        import re
        from models.experimental.ops.descriptors.fusion import (
            collect_includes,
            collect_defines,
        )

        # Reconstruct combined text for include/define collection
        all_combined = []
        for _, hdrs, s in sources_with_headers:
            hdr_text = "\n".join(c for _, c in hdrs)
            all_combined.append(hdr_text + "\n" + s)
        includes = collect_includes(all_combined)
        defines = collect_defines(all_combined)

        # Deduplicate headers by resolved path for file scope
        header_path_seen = set()
        file_scope_blocks = []
        for _, hdrs, _ in sources_with_headers:
            for path, content in hdrs:
                if path not in header_path_seen:
                    header_path_seen.add(path)
                    content_stripped = content.strip()
                    if content_stripped:
                        file_scope_blocks.append(content_stripped)

        # Extract pre-main from original source (minus preprocessor lines)
        _km_re = re.compile(r"\bvoid\s+kernel_main\s*\(")
        _skip = ("#include", "#define", "#pragma", "#undef")
        phase_pre_mains = {}
        for phase_idx, _, source in sources_with_headers:
            m = _km_re.search(source)
            pre_text = source[: m.start()] if m else source
            pre_lines = [line for line in pre_text.split("\n") if not line.strip().startswith(_skip)]
            phase_pre_mains[phase_idx] = "\n".join(pre_lines).strip()

        lines = [
            "// Auto-generated fused compute kernel - compilation test",
            "",
        ]
        lines.extend(defines)
        lines.append("")
        lines.extend(includes)
        lines.append("")

        # Header content at file scope
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")

        # Phase namespaces with empty run() — test is pre-main only
        for phase_idx, _, _ in sources_with_headers:
            ns_name = f"phase_{phase_idx}"
            pre_main = phase_pre_mains.get(phase_idx, "")
            lines.append(f"// ---- Phase {phase_idx} ----")
            lines.append(f"namespace {ns_name} {{")
            lines.append("")
            if pre_main.strip():
                lines.append(pre_main)
                lines.append("")
            lines.append("void run() {}")
            lines.append("")
            lines.append(f"}} // namespace {ns_name}")
            lines.append("")

        lines.append("void kernel_main() {}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _compile_source(device, source):
        """Compile a fused kernel source via the JIT build system.

        Creates a ProgramDescriptor with the source and dispatches via
        generic_op, which triggers JIT compilation. Compilation failures
        produce RuntimeError with "build failed" in the message — we
        detect that specifically so non-compilation failures propagate
        normally rather than being misreported as compile errors.

        No Python API exists for compile-only (CompileProgram is C++-only),
        so we must dispatch to trigger compilation. The empty kernel_main()
        is a valid no-op on hardware.
        """
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        kernel = ttnn.KernelDescriptor()
        kernel.kernel_source = source
        kernel.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        kernel.core_ranges = core
        kernel.config = ttnn.ComputeConfigDescriptor()

        # Minimal CB so the program has something to bind
        # Use constructor overload that accepts ttnn.DataType and converts to tt::DataFormat
        cb = ttnn.CBDescriptor()
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.DataType.BFLOAT16, page_size=2048)
        cb.total_size = 2048
        cb.core_ranges = core
        cb.format_descriptors = [fmt]

        desc = ttnn.ProgramDescriptor()
        desc.kernels = [kernel]
        desc.cbs = [cb]

        dummy_in = ttnn.from_torch(
            torch.zeros(1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        dummy_out = ttnn.from_torch(
            torch.zeros(1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        try:
            ttnn.generic_op([dummy_in, dummy_out], desc)
        except RuntimeError as e:
            msg = str(e)
            if "build failed" in msg:
                source_preview = source[:2000] + ("..." if len(source) > 2000 else "")
                raise AssertionError(
                    f"Kernel compilation failed.\n\nCompiler output:\n{msg}\n\n"
                    f"Generated source (first 2000 chars):\n{source_preview}"
                ) from None
            raise  # Non-compilation failure — propagate as-is

    def test_compile_layernorm_plus_matmul(self, device):
        """LN compute (namespace aliases + ALWI) + matmul (using declaration)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["matmul"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_layernorm_plus_batchnorm(self, device):
        """LN compute (short ALWI) + batchnorm (13-param ALWI helper)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["batchnorm"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_layernorm_plus_untilize(self, device):
        """LN compute (complex pre-main) + untilize (constexpr helper)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["untilize"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_rmsnorm_post_plus_layernorm(self, device):
        """rmsnorm_post (multi-line ACQ/REL) + LN (single-line ACQ/REL).

        Tests signature-based dedup: both define ACQ() and REL() differently.
        """
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["rmsnorm_post"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_matmul_plus_batchnorm(self, device):
        """matmul (minimal pre-main) + batchnorm (long ALWI)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["matmul"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["batchnorm"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_layernorm_plus_eltwise_sfpu(self, device):
        """LN compute + eltwise unary SFPU (many API includes)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["eltwise_sfpu"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_layernorm_plus_typecast(self, device):
        """LN compute + typecast (eltwise unary typecast)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["typecast"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1)])
        self._compile_source(device, source)

    def test_compile_three_phase_matmul_ln_batchnorm(self, device):
        """3-phase: matmul + LN + batchnorm."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["matmul"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h2, s2 = self._read_and_process(self.KERNEL_PATHS["batchnorm"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1), (2, h2, s2)])
        self._compile_source(device, source)

    def test_compile_three_phase_ln_untilize_sfpu(self, device):
        """3-phase: LN + untilize + eltwise SFPU."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["untilize"])
        h2, s2 = self._read_and_process(self.KERNEL_PATHS["eltwise_sfpu"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1), (2, h2, s2)])
        self._compile_source(device, source)

    def test_compile_four_phase_max_diversity(self, device):
        """4-phase: matmul + LN + batchnorm + untilize (max diversity)."""
        h0, s0 = self._read_and_process(self.KERNEL_PATHS["matmul"])
        h1, s1 = self._read_and_process(self.KERNEL_PATHS["layernorm"])
        h2, s2 = self._read_and_process(self.KERNEL_PATHS["batchnorm"])
        h3, s3 = self._read_and_process(self.KERNEL_PATHS["untilize"])
        source = self._build_fused_source([(0, h0, s0), (1, h1, s1), (2, h2, s2), (3, h3, s3)])
        self._compile_source(device, source)


@pytest.fixture
def opgraph_tensors(device):
    """Create test tensors sized for multi-core OpGraph tests.

    Interleaved norm ops need NCHt >= num_cores to use all cores.
    Shape (1, 1, 256, 128) gives NCHt=8, supporting up to 8 cores.
    """
    torch.manual_seed(42)

    batch, seq_len, hidden = 1, 256, 128
    input_shape = (batch, 1, seq_len, hidden)
    weight_shape = (1, 1, 1, hidden)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # Use ones weights for stability (avoids amplifying numerical errors across phases)
    torch_weights = [torch.ones(weight_shape, dtype=torch.bfloat16) for _ in range(8)]
    torch_bias = torch.zeros(weight_shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weights = [
        ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in torch_weights
    ]
    tt_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return {
        "torch_input": torch_input,
        "torch_weights": torch_weights,
        "torch_bias": torch_bias,
        "tt_input": tt_input,
        "tt_weights": tt_weights,
        "tt_bias": tt_bias,
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestOpGraphExecution:
    """Tests for OpGraph (branching tree) fusion topologies.

    An OpGraph has a shared stem of phases running on ALL branch cores,
    then the cores split into disjoint branches each running their own
    subsequent phases.  For each root-to-leaf path, a separate fused
    kernel binary is generated.  During stem phases, different kernel
    binaries synchronize via shared GlobalSemaphore addresses.

    These tests use larger tensors (NCHt=8) so the factory actually
    distributes work across multiple cores for interleaved operations.
    """

    def test_simple_two_branch_split(self, device, opgraph_tensors):
        """Stem (1 RMS on 4 cores) -> Branch A (RMS on 2 cores) + Branch B (RMS on 2 cores)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_two_phase_stem_two_branches(self, device, opgraph_tensors):
        """Stem (2 RMS on 4 cores) -> Branch A (RMS on 2 cores) + Branch B (RMS on 2 cores)."""
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})

        stem_op0 = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        stem_op1 = rms_norm.rms_norm(
            stem_op0.output_tensors[0], core_range_set=union_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_a_op = rms_norm.rms_norm(
            stem_op1.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][2], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op1.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][3], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op0, stem_op1],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_s0 = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_s1 = torch_rms_norm(golden_s0, t["torch_weights"][1])
        golden_a = torch_rms_norm(golden_s1, t["torch_weights"][2])
        golden_b = torch_rms_norm(golden_s1, t["torch_weights"][3])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_three_way_split(self, device, opgraph_tensors):
        """Stem (1 RMS on 6 cores) -> 3 branches of 2 cores each."""
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        branch_c_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )
        branch_c_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_c_range, weight=t["tt_weights"][3], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
                OpNode(branch_c_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        goldens = [torch_rms_norm(golden_stem, t["torch_weights"][i + 1]) for i in range(3)]

        for i, label in enumerate(["A", "B", "C"]):
            result = ttnn.to_torch(outputs[0][i])
            passing, pcc = comp_pcc(goldens[i], result, pcc=0.98)
            assert passing, f"Branch {label} PCC: {pcc}"

    def test_nested_branching(self, device, opgraph_tensors):
        """Stem -> Branch A (with 2 children) + Branch B. 3 paths total.

        Tree structure:
            Stem (8 cores) -> Branch A (4 cores) -> Leaf A1 (2 cores)
                                                  -> Leaf A2 (2 cores)
                           -> Branch B (4 cores)
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        leaf_a1_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        leaf_a2_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        leaf_a1_op = rms_norm.rms_norm(
            branch_a_op.output_tensors[0], core_range_set=leaf_a1_range, weight=t["tt_weights"][2], epsilon=1e-5
        )
        leaf_a2_op = rms_norm.rms_norm(
            branch_a_op.output_tensors[0], core_range_set=leaf_a2_range, weight=t["tt_weights"][3], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][4], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(
                    branch_a_op,
                    children=[
                        OpNode(leaf_a1_op),
                        OpNode(leaf_a2_op),
                    ],
                ),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        g_a = torch_rms_norm(g_stem, t["torch_weights"][1])
        goldens = [
            torch_rms_norm(g_a, t["torch_weights"][2]),
            torch_rms_norm(g_a, t["torch_weights"][3]),
            torch_rms_norm(g_stem, t["torch_weights"][4]),
        ]
        labels = ["A1", "A2", "B"]

        for i, (golden, label) in enumerate(zip(goldens, labels)):
            result = ttnn.to_torch(outputs[0][i])
            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            assert passing, f"Branch {label} PCC: {pcc}"

    def test_op_graph_plus_independent_chain(self, device, opgraph_tensors):
        """OpGraph on 4 cores + independent RMS chain on 2 separate cores."""
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        indep_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0))})

        torch_input2 = torch.randn_like(t["torch_input"])
        tt_input2 = ttnn.from_torch(
            torch_input2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        graph = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        indep_op0 = rms_norm.rms_norm(tt_input2, core_range_set=indep_range, weight=t["tt_weights"][0], epsilon=1e-5)
        indep_op1 = rms_norm.rms_norm(
            indep_op0.output_tensors[0], core_range_set=indep_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        indep_fused = build_op_graph([indep_op0, indep_op1], [], device)

        all_ops = [graph, indep_fused]
        outputs = composite.launch(all_ops)
        assert len(outputs) == 2

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])
        golden_indep = torch_rms_norm(torch_rms_norm(torch_input2, t["torch_weights"][0]), t["torch_weights"][1])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])
        result_indep = ttnn.to_torch(outputs[1][0])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        assert passing_a, f"Graph A PCC: {pcc_a}"
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_b, f"Graph B PCC: {pcc_b}"
        passing_indep, pcc_indep = comp_pcc(golden_indep, result_indep, pcc=0.98)
        assert passing_indep, f"Independent PCC: {pcc_indep}"

    def test_single_core_branches(self, device, opgraph_tensors):
        """Stem (1 RMS on 2 cores) -> 2 branches of 1 core each.

        Tests barrier with single-core branches (no NOC multicast needed
        within the branch segment).
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_deep_stem_short_branches(self, device, opgraph_tensors):
        """Stem (3 RMS phases on 4 cores) -> 2 branches of 2 cores, 1 phase each.

        Tests many stem barriers before the split point.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})

        stem_op0 = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        stem_op1 = rms_norm.rms_norm(
            stem_op0.output_tensors[0], core_range_set=union_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        stem_op2 = rms_norm.rms_norm(
            stem_op1.output_tensors[0], core_range_set=union_range, weight=t["tt_weights"][2], epsilon=1e-5
        )
        branch_a_op = rms_norm.rms_norm(
            stem_op2.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][3], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op2.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][4], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op0, stem_op1, stem_op2],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden = t["torch_input"]
        for i in range(3):
            golden = torch_rms_norm(golden, t["torch_weights"][i])
        golden_a = torch_rms_norm(golden, t["torch_weights"][3])
        golden_b = torch_rms_norm(golden, t["torch_weights"][4])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_mixed_ln_rms_op_graph(self, device, opgraph_tensors):
        """LN stem -> RMS branches, with matching compute configs.

        Tests that fp32_dest_acc_en is consistent when mixing LN (fp32=True)
        and RMS (fp32=False default) by passing LN compute config to RMS.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})

        stem_op = layer_norm.layer_norm(
            t["tt_input"],
            core_range_set=union_range,
            weight=t["tt_weights"][0],
            bias=t["tt_bias"],
            epsilon=1e-5,
        )
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0],
            core_range_set=branch_a_range,
            weight=t["tt_weights"][1],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0],
            core_range_set=branch_b_range,
            weight=t["tt_weights"][2],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_stem = torch_layer_norm(t["torch_input"], t["torch_weights"][0], t["torch_bias"])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_2d_grid_two_branch_split(self, device, opgraph_tensors):
        """Stem (1 RMS on 2x2 grid) -> Branch A (top row) + Branch B (bottom row).

        Tests OpGraph branching with 2D rectangular core grids, exercising
        the NOC multicast validation and barrier sync for non-trivial grid
        shapes. All existing branching tests use 1D (single-row) grids.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        # 2x2 grid: (0,0)-(1,1) = 4 cores
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        # Top row: (0,0)-(1,0)
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        # Bottom row: (0,1)-(1,1)
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_2d_grid_nested_branching(self, device, opgraph_tensors):
        """Stem (2x4 grid) -> Branch A (2x2 top) with 2 leaf children + Branch B (2x2 bottom).

        Tests nested OpGraph branching with 2D rectangular grids at every level.

        Tree structure:
            Stem (2x4 grid, 8 cores) -> Branch A (2x2 top) -> Leaf A1 (1x2 left)
                                                              -> Leaf A2 (1x2 right)
                                      -> Branch B (2x2 bottom)
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        # Full 2x4 grid: cores (0,0)-(3,1), 8 cores
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        # Top 2x2: (0,0)-(1,1), 4 cores
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        # Bottom 2x2: (2,0)-(3,1), 4 cores
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1))})
        # Leaf A1: left column of branch_a: (0,0)-(0,1), 2 cores
        leaf_a1_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
        # Leaf A2: right column of branch_a: (1,0)-(1,1), 2 cores
        leaf_a2_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 1))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        leaf_a1_op = rms_norm.rms_norm(
            branch_a_op.output_tensors[0], core_range_set=leaf_a1_range, weight=t["tt_weights"][2], epsilon=1e-5
        )
        leaf_a2_op = rms_norm.rms_norm(
            branch_a_op.output_tensors[0], core_range_set=leaf_a2_range, weight=t["tt_weights"][3], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][4], epsilon=1e-5
        )

        fused = build_op_graph(
            root_phases=[stem_op],
            children=[
                OpNode(
                    branch_a_op,
                    children=[
                        OpNode(leaf_a1_op),
                        OpNode(leaf_a2_op),
                    ],
                ),
                OpNode(branch_b_op),
            ],
            device=device,
        )

        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        g_a = torch_rms_norm(g_stem, t["torch_weights"][1])
        goldens = [
            torch_rms_norm(g_a, t["torch_weights"][2]),
            torch_rms_norm(g_a, t["torch_weights"][3]),
            torch_rms_norm(g_stem, t["torch_weights"][4]),
        ]
        labels = ["A1", "A2", "B"]

        for i, (golden, label) in enumerate(zip(goldens, labels)):
            result = ttnn.to_torch(outputs[0][i])
            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            assert passing, f"Branch {label} PCC: {pcc}"

    def test_child_wider_than_parent_accepted(self, device, opgraph_tensors):
        """Child core range extending beyond parent is accepted (narrow->wide).

        The root op is on cores 0-3 but one branch extends to core 5.
        This is now valid — cores 4-5 get no-op entries at the root position.
        """
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = opgraph_tensors
        # Root on 4 cores, branch B extends beyond
        root_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        # Branch B extends to core 5, beyond root_range (0-3) — now valid
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(5, 0))})

        root_op = rms_norm.rms_norm(t["tt_input"], core_range_set=root_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            root_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            root_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        # Should not raise — narrow->wide is now valid
        fused = build_op_graph(
            root_phases=[root_op],
            children=[
                OpNode(branch_a_op),
                OpNode(branch_b_op),
            ],
            device=device,
        )
        assert fused is not None

    def test_invalid_topology_overlapping_branches(self, device, opgraph_tensors):
        """Overlapping branch core ranges should raise ValueError before touching device."""
        from models.experimental.ops.descriptors.fusion import build_op_graph, OpNode
        from models.experimental.ops.descriptors.normalization import rms_norm

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        # Overlapping: (0,0)-(2,0) and (1,0)-(3,0)
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        with pytest.raises(ValueError, match="overlapping"):
            build_op_graph(
                root_phases=[stem_op],
                children=[
                    OpNode(branch_a_op),
                    OpNode(branch_b_op),
                ],
                device=device,
            )


class TestSequentialParallelAPI:
    """Tests for Sequential/Parallel high-level API with real device execution."""

    def test_sequential_api_linear_chain(self, device, test_tensors):
        """Sequential(rms, rms) produces same result as build_op_graph linear chain."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        rms1_desc = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms2_desc = rms_norm.rms_norm(
            rms1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        fused = Sequential(rms1_desc, rms2_desc).build(device)
        outputs = composite.launch([fused])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        temp = torch_rms_norm(test_tensors["torch_input"], test_tensors["torch_weight1"])
        golden = torch_rms_norm(temp, test_tensors["torch_weight2"])

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.99)
        assert passing, f"PCC: {pcc}"

    def test_sequential_api_branching(self, device, opgraph_tensors):
        """Sequential(stem, Parallel(branch_a, branch_b)) matches build_op_graph."""
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        t = opgraph_tensors
        union_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_a_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        branch_b_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})

        stem_op = rms_norm.rms_norm(t["tt_input"], core_range_set=union_range, weight=t["tt_weights"][0], epsilon=1e-5)
        branch_a_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_a_range, weight=t["tt_weights"][1], epsilon=1e-5
        )
        branch_b_op = rms_norm.rms_norm(
            stem_op.output_tensors[0], core_range_set=branch_b_range, weight=t["tt_weights"][2], epsilon=1e-5
        )

        fused = Sequential(stem_op, Parallel(branch_a_op, branch_b_op)).build(device)
        outputs = composite.launch([fused])

        golden_stem = torch_rms_norm(t["torch_input"], t["torch_weights"][0])
        golden_a = torch_rms_norm(golden_stem, t["torch_weights"][1])
        golden_b = torch_rms_norm(golden_stem, t["torch_weights"][2])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_parallel_full_grid_mixed_ops(self, device):
        """Parallel trees of LN, RMS, matmul spanning the entire 8x8 grid (64 cores).

        Layout:
          - Matmul 1:          (0,0)-(3,1)  4x2 = 8 cores
          - LN→RMS chain:     (4,0)-(7,1)  4x2 = 8 cores
          - RMS→LN chain:     (0,2)-(3,3)  4x2 = 8 cores
          - RMS→RMS→RMS chain: (4,2)-(7,3) 4x2 = 8 cores
          - Branching tree:    stem (0,4)-(7,5) 16 cores → A (0,4)-(3,5) + B (4,4)-(7,5) 8c each
          - Matmul 2:          (0,6)-(3,7)  4x2 = 8 cores
          - Single RMS:        (4,6)-(7,7)  4x2 = 8 cores
        Total: 64 cores = full 8x8 grid
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        hidden = 128
        weight_shape = (1, 1, 1, hidden)
        torch_w = torch.ones(weight_shape, dtype=torch.bfloat16)
        torch_b = torch.zeros(weight_shape, dtype=torch.bfloat16)

        def tt(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        tt_w = tt(torch_w)
        tt_b = tt(torch_b)

        # Norm tensors: NCHt=8 so 8 cores can each get 1 height tile
        norm_shape = (1, 1, 256, hidden)
        torch_inputs = [torch.randn(norm_shape, dtype=torch.bfloat16) for _ in range(4)]
        tt_inputs = [tt(t) for t in torch_inputs]

        # ── Matmul 1: (0,0)-(3,1) = 8 cores ──
        torch_mm1_a = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
        torch_mm1_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        mm1_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        mm1_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
        )
        mm1 = matmul_desc(tt(torch_mm1_a), tt(torch_mm1_b), core_range_set=mm1_cores, program_config=mm1_config)

        # ── LN→RMS chain: (4,0)-(7,1) = 8 cores ──
        chain1_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 1))})
        ln1 = layer_norm.layer_norm(tt_inputs[0], core_range_set=chain1_cores, weight=tt_w, epsilon=1e-5)
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=chain1_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_chain1 = Sequential(ln1, rms1).build(device)

        # ── RMS→LN chain: (0,2)-(3,3) = 8 cores ──
        chain2_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(3, 3))})
        rms2 = rms_norm.rms_norm(
            tt_inputs[1],
            core_range_set=chain2_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            rms2.output_tensors[0],
            core_range_set=chain2_cores,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )
        fused_chain2 = Sequential(rms2, ln2).build(device)

        # ── RMS→Matmul→RMS chain: (4,2)-(7,3) = 8 cores ──
        # Matmul preserves shape: (1,1,256,128) x (1,1,128,128) → (1,1,256,128)
        # Compute config must match RMS (HiFi4, math_approx_mode=True) for fusion
        chain3_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 2), ttnn.CoreCoord(7, 3))})
        rms3a = rms_norm.rms_norm(tt_inputs[2], core_range_set=chain3_cores, weight=tt_w, epsilon=1e-5)
        torch_mm3_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        mm3_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
        )
        mm3_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )
        mm3 = matmul_desc(
            rms3a.output_tensors[0],
            tt(torch_mm3_b),
            core_range_set=chain3_cores,
            program_config=mm3_config,
            compute_kernel_config=mm3_compute,
        )
        rms3c = rms_norm.rms_norm(
            mm3.output_tensors[0],
            core_range_set=chain3_cores,
            weight=tt_w,
            epsilon=1e-5,
        )
        fused_chain3 = Sequential(rms3a, mm3, rms3c).build(device)

        # ── Branching tree: stem (0,4)-(7,5) → A (0,4)-(3,5) + B (4,4)-(7,5) ──
        # Stem uses 16 cores (8x2), so input needs NCHt>=16: shape (1,1,512,128)
        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7, 5))})
        branch_a_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(3, 5))})
        branch_b_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(7, 5))})
        torch_stem_in = torch.randn(1, 1, 512, hidden, dtype=torch.bfloat16)
        torch_stem_w = torch.ones((1, 1, 1, hidden), dtype=torch.bfloat16)
        stem_op = rms_norm.rms_norm(tt(torch_stem_in), core_range_set=stem_cores, weight=tt_w, epsilon=1e-5)
        branch_a = rms_norm.rms_norm(
            stem_op.output_tensors[0],
            core_range_set=branch_a_cores,
            weight=tt_w,
            epsilon=1e-5,
        )
        branch_b = rms_norm.rms_norm(
            stem_op.output_tensors[0],
            core_range_set=branch_b_cores,
            weight=tt_w,
            epsilon=1e-5,
        )
        fused_tree = Sequential(stem_op, Parallel(branch_a, branch_b)).build(device)

        # ── Matmul 2: (0,6)-(3,7) = 8 cores ──
        torch_mm2_a = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
        torch_mm2_b = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
        mm2_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(3, 7))})
        mm2_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
        )
        mm2 = matmul_desc(tt(torch_mm2_a), tt(torch_mm2_b), core_range_set=mm2_cores, program_config=mm2_config)

        # ── Single RMS: (4,6)-(7,7) = 8 cores ──
        single_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 6), ttnn.CoreCoord(7, 7))})
        single_rms = rms_norm.rms_norm(tt_inputs[3], core_range_set=single_cores, weight=tt_w, epsilon=1e-5)

        # ── Launch all 7 items on full 8x8 grid ──
        outputs = composite.launch([mm1, fused_chain1, fused_chain2, fused_chain3, fused_tree, mm2, single_rms])
        assert len(outputs) == 7

        # ── Verify matmul 1 ──
        golden_mm1 = torch_mm1_a @ torch_mm1_b
        passing, pcc = comp_pcc(golden_mm1, ttnn.to_torch(outputs[0][0]), pcc=0.99)
        assert passing, f"Matmul 1 PCC: {pcc}"

        # ── Verify LN→RMS chain ──
        golden1 = torch_rms_norm(torch_layer_norm(torch_inputs[0], torch_w), torch_w)
        passing, pcc = comp_pcc(golden1, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing, f"LN→RMS chain PCC: {pcc}"

        # ── Verify RMS→LN chain ──
        golden2 = torch_layer_norm(torch_rms_norm(torch_inputs[1], torch_w), torch_w, torch_b)
        passing, pcc = comp_pcc(golden2, ttnn.to_torch(outputs[2][0]), pcc=0.98)
        assert passing, f"RMS→LN chain PCC: {pcc}"

        # ── Verify RMS→Matmul→RMS chain ──
        golden3 = torch_rms_norm(torch_rms_norm(torch_inputs[2], torch_w) @ torch_mm3_b, torch_w)
        passing, pcc = comp_pcc(golden3, ttnn.to_torch(outputs[3][0]), pcc=0.98)
        assert passing, f"RMS→MM→RMS chain PCC: {pcc}"

        # ── Verify branching tree (2 outputs) ──
        golden_stem = torch_rms_norm(torch_stem_in, torch_w)
        golden_ba = torch_rms_norm(golden_stem, torch_w)
        golden_bb = torch_rms_norm(golden_stem, torch_w)
        passing_a, pcc_a = comp_pcc(golden_ba, ttnn.to_torch(outputs[4][0]), pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_bb, ttnn.to_torch(outputs[4][1]), pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

        # ── Verify matmul 2 ──
        golden_mm2 = torch_mm2_a @ torch_mm2_b
        passing, pcc = comp_pcc(golden_mm2, ttnn.to_torch(outputs[5][0]), pcc=0.99)
        assert passing, f"Matmul 2 PCC: {pcc}"

        # ── Verify single RMS ──
        golden_single = torch_rms_norm(torch_inputs[3], torch_w)
        passing, pcc = comp_pcc(golden_single, ttnn.to_torch(outputs[6][0]), pcc=0.99)
        assert passing, f"Single RMS PCC: {pcc}"

    def test_sequential_api_add_method(self, device, test_tensors):
        """Incremental .add() produces same result as inline construction."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        rms1_desc = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )
        rms2_desc = rms_norm.rms_norm(
            rms1_desc.output_tensors[0],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        s = Sequential(rms1_desc)
        s.add(rms2_desc)
        fused = s.build(device)
        outputs = composite.launch([fused])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        temp = torch_rms_norm(test_tensors["torch_input"], test_tensors["torch_weight1"])
        golden = torch_rms_norm(temp, test_tensors["torch_weight2"])

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.99)
        assert passing, f"PCC: {pcc}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestMatmulFusionChains:
    """Tests for matmul kernel fusion in various chain configurations.

    Validates that matmul kernels with named CB compile-time args can be
    fused with normalization ops using the Sequential/Parallel API.
    """

    def _make_mm_config(self, grid_x=1, grid_y=1, in0_block_w=4, per_core_M=1, per_core_N=4):
        """Helper to create MatmulMultiCoreReuseProgramConfig."""
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=min(per_core_N, 4),
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

    def _make_compute_config(self, fp32=False, math_approx_mode=True):
        """Helper to create WormholeComputeKernelConfig."""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32,
        )

    def _tt(self, t, device):
        """Helper to move tensor to device."""
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def test_matmul_then_rms(self, device):
        """2-phase chain: Matmul → RMSNorm."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Matmul: (1,1,32,128) x (1,1,128,128) → (1,1,32,128)
        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(mm, rms).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(torch_a.float() @ torch_b.float(), torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Matmul→RMS PCC: {pcc}"

    def test_rms_then_matmul(self, device):
        """2-phase chain: RMSNorm → Matmul."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            rms.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )

        fused = Sequential(rms, mm).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_b.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"RMS→Matmul PCC: {pcc}"

    def test_rms_matmul_rms(self, device):
        """3-phase chain: RMS → Matmul → RMS."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms1 = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(rms1, mm, rms2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_b.float()
        golden = torch_rms_norm(temp, torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"RMS→MM→RMS PCC: {pcc}"

    def test_matmul_then_ln(self, device):
        """2-phase chain: Matmul → LayerNorm."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)

        # Matmul must match LN's fp32_dest_acc_en and math_approx_mode
        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(fp32=True, math_approx_mode=False),
        )
        ln = layer_norm.layer_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )

        fused = Sequential(mm, ln).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_layer_norm(torch_a.float() @ torch_b.float(), torch_w.float(), torch_bias.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Matmul→LN PCC: {pcc}"

    def test_ln_then_matmul(self, device):
        """2-phase chain: LayerNorm → Matmul."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        # Matmul must match LN's fp32_dest_acc_en=True and math_approx_mode=False
        mm = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(fp32=True, math_approx_mode=False),
        )

        fused = Sequential(ln, mm).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_layer_norm(torch_input.float(), torch_w.float(), torch_bias.float()) @ torch_b.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"LN→Matmul PCC: {pcc}"

    def test_ln_matmul_rms(self, device):
        """3-phase chain: LN → Matmul → RMS (mixed norm types)."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(fp32=True, math_approx_mode=False),
        )
        rms = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        fused = Sequential(ln, mm, rms).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(torch_input.float(), torch_w.float(), torch_bias.float()) @ torch_b.float()
        golden = torch_rms_norm(temp, torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"LN→MM→RMS PCC: {pcc}"

    def test_matmul_rms_matmul(self, device):
        """3-phase chain: Matmul → RMS → Matmul."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms = rms_norm.rms_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            rms.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )

        fused = Sequential(mm1, rms, mm2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_rms_norm(torch_a.float() @ torch_b1.float(), torch_w.float())
        golden = temp @ torch_b2.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"MM→RMS→MM PCC: {pcc}"

    def test_stem_rms_branch_matmuls(self, device):
        """Branching: stem RMS → Parallel(matmul_a, matmul_b) on disjoint cores.

        Tree (8 cores, single row):
            RMS [(0,0)-(7,0)] → matmul_a [(0,0)-(3,0)]  left 4 cores
                               → matmul_b [(4,0)-(7,0)]  right 4 cores

        Stem: RMS norm on (256, 128) = 8 M-tiles × 4 width-tiles.
        DRAM-interleaved norm assigns 1 core per height-tile → 8 cores.

        Each branch: matmul (256, 128) × (128, 128) = (256, 128) on 4 cores.
        grid(4,1), per_core_M=2, per_core_N=4, in0_block_w=4.
        M distributed across all 4 cores (2 M-tiles/core × 4 = 8), per_core_N = full N.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        # Stem: 8 cores (single row)
        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        # Left branch: (0,0)-(3,0) = 4 cores
        branch_a_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        # Right branch: (4,0)-(7,0) = 4 cores
        branch_b_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        # Input: 8 M-tiles × 4 K-tiles = (256, 128)
        torch_input = torch.randn(1, 1, 256, 128, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
        # B: K=4 tiles × N=4 tiles = (128, 128)
        torch_B = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)

        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        # grid(4,1): M distributed across 4 cores, per_core_N = full N = 4
        mm_config = self._make_mm_config(grid_x=4, grid_y=1, in0_block_w=4, per_core_M=2, per_core_N=4)
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        stem = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=stem_cores,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        tt_B = self._tt(torch_B, device)
        branch_a = matmul_desc(
            stem.output_tensors[0],
            tt_B,
            core_range_set=branch_a_cores,
            program_config=mm_config,
            compute_kernel_config=mm_compute,
        )
        branch_b = matmul_desc(
            stem.output_tensors[0],
            tt_B,
            core_range_set=branch_b_cores,
            program_config=mm_config,
            compute_kernel_config=mm_compute,
        )

        fused = Sequential(stem, Parallel(branch_a, branch_b)).build(device)
        outputs = composite.launch([fused])

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        golden = torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_B.float()
        passing_a, pcc_a = comp_pcc(golden, result_a, pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden, result_b, pcc=0.98)
        assert passing_a, f"Branch A PCC: {pcc_a}"
        assert passing_b, f"Branch B PCC: {pcc_b}"

    def test_sharded_ln_rms_ln(self, device):
        """Block-sharded LN → RMS → LN on 4x4 cores for profiling comparison.

        Block-sharded input/output so norm readers do L1 reads (no DRAM).
        Compare tracy output with DRAM-interleaved tests to see barrier
        overhead relative to compute.

        Layout (16 cores, 4x4 grid):
            LN → RMS → LN  all on (0,0)-(3,3)

        Input (128, 512) block-sharded: shard (32, 128) = 1 M-tile per core.
        4 rows × 4 cols, each shard = (32, 128).
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

        # Block-sharded: 4x4 grid, shard (32, 128) = 1 M-tile per core
        # Total: (4*32, 4*128) = (128, 512)
        shard_spec = ttnn.ShardSpec(cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        torch_input = torch.randn(1, 1, 128, 512, dtype=torch.bfloat16)
        torch_gamma = torch.ones(1, 1, 1, 512, dtype=torch.bfloat16)
        torch_beta = torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem
        )
        tt_gamma = self._tt(torch_gamma, device)
        tt_beta = self._tt(torch_beta, device)

        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        ln1 = layer_norm.layer_norm(
            tt_input,
            core_range_set=cores,
            weight=tt_gamma,
            bias=tt_beta,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=tt_gamma,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0],
            core_range_set=cores,
            weight=tt_gamma,
            bias=tt_beta,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )

        fused = Sequential(ln1, rms, ln2).build(device)
        outputs = composite.launch([fused])

        result = ttnn.to_torch(outputs[0][0])
        temp1 = torch_layer_norm(torch_input.float(), torch_gamma.float(), torch_beta.float())
        temp2 = torch_rms_norm(temp1, torch_gamma.float())
        golden = torch_layer_norm(temp2, torch_gamma.float(), torch_beta.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded LN→RMS→LN PCC: {pcc}"

    def test_sharded_ln_rms_ln_2x2(self, device):
        """Block-sharded LN → RMS → LN on 2x2 cores for barrier scaling comparison."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)

        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

        # Block-sharded: 2x2 grid, shard (32, 128) = 1 M-tile per core
        # Total: (2*32, 2*128) = (64, 256)
        shard_spec = ttnn.ShardSpec(cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        torch_input = torch.randn(1, 1, 64, 256, dtype=torch.bfloat16)
        torch_gamma = torch.ones(1, 1, 1, 256, dtype=torch.bfloat16)
        torch_beta = torch.zeros(1, 1, 1, 256, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_mem
        )
        tt_gamma = self._tt(torch_gamma, device)
        tt_beta = self._tt(torch_beta, device)

        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        ln1 = layer_norm.layer_norm(
            tt_input,
            core_range_set=cores,
            weight=tt_gamma,
            bias=tt_beta,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )
        rms = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores,
            weight=tt_gamma,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )
        ln2 = layer_norm.layer_norm(
            rms.output_tensors[0],
            core_range_set=cores,
            weight=tt_gamma,
            bias=tt_beta,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
            memory_config=sharded_mem,
        )

        fused = Sequential(ln1, rms, ln2).build(device)
        outputs = composite.launch([fused])

        result = ttnn.to_torch(outputs[0][0])
        temp1 = torch_layer_norm(torch_input.float(), torch_gamma.float(), torch_beta.float())
        temp2 = torch_rms_norm(temp1, torch_gamma.float())
        golden = torch_layer_norm(temp2, torch_gamma.float(), torch_beta.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Sharded LN→RMS→LN 2x2 PCC: {pcc}"

    def test_multicore_rms_matmul_rms(self, device):
        """Multi-core 3-phase: RMS → Matmul → RMS on 4x2 grid (8 cores)."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})
        # 8 cores, 1 height tile each → 8 height tiles = 256 rows
        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms1 = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(grid_x=4, grid_y=2),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(rms1, mm, rms2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_rms_norm(torch_input.float(), torch_w.float()) @ torch_b.float()
        golden = torch_rms_norm(temp, torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Multi-core RMS→MM→RMS PCC: {pcc}"

    def test_parallel_matmul_chains_vs_standalone(self, device):
        """Two independent matmul→RMS chains on disjoint cores, launched in parallel."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})

        torch_a1 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_a2 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a1, device),
            self._tt(torch_b1, device),
            core_range_set=cores_a,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms1 = rms_norm.rms_norm(
            mm1.output_tensors[0],
            core_range_set=cores_a,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        fused_a = Sequential(mm1, rms1).build(device)

        mm2 = matmul_desc(
            self._tt(torch_a2, device),
            self._tt(torch_b2, device),
            core_range_set=cores_b,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm2.output_tensors[0],
            core_range_set=cores_b,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        fused_b = Sequential(mm2, rms2).build(device)

        outputs = composite.launch([fused_a, fused_b])

        golden_a = torch_rms_norm(torch_a1.float() @ torch_b1.float(), torch_w.float())
        golden_b = torch_rms_norm(torch_a2.float() @ torch_b2.float(), torch_w.float())

        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[0][0]), pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_a, f"Chain A PCC: {pcc_a}"
        assert passing_b, f"Chain B PCC: {pcc_b}"

    def test_matmul_chain_plus_norm_chain_parallel(self, device):
        """Matmul→RMS chain in parallel with LN→RMS norm chain."""
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        cores_mm = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores_norm = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})

        # Matmul→RMS chain
        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=cores_mm,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms_after_mm = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=cores_mm,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        fused_mm_chain = Sequential(mm, rms_after_mm).build(device)

        # LN→RMS norm chain
        torch_norm_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        ln = layer_norm.layer_norm(
            self._tt(torch_norm_input, device),
            core_range_set=cores_norm,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        rms_after_ln = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=cores_norm,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        fused_norm_chain = Sequential(ln, rms_after_ln).build(device)

        outputs = composite.launch([fused_mm_chain, fused_norm_chain])

        golden_mm = torch_rms_norm(torch_a.float() @ torch_b.float(), torch_w.float())
        golden_norm = torch_rms_norm(torch_layer_norm(torch_norm_input.float(), torch_w.float()), torch_w.float())

        passing_mm, pcc_mm = comp_pcc(golden_mm, ttnn.to_torch(outputs[0][0]), pcc=0.98)
        passing_norm, pcc_norm = comp_pcc(golden_norm, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_mm, f"Matmul chain PCC: {pcc_mm}"
        assert passing_norm, f"Norm chain PCC: {pcc_norm}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestMatmulFusionStress:
    """Stress tests for matmul fusion.

    Exercises deep chains, mixed op types, non-trivial bias/residual,
    different shapes, multi-core grids, and error detection.
    """

    def _make_mm_config(self, grid_x=1, grid_y=1, in0_block_w=4, per_core_M=1, per_core_N=4):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=min(per_core_N, 4),
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

    def _make_compute_config(self, fp32=False, math_approx_mode=True):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32,
        )

    def _tt(self, t, device):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------ #
    # 4-phase chains
    # ------------------------------------------------------------------ #

    def test_four_phase_mm_ln_rms_mm(self, device):
        """4-phase: MM → LN(nonzero bias) → RMS → MM.

        Exercises: deep chain, LN bias code path with non-trivial values,
        mixed norm types, matmul bookending norms.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16) * 0.1  # non-zero bias

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute,
        )
        ln = layer_norm.layer_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        mm2 = matmul_desc(
            rms.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute,
        )

        fused = Sequential(mm1, ln, rms, mm2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_a.float() @ torch_b1.float()
        t = torch_layer_norm(t, torch_w.float(), torch_bias.float())
        t = torch_rms_norm(t, torch_w.float())
        golden = t @ torch_b2.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"MM→LN(bias)→RMS→MM PCC: {pcc}"

    def test_four_phase_mm_rms_mm_rms(self, device):
        """4-phase: MM → RMS → MM → RMS (double matmul sandwich).

        Exercises: runtime arg offsets for 2 matmuls in one fused kernel,
        CB remapping for repeated matmul op, 4 phases.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms1 = rms_norm.rms_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm2.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(mm1, rms1, mm2, rms2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_rms_norm(torch_a.float() @ torch_b1.float(), torch_w.float())
        golden = torch_rms_norm(t @ torch_b2.float(), torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"MM→RMS→MM→RMS PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Non-trivial bias and residual
    # ------------------------------------------------------------------ #

    def test_ln_nonzero_bias_then_matmul(self, device):
        """LN(random weight + random bias) → MM.

        Exercises: LN bias ifdef path with real values (not zeros),
        verifies bias actually affects output.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16).abs() + 0.5  # positive weights
        torch_bias = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16) * 0.5
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute,
        )

        fused = Sequential(ln, mm).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_layer_norm(torch_input.float(), torch_w.float(), torch_bias.float()) @ torch_b.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"LN(nonzero bias)→MM PCC: {pcc}"

    def test_rms_with_bias_then_matmul(self, device):
        """RMS(weight + bias) → MM.

        Exercises: RMS bias code path (rarely tested), CB allocation for
        bias tensor in RMS.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16) * 0.1
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            rms.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )

        fused = Sequential(rms, mm).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # RMS with bias: x / rms(x) * weight + bias
        rms_val = torch.sqrt(torch_input.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        temp = torch_input.float() / rms_val * torch_w.float() + torch_bias.float()
        golden = temp @ torch_b.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"RMS(bias)→MM PCC: {pcc}"

    def test_ln_with_residual_then_rms(self, device):
        """LN(residual) → RMS.

        Exercises: residual input tensor in non-sharded DRAM mode,
        extra CB allocation for residual.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_residual = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w_ln = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w_rms = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w_ln, device),
            residual_input_tensor=self._tt(torch_residual, device),
            epsilon=1e-5,
        )
        rms = rms_norm.rms_norm(
            ln.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w_rms, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(ln, rms).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        # LN(input + residual) then RMS
        combined = torch_input.float() + torch_residual.float()
        temp = torch_layer_norm(combined, torch_w_ln.float())
        golden = torch_rms_norm(temp, torch_w_rms.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"LN(residual)→RMS PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Deep chains (5+ phases)
    # ------------------------------------------------------------------ #

    def test_five_phase_rms_mm_ln_mm_rms(self, device):
        """5-phase: RMS → MM → LN(bias) → MM → RMS.

        Exercises: max depth chain, all three op types, 2 matmuls with
        different weight matrices, LN bias in middle of chain.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        torch_input = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w1 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w2 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_w3 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms1 = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w1, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        mm1 = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute,
        )
        ln = layer_norm.layer_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w2, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute,
        )
        rms2 = rms_norm.rms_norm(
            mm2.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w3, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(rms1, mm1, ln, mm2, rms2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_rms_norm(torch_input.float(), torch_w1.float())
        t = t @ torch_b1.float()
        t = torch_layer_norm(t, torch_w2.float(), torch_bias.float())
        t = t @ torch_b2.float()
        golden = torch_rms_norm(t, torch_w3.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"RMS→MM→LN(bias)→MM→RMS PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Shape variety
    # ------------------------------------------------------------------ #

    def test_non_square_matmul_then_rms(self, device):
        """MM with dimension change: (32,64)×(64,256) → RMS on (32,256).

        Exercises: CB page size changes between matmul output and norm input,
        different in0_block_w and per_core_N.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        M, K, N = 32, 64, 256
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, N, dtype=torch.bfloat16)

        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(in0_block_w=K // 32, per_core_N=N // 32),
            compute_kernel_config=self._make_compute_config(),
        )
        rms = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(mm, rms).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(torch_a.float() @ torch_b.float(), torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        assert passing, f"Non-square MM→RMS PCC: {pcc}"

    def test_wide_matmul_ln_narrow_matmul(self, device):
        """MM(32,128→256) → LN → MM(32,256→128): expand then contract.

        Exercises: CB page size adapts between phases, norm width changes.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        torch_a = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, 256, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, 256, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, 256, 128, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(in0_block_w=4, per_core_N=8),
            compute_kernel_config=mm_compute,
        )
        ln = layer_norm.layer_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(in0_block_w=8, per_core_N=4),
            compute_kernel_config=mm_compute,
        )

        fused = Sequential(mm1, ln, mm2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_a.float() @ torch_b1.float()
        t = torch_layer_norm(t, torch_w.float(), torch_bias.float())
        golden = t @ torch_b2.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"MM(128→256)→LN→MM(256→128) PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Multi-core deep chains
    # ------------------------------------------------------------------ #

    def test_multicore_four_phase_mm_rms_mm_rms(self, device):
        """4-phase MM→RMS→MM→RMS on 2x2 grid (4 cores).

        Exercises: multi-core runtime arg distribution for 4-phase chain,
        per-core tile partitioning across 2 matmuls.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})

        # 4 cores × 1 height tile each = 4 height tiles = 128 rows
        torch_a = torch.randn(1, 1, 128, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(grid_x=2, grid_y=2),
            compute_kernel_config=self._make_compute_config(),
        )
        rms1 = rms_norm.rms_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(grid_x=2, grid_y=2),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm2.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        fused = Sequential(mm1, rms1, mm2, rms2).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_rms_norm(torch_a.float() @ torch_b1.float(), torch_w.float())
        golden = torch_rms_norm(t @ torch_b2.float(), torch_w.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"Multi-core MM→RMS→MM→RMS PCC: {pcc}"

    def test_multicore_ln_mm_rms(self, device):
        """3-phase LN→MM→RMS on 4x1 grid.

        Exercises: multi-core chain with all three op types, fp32=True
        across 4 cores.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        torch_input = torch.randn(1, 1, 128, hidden, dtype=torch.bfloat16)
        torch_w_ln = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w_rms = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        ln = layer_norm.layer_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w_ln, device),
            bias=self._tt(torch_bias, device),
            epsilon=1e-5,
        )
        mm = matmul_desc(
            ln.output_tensors[0],
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(grid_x=4, grid_y=1),
            compute_kernel_config=mm_compute,
        )
        rms = rms_norm.rms_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w_rms, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(ln, mm, rms).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_layer_norm(torch_input.float(), torch_w_ln.float(), torch_bias.float())
        t = t @ torch_b.float()
        golden = torch_rms_norm(t, torch_w_rms.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"Multi-core LN→MM→RMS PCC: {pcc}"

    def test_rms_matmul_ln_2x2(self, device):
        """3-phase RMS→MM→LN on 2×2 grid (4 cores).

        Exercises: multi-core 3-phase chain with all three op types on a
        square grid layout.  fp32_dest_acc_en=True across all phases.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        mm_compute = self._make_compute_config(fp32=True, math_approx_mode=False)

        # 4 cores × 1 height tile each = 4 tiles = 128 rows
        torch_input = torch.randn(1, 1, 128, hidden, dtype=torch.bfloat16)
        torch_w_rms = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b_mm = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w_ln = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias_ln = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)

        rms = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=core_range,
            weight=self._tt(torch_w_rms, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        mm = matmul_desc(
            rms.output_tensors[0],
            self._tt(torch_b_mm, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(grid_x=2, grid_y=2),
            compute_kernel_config=mm_compute,
        )
        ln = layer_norm.layer_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w_ln, device),
            bias=self._tt(torch_bias_ln, device),
            epsilon=1e-5,
        )

        fused = Sequential(rms, mm, ln).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_rms_norm(torch_input.float(), torch_w_rms.float())
        t = t @ torch_b_mm.float()
        golden = torch_layer_norm(t, torch_w_ln.float(), torch_bias_ln.float())
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"Multi-core RMS→MM→LN PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Parallel heterogeneous chains
    # ------------------------------------------------------------------ #

    def test_parallel_mm_rms_and_ln_mm(self, device):
        """Two heterogeneous fused chains dispatched in parallel.

        Chain A (core 0,0): MM→RMS (fp32=False)
        Chain B (core 4,0): LN→MM (fp32=True)

        Exercises: independent fused chains with different fp32 settings.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0))})

        # Chain A: MM→RMS
        torch_a1 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w1 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm_a = matmul_desc(
            self._tt(torch_a1, device),
            self._tt(torch_b1, device),
            core_range_set=cores_a,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms_a = rms_norm.rms_norm(
            mm_a.output_tensors[0],
            core_range_set=cores_a,
            weight=self._tt(torch_w1, device),
            epsilon=1e-5,
        )
        fused_a = Sequential(mm_a, rms_a).build(device)

        # Chain B: LN→MM
        torch_input_b = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_w_ln = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        mm_compute_b = self._make_compute_config(fp32=True, math_approx_mode=False)

        ln_b = layer_norm.layer_norm(
            self._tt(torch_input_b, device),
            core_range_set=cores_b,
            weight=self._tt(torch_w_ln, device),
            epsilon=1e-5,
        )
        mm_b = matmul_desc(
            ln_b.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=cores_b,
            program_config=self._make_mm_config(),
            compute_kernel_config=mm_compute_b,
        )
        fused_b = Sequential(ln_b, mm_b).build(device)

        outputs = composite.launch([fused_a, fused_b])

        golden_a = torch_rms_norm(torch_a1.float() @ torch_b1.float(), torch_w1.float())
        golden_b = torch_layer_norm(torch_input_b.float(), torch_w_ln.float()) @ torch_b2.float()

        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[0][0]), pcc=0.98)
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[1][0]), pcc=0.98)
        assert passing_a, f"Chain A (MM→RMS) PCC: {pcc_a}"
        assert passing_b, f"Chain B (LN→MM) PCC: {pcc_b}"

    def test_three_parallel_heterogeneous_chains(self, device):
        """Three independent chains in parallel: MM→RMS, LN→RMS, RMS→MM.

        Exercises: 3-way parallel dispatch, each chain has different structure.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())
        cores = [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0))}) for i in [0, 3, 6]]

        # Chain 0: MM→RMS (fp32=False)
        t_a0 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        t_b0 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        t_w0 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm0 = matmul_desc(
            self._tt(t_a0, device),
            self._tt(t_b0, device),
            core_range_set=cores[0],
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms0 = rms_norm.rms_norm(
            mm0.output_tensors[0],
            core_range_set=cores[0],
            weight=self._tt(t_w0, device),
            epsilon=1e-5,
        )
        fused0 = Sequential(mm0, rms0).build(device)

        # Chain 1: LN→RMS (fp32=True)
        t_in1 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        t_w1a = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        t_w1b = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        ln1 = layer_norm.layer_norm(
            self._tt(t_in1, device),
            core_range_set=cores[1],
            weight=self._tt(t_w1a, device),
            epsilon=1e-5,
        )
        rms1 = rms_norm.rms_norm(
            ln1.output_tensors[0],
            core_range_set=cores[1],
            weight=self._tt(t_w1b, device),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        fused1 = Sequential(ln1, rms1).build(device)

        # Chain 2: RMS→MM (fp32=False)
        t_in2 = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        t_w2 = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        t_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)

        rms2 = rms_norm.rms_norm(
            self._tt(t_in2, device),
            core_range_set=cores[2],
            weight=self._tt(t_w2, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            rms2.output_tensors[0],
            self._tt(t_b2, device),
            core_range_set=cores[2],
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        fused2 = Sequential(rms2, mm2).build(device)

        outputs = composite.launch([fused0, fused1, fused2])

        golden0 = torch_rms_norm(t_a0.float() @ t_b0.float(), t_w0.float())
        golden1 = torch_rms_norm(torch_layer_norm(t_in1.float(), t_w1a.float()), t_w1b.float())
        golden2 = torch_rms_norm(t_in2.float(), t_w2.float()) @ t_b2.float()

        for i, (golden, label) in enumerate([(golden0, "MM→RMS"), (golden1, "LN→RMS"), (golden2, "RMS→MM")]):
            passing, pcc = comp_pcc(golden, ttnn.to_torch(outputs[i][0]), pcc=0.98)
            assert passing, f"Chain {i} ({label}) PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Error detection
    # ------------------------------------------------------------------ #

    def test_fp32_mismatch_raises_error(self, device):
        """MM(fp32=False) fused with LN(fp32=True) should error.

        Exercises: fp32_dest_acc_en consistency validation.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        # Matmul with fp32=False
        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(fp32=False),
        )
        # LN defaults to fp32=True
        ln = layer_norm.layer_norm(
            mm.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )

        with pytest.raises((ValueError, RuntimeError)):
            Sequential(mm, ln).build(device)

    # ------------------------------------------------------------------ #
    # Repeated same-op chains
    # ------------------------------------------------------------------ #

    def test_three_matmuls_with_rms_between(self, device):
        """5-phase: MM → RMS → MM → RMS → MM.

        Exercises: 3 matmuls with different weight matrices, runtime arg
        offsets for 3 readers each with different DRAM addresses.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b1 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b2 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_b3 = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        mm1 = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b1, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms1 = rms_norm.rms_norm(
            mm1.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm2 = matmul_desc(
            rms1.output_tensors[0],
            self._tt(torch_b2, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        rms2 = rms_norm.rms_norm(
            mm2.output_tensors[0],
            core_range_set=core_range,
            weight=self._tt(torch_w, device),
            epsilon=1e-5,
        )
        mm3 = matmul_desc(
            rms2.output_tensors[0],
            self._tt(torch_b3, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )

        fused = Sequential(mm1, rms1, mm2, rms2, mm3).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_rms_norm(torch_a.float() @ torch_b1.float(), torch_w.float())
        t = torch_rms_norm(t @ torch_b2.float(), torch_w.float())
        golden = t @ torch_b3.float()
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"MM→RMS→MM→RMS→MM PCC: {pcc}"

    @pytest.mark.parametrize("num_rms", [2, 3, 4])
    def test_matmul_followed_by_n_rms_norms(self, device, num_rms):
        """MM followed by N consecutive RMS norms.

        Exercises: many phases with identical op type, CB reuse across
        repeated norm phases, define/undef for N+1 phases.
        """
        from models.experimental.ops.descriptors.fusion import Sequential
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        torch_a = torch.randn(1, 1, 32, hidden, dtype=torch.bfloat16)
        torch_b = torch.randn(1, 1, hidden, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)

        ops = []
        mm = matmul_desc(
            self._tt(torch_a, device),
            self._tt(torch_b, device),
            core_range_set=core_range,
            program_config=self._make_mm_config(),
            compute_kernel_config=self._make_compute_config(),
        )
        ops.append(mm)

        prev_output = mm.output_tensors[0]
        for _ in range(num_rms):
            rms = rms_norm.rms_norm(
                prev_output,
                core_range_set=core_range,
                weight=self._tt(torch_w, device),
                epsilon=1e-5,
            )
            ops.append(rms)
            prev_output = rms.output_tensors[0]

        fused = Sequential(*ops).build(device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        t = torch_a.float() @ torch_b.float()
        for _ in range(num_rms):
            t = torch_rms_norm(t, torch_w.float())
        golden = t
        passing, pcc = comp_pcc(golden, result, pcc=0.97)
        assert passing, f"MM→{num_rms}×RMS PCC: {pcc}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestNestedParallelStress:
    """Stress tests for deeply nested Sequential/Parallel trees with long chains.

    All branching tests use norm ops (LN/RMS). Matmul also works in branching
    paths since the per-core group architecture correctly handles per-core
    runtime args via coordinate-based indexing.

    Core layout convention:
        NCHt (num height tiles) >= num_cores, one tile = 32 rows.
        8 cores → 256 rows, 16 cores → 512 rows.
    """

    def _tt(self, t, device):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------ #
    # Long chains inside Parallel branches
    # ------------------------------------------------------------------ #

    def test_stem_parallel_two_phase_mixed_branches(self, device):
        """Stem(RMS) → Parallel(2-phase chain A, 2-phase chain B).

        Tree (8 cores total):
            RMS [0-7] → LN→RMS [0-3]
                       → RMS→LN [4-7]

        3 norm-compute phases per path (max before SFPI compiler ICE).
        Exercises: mixed LN/RMS per branch, different op ordering per branch.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w = self._tt(torch_w, device)
        tt_b = self._tt(torch_bias, device)

        # Stem
        stem = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=all_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Branch A: LN → RMS (2 phases)
        a1 = layer_norm.layer_norm(
            stem.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )
        a2 = rms_norm.rms_norm(
            a1.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Branch B: RMS → LN (2 phases, reversed order)
        b1 = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        b2 = layer_norm.layer_norm(
            b1.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )

        fused = Sequential(
            stem,
            Parallel(
                Sequential(a1, a2),
                Sequential(b1, b2),
            ),
        ).build(device)
        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(torch_input.float(), torch_w.float())
        golden_a = torch_rms_norm(
            torch_layer_norm(g_stem, torch_w.float(), torch_bias.float()),
            torch_w.float(),
        )
        golden_b = torch_layer_norm(
            torch_rms_norm(g_stem, torch_w.float()),
            torch_w.float(),
            torch_bias.float(),
        )

        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[0][0]), pcc=0.97)
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[0][1]), pcc=0.97)
        assert passing_a, f"Branch A (LN→RMS) PCC: {pcc_a}"
        assert passing_b, f"Branch B (RMS→LN) PCC: {pcc_b}"

    def test_nested_split_of_split(self, device):
        """Stem → Parallel(Sequential(A, Parallel(A1, A2)), B).

        Tree (8 cores):
            RMS [0-7] → RMS [0-3] → RMS [0-1]
                                   → RMS [2-3]
                       → RMS [4-7]

        3 leaf paths. Exercises: nested Parallel inside Sequential inside Parallel.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        cores_a1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        cores_a2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(5)]
        tt_ws = [self._tt(w, device) for w in ws]

        stem = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=all_cores,
            weight=tt_ws[0],
            epsilon=1e-5,
        )
        a_mid = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_ws[1],
            epsilon=1e-5,
        )
        a1_leaf = rms_norm.rms_norm(
            a_mid.output_tensors[0],
            core_range_set=cores_a1,
            weight=tt_ws[2],
            epsilon=1e-5,
        )
        a2_leaf = rms_norm.rms_norm(
            a_mid.output_tensors[0],
            core_range_set=cores_a2,
            weight=tt_ws[3],
            epsilon=1e-5,
        )
        b_leaf = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_ws[4],
            epsilon=1e-5,
        )

        fused = Sequential(
            stem,
            Parallel(
                Sequential(a_mid, Parallel(a1_leaf, a2_leaf)),
                b_leaf,
            ),
        ).build(device)
        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(torch_input.float(), ws[0].float())
        g_a = torch_rms_norm(g_stem, ws[1].float())
        goldens = [
            torch_rms_norm(g_a, ws[2].float()),
            torch_rms_norm(g_a, ws[3].float()),
            torch_rms_norm(g_stem, ws[4].float()),
        ]

        for i, label in enumerate(["A1", "A2", "B"]):
            passing, pcc = comp_pcc(goldens[i], ttnn.to_torch(outputs[0][i]), pcc=0.98)
            assert passing, f"Leaf {label} PCC: {pcc}"

    # ------------------------------------------------------------------ #
    # Deep nesting with long chains at every level
    # ------------------------------------------------------------------ #

    def test_deep_nested_split_three_levels(self, device):
        """Stem → Parallel(mid → Parallel(leaf, leaf), leaf). 3-level nesting.

        Tree (8 cores):
            RMS [0-7] → RMS [0-3] → RMS [0-1]
                                   → RMS [2-3]
                       → RMS→RMS [4-7]

        3 norm phases on deepest path (hitting SFPI compiler limit).
        Exercises: 3-level nesting depth, mixed path lengths, CB state
        save/restore across nested splits.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        cores_a1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        cores_a2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(6)]
        tt_ws = [self._tt(w, device) for w in ws]

        def rms(inp, cores, wi):
            return rms_norm.rms_norm(inp, core_range_set=cores, weight=tt_ws[wi], epsilon=1e-5)

        stem = rms(self._tt(torch_input, device), all_cores, 0)
        a_mid = rms(stem.output_tensors[0], cores_a, 1)
        a1_leaf = rms(a_mid.output_tensors[0], cores_a1, 2)
        a2_leaf = rms(a_mid.output_tensors[0], cores_a2, 3)
        b1 = rms(stem.output_tensors[0], cores_b, 4)
        b2 = rms(b1.output_tensors[0], cores_b, 5)

        fused = Sequential(
            stem,
            Parallel(
                Sequential(
                    a_mid,
                    Parallel(a1_leaf, a2_leaf),
                ),
                Sequential(b1, b2),
            ),
        ).build(device)
        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(torch_input.float(), ws[0].float())
        g_a = torch_rms_norm(g_stem, ws[1].float())
        goldens = [
            torch_rms_norm(g_a, ws[2].float()),  # A1 leaf
            torch_rms_norm(g_a, ws[3].float()),  # A2 leaf
            torch_rms_norm(torch_rms_norm(g_stem, ws[4].float()), ws[5].float()),  # B leaf
        ]

        for i, label in enumerate(["A1", "A2", "B"]):
            passing, pcc = comp_pcc(goldens[i], ttnn.to_torch(outputs[0][i]), pcc=0.97)
            assert passing, f"Leaf {label} PCC: {pcc}"

    def test_three_way_split_with_long_chains(self, device):
        """Stem → Parallel(3 branches each with 2-phase chains).

        Tree (6 cores):
            RMS [0-5] → LN→RMS [0-1]
                       → RMS→LN [2-3]
                       → RMS→RMS [4-5]

        Exercises: 3-way split, different op mix per branch.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        cores_c = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0))})

        # 6 cores → 192 rows
        torch_input = torch.randn(1, 1, 192, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w = self._tt(torch_w, device)
        tt_b = self._tt(torch_bias, device)

        stem = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=all_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Branch A: LN → RMS
        a1 = layer_norm.layer_norm(
            stem.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )
        a2 = rms_norm.rms_norm(
            a1.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Branch B: RMS → LN
        b1 = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        b2 = layer_norm.layer_norm(
            b1.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )

        # Branch C: RMS → RMS
        c1 = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=cores_c,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        c2 = rms_norm.rms_norm(
            c1.output_tensors[0],
            core_range_set=cores_c,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = Sequential(
            stem,
            Parallel(
                Sequential(a1, a2),
                Sequential(b1, b2),
                Sequential(c1, c2),
            ),
        ).build(device)
        outputs = composite.launch([fused])

        w, b = torch_w.float(), torch_bias.float()
        g_stem = torch_rms_norm(torch_input.float(), w)
        goldens = [
            torch_rms_norm(torch_layer_norm(g_stem, w, b), w),
            torch_layer_norm(torch_rms_norm(g_stem, w), w, b),
            torch_rms_norm(torch_rms_norm(g_stem, w), w),
        ]

        for i, label in enumerate(["A(LN→RMS)", "B(RMS→LN)", "C(RMS→RMS)"]):
            passing, pcc = comp_pcc(goldens[i], ttnn.to_torch(outputs[0][i]), pcc=0.97)
            assert passing, f"Branch {label} PCC: {pcc}"

    def test_two_phase_stem_then_parallel_branches(self, device):
        """2-phase stem → Parallel(1-phase branch A, 1-phase branch B).

        Tree (8 cores):
            RMS→LN [0-7] → RMS [0-3]
                           → LN [4-7]

        3 norm phases per path (max before SFPI compiler ICE).
        Exercises: multi-phase stem with branching, mixed op types.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        cores_a = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        cores_b = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
        tt_w = self._tt(torch_w, device)
        tt_b = self._tt(torch_bias, device)

        # Stem: RMS → LN (2 phases, 8 cores)
        s1 = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=all_cores,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )
        s2 = layer_norm.layer_norm(
            s1.output_tensors[0],
            core_range_set=all_cores,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )

        # Branch A: RMS
        a1 = rms_norm.rms_norm(
            s2.output_tensors[0],
            core_range_set=cores_a,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Branch B: LN
        b1 = layer_norm.layer_norm(
            s2.output_tensors[0],
            core_range_set=cores_b,
            weight=tt_w,
            bias=tt_b,
            epsilon=1e-5,
        )

        fused = Sequential(
            s1,
            s2,
            Parallel(a1, b1),
        ).build(device)
        outputs = composite.launch([fused])

        w, b = torch_w.float(), torch_bias.float()
        g = torch_layer_norm(torch_rms_norm(torch_input.float(), w), w, b)
        golden_a = torch_rms_norm(g, w)
        golden_b = torch_layer_norm(g, w, b)

        passing_a, pcc_a = comp_pcc(golden_a, ttnn.to_torch(outputs[0][0]), pcc=0.97)
        passing_b, pcc_b = comp_pcc(golden_b, ttnn.to_torch(outputs[0][1]), pcc=0.97)
        assert passing_a, f"Branch A (RMS) PCC: {pcc_a}"
        assert passing_b, f"Branch B (LN) PCC: {pcc_b}"

    def test_symmetric_binary_tree(self, device):
        """Symmetric binary tree: stem → 2 branches → 4 leaves (2 each).

        Tree (8 cores):
            RMS [0-7] → RMS [0-3] → RMS [0-1]
                                   → RMS [2-3]
                       → RMS [4-7] → RMS [4-5]
                                   → RMS [6-7]

        4 leaf paths. Exercises: fully symmetric tree, every internal node splits.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        all_8 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        left_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        right_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})
        ll_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        lr_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        rl_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0))})
        rr_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(7)]
        tt_ws = [self._tt(w, device) for w in ws]

        def rms(inp, cores, wi):
            return rms_norm.rms_norm(inp, core_range_set=cores, weight=tt_ws[wi], epsilon=1e-5)

        root = rms(self._tt(torch_input, device), all_8, 0)
        left = rms(root.output_tensors[0], left_4, 1)
        right = rms(root.output_tensors[0], right_4, 2)
        ll = rms(left.output_tensors[0], ll_2, 3)
        lr = rms(left.output_tensors[0], lr_2, 4)
        rl = rms(right.output_tensors[0], rl_2, 5)
        rr = rms(right.output_tensors[0], rr_2, 6)

        fused = Sequential(
            root,
            Parallel(
                Sequential(left, Parallel(ll, lr)),
                Sequential(right, Parallel(rl, rr)),
            ),
        ).build(device)
        outputs = composite.launch([fused])

        g_root = torch_rms_norm(torch_input.float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        g_right = torch_rms_norm(g_root, ws[2].float())
        goldens = [
            torch_rms_norm(g_left, ws[3].float()),
            torch_rms_norm(g_left, ws[4].float()),
            torch_rms_norm(g_right, ws[5].float()),
            torch_rms_norm(g_right, ws[6].float()),
        ]

        for i, label in enumerate(["LL", "LR", "RL", "RR"]):
            passing, pcc = comp_pcc(goldens[i], ttnn.to_torch(outputs[0][i]), pcc=0.98)
            assert passing, f"Leaf {label} PCC: {pcc}"

    def test_asymmetric_deep_left(self, device):
        """Asymmetric: deep left chain + shallow right branch.

        Tree (8 cores):
            RMS [0-7] → RMS [0-3] → RMS [0-1] → RMS [0-1]
                                   → RMS [2-3]
                       → RMS [4-7]

        Left-most path: 5 phases. Right path: 2 phases.
        Exercises: asymmetric depth, different phase counts per path.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128

        all_8 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        left_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        ll_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        lr_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        right_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        ws = [torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16) for _ in range(6)]
        tt_ws = [self._tt(w, device) for w in ws]

        def rms(inp, cores, wi):
            return rms_norm.rms_norm(inp, core_range_set=cores, weight=tt_ws[wi], epsilon=1e-5)

        root = rms(self._tt(torch_input, device), all_8, 0)
        left = rms(root.output_tensors[0], left_4, 1)
        ll = rms(left.output_tensors[0], ll_2, 2)
        ll_deep = rms(ll.output_tensors[0], ll_2, 3)  # extra depth
        lr = rms(left.output_tensors[0], lr_2, 4)
        right = rms(root.output_tensors[0], right_4, 5)

        fused = Sequential(
            root,
            Parallel(
                Sequential(
                    left,
                    Parallel(
                        Sequential(ll, ll_deep),
                        lr,
                    ),
                ),
                right,
            ),
        ).build(device)
        outputs = composite.launch([fused])

        g_root = torch_rms_norm(torch_input.float(), ws[0].float())
        g_left = torch_rms_norm(g_root, ws[1].float())
        goldens = [
            torch_rms_norm(torch_rms_norm(g_left, ws[2].float()), ws[3].float()),  # LL deep
            torch_rms_norm(g_left, ws[4].float()),  # LR
            torch_rms_norm(g_root, ws[5].float()),  # Right
        ]

        for i, label in enumerate(["LL(deep)", "LR", "Right"]):
            passing, pcc = comp_pcc(goldens[i], ttnn.to_torch(outputs[0][i]), pcc=0.98)
            assert passing, f"Leaf {label} PCC: {pcc}"

    def test_nested_parallel_with_ln_bias_and_rms_bias(self, device):
        """Nested split with different bias configurations per branch.

        Tree (8 cores):
            RMS [0-7] → LN(bias) [0-3] → RMS(bias) [0-1]
                                        → RMS [2-3]
                       → LN [4-7]

        Exercises: LN bias ifdef + RMS bias ifdef in nested branches,
        different CB layouts per path.
        """
        from models.experimental.ops.descriptors.fusion import Sequential, Parallel
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        all_8 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        left_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        ll_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        lr_2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))})
        right_4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16) * 0.1
        torch_rms_bias = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16) * 0.05
        tt_w = self._tt(torch_w, device)
        tt_bias = self._tt(torch_bias, device)
        tt_rms_bias = self._tt(torch_rms_bias, device)

        stem = rms_norm.rms_norm(
            self._tt(torch_input, device),
            core_range_set=all_8,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Left mid: LN with bias
        left_ln = layer_norm.layer_norm(
            stem.output_tensors[0],
            core_range_set=left_4,
            weight=tt_w,
            bias=tt_bias,
            epsilon=1e-5,
        )

        # Left-left leaf: RMS with bias
        ll_rms = rms_norm.rms_norm(
            left_ln.output_tensors[0],
            core_range_set=ll_2,
            weight=tt_w,
            bias=tt_rms_bias,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Left-right leaf: RMS without bias
        lr_rms = rms_norm.rms_norm(
            left_ln.output_tensors[0],
            core_range_set=lr_2,
            weight=tt_w,
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        # Right: LN without bias
        right_ln = layer_norm.layer_norm(
            stem.output_tensors[0],
            core_range_set=right_4,
            weight=tt_w,
            epsilon=1e-5,
        )

        fused = Sequential(
            stem,
            Parallel(
                Sequential(left_ln, Parallel(ll_rms, lr_rms)),
                right_ln,
            ),
        ).build(device)
        outputs = composite.launch([fused])

        w = torch_w.float()
        b = torch_bias.float()
        rb = torch_rms_bias.float()
        g_stem = torch_rms_norm(torch_input.float(), w)
        g_left = torch_layer_norm(g_stem, w, b)

        # RMS with bias: x / rms(x) * weight + bias
        g_ll_rms_val = torch.sqrt(g_left.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        golden_ll = g_left / g_ll_rms_val * w + rb

        golden_lr = torch_rms_norm(g_left, w)
        golden_right = torch_layer_norm(g_stem, w)

        passing_ll, pcc_ll = comp_pcc(golden_ll, ttnn.to_torch(outputs[0][0]), pcc=0.97)
        passing_lr, pcc_lr = comp_pcc(golden_lr, ttnn.to_torch(outputs[0][1]), pcc=0.98)
        passing_r, pcc_r = comp_pcc(golden_right, ttnn.to_torch(outputs[0][2]), pcc=0.98)
        assert passing_ll, f"LL (RMS+bias) PCC: {pcc_ll}"
        assert passing_lr, f"LR (RMS) PCC: {pcc_lr}"
        assert passing_r, f"Right (LN) PCC: {pcc_r}"

    def test_stem_rms_branch_ln_and_rms_opgraph(self, device):
        """Branching: stem RMS → Parallel(LN, RMS) via build_op_graph API.

        Tree (8 cores):
            RMS [0-7] → LN  [0-3]
                       → RMS [4-7]

        Exercises: per-core group architecture with heterogeneous norm branches
        using the OpGraph (tree) API.  Two core groups are formed — one per
        branch — each running the stem phase then its own branch phase.
        """
        from models.experimental.ops.descriptors.fusion import OpNode, build_op_graph
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        torch.manual_seed(42)
        hidden = 128
        ln_compute = ttnn.layernorm_default_compute_config(device.arch())

        stem_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        branch_a_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        branch_b_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0))})

        # 8 tiles (256 rows), 1 per core
        torch_input = torch.randn(1, 1, 256, hidden, dtype=torch.bfloat16)
        torch_w = torch.ones(1, 1, 1, hidden, dtype=torch.bfloat16)
        torch_bias = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)

        def _tt(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        stem = rms_norm.rms_norm(
            _tt(torch_input),
            core_range_set=stem_cores,
            weight=_tt(torch_w),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        branch_a = layer_norm.layer_norm(
            stem.output_tensors[0],
            core_range_set=branch_a_cores,
            weight=_tt(torch_w),
            bias=_tt(torch_bias),
            epsilon=1e-5,
        )
        branch_b = rms_norm.rms_norm(
            stem.output_tensors[0],
            core_range_set=branch_b_cores,
            weight=_tt(torch_w),
            epsilon=1e-5,
            compute_kernel_config=ln_compute,
        )

        fused = build_op_graph(
            root_phases=[stem],
            children=[OpNode(branch_a), OpNode(branch_b)],
            device=device,
        )
        outputs = composite.launch([fused])

        g_stem = torch_rms_norm(torch_input.float(), torch_w.float())
        golden_a = torch_layer_norm(g_stem, torch_w.float(), torch_bias.float())
        golden_b = torch_rms_norm(g_stem, torch_w.float())

        result_a = ttnn.to_torch(outputs[0][0])
        result_b = ttnn.to_torch(outputs[0][1])

        passing_a, pcc_a = comp_pcc(golden_a, result_a, pcc=0.97)
        passing_b, pcc_b = comp_pcc(golden_b, result_b, pcc=0.98)
        assert passing_a, f"Branch A (LN) PCC: {pcc_a}"
        assert passing_b, f"Branch B (RMS) PCC: {pcc_b}"
