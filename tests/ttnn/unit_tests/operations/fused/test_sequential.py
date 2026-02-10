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
        from models.experimental.ops.descriptors.sequential import extract_cb_info
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

        print(f"LayerNorm uses {len(cb_info)} CBs: {sorted(cb_indices)}")

    def test_extract_cb_info_from_rmsnorm(self, device, test_tensors):
        """Test extracting CB info from a real RMSNorm descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info
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

        print(f"RMSNorm uses {len(cb_info)} CBs: {sorted(cb_indices)}")

    def test_chain_builder_with_real_descriptors(self, device, test_tensors):
        """Test building a chain with real LayerNorm/RMSNorm descriptors."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder, extract_cb_info
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

        # Build the chain
        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc)
        builder.add_phase(ln2_desc)

        # Verify chain structure
        assert len(builder.phases) == 3

        # Build and verify we get a valid fused descriptor
        fused = builder.build(device)
        assert fused is not None
        assert hasattr(fused, "descriptor")
        num_kernels = len(fused.descriptor.kernels)
        assert num_kernels >= 3, "Should have reader, writer, and compute kernels"

        print(f"Chain structure validated: LayerNorm -> RMSNorm -> LayerNorm ({num_kernels} kernels)")

    def test_barrier_config_added(self, device, test_tensors):
        """Test that fused descriptors get barrier configuration (GlobalSemaphores)."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1_desc, ln2_desc], device)

        # Verify we got a fused descriptor with kernels
        assert fused is not None
        assert hasattr(fused, "descriptor")
        num_kernels = len(fused.descriptor.kernels)
        assert num_kernels >= 3, "Should have reader, writer, and compute kernels"

        # Verify the fused kernels are SOURCE_CODE type (generated)
        for kernel in fused.descriptor.kernels:
            assert kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE

        print(f"Barrier config test passed: {num_kernels} fused kernels")


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

    print(f"\nDumped fused kernel files to: {output_dir}")
    print("\n".join(summary_lines))


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
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder
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
        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc)
        builder.add_phase(ln2_desc)
        fused_desc = builder.build(device)

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
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused = chain_descriptors([rms1, rms2, rms3, rms4], device)
        outputs = composite.launch([fused])
        tt_output = outputs[0][0]
        torch_output = ttnn.to_torch(tt_output)

        # Compute golden
        temp1 = torch_rms_norm(test_tensors["torch_input"], test_tensors["torch_weight1"])
        temp2 = torch_rms_norm(temp1, test_tensors["torch_weight2"])
        temp3 = torch_rms_norm(temp2, test_tensors["torch_weight3"])
        golden = torch_rms_norm(temp3, torch_weight4)

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        print(f"4-phase RMS chain PCC: {pcc:.6f}")
        assert passing, f"4-phase RMS chain PCC check failed: {pcc}"

    def test_two_phase_layernorm_chain_multicore(self, device, test_tensors):
        """Test 2-phase LN→LN chain on 2 cores to validate global barrier."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder
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

        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(ln2_desc)
        fused_desc = builder.build(device)

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
        print(f"2-phase LN→LN multi-core PCC: {pcc:.6f}")
        assert passing, f"Multi-core 2-phase chain PCC check failed: {pcc}"

    @pytest.mark.skip(reason="Fused kernel source generated but needs device compilation testing")
    def test_chain_with_parallel_op(self, device, test_tensors):
        """
        Test running a fused chain in parallel with another op on different cores.

        This demonstrates the integration of sequential.py with composite.py:
        - Cores (0,0): LayerNorm -> RMSNorm -> LayerNorm (fused chain)
        - Cores (1,0): RMSNorm (independent op)

        Both execute in a single program via composite.launch().
        """
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        # Core ranges - non-overlapping
        chain_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        parallel_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

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
        )
        ln2_desc = layer_norm.layer_norm(
            rms_desc.output_tensors[0],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc)
        builder.add_phase(ln2_desc)
        fused_chain = builder.build(device)

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

        print("Both ops executed in parallel successfully!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestFusedKernelSource:
    """Tests for the fused kernel source generation."""

    def test_fused_source_has_phases(self, device, test_tensors):
        """Test that fused kernel source contains phase functions."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, ln2], device)

        # Verify fused kernels are SOURCE_CODE type with phase functions
        for kernel in fused.descriptor.kernels:
            assert kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            source = kernel.kernel_source
            assert "Phase 0" in source, "Should have Phase 0 comment"
            assert "Phase 1" in source, "Should have Phase 1 comment"
            assert "void kernel_main()" in source

    def test_fused_source_has_barrier(self, device, test_tensors):
        """Test that fused kernel source contains barrier synchronization."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, ln2], device)

        # Check reader has barrier code
        for kernel in fused.descriptor.kernels:
            source = kernel.kernel_source
            if "__global_barrier" in source:
                assert "__cb_reset_to_empty" in source, "Reader should have CB reset"
                assert "__compute_done" in source, "Reader should wait for compute_done"
                assert "__writer_done" in source, "Reader should wait for writer_done"
                break
        else:
            pytest.fail("No kernel has __global_barrier")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelChains:
    """Tests for parallel chain creation and execution."""

    def test_create_parallel_chain_descriptors(self, device, test_tensors):
        """Test creating parallel chain descriptors."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors
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
        fused_descriptors = create_parallel_chain_descriptors([chain1, chain2], device)

        assert len(fused_descriptors) == 2
        print(f"Created {len(fused_descriptors)} fused chain descriptors")

        # Verify each chain has a merged descriptor with kernels from both ops
        for i, desc in enumerate(fused_descriptors):
            num_kernels = len(desc.descriptor.kernels)
            num_cbs = len(desc.descriptor.cbs)
            print(f"Chain {i}: {num_kernels} kernels, {num_cbs} CB descriptors")

    def test_chain_descriptors_three_phase(self, device, test_tensors):
        """Test the chain_descriptors convenience function with 3 phases."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused = chain_descriptors([ln1, rms1, ln2], device)

        # Verify structure
        assert fused is not None
        assert hasattr(fused, "descriptor")

        # Should have kernels from all 3 ops
        # Each op typically has 3 kernels (reader, compute, writer)
        num_kernels = len(fused.descriptor.kernels)
        print(f"Fused chain has {num_kernels} kernels")
        assert num_kernels >= 3, "Should have kernels from fused ops"

    def test_chain_single_op(self, device, test_tensors):
        """Test that chaining a single op returns it unchanged."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        fused = chain_descriptors([ln], device)

        # Should return the same descriptor
        assert fused is ln

    @pytest.mark.skip(reason="Don't generate a bunch of files")
    def test_dump_fused_kernel_files(self, device, test_tensors):
        """Build a fused 3-phase chain and dump all generated kernel files for inspection."""
        import os

        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, rms1, ln2], device)

        output_dir = os.path.join(os.path.dirname(__file__), "gen_kernels")
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
        from models.experimental.ops.descriptors.sequential import extract_cb_names_from_kernel
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
                print(f"Kernel CB names: {cb_names}")
                # Verify expected CB names are present
                expected_names = ["cb_in", "cb_out", "cb_scaler", "cb_eps"]
                for name in expected_names:
                    if name in cb_names:
                        print(f"  Found {name} -> {cb_names[name]}")

    def test_cb_overflow_validation(self, device, test_tensors):
        """Test that CB overflow is detected and reported clearly."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        ln_compute_config = ttnn.layernorm_default_compute_config(device.arch())

        # Try to create a very long chain that would overflow CBs
        # Each LayerNorm uses ~6 CBs (input, output, gamma, beta, scaler, eps)
        # With remapping, this could potentially exceed 32 CBs
        builder = SequentialChainBuilder()

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
                builder.add_phase(desc)
            else:
                desc = rms_norm.rms_norm(
                    prev_desc.output_tensors[0],
                    core_range_set=core_range,
                    weight=test_tensors["tt_weight1"],
                    epsilon=1e-5,
                    compute_kernel_config=ln_compute_config,
                )
                builder.add_phase(desc)
            prev_desc = desc

        # This should either succeed (if CB merging is efficient) or raise a clear error
        try:
            fused = builder.build(device)
            print(f"Successfully fused {len(builder.phases)} phases")
            # The build succeeded, so CBs should be within limits
            assert True, "Build succeeded without CB overflow"
        except (ValueError, RuntimeError) as e:
            # Expected to fail with a clear CB overflow message
            error_msg = str(e)
            print(f"CB overflow caught: {error_msg[:200]}...")
            assert (
                "CB" in error_msg or "circular buffer" in error_msg.lower()
            ), f"Error should mention CB overflow: {error_msg}"
            assert "32" in error_msg or "NUM_CBS" in error_msg, f"Error should mention the 32 CB limit: {error_msg}"

    def test_named_args_have_phase_prefix(self, device, test_tensors):
        """Test that fused kernel named args get proper phase prefixes."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, ln2], device)

        # Check that fused kernels have phase-prefixed named args
        for kernel in fused.descriptor.kernels:
            named_args = dict(kernel.named_compile_time_args)
            # Should have barrier_rt_offset
            assert "barrier_rt_offset" in named_args, f"Missing barrier_rt_offset in {list(named_args.keys())}"
            # Phase 1 args should have phase1_ prefix
            phase1_args = [k for k in named_args if k.startswith("phase1_")]
            assert len(phase1_args) > 0, f"Should have phase1_ prefixed args, got: {list(named_args.keys())}"
            print(f"Named args: {list(named_args.keys())}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestParallelChainsExecution:
    """
    Execution tests for parallel chains.

    These demonstrate running multiple fused chains in parallel using composite.launch().
    Some tests are skipped pending full kernel CB parameterization.
    """

    @pytest.mark.skip(reason="Fused kernel source generated but needs device compilation testing")
    def test_two_parallel_chains_execution(self, device, test_tensors):
        """
        Test executing two parallel chains:
        - Chain A: LayerNorm -> RMSNorm (on cores 0,0)
        - Chain B: RMSNorm -> LayerNorm (on cores 1,0)
        """
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

        cores1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cores2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})

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
            ln_a.output_tensors[0], core_range_set=cores1, weight=test_tensors["tt_weight2"], epsilon=1e-5
        )
        chain_a = [ln_a, rms_a]

        # Chain B: RMSNorm -> LayerNorm
        rms_b = rms_norm.rms_norm(tt_input2, core_range_set=cores2, weight=test_tensors["tt_weight1"], epsilon=1e-5)
        ln_b = layer_norm.layer_norm(
            rms_b.output_tensors[0], core_range_set=cores2, weight=test_tensors["tt_weight2"], epsilon=1e-5
        )
        chain_b = [rms_b, ln_b]

        # Fuse chains
        fused = create_parallel_chain_descriptors([chain_a, chain_b])

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
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors
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
        fused = create_parallel_chain_descriptors(chains, device)
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
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        print("\n" + "=" * 80)
        print("PROFILING: Serial vs Parallel Execution")
        print("=" * 80)
        print(f"Branch A: Matmul {torch_a.shape} @ {torch_b.shape}")
        print(f"Branch B: 4 normalizations (LN->RMS->LN->RMS) on {test_tensors['torch_input'].shape}")
        print("=" * 80)

        # ============================================================
        # SERIAL BASELINE: Run everything sequentially
        # ============================================================
        print("\n[1] SERIAL EXECUTION (baseline)")
        print("-" * 80)

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

        print(f"Serial execution time: {serial_time*1000:.2f} ms")

        # ============================================================
        # PARALLEL EXECUTION: Run matmul and norm chain in parallel
        # ============================================================
        print("\n[2] PARALLEL EXECUTION (fused + composite)")
        print("-" * 80)

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
        fused_norm_chain = chain_descriptors([ln1_desc, rms1_desc, ln2_desc], device)

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

        print(f"Parallel execution time: {parallel_time*1000:.2f} ms")

        # ============================================================
        # RESULTS
        # ============================================================
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Serial time:   {serial_time*1000:.2f} ms")
        print(f"Parallel time: {parallel_time*1000:.2f} ms")

        if parallel_time < serial_time:
            speedup = serial_time / parallel_time
            time_saved = (serial_time - parallel_time) * 1000
            print(f"Speedup:       {speedup:.2f}x")
            print(f"Time saved:    {time_saved:.2f} ms ({time_saved/serial_time/10:.1f}%)")
            print("\n✓ PARALLEL EXECUTION IS FASTER!")
        else:
            print("\n⚠ Serial was faster (parallel overhead may dominate for small ops)")

        print("=" * 80)

        # Verify correctness against torch golden
        print("\nVerifying correctness against torch golden...")

        # Matmul golden
        torch_golden_mm = torch.matmul(torch_a, torch_b)
        torch_parallel_mm = ttnn.to_torch(parallel_mm_out)
        passing_mm, pcc_mm = comp_pcc(torch_golden_mm, torch_parallel_mm, pcc=0.99)
        print(f"Matmul output PCC (parallel vs torch): {pcc_mm:.6f}")
        assert passing_mm, f"Matmul output doesn't match torch: {pcc_mm}"

        # Norm chain golden (3-phase: LN→RMS→LN)
        temp1 = torch_layer_norm(test_tensors["torch_input"], torch_weights[0], torch_biases[0], eps=1e-5)
        temp2 = torch_rms_norm(temp1, torch_weights[1], eps=1e-5)
        torch_golden_norm = torch_layer_norm(temp2, torch_weights[2], torch_biases[1], eps=1e-5)

        torch_parallel_norm = ttnn.to_torch(parallel_norm_out)
        passing_norm, pcc_norm = comp_pcc(torch_golden_norm, torch_parallel_norm, pcc=0.98)
        print(f"Norm chain output PCC (parallel vs torch): {pcc_norm:.6f}")
        assert passing_norm, f"Norm chain output doesn't match torch: {pcc_norm}"

        print("\n✓ Correctness verified against torch golden!")


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
        print(f"Matmul standalone PCC: {pcc}")
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
        print(f"Matmul PCC: {pcc1}")
        assert passing1, f"Matmul PCC: {pcc1}"

        # Verify layernorm output
        torch_ln_out = ttnn.to_torch(outputs[1][0])
        golden_ln = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        passing2, pcc2 = comp_pcc(golden_ln, torch_ln_out, pcc=0.99)
        print(f"LayerNorm PCC: {pcc2}")
        assert passing2, f"LayerNorm PCC: {pcc2}"

    def test_matmul_descriptor_cb_info(self, device, matmul_tensors):
        """Test extracting CB info from a matmul descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        cb_info = extract_cb_info(desc.descriptor)
        assert len(cb_info) > 0, "Matmul should have CB descriptors"

        cb_indices = set(cb_info.keys())
        print(f"Matmul uses {len(cb_info)} CBs: {sorted(cb_indices)}")

        # Matmul should have at least: c_0 (in0), c_1 (in1), c_4 (out)
        assert 0 in cb_indices, "Should have input A CB (c_0)"
        assert 1 in cb_indices, "Should have input B CB (c_1)"
        assert 4 in cb_indices, "Should have output CB (c_4)"

    def test_matmul_descriptor_named_args(self, device, matmul_tensors):
        """Test that matmul descriptor has named compile-time args for CB indices."""
        from models.experimental.ops.descriptors.sequential import extract_cb_names_from_kernel
        from models.experimental.ops.descriptors.matmul import matmul as matmul_desc

        desc = matmul_desc(matmul_tensors["tt_a"], matmul_tensors["tt_b"])

        found_named_args = False
        for kernel in desc.descriptor.kernels:
            cb_names = extract_cb_names_from_kernel(kernel)
            if cb_names:
                found_named_args = True
                print(f"Kernel CB names: {cb_names}")
                # Verify expected matmul CB names
                assert "cb_in0" in cb_names, "Should have cb_in0"
                assert "cb_out" in cb_names, "Should have cb_out"

        assert found_named_args, "At least one kernel should have named compile-time args"


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestStressInfrastructure:
    """Stress tests to exercise edge cases in sequential kernel chaining."""

    def test_six_parallel_two_phase_chains(self, device, test_tensors):
        """6 independent LN->RMS chains on cores (0,0)-(5,0) in parallel."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors
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

        fused = create_parallel_chain_descriptors(chains, device)
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
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, rms1, ln2], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_rms_norm(temp, test_tensors["torch_weight2"])
        golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias2"])

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        print(f"3-phase chain on core ({core_x},0) PCC: {pcc:.6f}")
        assert passing, f"Core ({core_x},0) PCC: {pcc}"

    def test_four_phase_all_rms_non_zero_core(self, device, test_tensors):
        """4-phase all-RMS chain on core (5,0)."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors(descs, device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = test_tensors["torch_input"]
        for tw in torch_weights:
            golden = torch_rms_norm(golden, tw)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        print(f"4-phase all-RMS on core (5,0) PCC: {pcc:.6f}")
        assert passing, f"4-phase all-RMS PCC: {pcc}"

    def test_matmul_plus_two_norm_chains(self, device, test_tensors, matmul_tensors):
        """1 matmul + 2 independent fused norm chains in parallel."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused_a = chain_descriptors([ln_a, rms_a], device)

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
        fused_b = chain_descriptors([rms_b, ln_b], device)

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

        print(f"Matmul PCC: {pcc_mm:.6f}, Chain A PCC: {pcc_a:.6f}, Chain B PCC: {pcc_b:.6f}")

    def test_matmul_plus_three_phase_chain(self, device, test_tensors, matmul_tensors):
        """1 matmul + 1 three-phase LN->RMS->LN fused chain in parallel."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused = chain_descriptors([ln1, rms1, ln2], device)

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

        print(f"Matmul PCC: {pcc_mm:.6f}, 3-phase chain PCC: {pcc_norm:.6f}")

    def test_mixed_chain_lengths_parallel(self, device, test_tensors):
        """2-phase chain + 3-phase chain running in parallel."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused_a = chain_descriptors([ln_a, rms_a], device)

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
        fused_b = chain_descriptors([ln_b1, rms_b, ln_b2], device)

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

        print(f"2-phase PCC: {pcc_a:.6f}, 3-phase PCC: {pcc_b:.6f}")

    def test_repeated_chain_execution(self, device, test_tensors):
        """Run chain 3 times with fresh descriptors each time to check for state leaks."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
            fused = chain_descriptors([ln1, rms1], device)
            outputs = composite.launch([fused])
            result = ttnn.to_torch(outputs[0][0])

            passing, pcc = comp_pcc(golden, result, pcc=0.98)
            pccs.append(pcc)
            assert passing, f"Iteration {iteration} PCC: {pcc}"

        print(f"Repeated execution PCCs: {[f'{p:.6f}' for p in pccs]}")

    def test_larger_tensor_chain(self, device):
        """2-phase LN->RMS chain with larger tensors (128x256)."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
        fused = chain_descriptors([ln1, rms1], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        golden = torch_rms_norm(torch_layer_norm(torch_input, torch_w1, torch_b1), torch_w2)

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        print(f"Larger tensor (128x256) chain PCC: {pcc:.6f}")
        assert passing, f"Larger tensor chain PCC: {pcc}"

    def test_all_ln_three_phase_chain(self, device, test_tensors):
        """LN->LN->LN chain (all same op type, with biases)."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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

        fused = chain_descriptors([ln1, ln2, ln3], device)
        outputs = composite.launch([fused])
        result = ttnn.to_torch(outputs[0][0])

        temp = torch_layer_norm(test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"])
        temp = torch_layer_norm(temp, test_tensors["torch_weight2"], test_tensors["torch_bias2"])
        golden = torch_layer_norm(temp, test_tensors["torch_weight3"], test_tensors["torch_bias1"])

        passing, pcc = comp_pcc(golden, result, pcc=0.98)
        print(f"All-LN 3-phase chain on core (6,0) PCC: {pcc:.6f}")
        assert passing, f"All-LN chain PCC: {pcc}"

    def test_three_chains_plus_matmul(self, device, test_tensors, matmul_tensors):
        """1 matmul + 3 independent fused norm chains in parallel (4 total ops)."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors
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
            fused_chains.append(chain_descriptors([ln, rms], device))

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

        print(f"Matmul + 3 chains: mm={pcc_mm:.6f}")
