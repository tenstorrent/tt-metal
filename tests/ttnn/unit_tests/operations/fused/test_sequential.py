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
            test_tensors["tt_input"],  # Placeholder - will be replaced by chain
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        ln2_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],  # Placeholder - will be replaced by chain
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        # Build the chain
        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc, input_cb=0, output_cb=16)
        builder.add_phase(rms_desc, input_from_previous=True, input_cb=0, output_cb=16)
        builder.add_phase(ln2_desc, input_from_previous=True, input_cb=0, output_cb=16)

        # Verify chain structure
        assert len(builder.phases) == 3
        assert len(builder.connections) == 2  # ln1->rms, rms->ln2

        # Verify connections
        conn1 = builder.connections[0]
        assert conn1.source_phase_idx == 0
        assert conn1.target_phase_idx == 1
        assert conn1.source_output_cb == 16
        assert conn1.target_input_cb == 0

        conn2 = builder.connections[1]
        assert conn2.source_phase_idx == 1
        assert conn2.target_phase_idx == 2

        print("Chain structure validated: LayerNorm -> RMSNorm -> LayerNorm")

    def test_cb_remapping_for_chain(self, device, test_tensors):
        """Test CB remapping logic for a 3-phase chain."""
        from models.experimental.ops.descriptors.sequential import (
            SequentialChainBuilder,
            CBRemapper,
            PhaseInfo,
            extract_cb_info,
        )
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        # Create descriptors
        ln1_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        rms_desc = rms_norm.rms_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )

        ln2_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        # Extract CB info from each
        ln1_cbs = extract_cb_info(ln1_desc.descriptor)
        rms_cbs = extract_cb_info(rms_desc.descriptor)
        ln2_cbs = extract_cb_info(ln2_desc.descriptor)

        print(f"Phase 0 (LayerNorm) CBs: {sorted(ln1_cbs.keys())}")
        print(f"Phase 1 (RMSNorm) CBs: {sorted(rms_cbs.keys())}")
        print(f"Phase 2 (LayerNorm) CBs: {sorted(ln2_cbs.keys())}")

        # Simulate CB remapping
        remapper = CBRemapper()

        # Phase 0
        phase0 = PhaseInfo(
            phase_idx=0,
            op_descriptor=ln1_desc,
            cb_info=ln1_cbs,
            input_cb_indices={0},
            output_cb_indices={16},
        )
        remap0 = remapper.allocate_for_phase(phase0)
        remapper.finish_phase(remap0, output_cb_original=16)

        print(f"Phase 0 remapping: {remap0}")

        # Phase 1 - chains from phase 0
        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=rms_desc,
            cb_info=rms_cbs,
            input_cb_indices={0},
            output_cb_indices={16},
        )
        remap1 = remapper.allocate_for_phase(
            phase1,
            chain_from_previous=True,
            previous_output_cb=remap0[16],
            target_input_cb=0,
        )
        remapper.finish_phase(remap1, output_cb_original=16)

        print(f"Phase 1 remapping: {remap1}")
        # Verify chaining: phase 1's input (originally 0) should map to phase 0's output
        assert remap1[0] == remap0[16], "Phase 1 input should chain from phase 0 output"

        # Phase 2 - chains from phase 1
        phase2 = PhaseInfo(
            phase_idx=2,
            op_descriptor=ln2_desc,
            cb_info=ln2_cbs,
            input_cb_indices={0},
            output_cb_indices={16},
        )
        remap2 = remapper.allocate_for_phase(
            phase2,
            chain_from_previous=True,
            previous_output_cb=remap1[16],
            target_input_cb=0,
        )

        print(f"Phase 2 remapping: {remap2}")
        assert remap2[0] == remap1[16], "Phase 2 input should chain from phase 1 output"

        total_cbs = remapper.get_total_cbs_used()
        print(f"Total unique CBs used across all phases: {total_cbs}")
        assert total_cbs <= 32, "Should not exceed 32 CBs"


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
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        ln2_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        # Build chain - CB indices are automatically remapped via named_compile_time_args
        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc, input_from_previous=True)
        builder.add_phase(ln2_desc, input_from_previous=True)
        fused_desc = builder.build()

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
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        ln2_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=chain_cores,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc, input_from_previous=True)
        builder.add_phase(ln2_desc, input_from_previous=True)
        fused_chain = builder.build()

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
class TestFusedKernelGenerator:
    """Tests for the FusedKernelGenerator class."""

    def test_generator_single_phase(self, device):  # noqa: ARG002 - device needed for fixture
        """Test generator with single phase."""
        _ = device  # Mark as used for parameterization
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()

        # Simple test kernel source
        compute_source = """
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "compute_kernel_api.h"

void kernel_main() {
    // Read from CB 0, write to CB 16
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    // ... compute operations ...
}
"""
        generator.add_phase(
            compute_source=compute_source,
            cb_remapping={0: 0, 16: 16},
            is_first=True,
            is_last=True,
        )

        fused = generator.generate_fused_compute()

        # Should contain the original source (with remapping applied)
        assert "void kernel_main()" in fused
        assert "cb_in" in fused
        assert "cb_out" in fused

    def test_generator_cb_remapping(self, device):  # noqa: ARG002
        """Test that CB remapping is applied correctly to source."""
        _ = device  # Mark as used for parameterization
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()

        compute_source = """
#include "compute_kernel_api.h"
void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
}
"""
        # Remap CB 0 -> 5, CB 16 -> 20
        generator.add_phase(
            compute_source=compute_source,
            cb_remapping={0: 5, 16: 20},
            is_first=True,
            is_last=True,
        )

        fused = generator.generate_fused_compute()

        # Check that remapping was applied
        assert "tt::CBIndex::c_5" in fused
        assert "tt::CBIndex::c_20" in fused
        # Original indices should be replaced
        assert "tt::CBIndex::c_0" not in fused or "tt::CBIndex::c_5" in fused
        assert "tt::CBIndex::c_16" not in fused or "tt::CBIndex::c_20" in fused

    def test_generator_multi_phase_includes(self, device):  # noqa: ARG002
        """Test that fused kernel collects includes from all phases."""
        _ = device  # Mark as used for parameterization
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()

        phase1 = """
#include "compute_kernel_api.h"
#include "special_api_1.h"
void kernel_main() { }
"""
        phase2 = """
#include "compute_kernel_api.h"
#include "special_api_2.h"
void kernel_main() { }
"""
        generator.add_phase(compute_source=phase1, is_first=True)
        generator.add_phase(compute_source=phase2, is_last=True)

        fused = generator.generate_fused_compute()

        # Should have both special includes (deduplicated)
        assert "#include" in fused
        # The fused kernel should reference both phases
        assert "Phase 0" in fused
        assert "Phase 1" in fused

    def test_generator_noop_reader_writer(self, device):  # noqa: ARG002
        """Test no-op reader/writer generation."""
        _ = device  # Mark as used for parameterization
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() {}",
            is_first=True,
            is_last=True,
        )

        # No reader/writer source provided - should generate no-ops
        reader = generator.generate_fused_reader()
        writer = generator.generate_fused_writer()

        assert "void kernel_main()" in reader
        assert "void kernel_main()" in writer
        assert "No-op" in reader
        assert "No-op" in writer


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
            test_tensors["tt_input"],
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
            tt_input2,
            core_range_set=cores2,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
        )
        chain2 = [rms2, ln2]

        # Create parallel chain descriptors
        fused_descriptors = create_parallel_chain_descriptors([chain1, chain2])

        assert len(fused_descriptors) == 2
        print(f"Created {len(fused_descriptors)} fused chain descriptors")

        # Verify each chain has a merged descriptor with kernels from both ops
        for i, desc in enumerate(fused_descriptors):
            num_kernels = len(desc.descriptor.kernels)
            num_cbs = len(desc.descriptor.cbs)
            print(f"Chain {i}: {num_kernels} kernels, {num_cbs} CB descriptors")

    def test_fuse_layernorm_chain(self, device, test_tensors):
        """Test the fuse_layernorm_chain convenience function."""
        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
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
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            epsilon=1e-5,
        )

        # Fuse the chain
        fused = fuse_layernorm_chain([ln1, rms1, ln2])

        # Verify structure
        assert fused is not None
        assert hasattr(fused, "descriptor")

        # Should have kernels from all 3 ops
        # Each op typically has 3 kernels (reader, compute, writer)
        num_kernels = len(fused.descriptor.kernels)
        print(f"Fused chain has {num_kernels} kernels")
        assert num_kernels >= 3, "Should have kernels from fused ops"

    def test_fuse_single_op(self, device, test_tensors):
        """Test that fusing a single op returns it unchanged."""
        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        fused = fuse_layernorm_chain([ln])

        # Should return the same descriptor
        assert fused is ln

    def test_dump_fused_kernel_files(self, device, test_tensors):
        """Build a fused 3-phase chain and dump all generated kernel files for inspection."""
        import os

        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
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
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = fuse_layernorm_chain([ln1, rms1, ln2])

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

    def test_cb_remapping_preserves_named_args(self, device, test_tensors):
        """Test that CB remapping works with named compile-time args."""
        from models.experimental.ops.descriptors.sequential import (
            remap_kernel_cb_indices,
            extract_cb_names_from_kernel,
        )
        from models.experimental.ops.descriptors.normalization import layer_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
        )

        # Remap CB indices
        cb_remapping = {0: 10, 16: 26}

        for kernel in desc.descriptor.kernels:
            original_names = extract_cb_names_from_kernel(kernel)
            if not original_names:
                continue

            print(f"Original CB names: {original_names}")

            # Apply remapping
            remapped_kernel = remap_kernel_cb_indices(
                kernel,
                cb_remapping,
            )

            # Extract names from remapped kernel
            remapped_names = extract_cb_names_from_kernel(remapped_kernel)
            print(f"Remapped CB names: {remapped_names}")

            # Verify remapping was applied
            for name, original_val in original_names.items():
                if original_val in cb_remapping:
                    expected_val = cb_remapping[original_val]
                    if name in remapped_names:
                        assert (
                            remapped_names[name] == expected_val
                        ), f"Expected {name} to be remapped from {original_val} to {expected_val}"


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
        chain_a = [
            layer_norm.layer_norm(
                test_tensors["tt_input"], core_range_set=cores1, weight=test_tensors["tt_weight1"], epsilon=1e-5
            ),
            rms_norm.rms_norm(
                test_tensors["tt_input"], core_range_set=cores1, weight=test_tensors["tt_weight2"], epsilon=1e-5
            ),
        ]

        # Chain B: RMSNorm -> LayerNorm
        chain_b = [
            rms_norm.rms_norm(tt_input2, core_range_set=cores2, weight=test_tensors["tt_weight1"], epsilon=1e-5),
            layer_norm.layer_norm(tt_input2, core_range_set=cores2, weight=test_tensors["tt_weight2"], epsilon=1e-5),
        ]

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

    @pytest.mark.skip(reason="Fused kernel source generated but needs device compilation testing")
    def test_four_parallel_chains(self, device, test_tensors):
        """
        Test running 4 parallel chains on 4 different cores.
        Each chain is: LayerNorm -> RMSNorm
        """
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

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

        # Create 4 chains
        chains = []
        for i in range(4):
            chain = [
                layer_norm.layer_norm(
                    tt_inputs[i], core_range_set=core_ranges[i], weight=test_tensors["tt_weight1"], epsilon=1e-5
                ),
                rms_norm.rms_norm(
                    tt_inputs[i], core_range_set=core_ranges[i], weight=test_tensors["tt_weight2"], epsilon=1e-5
                ),
            ]
            chains.append(chain)

        # Fuse and execute
        fused = create_parallel_chain_descriptors(chains)
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
class TestSequentialDebugHang:
    """
    Systematic debugging tests for the fused kernel device hang.

    These tests progressively increase complexity to isolate the root cause.
    The hang was caused by Phase 1 (RMSNorm) compiling as LayerNorm due to
    the RMSNORM preprocessor define not being resolved per-phase.
    """

    def test_debug_ln_ln_2phase(self, device, test_tensors):
        """
        Step 1: 2-phase LayerNorm -> LayerNorm chain.
        No RMSNORM involved — tests basic fusion infrastructure.
        """
        import os
        import signal

        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln1 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        ln2 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = fuse_layernorm_chain([ln1, ln2])

        # Dump for inspection
        output_dir = os.path.join(os.path.dirname(__file__), "gen_kernels")
        dump_fused_kernels(fused, output_dir, label="debug_ln_ln")

        # Launch with timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError("Device operation timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            outputs = composite.launch([fused])
            tt_output = outputs[0][0]
            torch_output = ttnn.to_torch(tt_output)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            pytest.fail("HANG DETECTED in LN->LN 2-phase chain")
        signal.signal(signal.SIGALRM, old_handler)

        # Golden: LayerNorm(LayerNorm(x)) — using ones weights, so effectively just normalization
        temp = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        golden = torch_layer_norm(temp, test_tensors["torch_weight2"], test_tensors["torch_bias2"], eps=1e-5)

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        print(f"Fused LN->LN PCC vs golden: {pcc}")

        # Also compare against unfused sequential: run two separate LN on device
        from models.experimental.ops.descriptors.normalization import layer_norm as ln_module

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        # Phase 0 alone
        single_ln = ln_module.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )
        single_out = composite.launch([single_ln])
        phase0_output = ttnn.to_torch(single_out[0][0])
        _, pcc_phase0 = comp_pcc(temp, phase0_output, pcc=0.99)
        print(f"Unfused Phase 0 PCC vs golden phase0: {pcc_phase0}")

        # Phase 1 (unfused, using Phase 0 output as input)
        tt_phase0_out = single_out[0][0]
        second_ln = ln_module.layer_norm(
            tt_phase0_out,
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )
        second_out = composite.launch([second_ln])
        unfused_sequential_output = ttnn.to_torch(second_out[0][0])
        _, pcc_unfused = comp_pcc(golden, unfused_sequential_output, pcc=0.99)
        print(f"Unfused sequential LN->LN PCC vs golden: {pcc_unfused}")

        # Compare fused vs unfused sequential
        _, pcc_fused_vs_unfused = comp_pcc(unfused_sequential_output, torch_output, pcc=0.99)
        print(f"Fused vs unfused sequential PCC: {pcc_fused_vs_unfused}")

        # Also compare fused output vs single LN output (is Phase 1 doing anything?)
        _, pcc_fused_vs_single = comp_pcc(phase0_output, torch_output, pcc=0.99)
        print(f"Fused output vs single LN (Phase 0 only) PCC: {pcc_fused_vs_single}")

        assert passing, f"LN->LN PCC check failed: {pcc}"

    def test_debug_ln_rms_2phase(self, device, test_tensors):
        """
        Step 2: 2-phase LayerNorm -> RMSNorm chain.
        This was the minimal hang case before the ifdef resolution fix.
        """
        import os
        import signal

        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

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
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )

        fused = fuse_layernorm_chain([ln1, rms1])

        # Dump for inspection
        output_dir = os.path.join(os.path.dirname(__file__), "gen_kernels")
        dump_fused_kernels(fused, output_dir, label="debug_ln_rms")

        # Verify the fix: Phase 1 source should NOT contain "#ifdef RMSNORM" or
        # "cb_reserve_back(cb_xmm, total_buffer_size)" in its body
        compute_path = os.path.join(output_dir, "debug_ln_rms_kernel_2_compute.cpp")
        with open(compute_path) as f:
            compute_src = f.read()

        # After fix, Phase 1 should have the RMSNorm code path where cb_xmm = cb_in
        assert "cb_xmm = cb_in" in compute_src, "Phase 1 (RMSNorm) should have cb_xmm = cb_in after ifdef resolution"

        # Launch with timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError("Device operation timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            outputs = composite.launch([fused])
            tt_output = outputs[0][0]
            torch_output = ttnn.to_torch(tt_output)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            pytest.fail("HANG DETECTED in LN->RMS 2-phase chain")
        signal.signal(signal.SIGALRM, old_handler)

        # Golden: RMSNorm(LayerNorm(x)) — using ones weights
        temp = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        golden = torch_rms_norm(temp, test_tensors["torch_weight2"], eps=1e-5)

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        assert passing, f"LN->RMS PCC check failed: {pcc}"

    def test_debug_full_3phase(self, device, test_tensors):
        """
        Step 3: Full 3-phase LayerNorm -> RMSNorm -> LayerNorm chain.
        The original hanging test case.
        """
        import os
        import signal

        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm
        from models.experimental.ops.descriptors import composite

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
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
            compute_kernel_config=ln_compute_config,
        )
        ln2 = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight3"],
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        fused = fuse_layernorm_chain([ln1, rms1, ln2])

        # Dump for inspection
        output_dir = os.path.join(os.path.dirname(__file__), "gen_kernels")
        dump_fused_kernels(fused, output_dir, label="debug_ln_rms_ln")

        # Launch with timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError("Device operation timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            outputs = composite.launch([fused])
            tt_output = outputs[0][0]
            torch_output = ttnn.to_torch(tt_output)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            pytest.fail("HANG DETECTED in LN->RMS->LN 3-phase chain")
        signal.signal(signal.SIGALRM, old_handler)

        # Golden: LayerNorm(RMSNorm(LayerNorm(x))) — using ones weights
        temp1 = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        temp2 = torch_rms_norm(temp1, test_tensors["torch_weight2"], eps=1e-5)
        golden = torch_layer_norm(temp2, test_tensors["torch_weight3"], test_tensors["torch_bias2"], eps=1e-5)

        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        assert passing, f"LN->RMS->LN PCC check failed: {pcc}"

    def test_debug_single_ln_source_code(self, device, test_tensors):
        """
        Step 0: Verify that a single LayerNorm works when its compute kernel
        is converted from FILE_PATH to SOURCE_CODE.

        If this test hangs, the issue is with SOURCE_CODE kernel compilation.
        If it passes, SOURCE_CODE works fine and the issue is in multi-phase fusion.
        """
        import os
        import signal

        from models.experimental.ops.descriptors.normalization import layer_norm
        from models.experimental.ops.descriptors import composite
        from models.experimental.ops.descriptors.sequential import _read_kernel_source_from_descriptor, _classify_kernel

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

        ln_desc = layer_norm.layer_norm(
            test_tensors["tt_input"],
            core_range_set=core_range,
            weight=test_tensors["tt_weight1"],
            bias=test_tensors["tt_bias1"],
            epsilon=1e-5,
        )

        # Find the compute kernel and convert it from FILE_PATH to SOURCE_CODE
        descriptor = ln_desc.descriptor
        kernels = list(descriptor.kernels)
        for kernel in kernels:
            ktype = _classify_kernel(kernel)
            if ktype == "compute":
                # Read the source from the file
                source = _read_kernel_source_from_descriptor(kernel)
                assert source, "Could not read compute kernel source"
                print(f"Compute kernel source: {len(source)} chars")
                print(f"Original source type: {kernel.source_type}")

                # Convert to SOURCE_CODE
                kernel.kernel_source = source
                kernel.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
                print(f"Converted to SOURCE_CODE")

                # Dump the source for inspection
                output_dir = os.path.join(os.path.dirname(__file__), "gen_kernels")
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "debug_single_ln_compute.cpp"), "w") as f:
                    f.write(source)
                break

        # Launch with timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError("Device operation timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(60)
        try:
            outputs = composite.launch([ln_desc])
            tt_output = outputs[0][0]
            torch_output = ttnn.to_torch(tt_output)
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            pytest.fail("HANG DETECTED: Single LN with SOURCE_CODE compute kernel")
        signal.signal(signal.SIGALRM, old_handler)

        # Golden: single LayerNorm
        golden = torch_layer_norm(
            test_tensors["torch_input"], test_tensors["torch_weight1"], test_tensors["torch_bias1"], eps=1e-5
        )
        passing, pcc = comp_pcc(golden, torch_output, pcc=0.98)
        print(f"Single LN SOURCE_CODE PCC: {pcc}")
        assert passing, f"Single LN SOURCE_CODE PCC check failed: {pcc}"


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
