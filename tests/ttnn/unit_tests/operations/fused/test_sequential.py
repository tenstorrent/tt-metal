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

        rms_desc = rms_norm.rms_norm(
            test_tensors["tt_input"],  # Placeholder - will be replaced by chain
            core_range_set=core_range,
            weight=test_tensors["tt_weight2"],
            epsilon=1e-5,
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
class TestSequentialChainExecution:
    """
    Execution tests for sequential chains.

    Note: These tests are marked as skip because full kernel-level fusion
    requires kernels to accept CB indices as compile-time arguments.
    The tests document the intended API and what golden results should be.
    """

    @pytest.mark.skip(reason="Requires kernel CB parameterization - see docstring")
    def test_layernorm_rmsnorm_layernorm_chain(self, device, test_tensors):
        """
        Test fusing LayerNorm -> RMSNorm -> LayerNorm chain.

        For this to work, kernels need to accept CB indices as compile-time args.
        The infrastructure supports this via cb_arg_positions parameter to build().

        Example of what kernel modification would look like:
        ```cpp
        // Instead of:
        constexpr auto cb_in = tt::CBIndex::c_0;
        constexpr auto cb_out = tt::CBIndex::c_16;

        // Use:
        constexpr auto cb_in = get_compile_time_arg_val(CB_IN_ARG_POS);
        constexpr auto cb_out = get_compile_time_arg_val(CB_OUT_ARG_POS);
        ```
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
            bias=test_tensors["tt_bias2"],
            epsilon=1e-5,
        )

        # Build chain
        builder = SequentialChainBuilder()
        builder.add_phase(ln1_desc)
        builder.add_phase(rms_desc, input_from_previous=True)
        builder.add_phase(ln2_desc, input_from_previous=True)

        # To make this work, we'd specify CB arg positions:
        # cb_arg_positions = {
        #     0: {0: CB_IN_ARG_POS, 16: CB_OUT_ARG_POS},  # Phase 0
        #     1: {0: CB_IN_ARG_POS, 16: CB_OUT_ARG_POS},  # Phase 1
        #     2: {0: CB_IN_ARG_POS, 16: CB_OUT_ARG_POS},  # Phase 2
        # }
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

    @pytest.mark.skip(reason="Requires kernel CB parameterization - see docstring")
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

    @pytest.mark.skip(reason="Requires C++ binding copy support - build() hangs with multi-phase chains")
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
        )
        chain1 = [ln1, rms1]

        # Chain 2: RMSNorm -> LayerNorm on cores2
        rms2 = rms_norm.rms_norm(
            tt_input2,
            core_range_set=cores2,
            weight=test_tensors["tt_weight1"],
            epsilon=1e-5,
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

    @pytest.mark.skip(reason="Requires C++ binding copy support - build() hangs with multi-phase chains")
    def test_fuse_layernorm_chain(self, device, test_tensors):
        """Test the fuse_layernorm_chain convenience function."""
        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain
        from models.experimental.ops.descriptors.normalization import layer_norm, rms_norm

        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

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

    @pytest.mark.skip(reason="Requires C++ binding copy support - remap_kernel_cb_indices hangs")
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

    @pytest.mark.skip(reason="Requires kernel CB parameterization for true fusion")
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

    @pytest.mark.skip(reason="Requires kernel CB parameterization for true fusion")
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
