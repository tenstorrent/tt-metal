# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for Sequential Kernel Chaining Infrastructure

These tests don't require ttnn to be imported, allowing testing of the
core infrastructure logic independently.
"""

import pytest
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any
from unittest.mock import MagicMock


# Mock ttnn before importing sequential
sys.modules["ttnn"] = MagicMock()


class TestCBRemapperStandalone:
    """Tests for the CBRemapper class without ttnn dependency."""

    def test_basic_allocation(self):
        """Test basic CB allocation."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()
        assert remapper.NUM_CBS == 32
        assert len(remapper.allocated) == 0

        # Create a phase with CBs
        phase = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={
                0: CBInfo(0, 2048, None, 2048, None),
                1: CBInfo(1, 2048, None, 2048, None),
                16: CBInfo(16, 2048, None, 2048, None),
            },
            input_cb_indices={0, 1},
            output_cb_indices={16},
        )

        remapping = remapper.allocate_for_phase(phase)

        # Should allocate 3 unique CB indices
        assert len(remapping) == 3
        assert len(set(remapping.values())) == 3

        # All original indices should be mapped
        assert 0 in remapping
        assert 1 in remapping
        assert 16 in remapping

    def test_sequential_allocation(self):
        """Test that CBs are allocated sequentially from 0."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()

        phase = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={
                0: CBInfo(0, 2048, None, 2048, None),
                5: CBInfo(5, 2048, None, 2048, None),
                10: CBInfo(10, 2048, None, 2048, None),
            },
        )

        remapping = remapper.allocate_for_phase(phase)

        # CBs should be allocated starting from 0
        allocated_values = sorted(remapping.values())
        assert allocated_values == [0, 1, 2]

    def test_chaining_reuses_output_cb(self):
        """Test that chaining reuses the previous phase's output CB."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()

        # Phase 0
        phase0 = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={
                0: CBInfo(0, 2048, None, 2048, None),
                16: CBInfo(16, 2048, None, 2048, None),
            },
            output_cb_indices={16},
        )

        remap0 = remapper.allocate_for_phase(phase0)
        remapper.finish_phase(remap0, output_cb_original=16)

        # The output CB should be in live_data_cbs
        assert remap0[16] in remapper.live_data_cbs

        # Phase 1 - chains from phase 0
        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=None,
            cb_info={
                0: CBInfo(0, 2048, None, 2048, None),  # This will be the chained input
                16: CBInfo(16, 2048, None, 2048, None),
            },
        )

        remap1 = remapper.allocate_for_phase(
            phase1,
            chain_from_previous=True,
            previous_output_cb=remap0[16],
            target_input_cb=0,
        )

        # Phase 1's CB 0 should be mapped to phase 0's CB 16
        assert remap1[0] == remap0[16]

    def test_cb_reuse_after_free(self):
        """Test that freed CBs are reused in subsequent phases."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()

        # Phase 0 uses CBs 0, 1, 2, 3
        phase0 = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={i: CBInfo(i, 2048, None, 2048, None) for i in [0, 1, 2, 3]},
            output_cb_indices={3},
        )

        remap0 = remapper.allocate_for_phase(phase0)
        assert len(remapper.allocated) == 4

        # Finish phase 0 - frees CBs 0, 1, 2 but keeps 3
        remapper.finish_phase(remap0, output_cb_original=3)
        assert len(remapper.allocated) == 1
        assert remap0[3] in remapper.allocated

        # Phase 1 should reuse freed CBs
        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=None,
            cb_info={
                0: CBInfo(0, 2048, None, 2048, None),
                1: CBInfo(1, 2048, None, 2048, None),
            },
            input_cb_indices={0},
            output_cb_indices={1},
        )

        remap1 = remapper.allocate_for_phase(
            phase1,
            chain_from_previous=True,
            previous_output_cb=remap0[3],
            target_input_cb=0,
        )

        # CB 0 is chained from phase 0's output
        assert remap1[0] == remap0[3]

        # CB 1 should reuse one of the freed indices (0, 1, or 2)
        assert remap1[1] in [0, 1, 2]

    def test_find_free_cb_exhaustion(self):
        """Test that exhausting all CBs raises an error."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()

        # Create a phase that uses all 32 CBs
        phase = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={i: CBInfo(i, 2048, None, 2048, None) for i in range(32)},
        )

        remap0 = remapper.allocate_for_phase(phase)
        assert len(remap0) == 32

        # Don't free any CBs, try to allocate more
        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=None,
            cb_info={32: CBInfo(32, 2048, None, 2048, None)},  # Try to allocate another
        )

        with pytest.raises(RuntimeError, match="No free CBs"):
            remapper.allocate_for_phase(phase1)

    def test_get_total_cbs_used(self):
        """Test counting total unique CBs used."""
        from models.experimental.ops.descriptors.sequential import CBRemapper, PhaseInfo, CBInfo

        remapper = CBRemapper()

        phase0 = PhaseInfo(
            phase_idx=0,
            op_descriptor=None,
            cb_info={0: CBInfo(0, 2048, None, 2048, None), 1: CBInfo(1, 2048, None, 2048, None)},
            output_cb_indices={1},
        )

        remap0 = remapper.allocate_for_phase(phase0)
        remapper.finish_phase(remap0, output_cb_original=1)

        # 2 CBs used so far
        assert remapper.get_total_cbs_used() == 2

        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=None,
            cb_info={0: CBInfo(0, 2048, None, 2048, None), 2: CBInfo(2, 2048, None, 2048, None)},
        )

        remap1 = remapper.allocate_for_phase(
            phase1,
            chain_from_previous=True,
            previous_output_cb=remap0[1],
            target_input_cb=0,
        )

        # Phase 1's CB 0 reuses phase 0's output (physical CB 1)
        # Phase 1's CB 2 reuses freed physical CB 0
        # Total unique physical CBs: {0, 1} = 2
        assert remapper.get_total_cbs_used() == 2


class TestPhaseInfo:
    """Tests for PhaseInfo dataclass."""

    def test_creation(self):
        """Test creating PhaseInfo."""
        from models.experimental.ops.descriptors.sequential import PhaseInfo, CBInfo

        phase = PhaseInfo(
            phase_idx=0,
            op_descriptor=MagicMock(),
            cb_info={0: CBInfo(0, 1024, None, 1024, None)},
            input_cb_indices={0},
            output_cb_indices={16},
        )

        assert phase.phase_idx == 0
        assert 0 in phase.cb_info
        assert phase.input_cb_indices == {0}
        assert phase.output_cb_indices == {16}
        assert phase.cb_remapping == {}  # Default empty


class TestCBInfo:
    """Tests for CBInfo dataclass."""

    def test_creation(self):
        """Test creating CBInfo."""
        from models.experimental.ops.descriptors.sequential import CBInfo

        cb = CBInfo(
            original_index=5,
            total_size=4096,
            data_format="Float16_b",
            page_size=2048,
            core_ranges="mock_ranges",
            is_input=True,
            is_output=False,
        )

        assert cb.original_index == 5
        assert cb.total_size == 4096
        assert cb.page_size == 2048
        assert cb.is_input is True
        assert cb.is_output is False


class TestRemapKernelCBIndices:
    """Tests for remap_kernel_cb_indices function."""

    def test_remap_via_compile_args(self):
        """Test remapping CB indices via compile-time args."""
        from models.experimental.ops.descriptors.sequential import remap_kernel_cb_indices

        kernel = MagicMock()
        kernel.compile_time_args = [100, 200, 0, 16, 300]  # CB 0 at pos 2, CB 16 at pos 3
        kernel.defines = []

        remapping = {0: 5, 16: 20}
        cb_arg_positions = {0: 2, 16: 3}

        result = remap_kernel_cb_indices(kernel, remapping, cb_arg_positions)

        assert result.compile_time_args[2] == 5
        assert result.compile_time_args[3] == 20
        assert result.compile_time_args[0] == 100  # Unchanged
        assert result.compile_time_args[1] == 200  # Unchanged
        assert result.compile_time_args[4] == 300  # Unchanged

    def test_remap_via_defines(self):
        """Test remapping CB indices via defines."""
        from models.experimental.ops.descriptors.sequential import remap_kernel_cb_indices

        kernel = MagicMock()
        kernel.compile_time_args = []
        kernel.defines = [("CB_IN", "0"), ("CB_OUT", "16"), ("OTHER", "42")]

        remapping = {0: 5, 16: 20}
        cb_defines = {0: "CB_IN", 16: "CB_OUT"}

        result = remap_kernel_cb_indices(kernel, remapping, cb_defines=cb_defines)

        defines_dict = dict(result.defines)
        assert defines_dict["CB_IN"] == "5"
        assert defines_dict["CB_OUT"] == "20"
        assert defines_dict["OTHER"] == "42"  # Unchanged

    def test_remap_adds_missing_defines(self):
        """Test that missing defines are added."""
        from models.experimental.ops.descriptors.sequential import remap_kernel_cb_indices

        kernel = MagicMock()
        kernel.compile_time_args = []
        kernel.defines = []  # No existing defines

        remapping = {0: 5}
        cb_defines = {0: "CB_NEW"}

        result = remap_kernel_cb_indices(kernel, remapping, cb_defines=cb_defines)

        defines_dict = dict(result.defines)
        assert defines_dict["CB_NEW"] == "5"

    def test_no_modification_without_mappings(self):
        """Test that nothing changes if no mappings provided."""
        from models.experimental.ops.descriptors.sequential import remap_kernel_cb_indices

        kernel = MagicMock()
        kernel.compile_time_args = [1, 2, 3]
        kernel.defines = [("X", "Y")]

        remapping = {0: 5}  # CB 0 -> 5, but no arg positions or defines for it

        result = remap_kernel_cb_indices(kernel, remapping)

        assert result.compile_time_args == [1, 2, 3]
        assert result.defines == [("X", "Y")]


class TestExtractCBInfo:
    """Tests for extract_cb_info function."""

    def test_extract_from_descriptor(self):
        """Test extracting CB info from a mock descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info

        # Create mock format descriptor
        fmt_desc = MagicMock()
        fmt_desc.buffer_index = 3
        fmt_desc.data_format = "Float16_b"
        fmt_desc.page_size = 2048

        # Create mock CB descriptor
        cb_desc = MagicMock()
        cb_desc.total_size = 8192
        cb_desc.core_ranges = "mock_ranges"
        cb_desc.format_descriptors = [fmt_desc]

        # Create mock program descriptor
        prog_desc = MagicMock()
        prog_desc.cbs = [cb_desc]

        result = extract_cb_info(prog_desc)

        assert 3 in result
        assert result[3].original_index == 3
        assert result[3].total_size == 8192
        assert result[3].page_size == 2048

    def test_extract_multiple_cbs(self):
        """Test extracting multiple CBs."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info

        # Create multiple CB descriptors
        fmt1 = MagicMock(buffer_index=0, data_format="F16", page_size=1024)
        fmt2 = MagicMock(buffer_index=16, data_format="F32", page_size=2048)

        cb1 = MagicMock(total_size=2048, core_ranges="r1", format_descriptors=[fmt1])
        cb2 = MagicMock(total_size=4096, core_ranges="r2", format_descriptors=[fmt2])

        prog_desc = MagicMock(cbs=[cb1, cb2])

        result = extract_cb_info(prog_desc)

        assert len(result) == 2
        assert 0 in result
        assert 16 in result
        assert result[0].page_size == 1024
        assert result[16].page_size == 2048


class TestSequentialChainBuilderBasic:
    """Basic tests for SequentialChainBuilder."""

    def test_creation(self):
        """Test creating a builder."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()
        assert len(builder.phases) == 0
        assert len(builder.connections) == 0
        assert builder._built is False

    def test_add_phase(self):
        """Test adding phases to the builder."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        assert len(builder.phases) == 1
        assert builder.phases[0].phase_idx == 0

        builder.add_phase(mock_desc, input_from_previous=True)
        assert len(builder.phases) == 2
        assert builder.phases[1].phase_idx == 1

        # Should have one connection (phase 0 -> phase 1)
        assert len(builder.connections) == 1

    def test_set_phase_io(self):
        """Test setting phase I/O configuration."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        builder.set_phase_io(0, input_cbs=[0, 1, 2], output_cbs=[16, 17])

        phase = builder.phases[0]
        assert phase.input_cb_indices == {0, 1, 2}
        assert phase.output_cb_indices == {16, 17}

    def test_connect_phases(self):
        """Test explicit phase connections."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        builder.add_phase(mock_desc)
        builder.connect_phases(0, 16, 1, 0)

        assert len(builder.connections) == 1
        conn = builder.connections[0]
        assert conn.source_phase_idx == 0
        assert conn.source_output_cb == 16
        assert conn.target_phase_idx == 1
        assert conn.target_input_cb == 0

    def test_method_chaining(self):
        """Test that builder methods return self."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = (
            builder.add_phase(mock_desc)
            .set_phase_io(0, input_cbs=[0], output_cbs=[16])
            .add_phase(mock_desc, input_from_previous=True)
        )

        assert result is builder
        assert len(builder.phases) == 2


class TestChainDescriptorsConvenience:
    """Tests for the chain_descriptors convenience function."""

    def test_single_descriptor(self):
        """Test that single descriptor is returned as-is."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = chain_descriptors([mock_desc])
        assert result is mock_desc

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors

        with pytest.raises(ValueError, match="no phases"):
            chain_descriptors([])


class TestFusedKernelGeneratorStandalone:
    """Standalone tests for FusedKernelGenerator without device dependency."""

    def test_generator_creation(self):
        """Test basic generator creation."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        assert generator.phases == []
        assert generator.include_paths == []

    def test_generator_with_include_paths(self):
        """Test generator with custom include paths."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator(include_paths=["/path/1", "/path/2"])
        assert generator.include_paths == ["/path/1", "/path/2"]

    def test_add_single_phase(self):
        """Test adding a single phase."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() { }",
            cb_remapping={0: 10},
            is_first=True,
            is_last=True,
        )

        assert len(generator.phases) == 1
        assert generator.phases[0]["is_first"] is True
        assert generator.phases[0]["is_last"] is True
        assert generator.phases[0]["cb_remapping"] == {0: 10}

    def test_add_multiple_phases(self):
        """Test adding multiple phases."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(compute_source="void kernel_main() { phase0(); }", is_first=True)
        generator.add_phase(compute_source="void kernel_main() { phase1(); }")
        generator.add_phase(compute_source="void kernel_main() { phase2(); }", is_last=True)

        assert len(generator.phases) == 3
        assert generator.phases[0]["is_first"] is True
        assert generator.phases[0]["is_last"] is False
        assert generator.phases[2]["is_last"] is True

    def test_generate_single_phase_compute(self):
        """Test generating compute for single phase."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="""#include "api.h"
void kernel_main() {
    // do something
}
""",
            is_first=True,
            is_last=True,
        )

        fused = generator.generate_fused_compute()

        # Single phase should return the source with remapping applied
        assert "void kernel_main()" in fused
        assert "do something" in fused

    def test_cb_remapping_in_source(self):
        """Test that CB indices are remapped in source code."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="""
void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_other = CB::c_5;
}
""",
            cb_remapping={0: 10, 16: 26, 5: 15},
            is_first=True,
            is_last=True,
        )

        fused = generator.generate_fused_compute()

        # Check remapping was applied
        assert "tt::CBIndex::c_10" in fused
        assert "tt::CBIndex::c_26" in fused
        assert "CB::c_15" in fused

    def test_generate_multi_phase_compute(self):
        """Test generating fused compute for multiple phases."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="""#include "api1.h"
void kernel_main() {
    // phase 0 code
}
""",
            is_first=True,
        )
        generator.add_phase(
            compute_source="""#include "api2.h"
void kernel_main() {
    // phase 1 code
}
""",
            is_last=True,
        )

        fused = generator.generate_fused_compute()

        # Should have auto-generated comment
        assert "Auto-generated fused compute kernel" in fused
        assert "Fuses 2 phases" in fused

        # Should have phase markers
        assert "Phase 0" in fused
        assert "Phase 1" in fused

        # Should have a single kernel_main
        assert fused.count("void kernel_main()") == 1

    def test_noop_reader_generation(self):
        """Test generating no-op reader when no reader source provided."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() {}",
            reader_source=None,  # No reader
            is_first=True,
            is_last=True,
        )

        reader = generator.generate_fused_reader()

        assert "No-op" in reader
        assert "void kernel_main()" in reader

    def test_noop_writer_generation(self):
        """Test generating no-op writer when no writer source provided."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() {}",
            writer_source=None,  # No writer
            is_first=True,
            is_last=True,
        )

        writer = generator.generate_fused_writer()

        assert "No-op" in writer
        assert "void kernel_main()" in writer

    def test_reader_from_first_phase(self):
        """Test that reader comes from first phase."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() {}",
            reader_source="void kernel_main() { first_reader(); }",
            is_first=True,
        )
        generator.add_phase(
            compute_source="void kernel_main() {}",
            reader_source="void kernel_main() { second_reader(); }",
            is_last=True,
        )

        reader = generator.generate_fused_reader()

        # Should use reader from first phase only
        assert "first_reader" in reader
        assert "second_reader" not in reader

    def test_writer_from_last_phase(self):
        """Test that writer comes from last phase."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="void kernel_main() {}",
            writer_source="void kernel_main() { first_writer(); }",
            is_first=True,
        )
        generator.add_phase(
            compute_source="void kernel_main() {}",
            writer_source="void kernel_main() { second_writer(); }",
            is_last=True,
        )

        writer = generator.generate_fused_writer()

        # Should use writer from last phase only
        assert "second_writer" in writer
        assert "first_writer" not in writer

    def test_empty_generator_raises(self):
        """Test that generating from empty generator raises."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()

        with pytest.raises(ValueError, match="No phases"):
            generator.generate_fused_compute()

        with pytest.raises(ValueError, match="No phases"):
            generator.generate_fused_reader()

        with pytest.raises(ValueError, match="No phases"):
            generator.generate_fused_writer()

    def test_extract_kernel_body(self):
        """Test extracting kernel body from source."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()

        source = """
#include "api.h"

void some_helper() {
    // helper code
}

void kernel_main() {
    int x = 1;
    int y = 2;
    compute(x, y);
}

void other_func() {
    // other
}
"""
        body = generator._extract_kernel_body(source)

        # Should extract content inside kernel_main
        assert "int x = 1" in body
        assert "int y = 2" in body
        assert "compute(x, y)" in body
        # Should not include other functions
        assert "helper code" not in body
        assert "other" not in body

    def test_collect_includes(self):
        """Test collecting includes from all phases."""
        from models.experimental.ops.descriptors.sequential import FusedKernelGenerator

        generator = FusedKernelGenerator()
        generator.add_phase(
            compute_source="""#include "common.h"
#include "phase0_specific.h"
void kernel_main() {}
""",
            is_first=True,
        )
        generator.add_phase(
            compute_source="""#include "common.h"
#include "phase1_specific.h"
void kernel_main() {}
""",
            is_last=True,
        )

        includes = generator._collect_includes()

        # Should collect unique includes
        assert len(includes) == 3
        assert '#include "common.h"' in includes
        assert '#include "phase0_specific.h"' in includes
        assert '#include "phase1_specific.h"' in includes


class TestParallelChainsStandalone:
    """Standalone tests for parallel chain functions."""

    def test_create_parallel_chain_empty(self):
        """Test creating parallel chains with empty list."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors

        result = create_parallel_chain_descriptors([])
        assert result == []

    def test_create_parallel_chain_single_ops(self):
        """Test creating parallel chains with single-op chains."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors

        mock_desc1 = MagicMock()
        mock_desc1.descriptor = MagicMock(cbs=[])
        mock_desc2 = MagicMock()
        mock_desc2.descriptor = MagicMock(cbs=[])

        chains = [[mock_desc1], [mock_desc2]]
        result = create_parallel_chain_descriptors(chains)

        # Single-op chains should return original descriptors
        assert len(result) == 2
        assert result[0] is mock_desc1
        assert result[1] is mock_desc2

    def test_fuse_layernorm_chain_empty_raises(self):
        """Test that fusing empty chain raises."""
        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain

        with pytest.raises(ValueError, match="No descriptors"):
            fuse_layernorm_chain([])

    def test_fuse_layernorm_chain_single(self):
        """Test that fusing single op returns it unchanged."""
        from models.experimental.ops.descriptors.sequential import fuse_layernorm_chain

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = fuse_layernorm_chain([mock_desc])
        assert result is mock_desc


class TestReadKernelSource:
    """Tests for read_kernel_source function."""

    def test_read_kernel_source(self, tmp_path):
        """Test reading kernel source from file."""
        from models.experimental.ops.descriptors.sequential import read_kernel_source

        # Create a temp file
        kernel_file = tmp_path / "test_kernel.cpp"
        kernel_content = """
#include "api.h"
void kernel_main() {
    // test kernel
}
"""
        kernel_file.write_text(kernel_content)

        result = read_kernel_source(str(kernel_file))
        assert result == kernel_content


class TestNamedCompileTimeArgs:
    """Tests for named compile-time args handling."""

    def test_remap_named_compile_time_args(self):
        """Test remapping named compile-time args."""
        from models.experimental.ops.descriptors.sequential import remap_kernel_cb_indices

        kernel = MagicMock()
        kernel.compile_time_args = []
        kernel.defines = []
        kernel.named_compile_time_args = [
            ("cb_in", 0),
            ("cb_out", 16),
            ("other_arg", 100),
        ]

        remapping = {0: 10, 16: 26}

        result = remap_kernel_cb_indices(kernel, remapping)

        # Check named args were remapped
        named_dict = dict(result.named_compile_time_args)
        assert named_dict["cb_in"] == 10
        assert named_dict["cb_out"] == 26
        assert named_dict["other_arg"] == 100  # Not in remapping, unchanged

    def test_extract_cb_names(self):
        """Test extracting CB names from kernel descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_names_from_kernel

        kernel = MagicMock()
        kernel.named_compile_time_args = [
            ("cb_in", 0),
            ("cb_out", 16),
            ("cb_scaler", 2),
            ("some_other_arg", 42),
        ]

        result = extract_cb_names_from_kernel(kernel)

        # Should only extract cb_ prefixed names
        assert result == {
            "cb_in": 0,
            "cb_out": 16,
            "cb_scaler": 2,
        }
        assert "some_other_arg" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
