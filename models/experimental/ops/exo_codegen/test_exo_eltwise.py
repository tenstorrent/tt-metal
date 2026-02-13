# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for Exo-generated eltwise unary kernels.

Tests that Exo-generated reader/compute/writer kernels produce correct
results when run on device via ttnn.generic_op(). Compares against
native ttnn operations for numerical verification.

Run:
    pytest models/experimental/ops/exo_codegen/test_exo_eltwise.py -v
"""

from __future__ import annotations

import torch
import pytest
import ttnn
from loguru import logger

from models.common.utility_functions import skip_for_blackhole


# ---------------------------------------------------------------------------
# Standalone tests (no device — verify generated source structure)
# ---------------------------------------------------------------------------


class TestCodeGenStandalone:
    """Verify generated kernel sources match expected structure (no device)."""

    def test_identity_generates_three_kernels(self):
        from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

        reader, compute, writer = generate_eltwise_kernels("identity")

        assert "void kernel_main()" in reader
        assert "void kernel_main()" in compute
        assert "void kernel_main()" in writer

    def test_reader_has_noc_read(self):
        from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

        reader, _, _ = generate_eltwise_kernels("identity")
        assert "noc_async_read_page" in reader
        assert "cb_reserve_back" in reader
        assert "cb_push_back" in reader
        assert "TensorAccessor" in reader

    def test_compute_identity_has_copy_tile(self):
        from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

        _, compute, _ = generate_eltwise_kernels("identity")
        assert "copy_tile" in compute
        assert "tile_regs_acquire" in compute
        assert "pack_tile" in compute
        assert "init_sfpu" in compute

    def test_compute_relu_has_relu_tile(self):
        from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

        _, compute, _ = generate_eltwise_kernels("relu")
        assert "relu_tile(0)" in compute
        assert "relu_tile_init" in compute

    def test_writer_has_noc_write(self):
        from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

        _, _, writer = generate_eltwise_kernels("identity")
        assert "noc_async_write_page" in writer
        assert "cb_wait_front" in writer
        assert "cb_pop_front" in writer
        assert "noc_async_write_barrier" in writer

    def test_scheduling_changes_loop_structure(self):
        """Verify that different block_dim values change the generated code."""
        from models.experimental.ops.exo_codegen.eltwise_unary import get_procs
        from exo import compile_procs_to_strings

        _, flat_compute, _ = get_procs("identity", block_dim=1)
        _, tiled_compute, _ = get_procs("identity", block_dim=4)

        flat_c, _ = compile_procs_to_strings([flat_compute], "f.h")
        tiled_c, _ = compile_procs_to_strings([tiled_compute], "t.h")

        # Flat has single loop, tiled has nested loops
        assert "block_idx" not in flat_c
        assert "block_idx" in tiled_c
        assert "tile_idx" in tiled_c

    def test_exo_proc_definitions_exist(self):
        """Verify all expected @proc definitions are importable."""
        from models.experimental.ops.exo_codegen.eltwise_unary import (
            eltwise_reader,
            eltwise_identity_compute,
            eltwise_relu_compute,
            eltwise_writer,
        )

        assert eltwise_reader is not None
        assert eltwise_identity_compute is not None
        assert eltwise_relu_compute is not None
        assert eltwise_writer is not None


# ---------------------------------------------------------------------------
# Device tests (require TT hardware)
# ---------------------------------------------------------------------------


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 4, 16, 64])
def test_exo_identity(device, num_tiles):
    """Exo-generated identity kernel produces correct output."""
    from models.experimental.ops.exo_codegen.program_builder import build_eltwise_program

    shape = [1, num_tiles, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program = build_eltwise_program(device, input_tensor, output_tensor, op="identity")
    output = ttnn.generic_op([input_tensor, output_tensor], program)

    torch_input = ttnn.to_torch(input_tensor)
    torch_output = ttnn.to_torch(output)

    logger.info(f"Identity test: num_tiles={num_tiles}")
    matching = torch.allclose(torch_input, torch_output)
    logger.info(f"Tensors matching: {matching}")
    assert matching, f"Identity mismatch for {num_tiles} tiles"


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 4, 16, 64])
def test_exo_relu(device, num_tiles):
    """Exo-generated relu kernel matches native ttnn.relu()."""
    from models.experimental.ops.exo_codegen.program_builder import build_eltwise_program

    shape = [1, num_tiles, 32, 32]
    # Use data with both positive and negative values
    data = (torch.rand(shape) * 2 - 1).to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program = build_eltwise_program(device, input_tensor, output_tensor, op="relu")
    output = ttnn.generic_op([input_tensor, output_tensor], program)

    # Compare against native ttnn relu
    golden = ttnn.relu(input_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    logger.info(f"ReLU test: num_tiles={num_tiles}")
    matching = torch.allclose(torch_golden, torch_output)
    logger.info(f"Tensors matching: {matching}")
    assert matching, f"ReLU mismatch for {num_tiles} tiles"


@skip_for_blackhole("Not tested / built for Blackhole")
def test_exo_generated_source_matches_handwritten(device):
    """Verify Exo-generated source is structurally equivalent to hand-written."""
    from models.experimental.ops.exo_codegen.codegen import generate_eltwise_kernels

    reader, compute, writer = generate_eltwise_kernels("identity")

    # Reader should have the same core structure
    assert "get_arg_val<uint32_t>(0)" in reader  # src_addr
    assert "get_arg_val<uint32_t>(1)" in reader  # num_pages
    assert "get_arg_val<uint32_t>(2)" in reader  # start_id
    assert "cb_reserve_back(cb_id_in0, 1)" in reader
    assert "noc_async_read_page" in reader
    assert "cb_push_back(cb_id_in0, 1)" in reader

    # Compute should have tile register protocol
    assert "get_compile_time_arg_val(0)" in compute  # block_cnt
    assert "get_compile_time_arg_val(1)" in compute  # block_dim
    assert "init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2)" in compute
    assert "tile_regs_acquire()" in compute
    assert "copy_tile(tt::CBIndex::c_0, 0, 0)" in compute
    assert "pack_tile(0, tt::CBIndex::c_2)" in compute
    assert "tile_regs_release()" in compute

    # Writer should have write barrier at end
    assert "noc_async_write_barrier()" in writer
    assert "cb_wait_front(cb_id_out, 1)" in writer

    logger.info("Source structure verification passed")
