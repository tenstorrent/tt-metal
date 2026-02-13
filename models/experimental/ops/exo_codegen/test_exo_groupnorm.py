# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Exo-generated GroupNorm kernels.

Standalone tests verify source generation without hardware.
Device tests run the kernels on TT hardware and compare against PyTorch.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_blackhole


# ---------------------------------------------------------------------------
# Standalone tests (no device required)
# ---------------------------------------------------------------------------


class TestGroupNormStandalone:
    """Test Exo compilation and source generation."""

    def test_target_loads(self):
        """All @instr definitions are accepted by Exo."""
        from models.experimental.ops.exo_codegen.groupnorm_target import (
            tt_gn_mask,
            tt_gn_reduce,
        )

        assert tt_gn_mask is not None
        assert tt_gn_reduce is not None

    def test_procs_load(self):
        """All @proc definitions are accepted by Exo."""
        from models.experimental.ops.exo_codegen.groupnorm import get_procs

        procs = get_procs()
        assert len(procs) == 7
        assert set(procs.keys()) == {
            "mask",
            "sub_mean",
            "square",
            "mul_invstd",
            "gamma",
            "beta",
            "reduce",
        }

    def test_procs_compile_to_c(self):
        """All procs compile to C via Exo."""
        from exo import compile_procs_to_strings
        from models.experimental.ops.exo_codegen.groupnorm import get_procs

        procs = get_procs()
        for name, proc in procs.items():
            c_code, _ = compile_procs_to_strings([proc], f"gn_{name}.h")
            assert "for" in c_code, f"{name} should contain a loop"

    def test_mask_loop_has_mul_tiles(self):
        """Mask proc generates mul_tiles call."""
        from exo import compile_procs_to_strings
        from models.experimental.ops.exo_codegen.groupnorm import get_procs

        procs = get_procs()
        c_code, _ = compile_procs_to_strings([procs["mask"]], "gn_mask.h")
        assert "mul_tiles" in c_code
        assert "tile_regs_acquire" in c_code
        assert "pack_tile" in c_code

    def test_reduce_loop_has_reduce_tile(self):
        """Reduce proc generates reduce_tile call."""
        from exo import compile_procs_to_strings
        from models.experimental.ops.exo_codegen.groupnorm import get_procs

        procs = get_procs()
        c_code, _ = compile_procs_to_strings([procs["reduce"]], "gn_reduce.h")
        assert "reduce_tile" in c_code
        # Reduce should NOT have tile_regs_acquire (wraps entire loop)
        assert "tile_regs_acquire" not in c_code

    def test_sub_mean_has_bcast_scalar(self):
        """Sub mean proc generates sub_tiles_bcast_scalar call."""
        from exo import compile_procs_to_strings
        from models.experimental.ops.exo_codegen.groupnorm import get_procs

        procs = get_procs()
        c_code, _ = compile_procs_to_strings([procs["sub_mean"]], "gn_sub.h")
        assert "sub_tiles_bcast_scalar" in c_code

    def test_compute_kernel_structure(self):
        """Generated compute kernel has correct structure."""
        from models.experimental.ops.exo_codegen.groupnorm_codegen import (
            generate_groupnorm_compute,
        )

        src = generate_groupnorm_compute()
        # Check includes
        assert '#include "api/compute/reduce.h"' in src
        assert '#include "api/compute/bcast.h"' in src
        # Check 3 passes
        assert "Pass 1: Mean" in src
        assert "Pass 2: Variance" in src
        assert "Pass 3: Normalize" in src
        # Check CB declarations
        assert "cb_in0" in src
        assert "cb_out0" in src
        assert "cb_ex_global" in src
        assert "cb_ex2pe" in src
        # Check Exo-generated operations
        assert "mul_tiles(cb_in0, cb_input_mask" in src
        assert "reduce_tile<REDUCE_OP" in src
        assert "sub_tiles_bcast_scalar(cb_in0, cb_ex_global" in src
        assert "mul_tiles_bcast_scalar(cb_x, cb_ex2pe" in src
        # Check inv_std calculation
        assert "rsqrt_tile" in src

    def test_reader_kernel_structure(self):
        """Generated reader kernel has correct structure."""
        from models.experimental.ops.exo_codegen.groupnorm_codegen import (
            generate_groupnorm_reader,
        )

        src = generate_groupnorm_reader()
        assert "noc_async_read_page" in src
        assert "cb_reserve_back" in src
        assert "cb_push_back" in src
        assert "TensorAccessorArgs<0>()" in src

    def test_writer_kernel_structure(self):
        """Generated writer kernel has correct structure."""
        from models.experimental.ops.exo_codegen.groupnorm_codegen import (
            generate_groupnorm_writer,
        )

        src = generate_groupnorm_writer()
        assert "generate_reduce_scaler" in src
        assert "generate_bcast_col_scalar" in src
        assert "noc_async_write_page" in src
        assert "noc_async_write_barrier" in src
        # Check mask generation (all 1.0)
        assert "0x3F803F80" in src

    def test_pack_bfloat16(self):
        """BFloat16 packing produces correct values."""
        from models.experimental.ops.exo_codegen.groupnorm_program import (
            _pack_bfloat16,
        )

        # 1.0 in bfloat16 = 0x3F80
        packed = _pack_bfloat16(1.0)
        assert packed == 0x3F803F80

        # 0.0 in bfloat16 = 0x0000
        packed = _pack_bfloat16(0.0)
        assert packed == 0x00000000


# ---------------------------------------------------------------------------
# Device tests
# ---------------------------------------------------------------------------


@skip_for_blackhole("Not tested for Blackhole")
class TestGroupNormDevice:
    """Test GroupNorm on TT hardware."""

    @pytest.mark.parametrize("num_tiles", [1, 2, 4])
    def test_groupnorm_identity(self, device, num_tiles):
        """GroupNorm with gamma=1, beta=0 normalizes correctly.

        Uses num_groups=1 so the entire input is one group.
        Input shape: (1, num_tiles, 32, 32) -> num_tiles tiles.
        """
        shape = [1, num_tiles, 32, 32]
        torch_input = torch.randn(shape).to(torch.bfloat16)

        dram_config = ttnn.DRAM_MEMORY_CONFIG

        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_config,
        )

        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            dram_config,
        )

        from models.experimental.ops.exo_codegen.groupnorm_program import (
            build_groupnorm_program,
        )

        program = build_groupnorm_program(
            device,
            input_tensor,
            output_tensor,
            num_groups=1,
            eps=1e-5,
        )

        io_tensors = [input_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program)

        # Golden: PyTorch GroupNorm with num_groups=1, no affine
        C = shape[1]
        golden = torch.nn.functional.group_norm(
            torch_input.float(),
            num_groups=1,
            weight=None,
            bias=None,
            eps=1e-5,
        ).to(torch.bfloat16)

        torch_output = ttnn.to_torch(output)
        torch_golden = golden

        # Use PCC (Pearson Correlation Coefficient) for comparison
        # BFloat16 precision limits exact matching
        output_flat = torch_output.float().flatten()
        golden_flat = torch_golden.float().flatten()

        if golden_flat.std() < 1e-6:
            # Near-constant input, check absolute error
            assert torch.allclose(
                output_flat, golden_flat, atol=0.05
            ), f"Absolute error too large for near-constant input"
        else:
            pcc = torch.corrcoef(torch.stack([output_flat, golden_flat]))[0, 1].item()
            assert pcc > 0.95, f"PCC {pcc:.4f} < 0.95 for num_tiles={num_tiles}"
