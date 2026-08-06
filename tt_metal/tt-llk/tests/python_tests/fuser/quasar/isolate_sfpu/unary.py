# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar, List

import torch
from fuser.base_isolate_sfpu import IsolateSfpu
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.isolate_sfpu_node import IsolateSfpuNode
from fuser.l1_operation import L1Operation
from helpers.llk_params import ApproximationMode, MathOperation

# Number of unpack/pack instructions placed in the SrcS auto-loop. One is enough
# for a whole tile: the auto-loop repeats it slice_count times (see llk_srcs.h).
SRCS_INSTRN_COUNT = 1

# Rows per SrcS slice in 16-bit mode; mirrors srcs_dims::YDIM_BASE. Needed here
# only to size the replay buffer, which is a call argument -- the addresses and
# iteration count themselves stay in C++, derived from srcs_dims.
SRCS_YDIM_BASE = 8


class IsolateUnarySfpu(IsolateSfpu):
    """Frame for a self-contained unary op on the SrcS path (L1 -> SrcS -> L1).

    Implements the whole lifecycle in terms of _llk_sfpu_srcs_init_ /
    _llk_sfpu_srcs_, which own the buffer-descriptor wiring, the SrcS auto-loop
    config and the tile/slice iteration. Concrete ops supply only the per-slice
    instruction sequence via slice_kernel(), so adding an op is a few lines.

    Subclasses set _OPERATION to the MathOperation they implement; the ctor
    signature is fixed by the isolate schema, which constructs every op class
    uniformly as cls(operation, approximation_mode, iterations).
    """

    _OPERATION: ClassVar[MathOperation] = None

    # LREG holding the result when the op's instructions have run. Ops that
    # write their result somewhere other than the input register override this.
    _RESULT_LREG: ClassVar[str] = "p_sfpu::LREG0"

    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
    ):
        if operation != type(self)._OPERATION:
            raise ValueError(
                f"{type(self).__name__} implements {type(self)._OPERATION}, "
                f"got {operation}"
            )
        self.operation = operation
        self.approx_mode = approx_mode
        self.iterations = iterations
        # Golden helpers (unary_sfpu_golden) read these off the unit.
        self.dest_idx = 0
        self.fill_const_value = 5

    def get_headers(self) -> List[str]:
        return [
            "cmath_common.h",
            "llk_math_common.h",
            "llk_sfpu_srcs.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: IsolateSfpuNode,
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        return self.unary_sfpu_golden(
            tensor, config, operation, compute_unit, batch_dims
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: IsolateSfpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        src_desc = compute_unit.src_a.cpp_srcs_desc_name
        out_desc = compute_unit.output.cpp_srcs_desc_name
        return (
            f"// Operation {stage}: Isolate {self.operation.cpp_enum_value} SFPU (SrcS)\n"
            f"_llk_sfpu_srcs_init_<{SRCS_INSTRN_COUNT}>({src_desc}, {out_desc});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: IsolateSfpuNode,
        block: BlockData,
    ) -> str:
        num_tiles = compute_unit.src_a.tile_count
        src_desc = compute_unit.src_a.cpp_srcs_desc_name
        out_desc = compute_unit.output.cpp_srcs_desc_name
        replay_buf_len = self._replay_buf_len(compute_unit, config)
        return (
            f"_llk_sfpu_srcs_<{SRCS_INSTRN_COUNT}>(\n"
            f"{num_tiles}, {src_desc}, {out_desc}, {replay_buf_len},\n"
            f"[](const int load_base_addr, const int store_base_addr, const int num_sfpu_iterations)\n"
            f"{{\n"
            f"{self.slice_kernel(config)}"
            f"}});\n"
        )

    @staticmethod
    def _srcs_ydim(compute_unit: IsolateSfpuNode) -> int:
        """Rows per SrcS slice, halved in 32-bit mode (mirrors srcs_dims::ydim)."""
        return (
            SRCS_YDIM_BASE // 2
            if compute_unit.src_a.data_format.is_32_bit()
            else SRCS_YDIM_BASE
        )

    @classmethod
    def _sfpu_iterations(cls, compute_unit: IsolateSfpuNode) -> int:
        """SFPU passes per slice; each pass covers SFP_ROWS (2) rows."""
        return cls._srcs_ydim(compute_unit) >> 1

    def _replay_buf_len(
        self, compute_unit: IsolateSfpuNode, config: GlobalConfig
    ) -> int:
        """Instruction count of one slice_kernel() expansion.

        The replay buffer is loaded with exactly this many instructions, so it
        is counted from the emitted body rather than declared per op: one line
        of slice_kernel() is one instruction, and the d-loop is unrolled by the
        compiler but issued num_sfpu_iterations times.
        """
        instructions = len(
            [
                line
                for line in self.slice_kernel(config).splitlines()
                if line.strip().startswith(("TT_", "TTI_"))
            ]
        )
        return instructions * self._sfpu_iterations(compute_unit)

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: IsolateSfpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_sfpu_srcs_done_();\n"

    def slice_kernel(self, config: GlobalConfig) -> str:
        """Load one SrcS slice, run the op, store the result back.

        One SFPU pass covers SFP_ROWS (2) rows of the slice, so the slice takes
        num_sfpu_iterations passes. Ops override sfpu_instructions(), not this.
        """
        return (
            f"#pragma GCC unroll 8\n"
            f"for (int d = 0; d < num_sfpu_iterations; d++)\n"
            f"{{\n"
            f"TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, load_base_addr + (d << 1));\n"
            f"{self.sfpu_instructions(config)}"
            f"TT_SFPSTORE({type(self)._RESULT_LREG}, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, store_base_addr + (d << 1));\n"
            f"}}\n"
        )

    def sfpu_instructions(self, config: GlobalConfig) -> str:
        """Return the op's SFPU instructions, transforming LREG0 into _RESULT_LREG.

        The surrounding load/store and slice iteration are emitted by
        slice_kernel(); an op contributes only its arithmetic.
        """
        return ""

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.operation})"
