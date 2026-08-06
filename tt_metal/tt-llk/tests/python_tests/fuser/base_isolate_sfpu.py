# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .block_data import BlockData
    from .fuser_config import GlobalConfig
    from .isolate_sfpu_node import IsolateSfpuNode
    from .l1_operation import L1Operation

from .golden import Golden


class IsolateSfpu(Golden):
    """Base class for fused test isolate-SFPU code generators (Quasar TRISC3).

    Subclasses represent specific SFPU operations running on the third compute
    thread (LLK_TRISC_ISOLATE_SFPU) over the SrcS register file, which has its
    own unpacker (UNP_S) and packer (PACK1). A self-contained isolate op streams
    L1 -> SrcS -> SFPU -> SrcS -> L1 without touching Dest and without any work
    on the UNPACK, MATH or PACK threads.

    Unlike Sfpu, which drives the MATH thread's inline SFPU over Dest data
    produced by a preceding FPU or datacopy stage, IsolateSfpu owns its whole
    data path: it configures its own buffer descriptors and drives its own
    unpack/pack. It therefore carries src_a/output operands (on IsolateSfpuNode)
    the way FpuNode does, which plain SfpuNode does not.

    The lifecycle called by IsolateSfpuNode is:
        init() -> calculate() -> uninit()

    calculate() emits the tile loop via _llk_sfpu_srcs_, handing the per-slice
    instruction sequence returned by slice_kernel() to that helper. Concrete ops
    therefore only implement slice_kernel() plus their headers and golden.

    To create a new isolate SFPU op:
        1. Subclass the arch's IsolateUnarySfpu (which implements the frame)
        2. Set _OPERATION to the MathOperation it implements
        3. Override slice_kernel() with the SFPU instruction sequence
        4. Override get_headers() if the op needs extra headers
        5. Register the class in the arch parser's ISOLATE_SFPU_MAP
    """

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "IsolateSfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that configures the SrcS unpack/pack path and the SFPU.

        Called once per operation, outside the tile loop. Override to emit the
        _llk_sfpu_srcs_init_() call.
        """
        return ""

    def calculate(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "IsolateSfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that streams every tile through the SFPU.

        Called once per operation between init() and uninit(). Override to emit
        the _llk_sfpu_srcs_() call wrapping slice_kernel().
        """
        return ""

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "IsolateSfpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that drains the SFPU, unpacker and packer.

        Called once per operation after calculate().
        """
        return ""

    def slice_kernel(self, config: "GlobalConfig") -> str:
        """Return the SFPU instruction sequence for one SrcS slice.

        Emitted inside the lambda passed to _llk_sfpu_srcs_, where
        load_base_addr, store_base_addr and num_sfpu_iterations are in scope.
        The sequence reads one slice from load_base_addr and writes the result
        to store_base_addr; SrcS bookkeeping (slice rotation, dvalid) is handled
        by the caller, so the op carries none of it.
        """
        return ""

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "IsolateSfpuNode",
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        """Compute the golden isolate SFPU result in Python.

        Operates on the tilized L1 input per block. batch_dims and
        batch_tile_cnt describe the current block's tile layout. Returns the
        transformed tensor.
        """
        return tensor

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this op.

        These headers are #included in the generated test source file's
        LLK_TRISC_ISOLATE_SFPU section.
        """
        return []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
