# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .block_data import BlockData
    from .compute_node import ComputeNode
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig


class Sfpu:
    """Base class for fused test SFPU code generators.

    Subclasses represent specific SFPU operations (e.g. UnarySfpu, BinarySfpu)
    and override methods to emit the C++ LLK calls that configure and drive the
    SFPU Unit, plus a Python golden function for test validation.

    Unlike Fpu, SFPU operates on dest register data that was already computed
    by a prior FPU stage or loaded via datacopy. It has no unpacker the
    ComputeNode enforces that sfpu and unpacker are mutually exclusive.

    The lifecycle called by ComputeNode.math_run() is:
        init() -> calculate() -> uninit()

    Entirely skipped during UNPACK_ISOLATE, PACK_ISOLATE, and L1_CONGESTION perf runs.

    To create a new SFPU:
        1. Subclass Sfpu
        2. Override get_headers() with the required LLK header files
        3. Override init(), calculate(), uninit() to emit the C++ LLK calls
        4. Override golden() to compute the expected SFPU result
    """

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the SFPU before calculation.

        Called once per block. Override to emit the
        _llk_math_eltwise_*_sfpu_init_<>() call.
        """
        return ""

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that performs the SFPU operation.

        Called once per block between init() and uninit().
        Override to emit the sfpu calls.
        """
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the SFPU after calculation.

        Called once per block after calculate(). Override if the SFPU
        requires explicit cleanup.
        """
        return ""

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        """Compute the golden SFPU result in Python.

        Operates on tilized dest data per block. batch_dims and batch_tile_cnt
        describe the current block's tile layout. Returns the transformed tensor.

        Called by ComputeNode.golden() on each block of the tilized dest tensor.
        """
        return tensor

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this SFPU.

        These headers are #included in the generated test source file.
        Override to return the headers that declare the SFPU functions
        used by init(), calculate() and uninit().
        """
        return []

    def __str__(self) -> str:
        return f"{self.__name__}"
