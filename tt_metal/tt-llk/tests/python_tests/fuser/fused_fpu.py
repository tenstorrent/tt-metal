# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .fpu_node import FpuNode
    from .block_data import BlockData


from .fused_loop import FusedLoop


class Fpu:
    """Base class for fused test FPU (math) code generators.

    Subclasses represent specific math operations (e.g. MatmulFpu, DatacopyFpu, etc.)
    and override methods to emit the C++ LLK calls that configure and drive the
    Math thread, plus a Python golden function for test validation.

    The lifecycle called by the pipeline is:
        init() -> loop.math_loop() [which calls calculate()] -> uninit()

    Override `loop` with an appropriate FusedLoop subclass to control
    the tile iteration pattern used by the math phase.

    Set `per_block_init = True` if init() needs block dimensions and must
    be called per-block inside the batch loop rather than hoisted out.

    To create a new FPU:
        1. Subclass Fpu
        2. Set `loop` to the desired FusedLoop variant
        3. Override get_headers() with the required LLK header files
        4. Override init(), calculate(), uninit() to emit the C++ LLK calls
        5. Override golden() to compute the expected math result
    """

    # Controls the tile iteration pattern for the math loop.
    loop: FusedLoop = FusedLoop()
    per_block_init: bool = False

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the math engine before the tile loop.

        Called once per block before the math loop begins. Override to emit
        the _llk_math_*_init_<>() call with the appropriate parameters.

        Skipped during UNPACK_ISOLATE, PACK_ISOLATE, and L1_CONGESTION perf runs.
        """
        return ""

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that performs the math operation on a single tile.

        Called inside the tile loop by FusedLoop.math_loop(). Use block.tile_id_block
        for the dest register index. Override to emit the _llk_math_*_<>() call
        that executes the FPU operation on data in the source register files.
        """
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the math engine after the tile loop.

        Called once per block after the math loop completes. Override to emit
        the _llk_math_*_uninit_() call that restores math state.

        Skipped during UNPACK_ISOLATE, PACK_ISOLATE, and L1_CONGESTION perf runs.
        """
        return ""

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the golden math result in Python.

        Returns (tensor_a, tensor_b, tensor_dst) where tensor_dst is the expected
        output after the math operation. tensor_a and tensor_b are passed through
        (possibly modified) for downstream stages.

        Called by FpuNode.golden() after the unpack golden. The input tensors
        are the outputs of the unpacker's golden().
        """
        return (tensor_a, tensor_b, tensor_dst)

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this FPU.

        These headers are #included in the generated test source file. Override to
        return the headers that declare the _llk_math_*_ functions used by init(),
        calculate(), and uninit().
        """
        return []

    def __str__(self) -> str:
        return self.__class__.__name__
