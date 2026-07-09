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


class Unpacker:
    """Base class for fused test unpacker code generators.

    Subclasses represent specific unpack operations (e.g. UnpackerA, MatmulUnpacker, etc.)
    and override methods to emit the C++ LLK calls that configure and
    drive the Unpack thread, plus a Python golden function for test validation.

    The lifecycle called by the pipeline is:
        init() -> loop.unpack_loop() [which calls unpack()] -> uninit()

    Override `loop` with an appropriate FusedLoop subclass to control
    the tile iteration pattern used by the unpack phases.

    Set `per_block_init = True` if init() needs block dimensions and must
    be called per-block inside the batch loop rather than hoisted out.

    To create a new unpacker:
        1. Subclass Unpacker
        2. Set `loop` to the desired FusedLoop variant
        3. Override get_headers() with the required LLK header files
        4. Override init(), unpack(), uninit() to emit the C++ LLK calls
        5. Override golden() to compute the expected unpack transformation
        6. Override perf_set_valid() / perf_clear_valid() for perf isolation
    """

    # Controls the tile iteration pattern for unpack and math loops.
    loop: FusedLoop = FusedLoop()
    per_block_init: bool = False

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the unpacker.

        Called once per block before the unpack loop begins. Override to emit
        the _llk_unpack_*_init_<>() call with the appropriate parameters

        Skipped during PACK_ISOLATE and MATH_ISOLATE perf runs.
        """
        return ""

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that unpacks a single tile (or tile group).

        Called inside the tile loop by FusedLoop.unpack_loop(). Use block.tile_id_global
        for the L1 buffer index and block.tile_id_block for the dest register index.
        Override to emit the _llk_unpack_*_<>() call that moves data from L1 into
        source register files.
        """
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the unpacker after the tile loop.

        Called once per block after the unpack loop completes. Override to emit
        the _llk_unpack_*_uninit_() call that restores unpacker state.

        Skipped during PACK_ISOLATE and MATH_ISOLATE perf runs.
        """
        return ""

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that mocks unpacker output for MATH_ISOLATE perf runs.

        During MATH_ISOLATE, real unpack is skipped. Override to call the correct
        number of set dvalids to match the unpack pattern of the operation.
        """
        return ""

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that mocks math consumption for UNPACK_ISOLATE perf runs.

        During UNPACK_ISOLATE, real math is skipped. Override to call the correct
        number of clear dvalids to match the math consumption pattern of the operation.
        """
        return ""

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this unpacker.

        These headers are #included in the generated test source file. Override to
        return the headers that declare the _llk_unpack_*_ functions used by init(),
        unpack(), and uninit().
        """
        return []

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "FpuNode" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the golden unpack transformation in Python.

        Returns (tensor_a, tensor_b) after applying the unpack transforms
        (transpose, broadcast, tilize, etc.). Set an output tensor to None
        to indicate that operand is unused by downstream math.

        Called by FpuNode.golden() before the math golden. The returned tensors
        become the math unit's inputs.
        """
        return tensor_a, tensor_b
