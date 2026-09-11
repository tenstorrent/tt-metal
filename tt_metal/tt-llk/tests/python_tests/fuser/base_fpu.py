# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig
    from .fpu_node import FpuNode
    from .block_data import BlockData

from .indexing import InvocationGranularity


class Fpu:
    """Base class for fused test FPU (math) code generators.

    Subclasses represent specific math operations (e.g. MatmulFpu, DatacopyFpu, etc.)
    and override methods to emit the C++ LLK calls that configure and drive the
    Math thread.

    The lifecycle called by the pipeline is:
        init() -> planned calls to calculate() -> uninit()

    Set `granularity` to the number of tiles one call covers, to control
    the tile iteration pattern used by the math phase.

    Set `per_block_init = True` if init() needs block dimensions and must
    be called per-block inside the batch loop rather than hoisted out.

    To create a new FPU:
        1. Subclass Fpu
        2. Set `granularity` to the tiles one call covers
        3. Override get_headers() with the required LLK header files
        4. Override init(), calculate(), uninit() to emit the C++ LLK calls
        5. Bind the corresponding callable from fuser.golden.fpu.
    """

    # Controls the tile iteration pattern for the math loop.
    granularity = InvocationGranularity.NONE
    per_block_init: bool = False

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the math engine before the tile loop."""
        return ""

    def calculate(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that performs one planned math call (dest index in
        block.tile_id_dest)."""
        return ""

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the math engine after the tile loop."""
        return ""

    def get_headers(self) -> List[str]:
        """Return the LLK header filenames that declare this FPU's generated calls."""
        return []

    def __str__(self) -> str:
        return self.__class__.__name__
