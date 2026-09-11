# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig
    from .fpu_node import FpuNode
    from .block_data import BlockData

from .golden.state import OutputLayout
from .indexing import InvocationGranularity


class Unpacker:
    """Base class for fused test unpacker code generators.

    Subclasses represent specific unpack operations (e.g. UnpackerA, MatmulUnpacker, etc.)
    and override methods to emit the C++ LLK calls that configure and
    drive the Unpack thread.

    The lifecycle called by the pipeline is:
        init() -> planned calls to unpack() -> uninit()

    Set `granularity` to the number of tiles one call covers, to control
    the tile iteration pattern used by the unpack phases.

    Set `per_block_init = True` if init() needs block dimensions and must
    be called per-block inside the batch loop rather than hoisted out.

    To create a new unpacker:
        1. Subclass Unpacker
        2. Set `granularity` to the tiles one call covers
        3. Override get_headers() with the required LLK header files
        4. Override init(), unpack(), uninit() to emit the C++ LLK calls
        5. Bind the corresponding callable from fuser.golden.unpack
        6. Override perf_set_valid() / perf_clear_valid() for perf isolation
    """

    # Controls the tile iteration pattern for unpack and math loops.
    granularity = InvocationGranularity.NONE
    per_block_init: bool = False

    output_layout = OutputLayout.ROW_MAJOR

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the unpacker before the tile loop."""
        return ""

    def unpack(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code for one planned unpack call (src_a index in
        block.tile_id_src_a, src_b in block.tile_id_src_b, dest in
        block.tile_id_dest)."""
        return ""

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        compute_unit: "FpuNode",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the unpacker after the tile loop."""
        return ""

    def perf_set_valid(
        self,
        operation: "L1Operation",
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
        operation: "L1Operation",
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
        """Return the LLK header filenames that declare this unpacker's generated calls."""
        return []
