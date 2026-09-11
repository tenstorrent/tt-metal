# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .pack_node import PackNode

from .golden.state import OutputLayout
from .indexing import InvocationGranularity


class Packer:
    """Base class for fused test packer code generators.

    Subclasses override methods to emit the C++ LLK calls that configure and
    drive the Pack thread.

    The pack lifecycle is driven by the planned call nest, which iterates
    over tiles in the block and calls pack() for each one:
        init() -> planned calls to pack() -> uninit()

    To create a new packer:
        1. Subclass Packer
        2. Override get_headers() with the required LLK header files
        3. Override init(), pack(), uninit() to emit the C++ LLK calls
        4. Bind the corresponding callable from fuser.golden.pack
    """

    # Controls the tile iteration pattern for the pack loop.
    granularity = InvocationGranularity.NONE

    # Set `per_block_init = True` if init() needs block dimensions and must
    # be called per-block inside the batch loop rather than hoisted out.
    per_block_init: bool = False

    pack_mode: str = "PackMode::Default"

    output_layout = OutputLayout.ROW_MAJOR

    requires_dest_remap: bool = False

    def get_headers(self) -> List[str]:
        """Return the LLK header filenames that declare this packer's generated calls."""
        return []

    def init(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the packer before the pack loop."""
        return ""

    def pack(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code for one planned pack call (dest index in
        block.tile_id_dest, L1 output index in block.tile_id_out)."""
        return ""

    def uninit(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that tears down the packer after the pack loop."""
        return ""
