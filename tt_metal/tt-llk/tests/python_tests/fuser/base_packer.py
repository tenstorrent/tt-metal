# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .pack_node import PackNode

from .block_data import InvocationGranularity
from .golden import Golden


class Packer(Golden):
    """Base class for fused test packer code generators.

    Subclasses override methods to emit the C++ LLK calls that configure and
    drive the Pack thread, plus a Python golden function for test validation.

    The pipeline calls pack() for each resolved invocation:
        init() -> planned calls to pack() -> uninit()

    To create a new packer:
        1. Subclass Packer
        2. Override get_headers() with the required LLK header files
        3. Override init(), pack(), uninit() to emit the C++ LLK calls
        4. Override golden() to compute the expected pack result,
           calling self.relu_golden() and self.l1_acc_golden() as needed
    """

    granularity = InvocationGranularity.TILE

    # Set `per_block_init = True` if init() needs block dimensions and must
    # be called per-block inside the batch loop rather than hoisted out.
    per_block_init: bool = False

    pack_mode: str = "PackMode::Default"

    requires_dest_remap: bool = False

    def get_headers(self) -> List[str]:
        """Return the list of C++ LLK header filenames required by this packer.

        These headers are #included in the generated test source file. Override to
        return the headers that declare the _llk_pack_*_ functions used by init(),
        pack(), and uninit().
        """
        return []

    def golden(
        self,
        tensor: torch.Tensor,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        """Compute the golden pack result in Python.

        Returns the tensor after applying pack transforms.
        Override and call self.relu_golden() or self.l1_acc_golden()
        as needed based on the pack_node config.
        """
        return tensor

    def init(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that initializes the packer before the pack loop.

        Called once per block. Override to emit the _llk_pack_init_<>()
        calls with the appropriate parameters
        """
        return ""

    def pack(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that packs a single tile from dest to L1.

        Called for each planned invocation. Use block.tile_id_block for the Dest
        register index and
        block.tile_id_global for the L1 output buffer index.
        Override to emit the _llk_pack_<>() call.
        """
        return ""

    def uninit(
        self,
        pack_node: "PackNode",
        operation: "L1Operation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that uninitializes the packer after the pack loop.

        Called once per block after the pack loop completes. Override if the
        packer requires explicit cleanup.
        """
        return ""
