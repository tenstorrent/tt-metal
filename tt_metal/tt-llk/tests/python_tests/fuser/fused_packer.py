# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .pack_node import PackNode

from helpers.golden_generators import PackGolden
from helpers.tilize_untilize import tilize_block, untilize_block

from .fused_loop import FusedLoop


class Packer:
    """Base class for fused test packer code generators.

    Subclasses override methods to emit the C++ LLK calls that configure and
    drive the Pack thread, plus a Python golden function for test validation.

    The pack lifecycle is driven by FusedLoop.pack_loop(), which iterates
    over tiles in the block and calls pack() for each one:
        init() -> pack_loop() [which calls pack()] -> uninit()

    To create a new packer:
        1. Subclass Packer
        2. Override get_headers() with the required LLK header files
        3. Override init(), pack(), uninit() to emit the C++ LLK calls
        4. Override golden() to compute the expected pack result,
           calling _relu_golden() and _l1_acc_golden() as needed
    """

    # Controls the tile iteration pattern for the pack loop.
    loop: FusedLoop = FusedLoop()

    @staticmethod
    def _l1_acc_golden(
        tensor: torch.Tensor,
        pack_node: "PackNode",
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        """Golden helper: simulate L1 accumulation across blocks."""
        output_dims = pack_node.output.dimensions
        output_format = pack_node.output.data_format
        tile_size = pack_node.output.tile_shape.total_tile_size()
        tile_count_x = pack_node.output.tile_count_x
        tile_count_y = pack_node.output.tile_count_y
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y

        tile_dims = (
            pack_node.output.tile_shape.total_row_dim(),
            pack_node.output.tile_shape.total_col_dim(),
        )
        num_faces = pack_node.output.tile_shape.total_num_faces()
        tensor = tilize_block(
            tensor,
            output_dims,
            output_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        ).flatten()
        tile_grid = tensor.view(tile_count_y, tile_count_x, tile_size)

        accumulated = torch.zeros(
            block_tiles_y, block_tiles_x, tile_size, dtype=tensor.dtype
        )
        for by in range(0, tile_count_y, block_tiles_y):
            for bx in range(0, tile_count_x, block_tiles_x):
                bty = min(block_tiles_y, tile_count_y - by)
                btx = min(block_tiles_x, tile_count_x - bx)
                accumulated[:bty, :btx] += tile_grid[by : by + bty, bx : bx + btx]

        result_grid = torch.zeros(
            tile_count_y, tile_count_x, tile_size, dtype=tensor.dtype
        )
        result_grid[:block_tiles_y, :block_tiles_x] = accumulated
        return untilize_block(
            result_grid.flatten(),
            output_format,
            output_dims,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

    @staticmethod
    def _relu_golden(
        tensor: torch.Tensor,
        pack_node: "PackNode",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        """Golden helper: apply packer ReLU activation."""
        intermediate_format = config.sentinel.golden_pack_src
        relu_config = PackGolden.generate_relu_config(
            pack_node.pack_relu, pack_node.relu_threshold, intermediate_format
        )
        return PackGolden.apply_relu(tensor, relu_config, intermediate_format)

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
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        """Compute the golden pack result in Python.

        Returns the tensor after applying pack transforms.
        Override and call _relu_golden() or _l1_acc_golden()
        as needed based on the pack_node config.
        """
        return tensor

    def init(
        self,
        pack_node: "PackNode",
        operation: "FusedOperation",
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
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that packs a single tile from dest to L1.

        Called inside the tile loop by FusedLoop.pack_loop(). Use
        block.tile_id_block for the dest register index and
        block.tile_id_global for the L1 output buffer index.
        Override to emit the _llk_pack_<>() call.
        """
        return ""

    def uninit(
        self,
        pack_node: "PackNode",
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: "BlockData",
    ) -> str:
        """Return C++ code that uninitializes the packer after the pack loop.

        Called once per block after the pack loop completes. Override if the
        packer requires explicit cleanup.
        """
        return ""
