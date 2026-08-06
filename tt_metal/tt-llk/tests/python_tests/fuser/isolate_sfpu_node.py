# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation

from helpers.tilize_untilize import tilize_block, untilize_block

from .base_isolate_sfpu import IsolateSfpu
from .block_data import BlockData
from .operand import Operand


class IsolateSfpuNode:
    """Wraps an isolate SFPU op with its operands and SrcS/Dest path selection.

    Analogous to FpuNode on the math side: the node owns the operands, the unit
    owns the code generation. Each source and the output is either an L1 operand
    (the SrcS path, unpacked by UNP_S / packed by PACK1 on TRISC3 itself) or a
    Dest tile, in which case the operand is None and the matching *_dest_index
    gives the tile index within the block.

    A node whose sources and output are all L1 operands is self-contained: it
    needs no synchronization with the MATH thread at all. A node touching Dest
    participates in the FPU_SFPU/SFPU_FPU handshake instead.
    """

    def __init__(
        self,
        sfpu: IsolateSfpu,
        src_a: Optional[Operand] = None,
        output: Optional[Operand] = None,
        src_b: Optional[Operand] = None,
        src_a_dest_index: Optional[int] = None,
        src_b_dest_index: Optional[int] = None,
        output_dest_index: Optional[int] = None,
    ):
        self.sfpu = sfpu
        self.src_a = src_a
        self.src_b = src_b
        self.output = output
        self.src_a_dest_index = src_a_dest_index
        self.src_b_dest_index = src_b_dest_index
        self.output_dest_index = output_dest_index

    @property
    def reads_dest(self) -> bool:
        return self.src_a_dest_index is not None or self.src_b_dest_index is not None

    @property
    def writes_dest(self) -> bool:
        return self.output_dest_index is not None

    @property
    def self_contained(self) -> bool:
        """True when the whole data path is L1 -> SrcS -> SFPU -> SrcS -> L1."""
        return not (self.reads_dest or self.writes_dest)

    def sfpu_init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if config.skip_math_init:
            return ""
        return self.sfpu.init(operation, config, self, block)

    def sfpu_run(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if config.skip_math_init:
            return ""
        return self.sfpu.calculate(operation, config, self, block)

    def sfpu_uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if config.skip_math_init:
            return ""
        return self.sfpu.uninit(operation, config, self, block)

    def golden(
        self,
        input_tensor: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        """Apply the op's golden to an L1 input tensor, block by block.

        Tilizes the input, walks it in the same block decomposition the
        generated tile loop uses, and untilizes the result. Mirrors
        SfpuNode.golden, but sources from and returns an L1 operand tensor
        rather than threading the Dest tensor through the pipeline.
        """
        tile_shape = operation.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        num_faces = tile_shape.total_num_faces()
        dimensions = self.output.dimensions

        tilized = tilize_block(
            input_tensor,
            dimensions,
            config.sentinel.golden_math_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        )

        tile_count_x = dimensions[1] // tile_dims[1]
        tile_count_y = dimensions[0] // tile_dims[0]
        tile_size = tilized.shape[1]

        for tile_y in range(tile_count_y):
            for tile_x in range(tile_count_x):
                tile_id = tile_count_x * tile_y + tile_x
                block_tensor = tilized[[tile_id], :].clone().flatten()
                block_tensor = self.sfpu.golden(
                    block_tensor,
                    operation,
                    config,
                    self,
                    (tile_dims[0], tile_dims[1]),
                    1,
                )
                tilized[[tile_id], :] = block_tensor.view(1, tile_size)

        return untilize_block(
            tilized.flatten(),
            config.sentinel.golden_math_format,
            dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        ).reshape(dimensions)

    def get_headers(self) -> List[str]:
        return self.sfpu.get_headers()

    def __str__(self) -> str:
        return f"IsolateSfpuNode({self.sfpu}, output={self.output})"
