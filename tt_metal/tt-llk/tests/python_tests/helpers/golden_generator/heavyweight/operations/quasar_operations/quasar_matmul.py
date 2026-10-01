# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar matmul."""

import torch

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..matmul import MatmulGolden


class QuasarMatmulGolden(MatmulGolden):
    """Quasar computes ``Dest = SrcB @ SrcA``, not ``SrcA @ SrcB``.

    ``_llk_unpack_matmul_init_`` sends its first argument to SrcB and its second
    to SrcA, so ``OPERAND_REGISTERS`` routes them the same way and
    ``run([arg0, arg1])`` produces ``arg0 @ arg1`` exactly as the kernel does.
    Both halves of that are load-bearing: routing without the product order, or
    the product order without the routing, silently transposes the result into
    something wrong everywhere but still plausible-looking.

    The FPU multiplies 7x7 mantissa bits per phase, so a src datum's 10
    explicit bits split 7 high / 3 low on both operands -- symmetric, unlike
    Wormhole/Blackhole, where SrcA also loses its least significant bit.
    """

    blocks_class = QuasarDataTransferBlocks
    op_name = "matmul(srcB@srcA)"
    MANTISSA_SPLIT = (7, 7)
    OPERAND_REGISTERS = ("srcB", "srcA")

    def _product(self, srcA: torch.Tensor, srcB: torch.Tensor) -> torch.Tensor:
        return (
            (self._as_tile(srcB).double() @ self._as_tile(srcA).double())
            .reshape(-1)
            .float()
        )
