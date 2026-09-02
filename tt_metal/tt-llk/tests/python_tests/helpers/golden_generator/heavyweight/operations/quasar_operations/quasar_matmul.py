# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar matmul."""

import torch

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..chain import Registers
from ..matmul import MatmulGolden


class QuasarMatmulGolden(MatmulGolden):
    """Quasar computes ``Dest = SrcB @ SrcA``, not ``SrcA @ SrcB``.

    ``_llk_unpack_matmul_init_`` sends its first argument to SrcB and its second
    to SrcA, so the operand that reads as "first" in a kernel is the *right*
    factor here. Getting this backwards gives a result that is wrong everywhere
    but still plausible-looking, so it is worth being explicit.
    """

    blocks_class = QuasarDataTransferBlocks
    op_name = "matmul(srcB@srcA)"

    def apply(self, regs: Registers) -> torch.Tensor:
        return (self._as_tile(regs["srcB"]) @ self._as_tile(regs["srcA"])).reshape(-1)
