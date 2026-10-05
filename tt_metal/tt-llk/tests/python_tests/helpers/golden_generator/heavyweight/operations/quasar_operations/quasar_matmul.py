# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar matmul."""

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..matmul import MatmulGolden
from .quasar_fidelity import QUASAR_MANTISSA_SPLIT


class QuasarMatmulGolden(MatmulGolden):
    """Quasar matmul: the base operand routing, with a symmetric mantissa split.

    ``Dest = SrcB @ SrcA`` and the arg0 -> SrcB routing that goes with it are
    both inherited -- every architecture does it that way, so neither is a
    Quasar quirk. See :attr:`MatmulGolden.OPERAND_REGISTERS`.

    What *is* specific: the FPU multiplies 7x7 mantissa bits per phase, so a src
    datum's 10 explicit bits split 7 high / 3 low on both operands. That is
    symmetric, unlike Wormhole/Blackhole, where SrcA also loses its least
    significant bit.
    """

    blocks_class = QuasarDataTransferBlocks
    MANTISSA_SPLIT = QUASAR_MANTISSA_SPLIT
