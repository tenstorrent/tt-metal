# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar element-wise binary operations."""

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..eltwise import EltwiseBinaryGolden


class QuasarEltwiseBinaryGolden(EltwiseBinaryGolden):
    """Element-wise binary on Quasar.

    The FPU multiplies 7x7 mantissa bits per phase, so a src datum's 10 explicit
    bits split 7 high / 3 low on both operands.

    A bf16 operand carries only 7 explicit bits, so its low half is always zero
    and every phase after AH_BH contributes nothing — fidelity is a no-op for
    bf16 and only changes the result for fp16, Tf32 and Float32 sources.
    """

    blocks_class = QuasarDataTransferBlocks
    MANTISSA_SPLIT = (7, 7)
