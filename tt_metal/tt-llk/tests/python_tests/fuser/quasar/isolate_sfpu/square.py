# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from fuser.fuser_config import GlobalConfig
from helpers.llk_params import MathOperation

from .unary import IsolateUnarySfpu


class SquareIsolateSfpu(IsolateUnarySfpu):
    """x * x on the SrcS path, as a single SFPMUL."""

    _OPERATION: ClassVar[MathOperation] = MathOperation.Square

    def sfpu_instructions(self, config: GlobalConfig) -> str:
        return "TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);\n"
