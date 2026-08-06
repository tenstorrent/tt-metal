# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar, List

from fuser.fuser_config import GlobalConfig
from helpers.llk_params import MathOperation

from .unary import IsolateUnarySfpu


class ExpIsolateSfpu(IsolateUnarySfpu):
    """exp(x) on the SrcS path, via the SFPU nonlinear unit's EXP LUT.

    Quasar computes exp in hardware (p_sfpnonlinear::EXP_MODE), so this is one
    instruction rather than the polynomial expansion the Dest-path
    calculate_exponential() uses for its fp32-accurate mode. Accuracy therefore
    matches the LUT, which is what approximate-mode exp uses on the Dest path
    too -- the golden is generated in approximate mode to match.
    """

    _OPERATION: ClassVar[MathOperation] = MathOperation.Exp
    # SFPNONLINEAR writes its result to a second register rather than in place.
    _RESULT_LREG: ClassVar[str] = "p_sfpu::LREG1"

    def get_headers(self) -> List[str]:
        return super().get_headers() + ["ckernel_sfpu.h"]

    def sfpu_instructions(self, config: GlobalConfig) -> str:
        return "TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::EXP_MODE);\n"
