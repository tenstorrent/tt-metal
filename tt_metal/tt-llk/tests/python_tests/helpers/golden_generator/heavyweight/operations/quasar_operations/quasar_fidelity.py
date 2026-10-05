# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Quasar FPU's fidelity split, shared by every op that models it."""

from typing import Tuple

#: Explicit mantissa bits the multiplier takes from each operand per fidelity
#: phase, as (srcA, srcB).
#:
#: The FPU multiplies 7x7 mantissa bits per phase, so a src datum's 10 explicit
#: bits split 7 high / 3 low on **both** operands -- symmetric, unlike Wormhole
#: and Blackhole, whose split is 4/6 and whose SrcA additionally drops its least
#: significant bit at every fidelity.
#:
#: Defined here rather than on one op because the three Quasar ops that model
#: fidelity descend from three different bases and so cannot inherit it from
#: each other.
QUASAR_MANTISSA_SPLIT: Tuple[int, int] = (7, 7)
