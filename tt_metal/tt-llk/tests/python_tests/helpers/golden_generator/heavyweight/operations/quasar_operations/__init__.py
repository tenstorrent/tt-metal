# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar operation goldens.

Each binds the Quasar data-transfer blocks, so a test constructs one with no
arguments and never touches a block class.
"""

from .quasar_datacopy import QuasarDataCopyGolden
from .quasar_eltwise import QuasarEltwiseBinaryGolden
from .quasar_matmul import QuasarMatmulGolden

__all__ = [
    "QuasarDataCopyGolden",
    "QuasarEltwiseBinaryGolden",
    "QuasarMatmulGolden",
]
