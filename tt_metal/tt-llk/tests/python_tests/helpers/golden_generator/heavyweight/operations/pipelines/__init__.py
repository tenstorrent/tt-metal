# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One module per operation: the pipelines themselves.

Each declares a chain of data-transfer blocks and supplies the arithmetic
between them. The machinery they are built from (``Chain``, ``Golden``, the
fidelity phases) sits one level up, because it knows nothing about any
particular operation.

Architectures are not here either: an operation takes one as an argument, so
these classes are architecture-independent. See :mod:`..arch`.
"""

from .datacopy import DataCopyGolden
from .eltwise import EltwiseBinaryGolden
from .matmul import MatmulGolden
from .reuse_dest import EltwiseBinaryReuseDestGolden

__all__ = [
    "DataCopyGolden",
    "EltwiseBinaryGolden",
    "EltwiseBinaryReuseDestGolden",
    "MatmulGolden",
]
