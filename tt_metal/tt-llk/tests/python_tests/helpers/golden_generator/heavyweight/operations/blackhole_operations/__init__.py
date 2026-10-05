# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole operation goldens."""

from .blackhole_datacopy import BlackholeDataCopyGolden
from .blackhole_eltwise import BlackholeEltwiseBinaryGolden
from .blackhole_matmul import BlackholeMatmulGolden

__all__ = [
    "BlackholeDataCopyGolden",
    "BlackholeEltwiseBinaryGolden",
    "BlackholeMatmulGolden",
]
