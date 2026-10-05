# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole operation goldens."""

from .wormhole_datacopy import WormholeDataCopyGolden
from .wormhole_eltwise import WormholeEltwiseBinaryGolden
from .wormhole_matmul import WormholeMatmulGolden

__all__ = [
    "WormholeDataCopyGolden",
    "WormholeEltwiseBinaryGolden",
    "WormholeMatmulGolden",
]
