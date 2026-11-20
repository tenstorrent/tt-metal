# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The video sizes Wan serves on each mesh, and the request check over them.

Sizes are (width, height). This module imports no device code at load, so a server process can
import it to validate requests before any worker opens a device.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Wan 2.2's 480p and 720p sizes in 16:9 and 9:16.
WAN_RESOLUTIONS_480P = ((832, 480), (480, 832))
WAN_RESOLUTIONS_720P = ((1280, 720), (720, 1280))

WAN_DEFAULT_RESOLUTION = (832, 480)

_SERVED_RESOLUTIONS: dict[tuple[str, tuple[int, ...]], tuple[tuple[int, int], ...]] = {
    # 720p runs out of DRAM in the VAE decoder.
    ("blackhole", (2, 2)): WAN_RESOLUTIONS_480P,
    ("blackhole", (4, 8)): WAN_RESOLUTIONS_720P + WAN_RESOLUTIONS_480P,
    ("blackhole", (4, 32)): WAN_RESOLUTIONS_720P + WAN_RESOLUTIONS_480P,
}


@functools.cache
def _is_blackhole() -> bool:
    import ttnn

    return ttnn.device.is_blackhole()


def served_resolutions(mesh_shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    arch = "blackhole" if _is_blackhole() else "wormhole_b0"
    return _SERVED_RESOLUTIONS.get((arch, tuple(mesh_shape)), (WAN_DEFAULT_RESOLUTION,))


def validate_request(*, mesh_shape: Sequence[int], width: int, height: int) -> None:
    served = served_resolutions(mesh_shape)
    if (width, height) not in served:
        msg = f"unsupported size {width}x{height}; this deployment serves {list(served)}"
        raise ValueError(msg)
