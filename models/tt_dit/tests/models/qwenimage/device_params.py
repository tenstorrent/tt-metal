# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import is_blackhole
from models.tt_dit.utils.test import line_params_req_exact_devices

# The warm-up traces every resolution of the preset. Measured per chip: BH 2x2 759 MB, BH 2x4
# 738 MB, BH 4x8 779 MB, WH 2x4 (1024² only) 108 MB, WH 4x8 555 MB.
_WH_TRACE_REGION_SIZES = {(2, 4): 110_000_000, (4, 8): 560_000_000}


def _trace_region_size(shape: tuple[int, int]) -> int:
    if is_blackhole():
        return 790_000_000
    return _WH_TRACE_REGION_SIZES.get(shape, 0)


# (mesh_device, device_params) rows. A row whose mesh the machine doesn't have skips itself.
MESH_DEVICE_PARAMS = [
    pytest.param(
        shape,
        {**line_params_req_exact_devices, "trace_region_size": _trace_region_size(shape)},
        id=f"{shape[0]}x{shape[1]}",
    )
    for shape in [(2, 2), (2, 4), (4, 8)]
]

# The edit checkpoints trace one spatial bucket per number of input images (up to 3 for Edit-2509 and
# 2511) instead of one per resolution. Measured on WH 2x4: Edit 95 MB, Edit-2509 187 MB. WH 4x8 is
# not measured.
_WH_EDIT_TRACE_REGION_SIZES = {(2, 4): 200_000_000, (4, 8): 600_000_000}


def _edit_trace_region_size(shape: tuple[int, int]) -> int:
    if is_blackhole():
        return 790_000_000
    return _WH_EDIT_TRACE_REGION_SIZES.get(shape, 0)


EDIT_MESH_DEVICE_PARAMS = [
    pytest.param(
        shape,
        {**line_params_req_exact_devices, "trace_region_size": _edit_trace_region_size(shape)},
        id=f"{shape[0]}x{shape[1]}",
    )
    for shape in [(2, 2), (2, 4), (4, 8)]
]
