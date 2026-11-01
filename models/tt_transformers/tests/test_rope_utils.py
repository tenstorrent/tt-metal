# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.rope import get_batch_size_per_device_group


@pytest.mark.parametrize(
    ("batch_size", "use_qk_fused", "num_devices", "mesh_shape", "expected"),
    [
        (1, True, 32, (8, 4), 2),
        (1, False, 32, (8, 4), 1),
        (32, True, 32, (8, 4), 16),
        (1, True, 1, (), 2),
    ],
)
def test_get_batch_size_per_device_group(batch_size, use_qk_fused, num_devices, mesh_shape, expected):
    assert get_batch_size_per_device_group(batch_size, use_qk_fused, num_devices, mesh_shape, 1) == expected
