# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract checks for the KDA cache-adapter ablation."""

import pytest

from models.demos.deepseek_v3_d_p.tests.kda.cache_adapters import (
    KDA_CONV_SEGMENT_BYTES,
    KDA_S_SEGMENT_BYTES,
    KdaCacheGeometry,
)


@pytest.mark.parametrize("sp,tp", [(1, 8), (2, 4), (4, 2)])
def test_k3_cache_adapter_geometry(sp: int, tp: int) -> None:
    geometry = KdaCacheGeometry(sp, tp)

    assert geometry.unique_recurrent_segments == 384
    assert geometry.unique_convolution_segments == 576
    assert KDA_S_SEGMENT_BYTES == 16_384
    assert KDA_CONV_SEGMENT_BYTES == 384
    assert geometry.recurrent_bytes_per_device == geometry.local_heads * 128 * 128 * 4
    assert geometry.convolution_bytes_per_device == geometry.local_heads * 3 * 3 * 128 * 2
    assert geometry.physical_recurrent_bytes == sp * 384 * KDA_S_SEGMENT_BYTES
    assert geometry.physical_convolution_bytes == sp * 576 * KDA_CONV_SEGMENT_BYTES


@pytest.mark.parametrize("sp,tp", [(1, 4), (2, 8), (8, 2)])
def test_k3_cache_adapter_geometry_rejects_non_eight_device_layouts(sp: int, tp: int, expect_error) -> None:
    with expect_error(ValueError, "exactly eight devices"):
        KdaCacheGeometry(sp, tp)
