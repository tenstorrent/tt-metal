# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host descriptor IDs may be wide; program creation still validates hardware IDs."""

import pytest
import ttnn


@pytest.mark.parametrize("logical_id", [63, 255, 256, 1024, 65535])
def test_cb_descriptor_logical_id_roundtrip(logical_id):
    fmt = ttnn.CBFormatDescriptor(buffer_index=logical_id, data_format=ttnn.bfloat16, page_size=64)
    assert fmt.buffer_index == logical_id
    fmt.buffer_index = 0
    fmt.buffer_index = logical_id
    assert fmt.buffer_index == logical_id
    descriptor = ttnn.CBDescriptor()
    descriptor.format_descriptors = [fmt]
    assert descriptor.format_descriptors[0].buffer_index == logical_id
    # Model the final logical-to-physical remap before program creation.
    fmt.buffer_index = 63
    descriptor.format_descriptors = [fmt]
    assert descriptor.format_descriptors[0].buffer_index == 63


@pytest.mark.parametrize("logical_id", [-1, 65536])
def test_cb_descriptor_rejects_out_of_uint16_range(logical_id, expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.CBFormatDescriptor(buffer_index=logical_id, data_format=ttnn.bfloat16, page_size=64)
