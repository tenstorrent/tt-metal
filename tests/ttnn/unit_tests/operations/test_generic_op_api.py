# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_sram_preparation_apis_are_experimental():
    assert not hasattr(ttnn, "prepare_generic_op")
    assert not hasattr(ttnn, "create_sharded_tensor_view")
    assert ttnn.experimental.prepare_generic_op is ttnn._ttnn.operations.experimental.prepare_generic_op
    assert ttnn.experimental.create_sharded_tensor_view is ttnn._ttnn.operations.experimental.create_sharded_tensor_view
