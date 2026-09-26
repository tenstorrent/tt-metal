# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_sram_preparation_apis_are_experimental():
    assert not hasattr(ttnn, "prepare_generic_op"), "prepare_generic_op must not be exported at the ttnn root"
    assert not hasattr(
        ttnn, "create_sharded_tensor_view"
    ), "create_sharded_tensor_view must not be exported at the ttnn root"
    assert (
        ttnn.experimental.prepare_generic_op is ttnn._ttnn.operations.experimental.prepare_generic_op
    ), "ttnn.experimental.prepare_generic_op must be the experimental binding itself, not a wrapper"
    assert (
        ttnn.experimental.create_sharded_tensor_view is ttnn._ttnn.operations.experimental.create_sharded_tensor_view
    ), "ttnn.experimental.create_sharded_tensor_view must be the experimental binding itself, not a wrapper"
