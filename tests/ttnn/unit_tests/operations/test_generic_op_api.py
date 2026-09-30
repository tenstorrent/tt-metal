# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_preparation_api_is_experimental():
    assert not hasattr(ttnn, "prepare_generic_op"), "prepare_generic_op must not be exported at the ttnn root"
    assert (
        ttnn.experimental.prepare_generic_op is ttnn._ttnn.operations.experimental.prepare_generic_op
    ), "ttnn.experimental.prepare_generic_op must be the experimental binding itself, not a wrapper"
