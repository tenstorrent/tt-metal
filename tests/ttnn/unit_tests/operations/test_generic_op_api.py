# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_prepare_generic_op_is_public():
    assert ttnn.prepare_generic_op is ttnn._ttnn.operations.generic.prepare_generic_op
