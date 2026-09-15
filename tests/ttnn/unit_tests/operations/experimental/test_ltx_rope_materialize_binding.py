# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_ltx_rope_materialize_binding_is_visible():
    assert callable(ttnn.experimental.ltx_rope_materialize)
    doc = ttnn.experimental.ltx_rope_materialize.__doc__
    assert "compact self tables" in doc
    assert "caller-supplied BF16" in doc
    assert "sp_axis" in doc
    assert "tp_axis" in doc
