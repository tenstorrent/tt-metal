# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``untilize_rows``: kernels for one row tile ``[1, 1, rows, W]`` TILE -> ROW_MAJOR, hosted by ``moe_dense``.

They run as a kernel group inside the host program, where ``ttnn.to_layout`` would be a program of its own. Pure data
movement, so bitwise. Elements of 2 or 4 bytes, widths of whole tiles.
"""

from __future__ import annotations

from .. import program as fp

NAME = "untilize_rows"
READER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER = fp.kernel_source(NAME, "writer_rows.cpp")
