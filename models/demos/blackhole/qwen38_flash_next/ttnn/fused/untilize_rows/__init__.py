# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``untilize_rows``: the kernels of one row tile ``[1, 1, rows, W]`` TILE -> ROW_MAJOR (``ttnn.to_layout``), hosted
by ``moe_dense`` for the routed dispatch untilize.  Hand-rolled because the composite runs them as one kernel group of
its own program beside the top-k and the shared expert; ``ttnn.to_layout`` would be a program of its own.

Pure data movement (the stock interleaved reader feeds one tile per CB page; the writer scatters the valid rows out of
the tile's faces), so the output is bitwise by construction.  Elements of 2 or 4 bytes, widths of whole tiles.
"""

from __future__ import annotations

from .. import program as fp

NAME = "untilize_rows"
READER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER = fp.kernel_source(NAME, "writer_rows.cpp")
