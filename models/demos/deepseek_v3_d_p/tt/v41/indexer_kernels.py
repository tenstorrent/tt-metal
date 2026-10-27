# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Row-major selection kernels of the DeepSeek-V4.1 indexer (bead F10; ``tt/v41/kernels``, launched through
``ttnn.generic_op``). Both work row by row on a DRAM-interleaved row-major bf16 tensor (one page per row), such as
the indexer's score, so every chip's rows stay local (no collectives; replicated or sharded mesh tensors work).

* :func:`segment_max`: maxima of consecutive runs of 8 or 32 elements of every row in one read of the row, with each
  row's newest element's run pinned to +inf (the candidate source's pin).
* :func:`gather_runs`: per row, the runs of 8 or 32 elements named by a row of run ids, in id order (-inf for an id
  past the row, e.g. the 0xFFFFFFFF sentinel).
* :func:`candidate_scores`: a candidate index source's score of each query's candidate rows only (the rows of its
  candidate blocks, in rank order), from the query, head weights and the row-major index-K cache: the same row
  layout ``gather_runs`` makes of a full score.
"""

import os

import ttnn

_KERNEL_DIR = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE_BYTES = 32 * 32 * 2  # bf16 tile
SEGMAX_BLOCK = 4  # tiles per segment-max chunk: 4096 consecutive row elements
PIN_PAGE = 8  # entries of the per-chip ``query_first`` pin page (V41ChunkTables.query_first)
SENTINEL = 0xFFFFFFFF
GATHER_SEGMENT_BYTES = 64 << 10  # gather_runs streams each source row through L1 in segments of this size
GATHER_MAX_SEGMENTS = 64  # gather_runs.cpp MAX_SEGS


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _cores(device, n: int):
    """The first ``n`` cores of the compute grid, row-major -> ([(x, y)], CoreRangeSet)."""
    grid = device.compute_with_storage_grid_size()
    n = min(n, grid.x * grid.y)
    last = ttnn.CoreCoord((n - 1) % grid.x, (n - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    return [(i % grid.x, i // grid.x) for i in range(n)], ttnn.CoreRangeSet(ranges)


def _split(total: int, parts: int) -> list[tuple[int, int]]:
    """``total`` items in ``parts`` contiguous (start, count) ranges, the first ``total % parts`` one longer."""
    base, extra = divmod(total, parts)
    out, start = [], 0
    for i in range(parts):
        count = base + (i < extra)
        out.append((start, count))
        start += count
    return out


def _cb(index: int, total: int, cores, dtype=ttnn.bfloat16, page: int = _TILE_BYTES) -> ttnn.CBDescriptor:
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=total, core_ranges=cores, format_descriptors=[fmt])


def _check_row_major(t, dtype, what: str):
    memory = t.memory_config()
    assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.dtype == dtype, f"{what} must be row-major {dtype}"
    assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), f"{what} must be DRAM-interleaved"


def _pin_rows(pin, rows: int) -> int:
    """The pin contract of a writer kernel: 0 for the per-chip ``query_first`` page ([1, 1, 1, 8]: ``q_i = query_first
    + i``), else the entries of a per-row page ([1, 1, 1, P] with P >= rows: ``q_i = pin[i]``, the sentinel pins
    nothing; the decode tile's ``newest`` rows)."""
    _check_row_major(pin, ttnn.uint32, "pin")
    assert pin.shape[2] == 1, tuple(pin.shape)
    if pin.shape[3] == PIN_PAGE:
        return 0
    assert pin.shape[3] >= rows, (tuple(pin.shape), rows)
    return pin.shape[3]


def segment_max(x, seg: int, pin, pin_base: int, pin_mask: int):
    """Row-major bf16 ``x`` [1, 1, R, W] (DRAM-interleaved, W a multiple of 32) -> row-major bf16
    [1, 1, R, round_up(W / seg, 32)]: output j of row i is ``max(x[i, seg * j : seg * (j + 1)])`` (-inf past W / seg),
    except the row's pinned output: +inf at ``e // seg`` with ``e = (pin_base + q_i) & pin_mask`` when ``e < W``,
    ``q_i`` the chunk query index of the chip's row i (``pin``: ``V41ChunkTables.query_first()``), or, with a per-row
    ``pin`` (``_pin_rows``), ``q_i = pin[i]`` (none when it is the sentinel). ``seg`` 8 or 32. One fused op
    (kernels/segmax_*): per 4096-element chunk a tilize and a MAX row reduce (for 8, after adding four column
    masks)."""
    assert seg in (8, 32)
    _check_row_major(x, ttnn.bfloat16, "segment_max input")
    rows, width = x.shape[2], x.shape[3]
    assert width % 32 == 0, f"row width {width} is not a multiple of 32"
    pin_rows = _pin_rows(pin, rows)
    pin_slot = _round_up(max(32, pin_rows * 4), 64)
    out_width = _round_up(width // seg, 32)
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, out_width]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    block = SEGMAX_BLOCK
    per_tile = 4 if seg == 8 else 1
    results, out_per_chunk = block * per_tile, block * 32 * per_tile
    chunks = -(-width // (block * 1024))
    coords, cores = _cores(device, rows)
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (first, count) in zip(coords, _split(rows, len(coords))):
        reader_args[cx][cy] = [x.buffer_address(), first, count, width * 2]
        writer_args[cx][cy] = [out.buffer_address(), first, count, chunks, pin.buffer_address(), pin_base, pin_mask]
        compute_args[cx][cy] = [count * chunks]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_reader.cpp",
            core_ranges=cores,
            compile_time_args=[0, 1, 2, seg, block] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_writer.cpp",
            core_ranges=cores,
            compile_time_args=[16, 5, seg, block, out_width, width // seg, pin_rows]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pin).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/segmax_compute.cpp",
            core_ranges=cores,
            compile_time_args=[seg, block],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
            ),
        ),
    ]
    scratch = 64 + pin_slot + 2 * out_per_chunk * 2  # 64: the writer aligns the pin page
    cbs = [
        _cb(0, 2 * block * _TILE_BYTES, cores),
        _cb(1, _TILE_BYTES, cores),
        _cb(3, 2 * block * _TILE_BYTES, cores),
        _cb(16, 2 * results * _TILE_BYTES, cores),
        _cb(5, scratch, cores, page=scratch),
        _cb(2, 4 * _TILE_BYTES, cores),  # column masks (8 only; the compute kernel binds every CB)
        _cb(4, 4 * block * _TILE_BYTES, cores),
    ]
    return ttnn.generic_op([x, pin, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def gather_runs(x, ids, run: int):
    """Row-major bf16 ``x`` [1, 1, R, W] and uint32 ``ids`` [1, 1, R, K] (both DRAM-interleaved) -> row-major bf16
    [1, 1, R, K * run]: run r of row i is ``x[i, run * c : run * (c + 1)]`` for ``c = ids[i, r]``, -inf where
    ``c >= W / run`` (the sentinel included). ``run`` 8 or 32 (16- or 64-byte runs). One data-movement op
    (kernels/gather_runs.cpp) on both RISC-Vs of every core: each source row is read once, sequentially."""
    assert run in (8, 32)
    _check_row_major(x, ttnn.bfloat16, "gather_runs input")
    _check_row_major(ids, ttnn.uint32, "gather_runs ids")
    rows, width, k = x.shape[2], x.shape[3], ids.shape[3]
    assert ids.shape[2] == rows and width % 32 == 0
    run_bytes = run * 2
    assert -(-width * 2 // GATHER_SEGMENT_BYTES) <= GATHER_MAX_SEGMENTS, f"row width {width} is too wide"
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, k * run]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords, cores = _cores(device, -(-rows // 2))
    units = _split(rows, 2 * len(coords))  # unit 2c on core c's reader RISC-V, 2c + 1 on its writer
    accessors = (
        ttnn.TensorAccessorArgs(x).get_compile_time_args()
        + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
    )
    scratch = 64 + GATHER_SEGMENT_BYTES + k * run_bytes + 2 * k * 4
    kernels = []
    for risc, config in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
        args = ttnn.RuntimeArgs()
        for c, (cx, cy) in enumerate(coords):
            first, count = units[2 * c + risc]
            args[cx][cy] = [
                x.buffer_address(),
                ids.buffer_address(),
                out.buffer_address(),
                first,
                count,
                width // run,
                width * 2,
            ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNEL_DIR}/gather_runs.cpp",
                core_ranges=cores,
                compile_time_args=[risc, run_bytes, k, GATHER_SEGMENT_BYTES] + accessors,
                runtime_args=args,
                config=config,
            )
        )
    cbs = [_cb(risc, scratch, cores, page=scratch) for risc in (0, 1)]
    return ttnn.generic_op([x, ids, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


CANDSCORE_BLOCK = 8  # rows per candidate block (CANDIDATE_BLOCK_SIZE)


def _largest_divisor(n: int, options=(8, 4, 2, 1)) -> int:
    return next(b for b in options if n % b == 0)


def candidate_scores(q, w, k, ids, pin, start: int, rows_visible: int, block_base: int = 0, base_page=None):
    """A ratio-1 candidate index source's score over each query's candidate rows only.

    ``q`` [1, 1, R, 32 * 128] bf16 row-major (the query's 32 heads, head-major, after RoPE and QDQ), ``w`` [1, 1, R, 32]
    bf16 row-major (scaled head weights), ``k`` the index-K cache: [1, 1, T, 128] bf16 row-major (T a multiple of
    8) or already as block pages [1, 1, P, 8 * 128] (the decode pool's ``index_k_blocks``: ``block_base`` the page of
    the slot's block 0, no copy), ``ids`` [1, 1, R, K] uint32 row-major (candidate block ids, sentinel tail), all
    DRAM-interleaved; ``pin`` the per-chip ``V41ChunkTables.query_first()`` or a per-row page (``_pin_rows``:
    ``p_i = start + pin[i]``, every row visible to a sentinel entry); ``start`` the chunk start; ``rows_visible``
    the index-K rows any query of the chunk can see (a multiple of 8, at most the rows from the slot's block 0).
    -> bf16 row-major [1, 1, R, K * 8]: element 8 r + j of row i is ``sum_h w[i, h] * relu(q[i, h]. k[t])`` for
    ``t = 8 ids[i, r] + j``, -inf where ``ids[i, r]`` is the sentinel or ``t > p_i`` (the query's position,
    ``start + query_first + i``), i.e. ``gather_runs(score, ids, 8)`` of the full masked score. A row-major cache
    is first copied into blocks of 8 rows per page ([rows_visible / 8, 8 * 128]: a whole block is one 2 KB DRAM
    read; random 256-byte row reads are bound by the DRAM request rate, measured 20 ms vs ~5 ms of bandwidth for
    640 x 16,384 rows); then one op (kernels/candscore_*): per query its candidate blocks are gathered and tilized,
    ``relu(Q @ K^T)`` rounded to bf16 per head, and the heads summed by ``W @ S`` in fp32 (HiFi4). ``base_page``
    (uint32 row-major [1, 1, 1, P], P >= 32): the page of the slot's block 0 is read from its word 1 on device
    instead of ``block_base`` (the one-live-row capture: the live user is a per-step input, not
    a constant of the capture); ``ids``, ``pin`` and ``rows_visible`` stay the slot's own, as with ``block_base``."""
    for t, dtype, what in ((q, ttnn.bfloat16, "query"), (w, ttnn.bfloat16, "weights"), (k, ttnn.bfloat16, "index-K")):
        _check_row_major(t, dtype, f"candidate_scores {what}")
    _check_row_major(ids, ttnn.uint32, "candidate_scores ids")
    dyn = base_page is not None
    if dyn:
        _check_row_major(base_page, ttnn.uint32, "candidate_scores base page")
        assert base_page.shape[2] == 1 and base_page.shape[3] >= 32 and block_base == 0, (
            tuple(base_page.shape),
            block_base,
        )
    rows, nk = q.shape[2], ids.shape[3]
    pin_rows = _pin_rows(pin, rows)
    pin_slot = _round_up(max(32, pin_rows * 4), 64)
    assert q.shape[3] == 32 * 128 and (w.shape[2], w.shape[3]) == (rows, 32) and ids.shape[2] == rows
    assert rows_visible % CANDSCORE_BLOCK == 0, rows_visible
    tile_rows = nk * CANDSCORE_BLOCK // 32
    assert tile_rows * 32 == nk * CANDSCORE_BLOCK, f"{nk} candidate blocks do not fill whole tiles"
    batch = _largest_divisor(tile_rows)
    per_read = _largest_divisor(tile_rows, (8, 4, 2, 1))
    nblocks = rows_visible // CANDSCORE_BLOCK
    if k.shape[3] == CANDSCORE_BLOCK * 128:  # block pages already (persistent): no copy
        assert (nblocks if dyn else block_base + nblocks) <= k.shape[2], (block_base, nblocks, tuple(k.shape))
    else:
        assert k.shape[3] == 128 and block_base == 0 and rows_visible <= k.shape[2], (tuple(k.shape), rows_visible)
        if rows_visible < k.shape[2]:
            k = ttnn.slice(k, [0, 0, 0, 0], [1, 1, rows_visible, 128])
        k = ttnn.reshape(k, [1, 1, nblocks, CANDSCORE_BLOCK * 128])
    device = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, nk * CANDSCORE_BLOCK]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    coords, cores = _cores(device, rows)
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for (cx, cy), (first, count) in zip(coords, _split(rows, len(coords))):
        reader_args[cx][cy] = [
            q.buffer_address(),
            w.buffer_address(),
            ids.buffer_address(),
            k.buffer_address(),
            first,
            count,
            nblocks,
            block_base,
            base_page.buffer_address() if dyn else 0,
        ]
        writer_args[cx][cy] = [
            ids.buffer_address(),
            out.buffer_address(),
            pin.buffer_address(),
            first,
            count,
            start,
            nblocks,
        ]
        compute_args[cx][cy] = [count]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/candscore_reader.cpp",
            core_ranges=cores,
            compile_time_args=[0, 1, 2, 3, nk, per_read, int(dyn)]
            + ttnn.TensorAccessorArgs(q).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
            + ttnn.TensorAccessorArgs(k).get_compile_time_args()
            + ttnn.TensorAccessorArgs(base_page if dyn else ids).get_compile_time_args(),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/candscore_writer.cpp",
            core_ranges=cores,
            compile_time_args=[16, 8, nk, pin_rows]
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pin).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/candscore_compute.cpp",
            core_ranges=cores,
            compile_time_args=[nk, batch],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]
    writer_scratch = 64 + pin_slot + nk * 4 + 2 * nk * CANDSCORE_BLOCK * 2  # 64: the writer aligns the pin page
    cbs = [
        _cb(0, 2 * 4 * _TILE_BYTES, cores),  # query rows (two queries)
        _cb(1, 2 * _TILE_BYTES, cores),  # head-weight row pages (row 0; rows 1-31 zero)
        _cb(2, 2 * 4 * per_read * _TILE_BYTES, cores),  # gathered index-K rows
        _cb(3, nk * 4, cores, dtype=ttnn.uint32, page=nk * 4),  # reader: the query's ids
        _cb(4, 4 * _TILE_BYTES, cores),  # Q tiles [heads, dims]
        _cb(5, _TILE_BYTES, cores),  # W tile
        _cb(6, 4 * batch * _TILE_BYTES, cores),  # K tiles [rows, dims]
        _cb(7, batch * _TILE_BYTES, cores),  # S = relu(Q K^T) [heads, rows]
        _cb(16, 2 * batch * _TILE_BYTES, cores),  # O = W S (row 0)
        _cb(8, writer_scratch, cores, page=writer_scratch),  # writer: pin | ids | two output rows
    ]
    inputs = [q, w, k, ids, pin] + ([base_page] if dyn else []) + [out]
    return ttnn.generic_op(inputs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


RANGE_CHUNK = 8  # key tile rows (of 32 keys) per range_scores chunk: 64 KB of keys read, 512 B of score written
RANGE_ROWS = int(os.environ.get("V41_RANGE_ROWS", "32"))  # rows (cores) a range_scores launch splits the keys over
SPLIT_MIN_WIDTH = 512  # a split row must hold a top-k of INDEX_TOPK (topk_large_indices: n >= k)
# up to this width the score is one row: the one-core selection over it (~4.5 us per 1K columns for a top-k, as much
# again for the block maxima) costs less than the split selection's merge (~25 small ops, ~3 ms a 40-layer step at
# 16K)
SPLIT_ONE_ROW_MAX_WIDTH = 1 << 16


def split_geometry(width: int, rows: int | None = None) -> tuple[int, int, int, int]:
    """The split layout of a ``width``-wide score over ``rows`` rows -> (rows, chunks, chunks per row, row width):
    the key tile rows in chunks of RANGE_CHUNK, chunk ``c`` at row ``c % rows``, slot ``c // rows``; the rows are a
    power of two, at most RANGE_ROWS, and wide enough for a top-k of SPLIT_MIN_WIDTH. Element ``(r, j)`` of the
    layout is key ``((j // CW) * rows + r) * CW + j % CW`` with ``CW = RANGE_CHUNK * 32``; slots past the last
    chunk are -inf."""
    assert width % 32 == 0, width
    chunk_cols = RANGE_CHUNK * 32
    nchunks = -(-width // chunk_cols)
    if rows is None:
        rows = 1 if width <= SPLIT_ONE_ROW_MAX_WIDTH else RANGE_ROWS
    rows = min(rows, max(1, nchunks * chunk_cols // SPLIT_MIN_WIDTH))
    rows = 1 << (rows.bit_length() - 1)  # a power of two: the selection's id arithmetic is shifts and masks
    per_row = -(-nchunks // rows)
    return rows, nchunks, per_row, per_row * chunk_cols


def range_scores(
    q, w, k, pin, user: int, width: int, rows: int | None = None, dyn: bool = False, row: int | None = None
):
    """One user's query of the decode tile against the visible prefix of its own index keys, in the pool's tiled
    form (the single-query scorer of the densely scored sources).

    ``q`` [1, 1, R, 32 * 128] bf16 row-major and ``w`` [1, 1, R, 32] bf16 row-major (``TtV41Indexer.decode_query``;
    row ``row`` is read, default ``user``: a verify row of speculative decode is a tile row of its user's keys
    other than the user's own, with its own pin entry), ``k`` [U, 1, width, 128] bf16 TILE_LAYOUT DRAM-interleaved (the pool's ``index_k[src]``:
    user ``u``'s key tile row ``g`` is the tile pages ``(u * width / 32 + g) * 4 + d``), ``pin`` [1, 1, 1, P] uint32
    row-major with ``P > user`` (the step's ``newest`` page: entry ``user`` the user's newest visible row, the
    sentinel for none). -> bf16 row-major [1, 1, rows, row width] in the split layout of ``split_geometry(width,
    rows)``: key ``t``'s element is ``sum_h w[user, h] * relu(q[user, h]. k[user, t])`` (``candidate_scores``'s
    arithmetic) for ``t <= newest``, -inf past it (and everywhere for the sentinel). Chunk ``c`` of key tile rows is
    scored by core ``c % rows`` into slot ``c // rows`` of its row; the bound is read on device, so only the visible
    chunks are read and computed and the op is trace-safe. One op (kernels/rangescore_*). ``dyn``: the user and its
    newest row are words 0 and 2 of ``pin`` (a per-step host page; ``user`` is ignored): the one-live-row capture's
    selection of the live user alone, whichever slot it is."""
    for t, what in ((q, "query"), (w, "weights")):
        _check_row_major(t, ttnn.bfloat16, f"range_scores {what}")
    _check_row_major(pin, ttnn.uint32, "range_scores pin")
    memory = k.memory_config()
    assert k.layout == ttnn.TILE_LAYOUT and k.dtype == ttnn.bfloat16, "range_scores index-K must be tiled bf16"
    assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), "index-K must be DRAM-interleaved"
    assert width % 32 == 0 and (k.shape[2], k.shape[3]) == (width, 128), (width, tuple(k.shape))
    row = user if row is None else row
    assert 0 <= user < k.shape[0] and q.shape[2] > row and w.shape[2] > row, (user, row, tuple(k.shape))
    assert q.shape[3] == 32 * 128 and w.shape[3] == 32, (tuple(q.shape), tuple(w.shape))
    assert pin.shape[2] == 1 and pin.shape[3] > row, (tuple(pin.shape), row)
    pin_rows = pin.shape[3]
    pin_slot = _round_up(pin_rows * 4, 64)
    tile_rows = width // 32
    chunk = RANGE_CHUNK
    nrows, _, per_row, row_width = split_geometry(width, rows)
    device = q.device()
    coords, core_set = _cores(device, nrows)
    assert len(coords) == nrows, (len(coords), nrows)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, nrows, row_width]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(coords):
        reader_args[cx][cy] = [
            q.buffer_address(),
            w.buffer_address(),
            k.buffer_address(),
            pin.buffer_address(),
            user,
            i,
            row,
        ]
        writer_args[cx][cy] = [out.buffer_address(), pin.buffer_address(), user, i, row]
        compute_args[cx][cy] = [i]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rangescore_reader.cpp",
            core_ranges=core_set,
            compile_time_args=[0, 1, 2, 3, chunk, tile_rows, nrows, pin_rows, int(dyn)]
            + ttnn.TensorAccessorArgs(q).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(k).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pin).get_compile_time_args(),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rangescore_writer.cpp",
            core_ranges=core_set,
            compile_time_args=[16, 8, chunk, tile_rows, nrows, pin_rows, per_row, int(dyn)]
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pin).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rangescore_compute.cpp",
            core_ranges=core_set,
            compile_time_args=[chunk, tile_rows, nrows],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]
    meta = 64 + pin_slot  # the reader aligns the pin page inside it; word 0 is the visible tile-row count
    writer_scratch = 64 + pin_slot + 2 * chunk * 64  # 64: the writer aligns the pin page; two chunk buffers
    cbs = [
        _cb(0, 4 * _TILE_BYTES, core_set),  # the query's row-major [32 heads, 128] block (4 tile-sized pages)
        _cb(1, 2 * _TILE_BYTES, core_set),  # head-weight row page (row 0; rows 1-31 zero)
        _cb(2, 2 * 4 * chunk * _TILE_BYTES, core_set),  # K tiles [rows, dims] of two chunks
        _cb(3, meta, core_set, dtype=ttnn.uint32, page=meta),  # reader -> compute: the visible tile rows
        _cb(4, 4 * _TILE_BYTES, core_set),  # Q tiles [heads, dims]
        _cb(5, _TILE_BYTES, core_set),  # W tile
        _cb(7, chunk * _TILE_BYTES, core_set),  # S = relu(Q K^T) [heads, rows]
        _cb(16, 2 * chunk * _TILE_BYTES, core_set),  # O = W S (row 0)
        _cb(8, writer_scratch, core_set, page=writer_scratch),  # writer: pin | two chunk buffers
    ]
    return ttnn.generic_op([q, w, k, pin, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
