# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_post_rows``: the GDN prefill rows chain after the chunk scan, as two ``generic_op`` programs.

The chain it mirrors is ``_gate_and_project_rows`` (``ttnn/gdn.py``) from the scan's output to the gate, plus the
history tile ``commit_rows_full`` builds for the next pass: ``typecast(o fp32 -> bf16)``, ``rms_norm(weight=norm,
epsilon=1e-6)``, the twelve head slices and the width concat, ``multiply(normalized, sigmoid_bf16)``, and the last
tile row's q|k|v rows 29..31 moved into a fresh 32-row tile.  The z sigmoid is NOT here: it does not depend on the
scan, so ``gdn_pre_rows`` produces it into the persistent ``sig`` buffer and this program takes it as an input.

Two programs, because a compute kernel has one destination width and the chain's two arithmetic ops disagree:

* ``post_cast`` runs ``ttnn.typecast(fp32 -> bf16)``, which the op runs in a 32-bit destination with the input
  unpacked straight to it (``preserve_fp32_precision`` is set for an fp32 input);
* ``post_norm`` runs the weighted RMSNorm and the gate, which the ops run in a 16-bit destination (the rms_norm
  default compute config is ``math_approx_mode=True``, ``fp32_dest_acc_en=False``).

Both compute kernels are the ops' own LLK call sequences with the ops' operand order and one pack per op, so the
class is bitwise by construction; the head fold, the row mask and the history tile are data movement.  The gate's
proof is the device test's ``chain_on_device`` (the chain's ttnn ops on the same inputs), not torch: ``reference``
below is the structural bound only.

Page maps (per device local shapes; T = 32 * NC):

===================  ========================  =========================================
tensor               shape                     page of a tile
===================  ========================  =========================================
``o`` / ``o16``      ``[12, T, 128]``          ``(h * NC + c) * 4 + d``   (head-major)
``sig`` / ``gated``  ``[1, 1, T, 1536]``       ``c * 48 + 4 * h + d``     (token-major)
``projected``        ``[1, 1, T, 4160]``       ``c * 130 + column``
``history_next``     ``[1, 1, 32, 2560]``      ``column``
===================  ========================  =========================================

A work unit of both programs is one (value head, tile row) = the four column tiles of one head's 128 columns, and
unit ``u = h * NC + c`` so a core's run is contiguous in the head-major order and the ten history units (the last
tile row, heads 0..9) land on ten different cores.
"""

from __future__ import annotations

import struct

import torch

import ttnn

from .. import program as fp
from ..gdn_rows_reference import (
    DEFAULT as DEFAULT_ROUNDING,
    HEAD_DIM,
    HEADS,
    HISTORY_ROWS,
    PROJECTION_WIDTH,
    QKV_WIDTH,
    RMS_NORM_EPS,
    TILE,
    VALUE_WIDTH,
    fold_heads,
    history_next,
    multiply_bf16,
    post_reference,
    rms_norm_bf16,
    typecast_to_bf16,
)

NAME = "gdn_post_rows"
READER_CAST = fp.kernel_source(NAME, "reader_cast.cpp")
COMPUTE_CAST = fp.kernel_source(NAME, "compute_cast.cpp")
WRITER_CAST = fp.kernel_source(NAME, "writer_cast.cpp")
READER_NORM = fp.kernel_source(NAME, "reader_norm.cpp")
COMPUTE_NORM = fp.kernel_source(NAME, "compute_norm.cpp")
WRITER_NORM = fp.kernel_source(NAME, "writer_norm.cpp")

BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE_BF16, TILE_FP32 = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
FACE = fp.FACE

HEAD_TILES = HEAD_DIM // TILE  # 4: the Wt of the norm
VALUE_TILES = VALUE_WIDTH // TILE  # 48
QKV_TILES = QKV_WIDTH // TILE  # 80
PROJECTION_TILES = PROJECTION_WIDTH // TILE  # 130
HISTORY_COLUMNS = 8  # projection column tiles per history unit
HISTORY_HEADS = QKV_TILES // HISTORY_COLUMNS  # 10 of the 12 heads carry the history
SCALAR_TILES = 2  # the reduce scaler tile and the eps tile, side by side
RECIP_HEAD_DIM_BITS = 0x3C000000  # bit_cast<uint32_t>(1.0f / 128), numeric.h's row_wise_mean epilogue

# (index, dtype, pages); the indices must match the kernels' CB_* constants
CB_CAST_IN, CB_CAST_OUT = 0, 16
CAST_CBS = ((CB_CAST_IN, FP32, 2 * HEAD_TILES), (CB_CAST_OUT, BF16, 2 * HEAD_TILES))
# fp32 CBs the compute consumes with copy_tile: the typecast op unpacks its fp32 input straight to the destination
CAST_FP32_COPY_CBS = (CB_CAST_IN,)

CB_X, CB_SIG, CB_SCALER, CB_EPS, CB_GAMMA, CB_PROJ = 0, 1, 2, 3, 4, 5
CB_XMM2, CB_EX2, CB_EX2PE, CB_FUSION, CB_NRM = 6, 7, 8, 9, 10
CB_OUT, CB_HIST = 16, 17
NORM_CBS = (
    (CB_X, BF16, 2 * HEAD_TILES),
    (CB_SIG, BF16, 2 * HEAD_TILES),
    (CB_SCALER, BF16, 1),
    (CB_EPS, BF16, 1),
    (CB_GAMMA, BF16, HEAD_TILES),
    (CB_PROJ, BF16, 2 * HISTORY_COLUMNS),
    (CB_XMM2, BF16, HEAD_TILES),
    (CB_EX2, BF16, 1),
    (CB_EX2PE, BF16, 1),
    (CB_FUSION, BF16, HEAD_TILES),
    (CB_NRM, BF16, HEAD_TILES),
    (CB_OUT, BF16, 2 * HEAD_TILES),
    (CB_HIST, BF16, HISTORY_COLUMNS),
)


# --------------------------------------------------------------------------------------------- page maps and units


def head_tile_page(head: int, chunk: int, column: int, chunks: int) -> int:
    """The page of tile (head, tile row, column tile) of a head-major ``[12, T, 128]`` TILE tensor."""

    return (head * chunks + chunk) * HEAD_TILES + column


def token_tile_page(chunk: int, head: int, column: int) -> int:
    """The page of head ``head``'s column tile ``column`` in a token-major ``[1, 1, T, 1536]`` TILE tensor."""

    return chunk * VALUE_TILES + head * HEAD_TILES + column


def projection_tile_page(chunk: int, column: int) -> int:
    """The page of column tile ``column`` of tile row ``chunk`` in ``[1, 1, T, 4160]``."""

    return chunk * PROJECTION_TILES + column


def units(chunks: int) -> list[tuple[int, int]]:
    """The (value head, tile row) work units in program order: unit ``u`` is ``(u // NC, u % NC)``, so a core's run
    of units is a contiguous run of head-major pages and the history units are ``NC`` apart."""

    if chunks < 1:
        raise ValueError(f"chunks must be at least 1, got {chunks}")
    return [(u // chunks, u % chunks) for u in range(HEADS * chunks)]


def history_units(chunks: int) -> dict[tuple[int, int], tuple[int, ...]]:
    """The units that also write the history tile: the last tile row's heads 0..9, unit (h, NC - 1) taking the
    projection's q|k|v column tiles ``8h .. 8h + 7``.  Heads 10 and 11 write no history."""

    return {
        (head, chunks - 1): tuple(range(head * HISTORY_COLUMNS, (head + 1) * HISTORY_COLUMNS))
        for head in range(HISTORY_HEADS)
    }


# ------------------------------------------------------------------------------------- the two constant tiles


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16_of_bits(bits: int) -> torch.Tensor:
    """The bf16 value whose 16 bits are ``bits`` (torch has no unsigned 16-bit scalar to view through)."""

    signed = bits - 0x10000 if bits >= 0x8000 else bits
    return torch.tensor([signed], dtype=torch.int16).view(torch.bfloat16)[0]


def scalar_tiles(eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """The two constant tiles the layernorm reader generates, as one ``[1, 1, 32, 64]`` bf16 host tensor: tile 0 the
    SUM / REDUCE_ROW scaler, tile 1 the eps tile.

    ``calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`` (reduce_helpers_dataflow.inl) zeroes the tile and writes
    an exact 1.0 into row 0 of every face, which is rows 0 and 16 of the whole tile across all 32 columns.
    ``generate_bcast_col_scalar`` (generate_bcast_scalar_metal2.hpp) writes ``bits(eps) >> 16`` -- a TRUNCATION to
    bf16, not a rounding -- into column 0 of every row.  Both are read by this program from DRAM instead of being
    generated on the RISC, so the values are pinned by a test rather than by a second copy of the fill loops."""

    scaler = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    scaler[0] = 1.0
    scaler[FACE] = 1.0
    epsilon = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    epsilon[:, 0] = _bf16_of_bits(_bits(eps) >> 16)
    return torch.cat([scaler, epsilon], dim=1).reshape(1, 1, TILE, SCALAR_TILES * TILE)


def scalar_tensor(mesh, eps: float = RMS_NORM_EPS):
    """``scalar_tiles`` on the mesh (replicated: the tiles are the same on every device)."""

    return ttnn.from_torch(
        scalar_tiles(eps),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ------------------------------------------------------------------------------------------------- the programs


def _chunks_of(tensor, shape_label: str, chunks: int | None = None) -> int:
    rows = tensor.shape[-2]
    if rows % TILE:
        raise ValueError(f"{shape_label} must hold whole tile rows, got {rows}")
    found = rows // TILE
    if chunks is not None and found != chunks:
        raise ValueError(f"{shape_label} has {found} tile rows, expected {chunks}")
    return found


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def allocate_outputs(mesh, chunks: int, *, history: bool = True, gated_memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The programs' three output buffers for ``T = 32 * chunks`` rows: ``o16`` (post_cast's output and post_norm's
    input), ``gated`` and, with ``history``, ``history_next``.  Persistent per slab body, as the chain's rows state
    buffers are."""

    rows = chunks * TILE
    return {
        "o16": fp.allocate((HEADS, rows, HEAD_DIM), BF16, ttnn.TILE_LAYOUT, mesh),
        "gated": fp.allocate((1, 1, rows, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh, gated_memory_config),
        "history_next": (fp.allocate((1, 1, TILE, QKV_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh) if history else None),
    }


def post_cast(o, o16):
    """``ttnn.typecast(o, bfloat16)`` on the scan's output: ``o`` ``[12, T, 128]`` fp32 TILE -> ``o16`` the same
    shape bf16, page for page.  One unit = the four column tiles of one (value head, tile row)."""

    _require(
        tuple(o.shape)[:1] == (HEADS,) and o.shape[-1] == HEAD_DIM and o.dtype == FP32,
        f"post_cast input must be [{HEADS}, T, {HEAD_DIM}] fp32, got {tuple(o.shape)} {o.dtype}",
    )
    _require(
        tuple(o16.shape) == tuple(o.shape) and o16.dtype == BF16,
        f"post_cast output must be {tuple(o.shape)} bf16, got {tuple(o16.shape)} {o16.dtype}",
    )
    chunks = _chunks_of(o, "post_cast input")
    mesh = o.device()
    work = fp.split_work(HEADS * chunks, mesh)
    cores = fp.core_rectangle(work, mesh)

    cbs = [fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores) for index, dtype, pages in CAST_CBS]
    reader = fp.reader_kernel(
        READER_CAST,
        cores,
        fp.accessor_args(o),
        [(w.core, [o.buffer_address(), w.count, w.start]) for w in work],
    )
    writer = fp.writer_kernel(
        WRITER_CAST,
        cores,
        fp.accessor_args(o16),
        [(w.core, [o16.buffer_address(), w.count, w.start]) for w in work],
    )
    compute = fp.compute_kernel(
        COMPUTE_CAST,
        cores,
        [],
        [(w.core, [w.count]) for w in work],
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        approx=False,
        unpack_to_dest_fp32=CAST_FP32_COPY_CBS,
    )
    # every page of o read and of o16 written once; the typecast's explicit round-to-nearest-even is one
    # operation per element
    meta = fp.program_meta(
        NAME,
        "cast_verify" if chunks == 1 else "cast",
        chunks * TILE,
        reads=(o,),
        writes=(o16,),
        flops=HEADS * chunks * TILE * HEAD_DIM,
        cores=len(work),
    )
    return fp.run_program([o, o16], fp.program_descriptor([reader, writer, compute], cbs), meta=meta)


def post_norm(o16, sig, norm, scalars, projected, gated, history_next=None, *, rows=None, history=True):
    """The weighted RMSNorm, the head fold and the sigmoid gate, plus the next pass's history tile.

    ``o16`` ``[12, T, 128]`` bf16 (post_cast's output), ``sig`` ``[1, 1, T, 1536]`` bf16 (the pre program's
    ``bf16(sigmoid(fp32 z))``, token-major), ``norm`` ``[1, 1, 1, 128]`` bf16 (the gamma row), ``scalars`` from
    ``scalar_tensor``, ``projected`` ``[1, 1, T, 4160]`` bf16 (only the last tile row's q|k|v column tiles are read,
    and only when ``history``).  Writes ``gated`` ``[1, 1, T, 1536]`` bf16 token-major and, with ``history``,
    ``history_next`` ``[1, 1, 32, 2560]`` bf16.  ``rows`` (default all of T) is the MTP verify form's valid row
    count: the rest of the partial tile row is written as exact zeros, as the chain's row-mask multiply leaves it."""

    chunks = _chunks_of(o16, "post_norm o16")
    _require(
        tuple(o16.shape) == (HEADS, chunks * TILE, HEAD_DIM) and o16.dtype == BF16,
        f"post_norm o16 must be [{HEADS}, T, {HEAD_DIM}] bf16, got {tuple(o16.shape)} {o16.dtype}",
    )
    for label, tensor in (("sig", sig), ("gated", gated)):
        _require(
            tuple(tensor.shape) == (1, 1, chunks * TILE, VALUE_WIDTH) and tensor.dtype == BF16,
            f"post_norm {label} must be [1, 1, {chunks * TILE}, {VALUE_WIDTH}] bf16, got "
            f"{tuple(tensor.shape)} {tensor.dtype}",
        )
    _require(
        tuple(norm.shape) == (1, 1, 1, HEAD_DIM) and norm.dtype == BF16,
        f"post_norm weight must be [1, 1, 1, {HEAD_DIM}] bf16, got {tuple(norm.shape)} {norm.dtype}",
    )
    _require(
        tuple(scalars.shape) == (1, 1, TILE, SCALAR_TILES * TILE) and scalars.dtype == BF16,
        f"post_norm scalars must be [1, 1, {TILE}, {SCALAR_TILES * TILE}] bf16 (scaler | eps), got "
        f"{tuple(scalars.shape)} {scalars.dtype}",
    )
    _require(
        tuple(projected.shape) == (1, 1, chunks * TILE, PROJECTION_WIDTH) and projected.dtype == BF16,
        f"post_norm projection must be [1, 1, {chunks * TILE}, {PROJECTION_WIDTH}] bf16, got "
        f"{tuple(projected.shape)} {projected.dtype}",
    )
    if history:
        _require(
            history_next is not None
            and tuple(history_next.shape) == (1, 1, TILE, QKV_WIDTH)
            and history_next.dtype == BF16,
            f"post_norm history tile must be [1, 1, {TILE}, {QKV_WIDTH}] bf16",
        )
    rows = chunks * TILE if rows is None else rows
    _require(1 <= rows <= chunks * TILE, f"rows must be in [1, {chunks * TILE}], got {rows}")

    mesh = o16.device()
    # With history off there is no second output to bind: the slot repeats the gated tensor and the kernels never
    # touch it (the flag is 0), the way the stream reader's unused accessor slots repeat a used tensor's.
    hist_out = history_next if history else gated
    work = fp.split_work(HEADS * chunks, mesh)
    cores = fp.core_rectangle(work, mesh)

    cbs = [fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores) for index, dtype, pages in NORM_CBS]
    reader_cta = []
    for tensor in (o16, sig, norm, scalars, projected):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        o16.buffer_address(),
        sig.buffer_address(),
        norm.buffer_address(),
        scalars.buffer_address(),
        projected.buffer_address(),
    ]
    writer_cta = [*fp.accessor_args(gated), *fp.accessor_args(hist_out)]
    writer_addrs = [gated.buffer_address(), hist_out.buffer_address()]
    flag = 1 if history else 0

    reader = fp.reader_kernel(
        READER_NORM,
        cores,
        reader_cta,
        [(w.core, [*reader_addrs, chunks, flag, w.count, w.start]) for w in work],
    )
    writer = fp.writer_kernel(
        WRITER_NORM,
        cores,
        writer_cta,
        [(w.core, [*writer_addrs, chunks, rows, flag, w.count, w.start]) for w in work],
    )
    compute = fp.compute_kernel(
        COMPUTE_NORM,
        cores,
        [],
        [(w.core, [w.count]) for w in work],
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=False,
        approx=False,
    )
    io = [o16, sig, norm, scalars, projected, gated]
    if history:
        io.append(history_next)
    # o16, sig and gated a page per unit (each exactly once), the four gamma tiles and the scaler and eps tiles
    # once per core, and, with the history on, the eighty q|k|v column tiles of the last tile row of the
    # projection and the eighty pages of history_next.  One operation per element for the norm's square, row sum,
    # rsqrt multiply and gamma multiply and for the gate's multiply.
    meta = fp.program_meta(
        NAME,
        "norm" if history else "norm_verify",
        int(rows),
        reads=(o16, sig),
        writes=(gated, history_next) if history else (gated,),
        partial=((projected, HISTORY_HEADS * HISTORY_COLUMNS * fp.TILE_BYTES[BF16]),) if history else (),
        dram_bytes=len(work) * (HEAD_TILES + SCALAR_TILES) * fp.TILE_BYTES[BF16],
        flops=5 * HEADS * chunks * TILE * HEAD_DIM,
        cores=len(work),
    )
    fp.run_program(io, fp.program_descriptor([reader, writer, compute], cbs), meta=meta)
    return gated, (history_next if history else None)


# ------------------------------------------------------------------------------------------------- the references


def reference(o, z_or_sig, norm, projected, rows=None, history=True, *, sigmoid_applied=False, rounding=None):
    """The two programs in torch: ``gdn_rows_reference.post_reference`` plus ``history_next`` and the row mask.

    ``o`` ``[12, T, 128]`` fp32, ``z_or_sig`` ``[T, 1536]`` bf16 (the raw z, or the pre program's ``sig`` with
    ``sigmoid_applied``), ``norm`` ``[128]`` or ``[1, 128]`` bf16, ``projected`` ``[T, 4160]`` bf16.  Returns
    ``{"gated": [T, 1536] bf16, "history_next": [32, 2560] bf16 or None}``.  STRUCTURAL: the norm's device chain is
    a 16-bit-destination LLK sequence no torch expression reproduces (see ``rms_norm_bf16``), so this bounds the
    kernels to a few bf16 ulps; the bitwise oracle is ``chain_on_device``."""

    rounding = DEFAULT_ROUNDING if rounding is None else rounding
    total = o.shape[1]
    if sigmoid_applied:
        heads = typecast_to_bf16(o, rounding)
        normalized = rms_norm_bf16(heads, RMS_NORM_EPS, weight=norm, rounding=rounding)
        gated = multiply_bf16(fold_heads(normalized), z_or_sig, rounding)
    else:
        gated = post_reference(o, z_or_sig, norm, rounding)
    if rows is not None and rows < total:
        gated = gated.clone()
        gated[rows:] = 0.0  # the chain's row-mask multiply: x * 0.0 = +0 under the bf16 multiply's clamp
    return {
        "gated": gated,
        "history_next": history_next(projected[:, :QKV_WIDTH]) if history else None,
    }


def chain_on_device(mesh, o, z, norm, projected, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Today's chain on the device, op for op: ``_gate_and_project_rows``' slab branch (``ttnn/gdn.py``) from the
    scan's output to the gated rows, and ``commit_rows_full``'s history tile.

    Returns ``(gated [1, 1, T, 1536], history [1, 1, 32, 2560], sigmoid_bf16 [1, 1, T, 1536])``.  The sigmoid is
    returned because ``gdn_post_rows`` takes it as an input (the pre program owns it), so the two arms must be fed
    the same bits.  The mesh-topology retags of the chain are bookkeeping, not arithmetic, and are left out."""

    tile_rows = o.shape[-2]
    head_rows = ttnn.reshape(o, (1, HEADS, tile_rows, HEAD_DIM))
    head_rows_bf16 = ttnn.typecast(head_rows, ttnn.bfloat16, memory_config=memory_config)
    normalized_heads = ttnn.rms_norm(head_rows_bf16, weight=norm, epsilon=RMS_NORM_EPS, memory_config=memory_config)
    z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=memory_config)
    sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=memory_config)
    sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=memory_config)
    heads = [
        ttnn.slice(normalized_heads, (0, head, 0, 0), (1, head + 1, tile_rows, HEAD_DIM), memory_config=memory_config)
        for head in range(HEADS)
    ]
    normalized = ttnn.concat(heads, dim=3, memory_config=memory_config)
    gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=memory_config)

    last_tile = ttnn.slice(
        projected, (0, 0, tile_rows - TILE, 0), (1, 1, tile_rows, QKV_WIDTH), memory_config=memory_config
    )
    last_rm = ttnn.to_layout(last_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    tail = ttnn.slice(last_rm, (0, 0, TILE - HISTORY_ROWS, 0), (1, 1, TILE, QKV_WIDTH), memory_config=memory_config)
    padded = ttnn.pad(tail, [(0, 0), (0, 0), (0, TILE - HISTORY_ROWS), (0, 0)], 0.0, memory_config=memory_config)
    history = ttnn.to_layout(padded, ttnn.TILE_LAYOUT, memory_config=memory_config)

    for tensor in (
        head_rows_bf16,
        z_fp32,
        sigmoid_fp32,
        normalized_heads,
        *heads,
        normalized,
        last_tile,
        last_rm,
        tail,
        padded,
    ):
        ttnn.deallocate(tensor)
    return gated, history, sigmoid_bf16
