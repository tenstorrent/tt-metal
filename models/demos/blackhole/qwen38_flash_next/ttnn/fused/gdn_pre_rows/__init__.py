# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_pre_rows``: the GDN prefill rows chain from the projection to the chunk prims' inputs, one program.

Replaces the slab branch's P1 landing slices, ``_shifted_rows_slab`` (the FIR row shifts), ``_causal_conv_rows``
(multiply + three ``mac`` + SiLU), ``_make_chunk_inputs`` (the GQA expand, both q/k rms_norms and scales, the v row
mask, the beta and log-decay gates) and the composite's own q scale and head-major relayout, plus the z sigmoid the
gated epilogue no longer has to wait for the scan to compute.  Every arithmetic step is the chain op's own LLK
sequence in the chain op's DEST width with one pack per op, so the outputs are the chain's bits; the row shifts, the
GQA copy, the page maps and the beta/g column transposes are data movement.

One program, two core groups on disjoint core rectangles, six kernels:

* **group A** (``reader_qkv`` / ``compute_qkv`` / ``writer_qkv``, ``fp32_dest=False``, ``approx=False``, HiFi4): the
  20 column groups of four tiles (q key heads 0..3, k key heads 0..3, v value heads 0..11) x the tile rows.  Per
  unit: the three shifted taps by exact 0/1 selection matmuls, the conv mirror (``mul_binary_tile`` then three
  ``mac_tile<Float16_b>`` against the materialised row-broadcast tap tiles), ``silu_tile``; for q/k the
  ``ttnn.rms_norm`` mirror in the 16-bit DEST and the ``x 128^-0.5`` scale (twice for q: the chain's and the
  composite's).  A core's units are consecutive tile rows of one column group, which is what lets the four tap
  tiles be read and broadcast once per group; the reader hands the compute one whole window per unit (the previous
  tile row, then this one), pushed and popped as one block so it is always contiguous in the CB.  A q unit costs
  more than a k unit and a k unit more than a v one (the norm and the scale passes), so the runs are cut by
  measured COST, not by unit count: ``UNIT_COST`` and :func:`_run_counts`.
* **group B** (``reader_gates`` / ``compute_gates`` / ``writer_gates``, ``fp32_dest=True``, ``INP_FLOAT32``,
  ``unpack_to_dest_fp32`` on the fp32 constant CB): per tile row the a/b gate unit (beta and the log decay, fp32
  throughout) and the 12 z units (``bf16(sigmoid(fp32(z)))``).

Per-core L1 (the CB table below; a Blackhole Tensix has 1.5 MB): **group A 190,464 B (186 KB)**, **group B 61,440 B
(60 KB)**.  Group A's three largest CBs are the 16 tap tiles, the 16 materialised broadcast tap tiles (both rebuilt
only when a core's run crosses a column group) and the 12-deep input window.

Circular-buffer indices are per program, so group A's 0..14 and group B's 20..26 only have to be disjoint from each
other (they are); ``gdn_post_rows``' own buffers are a different ``generic_op`` call and never coexist with these on a
core.  Kernel-side constant names avoid a bare ``BF16`` / ``FP32``: the JIT injects macros of those names on
Blackhole and a template argument spelled with them fails to parse (the sibling program hit it), so the data-format
constants here are ``DF_BF16`` / ``DF_FP32``.

What this module does NOT do: the history tile of the next pass (``commit_rows_full`` keeps it), the scan, and the
gated epilogue (``gdn_post_rows``).  No registry entry yet: the model wiring stage adds ``gdn_prefill_rows`` with
the composed chain.

``v`` runs the chain's own row-mask multiply (``ttnn.multiply(v_slice, row_mask_bf16_col)``, binary_ng's
column-broadcast SFPU kernel) rather than being written as the SiLU left it: that op carries the rows mask AND the
bf16 multiply's ``0 * x = +0`` clamp, so a ``v`` element the SiLU leaves at ``-0.0`` gets the chain's ``+0.0``.
q and k take their rows mask in the writer instead, because their chain mask multiplies a whole padded
``[1, T, 32, 128]`` token tile by ``0.0``, which the same clamp makes ``+0`` in every element -- the same bits as
zeroing the tile.  ``sig`` takes no mask at all: the chain's z sigmoid has none.
"""

from __future__ import annotations

import struct

import torch

import ttnn

from .. import gdn_rows_reference as ref
from .. import program as fp

NAME = "gdn_pre_rows"
KERNELS = {
    role: fp.kernel_source(NAME, f"{role}.cpp")
    for role in ("reader_qkv", "compute_qkv", "writer_qkv", "reader_gates", "compute_gates", "writer_gates")
}

BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE = ref.TILE
HEADS = ref.HEADS  # 12 value heads per device
QK_HEADS = ref.QK_HEADS  # 4 key heads per device
HEAD_DIM = ref.HEAD_DIM  # 128
QK_WIDTH = ref.QK_WIDTH  # 512
VALUE_WIDTH = ref.VALUE_WIDTH  # 1536
QKV_WIDTH = ref.QKV_WIDTH  # 2560
A_COLUMN = ref.A_COLUMN  # 4096
B_COLUMN = ref.B_COLUMN  # 4128
PROJECTION_WIDTH = ref.PROJECTION_WIDTH  # 4160
CONV_KERNEL = ref.CONV_KERNEL  # 4
HISTORY_ROWS = ref.HISTORY_ROWS  # 3
QK_SCALE = ref.QK_SCALE  # 128 ** -0.5
NORM_EPS = ref.QK_L2_NORM_EPS / HEAD_DIM  # the epsilon ttnn.rms_norm is given

HEAD_TILES = HEAD_DIM // TILE  # 4 column tiles per head
PROJECTION_TILES = PROJECTION_WIDTH // TILE  # 130 column tiles per tile row
QKV_TILES = QKV_WIDTH // TILE  # 80
VALUE_TILES = VALUE_WIDTH // TILE  # 48
K_TILE0 = QK_WIDTH // TILE  # 16
V_TILE0 = 2 * K_TILE0  # 32
Z_TILE0 = QKV_TILES  # 80
A_TILE = A_COLUMN // TILE  # 128
B_TILE = B_COLUMN // TILE  # 129

QK_GROUPS = 2 * QK_HEADS  # column groups 0..3 = q key heads, 4..7 = k key heads
GROUPS = QK_GROUPS + HEADS  # 20 column groups of four tiles
GATE_UNITS = 1 + HEADS  # per tile row: the a/b unit, then the 12 z units

# The nine 0/1 selection tiles, in the order ``selection_tiles`` stacks them (and the pages of ``selects``).
SELECT_CUR, SELECT_PREV, SELECT_HIST = 0, 1, 2
SELECT_KINDS = 3
SELECT_TILES = HISTORY_ROWS * SELECT_KINDS  # 9
# The three small tiles of ``scalars``, in page order.
SCALAR_REDUCE, SCALAR_EPS, SCALAR_SCALE = 0, 1, 2
SCALAR_TILES = 3
# The two full fp32 constant tiles of ``constants``, in page order.
CONST_DT_BIAS, CONST_NEG_EXP_A = 0, 1
CONST_TILES = 2

# Default: group B takes the last two grid columns and group A every column before them.  Measured on one chip at
# T = 2048 (the lane's perf tool, a dev tool: traced us per call, the outputs bitwise the chain at every point): one column
# 608.9 us, two 389.3, three 413.3, four 484.2, five 557.7, six 650.8.  At two columns group A alone is 388.5 and
# group B alone 317.2, so A is the critical path there and at every wider split while B is the critical path at one
# column: two is the minimum of the two curves, and the program is max(A, B) to within a microsecond.
GROUP_B_COLUMNS = 2


def bits(value: float) -> int:
    """The fp32 bit pattern of ``value`` as an unsigned 32-bit integer."""

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


EPS_BITS = bits(NORM_EPS)
ONE_OVER_W_BITS = bits(1.0 / HEAD_DIM)  # 0x3C000000: the layernorm reduce's 1/W scale
SCALE_BF16_BITS = bits(QK_SCALE) >> 16  # 0x3DB5: bfloat16(128 ** -0.5), rounded on the host by binary_ng

# ------------------------------------------------------------------------------------------------- the CB table
# (name, index, dtype, pages, group).  Indices must not collide across the two groups: both compute kernels declare
# the subset they use and the static test checks all six kernels against this table.
CBS = (
    ("in", 0, BF16, 4 * HEAD_TILES, "A"),  # two windows of (previous tile row | this one): pushed and popped whole
    ("sel", 1, BF16, SELECT_TILES, "A"),  # the nine selection tiles, pushed once
    ("tap", 2, BF16, CONV_KERNEL * HEAD_TILES, "A"),  # the four taps' four column tiles of the group
    ("tapfull", 3, BF16, CONV_KERNEL * HEAD_TILES, "A"),  # the same rows broadcast to full tiles
    ("shift", 4, BF16, HISTORY_ROWS, "A"),  # taps 0..2 of one column tile
    ("conv", 5, BF16, 2, "A"),  # the running FIR accumulator
    ("x", 6, BF16, HEAD_TILES, "A"),  # silu(conv): the unit's four column tiles
    ("xmm2", 7, BF16, HEAD_TILES, "A"),  # x * x
    ("scaler", 8, BF16, 1, "A"),  # the reduce scaler tile (1.0 in row 0 of every face), pushed once
    ("eps", 9, BF16, 1, "A"),  # the epsilon tile (column 0, truncated bf16), pushed once
    ("scale", 10, BF16, 1, "A"),  # bfloat16(128 ** -0.5) in every element, pushed once
    ("ex2", 11, BF16, 2, "A"),  # mean(x * x)
    ("ex2pe", 12, BF16, 2, "A"),  # rsqrt(mean + eps)
    ("unit", 13, BF16, HEAD_TILES, "A"),  # x * rsqrt(...)
    # what the writer emits, two units deep so the writer drains one unit while the compute runs the next
    ("out", 14, BF16, 2 * HEAD_TILES, "A"),
    ("mask", 15, BF16, 2, "A"),  # the chain's row_mask_bf16_col tile for a v unit's tile row
    ("maskfull", 16, BF16, 2, "A"),  # the same column broadcast across the tile, as binary_ng materialises it
    ("ab", 20, BF16, 2, "B"),  # the a and b column tiles of the tile row
    ("z", 21, BF16, HEAD_TILES, "B"),  # one z value head's four column tiles
    ("const", 22, FP32, CONST_TILES, "B"),  # dt_bias and neg_exp_A as full fp32 tiles, pushed once
    ("beta", 23, FP32, 2, "B"),
    ("g", 24, FP32, 2, "B"),
    ("sig", 25, BF16, 2 * HEAD_TILES, "B"),  # two units deep, as CB_OUT
    ("col", 26, FP32, 2, "B"),  # the writer's scratch: one head's 32 values down column 0 of a zeroed tile
)
CB_INDEX = {name: index for name, index, _dtype, _pages, _group in CBS}
CB_GROUP = {name: group for name, _index, _dtype, _pages, group in CBS}
# The fp32 CBs group B's compute reads with copy_tile: without this they would arrive through the 19-bit source path.
FP32_COPY_CBS = (CB_INDEX["const"],)

# The fixed leading runtime args of each kernel, in order (the per-unit pairs follow).  The static test reads the
# kernels' literal ``get_arg_val<uint32_t>(i)`` indices against these.
READER_QKV_ARGS = ("projected", "history", "tap0", "tap1", "tap2", "tap3", "selects", "scalars", "rows", "units")
COMPUTE_QKV_ARGS = ("units",)
WRITER_QKV_ARGS = ("q_c", "k_c", "v", "rows", "units")
READER_GATES_ARGS = ("projected", "constants", "units")
COMPUTE_GATES_ARGS = ("units",)
WRITER_GATES_ARGS = ("beta_c", "g_c", "sig", "rows", "units")


def l1_bytes(group: str) -> int:
    """The per-core L1 the group's circular buffers take."""

    return sum(pages * fp.TILE_BYTES[dtype] for _n, _i, dtype, pages, g in CBS if g == group)


# ------------------------------------------------------------------------------------------------- the page maps


def projected_page(chunk: int, column_tile: int) -> int:
    """Tile ``column_tile`` of tile row ``chunk`` of ``projected`` ``[1, 1, T, 4160]``."""

    return chunk * PROJECTION_TILES + column_tile


def group_first_tile(group: int) -> int:
    """The first of the four projection column tiles a group owns: q key head g, k key head g - 4, v value head
    g - 8."""

    if not 0 <= group < GROUPS:
        raise ValueError(f"column group must be in [0, {GROUPS}), got {group}")
    if group < QK_HEADS:
        return group * HEAD_TILES
    if group < QK_GROUPS:
        return K_TILE0 + (group - QK_HEADS) * HEAD_TILES
    return V_TILE0 + (group - QK_GROUPS) * HEAD_TILES


def group_kind(group: int) -> str:
    """``"q"``, ``"k"`` or ``"v"``: which output a column group feeds (and so whether it norms and scales)."""

    return "q" if group < QK_HEADS else ("k" if group < QK_GROUPS else "v")


def group_value_heads(group: int) -> tuple[int, ...]:
    """The value heads a group writes: a q/k key head expands to its three value heads (the GQA copy), a v group is
    its own value head."""

    kind = group_kind(group)
    if kind == "v":
        return (group - QK_GROUPS,)
    key_head = group if kind == "q" else group - QK_HEADS
    return tuple(range(key_head * (HEADS // QK_HEADS), (key_head + 1) * (HEADS // QK_HEADS)))


def qk_page(head: int, chunk: int, column: int, chunks: int) -> int:
    """Tile ``column`` of (value head, chunk) of ``q_c`` / ``k_c`` ``[12, NC, 32, 128]``: what the prep reader's
    head-major read ``hc * Ct * Kt + t`` indexes with ``hc = head * NC + chunk``."""

    return (head * chunks + chunk) * HEAD_TILES + column


def vec_page(head: int, chunk: int, chunks: int) -> int:
    """The one tile of (value head, chunk) of ``beta_c`` / ``g_c`` ``[12, NC, 32, 1]`` (the reader's ``hc * Ct``)."""

    return head * chunks + chunk


def flat_page(chunk: int, head: int, column: int) -> int:
    """Tile ``column`` of value head ``head`` in tile row ``chunk`` of a token-major ``[1, 1, T, 1536]`` tensor (``v``
    and ``sig``): the prep reader's flat-v page ``(c * Ct + rt) * HV * Vt + hv * Vt + ct`` at ``Ct = 1``."""

    return chunk * VALUE_TILES + head * HEAD_TILES + column


def select_index(kind: int, shift: int) -> int:
    """The page of one selection tile: ``kind`` is ``SELECT_CUR`` / ``SELECT_PREV`` / ``SELECT_HIST`` and ``shift``
    the row shift s in 1..3."""

    if kind not in (SELECT_CUR, SELECT_PREV, SELECT_HIST) or not 1 <= shift <= HISTORY_ROWS:
        raise ValueError(f"no selection tile for kind {kind} shift {shift}")
    return (shift - 1) * SELECT_KINDS + kind


# ------------------------------------------------------------------------------------- the host-built constants


def selection_tiles() -> torch.Tensor:
    """The nine 0/1 selection tiles ``[9, 32, 32]`` bf16, ``select_index``-ordered.

    ``shifted_s`` -- the FIR input shifted down by s rows, from which tap 0 is ``shifted_3``, tap 1 ``shifted_2``,
    tap 2 ``shifted_1`` and tap 3 the rows themselves -- is ``Sel_prev(s) @ previous + Sel_cur(s) @ current`` for one
    column tile, two matmuls into one DEST tile.  Row r of the result is row ``32 c + r - s`` of the FIR input, which
    is row ``r - s`` of the current tile when ``r >= s`` (``Sel_cur``), row ``32 - s + r`` of the previous tile row
    when ``r < s`` (``Sel_prev``), and -- at tile row 0, where the previous tile is the history tile whose rows 0..2
    are the three rows before row 0 -- history row ``3 - s + r`` (``Sel_hist``).  Exactly one 1.0 per output element,
    so the HiFi4 fp32-accumulated matmul returns that bf16 value unchanged.
    """

    tiles = torch.zeros(SELECT_TILES, TILE, TILE)
    for shift in range(1, HISTORY_ROWS + 1):
        for row in range(TILE):
            if row >= shift:
                tiles[select_index(SELECT_CUR, shift), row, row - shift] = 1.0
            else:
                tiles[select_index(SELECT_PREV, shift), row, TILE - shift + row] = 1.0
                tiles[select_index(SELECT_HIST, shift), row, HISTORY_ROWS - shift + row] = 1.0
    return tiles.to(torch.bfloat16)


def _bf16_of_bits(pattern: torch.Tensor) -> torch.Tensor:
    """A bf16 tensor holding exactly these 16-bit patterns (``pattern`` int64 in 0..0xFFFF)."""

    signed = torch.where(pattern >= 0x8000, pattern - 0x10000, pattern).to(torch.int16)
    return signed.view(torch.bfloat16)


def scalar_tiles(eps_bits: int = EPS_BITS, scale_bits: int = SCALE_BF16_BITS) -> torch.Tensor:
    """The three small bf16 tiles ``[3, 32, 32]`` the norm and the scale need, each filled exactly as the op's own
    dataflow kernel fills it.

    * ``SCALAR_REDUCE``: the layernorm reader's ``calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`` on a zeroed
      tile fills row 0 of each of the four faces with 1.0 (``fill_each_face_row0``: 8 u32 words per face), which in
      (row, column) terms is rows 0 and 16, all 32 columns.  Every other element stays zero.
    * ``SCALAR_EPS``: ``generate_bcast_col_scalar(eps)`` writes ``bits >> 16`` -- the epsilon TRUNCATED to bf16, not
      rounded -- into column 0 of all 32 rows (u16 indices 16j and 512 + 16j).
    * ``SCALAR_SCALE``: binary_ng's scalar writer ``fill_with_val_bfloat16`` fills every element with the host's
      ``bfloat16(128 ** -0.5)`` = 0x3DB5 (RNE on the host, so NOT the fp32 0x3DB504F3).
    """

    pattern = torch.zeros(SCALAR_TILES, TILE, TILE, dtype=torch.int64)
    pattern[SCALAR_REDUCE, 0, :] = 0x3F80
    pattern[SCALAR_REDUCE, TILE // 2, :] = 0x3F80
    pattern[SCALAR_EPS, :, 0] = int(eps_bits) >> 16
    pattern[SCALAR_SCALE, :, :] = int(scale_bits) & 0xFFFF
    return _bf16_of_bits(pattern)


def constant_row_tiles(dt_bias: torch.Tensor, neg_exp_A: torch.Tensor) -> torch.Tensor:
    """The two full fp32 constant tiles ``[2, 32, 32]``: the row ``[dt_bias[0..11], 0 x 20]`` in all 32 rows, and the
    same for ``neg_exp_A``.

    The chain's ``ttnn.add(a_fp32, dt_bias)`` and ``ttnn.multiply(neg_exp_A, softplus)`` are binary_ng ROW-broadcast
    ops whose Blackhole reader fills the tile with ``fill_tile_with_first_row`` (a bit copy of row 0 into 32 rows)
    because the fp32 operands force ``fp32_dest_acc_en`` and with it ``use_llk_bcast = false``; both operands then
    reach the SFPU through ``UnpackToDestFp32``, exactly.  Building the same tile on the host is the same bits and
    keeps the constant off the RISC's critical path.
    """

    tiles = torch.zeros(CONST_TILES, TILE, TILE, dtype=torch.float32)
    tiles[CONST_DT_BIAS, :, :HEADS] = dt_bias.float().reshape(-1)[:HEADS]
    tiles[CONST_NEG_EXP_A, :, :HEADS] = neg_exp_A.float().reshape(-1)[:HEADS]
    return tiles


def _upload(host: torch.Tensor, dtype, mesh, *, per_device: bool):
    """``host`` to the mesh: replicated, or with a leading device axis sharded over the mesh rows."""

    if per_device:
        return ttnn.from_torch(
            host.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 0)),
        )
    return ttnn.from_torch(
        host.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def constant_tiles(mesh, dt_bias, neg_exp_A):
    """The per-device ``constants`` tensor ``[1, 2, 32, 32]`` fp32 from the layer's device ``dt_bias`` and
    ``neg_exp_A`` ``[1, 1, 1, 12]`` fp32 (the gdn_step pattern: one upload per layer, cached and resident before a
    trace capture)."""

    def per_device(tensor):
        return [ttnn.to_torch(local).float().reshape(-1)[:HEADS] for local in ttnn.get_device_tensors(tensor)]

    host = torch.stack(
        [constant_row_tiles(dt, na) for dt, na in zip(per_device(dt_bias), per_device(neg_exp_A))]
    ).contiguous()
    return _upload(host, FP32, mesh, per_device=True)


def upload_constants(mesh, dt_bias, neg_exp_A, *, eps: float = NORM_EPS, scale: float = QK_SCALE):
    """Every constant the program reads, uploaded once per layer: ``(constants, selects, scalars)``.

    ``selects`` ``[1, 9, 32, 32]`` bf16 and ``scalars`` ``[1, 3, 32, 32]`` bf16 are the same on every device
    (replicated); ``constants`` carries the device's own ``dt_bias`` / ``neg_exp_A`` row.
    """

    selects = _upload(selection_tiles().reshape(1, SELECT_TILES, TILE, TILE), BF16, mesh, per_device=False)
    scalars = _upload(
        scalar_tiles(bits(eps), bits(scale) >> 16).reshape(1, SCALAR_TILES, TILE, TILE), BF16, mesh, per_device=False
    )
    return constant_tiles(mesh, dt_bias, neg_exp_A), selects, scalars


# -------------------------------------------------------------------------------------------- the work split


def group_a_units(chunks: int) -> list[tuple[int, int]]:
    """The (column group, tile row) units in COLUMN-MAJOR order, so a contiguous run is consecutive tile rows of one
    column group and every unit but the first of a run finds its FIR history -- the previous tile row -- in L1."""

    return [(group, chunk) for group in range(GROUPS) for chunk in range(chunks)]


def group_b_units(chunks: int) -> list[tuple[int, int]]:
    """The (tile row, kind) units of group B: kind 0 is the a/b gate unit of the tile row, kinds 1..12 its z value
    heads."""

    return [(chunk, kind) for chunk in range(chunks) for kind in range(GATE_UNITS)]


def _column_work(units: int, grid, first_column: int, cores: int) -> list[fp.CoreWork]:
    """``units`` items over ``cores`` cores laid out column-major from grid column ``first_column``."""

    cores = max(1, min(cores, units))
    base, extra = divmod(units, cores)
    work, start = [], 0
    for i in range(cores):
        count = base + (1 if i < extra else 0)
        work.append(fp.CoreWork(ttnn.CoreCoord(first_column + i // grid.y, i % grid.y), start, count))
        start += count
    return work


# The core time of one group A unit by the kind of its column group, in hundredths of a microsecond.  MEASURED on
# one chip at T = 2048 with the lane's perf tool (``--probe``: the program with group A's unit list made of a single
# kind and group B idle, over the 15 units a core then holds -- all q 393.20 us, all k 350.39, all v 315.49).  A q
# unit runs the rms_norm mirror and two scale passes, a k unit the norm and one, a v unit neither but the row-mask
# multiply, and the three costs are flat in NC because a unit is always one tile row of one column group.  Only the
# RATIO matters to the split below; whole numbers keep the cut exact and identical on every host.
UNIT_COST = {"q": 2621, "k": 2336, "v": 2103}


def group_a_weights(chunks: int) -> list[int]:
    """The cost weight of every unit of :func:`group_a_units`, in that order."""

    return [UNIT_COST[group_kind(group)] for group, _chunk in group_a_units(chunks)]


def group_a_weight_blocks(chunks: int) -> list[tuple[int, int]]:
    """:func:`group_a_weights` as ``(weight, units)`` blocks of equal weight.

    Column-major order puts the four q groups first, then the four k groups, then the twelve v groups, so group A
    is three blocks whatever NC is and the cut below is arithmetic on three items instead of a pass over every
    unit (the split is built on the host on every call).
    """

    blocks: list[tuple[int, int]] = []
    for group in range(GROUPS):
        weight = UNIT_COST[group_kind(group)]
        if blocks and blocks[-1][0] == weight:
            blocks[-1] = (weight, blocks[-1][1] + chunks)
        else:
            blocks.append((weight, chunks))
    return blocks


def _runs_needed(blocks, limit: int) -> int:
    """How many contiguous runs a greedy left-to-right fill of ``blocks`` takes when no run may weigh more than
    ``limit``; one more than the unit count (i.e. impossible) when a single unit is already over it."""

    units = sum(count for _weight, count in blocks)
    if any(weight > limit for weight, _count in blocks):
        return units + 1
    runs, load = 1, 0
    for weight, count in blocks:
        while count:
            fits = (limit - load) // weight  # how many of this block still fit in the run being filled
            if fits >= count:
                load += count * weight
                count = 0
            else:
                count -= fits
                runs += 1
                load = 0
    return runs


def _run_counts(blocks, cores: int) -> list[int]:
    """The unit counts of ``cores`` CONSECUTIVE runs over the units of ``blocks`` whose heaviest run is as light as
    a consecutive cut can make it.

    The linear partition: binary-search the smallest limit a greedy fill meets in ``cores`` runs, then lay the runs
    out under it.  The search is over whole numbers between the two bounds it cannot beat -- the heaviest unit and
    the mean run, neither of which any cut undercuts -- and the mean plus the heaviest unit, which greedy always
    meets (every run but the last is loaded past ``limit - heaviest``, so ``cores`` of them hold more than the
    total).  Every core gets a unit: splitting a run only lowers the maximum, so spending the rest of the grid is
    free, and the kernels take a core whose run crosses a column group anyway (they rebuild the tap tiles at every
    change, not only at the first unit).
    """

    units = sum(count for _weight, count in blocks)
    cores = max(1, min(cores, units))
    heaviest = max(weight for weight, _count in blocks)
    total = sum(weight * count for weight, count in blocks)
    mean = -(-total // cores)
    low, high = max(heaviest, mean), max(heaviest, mean) + heaviest
    while low < high:
        middle = (low + high) // 2
        if _runs_needed(blocks, middle) <= cores:
            high = middle
        else:
            low = middle + 1

    counts, index, offset, left = [], 0, 0, units
    for core in range(cores):
        room = left - (cores - core - 1)  # leave one unit for every core after this one
        load, taken = 0, 0
        while taken < room and index < len(blocks):
            weight, count = blocks[index]
            take = min((low - load) // weight, count - offset, room - taken)
            if taken == 0:
                take = max(take, 1)  # the first unit always fits: the limit is never under the heaviest one
            if take < 1:
                break
            load += take * weight
            taken += take
            offset += take
            if offset == count:
                index, offset = index + 1, 0
        counts.append(taken)
        left -= taken
    if left:  # unreachable at the limit the search returns; a run of the tail rather than a lost unit
        counts[-1] += left
    return counts


def _weighted_work(blocks, grid, first_column: int, cores: int) -> list[fp.CoreWork]:
    """:func:`_column_work`'s core order and consecutive runs over the same unit list, cut by accumulated COST
    instead of by unit count."""

    units = sum(count for _weight, count in blocks)
    work, start = [], 0
    for i, count in enumerate(_run_counts(blocks, max(1, min(cores, units)))):
        work.append(fp.CoreWork(ttnn.CoreCoord(first_column + i // grid.y, i % grid.y), start, count))
        start += count
    return work


def _column_ranges(work, grid, first_column: int) -> ttnn.CoreRangeSet:
    """The cores of a column-major run as one CoreRange (whole columns) or two (whole columns plus the partial one):
    one kernel group per range, so the dispatcher multicasts each binary once instead of once per core."""

    columns, rest = divmod(len(work), grid.y)
    ranges = []
    if columns:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(first_column, 0), ttnn.CoreCoord(first_column + columns - 1, grid.y - 1))
        )
    if rest:
        x = first_column + columns
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, rest - 1)))
    return ttnn.CoreRangeSet(ranges)


class Split:
    """The two core groups of one call: their unit lists, their per-core runs and their core rectangles."""

    __slots__ = ("a_units", "a_work", "a_cores", "b_units", "b_work", "b_cores")

    def __init__(self, a_units, a_work, a_cores, b_units, b_work, b_cores):
        self.a_units, self.a_work, self.a_cores = a_units, a_work, a_cores
        self.b_units, self.b_work, self.b_cores = b_units, b_work, b_cores


def split_units(mesh, chunks: int, group_b_cores: int | None = None) -> Split:
    """The program's work split for ``chunks`` tile rows: group A takes the leading grid columns and group B the
    trailing ``ceil(group_b_cores / grid.y)`` ones, so the two kernel groups sit on disjoint rectangles.

    ``group_b_cores`` defaults to ``GROUP_B_COLUMNS`` whole columns, the measured crossing of the two groups'
    core-time at T = 2048; only whole columns move group A's core count, so the values between two multiples of
    ``grid.y`` buy nothing for group A and every core they leave out costs group B.

    Group A's runs are cut by ``UNIT_COST``, not by unit count: its units are the same unit list in the same
    column-major order, but a run ends where the core's accumulated cost reaches its share, so the q cores stop
    being the ones the program waits for.  MEASURED at T = 2048 on 90 cores: the heaviest run goes from 393.2 to
    336.5 hundredths-of-a-microsecond of modelled core time.  Group B's units are split by count (its z units are
    each other's equals and it is not the critical path).
    """

    if chunks < 1:
        raise ValueError(f"at least one tile row, got {chunks}")
    grid = mesh.compute_with_storage_grid_size()
    a_units, b_units = group_a_units(chunks), group_b_units(chunks)
    wanted = GROUP_B_COLUMNS * grid.y if group_b_cores is None else int(group_b_cores)
    cores_b = max(1, min(wanted, len(b_units), (grid.x - 1) * grid.y))
    columns_b = -(-cores_b // grid.y)
    first_b = grid.x - columns_b
    if first_b < 1:
        raise ValueError(f"group B wants {columns_b} of {grid.x} grid columns, leaving none for group A")
    a_work = _weighted_work(group_a_weight_blocks(chunks), grid, 0, min(len(a_units), first_b * grid.y))
    b_work = _column_work(len(b_units), grid, first_b, cores_b)
    return Split(
        a_units,
        a_work,
        _column_ranges(a_work, grid, 0),
        b_units,
        b_work,
        _column_ranges(b_work, grid, first_b),
    )


# ------------------------------------------------------------------------------------------------ the program


def _check(tensor, shape, dtype, label):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"{label} must be TILE {dtype} {tuple(shape)}, got {tensor.layout} {tensor.dtype} {tuple(tensor.shape)}"
        )


def allocate_outputs(mesh, rows_total: int, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The six output buffers of ``run`` for ``rows_total = 32 * NC`` rows, in argument order."""

    if rows_total % TILE or rows_total < TILE:
        raise ValueError(f"rows must be a positive multiple of {TILE}, got {rows_total}")
    chunks = rows_total // TILE
    qk = (HEADS, chunks, TILE, HEAD_DIM)
    vec = (HEADS, chunks, TILE, 1)
    flat = (1, 1, rows_total, VALUE_WIDTH)
    return (
        fp.allocate(qk, BF16, ttnn.TILE_LAYOUT, mesh, memory_config),
        fp.allocate(qk, BF16, ttnn.TILE_LAYOUT, mesh, memory_config),
        fp.allocate(flat, BF16, ttnn.TILE_LAYOUT, mesh, memory_config),
        fp.allocate(vec, FP32, ttnn.TILE_LAYOUT, mesh, memory_config),
        fp.allocate(vec, FP32, ttnn.TILE_LAYOUT, mesh, memory_config),
        fp.allocate(flat, BF16, ttnn.TILE_LAYOUT, mesh, memory_config),
    )


def run(
    projected,
    history,
    taps,
    constants,
    selects,
    scalars,
    q_c,
    k_c,
    v,
    beta_c,
    g_c,
    sig,
    *,
    rows: int,
    group_b_cores: int | None = None,
):
    """The program on explicit tensors (per-device local shapes).

    Inputs: ``projected`` ``[1, 1, T, 4160]`` bf16 TILE (T = 32 NC), ``history`` ``[1, 1, 32, 2560]`` bf16 (rows 0..2
    = the three rows before row 0), ``taps`` four ``[1, 1, 1, 2560]`` bf16 conv weights, and the three constant
    tensors of :func:`upload_constants`.  Outputs (pre-allocated by the caller, returned): ``q_c`` / ``k_c``
    ``[12, NC, 32, 128]`` bf16, ``v`` ``[1, 1, T, 1536]`` bf16, ``beta_c`` / ``g_c`` ``[12, NC, 32, 1]`` fp32,
    ``sig`` ``[1, 1, T, 1536]`` bf16.  ``rows`` (1..T) is the valid row count: every output row at or past it is
    zeroed by the writers, which is what the chain's ``x * row_mask`` multiplies do.
    """

    if len(tuple(projected.shape)) != 4 or tuple(projected.shape)[:2] != (1, 1):
        raise ValueError(f"gdn_pre_rows projection must be [1, 1, T, {PROJECTION_WIDTH}], got {projected.shape}")
    rows_total = int(projected.shape[-2])
    if rows_total % TILE or rows_total < TILE or int(projected.shape[-1]) != PROJECTION_WIDTH:
        raise ValueError(f"gdn_pre_rows projection must be [1, 1, 32k, {PROJECTION_WIDTH}], got {projected.shape}")
    chunks = rows_total // TILE
    if not 1 <= int(rows) <= rows_total:
        raise ValueError(f"rows must be in [1, {rows_total}], got {rows}")
    _check(projected, (1, 1, rows_total, PROJECTION_WIDTH), BF16, "gdn_pre_rows projection")
    _check(history, (1, 1, TILE, QKV_WIDTH), BF16, "gdn_pre_rows history")
    if len(taps) != CONV_KERNEL:
        raise ValueError(f"gdn_pre_rows takes {CONV_KERNEL} conv taps, got {len(taps)}")
    for index, tap in enumerate(taps):
        _check(tap, (1, 1, 1, QKV_WIDTH), BF16, f"gdn_pre_rows conv tap {index}")
    _check(constants, (1, CONST_TILES, TILE, TILE), FP32, "gdn_pre_rows constants")
    _check(selects, (1, SELECT_TILES, TILE, TILE), BF16, "gdn_pre_rows selects")
    _check(scalars, (1, SCALAR_TILES, TILE, TILE), BF16, "gdn_pre_rows scalars")
    _check(q_c, (HEADS, chunks, TILE, HEAD_DIM), BF16, "gdn_pre_rows q_c")
    _check(k_c, (HEADS, chunks, TILE, HEAD_DIM), BF16, "gdn_pre_rows k_c")
    _check(v, (1, 1, rows_total, VALUE_WIDTH), BF16, "gdn_pre_rows v")
    _check(beta_c, (HEADS, chunks, TILE, 1), FP32, "gdn_pre_rows beta_c")
    _check(g_c, (HEADS, chunks, TILE, 1), FP32, "gdn_pre_rows g_c")
    _check(sig, (1, 1, rows_total, VALUE_WIDTH), BF16, "gdn_pre_rows sig")

    mesh = projected.device()
    split = split_units(mesh, chunks, group_b_cores)

    def pairs(units, w):
        return [value for item in units[w.start : w.start + w.count] for value in item]

    reader_a_cta = []  # the readers address the projection by its fixed 130-tile row stride, not by NC
    for tensor in (projected, history, *taps, selects, scalars):
        reader_a_cta.extend(fp.accessor_args(tensor))
    reader_a_addrs = [
        projected.buffer_address(),
        history.buffer_address(),
        *(tap.buffer_address() for tap in taps),
        selects.buffer_address(),
        scalars.buffer_address(),
        int(rows),
    ]
    writer_a_cta = [chunks, *fp.accessor_args(q_c), *fp.accessor_args(k_c), *fp.accessor_args(v)]
    writer_a_addrs = [q_c.buffer_address(), k_c.buffer_address(), v.buffer_address(), int(rows)]
    reader_b_cta = [*fp.accessor_args(projected), *fp.accessor_args(constants)]
    reader_b_addrs = [projected.buffer_address(), constants.buffer_address()]
    writer_b_cta = [chunks, *fp.accessor_args(beta_c), *fp.accessor_args(g_c), *fp.accessor_args(sig)]
    writer_b_addrs = [beta_c.buffer_address(), g_c.buffer_address(), sig.buffer_address(), int(rows)]

    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, split.a_cores if group == "A" else split.b_cores)
        for _name, index, dtype, pages, group in CBS
    ]
    kernels = [
        fp.reader_kernel(
            KERNELS["reader_qkv"],
            split.a_cores,
            reader_a_cta,
            [(w.core, [*reader_a_addrs, w.count, *pairs(split.a_units, w)]) for w in split.a_work],
        ),
        fp.compute_kernel(
            KERNELS["compute_qkv"],
            split.a_cores,
            [],
            [(w.core, [w.count, *pairs(split.a_units, w)]) for w in split.a_work],
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=False,
            approx=False,
        ),
        fp.writer_kernel(
            KERNELS["writer_qkv"],
            split.a_cores,
            writer_a_cta,
            [(w.core, [*writer_a_addrs, w.count, *pairs(split.a_units, w)]) for w in split.a_work],
        ),
        fp.reader_kernel(
            KERNELS["reader_gates"],
            split.b_cores,
            reader_b_cta,
            [(w.core, [*reader_b_addrs, w.count, *pairs(split.b_units, w)]) for w in split.b_work],
        ),
        fp.compute_kernel(
            KERNELS["compute_gates"],
            split.b_cores,
            [],
            [(w.core, [w.count, *pairs(split.b_units, w)]) for w in split.b_work],
            defines=[("INP_FLOAT32", "1")],
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            approx=False,
            unpack_to_dest_fp32=FP32_COPY_CBS,
        ),
        fp.writer_kernel(
            KERNELS["writer_gates"],
            split.b_cores,
            writer_b_cta,
            [(w.core, [*writer_b_addrs, w.count, *pairs(split.b_units, w)]) for w in split.b_work],
        ),
    ]
    io = [projected, history, *taps, constants, selects, scalars, q_c, k_c, v, beta_c, g_c, sig]
    # What the program addresses and issues, by construction from the two unit lists.  Group A reads one whole
    # window per unit (the previous tile row's four column tiles and this one's), the four tap tiles once per
    # column-group run, and the nine selection and three scalar tiles once per core; group B reads the a/b pair or
    # one z head per unit and the two constant tiles once per core.  The history tile and all six outputs are read
    # or written exactly once -- the GQA expand writes each page of q_c and k_c once.  The arithmetic is the six
    # exact selection matmuls at 2 M N K plus one operation per element of every pass: the conv's multiply and
    # three macs (two each), the SiLU, the norm's square, row sum and column-broadcast multiply, q's two scale
    # passes and k's one, v's row-mask multiply, and group B's sigmoid, add + softplus + multiply and z sigmoid
    # with its typecast.
    tile, fp32_tile = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
    tap_runs = sum(len({split.a_units[i][0] for i in range(w.start, w.start + w.count)}) for w in split.a_work)
    unit_elements = HEAD_TILES * TILE * TILE
    fir_flops = HEAD_TILES * HISTORY_ROWS * 2 * (2 * TILE**3)
    unit_passes = {"q": 7 + 1 + 3 + 2, "k": 7 + 1 + 3 + 1, "v": 7 + 1 + 1}
    meta = fp.program_meta(
        NAME,
        "verify_rows" if chunks == 1 else "slab",
        int(rows),
        reads=(history,),
        writes=(q_c, k_c, v, beta_c, g_c, sig),
        dram_bytes=(
            len(split.a_units) * 2 * HEAD_TILES * tile
            + tap_runs * CONV_KERNEL * HEAD_TILES * tile
            + chunks * (2 + HEADS * HEAD_TILES) * tile
            + len(split.a_work) * (SELECT_TILES + SCALAR_TILES) * tile
            + len(split.b_work) * CONST_TILES * fp32_tile
        ),
        flops=(
            sum(fir_flops + unit_passes[group_kind(group)] * unit_elements for group, _chunk in split.a_units)
            + chunks * (4 + 2 * HEADS * HEAD_TILES) * TILE * TILE
        ),
        cores=len(split.a_work) + len(split.b_work),
    )
    fp.run_program(io, fp.program_descriptor(kernels, cbs), meta=meta)
    return q_c, k_c, v, beta_c, g_c, sig


# ------------------------------------------------------------------------------------------------ the oracles


def mask_rows(outputs: dict, rows: int, rows_total: int) -> dict:
    """Zero every row at or past ``rows`` of q_c, k_c, v, beta_c and g_c, in each output's own layout: what the
    writers do and what the chain's ``x * row_mask`` multiplies do (``x * 1.0`` is exact, ``x * 0.0`` is ``+0`` --
    the bf16 multiply clamps it and the die pins the fp32 one at ``+0`` too).

    ``sig`` is NOT masked: the chain's z sigmoid (``_gate_and_project_rows``) carries no row mask; only the five
    inputs the scan reads are masked (``_make_chunk_inputs``)."""

    if not 1 <= int(rows) <= rows_total:
        raise ValueError(f"rows must be in [1, {rows_total}], got {rows}")
    if int(rows) == rows_total:
        return outputs
    chunks = rows_total // TILE
    keep = torch.arange(rows_total) < int(rows)
    kept = keep.reshape(chunks, TILE)
    masked = dict(outputs)
    for name in ("q_c", "k_c"):
        value = outputs[name].clone()
        value[:, ~kept] = 0.0
        masked[name] = value
    for name in ("beta_c", "g_c"):
        value = outputs[name].clone()
        value[:, ~kept] = 0.0
        masked[name] = value
    value = outputs["v"].clone()
    value[~keep] = 0.0
    masked["v"] = value
    return masked


def reference(projected, history, conv_weights, dt_bias, neg_exp_A, *, rows=None, rounding=ref.DEFAULT) -> dict:
    """The six outputs in their device layouts from ``gdn_rows_reference.pre_reference``, plus the z sigmoid the
    post chain's first three ops produce.

    Host shapes: ``projected`` ``[T, 4160]`` bf16, ``history`` ``[32, 2560]`` bf16, ``conv_weights`` ``[4, 2560]``
    bf16, ``dt_bias`` / ``neg_exp_A`` ``[12]`` fp32.  STRUCTURAL, not bitwise: the reference's SiLU, sigmoid,
    softplus and rms_norm are torch expressions, not the SFPU's polynomials (the device test's oracle is
    :func:`chain_on_device`, never this).
    """

    out = ref.pre_reference(projected, history, conv_weights, dt_bias, neg_exp_A, rounding)
    sig = ref.typecast_to_bf16(ref.sigmoid_fp32(ref.typecast_to_fp32(out["z"])), rounding)
    six = {
        "q_c": out["q_c"],
        "k_c": out["k_c"],
        "v": out["v"],
        "beta_c": out["beta_c"],
        "g_c": out["g_c"],
        "sig": sig,
    }
    rows_total = int(projected.shape[0])
    return six if rows is None else mask_rows(six, rows, rows_total)


def _qk_expand_matrix() -> torch.Tensor:
    """The chain's ``qk_expand`` ``[512, 1536]`` 0/1 matrix: key head h feeds value heads 3h .. 3h + 2."""

    expand = torch.zeros(QK_WIDTH, VALUE_WIDTH)
    for head in range(HEADS):
        key_head = head // (HEADS // QK_HEADS)
        for d in range(HEAD_DIM):
            expand[key_head * HEAD_DIM + d, head * HEAD_DIM + d] = 1.0
    return expand


def chain_on_device(mesh, projected, history, taps, dt_bias, neg_exp_A, *, rows: int | None = None):
    """Today's chain, op for op, on the same device tensors ``run`` takes -- the bitwise oracle of the device test.

    Transcribed from ``ttnn/gdn.py``: ``_shifted_rows_slab`` 2544-2571 (the FIR taps by untilize / slice / concat /
    tilize), ``_causal_conv_rows`` 2573-2598, ``_make_chunk_inputs`` 2600-2684 (the q/k GQA expand, ``rms_norm``,
    scale and row masks; the v row mask; the beta and log-decay gates), the composite's ``head_split_tile`` +
    ``q * scale`` + ``to_chunks_tile`` (``chunk_gated_delta_rule.cpp`` 35-58, 204-206, 227-236) reproduced here as
    ``ttnn.permute`` + ``ttnn.reshape``, and the z sigmoid of ``_gate_and_project_rows`` 2774-2778.  Argument order,
    dtypes, broadcast forms, memory configs and the absence of compute configs follow the chain; the only compute
    config is the chain's own ``self.compute_config`` (HiFi4, no approximations, fp32 accumulation) on the two
    exact 0/1 matmuls.  ``ttnn.multiply(..., output_tensor=buffer)`` of the chain is the same program as
    ``ttnn.multiply(..., memory_config=...)``; the buffer only names where it lands.

    ``rows`` (default: all of them) takes the chain's masked branch: the extra ``x * row_mask`` multiply per output.
    Returns ``(q_c, k_c, v, beta_c, g_c, sig)`` in ``run``'s layouts.
    """

    # The chain's own memory configs are L1 for most intermediates; at 2048 rows those are 10.5 MB tensors that do
    # not fit beside this program's circular buffers on one die, and an interleaved L1 tensor and an interleaved
    # DRAM tensor take the same program factory for every op below, so the oracle runs entirely in DRAM: the
    # arithmetic, the operand order and the rounding points are the chain's either way.
    dram = ttnn.DRAM_MEMORY_CONFIG
    l1 = dram
    rows_total = int(projected.shape[-2])
    chunks = rows_total // TILE
    rows = rows_total if rows is None else int(rows)
    full_rows = rows == rows_total
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    keep = (torch.arange(rows_total) < rows).float()
    expand = ttnn.from_torch(
        _qk_expand_matrix().reshape(1, 1, QK_WIDTH, VALUE_WIDTH).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    mask_bf16 = ttnn.from_torch(
        keep.reshape(1, rows_total, 1, 1).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    mask_bf16_col = ttnn.from_torch(
        keep.reshape(1, 1, rows_total, 1).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    mask_fp32 = ttnn.from_torch(
        keep.reshape(1, 1, rows_total, 1), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram
    )

    # P1: the landing slices of _project_rows.
    qkv = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, rows_total, QKV_WIDTH), memory_config=dram)
    z = ttnn.slice(projected, (0, 0, 0, QKV_WIDTH), (1, 1, rows_total, A_COLUMN), memory_config=dram)
    # gdn.py slices a and b to the twelve valid head columns (_project_rows 2484-2495), not to the tile width: the
    # broadcast add of dt_bias and the neg_exp_A multiply are [1, 1, T, 12] ops.
    a = ttnn.slice(projected, (0, 0, 0, A_COLUMN), (1, 1, rows_total, A_COLUMN + HEADS), memory_config=dram)
    b = ttnn.slice(projected, (0, 0, 0, B_COLUMN), (1, 1, rows_total, B_COLUMN + HEADS), memory_config=dram)

    # C1: _shifted_rows_slab -- the three shifted taps as row-major slices and concats, tilized.
    history_rm = ttnn.to_layout(history, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    qkv_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    pieces = []
    for tap in range(HISTORY_ROWS):
        kept = ttnn.slice(history_rm, (0, 0, tap, 0), (1, 1, HISTORY_ROWS, QKV_WIDTH), memory_config=dram)
        new = ttnn.slice(qkv_rm, (0, 0, 0, 0), (1, 1, rows_total - HISTORY_ROWS + tap, QKV_WIDTH), memory_config=dram)
        shifted_rm = ttnn.concat([kept, new], dim=2, memory_config=dram)
        pieces.append(ttnn.to_layout(shifted_rm, ttnn.TILE_LAYOUT, memory_config=l1))
    pieces.append(qkv)

    # C2: _causal_conv_rows -- multiply, three mac, silu.
    conv = ttnn.multiply(pieces[0], taps[0], memory_config=l1)
    for index in range(1, CONV_KERNEL):
        conv = ttnn.mac(pieces[index], taps[index], conv)
    conv = ttnn.silu(conv, memory_config=l1)

    # Q1 / V1: _make_chunk_inputs.
    q_slice = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, rows_total, QK_WIDTH), memory_config=l1)
    k_slice = ttnn.slice(conv, (0, 0, 0, QK_WIDTH), (1, 1, rows_total, 2 * QK_WIDTH), memory_config=l1)
    v_slice = ttnn.slice(conv, (0, 0, 0, 2 * QK_WIDTH), (1, 1, rows_total, QKV_WIDTH), memory_config=l1)
    heads = []
    for source in (q_slice, k_slice):
        expanded = ttnn.matmul(source, expand, memory_config=l1, compute_kernel_config=compute_config)
        heads_tensor = ttnn.reshape(expanded, (1, rows_total, HEADS, HEAD_DIM), pad_value=0.0)
        normed = ttnn.rms_norm(heads_tensor, epsilon=NORM_EPS)
        scaled = ttnn.multiply(normed, QK_SCALE, memory_config=l1)
        heads.append(scaled if full_rows else ttnn.multiply(scaled, mask_bf16, memory_config=l1))
    q_rows, k_rows = heads
    v = ttnn.multiply(v_slice, mask_bf16_col, memory_config=dram)

    # B1 / G1: the gates, fp32 throughout.
    b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=l1)
    beta = ttnn.sigmoid(b_fp32, memory_config=dram if full_rows else l1)
    if not full_rows:
        beta = ttnn.multiply(beta, mask_fp32, memory_config=dram)
    a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=l1)
    shifted = ttnn.add(a_fp32, dt_bias, memory_config=l1)
    softplus = ttnn.softplus(shifted, beta=1.0, threshold=20.0, memory_config=l1)
    g = ttnn.multiply(neg_exp_A, softplus, memory_config=dram if full_rows else l1)
    if not full_rows:
        g = ttnn.multiply(g, mask_fp32, memory_config=dram)

    # K0: the composite's head_split_tile, its own q scale and to_chunks_tile / the g and beta reshapes.
    def to_chunks_qk(tensor, *, scale: bool):
        head_major = ttnn.reshape(ttnn.permute(tensor, (0, 2, 1, 3)), (HEADS, rows_total, HEAD_DIM))
        if scale:
            head_major = ttnn.multiply(head_major, QK_SCALE)
        return ttnn.reshape(head_major, (HEADS, chunks, TILE, HEAD_DIM))

    def to_chunks_vec(tensor):
        rows_view = ttnn.reshape(tensor, (1, rows_total, HEADS))
        head_major = ttnn.reshape(ttnn.permute(rows_view, (0, 2, 1)), (HEADS, rows_total))
        return ttnn.reshape(head_major, (HEADS, chunks, TILE, 1))

    q_out = to_chunks_qk(q_rows, scale=True)
    k_out = to_chunks_qk(k_rows, scale=False)
    beta_out, g_out = to_chunks_vec(beta), to_chunks_vec(g)

    # E1's first three ops on z: the fp32 sigmoid packed back to bf16 (moved here: it never waits for the scan).
    z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=l1)
    sig = ttnn.typecast(ttnn.sigmoid(z_fp32, memory_config=l1), ttnn.bfloat16, memory_config=dram)
    return q_out, k_out, v, beta_out, g_out, sig
