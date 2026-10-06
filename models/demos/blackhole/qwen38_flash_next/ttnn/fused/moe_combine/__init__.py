# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``moe_combine``: the one-call prefill slab's MoE combine glue as ONE program per layer.

The composed chain (``Qwen38TTNNMoE._weighted_reduce_slab_blocks``): per 512-row block of ``moe_compute``'s
``[10, rows, 2560]`` ROW_MAJOR bf16 combine page, a k-strided ``slice``, a ``tilize`` (``to_layout``), two routing
``slice``s and ``deepseek_moe_fast_reduce_nc_fused`` (the score-weighted sum over the ten slots with the slots this
device does not own scored +0.0: ``mul_tiles_bcast_cols`` with ``acc_to_dest`` into the fp32 DEST, one bf16 pack), then
the ``concat`` of the four partials: 21 programs and 2.63 ms of kernel per layer at 2048 rows (the arm-1 slab profile
of 2026-09-25; 128 ms per 2048-row slab over the 48 layers).

This program: the output ``[1, 1, rows, 2560]`` bf16 TILE is split into work units of one row tile (32 tokens) by
``cols`` column tiles (``groups = 80 / cols`` per row tile), spread contiguously over the compute grid.  Per unit the
reader streams the ten slots' ROW_MAJOR blocks (32 token rows x ``cols x 64`` contiguous bytes of the page, the tilize
op's own reader pattern) and builds the ten score tiles exactly as the fused reduce's reader does (column 0, row
``j`` = the row's bf16 score for the slot when this device owns the slot's expert, else +0.0); the compute kernel runs
the chain's own tilize (``compute_kernel_lib::tilize``, the fast Blackhole path the ``to_layout`` op takes on bf16)
then the chain's MAC, slot order 0..9 into DEST tile 0 under fp32 accumulation and HiFi4, one ``pack_tile``; the
writer stores the tiled output pages.  So every output element is the chain's arithmetic in the chain's order
(tolerance class BITWISE), and the slices, the tilize round trip through DRAM and the concat are gone.

Unowned slots (``read_all`` off, the default): the reader zero-fills the rows of the slots this device does not own
instead of reading them.  The chain reads them and multiplies by an exact +0.0; the model keeps every value the
page ever holds finite (zero at allocation, then expert outputs: ttnn/moe.py, the fill comment), and for a finite
``x`` the FPU's ``x * (+0.0)`` is a signed zero whose accumulation into a non-negative-zero DEST leaves it
unchanged, so the two forms agree bit for bit (the device test runs NaN garbage through the unowned slots to prove
the reader never touches them).  ``QWEN38_MOE_COMBINE_READ_ALL=1`` reads every slot, the chain's traffic, for A/B.

Two deliberate departures from the chain's reader, both outside its contract: an expert index at or past 512 counts
as unowned here (the chain indexes its mapping page past its end), and the owner row is device ``d``'s ``[1, 512]``
0/1 row (``moe_post.owner_rows``), which the model shards over the mesh; the replicated ``expert_mapping`` (device
ids) has the same shape and dtype and would be a silently wrong input, so the model site builds the row itself.  The
tilize runs on the fast Blackhole path in the 16-bit dest exactly as the ``to_layout`` op does (the fast tilize init
clears the dest's fp32 mode and its uninit restores it for the MAC); ``cols = 1`` takes the plain ``tilize_block``
through the 32-bit dest, bf16 bits either way (the device test covers every admitted ``cols``).

Knobs (read once from the environment, ``configured`` overrides them in tests): ``QWEN38_MOE_COMBINE_COLS`` (column
tiles per unit, a divisor of 80 up to 20; default 16: 320 units at 2048 rows over the 110-130 cores of a die, the
fastest measured),
``QWEN38_MOE_COMBINE_READ_ALL``; the study build's phase zones come from the shared ``QWEN38_FUSED_ZONES``.  The CB
plan takes ``l1_bytes(cols)`` of a core's L1 (749 KB at 16, 397 KB at 8, 925 KB at 20) beside whatever the model keeps
resident; a plan that does not fit fails at ``run_program``, and ``cols`` 8 is the fallback.
The slab's default form (``DEFAULT_ON``; ``QWEN38_FUSED_OFF=moe_combine`` restores the 512-row blocks); the model
resolves it with ``resolve_admitted`` (``admits`` = the page, routing and owner contracts below), so a shape outside
the contract takes the chain.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator

import ttnn

from .. import program as fp
from ..moe_post import (
    EXPERTS,
    EXPERTS_PER_DEVICE,
    HIDDEN,
    HIDDEN_TILES,
    OWNER_BYTES,
    TOP_K,
    chain_compute_config,
    chain_mapping,
    owner_rows,
)
from ..registry import BITWISE, FusedKernel, register

NAME = "moe_combine"
BF16, U16 = ttnn.bfloat16, ttnn.uint16
DRAM = ttnn.DRAM_MEMORY_CONFIG
KERNELS = {name: fp.kernel_source(NAME, f"{name}_moe_combine.cpp") for name in ("reader", "compute", "writer")}
ROW_BYTES = HIDDEN * 2  # one ROW_MAJOR page of the combine buffer (one token row of one slot)
ROUTE_PAGE_BYTES = 32 * 64  # 32 routing rows staged at a 64-byte pitch (a DRAM page lands at its own 64-byte phase)
MIN_ROWS = fp.TILE
MAX_ROWS = 4096  # contracts.MAX_SLAB_ROWS (pinned by the static test); the L1 plan does not depend on the rows
CHAIN_BLOCK_ROWS = 512  # moe.SLAB_REDUCE_BLOCK_ROWS: the composed chain's reduce block
COLS_ENV = "QWEN38_MOE_COMBINE_COLS"
READ_ALL_ENV = "QWEN38_MOE_COMBINE_READ_ALL"
COLS_ADMITTED = (1, 2, 4, 5, 8, 10, 16, 20)  # divisors of 80 whose L1 plan fits (see l1_bytes)
DEFAULT_COLS = (
    16  # measured 2026-09-25 on a 110-core die at 2048 rows: 149 us (16), 155 (20), 173 (10), 189 (8), 252 (5)
)
L1_BUDGET = 1_200_000  # bytes of CB space a plan may take on a Blackhole core (1.5 MB L1 less the reserved regions)
RM_DEPTH = TOP_K  # the reader may run a whole unit's ten slot blocks ahead of the compute
# (name, index, dtype, page bytes, pages(cols)); the names are the kernels' named compile-time args
CBS = (
    ("cb_rm", 0, BF16, fp.TILE_BYTES[BF16], lambda cols: cols * RM_DEPTH),  # ROW_MAJOR slot blocks, 32 rows each
    ("cb_tiled", 1, BF16, fp.TILE_BYTES[BF16], lambda cols: cols * TOP_K),  # the unit's ten slots, tilized
    ("cb_scores", 2, BF16, fp.TILE_BYTES[BF16], lambda cols: 2 * TOP_K),  # ten score tiles per row tile, two deep
    ("cb_owner", 3, U16, OWNER_BYTES, lambda cols: 1),
    ("cb_route", 4, U16, ROUTE_PAGE_BYTES, lambda cols: 2),  # indices rows, then scores rows
    ("cb_out", 16, BF16, fp.TILE_BYTES[BF16], lambda cols: 2 * cols),
)
CB_INDEX = {name: index for name, index, _dtype, _bytes, _pages in CBS}
READER_ARGS = ("combine", "scores", "indices", "owner", "unit_start", "unit_count", "rows")
COMPUTE_ARGS = ("unit_start", "unit_count")
WRITER_ARGS = ("out", "unit_start", "unit_count")


@dataclass(frozen=True)
class Settings:
    cols: int = DEFAULT_COLS
    read_all: bool = False

    def __post_init__(self) -> None:
        if self.cols not in COLS_ADMITTED:
            raise ValueError(f"{COLS_ENV} must be one of {COLS_ADMITTED}, got {self.cols}")
        if l1_bytes(self.cols) > L1_BUDGET:
            raise ValueError(f"{NAME}: cols={self.cols} needs {l1_bytes(self.cols)} bytes of L1, over {L1_BUDGET}")


def l1_bytes(cols: int) -> int:
    """The CB space of a plan with ``cols`` column tiles per unit."""

    return sum(page_bytes * pages(cols) for _name, _index, _dtype, page_bytes, pages in CBS)


def settings_from_env(environ=os.environ) -> Settings:
    return Settings(
        cols=int(environ.get(COLS_ENV, DEFAULT_COLS)),
        read_all=environ.get(READ_ALL_ENV, "0") == "1",
    )


_settings: Settings | None = None


def settings() -> Settings:
    """The process's settings: the environment, read once on first use."""

    global _settings
    if _settings is None:
        _settings = settings_from_env()
    return _settings


@contextmanager
def configured(**changes) -> Iterator[Settings]:
    """Temporarily override the settings (tests and the microtest): ``with configured(cols=16, read_all=True):``."""

    global _settings
    previous = settings()
    _settings = replace(previous, **changes)
    try:
        yield _settings
    finally:
        _settings = previous


# ----------------------------------------------------------------------------------------------------------------
# contracts
# ----------------------------------------------------------------------------------------------------------------


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def routing_rows_of(scores, indices) -> int:
    """The token rows of the ROW_MAJOR routing tables ``[.., rows, 10]`` (leading dimensions 1)."""

    for name, tensor, dtype in (("scores", scores, BF16), ("indices", indices, U16)):
        shape = _shape(tensor)
        if len(shape) < 2 or shape[-1] != TOP_K or any(v != 1 for v in shape[:-2]):
            raise ValueError(f"{NAME} {name} must be [.., rows, {TOP_K}] with leading 1s, got {list(shape)}")
        if tensor.dtype != dtype or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"{NAME} {name} must be ROW_MAJOR {dtype}, got {tensor.layout} {tensor.dtype}")
    if _shape(scores)[-2] != _shape(indices)[-2]:
        raise ValueError(f"{NAME} scores and indices disagree on the rows")
    return _shape(scores)[-2]


def _check_inputs(combine, scores, indices, owner) -> int:
    rows = routing_rows_of(scores, indices)
    if rows % fp.TILE or not MIN_ROWS <= rows <= MAX_ROWS:
        raise ValueError(f"{NAME} rows must be whole tiles in {MIN_ROWS}..{MAX_ROWS}, got {rows}")
    if _shape(combine) != (TOP_K, rows, HIDDEN) or combine.dtype != BF16 or combine.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"{NAME} combine must be ROW_MAJOR bf16 [{TOP_K}, {rows}, {HIDDEN}], got {list(_shape(combine))} "
            f"{combine.layout} {combine.dtype}"
        )
    if _shape(owner) != (1, EXPERTS) or owner.dtype != U16 or owner.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"{NAME} owner must be ROW_MAJOR uint16 [1, {EXPERTS}], got {list(_shape(owner))}")
    tensors = (combine, scores, indices, owner)
    if all(hasattr(t, "device") for t in tensors):
        devices = {_device_key(t) for t in tensors}
        if len(devices) != 1:
            raise ValueError(f"{NAME} inputs must share one device, got {sorted(map(str, devices))}")
    return rows


def _device_key(tensor):
    device = tensor.device()
    ident = getattr(device, "id", None)
    return ident() if callable(ident) else id(device)


def admits(combine, scores, indices, owner, *, memory_config=DRAM) -> bool:
    """The input contract as a predicate (``resolve_admitted``): the shapes, dtypes and layouts above."""

    try:
        _check_inputs(combine, scores, indices, owner)
    except (ValueError, AttributeError):
        return False
    return True


# ----------------------------------------------------------------------------------------------------------------
# the program
# ----------------------------------------------------------------------------------------------------------------


def unit_count(rows: int, cols: int) -> int:
    """Work units: one row tile by ``cols`` column tiles each."""

    if HIDDEN_TILES % cols:
        raise ValueError(f"cols must divide {HIDDEN_TILES}, got {cols}")
    return (rows // fp.TILE) * (HIDDEN_TILES // cols)


def moe_combine_program(combine, scores, indices, owner, out, *, rows: int, cfg: Settings) -> "ttnn.ProgramDescriptor":
    mesh = combine.device()
    cols = cfg.cols
    work = fp.split_work(unit_count(rows, cols), mesh)
    cores = fp.core_rectangle(work, mesh)
    named = [(name, index) for name, index, _dtype, _bytes, _pages in CBS] + [
        ("cols", cols),
        ("top_k", TOP_K),
        ("groups", HIDDEN_TILES // cols),
        ("hidden_tiles", HIDDEN_TILES),
        ("experts", EXPERTS),
        ("owner_bytes", OWNER_BYTES),
        ("read_all", 1 if cfg.read_all else 0),
    ]
    cbs = [
        fp.cb_descriptor(index, dtype, page_bytes, pages(cols), cores) for _n, index, dtype, page_bytes, pages in CBS
    ]
    reader = fp.reader_kernel(
        KERNELS["reader"],
        cores,
        [*fp.accessor_args(combine), *fp.accessor_args(scores), *fp.accessor_args(indices), *fp.accessor_args(owner)],
        [
            (
                w.core,
                [
                    combine.buffer_address(),
                    scores.buffer_address(),
                    indices.buffer_address(),
                    owner.buffer_address(),
                    w.start,
                    w.count,
                    rows,
                ],
            )
            for w in work
        ],
        named=named,
    )
    writer = fp.writer_kernel(
        KERNELS["writer"],
        cores,
        fp.accessor_args(out),
        [(w.core, [out.buffer_address(), w.start, w.count]) for w in work],
        named=named,
    )
    compute = fp.compute_kernel(
        KERNELS["compute"],
        cores,
        [],
        [(w.core, [w.start, w.count]) for w in work],
        named=named,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
    )
    return fp.program_descriptor([reader, writer, compute], cbs=cbs)


def moe_combine(combine, scores, indices, owner, *, memory_config=DRAM):
    """The layer's weighted routed reduce ``[1, 1, rows, 2560]`` bf16 TILE: the score-weighted sum of the owned slots
    of ``combine`` (``moe_compute``'s ``[10, rows, 2560]`` ROW_MAJOR one-call page), the chain's ``partial``.
    ``scores`` / ``indices`` are the ROW_MAJOR routing rows ``[1, 1, rows, 10]``, ``owner`` this device's ``[1, 512]``
    uint16 expert owner row (1 = owned; ``moe_post.owner_rows``)."""

    rows = _check_inputs(combine, scores, indices, owner)
    cfg = settings()
    mesh = combine.device()
    out = fp.allocate((1, 1, rows, HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh, memory_config)
    fp.run_program(
        [combine, scores, indices, owner, out],
        moe_combine_program(combine, scores, indices, owner, out, rows=rows, cfg=cfg),
        meta=moe_combine_meta(combine, scores, indices, owner, out, rows=rows, cfg=cfg),
    )
    return out


def moe_combine_meta(combine, scores, indices, owner, out, *, rows: int, cfg: Settings) -> "fp.FusedProgramMeta":
    """What the program moves and issues, by construction (the census's model): the page's owned rows at their bound
    (every slot owned: the whole ``[10, rows, 2560]``), the tiled sum out, per core the owner row once and the 32
    routing rows of every row tile its units touch; per output element the ten-slot MAC (two FLOPs per slot)."""

    work = fp.split_work(unit_count(rows, cfg.cols), combine.device())
    groups = HIDDEN_TILES // cfg.cols
    row_tiles_touched = sum((w.start + w.count - 1) // groups - w.start // groups + 1 for w in work)
    routing_row_bytes = 2 * TOP_K * 2  # one token's scores and indices (two 20-byte DRAM pages)
    per_core = len(work) * fp.tensor_bytes(owner) + row_tiles_touched * fp.TILE * routing_row_bytes
    return fp.program_meta(
        NAME,
        "combine_read_all" if cfg.read_all else "combine",
        rows,
        reads=(combine,),
        writes=(out,),
        dram_bytes=per_core,
        flops=rows * HIDDEN * 2 * TOP_K,
        cores=len(work),
        outputs=((out, None),),
    )


# ----------------------------------------------------------------------------------------------------------------
# the composed chain (one-chip replica)
# ----------------------------------------------------------------------------------------------------------------


def moe_combine_composed(combine, scores, indices, owner, *, memory_config=DRAM):
    """The chain on one chip, op for op as ``_weighted_reduce_slab_blocks`` issues it: per 512-row block the k-strided
    slice of the page, ``reshape`` + ``to_layout(TILE)`` + ``view``, the routing slices and
    ``deepseek_moe_fast_reduce_nc_fused`` (HiFi4, fp32 DEST) with this device's ownership (``cluster_axis`` 1 on a
    1x1 mesh reads the mapping: owner 0 = on-axis, so the mapping is ``1 - owner``); the partials concatenated.  Rows
    that are not a multiple of 512 end in a shorter block here (the one-call slab refuses such row counts at
    construction, moe.py; the row contract itself admits steps of 128).  The chain reads every slot: the caller hands
    in pages whose unowned slots are finite (the model's invariant)."""

    rows = _check_inputs(combine, scores, indices, owner)
    mesh = combine.device()
    if mesh.get_num_devices() != 1:
        raise ValueError(f"{NAME}_composed is the one-chip replica of the chain; the model runs its inline chain")
    mapping = chain_mapping(owner)
    config = chain_compute_config(mesh)
    scores4 = ttnn.reshape(scores, (1, 1, rows, TOP_K))
    indices4 = ttnn.reshape(indices, (1, 1, rows, TOP_K))
    partials = []
    for start in range(0, rows, CHAIN_BLOCK_ROWS):
        block = min(CHAIN_BLOCK_ROWS, rows - start)
        pages = ttnn.slice(combine, (0, start, 0), (TOP_K, start + block, HIDDEN), memory_config=DRAM)
        stack = ttnn.experimental.view(
            ttnn.to_layout(
                ttnn.reshape(pages, (TOP_K * block, HIDDEN)), ttnn.TILE_LAYOUT, memory_config=DRAM, pad_value=0.0
            ),
            (TOP_K, 1, block, HIDDEN),
        )
        block_scores = ttnn.slice(scores4, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=DRAM)
        block_indices = ttnn.slice(indices4, (0, 0, start, 0), (1, 1, start + block, TOP_K), memory_config=DRAM)
        fast_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            stack,
            block_indices,
            mapping,
            reduce_dim=0,
            split_size=HIDDEN,
            cluster_axis=1,
            output_memory_config=DRAM,
            scores_tensor=ttnn.reshape(block_scores, (block, 1, 1, TOP_K)),
            num_shared_experts=0,
            shared_expert_scale=1.0,
            compute_kernel_config=config,
        )
        for t in (pages, stack, block_scores, block_indices):
            # a slice spanning the whole tensor hands back the input itself (one block): never free that
            if all(t.buffer_address() != src.buffer_address() for src in (combine, scores, indices)):
                ttnn.deallocate(t)
        partials.append(fast_outputs[0])
    ttnn.deallocate(mapping)
    if len(partials) == 1:  # one block: the chain's concat has nothing to join
        partial = partials[0]
        if memory_config == DRAM:
            return partial
        moved = ttnn.to_memory_config(partial, memory_config)
        ttnn.deallocate(partial)
        return moved
    partial = ttnn.concat(partials, dim=2, memory_config=memory_config)
    for t in partials:
        ttnn.deallocate(t)
    return partial


__all__ = [
    "NAME",
    "Settings",
    "admits",
    "configured",
    "l1_bytes",
    "moe_combine",
    "moe_combine_composed",
    "owner_rows",
    "settings",
    "settings_from_env",
    "unit_count",
]

register(
    FusedKernel(
        name=NAME,
        replaces="the one-call slab's MoE combine glue: 4 x (k-strided slice, tilize, 2 routing slices, deepseek_moe_fast_reduce_nc_fused) + concat (21 programs per layer)",
        tolerance=BITWISE,
        fused=moe_combine,
        composed=moe_combine_composed,
        gate=None,
        admits=admits,
    )
)
