# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``router_tail``: the MoE router tail as one program.  fp32 logits ``[1, 1, rows, 512]`` (TILE, DRAM) -> ROW_MAJOR
bf16 scores and uint16 indices ``[1, 1, rows, top_k]`` (DRAM); the chain ``softmax(numeric_stable) -> topk(k, largest,
sorted) -> sum -> div -> typecast(bf16) -> to_layout(ROW_MAJOR) x2 -> typecast(uint16)`` of ``Qwen38TTNNMoE._route``.

Two program forms, the same three kernels.  The single-core form (``QWEN38_ROUTER_TAIL_LANES=0``): one core per
32-row tile; the compute kernel issues each replaced op's instruction sequence on CBs of that op's data format and
unpack mode (``kernels/compute_router_tail.cpp``), so the values and the top-k tie order are the composed chain's:
tolerance class BITWISE.  ``rows`` 1..32 is one tile; 128 rows (the long chunk) is four tiles on four cores (not
proven on device yet; the model keeps the chunk on the composed chain).

The lane form (the default, rows <= 32): the top-k LLK sorts a tile's 32 token columns in four
independent passes of eight tokens each (``kernels/topk_lanes.h``); one core per pass that holds a live token, every
core running the whole softmax and the whole insertion chain but only its pass of every sort, and writing only its
tokens.  Same instructions on the same DST rows for every live token: bitwise with the single-core form by
construction, at a quarter of the sort work per core (one core and one pass for rows <= 8 -- the decode body).
``PASS_TOKENS`` (which tokens each pass holds: face by row half, pass by row parity) was measured on device by the
lane probe, a development tool that is not shipped: rows 1 needs one core, rows 2..16 two, rows 17..32 four.
"""

from __future__ import annotations

import os

import torch

import ttnn

from .. import placement
from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register

NAME = "router_tail"
EXPERTS = 512
WIDTH_TILES = EXPERTS // fp.TILE
TOP_K = 10
STAGE_PAGES = 4  # 8 KB of bf16 tile pages for the writer's two 2 KB row stages plus alignment
KERNELS = {name: fp.kernel_source(NAME, f"{name}_router_tail.cpp") for name in ("reader", "compute", "writer")}
LANES_HEADER = fp.kernel_source(NAME, "topk_lanes.h")
EXP_LIVE_HEADER = fp.kernel_source(NAME, "exp_live.h")
EXP_LIVE_ENV = "QWEN38_ROUTER_TAIL_EXP_LIVE"
LANES_ENV = "QWEN38_ROUTER_TAIL_LANES"
# The core rectangle the top-k asks the placement helper for (placement.free_rectangle): 4 x 5 = the exact multi-core
# form's 16 worker cores plus up to four lane cores (q38-router-topk-exact); the lane form takes the rectangle's first
# row, so the top-k lands on the same cores standalone and hosted by the MoE dense composite.  Off row 0: today's
# lane core (0, 0) was storage core 0 of the dense linears, which serialized the shared expert chain behind the top-k.
RECTANGLE = (4, 5)
PASSES = 4  # the LLK local sort's (face, col) passes; pass p = bit p of the compute kernel's pass_mask
ALL_TOKENS = (1 << fp.TILE) - 1
# The eight token rows (of the untransposed tile) each sort pass covers: pass (face, col) = tokens of face `face`
# (rows 0-15 / 16-31) with row parity `col`.  Measured on one Blackhole chip (the lane probe, 2026-09-18: with one
# pass enabled exactly these rows stay bitwise against the chain; the four sets partition the tile).  A change here changes which core writes which row, nothing in the arithmetic.
PASS_TOKENS: tuple[frozenset[int], ...] = tuple(
    frozenset(range(16 * face + col, 16 * face + 16, 2)) for face in range(2) for col in range(2)
)

# Circular buffers: (name, index, dtype, pages).  The compute kernel unpacks cb_probs, cb_vals_t, cb_pad_reduce,
# cb_pad_div and cb_denom straight into the 32-bit dest (as topk, the accurate fp32 reduce and binary_ng unpack their
# fp32 operands); every other fp32 CB unpacks to the source registers (as softmax does).  Names are the kernels' named compile-time args.
CBS = (
    ("cb_in0", 0, ttnn.float32, WIDTH_TILES),
    ("cb_max_scaler", 1, ttnn.float32, 1),
    ("cb_sum_scaler", 2, ttnn.float32, 1),
    ("cb_norm_scaler", 3, ttnn.float32, 1),
    ("cb_max", 4, ttnn.float32, 1),
    ("cb_exps", 5, ttnn.float32, WIDTH_TILES),
    ("cb_recip", 6, ttnn.float32, 1),
    ("cb_probs", 7, ttnn.float32, WIDTH_TILES),
    ("cb_index", 8, ttnn.uint32, WIDTH_TILES),
    ("cb_vals_t", 9, ttnn.float32, 1),
    ("cb_idx_t", 10, ttnn.uint32, 1),
    ("cb_vals", 11, ttnn.float32, 1),  # [token, k] values; the reader zeroes the padding in place -> the sum's input
    ("cb_vals_ready", 12, ttnn.uint16, 1),  # token: the reader finished the zero fill
    ("cb_pad_div", 13, ttnn.float32, 1),  # [token, k] values for the division
    ("cb_sums", 14, ttnn.float32, 1),  # row sums; the reader broadcasts column 0 in place -> the division's rhs
    ("cb_sums_ready", 15, ttnn.uint16, 1),  # token: the reader finished the broadcast
    ("cb_scores", 16, ttnn.bfloat16, 1),
    ("cb_stage", 17, ttnn.bfloat16, STAGE_PAGES),  # writer staging: 2 x 32 rows x 64 B, 64-B aligned inside
)
TOKEN_CBS = ("cb_vals_ready", "cb_sums_ready")
TOKEN_PAGE_BYTES = 32
UNPACK_TO_DEST_FP32 = ("cb_probs", "cb_vals_t", "cb_vals", "cb_pad_div", "cb_sums")
CB_INDEX = {name: index for name, index, _dtype, _pages in CBS}
READER_ARGS = ("logits_addr", "index_addr", "tile_row", "rows_in_tile", "token_mask")
COMPUTE_ARGS = ("pass_mask", "token_mask")
WRITER_ARGS = ("scores_addr", "indices_addr", "tile_row", "rows_in_tile", "token_mask")


def _compute_kernel_config(logits):
    return ttnn.init_device_compute_kernel_config(
        logits.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _rows_of(logits) -> int:
    shape = tuple(int(v) for v in logits.shape)
    rows = shape[-2] if len(shape) == 4 else 0
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[3] != EXPERTS
        or not (1 <= rows <= fp.TILE or rows % fp.TILE == 0)
    ):
        raise ValueError(f"router tail logits must be [1, 1, rows (1..32 or n x 32), {EXPERTS}], got {list(shape)}")
    if logits.dtype not in (ttnn.float32, ttnn.bfloat16) or logits.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"router tail logits must be fp32 or bf16 TILE, got {logits.dtype} {logits.layout}")
    return rows


_INDEX_TEMPLATES: dict[int, object] = {}


def router_tail_prepare(mesh):
    """The constant index tiles the top-k sorts alongside the values, pre-transposed: tile ``w`` holds ``w*32+k`` in
    every column of row ``k`` (the transpose of the topk reader's ``w*32+c`` tile), as one uint32 TILE tensor
    ``[1, 1, 32, 512]`` in DRAM, allocated once per mesh (call before trace capture)."""

    key = id(mesh)
    if key not in _INDEX_TEMPLATES:
        placement.free_rectangle(mesh, *RECTANGLE)  # the placement's device reads, memoized before any capture
        k = torch.arange(fp.TILE, dtype=torch.int32).reshape(fp.TILE, 1)
        tile_base = (torch.arange(EXPERTS, dtype=torch.int32) // fp.TILE * fp.TILE).reshape(1, EXPERTS)
        template = (tile_base + k).reshape(1, 1, fp.TILE, EXPERTS)
        _INDEX_TEMPLATES[key] = ttnn.from_torch(
            template, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return _INDEX_TEMPLATES[key]


def lanes_enabled(environ=os.environ) -> bool:
    """The lane form serves by default; ``QWEN38_ROUTER_TAIL_LANES=0`` is the single-core form (the same kernels, one
    core per tile, the LLK's own four-pass sort; its token_mask is every row, so its exp runs over every vector pair).
    """

    return environ.get(LANES_ENV, "1") != "0"


def lanes_plan(rows: int, pass_tokens=None) -> list[tuple[int, int]]:
    """``(pass_mask, token_mask)`` per core of the lane form for ``rows`` live tokens: one core per sort pass that holds
    a live token, in pass order.  ``pass_tokens`` defaults to the pinned ``PASS_TOKENS``."""

    pass_tokens = PASS_TOKENS if pass_tokens is None else pass_tokens
    if not 1 <= rows <= fp.TILE:
        raise ValueError(f"router tail lanes: rows must be 1..{fp.TILE}, got {rows}")
    plan = []
    for p, tokens in enumerate(pass_tokens):
        owned = [t for t in range(rows) if t in tokens]
        if owned:
            plan.append((1 << p, sum(1 << t for t in owned)))
    return plan


def _core_plan(rows: int) -> tuple[list[tuple[int, int, int]], bool]:
    """``(tile_row, pass_mask, token_mask)`` per core and whether the lane form was chosen: the lane form for one tile
    unless ``QWEN38_ROUTER_TAIL_LANES=0``, else one core per tile row (pass_mask = the dev knob, default 0 = the LLK's
    four-pass sort)."""

    tile_rows = -(-rows // fp.TILE)
    if tile_rows == 1 and lanes_enabled():
        return [(0, pass_mask, token_mask) for pass_mask, token_mask in lanes_plan(rows)], True
    return [(t, _dev_pass_mask(), ALL_TOKENS) for t in range(tile_rows)], False


def lane_cores(rectangle: "ttnn.CoreRange", count: int) -> list["ttnn.CoreCoord"]:
    """The ``count`` lane cores of the form: the first ``count`` cores of the rectangle's first row."""

    width = rectangle.end.x - rectangle.start.x + 1
    if not 1 <= count <= width:
        raise ValueError(f"router tail needs {count} lane cores, the rectangle is {width} wide")
    return [ttnn.CoreCoord(rectangle.start.x + i, rectangle.start.y) for i in range(count)]


def program_parts(
    logits, index_template, scores, indices, *, rows: int, top_k: int, rectangle=None, sem_base: int = 0
) -> tuple[list, list, list]:
    """``(kernels, cbs, semaphores)`` of the top-k program on the lane cores of ``rectangle`` (default: the placement
    helper's ``RECTANGLE`` for the logits' mesh), for a program that hosts them beside other kernel groups; the
    standalone program is ``fp.program_descriptor(*program_parts(...))``.  ``sem_base`` offsets the program-local
    semaphore ids this form declares (none today; the multi-core form reserves five)."""

    plan, _lanes = _core_plan(rows)
    if rectangle is None:
        rectangle = placement.free_rectangle(logits.device(), *RECTANGLE)
    replicas = _dev_replicas()
    if replicas == 1:
        cores = lane_cores(rectangle, len(plan))
    else:  # dev knob: the plan's first core plus silent replicas over the rectangle in row-major order
        tile_row, pass_mask, _token_mask = plan[0]
        plan = plan[:1] + [(tile_row, pass_mask, 0)] * (replicas - 1)
        cores = placement.cores_of(rectangle)[:replicas]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[-1])])
    named = [(name, index) for name, index, _dtype, _pages in CBS] + [
        ("Wt", WIDTH_TILES),
        ("top_k", top_k),
        ("stage_pages", STAGE_PAGES),
    ]
    cbs = [
        fp.cb_descriptor(
            index,
            logits.dtype if name == "cb_in0" else dtype,
            TOKEN_PAGE_BYTES if name in TOKEN_CBS else fp.TILE_BYTES[logits.dtype if name == "cb_in0" else dtype],
            pages,
            grid,
        )
        for name, index, dtype, pages in CBS
    ]

    def per_core(args_of):
        return [
            (core, args_of(tile_row, min(fp.TILE, rows - tile_row * fp.TILE), pass_mask, token_mask))
            for core, (tile_row, pass_mask, token_mask) in zip(cores, plan)
        ]

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64  # one per circular buffer of the runtime (64)
    for name in UNPACK_TO_DEST_FP32:
        modes[CB_INDEX[name]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes

    def kernel(source, compile_time_args, runtime_args, config):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=[int(a) for a in compile_time_args],
            named_compile_time_args=named,
            defines=fp.zone_defines(),  # the study build's phase zones (kernels/zones.h), nothing otherwise
            runtime_args=runtime_args,
            config=config,
        )

    reader = kernel(
        KERNELS["reader"],
        fp.accessor_args(logits) + fp.accessor_args(index_template),
        per_core(lambda t, r, _p, m: [logits.buffer_address(), index_template.buffer_address(), t, r, m]),
        ttnn.ReaderConfigDescriptor(),
    )
    writer = kernel(
        KERNELS["writer"],
        fp.accessor_args(scores) + fp.accessor_args(indices),
        per_core(lambda t, r, _p, m: [scores.buffer_address(), indices.buffer_address(), t, r, m]),
        ttnn.WriterConfigDescriptor(),
    )
    compute = kernel(KERNELS["compute"], [], per_core(lambda _t, _r, p, m: [p, m]), compute_config)
    compute.defines = _dev_defines() + fp.zone_defines()
    del sem_base  # no program-local semaphores in this form
    return [reader, writer, compute], cbs, []


def router_tail_program(
    logits, index_template, scores, indices, *, rows: int, top_k: int, rectangle=None
) -> "ttnn.ProgramDescriptor":
    kernels, cbs, semaphores = program_parts(
        logits, index_template, scores, indices, rows=rows, top_k=top_k, rectangle=rectangle
    )
    return fp.program_descriptor(kernels, cbs=cbs, semaphores=semaphores)


def _dev_defines() -> list[tuple[str, str]]:
    """Study knobs (dev only; every one of them breaks the output): QWEN38_ROUTER_TAIL_TOPK_TILES=N sorts only the
    first N width tiles, QWEN38_ROUTER_TAIL_SOFTMAX_COPY_ONLY=1 skips the softmax math, QWEN38_ROUTER_TAIL_TOPK_SORT_SKIP=1
    keeps the transposes and copies but skips the sorts, QWEN38_ROUTER_TAIL_TOPK_SPLIT=2 emulates a two-core width
    split on one core (the study's tie-order counter-example), QWEN38_ROUTER_TAIL_EXP_ITERATIONS / _SORT_PHASES time the
    exact form's anchors.  QWEN38_ROUTER_TAIL_EXP_LIVE=0 is the bitwise A/B switch back to the full exp_tile (not a
    study knob: the output is the same)."""

    defines = []
    tiles = os.environ.get("QWEN38_ROUTER_TAIL_TOPK_TILES")
    if tiles:
        defines.append(("FRT_TOPK_TILES", str(int(tiles))))
    if os.environ.get("QWEN38_ROUTER_TAIL_SOFTMAX_COPY_ONLY") == "1":
        defines.append(("FRT_SOFTMAX_COPY_ONLY", "1"))
    if os.environ.get("QWEN38_ROUTER_TAIL_TOPK_SORT_SKIP") == "1":
        defines.append(("FRT_TOPK_SORT_SKIP", "1"))
    split = os.environ.get("QWEN38_ROUTER_TAIL_TOPK_SPLIT")
    if split:
        defines.append(("FRT_TOPK_SPLIT", str(int(split))))
    if os.environ.get(EXP_LIVE_ENV, "1") == "0":
        defines.append(
            ("FRT_EXP_LIVE", "0")
        )  # A/B switch: the full exp_tile over all 32 vectors (today's instruction stream)
    exp_iterations = os.environ.get("QWEN38_ROUTER_TAIL_EXP_ITERATIONS")
    if exp_iterations:
        if not 0 <= int(exp_iterations) <= 8:
            raise ValueError(f"QWEN38_ROUTER_TAIL_EXP_ITERATIONS must be 0..8, got {exp_iterations}")
        defines.append(("FRT_EXP_ITERATIONS", str(int(exp_iterations))))
    phases = os.environ.get("QWEN38_ROUTER_TAIL_SORT_PHASES")
    if phases:
        start, end = (int(v) for v in phases.split(":"))
        if not 0 <= start <= end <= 5:
            raise ValueError(f"QWEN38_ROUTER_TAIL_SORT_PHASES must be start:end within 0..5, got {phases}")
        defines.append(("FRT_SORT_PHASE_START", str(start)))
        defines.append(("FRT_SORT_PHASE_END", str(end)))
    return defines


def _dev_replicas() -> int:
    """Study knob (dev only): QWEN38_ROUTER_TAIL_DEV_REPLICAS=N runs the program on N cores of one rectangle, the
    extra cores reading and sorting everything and writing nothing (token_mask 0, so with the live exp they skip the
    exp too): the concurrent-read timing of a multi-core form.  N must be 1 (off) or a multiple of 4 up to 20 (the
    placement rectangle's cores; more would truncate silently against it)."""

    replicas = int(os.environ.get("QWEN38_ROUTER_TAIL_DEV_REPLICAS", "1"))
    if replicas != 1 and not (replicas % 4 == 0 and 4 <= replicas <= 20):
        raise ValueError(
            f"QWEN38_ROUTER_TAIL_DEV_REPLICAS must be 1 or a multiple of 4 up to 20 (the rectangle), got {replicas}"
        )
    return replicas


def _dev_pass_mask() -> int:
    """Study knob (dev only): QWEN38_ROUTER_TAIL_TOPK_PASS_MASK=<0..15> makes the single-core form sort only those
    passes of the LLK network (topk_lanes.h); the tokens of the other passes come out wrong.  0 = the LLK's own sort."""

    mask = int(os.environ.get("QWEN38_ROUTER_TAIL_TOPK_PASS_MASK", "0"))
    if not 0 <= mask < (1 << PASSES):
        raise ValueError(f"QWEN38_ROUTER_TAIL_TOPK_PASS_MASK must be 0..{(1 << PASSES) - 1}, got {mask}")
    return mask


def router_tail(logits, *, top_k: int = TOP_K, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """``(scores, indices)`` ROW_MAJOR bf16 / uint16 ``[1, 1, rows, top_k]``; ``compute_kernel_config`` is the chain's
    (HiFi4, fp32 dest) and is pinned inside the kernel.  ``logits`` are the fp32 tiles the chain drains, or the router
    linear's bf16 L1 shard itself (the same values: a bf16 value is exact in the 19-bit source registers the softmax
    unpacks fp32 into)."""

    rows = _rows_of(logits)
    mesh = logits.device()
    scores = fp.allocate((1, 1, rows, top_k), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    indices = fp.allocate((1, 1, rows, top_k), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    return router_tail_into(logits, scores, indices, top_k=top_k)


def router_tail_into(logits, scores, indices, *, top_k: int = TOP_K):
    """``router_tail`` writing its ROW_MAJOR rows into the pre-allocated ``scores`` (bf16) and ``indices`` (uint16)
    of ``rows x top_k`` rows in any layout (e.g. the drain-core L1 shards ``moe_compute`` reads); returns them."""

    rows = _rows_of(logits)
    if not 1 <= top_k <= 16:
        raise ValueError(f"router tail top_k must be 1..16 (one face of the sorted tile), got {top_k}")
    for name, tensor, dtype in (("scores", scores, ttnn.bfloat16), ("indices", indices, ttnn.uint16)):
        shape = tuple(int(v) for v in tensor.shape)
        if (
            shape[-2:] != (rows, top_k)
            or any(v != 1 for v in shape[:-2])
            or tensor.dtype != dtype
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
        ):
            raise ValueError(
                f"router tail {name} output must be ROW_MAJOR {dtype} [.., {rows}, {top_k}], got {list(shape)} {tensor.dtype}"
            )
    index_template = router_tail_prepare(logits.device())
    plan, lanes = _core_plan(rows)
    # every core streams the logits tile row (the fp32 tiles, or the router linear's bf16 L1 shard) and the index
    # template; the scores and indices rows out; per token the softmax (max, exp, sum, reciprocal, scale), the
    # bitonic top-k over the 16 width tiles and the sum / division / typecasts of the k scores
    meta = fp.program_meta(
        NAME,
        "lanes" if lanes else "single_core",
        rows,
        writes=(scores, indices),
        dram_bytes=len(plan) * (fp.tensor_bytes(index_template) + (0 if fp.in_l1(logits) else fp.tensor_bytes(logits))),
        l1_bytes=len(plan) * fp.tensor_bytes(logits) if fp.in_l1(logits) else 0,
        flops=rows * EXPERTS * (5 + 2 * WIDTH_TILES) + rows * top_k * 3,
        cores=len(plan),
    )
    fp.run_program(
        [logits, index_template, scores, indices],
        router_tail_program(logits, index_template, scores, indices, rows=rows, top_k=top_k),
        meta=meta,
    )
    return scores, indices


def router_tail_composed(
    logits, *, top_k: int = TOP_K, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """The composed chain, op for op as ``Qwen38TTNNMoE._route`` issues it for one row tile."""

    _rows_of(logits)
    compute_config = _compute_kernel_config(logits) if compute_kernel_config is None else compute_kernel_config
    l1 = ttnn.L1_MEMORY_CONFIG
    probabilities = ttnn.softmax(
        logits, dim=-1, numeric_stable=True, memory_config=l1, compute_kernel_config=compute_config
    )
    scores, indices = ttnn.topk(probabilities, k=top_k, dim=-1, largest=True, sorted=True, memory_config=l1)
    ttnn.deallocate(probabilities)
    denominator = ttnn.sum(scores, dim=-1, keepdim=True, memory_config=l1, compute_kernel_config=compute_config)
    normalized_fp32 = ttnn.div(scores, denominator, memory_config=l1)
    ttnn.deallocate(scores)
    ttnn.deallocate(denominator)
    normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=l1)
    ttnn.deallocate(normalized_fp32)
    scores_rm = ttnn.to_layout(normalized, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(normalized)
    ttnn.deallocate(indices)
    if indices_rm.dtype != ttnn.uint16:
        converted = ttnn.typecast(indices_rm, ttnn.uint16, memory_config=memory_config)
        ttnn.deallocate(indices_rm)
        indices_rm = converted
    return scores_rm, indices_rm


def routing_table(result) -> torch.Tensor:
    """``[rows, 2 * top_k]`` fp32: the bf16 scores widened, then the indices; bitwise-comparable as fp32 bits."""

    scores, indices = result
    k = int(scores.shape[-1])
    return torch.cat(
        [ttnn.to_torch(scores).reshape(-1, k).float(), ttnn.to_torch(indices).reshape(-1, k).to(torch.int64).float()],
        dim=-1,
    )


def _gate_inputs(mesh, capture, positions, layer):
    index = [capture["positions"].index(p) for p in positions]
    logits = capture["router_logits"][index, layer].float().reshape(1, 1, len(positions), EXPERTS)
    return {
        "logits": ttnn.from_torch(
            logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    }


def _gate_reference(oracle, positions, layer):
    index = [oracle["positions"].index(p) for p in positions]
    scores = oracle["router_scores"][index, layer].to(torch.bfloat16).float()
    indices = oracle["router_indices"][index, layer].to(torch.int64).float()
    return torch.cat([scores, indices], dim=-1)


register(
    FusedKernel(
        name=NAME,
        replaces="softmax, topk, fill padding, sum, div, typecast bf16, to_layout x2, typecast uint16 (the 12-program router tail per layer)",
        tolerance=BITWISE,
        fused=router_tail,
        composed=router_tail_composed,
        gate=GateSpec(inputs=_gate_inputs, output=routing_table, reference=_gate_reference, layers=tuple(range(48))),
    )
)
