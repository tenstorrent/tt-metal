# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``router_tail``: the MoE router tail as one program.  fp32 logits ``[1, 1, rows, 512]`` (TILE, DRAM) -> ROW_MAJOR
bf16 scores and uint16 indices ``[1, 1, rows, top_k]`` (DRAM); the chain ``softmax(numeric_stable) -> topk(k, largest,
sorted) -> sum -> div -> typecast(bf16) -> to_layout(ROW_MAJOR) x2 -> typecast(uint16)`` of ``Qwen38TTNNMoE._route``.

``rows`` 1..32, one tile. The compute kernel issues each replaced op's instruction sequence on CBs of that op's data
format and unpack mode, so the values and the top-k tie order are the chain's (BITWISE). The top-k LLK sorts a tile's
32 token columns in four independent passes of eight tokens (``kernels/topk_lanes.h``); there is one core per pass
that holds a live token, and each core runs only its pass of every sort and writes only its tokens.
"""

from __future__ import annotations

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
# The rectangle the top-k asks placement.free_rectangle for; the lane form takes its first row, so the top-k lands on
# the same cores standalone and hosted by the MoE dense composite.
RECTANGLE = (4, 5)
PASSES = 4  # the LLK local sort's (face, col) passes; pass p = bit p of the compute kernel's pass_mask
# The eight token rows each sort pass covers: pass (face, col) = rows of face `face` with row parity `col`; from a
# device measurement. The four sets partition the tile; a change here moves rows between cores, not the arithmetic.
PASS_TOKENS: tuple[frozenset[int], ...] = tuple(
    frozenset(range(16 * face + col, 16 * face + 16, 2)) for face in range(2) for col in range(2)
)

# Circular buffers: (name, index, dtype, pages); names are the kernels' named compile-time args. The CBs of
# UNPACK_TO_DEST_FP32 unpack straight into the 32-bit dest as the chain's ops do; other fp32 CBs unpack to SrcA/B.
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
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != EXPERTS or not 1 <= rows <= fp.TILE:
        raise ValueError(f"router tail logits must be [1, 1, rows (1..{fp.TILE}), {EXPERTS}], got {list(shape)}")
    if logits.dtype not in (ttnn.float32, ttnn.bfloat16) or logits.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"router tail logits must be fp32 or bf16 TILE, got {logits.dtype} {logits.layout}")
    return rows


_INDEX_TEMPLATES: dict[int, object] = {}


def router_tail_prepare(mesh):
    """The pre-transposed index tiles the top-k sorts alongside the values: tile ``w`` holds ``w*32+k`` in row ``k``.

    One uint32 TILE tensor ``[1, 1, 32, 512]`` in DRAM, allocated once per mesh (call before trace capture)."""

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


def lanes_plan(rows: int, pass_tokens=None) -> list[tuple[int, int]]:
    """``(pass_mask, token_mask)`` per core for ``rows`` live tokens: one core per sort pass holding a live token."""

    pass_tokens = PASS_TOKENS if pass_tokens is None else pass_tokens
    if not 1 <= rows <= fp.TILE:
        raise ValueError(f"router tail lanes: rows must be 1..{fp.TILE}, got {rows}")
    plan = []
    for p, tokens in enumerate(pass_tokens):
        owned = [t for t in range(rows) if t in tokens]
        if owned:
            plan.append((1 << p, sum(1 << t for t in owned)))
    return plan


def _core_plan(rows: int) -> list[tuple[int, int, int]]:
    """``(tile_row, pass_mask, token_mask)`` per core of the one tile."""

    return [(0, pass_mask, token_mask) for pass_mask, token_mask in lanes_plan(rows)]


def lane_cores(rectangle: "ttnn.CoreRange", count: int) -> list["ttnn.CoreCoord"]:
    """The ``count`` lane cores of the form: the first ``count`` cores of the rectangle's first row."""

    width = rectangle.end.x - rectangle.start.x + 1
    if not 1 <= count <= width:
        raise ValueError(f"router tail needs {count} lane cores, the rectangle is {width} wide")
    return [ttnn.CoreCoord(rectangle.start.x + i, rectangle.start.y) for i in range(count)]


def program_parts(
    logits, index_template, scores, indices, *, rows: int, top_k: int, rectangle=None, sem_base: int = 0
) -> tuple[list, list, list]:
    """``(kernels, cbs, semaphores)`` of the top-k program on the lane cores of ``rectangle``, for a hosting program.

    The standalone program is ``fp.program_descriptor(*program_parts(...))``; ``sem_base`` is unused (no semaphores)."""

    plan = _core_plan(rows)
    if rectangle is None:
        rectangle = placement.free_rectangle(logits.device(), *RECTANGLE)
    cores = lane_cores(rectangle, len(plan))
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
    del sem_base  # no program-local semaphores in this form
    return [reader, writer, compute], cbs, []


def router_tail_program(
    logits, index_template, scores, indices, *, rows: int, top_k: int, rectangle=None
) -> "ttnn.ProgramDescriptor":
    kernels, cbs, semaphores = program_parts(
        logits, index_template, scores, indices, rows=rows, top_k=top_k, rectangle=rectangle
    )
    return fp.program_descriptor(kernels, cbs=cbs, semaphores=semaphores)


def router_tail(logits, *, top_k: int = TOP_K, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """``(scores, indices)`` ROW_MAJOR bf16 / uint16 ``[1, 1, rows, top_k]``; the chain's compute config is pinned.

    ``logits`` may be the fp32 tiles or the router linear's bf16 L1 shard: a bf16 value is exact in the 19-bit source
    registers the softmax unpacks fp32 into."""

    rows = _rows_of(logits)
    mesh = logits.device()
    scores = fp.allocate((1, 1, rows, top_k), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    indices = fp.allocate((1, 1, rows, top_k), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    return router_tail_into(logits, scores, indices, top_k=top_k)


def router_tail_into(logits, scores, indices, *, top_k: int = TOP_K):
    """``router_tail`` into pre-allocated ROW_MAJOR ``scores`` (bf16) and ``indices`` (uint16) in any memory layout."""

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
    fp.run_program(
        [logits, index_template, scores, indices],
        router_tail_program(logits, index_template, scores, indices, rows=rows, top_k=top_k),
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
