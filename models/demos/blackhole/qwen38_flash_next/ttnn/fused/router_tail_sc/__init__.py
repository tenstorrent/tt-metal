# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``router_tail_sc``: the router top-k with the sort on its own core ("sort core in parallel"), default off.

``QWEN38_ROUTER_SORTCORE=1`` and one live row (the decode body) select a two-core form of ``router_tail``'s lane
program; every other case (flag unset, rows != 1) gets ``router_tail.program_parts`` unchanged.

  core A  today's softmax (max, precise exp, sum, reciprocal, mul_bcast -> cb_probs), today's sum and division of the
          ten selected probabilities, today's writer.  Its reader gathers the ten probabilities by the sorted indices.
  core B  today's insertion sort (the same LLK network, the same 16 tiles in the same fold order, pass_mask 1) on the
          RAW logits at the same time.  Softmax is strictly monotone and keeps ties, so every compare and tie of the
          network is the one it makes on the probabilities (x18: indices 259 / 259 equal).  B sends its transposed
          index tile into A's cb_idx_t by NoC write, then increments a semaphore on A.

Indices, slot order and the ten probability bits are today's; the sum, the division and the typecast see the same
tile, so the output is bitwise today's.  The kernels are new files here; ``router_tail`` is not changed.  The noc
coordinates of core A come from ``gr_read.noc_map`` (a once-per-mesh probe: call ``prepare`` before a trace capture).
"""

from __future__ import annotations

import os

import ttnn

from .. import gr_read, placement
from .. import program as fp
from .. import router_tail as rt

NAME = "router_tail_sc"
SORTCORE_ENV = "QWEN38_ROUTER_SORTCORE"
KERNELS = {
    "reader_main": fp.kernel_source(NAME, "reader_rt_main.cpp"),
    "reader_sort": fp.kernel_source(NAME, "reader_rt_sort.cpp"),
    "compute_main": fp.kernel_source(NAME, "compute_rt_main.cpp"),
    "compute_sort": fp.kernel_source(NAME, "compute_rt_sort.cpp"),
    "writer": rt.KERNELS["writer"],  # today's writer, unchanged, on core A
}
FAST_ENV = "QWEN38_ROUTER_TAIL_FAST"  # 1: the shortened tail (kernels/*_rtf_*.cpp), default off
FAST_OPTS_ENV = "QWEN38_ROUTER_TAIL_FAST_OPTS"  # study: comma list of FAST_OPTS to enable instead of the default set
FAST_OPTS = ("row0", "nostart", "incr", "writer", "direct", "inline", "idxf", "notoken", "split")
FAST_DEFAULT_OPTS = ("row0", "writer", "direct", "inline", "idxf", "notoken", "split")
FAST_KERNELS = {
    "reader_main": fp.kernel_source(NAME, "reader_rtf_main.cpp"),
    "reader_sort": fp.kernel_source(NAME, "reader_rtf_sort.cpp"),
    "compute_main": fp.kernel_source(NAME, "compute_rtf_main.cpp"),
    "compute_sort": fp.kernel_source(NAME, "compute_rtf_sort.cpp"),
    "writer": fp.kernel_source(NAME, "writer_rtf.cpp"),
    "index": fp.kernel_source(NAME, "writer_rtf_index.cpp"),
}
SORT_PASS_MASK = 1  # the sort pass that holds token 0 (rt.PASS_TOKENS[0])
SORT_CORES = 2
SEMAPHORES = 1  # ids sem_base .. sem_base + 0: B -> A "index tile arrived"


def enabled(rows: int, environ=os.environ) -> bool:
    """The two-core form serves ``rows == 1`` with ``QWEN38_ROUTER_SORTCORE=1`` and the lane form on."""

    return (
        environ.get(SORTCORE_ENV, "1") == "1" and rows == 1 and rt.lanes_enabled(environ)
    )  # default on (x36 A/B -0.65 ms)


def fast_opts(environ=os.environ) -> tuple[str, ...]:
    """The shortened-tail options in force: () unless ``QWEN38_ROUTER_TAIL_FAST=1`` (then the default set, or the
    comma list of ``QWEN38_ROUTER_TAIL_FAST_OPTS``, a study switch)."""

    if environ.get(FAST_ENV, "1") != "1":  # default on (x42 A/B -0.19 ms)
        return ()
    listed = environ.get(FAST_OPTS_ENV)
    opts = FAST_DEFAULT_OPTS if listed is None else tuple(o for o in listed.split(",") if o)
    unknown = set(opts) - set(FAST_OPTS)
    if unknown:
        raise ValueError(f"{FAST_OPTS_ENV}: unknown options {sorted(unknown)}")
    return opts


_FLOAT_INDEX: dict[int, object] = {}


def float_index_template(mesh):
    """``rt.router_tail_prepare``'s index tiles as fp32 (the integers are exact), for the sort without srcA reconfigs."""

    import torch

    key = id(mesh)
    if key not in _FLOAT_INDEX:
        k = torch.arange(fp.TILE, dtype=torch.int32).reshape(fp.TILE, 1)
        tile_base = (torch.arange(rt.EXPERTS, dtype=torch.int32) // fp.TILE * fp.TILE).reshape(1, rt.EXPERTS)
        template = (tile_base + k).float().reshape(1, 1, fp.TILE, rt.EXPERTS)
        _FLOAT_INDEX[key] = ttnn.from_torch(
            template, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return _FLOAT_INDEX[key]


def prepare(mesh):
    """Memoize what the form needs before a trace capture: ``rt.router_tail_prepare`` and the NoC map."""

    table = rt.router_tail_prepare(mesh)
    gr_read.noc_map(mesh)
    if "idxf" in fast_opts():
        float_index_template(mesh)
    return table


def program_parts(
    logits,
    index_template,
    scores,
    indices,
    *,
    rows: int,
    top_k: int,
    rectangle=None,
    sem_base: int = 0,
    index_signal=None,
) -> tuple[list, list, list]:
    """``rt.program_parts``'s contract: ``(kernels, cbs, semaphores)``; the two-core form when ``enabled(rows)``."""

    if not enabled(rows):
        if index_signal is not None:
            raise ValueError("an index signal needs the two-core form of the top-k")
        return rt.program_parts(
            logits, index_template, scores, indices, rows=rows, top_k=top_k, rectangle=rectangle, sem_base=sem_base
        )
    return two_core_parts(
        logits,
        index_template,
        scores,
        indices,
        rows=rows,
        top_k=top_k,
        rectangle=rectangle,
        sem_base=sem_base,
        index_signal=index_signal,
    )


def two_core_parts(
    logits,
    index_template,
    scores,
    indices,
    *,
    rows: int,
    top_k: int,
    rectangle=None,
    sem_base: int = 0,
    index_signal=None,
) -> tuple[list, list, list]:
    """The two-core form itself (rows == 1), whatever the flag says.  ``index_signal`` = ``(semaphore id, [(noc x, noc y)])``
    (the moe_experts fused form, QWEN38_MOE_FUSE=1): core A's fast writer raises that semaphore on those cores once the
    index rows are in the drain shard (define MOE_FUSE_SEM; with None the kernels, defines and arguments are exactly
    the unsignalled form's)."""

    if rows != 1:
        raise ValueError(f"router_tail_sc serves rows == 1, got {rows}")
    if rectangle is None:
        rectangle = placement.free_rectangle(logits.device(), *rt.RECTANGLE)
    # "direct" sorts the logits CB as is: fp32 logits; QWEN38_ROUTER_L1=1 feeds the router linear's bf16 L1 shard, whose
    # CB the sort core also unpacks to DST in fp32 (the same values the drain widened)
    opts = fast_opts() if (logits.dtype == ttnn.float32 or os.environ.get("QWEN38_ROUTER_L1", "0") == "1") else ()
    if (
        logits.dtype != ttnn.float32
    ):  # a bf16 CB takes the copy pass (the direct bf16 transpose measured wrong: 0 / 768)
        opts = tuple(o for o in opts if o not in ("direct", "inline"))
    kernels_of = FAST_KERNELS if opts else KERNELS
    if opts and "writer" not in opts:
        kernels_of = {**FAST_KERNELS, "writer": KERNELS["writer"]}
    if "idxf" in opts:
        index_template = float_index_template(logits.device())
    core_a, core_b = rt.lane_cores(rectangle, SORT_CORES)
    noc = gr_read.noc_map(logits.device())
    a_x, a_y = noc[(core_a.x, core_a.y)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core_a, core_b)])
    set_a = ttnn.CoreRangeSet([ttnn.CoreRange(core_a, core_a)])
    set_b = ttnn.CoreRangeSet([ttnn.CoreRange(core_b, core_b)])
    named = [(name, index) for name, index, _dtype, _pages in rt.CBS] + [
        ("Wt", rt.WIDTH_TILES),
        ("top_k", top_k),
        ("stage_pages", rt.STAGE_PAGES),
        ("sem_id", sem_base),
    ]
    cbs = [
        fp.cb_descriptor(
            index,
            ttnn.float32
            if "idxf" in opts and name in ("cb_index", "cb_idx_t")
            else logits.dtype
            if name == "cb_in0"
            else dtype,
            rt.TOKEN_PAGE_BYTES if name in rt.TOKEN_CBS else fp.TILE_BYTES[logits.dtype if name == "cb_in0" else dtype],
            pages,
            grid,
        )
        for name, index, dtype, pages in rt.CBS
    ]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64  # one per circular buffer of the runtime (64)
    for name in rt.UNPACK_TO_DEST_FP32:
        modes[rt.CB_INDEX[name]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    compute_sort_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
    )
    sort_modes = list(modes)
    if "idxf" in opts:
        sort_modes[rt.CB_INDEX["cb_index"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if (
        "direct" in opts or "inline" in opts
    ) and logits.dtype == ttnn.float32:  # the sort transposes cb_in0 straight into DST
        sort_modes[
            rt.CB_INDEX["cb_in0"]
        ] = ttnn.UnpackToDestMode.UnpackToDestFp32  # (a bf16 CB keeps the srcA transpose)
    compute_sort_config.unpack_to_dest_mode = sort_modes
    define = lambda *names: [(f"RTF_{n.upper()}", "1") for n in names if n in opts]  # noqa: E731

    def kernel(source, cores, compile_time_args, runtime_args, config, defines=()):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[int(a) for a in compile_time_args],
            named_compile_time_args=named,
            defines=[*defines, *fp.zone_defines()],
            runtime_args=runtime_args,
            config=config,
        )

    logits_addr, index_addr = logits.buffer_address(), index_template.buffer_address()
    # one live token, row 0: pass_mask = SORT_PASS_MASK, token_mask = 1, tile row 0, rows_in_tile 1
    reader_main = kernel(
        kernels_of["reader_main"],
        set_a,
        fp.accessor_args(logits),
        [(core_a, [logits_addr, 0, 1])],
        ttnn.ReaderConfigDescriptor(),
        defines=define("row0", "idxf", "notoken"),
    )
    writer_args = [scores.buffer_address(), indices.buffer_address(), 0, 1, 1]
    writer_defines = define("idxf")
    if index_signal is not None:
        if kernels_of["writer"] != FAST_KERNELS["writer"]:
            raise ValueError("the index signal is in the fast writer only (QWEN38_ROUTER_TAIL_FAST)")
        sem_id, signalled = index_signal
        writer_args += [len(signalled), *[v for xy in signalled for v in xy]]
        writer_defines = [*writer_defines, ("MOE_FUSE_SEM", str(sem_id))]
    writer = kernel(
        kernels_of["writer"],
        set_a,
        fp.accessor_args(scores) + fp.accessor_args(indices),
        [(core_a, writer_args)],
        ttnn.WriterConfigDescriptor(),
        defines=writer_defines,
    )
    compute_main = kernel(
        kernels_of["compute_main"],
        set_a,
        [],
        [(core_a, [SORT_PASS_MASK, 1])],
        compute_config,
        defines=[*rt._compute_defines(), *define("nostart", "notoken")],
    )
    reader_sort = kernel(
        kernels_of["reader_sort"],
        set_b,
        fp.accessor_args(logits) + fp.accessor_args(index_template),
        [(core_b, [logits_addr, index_addr, 0, a_x, a_y])],
        ttnn.ReaderConfigDescriptor(),
        defines=define("incr", "idxf", "split"),
    )
    compute_sort = kernel(
        kernels_of["compute_sort"],
        set_b,
        [],
        [(core_b, [SORT_PASS_MASK, 1])],
        compute_sort_config,
        defines=define("incr", "direct", "inline", "idxf"),
    )
    semaphores = [fp.semaphore_descriptor(sem_base, set_a)]
    kernels = [reader_main, writer, compute_main, reader_sort, compute_sort]
    if "split" in opts:  # core B's second data-movement RISC reads the index tiles while the reader reads the logits
        kernels.append(
            kernel(
                FAST_KERNELS["index"],
                set_b,
                fp.accessor_args(index_template),
                [(core_b, [index_addr])],
                ttnn.WriterConfigDescriptor(),
            )
        )
    return kernels, cbs, semaphores


def router_tail_sc_into(logits, scores, indices, *, top_k: int = rt.TOP_K):
    """``rt.router_tail_into`` on the two-core form (rows == 1, regardless of the flag); a test and bench entry."""

    rows = rt._rows_of(logits)
    if rows != 1:
        raise ValueError(f"router_tail_sc serves rows == 1, got {rows}")
    index_template = prepare(logits.device())
    meta = fp.program_meta(
        NAME,
        "sort_core",
        rows,
        writes=(scores, indices),
        dram_bytes=fp.tensor_bytes(index_template) + SORT_CORES * (0 if fp.in_l1(logits) else fp.tensor_bytes(logits)),
        l1_bytes=SORT_CORES * fp.tensor_bytes(logits) if fp.in_l1(logits) else 0,
        flops=rows * rt.EXPERTS * (5 + 2 * rt.WIDTH_TILES) + rows * top_k * 3,
        cores=SORT_CORES,
    )
    descriptor = fp.program_descriptor(*two_core_parts(logits, index_template, scores, indices, rows=rows, top_k=top_k))
    fp.run_program([logits, index_template, scores, indices], descriptor, meta=meta)
    return scores, indices
