# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``moe_dense``: the MoE block's router top-k co-scheduled with the shared expert chain and the routed dispatch
untilize as ONE program on disjoint core groups.

Today the block runs, per layer, the router top-k (``router_tail``, one lane core, 51 us) and then the shared expert's
eltwise program (5 cores, 4.6 us), its down linear (6.9 us) and the routed dispatch untilize (1.9 us) one after another,
although the four depend on nothing but the gathered hidden and the linears before them (programs never overlap:
the census of record's kernel time plus its launch gaps is the step; the top-k's lane core was also storage core 0 of
the shared linear).  This program hosts the four as kernel groups on disjoint cores, so the 13.4 us of shared work run
under the top-k and three launch boundaries disappear:

  G1  the top-k: ``router_tail.program_parts`` unchanged, on the placement helper's 4 x 5 rectangle (its first row);
  G2  the shared eltwise: ``shared_expert``'s reader and compute unchanged on the five storage cores ``(0..4, 0)``;
      its writer is gr_read's multicast writer, which multicasts the core's intermediate tile into the down workers'
      in0 CB and raises their semaphore (and writes the sigmoid tile from core 0);
  G3  the down linear: 16 worker cores (a NoC-contiguous rectangle), each streaming five weight columns while the
      eltwise computes, then gr_read's streaming-matmul kernel per column (``kernels/down_compute.cpp``: fp32 dest,
      the DRAM-sharded matmul's spill/reload after every K tile since its in0_block_w is 1 for K = 160 over five
      storage cores, packed bf16 into the partial's five-core width shard);
  G4  the dispatch untilize: ``untilize_rows``' kernels (the stock interleaved reader, ``writer_rows.cpp``) on eight
      cores, the gathered hidden's 80 tiles as ROW_MAJOR rows for ``moe_compute``.

Outputs and their consumers are unchanged: the routing rows land where the caller allocated them (the drain-core L1
shards or DRAM rows), the ungated partial in the five-core width shard and the sigmoid tile go to ``moe_post``, the
rows to ``moe_compute``.  Tolerance class BITWISE: G1, G2 and G4 are the existing kernels by path (G4 pure data
movement), G3 reproduces ``ttnn.linear``'s arithmetic order (the gr_read DOWN precedent) and is proven against it on
the device at both weight formats (the device test and the audit-capture microtest: rows 1/5/32, bf16 and bf8, both
routing placements, 2026-09-25).  Rows 1..32 (one tile); the MTP verify rows (5) are two lane cores of G1.  Serves by
default since 2026-09-26; ``QWEN38_FUSED_OFF=moe_dense`` restores the four programs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .. import gr_read, placement
from .. import program as fp
from .. import router_tail as rt
from .. import shared_expert as se
from .. import untilize_rows as ur
from ..registry import BITWISE, FusedKernel, GateSpec, register
from ...decode_matmul import DENSE_MATH_FIDELITY_NAMES

NAME = "moe_dense"
HIDDEN = se.HIDDEN
TOP_K = rt.TOP_K
BF16 = ttnn.bfloat16
K_TILES = se.LOCAL_INTERMEDIATE // fp.TILE  # 5: the down linear's K
N_TILES = HIDDEN // fp.TILE  # 80 output column tiles
DOWN_WORKERS = 16
N_PER_WORKER = N_TILES // DOWN_WORKERS  # 5
# The multicast reader streams its WHOLE weight stream into CB_IN1 before it waits for the eltwise cores' semaphore
# and the compute waits for the whole in0 row: a CB smaller than its stream would block the reader on a compute that
# is itself blocked (a hang on the partition), so the CB page counts and the stream lengths share these constants.
DOWN_STREAM_TILES = N_PER_WORKER * K_TILES  # weight tiles per worker = CB_IN1 pages
DOWN_SHAPES = ((4, 4), (8, 2), (2, 8), (16, 1))  # the worker rectangle shapes tried, in order
UNTILIZE_SHAPE = (4, 2)
# The DRAM-sharded down linear's K block is one tile (decode_matmul: in0_block_w = the largest divisor <= 8 of the K
# tiles per storage core, 160 / (32 x 5) = 1), so it spills and reloads its fp32 partial after every K tile.
DOWN_SPILL = 1
DOWN_BLK = 1
DOWN = fp.kernel_source(NAME, "down_compute.cpp")
# Circular buffers of G3 (indices free of G2's 0-4 / 16-17 on the storage cores, which also declare CB_IN0 as the
# multicast source view); G1 and G4 keep their own tables on their own cores.
CB_IN0, CB_IN1, CB_INTERM, CB_OUT = 8, 9, 10, 11
UNTILIZE_CB = 0  # the stock interleaved reader and writer_rows.cpp pin cb 0
TOPK_SEMAPHORES = 5  # ids reserved for G1 (the multi-core top-k's one semaphore, its fallback's four more)
SEM_DOWN = TOPK_SEMAPHORES
_PLANS: dict[int, "DensePlan"] = {}


@dataclass(frozen=True)
class DensePlan:
    """The composite's core groups on one mesh: the top-k rectangle (G1), the down workers (G3; row-major inside a
    NoC-contiguous rectangle, with the NoC rectangle the eltwise cores multicast to) and the untilize cores (G4)."""

    topk: ttnn.CoreRange
    workers: ttnn.CoreRange
    workers_noc: tuple[int, int, int, int]
    untilize: ttnn.CoreRange

    def worker_cores(self) -> list[ttnn.CoreCoord]:
        return placement.cores_of(self.workers)

    def untilize_cores(self) -> list[ttnn.CoreCoord]:
        return placement.cores_of(self.untilize)


def _noc_rectangle(noc, rect) -> tuple[int, int, int, int] | None:
    """The NoC-0 rectangle of the logical rectangle when its cores are NoC-contiguous, else None."""

    cores = placement.rectangle_cores(rect)
    xs = sorted({noc[c][0] for c in cores})
    ys = sorted({noc[c][1] for c in cores})
    width, height = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
    if xs != list(range(xs[0], xs[0] + width)) or ys != list(range(ys[0], ys[0] + height)):
        return None
    return (xs[0], ys[0], xs[-1], ys[-1])


def plan_for(mesh) -> DensePlan:
    """The placement on ``mesh`` (memoized; the NoC map probe and the reserved-core reads run once, before any
    capture): the top-k's rectangle first (the same call the standalone top-k makes), then the down workers as the
    first free shape of ``DOWN_SHAPES`` that is a contiguous NoC rectangle, then the untilize cores."""

    key = id(mesh)
    if key not in _PLANS:
        topk = placement.free_rectangle(mesh, *rt.RECTANGLE)
        avoid = set(placement.rectangle_cores(placement.range_rect(topk)))
        noc = gr_read.noc_map(mesh)
        workers = None
        for width, height in DOWN_SHAPES:
            try:
                workers = placement.free_rectangle(
                    mesh, width, height, avoid=avoid, accept=lambda rect: _noc_rectangle(noc, rect) is not None
                )
                break
            except RuntimeError:
                continue
        if workers is None:
            raise RuntimeError(f"no NoC-contiguous free rectangle of {DOWN_WORKERS} cores for the down linear")
        avoid |= set(placement.rectangle_cores(placement.range_rect(workers)))
        untilize = placement.free_rectangle(mesh, *UNTILIZE_SHAPE, avoid=avoid)
        _PLANS[key] = DensePlan(topk, workers, _noc_rectangle(noc, placement.range_rect(workers)), untilize)
    return _PLANS[key]


def _fidelity_of(compute_kernel_config):
    """The down linear's fidelity from the model's compute kernel config; the rest of that config must be what the
    kernel hard-codes (fp32 dest accumulation, no packer L1 accumulation, no approximations), or the stock linear's
    spill format and accumulation would differ from the composite's."""

    fidelity = getattr(compute_kernel_config, "math_fidelity", None)
    if fidelity is None:
        raise ValueError("moe_dense needs the shared linears' device compute kernel config (math_fidelity)")
    settings = {
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "math_approx_mode": False,
    }
    for name, expected in settings.items():
        if bool(getattr(compute_kernel_config, name, not expected)) is not expected:
            raise ValueError(
                f"moe_dense reproduces the down linear only with {name}={expected}, got {getattr(compute_kernel_config, name, None)}"
            )
    return fidelity


def moe_dense_program(
    *,
    logits,
    index_template,
    scores,
    indices,
    gate_up_scalar_ws,
    sigmoid,
    down,
    partial,
    full_hidden,
    sparse_rows,
    rows: int,
    top_k: int,
    plan: DensePlan,
    fidelity,
) -> "ttnn.ProgramDescriptor":
    # G1: the top-k on the rectangle's lane cores
    topk_kernels, topk_cbs, topk_semaphores = rt.program_parts(
        logits, index_template, scores, indices, rows=rows, top_k=top_k, rectangle=plan.topk, sem_base=0
    )
    # G2: the shared eltwise on the storage cores, multicasting the intermediate into G3's in0 CB
    storage = [ttnn.CoreCoord(c, 0) for c in range(se.STORAGE_CORES)]
    storage_set = ttnn.CoreRangeSet([ttnn.CoreRange(storage[0], storage[-1])])
    workers = plan.worker_cores()
    worker_set = ttnn.CoreRangeSet([plan.workers])
    all_set = ttnn.CoreRangeSet([ttnn.CoreRange(storage[0], storage[-1]), plan.workers])
    se_named = [(name, index) for name, index, _dtype, _pages in se.CBS] + [("gate_tiles", se.GATE_TILES)]
    eltwise_cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, storage_set) for _n, index, dtype, pages in se.CBS
    ]
    in0_cb = fp.cb_descriptor(CB_IN0, BF16, fp.TILE_BYTES[BF16], K_TILES, all_set)
    eltwise_reader = fp.reader_kernel(
        se.KERNELS["reader"],
        storage_set,
        fp.accessor_args(gate_up_scalar_ws),
        [(core, [gate_up_scalar_ws.buffer_address(), c, int(c == 0)]) for c, core in enumerate(storage)],
        named=se_named,
    )
    eltwise_compute = fp.compute_kernel(
        se.KERNELS["compute"],
        storage_set,
        [],
        [(core, [int(c == 0)]) for c, core in enumerate(storage)],
        named=se_named,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=False,
    )
    eltwise_writer = gr_read._mcast_writer(
        storage_set,
        [(core, (c, plan.workers_noc, (0, 1), (1 if c == 0 else 0, 0, 1, 1))) for c, core in enumerate(storage)],
        src_cb=se.CB_INDEX["cb_inter"],
        dst_cb=CB_IN0,
        tiles=1,
        tiles_tensor=None,
        extra=(sigmoid, se.CB_INDEX["cb_sig_bcast"]),
        sem=SEM_DOWN,
    )
    # G3: the down linear, N_PER_WORKER output tiles per worker (CB_IN1 holds the whole DOWN_STREAM_TILES stream)
    down_cbs = [
        fp.cb_descriptor(CB_IN1, down.dtype, fp.TILE_BYTES[down.dtype], DOWN_STREAM_TILES, worker_set),
        fp.cb_descriptor(CB_INTERM, ttnn.float32, fp.TILE_BYTES[ttnn.float32], 1, worker_set),
        fp.cb_descriptor(CB_OUT, BF16, fp.TILE_BYTES[BF16], 2, worker_set),
    ]
    down_reader = gr_read._mcast_reader(
        worker_set,
        [(down, CB_IN1)],
        [
            (core, [gr_read._stream(down, N_PER_WORKER, K_TILES, w * N_PER_WORKER, N_TILES, 1, DOWN_BLK)])
            for w, core in enumerate(workers)
        ],  # outer x inner = DOWN_STREAM_TILES tiles: the CB_IN1 capacity
        recv_cb=CB_IN0,
        recv_tiles=K_TILES,  # = the CB_IN0 pages: the five eltwise cores' tiles
        senders=se.STORAGE_CORES,
        sem=SEM_DOWN,
    )
    down_compute = fp.compute_kernel(
        DOWN,
        worker_set,
        [K_TILES, DOWN_BLK, CB_IN0, CB_IN1, CB_OUT, DOWN_SPILL, CB_INTERM, N_PER_WORKER],
        fidelity=fidelity,
        fp32_dest=True,
    )
    down_writer = gr_read._writer(
        worker_set,
        [(partial, CB_OUT)],
        [(core, [(N_PER_WORKER, w * N_PER_WORKER, 1, 1)]) for w, core in enumerate(workers)],
    )
    # G4: the routed dispatch untilize
    untilize_cores = plan.untilize_cores()
    untilize_set = ttnn.CoreRangeSet([plan.untilize])
    per_core, extra = divmod(N_TILES, len(untilize_cores))
    starts, start = [], 0
    for i in range(len(untilize_cores)):
        count = per_core + (1 if i < extra else 0)
        starts.append((start, count))
        start += count
    untilize_cb = fp.cb_descriptor(UNTILIZE_CB, BF16, fp.TILE_BYTES[BF16], 2, untilize_set)
    untilize_reader = fp.reader_kernel(
        ur.READER,
        untilize_set,
        fp.accessor_args(full_hidden),
        [(core, [full_hidden.buffer_address(), count, first]) for core, (first, count) in zip(untilize_cores, starts)],
    )
    untilize_writer = fp.writer_kernel(
        ur.WRITER,
        untilize_set,
        [fp.ELEMENT_BYTES[BF16], rows, *fp.accessor_args(sparse_rows)],
        [(core, [sparse_rows.buffer_address(), count, first]) for core, (first, count) in zip(untilize_cores, starts)],
    )
    return fp.program_descriptor(
        [
            *topk_kernels,
            eltwise_reader,
            eltwise_compute,
            eltwise_writer,
            down_reader,
            down_compute,
            down_writer,
            untilize_reader,
            untilize_writer,
        ],
        cbs=[*topk_cbs, *eltwise_cbs, in0_cb, *down_cbs, untilize_cb],
        semaphores=[*topk_semaphores, fp.semaphore_descriptor(SEM_DOWN, all_set)],
    )


def _check(logits, gate_up_scalar_ws, down, full_hidden, scores, indices, top_k) -> int:
    rows = rt._rows_of(logits)
    if rows > fp.TILE:
        raise ValueError(f"moe_dense takes one row tile (rows 1..{fp.TILE}), got {rows}")
    if fp.rows_of(full_hidden) != rows or fp.tile_width_of(full_hidden) != HIDDEN or full_hidden.dtype != BF16:
        raise ValueError(f"moe_dense hidden must be bf16 TILE [1, 1, {rows}, {HIDDEN}], got {list(full_hidden.shape)}")
    if tuple(int(v) for v in gate_up_scalar_ws.shape) != (1, 1, rows, se.CAT_WIDTH) or gate_up_scalar_ws.dtype != BF16:
        raise ValueError(
            f"moe_dense takes the [gate | up | scalar] linear's bf16 [1, 1, {rows}, {se.CAT_WIDTH}] shard, got "
            f"{list(gate_up_scalar_ws.shape)}"
        )
    if tuple(int(v) for v in down.shape) != (1, 1, se.LOCAL_INTERMEDIATE, HIDDEN) or down.dtype not in (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ):
        raise ValueError(
            f"moe_dense down weight must be bf16 or bf8 TILE [1, 1, {se.LOCAL_INTERMEDIATE}, {HIDDEN}], got "
            f"{list(down.shape)} {down.dtype}"
        )
    for name, tensor, dtype in (("scores", scores, BF16), ("indices", indices, ttnn.uint16)):
        shape = tuple(int(v) for v in tensor.shape)
        if shape[-2:] != (rows, top_k) or tensor.dtype != dtype or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"moe_dense {name} output must be ROW_MAJOR {dtype} [.., {rows}, {top_k}], got {shape}")
    return rows


def moe_dense(
    logits,
    gate_up_scalar_ws,
    down,
    full_hidden,
    *,
    scores,
    indices,
    top_k: int = TOP_K,
    compute_kernel_config,
    partial_memory_config,
    **_composed_only,
):
    """One program: the top-k of ``logits`` (fp32 TILE ``[1, 1, rows, 512]``) into the pre-allocated ROW_MAJOR
    ``scores`` / ``indices``; the shared eltwise on the ``[gate | up | scalar]`` linear's shard and its down linear
    (``down`` DRAM width-sharded ``[160, 2560]``; the fidelity of ``compute_kernel_config``) into the ungated
    ``partial`` (``partial_memory_config``: the five-core width shard) and the sigmoid tile; the gathered hidden's rows
    as ``sparse_rows`` (ROW_MAJOR bf16, DRAM).  Returns ``(scores, indices, partial, sigmoid, sparse_rows)``."""

    rows = _check(logits, gate_up_scalar_ws, down, full_hidden, scores, indices, top_k)
    mesh = logits.device()
    plan = plan_for(mesh)
    index_template = rt.router_tail_prepare(mesh)
    partial = fp.allocate((1, 1, rows, HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh, partial_memory_config)
    sigmoid = fp.allocate((1, 1, fp.TILE, fp.TILE), BF16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    sparse_rows = fp.allocate((1, 1, rows, HIDDEN), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    fidelity = _fidelity_of(compute_kernel_config)
    lanes = len(rt.lane_cores(plan.topk, len(rt._core_plan(rows)[0])))
    # G1 as router_tail states it (every lane core streams the logits and the index template); G2 the [gate | up |
    # scalar] shard once and its multicast of the intermediate into the 16 workers; G3 the down weight once (each
    # worker its five columns) and the partial out; G4 the hidden shard once and the rows out.  FLOPs: the top-k's
    # softmax + sort + normalization, the eltwise's silu / product / sigmoid, the down matmul 2 x rows x K x N.
    meta = fp.program_meta(
        NAME,
        "composite",
        rows,
        reads=(gate_up_scalar_ws, down, full_hidden),
        writes=(scores, indices, partial, sigmoid, sparse_rows),
        dram_bytes=lanes * (fp.tensor_bytes(index_template) + fp.tensor_bytes(logits)),
        l1_bytes=DOWN_WORKERS * K_TILES * fp.TILE_BYTES[BF16],
        flops=rows * rt.EXPERTS * (5 + 2 * rt.WIDTH_TILES)
        + rows * top_k * 3
        + rows * se.LOCAL_INTERMEDIATE * 3
        + rows * fp.TILE
        + 2 * rows * se.LOCAL_INTERMEDIATE * HIDDEN,
        cores=lanes + se.STORAGE_CORES + DOWN_WORKERS + len(plan.untilize_cores()),
        # every output is per device the same shape as the chain's (replicated placements, as the caller stamped them)
        outputs=((scores, None), (indices, None), (partial, None), (sigmoid, None), (sparse_rows, None)),
    )
    fp.run_program(
        [logits, index_template, gate_up_scalar_ws, down, full_hidden, scores, indices, partial, sigmoid, sparse_rows],
        moe_dense_program(
            logits=logits,
            index_template=index_template,
            scores=scores,
            indices=indices,
            gate_up_scalar_ws=gate_up_scalar_ws,
            sigmoid=sigmoid,
            down=down,
            partial=partial,
            full_hidden=full_hidden,
            sparse_rows=sparse_rows,
            rows=rows,
            top_k=top_k,
            plan=plan,
            fidelity=fidelity,
        ),
        meta=meta,
    )
    # generic_op rewrites the mesh topology of the io tensor it returns (the last one) from its inputs: with the
    # width-sharded down weight among them the rows came back [Replicate, Shard] and the served model's replicated
    # contract refused them (the first served run, 2026-09-26).  Every output keeps the gathered hidden's replicated topology.
    for tensor in (scores, indices, partial, sigmoid, sparse_rows):
        fp.stamp_topology(tensor, full_hidden)
    return scores, indices, partial, sigmoid, sparse_rows


def moe_dense_composed(
    logits,
    gate_up_scalar_ws,
    down,
    full_hidden,
    *,
    scores,
    indices,
    top_k: int = TOP_K,
    compute_kernel_config,
    down_program_config,
    intermediate_memory_config,
    **_fused_only,
):
    """The four programs as the model issues them today: ``router_tail_into``, the shared eltwise program, the down
    ``ttnn.linear`` and ``ttnn.to_layout`` of the gathered hidden.  Same returns as ``moe_dense``."""

    rows = _check(logits, gate_up_scalar_ws, down, full_hidden, scores, indices, top_k)
    mesh = logits.device()
    rt.router_tail_into(logits, scores, indices, top_k=top_k)
    intermediate = fp.stamp_topology(
        fp.allocate((1, 1, rows, se.LOCAL_INTERMEDIATE), BF16, ttnn.TILE_LAYOUT, mesh, intermediate_memory_config),
        full_hidden,
    )
    sigmoid = fp.stamp_topology(
        fp.allocate((1, 1, fp.TILE, fp.TILE), BF16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG), full_hidden
    )
    fp.run_program(
        [gate_up_scalar_ws, intermediate, sigmoid],
        se.shared_eltwise_program(gate_up_scalar_ws, intermediate, sigmoid),
        meta=fp.program_meta(  # the chain's eltwise program as shared_expert states it
            NAME,
            "composed_eltwise",
            rows,
            reads=(gate_up_scalar_ws,),
            writes=(intermediate, sigmoid),
            flops=rows * se.LOCAL_INTERMEDIATE * 3 + rows * fp.TILE,
            cores=se.STORAGE_CORES,
            outputs=((intermediate, None), (sigmoid, None)),
        ),
    )
    partial = ttnn.linear(
        intermediate,
        down,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=down_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(intermediate)
    sparse_rows = ttnn.to_layout(full_hidden, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return scores, indices, partial, sigmoid, sparse_rows


def dense_rows(result) -> torch.Tensor:
    """``[rows, 20 + 2560 + 1 + 2560]`` fp32: the routing table (bf16 scores widened, indices), the ungated partial,
    the sigmoid of the row (column 0 of the broadcast tile) and the untilized hidden row; bitwise-comparable."""

    scores, indices, partial, sigmoid, sparse_rows = result
    rows = int(scores.shape[-2])
    routing = rt.routing_table((scores, indices))
    partial_rows = ttnn.to_torch(partial).float().reshape(-1, HIDDEN)[:rows]
    sigmoid_col = ttnn.to_torch(sigmoid).float().reshape(fp.TILE, fp.TILE)[:rows, :1]
    hidden_rows = ttnn.to_torch(sparse_rows).float().reshape(-1, HIDDEN)[:rows]
    return torch.cat([routing, partial_rows, sigmoid_col, hidden_rows], dim=-1)


def configs(mesh, weight_dtype=BF16) -> dict:
    """The model's shared-linear configs for the composite and its chain: the down program config, the intermediate
    shard, the partial's five-core width shard (the hidden's), the compute config at the weight format's fidelity."""

    shared = se.gate_configs(mesh)
    return {
        "compute_kernel_config": ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, DENSE_MATH_FIDELITY_NAMES[weight_dtype]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
        "down_program_config": shared["down_program_config"],
        "intermediate_memory_config": shared["intermediate_memory_config"],
        "partial_memory_config": shared["hidden_act_memory_config"],
    }


def _gate_inputs(mesh, capture, positions, layer):
    """The captured router logits and MoE input rows of ``positions`` at ``layer``; deterministic bf16 shared weights
    (the class is bitwise between the two forms); the ``[gate | up | scalar]`` linear run on the device; DRAM routing
    rows pre-allocated for both forms."""

    index = [capture["positions"].index(p) for p in positions]
    rows = len(positions)
    logits = ttnn.from_torch(
        capture["router_logits"][index, layer].float().reshape(1, 1, rows, rt.EXPERTS),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    generator = torch.Generator().manual_seed(20260925 + layer)
    gate_t = torch.randn(HIDDEN, se.INTERMEDIATE, generator=generator) * 0.02
    up_t = torch.randn(HIDDEN, se.INTERMEDIATE, generator=generator) * 0.02
    scalar_t = torch.randn(HIDDEN, 1, generator=generator) * 0.02
    down_t = torch.randn(se.INTERMEDIATE, HIDDEN, generator=generator) * 0.02
    shared = se.gate_configs(mesh)
    weights = se.upload_weights(mesh, gate_t, up_t, scalar_t, down_t)
    hidden_dram = ttnn.from_torch(
        capture["mlp_in"][index, layer].to(torch.bfloat16).reshape(1, 1, rows, HIDDEN),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    full_hidden = ttnn.to_memory_config(hidden_dram, shared["hidden_act_memory_config"])
    ttnn.deallocate(hidden_dram)
    gate_up_scalar_ws = ttnn.linear(
        full_hidden,
        weights["gate_up_scalar"],
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=shared["gate_up_scalar_program_config"],
        compute_kernel_config=shared["compute_kernel_config"],
    )
    return {
        "logits": logits,
        "gate_up_scalar_ws": gate_up_scalar_ws,
        "down": weights["down"],
        "full_hidden": full_hidden,
        "scores": fp.allocate((1, 1, rows, TOP_K), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh),
        "indices": fp.allocate((1, 1, rows, TOP_K), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh),
        **configs(mesh),
    }


register(
    FusedKernel(
        name=NAME,
        replaces="router_tail top-k, the shared eltwise program, the shared down linear and the routed dispatch "
        "untilize (4 programs per layer -> 1, the shared work under the top-k)",
        tolerance=BITWISE,
        fused=moe_dense,
        composed=moe_dense_composed,
        gate=GateSpec(inputs=_gate_inputs, output=dense_rows, reference=None, layers=tuple(range(48))),
    )
)
