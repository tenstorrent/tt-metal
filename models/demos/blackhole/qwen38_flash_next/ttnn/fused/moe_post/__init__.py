# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``moe_post``: everything after ``moe_compute`` in one program per layer.

The composed chain (``Qwen38TTNNMoE._routed_partial`` and the partial add in ``forward``): ``fill`` of the local
combine buffer, ``unsqueeze`` + ``to_layout(TILE)`` of the ``[10, rows, 2560]`` ROW_MAJOR pages,
``deepseek_moe_fast_reduce_nc_fused`` (the score-weighted sum over the ten slots, the slots this device does not own
scored 0), then ``ttnn.add`` of the routed and the shared partial -- with the shared expert's ``x sigmoid`` multiply
folded in when the shared partial arrives ungated.  One program of 80 cores (one output column tile each): the reader
zeroes the activation pages with the NoC and gathers only the owned ``(row, slot)`` fragments of the pages, the compute kernel issues the chain's MAC, multiply
and add instruction sequences (``kernels/compute_moe_post.cpp``), so the output is the chain's bit for bit
(tolerance class BITWISE).  Rows 1..32 (one row tile).
"""

from __future__ import annotations

import os

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register

NAME = "moe_post"
TOP_K = 10
HIDDEN = 2560
HIDDEN_TILES = HIDDEN // fp.TILE
EXPERTS = 512
EXPERTS_PER_DEVICE = EXPERTS // 4
OWNER_BYTES = EXPERTS * 2
ROUTE_PAGE_BYTES = 32 * 64  # 32 staged routing rows at a 64-byte pitch (a DRAM page's own alignment phase)
FRAGMENT_BYTES = 64
KERNELS = {name: fp.kernel_source(NAME, f"{name}_moe_post.cpp") for name in ("reader", "compute", "writer")}
BF16, U16 = ttnn.bfloat16, ttnn.uint16
# (name, index, dtype, page bytes, pages); the names are the kernels' named compile-time args
CBS = (
    ("cb_act", 0, BF16, fp.TILE_BYTES[BF16], TOP_K),
    ("cb_scores", 1, BF16, fp.TILE_BYTES[BF16], TOP_K),
    ("cb_routed", 2, BF16, fp.TILE_BYTES[BF16], 1),
    ("cb_shared", 3, BF16, fp.TILE_BYTES[BF16], 1),
    ("cb_sig", 4, BF16, fp.TILE_BYTES[BF16], 1),
    ("cb_gated", 5, BF16, fp.TILE_BYTES[BF16], 1),
    ("cb_owner", 6, U16, OWNER_BYTES, 1),
    ("cb_route", 7, U16, ROUTE_PAGE_BYTES, 2),  # indices rows, then scores rows
    ("cb_stage", 8, BF16, FRAGMENT_BYTES, None),  # rows x top_k owned fragments + one for the 64-byte alignment
    ("cb_out", 16, BF16, fp.TILE_BYTES[BF16], 1),
)
CB_INDEX = {name: index for name, index, _dtype, _bytes, _pages in CBS}
READER_ARGS = ("pages", "scores", "indices", "owner", "shared", "sig", "tile_col", "rows")
WRITER_ARGS = ("out", "tile_col")
DEVICE_ENV = "QWEN38_FUSED_MOE_POST_DEVICE"  # the gate's device index (owner row) on one chip


def owner_rows(devices: int = 4) -> torch.Tensor:
    """``[devices, 512]`` uint16: row ``d`` marks the experts device ``d`` owns (the 128-expert EP4 shards)."""

    experts = torch.arange(EXPERTS) // EXPERTS_PER_DEVICE
    return (experts.unsqueeze(0) == torch.arange(devices).unsqueeze(1)).to(torch.int16)


def routing_rows_of(scores, indices) -> int:
    """The token rows of the ROW_MAJOR routing tables (``[1, 1, rows, 10]`` in DRAM or ``[1, rows, 10]`` in L1)."""

    for name, tensor, dtype in (("scores", scores, BF16), ("indices", indices, U16)):
        shape = tuple(int(v) for v in tensor.shape)
        if shape[-1] != TOP_K or not 1 <= shape[-2] <= fp.TILE or any(v != 1 for v in shape[:-2]):
            raise ValueError(f"moe_post {name} must be [.., rows (1..32), {TOP_K}], got {list(shape)}")
        if tensor.dtype != dtype or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"moe_post {name} must be ROW_MAJOR {dtype}, got {tensor.layout} {tensor.dtype}")
    if tuple(scores.shape)[-2] != tuple(indices.shape)[-2]:
        raise ValueError("moe_post scores and indices disagree on the rows")
    return int(scores.shape[-2])


def _check_inputs(pages, scores, indices, owner, shared, sigmoid) -> int:
    rows = routing_rows_of(scores, indices)
    if (
        tuple(int(v) for v in pages.shape) != (TOP_K, rows, HIDDEN)
        or pages.dtype != BF16
        or pages.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"moe_post pages must be ROW_MAJOR bf16 [{TOP_K}, {rows}, {HIDDEN}], got {list(pages.shape)}")
    if (
        tuple(int(v) for v in owner.shape) != (1, EXPERTS)
        or owner.dtype != U16
        or owner.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"moe_post owner must be ROW_MAJOR uint16 [1, {EXPERTS}], got {list(owner.shape)}")
    if (
        fp.rows_of(shared) != rows
        or fp.tile_width_of(shared) != HIDDEN
        or shared.dtype != BF16
        or shared.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(
            f"moe_post shared partial must be TILE bf16 [1, 1, {rows}, {HIDDEN}], got {list(shared.shape)}"
        )
    if sigmoid is not None and (
        tuple(int(v) for v in sigmoid.shape) != (1, 1, fp.TILE, fp.TILE)
        or sigmoid.dtype != BF16
        or sigmoid.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"moe_post sigmoid must be one TILE bf16 [1, 1, 32, 32], got {list(sigmoid.shape)}")
    return rows


def route_contiguous(scores, indices) -> int:
    """1 when both routing tensors are one L1 shard on one core (moe_compute's drain-core shard: the rows lie at the
    shard's page pitch and the reader takes each tensor in one read), else 0 (one page read per row)."""

    for tensor in (scores, indices):
        config = tensor.memory_config()
        if (
            not config.is_sharded()
            or config.buffer_type != ttnn.BufferType.L1
            or config.shard_spec.grid.num_cores() != 1
        ):
            return 0
    return 1


def moe_post_program(pages, scores, indices, owner, shared, sigmoid, out, *, rows: int) -> "ttnn.ProgramDescriptor":
    mesh = pages.device()
    work = fp.split_work(HIDDEN_TILES, mesh)
    cores = fp.core_rectangle(work)
    stage_pages = rows * TOP_K + 1
    named = [(name, index) for name, index, _dtype, _bytes, _pages in CBS] + [
        ("top_k", TOP_K),
        ("has_sig", 0 if sigmoid is None else 1),
        ("stage_pages", stage_pages),
        ("owner_bytes", OWNER_BYTES),
        ("route_contiguous", route_contiguous(scores, indices)),
    ]
    cbs = [
        fp.cb_descriptor(index, dtype, page_bytes, stage_pages if pages_ is None else pages_, cores)
        for _name, index, dtype, page_bytes, pages_ in CBS
    ]
    sig = shared if sigmoid is None else sigmoid  # the absent sigmoid's accessor args still have to be laid out
    reader = fp.reader_kernel(
        KERNELS["reader"],
        cores,
        [
            *fp.accessor_args(pages),
            *fp.accessor_args(scores),
            *fp.accessor_args(indices),
            *fp.accessor_args(owner),
            *fp.accessor_args(shared),
            *fp.accessor_args(sig),
        ],
        [
            (
                w.core,
                [
                    pages.buffer_address(),
                    scores.buffer_address(),
                    indices.buffer_address(),
                    owner.buffer_address(),
                    shared.buffer_address(),
                    sig.buffer_address(),
                    w.start,
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
        [(w.core, [out.buffer_address(), w.start]) for w in work],
        named=named,
    )
    compute = fp.compute_kernel(
        KERNELS["compute"], cores, [], named=named, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True
    )
    return fp.program_descriptor([reader, writer, compute], cbs=cbs)


def moe_post(pages, scores, indices, owner, shared, *, sigmoid=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The layer's local MoE sum ``[1, 1, rows, 2560]`` bf16 TILE: the score-weighted sum of the owned slots of
    ``pages`` (``moe_compute``'s ``[10, rows, 2560]`` ROW_MAJOR local combine buffer; unowned slots are never read)
    plus ``shared`` (TILE ``[1, 1, rows, 2560]``: the gated shared partial, or the ungated down-linear shard when
    ``sigmoid`` -- its column-broadcast sigmoid tile -- is given).  ``scores`` / ``indices`` are the ROW_MAJOR routing
    rows, ``owner`` this device's ``[1, 512]`` uint16 expert owner row."""

    rows = _check_inputs(pages, scores, indices, owner, shared, sigmoid)
    mesh = pages.device()
    out = fp.stamp_topology(fp.allocate((1, 1, rows, HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh, memory_config), pages)
    io = [pages, scores, indices, owner, shared] + ([] if sigmoid is None else [sigmoid]) + [out]
    fp.run_program(
        io,
        moe_post_program(pages, scores, indices, owner, shared, sigmoid, out, rows=rows),
        meta=moe_post_meta(pages, scores, indices, owner, shared, sigmoid, out, rows=rows),
    )
    return out


def moe_post_meta(pages, scores, indices, owner, shared, sigmoid, out, *, rows: int) -> "fp.FusedProgramMeta":
    """What the program moves and issues, by construction (the census's model): the owned fragments of the pages at
    their bound (every slot owned: the whole ``[10, rows, 2560]``), the routing rows and the owner row once per core
    (the routing rows from the drain core's L1 shard when they are sharded there), the shared partial and the sigmoid
    tile once (each core its column), the sum out; per element of the ``rows x 2560`` output the ten-slot MAC, the
    shared add and, with the sigmoid, its multiply."""

    cores = HIDDEN_TILES
    routing = fp.tensor_bytes(scores) + fp.tensor_bytes(indices)
    per_core = cores * (routing + fp.tensor_bytes(owner))
    l1 = per_core if fp.in_l1(scores) else 0
    reads = (pages, shared) + (() if sigmoid is None else (sigmoid,))
    return fp.program_meta(
        NAME,
        "post" if sigmoid is None else "post_sigmoid",
        rows,
        reads=reads,
        writes=(out,),
        dram_bytes=0 if l1 else per_core,
        l1_bytes=l1,
        flops=rows * HIDDEN * (2 * TOP_K + 1 + (0 if sigmoid is None else 1)),
        cores=cores,
    )


def _compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def moe_post_composed(pages, scores, indices, owner, shared, *, sigmoid=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The composed chain on one chip, op for op as the model issues it: tilize of the pages,
    ``deepseek_moe_fast_reduce_nc_fused`` with this device's ownership (the model's reader keeps the scores of the
    experts whose owner is on its cluster axis; on one chip ``cluster_axis=1`` keeps expert ``e`` when the mapping
    row says owner 0, so the mapping is ``1 - owner``), the shared expert's ``x sigmoid`` multiply and the add.  The
    chain reads every slot: the caller hands in pages whose unowned slots are +0.0 (the fill's state)."""

    rows = _check_inputs(pages, scores, indices, owner, shared, sigmoid)
    mesh = pages.device()
    if mesh.get_num_devices() != 1:
        raise ValueError("moe_post_composed is the one-chip replica of the chain; the model runs its inline chain")
    mapping = _chain_mapping(owner)
    local_stack = ttnn.to_layout(
        ttnn.unsqueeze(pages, dim=1), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, pad_value=0.0
    )
    indices4 = ttnn.reshape(indices, (1, 1, rows, TOP_K))
    scores4 = ttnn.reshape(scores, (rows, 1, 1, TOP_K))
    routed = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        local_stack,
        indices4,
        mapping,
        reduce_dim=0,
        split_size=HIDDEN,
        cluster_axis=1,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        scores_tensor=scores4,
        num_shared_experts=0,
        shared_expert_scale=1.0,
        compute_kernel_config=_compute_config(mesh),
    )[0]
    ttnn.deallocate(local_stack)
    ttnn.deallocate(mapping)
    rhs = shared
    if sigmoid is not None:
        gate = ttnn.slice(sigmoid, (0, 0, 0, 0), (1, 1, rows, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rhs = ttnn.mul(shared, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
    out = ttnn.add(routed, rhs, memory_config=memory_config)
    ttnn.deallocate(routed)
    if rhs is not shared:
        ttnn.deallocate(rhs)
    return out


_ONES_ROWS: dict[int, object] = {}


def _chain_mapping(owner):
    """``1 - owner`` as the chain's ``[1, 512]`` uint16 mapping row, computed on device (a ones row per mesh, built
    once; no host read, so the replica runs inside a trace capture too)."""

    key = id(owner.device())
    if key not in _ONES_ROWS:
        _ONES_ROWS[key] = ttnn.from_torch(
            torch.ones(1, EXPERTS, dtype=torch.int16),
            dtype=U16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=owner.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return ttnn.subtract(_ONES_ROWS[key], owner, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# public names for the other MoE kernels' one-chip chain replicas (moe_combine)
chain_mapping = _chain_mapping
chain_compute_config = _compute_config


def local_sum_rows(result) -> torch.Tensor:
    """``[rows, 2560]`` fp32 (the bf16 bits widened): the comparable form of the local sum."""

    return ttnn.to_torch(result).float().reshape(-1, HIDDEN)


def _gate_inputs(mesh, capture, positions, layer):
    """The capture's ``moe_pages`` (the ten slots' expert outputs, every slot filled by its owner), routing and one
    device's gated shared partial as the chain's inputs; the device is ``QWEN38_FUSED_MOE_POST_DEVICE`` (default 0),
    and the pages of the slots it does not own are zeroed as the fill leaves them."""

    device = int(os.environ.get(DEVICE_ENV, "0"))
    index = [capture["positions"].index(p) for p in positions]
    rows = len(positions)
    owner = owner_rows()[device]
    indices = capture["router_indices"][index, layer].to(torch.int64).reshape(rows, TOP_K)
    scores = capture["router_scores"][index, layer].to(torch.bfloat16).reshape(rows, TOP_K)
    pages = (
        capture["moe_pages"][index, layer]
        .to(torch.bfloat16)
        .reshape(rows, TOP_K, -1)[..., :HIDDEN]
        .permute(1, 0, 2)
        .clone()
    )
    owned = owner[indices].bool().t()  # [top_k, rows]
    pages[~owned] = 0.0
    shared = capture["shared_partial"][index, layer, device].to(torch.bfloat16).reshape(1, 1, rows, HIDDEN)
    dram = ttnn.DRAM_MEMORY_CONFIG
    return {
        "pages": ttnn.from_torch(
            pages.contiguous(), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram
        ),
        "scores": ttnn.from_torch(
            scores.reshape(1, 1, rows, TOP_K), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram
        ),
        "indices": ttnn.from_torch(
            indices.to(torch.int16).reshape(1, 1, rows, TOP_K),
            dtype=U16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=dram,
        ),
        "owner": ttnn.from_torch(
            owner.reshape(1, EXPERTS), dtype=U16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram
        ),
        "shared": ttnn.from_torch(shared, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram),
    }


register(
    FusedKernel(
        name=NAME,
        replaces="fill, unsqueeze + tilize, deepseek_moe_fast_reduce_nc_fused, the partial add and the shared x sigmoid multiply (7 programs per layer)",
        tolerance=BITWISE,
        fused=moe_post,
        composed=moe_post_composed,
        gate=GateSpec(inputs=_gate_inputs, output=local_sum_rows, reference=None, layers=tuple(range(48))),
    )
)
