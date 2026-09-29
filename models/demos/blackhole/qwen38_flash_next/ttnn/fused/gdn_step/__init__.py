# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_step``: one GDN decode step from the fused projection to the gated output, one program per layer.

Replaces programs 4-52 of the GDN block (the slice into the conv ring slot, the z/a/b slices, conv + SiLU, the head
split, the decay and write gates, both l2 norms, the fp32 delta-rule state update, the read-out, the gated RMSNorm
and the sigmoid gate: 49 programs, 119 us kernel / 213 us occupancy per layer in the 2026-09-12 census).  One core
per (lane, value head) item; the state stays fp32 and is updated in place; the newest ring slot is written from the
projection.  The 1-row body runs it at rows = 1 and the batched-lanes body at rows = B (``_forward_decode_lanes_fused``
in ttnn/gdn.py: lane u = state slot u and ring row u, so B independent users are 12 B items of one program).
Rounding follows tt/gdn.py (the oracle), not today's chain, so the tolerance class is COMPONENT (note
work/FUSE-GDN-STEP-20260913.md section 1).  ``reference_step`` is the kernel's arithmetic in torch for the tests.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import ttnn

from .. import program as fp
from ..registry import COMPONENT, FusedKernel, register

NAME = "gdn_step"
READER = fp.kernel_source(NAME, "reader.cpp")
WRITER = fp.kernel_source(NAME, "writer.cpp")
COMPUTE = fp.kernel_source(NAME, "compute.cpp")

HEADS = 12
QK_HEADS = 4
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = HEADS * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
A_COLUMN = QKV_WIDTH + VALUE_WIDTH
B_COLUMN = A_COLUMN + fp.TILE
PROJECTION_WIDTH = B_COLUMN + fp.TILE
STATE_TILES = 16
# debug taps in program order (one fp32 tile each); the writer stores them per item as [rows*12, DEBUG_TILES, 32, 32]
DEBUG_TAPS = (
    "conv_sum0",
    *(f"conv{t}" for t in range(12)),
    "q_sum",
    "q_sum_eps",
    "q_rsqrt",
    "q_unit0",
    "beta",
    "decay",
    "sdec0",
    "vread0",
    "deltab0",
    "kcol0",
    "snew0",
    "o0",
    "o1",
    "o2",
    "o3",
)
DEBUG_TILES = len(DEBUG_TAPS)
EPS = 1.0e-6

BF16, FP32 = ttnn.bfloat16, ttnn.float32
# (index, dtype, pages); indices must match the three kernels
CBS = (
    (0, BF16, 12),
    (1, BF16, 36),
    (2, BF16, 48),
    (3, BF16, 4),
    (4, BF16, 2),
    (5, FP32, 2),
    (6, BF16, 4),
    (7, FP32, 16),
    (8, BF16, 1),
    (9, FP32, 2),
    (10, BF16, 4),
    (11, BF16, 12),
    (12, BF16, 4),
    (13, BF16, 2),
    (14, BF16, 4),
    (15, FP32, 4),
    (16, FP32, 4),
    (17, FP32, 4),
    (18, FP32, 1),
    (19, FP32, 1),
    (20, FP32, 16),
    (21, FP32, 16),
    (22, FP32, 4),
    (23, FP32, 4),
    (26, BF16, 4),
    (27, FP32, 1),
)
CB_SNEW, CB_OUTS, CB_DEBUG = 24, 25, 28
# fp32 CBs consumed only by copy_tile (exact unpack to DST); matmul, reduce and broadcast operands stay Default
FP32_COPY_CBS = (5, 7, 18, 19, 21, 22)


def constant_tiles(mesh, dt_bias, neg_exp_A):
    """Per device ``[1, 24, 32, 32]`` fp32: tiles 0..11 = dt_bias[h], 12..23 = neg_exp_A[h], each filling a tile.

    Full tiles keep the fp32 constants exact on the SFPU path (a broadcast through the source registers would
    truncate them to 19 bits)."""

    def per_device(tensor):
        return [ttnn.to_torch(local).float().reshape(-1)[:HEADS] for local in ttnn.get_device_tensors(tensor)]

    host = torch.stack(
        [
            torch.cat(
                [
                    dt.view(HEADS, 1, 1).expand(HEADS, fp.TILE, fp.TILE),
                    na.view(HEADS, 1, 1).expand(HEADS, fp.TILE, fp.TILE),
                ]
            )
            for dt, na in zip(per_device(dt_bias), per_device(neg_exp_A))
        ]
    ).contiguous()
    return ttnn.from_torch(
        host,
        dtype=FP32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 0)),
    )


def run(projected, older, newest, taps, constants, norm, recurrent, out, *, debug=None):
    """The program on explicit tensors (local shapes, one tile row): ``projected`` [1,1,rows,4160] bf16;
    ``older`` the three older ring slots and ``newest`` the slot this token lands in, [1,1,rows,2560] bf16;
    ``taps`` 4 x [1,1,1,2560] bf16; ``constants`` from ``constant_tiles``; ``norm`` [1,1,1,128] bf16;
    ``recurrent`` [rows,12,128,128] fp32 (updated in place); ``out`` [1,1,rows,1536] bf16 (any memory layout the
    TensorAccessor addresses); ``debug`` an optional [rows*12*18, 32, 32] fp32 tap tensor."""

    rows = fp.rows_of(projected)
    if fp.tile_width_of(projected) != PROJECTION_WIDTH:
        raise ValueError(f"gdn_step projection width must be {PROJECTION_WIDTH}, got {projected.shape[-1]}")
    if tuple(recurrent.shape) != (rows, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(
            f"gdn_step state must be [{rows}, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape} {recurrent.dtype}"
        )
    if len(older) != 3 or len(taps) != 4:
        raise ValueError("gdn_step takes three older ring slots and four taps")
    mesh = projected.device()
    items = [(lane, head) for lane in range(rows) for head in range(HEADS)]
    work = fp.split_work(len(items), mesh)
    cores = fp.core_set(work)

    def pairs(w):
        return [value for item in items[w.start : w.start + w.count] for value in item]

    reader_cta = [rows]
    for tensor in (projected, *older, *taps, constants, norm, recurrent, newest):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        *(t.buffer_address() for t in older),
        *(t.buffer_address() for t in taps),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
        newest.buffer_address(),
    ]
    writer_cta = [rows, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]
    cbs = [fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores) for index, dtype, pages in CBS]
    # the new state is read back by the compute (CB_SNEW) and drained by the writer (CB_OUTS): one allocation
    cbs.append(
        ttnn.CBDescriptor(
            total_size=STATE_TILES * fp.TILE_BYTES[FP32],
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    if debug is not None:
        writer_cta.extend(fp.accessor_args(debug))
        writer_addrs.append(debug.buffer_address())
        defines.append(("DEBUG_TAPS", "1"))
        cbs.append(fp.cb_descriptor(CB_DEBUG, FP32, fp.TILE_BYTES[FP32], DEBUG_TILES, cores))

    reader = fp.reader_kernel(READER, cores, reader_cta, [(w.core, [*reader_addrs, w.count, *pairs(w)]) for w in work])
    writer = fp.writer_kernel(
        WRITER, cores, writer_cta, [(w.core, [*writer_addrs, w.count, *pairs(w)]) for w in work], defines=defines
    )
    compute = fp.compute_kernel(
        COMPUTE,
        cores,
        [],
        [(w.core, [w.count]) for w in work],
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=FP32_COPY_CBS,
    )
    io = [projected, *older, *taps, constants, norm, recurrent, newest]
    if debug is not None:
        io.append(debug)
    # the projection, the ring slots and the taps once (each core its head's columns), the fp32 state read and
    # written in place, the newest slot and the output written, the constant tiles once per core; per lane: the
    # 4-tap conv, the two l2 norms, the fp32 delta-rule update (outer products, decay, delta) and read-out on 12
    # heads of 128 x 128, the gated norm
    meta = fp.program_meta(
        NAME,
        "step",
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out, *([debug] if debug is not None else [])),
        dram_bytes=len(work) * 2 * fp.TILE_BYTES[FP32],
        flops=rows * (2 * 4 * QKV_WIDTH + 6 * 2 * QK_WIDTH + 8 * HEADS * HEAD_DIM * HEAD_DIM + 6 * VALUE_WIDTH),
        cores=len(work),
    )
    fp.run_program([*io, out], fp.program_descriptor([reader, writer, compute], cbs), meta=meta)
    return out


def _constants(gdn):
    tiles = getattr(gdn, "_fused_gdn_step_constants", None)
    if tiles is None:
        tiles = constant_tiles(gdn.mesh_device, gdn.weights.dt_bias, gdn.weights.neg_exp_A)
        gdn._fused_gdn_step_constants = tiles
    return tiles


def gdn_step(gdn, projected, window, state):
    """The fused step at the chain site: ``projected`` from ``_project`` (one row) or ``_project_lanes_unsplit`` (B
    lane rows), ``window`` the conv ring (oldest first, the last slot receives this token), ``state`` the layer's GDN
    state at the same rows; returns the gated output in the out-projection's activation layout and updates
    ``state.recurrent`` and the newest slot in place."""

    rows = fp.rows_of(projected)
    out = fp.allocate(
        (1, 1, rows, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device, gdn.out_proj_act_memory_config
    )
    run(
        projected, window[:3], window[3], gdn.weights.conv_taps, _constants(gdn), gdn.weights.norm, state.recurrent, out
    )
    ttnn.deallocate(projected)
    return out


def admits(gdn, projected, window, state) -> bool:
    """The fused step's input contract for one call, as ``run`` asserts it before building the program: ``projected``
    one row tile of the projection width, the ring window's four slots, the state's fp32 recurrent tensor with one
    lane per row.  The served decode tensors satisfy it; host fakes whose padded shape is their logical shape do not
    and take the composed chain (``registry.resolve_admitted``)."""

    if not (fp.is_row_tile(projected) and fp.is_tile_width(projected, PROJECTION_WIDTH)):
        return False
    try:
        slots = len(window)
    except TypeError:
        return False
    recurrent = getattr(state, "recurrent", None)
    return (
        slots == 4
        and recurrent is not None
        and tuple(recurrent.shape) == (projected.shape[-2], HEADS, HEAD_DIM, HEAD_DIM)
        and recurrent.dtype == FP32
    )


def gdn_step_composed(gdn, projected, window, state):
    """Today's chain from the projection to the gated output (ttnn/gdn.py)."""

    z, a, b = gdn._split_projection(projected, window[3])
    conv = gdn._causal_conv_decode(window)
    q, k, v, beta, log_decay, producers = gdn._make_recurrent_inputs(conv, a, b)
    return gdn._gate(gdn._recurrent_decode(q, k, v, beta, log_decay, producers, state), z)


def reference_step(projected, older, taps, dt_bias, neg_exp_A, norm, state):
    """tt/gdn.py's arithmetic per lane on host tensors: ``projected`` [rows, 4160] bf16, ``older`` 3 x [rows, 2560]
    bf16 (oldest first), ``taps`` 4 x [2560] bf16, ``dt_bias`` / ``neg_exp_A`` [12] fp32, ``norm`` [128] bf16,
    ``state`` [rows, 12, 128, 128] fp32.  Returns (new_state fp32, gated [rows, 1536] bf16, conv [rows, 2560] bf16,
    o [rows, 12, 128] bf16, beta [rows, 12] bf16, decay [rows, 12] fp32)."""

    from models.demos.blackhole.qwen38_flash_next.reference import gated_delta_recurrent

    rows = projected.shape[0]
    window = [*older, projected[:, :QKV_WIDTH]]
    conv = sum(w.float() * t.float() for w, t in zip(window, taps)).to(torch.bfloat16)
    conv = F.silu(conv)
    q = conv[:, :QK_WIDTH].reshape(rows, 1, QK_HEADS, HEAD_DIM).repeat_interleave(HEADS // QK_HEADS, dim=2)
    k = (
        conv[:, QK_WIDTH : 2 * QK_WIDTH]
        .reshape(rows, 1, QK_HEADS, HEAD_DIM)
        .repeat_interleave(HEADS // QK_HEADS, dim=2)
    )
    v = conv[:, 2 * QK_WIDTH :].reshape(rows, 1, HEADS, HEAD_DIM)
    z = projected[:, QKV_WIDTH:A_COLUMN].reshape(rows, 1, HEADS, HEAD_DIM)
    a = projected[:, A_COLUMN : A_COLUMN + HEADS].reshape(rows, 1, HEADS)
    b = projected[:, B_COLUMN : B_COLUMN + HEADS].reshape(rows, 1, HEADS)
    beta = b.sigmoid()
    log_decay = neg_exp_A.float() * F.softplus(a.float() + dt_bias.float())
    output, new_state = gated_delta_recurrent(q, k, v, log_decay, beta, state, l2_normalize_qk=True)
    normalized = output.float() * torch.rsqrt(output.float().square().mean(dim=-1, keepdim=True) + EPS)
    normalized = norm * normalized.to(output.dtype)
    gated = (normalized * torch.sigmoid(z.float())).to(output.dtype)
    return (
        new_state,
        gated.reshape(rows, VALUE_WIDTH),
        conv,
        output.reshape(rows, HEADS, HEAD_DIM),
        beta[:, 0],
        log_decay[:, 0].exp(),
    )


register(
    FusedKernel(
        name=NAME,
        replaces="GDN decode programs 4-52: ring-slot slice, z/a/b slices, conv + SiLU, head split, gates, l2 norms, "
        "fp32 delta-rule update, read-out, gated RMSNorm, sigmoid gate (49 programs per layer); the same programs of "
        "the batched-lanes body on B rows (one item per (lane, value head))",
        tolerance=COMPONENT,
        fused=gdn_step,
        composed=gdn_step_composed,
        admits=admits,
        component_proof="the fused GDN step probe against the CPU oracle, four p150, 2026-09-25: state error 0.0048 and "
        "gated output 0.0098 for the fused step, 0.0081 and 0.0234 for the composed chain",
        # The audit capture holds no GDN recurrent state; the component gate is the real-input probe of the dev tools
        # (fused_gdn_step_probe: layer-0 inputs, the oracle state, 64 consecutive steps, per-component taps).
        gate=None,
    )
)
