# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_step``: one GDN decode step from the fused projection to the gated output, one program per layer.

Covers the conv ring slot write, the z/a/b slices, conv + SiLU, the head split, the decay and write gates, both l2
norms, the fp32 delta-rule state update, the read-out, the gated RMSNorm and the sigmoid gate. One core per (lane,
value head) item; the state stays fp32 and is updated in place. The 1-row body runs it at rows = 1 and the
batched-lanes body (``_forward_decode_lanes_fused`` in ttnn/gdn.py) at rows = B, lane u = state slot u and ring row u.
Rounding follows tt/gdn.py (the oracle), not the chain, so the tolerance class is COMPONENT; ``reference_step`` is the
kernel's arithmetic in torch.
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
CB_SNEW, CB_OUTS = 24, 25
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


def run(projected, older, newest, taps, constants, norm, recurrent, out):
    """The program on explicit tensors (local shapes, one tile row): ``projected`` [1,1,rows,4160] bf16;
    ``older`` the three older ring slots and ``newest`` the slot this token lands in, [1,1,rows,2560] bf16;
    ``taps`` 4 x [1,1,1,2560] bf16; ``constants`` from ``constant_tiles``; ``norm`` [1,1,1,128] bf16;
    ``recurrent`` [rows,12,128,128] fp32 (updated in place); ``out`` [1,1,rows,1536] bf16 (any memory layout the
    TensorAccessor addresses)."""

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
    fp.run_program([*io, out], fp.program_descriptor([reader, writer, compute], cbs))
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
    """The fused step's input contract for one call, as ``run`` asserts it: ``projected`` one row tile of the
    projection width, four ring slots, and the fp32 recurrent state with one lane per row. Host fakes whose padded
    shape is their logical shape fail it and take the composed chain (``registry.resolve_admitted``)."""

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
    """The chain from the projection to the gated output (ttnn/gdn.py)."""

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
        component_proof="the fused step's state and gated output against the CPU oracle on real inputs",
        # the audit capture holds no GDN recurrent state; checked by the real-input step probe against the oracle
        gate=None,
    )
)
