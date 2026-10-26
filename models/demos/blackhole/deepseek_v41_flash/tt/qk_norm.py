# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused attention-front norm (lever L06): slice + q RMSNorm [1280] + kv RMSNorm [512] of the wqkv output in ONE program (4 ops -> 1),
bit for bit equal to ``ttnn.slice`` + ``ttnn.rms_norm`` (default compute config) of both parts.

y [1,1,T,1792] bf16 TILE (= [wq_a | wkv] output) -> (qr [1,1,T,1280], kvn [1,1,T,512]) bf16 TILE DRAM.  The q norm runs on GQ cores, the kv norm on GK cores (one
group each).  The cores run the LLK sequence of the reference op (square, reduce_tile over all tiles in order, * 1/N, + eps, rsqrt, column-broadcast
multiply, gamma multiply; bf16 intermediates, fp32_dest_acc off, approx mode on).  The accumulation order of the row sum is the reference one
(a split of the sum over several cores changes the roundings: measured 7-35% equal elements): the cores square their tiles, the other cores of a
group write their square tiles to the reducer (core 0 of the group), which reduces all tiles in order and sends rs back.
Env DSV41_QKN_GQ / DSV41_QKN_GK: cores of the q / kv group (divisors of 40 / 16).
gamma: bf16 TILE [1,1,32,1792] = [q_norm | kv_norm] rows replicated (``make_gamma``).  See tt/attn_kernels/qkn_*.cpp."""

import os
import struct

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_fused import _kernel
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash

Q_LORA, HEAD_DIM = 1280, 512
GX = 4  # worker columns 0..3 are contiguous in physical NOC coordinates


def make_gamma(q_norm, kv_norm):
    """torch [1280], [512] -> torch [1,1,32,1792] (rows replicated; upload as bf16)."""
    g = torch.cat([q_norm.float(), kv_norm.float()])
    return g.reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()


def qk_norm(y, gamma, T, eps=1e-20):
    assert y.dtype == ttnn.bfloat16 and gamma.dtype == ttnn.bfloat16 and T <= 16
    GQ = int(os.environ.get("DSV41_QKN_GQ", "8"))
    GK = int(os.environ.get("DSV41_QKN_GK", "4"))
    assert 40 % GQ == 0 and 16 % GK == 0
    mesh = y.device()
    qr = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, Q_LORA]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    kvn = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, HEAD_DIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    NC = GQ + GK
    gy = -(-NC // GX)
    cores = [(k % GX, k // GX) for k in range(NC)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(GX - 1, gy - 1))
    assert p1.x - p0.x == GX - 1 and p1.y - p0.y == gy - 1
    qpc, kpc = (Q_LORA // 32) // GQ, (HEAD_DIM // 32) // GK
    gmax = max(qpc, kpc)
    wall = Q_LORA // 32
    fb = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
    rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        if k < GQ:
            rt_r[cx][cy] = [int(k == 0), 0, GQ, k, k * qpc, qpc, fb(eps)]
            rt_w[cx][cy] = [0, k * qpc, qpc]
            rt_c[cx][cy] = [qpc, fb(1.0 / Q_LORA), int(k == 0), Q_LORA // 32]
        else:
            kl = k - GQ
            rt_r[cx][cy] = [int(kl == 0), GQ, GK, kl, Q_LORA // 32 + kl * kpc, kpc, fb(eps)]
            rt_w[cx][cy] = [1, kl * kpc, kpc]
            rt_c[cx][cy] = [kpc, fb(1.0 / HEAD_DIM), int(kl == 0), HEAD_DIM // 32]
    bf = ttnn.bfloat16
    X, W, SC, EPS, SQ, SQALL, EX, RS, XR, O = range(10)
    cbs = [
        _cb(core_set, X, gmax, 2048, bf),
        _cb(core_set, W, gmax, 2048, bf),
        _cb(core_set, SC, 1, 2048, bf),
        _cb(core_set, EPS, 1, 2048, bf),
        _cb(core_set, SQ, gmax, 2048, bf),
        _cb(core_set, SQALL, wall, 2048, bf),
        _cb(core_set, EX, 1, 2048, bf),
        _cb(core_set, RS, 1, 2048, bf),
        _cb(core_set, XR, gmax, 2048, bf),
        _cb(core_set, O, gmax, 2048, bf),
    ]
    reader = _kernel(
        "qkn_reader.cpp",
        core_set,
        [X, W, SC, EPS, SQ, SQALL, RS, p0.x, p0.y, GX] + _acc(y) + _acc(gamma),
        rt_r,
        ttnn.ReaderConfigDescriptor(),
        [y.buffer_address(), gamma.buffer_address()],
    )
    writer = _kernel(
        "qkn_writer.cpp",
        core_set,
        [O] + _acc(qr) + _acc(kvn),
        rt_w,
        ttnn.WriterConfigDescriptor(),
        [qr.buffer_address(), kvn.buffer_address()],
    )
    compute = _kernel(
        "qkn_compute.cpp",
        core_set,
        [X, W, SC, EPS, SQ, SQALL, EX, RS, XR, O],
        rt_c,
        ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=True
        ),
        [],
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(2)],
        cbs=cbs,
    )
    prog.custom_program_hash = _hash(
        0xA72, T, GQ, GK, tuple(_acc(y)), tuple(_acc(gamma)), tuple(_acc(qr)), tuple(_acc(kvn))
    )
    ttnn.generic_op([y, gamma, qr, kvn], prog)
    return qr, kvn
