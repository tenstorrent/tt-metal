# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused sLSTM recurrence: every step of ``tt/layers.py: slstm_recurrence`` in one ``ttnn.generic_op``.

R is block-diagonal by head (columns gate, head, d), so head h's gates only read head h's y. A unit (one
32-row tile row x one head) is independent of every other unit. The work item is one d-tile of a unit (4 gate
columns x 32 rows); items are split contiguously over the cores so no core runs much more than one unit's
worth. A unit whose 4 d-tiles land on several cores shares its bf16 y: after each step every owner writes its
y tile into the same L1 slot of the other touching cores (fixed y buffer, double-buffered by step parity) and
bumps their per-unit semaphore; a core's reader waits for 4 tiles per unit per step before releasing the next
step's matmul. Per step and item: 4 Wx tiles in, pre = Wx + y_bf16 @ R_h (HiFi4, fp32 DST), the gate math in
DST, 1 fp32 y tile out.
"""

from __future__ import annotations

import os
import struct

import ttnn

TILE = 32
HEADS = 4
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

WX_CB, R_CB, Y_CB, C_CB, N_CB, M_CB, READY_CB = 0, 1, 2, 3, 4, 5, 7
YOUT_CB, COUT_CB, NOUT_CB, MOUT_CB = 16, 17, 18, 19


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _cb(index, pages, dtype, cores):
    page = 4096 if dtype == ttnn.float32 else 2048
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def slstm_steps(
    WxT: ttnn.Tensor, r: ttnn.Tensor, steps: int | None = None, record: bool = False, ablate: int = 0, gate_ablate: int = 0,
    compute_ablate: int = 0,
):
    """Run the sLSTM loop over ``WxT [T, 1, B, 2048]`` fp32 with recurrent weight ``r [512, 2048]`` bf16.

    Returns ``(ys, c, n, m)``: ``ys [T, 1, B, 512]`` fp32 and the final states ``[1, 1, B, 512]`` fp32, or with
    ``record`` the states after every step ``[T, 1, B, 512]`` (tests).
    """
    T, _, B, G = tuple(WxT.shape)
    T = T if steps is None else steps
    if G != 2048 or B % TILE or WxT.dtype != ttnn.float32 or r.dtype != ttnn.bfloat16:
        raise ValueError(f"slstm_steps needs WxT [T, 1, 32k, 2048] fp32 and bf16 r, got {WxT.shape} {WxT.dtype}")
    Bt = B // TILE
    device = WxT.device()
    dram = ttnn.DRAM_MEMORY_CONFIG
    ys = ttnn.allocate_tensor_on_device(ttnn.Shape([T, 1, B, 512]), ttnn.float32, ttnn.TILE_LAYOUT, device, dram)
    s_shape = ttnn.Shape([T if record else 1, 1, B, 512])
    cs, ns, ms = (
        ttnn.allocate_tensor_on_device(s_shape, ttnn.float32, ttnn.TILE_LAYOUT, device, dram) for _ in range(3)
    )

    key = (T, B, bool(record), int(ablate), int(gate_ablate), int(compute_ablate), device.id())
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _plan(device, T, Bt, record, ablate, gate_ablate, compute_ablate)
    work, writer_tail, all_cores, cbs, sems, compute, rt, L = plan
    reader = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "reader_slstm_step.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[*_accessor(WxT), *_accessor(r)],
        runtime_args=[((c.x, c.y), [WxT.buffer_address(), r.buffer_address(), cnt, s, *rt, L]) for c, cnt, s in work],
        config=ttnn.ReaderConfigDescriptor(),
    )
    addrs = [ys.buffer_address(), cs.buffer_address(), ns.buffer_address(), ms.buffer_address()]
    writer = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "writer_slstm_step.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[*_accessor(ys), *_accessor(cs)],
        runtime_args=[(xy, addrs + tail) for xy, tail in writer_tail],
        config=ttnn.WriterConfigDescriptor(),
    )
    ttnn.generic_op(
        [WxT, r, ys, cs, ns, ms], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs)
    )
    return ys, cs, ns, ms


# Address-free part of the program per (shape, knobs, device): work split, peer lists, CBs, semaphores and the
# compute kernel, built once instead of on every call (host cost per launch).
_PLANS: dict = {}


def _plan(device, T, Bt, record, ablate, gate_ablate, compute_ablate):
    grid = device.compute_with_storage_grid_size()
    num_items = Bt * HEADS * 4
    _, all_cores, g1, _, n1, n2 = ttnn.split_work_to_cores(grid, num_items, True)
    work, start = [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(all_cores, row_wise=True)):
        cnt = n1 if i < g1.num_cores() else n2
        work.append((core, cnt, start))
        start += cnt
    # Units each core touches (items first .. first + cnt - 1); local unit slot lu = unit - first // 4.
    touch = [(s // 4, (s + cnt - 1) // 4) for _, cnt, s in work]
    L = max(u1 - u0 + 1 for u0, u1 in touch)
    noc = [device.worker_core_from_logical_core(c) for c, _, _ in work]

    def dests(ci, item):
        u, out = item // 4, []
        for cj, (u0, u1) in enumerate(touch):
            if cj != ci and u0 <= u <= u1:
                out += [noc[cj].x, noc[cj].y, u - u0]
        if len(out) > 9:
            raise ValueError("a unit spans more than 4 cores")
        return [len(out) // 3, *out, *[0] * (9 - len(out))]

    f32, bf16 = ttnn.float32, ttnn.bfloat16
    cbs = [
        _cb(WX_CB, 8 * n1, f32, all_cores),
        _cb(R_CB, 16 * n1, bf16, all_cores),
        _cb(Y_CB, 8 * L, bf16, all_cores),
        _cb(C_CB, n1 + 1, f32, all_cores),
        _cb(N_CB, n1 + 1, f32, all_cores),
        _cb(M_CB, n1 + 1, f32, all_cores),
        _cb(READY_CB, 2, bf16, all_cores),
        _cb(YOUT_CB, 4, f32, all_cores),
        _cb(COUT_CB, 2, f32, all_cores),
        _cb(NOUT_CB, 2, f32, all_cores),
        _cb(MOUT_CB, 2, f32, all_cores),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=0) for i in range(L)]
    rt = [T, int(record), Bt, int(ablate)]
    writer_tail = []
    for ci, (c, cnt, s) in enumerate(work):
        args = [cnt, s, *rt]
        for e in range(cnt):
            args += dests(ci, s + e)
        writer_tail.append(((c.x, c.y), args))
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("SLSTM_FIDELITY", "HiFi4")), math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (WX_CB, C_CB, N_CB, M_CB):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "compute_slstm_step.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[_f32_bits(-80.0), _f32_bits(1.0), _f32_bits(20.0)],
        defines=[("INP_FLOAT32", "1"), ("SLSTM_GABL", str(int(gate_ablate))), ("SLSTM_CABL", str(int(compute_ablate))), ("SLSTM_FASTEXP", os.environ.get("SLSTM_FASTEXP", "2"))],
        runtime_args=[((c.x, c.y), [cnt, *rt, s, L]) for c, cnt, s in work],
        config=compute_cfg,
    )
    return work, writer_tail, all_cores, cbs, sems, compute, rt, L
