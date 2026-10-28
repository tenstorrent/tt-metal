# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Route K (CONTRACT §6 B0): the diagonal complex LTI convolution x_t = Lbar x_{t-1} + Bu_t as one generic_op.

Bu and x are (L, B, P) fp32 TILE, real and imaginary parts separate. A unit is one 32 x 32 tile of (B, P); units
are independent and split contiguously over the cores. Each core walks t = 0..L-1 and, per unit, reads 2 Bu tiles,
does one fp32 SFPU pass (x' = Lbar x + Bu) with the state carried through an L1 FIFO, and writes 2 x tiles.
Lbar is the reference's complex64 Lbar (= kernel[:, -2]); repeated multiplication is the recurrence itself.
"""

from __future__ import annotations

import os

import ttnn

TILE = 32
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
IN_CB, LBAR_CB, STATE_CB, OUT_CB = 0, 1, 2, 16


def _cb(index, pages, cores):
    return ttnn.CBDescriptor(
        total_size=pages * 4096,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=4096)],
    )


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def lbar_tiles(device, lbar):
    """lbar (P,) complex64 -> (re, im) each (32, P) fp32 TILE, every row = Lbar."""
    rows = lbar.unsqueeze(0).expand(TILE, -1).contiguous()
    return [
        ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        for t in (rows.real, rows.imag)
    ]


_PLANS: dict = {}


def _plan(device, units):
    grid = device.compute_with_storage_grid_size()
    _, all_cores, g1, _, n1, n2 = ttnn.split_work_to_cores(grid, units, True)
    work, start = [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(all_cores, row_wise=True)):
        cnt = n1 if i < g1.num_cores() else n2
        work.append((core, cnt, start))
        start += cnt
    cbs = [
        _cb(IN_CB, 4, all_cores),
        _cb(LBAR_CB, 2 * n1, all_cores),
        _cb(STATE_CB, 2 * n1 + 2, all_cores),
        _cb(OUT_CB, 4, all_cores),
    ]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (IN_CB, LBAR_CB, STATE_CB):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    return work, all_cores, cbs, cfg


def lti_scan(bu_re, bu_im, l_tiles):
    """Split layout: bu (L, B, P) fp32 TILE (B padded to 32 by the layout) -> x (re, im) (L, B, P) fp32 TILE."""
    L, B, P = tuple(bu_re.shape)
    if bu_re.dtype != ttnn.float32 or P % TILE:
        raise ValueError(f"lti_scan needs fp32 (L, B, 32k) Bu, got {bu_re.shape} {bu_re.dtype}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    x_re, x_im = (
        ttnn.allocate_tensor_on_device(ttnn.Shape([L, B, P]), ttnn.float32, ttnn.TILE_LAYOUT, bu_re.device(), dram)
        for _ in range(2)
    )
    _launch(bu_re, bu_im, x_re, x_im, l_tiles, L, -(-B // TILE), P // TILE, P // TILE, 0)
    return x_re, x_im


def lti_scan_fused(bu, l_tiles, L):
    """Fused layout: bu (1, 1, L * Bp, 2P) fp32 TILE, rows (t, b), columns [Bu_re | Bu_im] as the B projection
    writes it, Bp a multiple of 32 -> x (1, 1, L * Bp, 2P) fp32 TILE, [x_re | x_im] as the C projection reads it."""
    M, P2 = tuple(bu.shape)[-2:]
    if bu.dtype != ttnn.float32 or M % (L * TILE) or P2 % (2 * TILE):
        raise ValueError(f"lti_scan_fused needs fp32 (L * 32k, 64k) Bu, got {bu.shape} {bu.dtype}")
    Pt = P2 // (2 * TILE)
    x = ttnn.allocate_tensor_on_device(bu.shape, ttnn.float32, ttnn.TILE_LAYOUT, bu.device(), ttnn.DRAM_MEMORY_CONFIG)
    _launch(bu, bu, x, x, l_tiles, L, M // L // TILE, Pt, 2 * Pt, Pt)
    return x


def _launch(bu_re, bu_im, x_re, x_im, l_tiles, L, Bt, Pt, row_tiles, im_off):
    device = bu_re.device()
    key = (Bt * Pt, device.id())
    if key not in _PLANS:
        _PLANS[key] = _plan(device, Bt * Pt)
    work, all_cores, cbs, cfg = _PLANS[key]
    l_re, l_im = l_tiles
    reader = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "reader_lti_scan.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=[*_accessor(bu_re), *_accessor(l_re)],
        runtime_args=[
            (
                (c.x, c.y),
                [
                    bu_re.buffer_address(),
                    bu_im.buffer_address(),
                    l_re.buffer_address(),
                    l_im.buffer_address(),
                    cnt,
                    s,
                    L,
                    Bt,
                    Pt,
                    row_tiles,
                    im_off,
                ],
            )
            for c, cnt, s in work
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "writer_lti_scan.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        compile_time_args=_accessor(x_re),
        runtime_args=[
            ((c.x, c.y), [x_re.buffer_address(), x_im.buffer_address(), cnt, s, L, Bt, Pt, row_tiles, im_off])
            for c, cnt, s in work
        ],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=os.path.join(_KERNELS, "compute_lti_scan.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores,
        runtime_args=[((c.x, c.y), [cnt, L]) for c, cnt, s in work],
        config=cfg,
    )
    io = [bu_re] + ([bu_im] if bu_im is not bu_re else []) + [l_re, l_im, x_re] + ([x_im] if x_im is not x_re else [])
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs))
