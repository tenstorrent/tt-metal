# SPDX-License-Identifier: Apache-2.0
"""Final fused-SwiGLU driver: transpose-eliminating + expert-skipping + trace-safe.

Reads only the n_active experts (ids from device tensor `idx` [1,1,1,NACT] uint32)
directly from the raw fused gate/up matmul output `raw` [1,E,1,2I] (bias pre-added),
computes clamp+silu+clamp+1+mul, and SCATTERS the result to the active expert slots
of `out` [1,E,1,I]. Replaces the whole reshape->transpose->slice->bias->swiglu chain
AND skips the 28 inactive experts.
"""
import os
import struct

import ttnn

_KDIR = "models/demos/gpt_oss/kernels"
_TILE = 32
_TILE_BYTES = 2 * 1024
_CACHE = {}


def _best_ncores(total, max_cores=64):
    best = 1
    for d in range(1, min(total, max_cores) + 1):
        if total % d == 0:
            best = d
    return best


def fused_swiglu_final(raw, bias, idx, out, nact, cap, limit, ncores=None):
    """raw,bias: [1,E,1,2I]; idx: [1,1,1,nact] uint32; out: [1,E,1,I]. Reads gate/up
    + gate/up bias for active experts, folds bias inside the kernel (no wide add)."""
    E = raw.shape[-3]
    twoI = raw.shape[-1]
    I = twoI // 2
    Ht = I // _TILE
    Wt2 = twoI // _TILE
    total_tiles = nact * Ht
    key = (id(raw.device()), E, I, nact)
    if key not in _CACHE:
        # 45 cores was the in-model optimum for the 4-expert x 90-tile decode SwiGLU
        # (swept 45/60: both ~57.5-57.8 tok/s, 45 marginally better + fewer cores).
        NC = ncores or int(os.environ.get("SWIGLU_NCORES", "0")) or 45
        while total_tiles % NC != 0:
            NC -= 1
        per_core = total_tiles // NC
        gx = min(8, NC)
        gy = (NC + gx - 1) // gx
        core_list = [(x, y) for y in range(gy) for x in range(gx)][:NC]
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in core_list])
        cb_gate, cb_up, cb_gb, cb_ub, cb_out, cb_idxr, cb_idxw = 0, 1, 2, 3, 16, 4, 5
        cap_u = struct.unpack("<I", struct.pack("<f", cap))[0]
        upmin_u = struct.unpack("<I", struct.pack("<f", -limit))[0]
        upmax_u = struct.unpack("<I", struct.pack("<f", limit))[0]
        reader_ct = [cb_gate, cb_up, cb_gb, cb_ub, cb_idxr, Ht, Wt2, nact]
        reader_ct += ttnn.TensorAccessorArgs(raw).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(bias).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(idx).get_compile_time_args()
        writer_ct = [cb_out, cb_idxw, Ht, nact]
        writer_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        writer_ct += ttnn.TensorAccessorArgs(idx).get_compile_time_args()
        compute_ct = [per_core, cb_gate, cb_up, cb_out, cap_u, upmin_u, upmax_u, cb_gb, cb_ub]
        idx_bytes = ((nact * 4 + _TILE_BYTES - 1) // _TILE_BYTES) * _TILE_BYTES or _TILE_BYTES
        _CACHE[key] = (
            core,
            core_list,
            per_core,
            cb_gate,
            cb_up,
            cb_gb,
            cb_ub,
            cb_out,
            cb_idxr,
            cb_idxw,
            reader_ct,
            writer_ct,
            compute_ct,
            idx_bytes,
        )

    (
        core,
        core_list,
        per_core,
        cb_gate,
        cb_up,
        cb_gb,
        cb_ub,
        cb_out,
        cb_idxr,
        cb_idxw,
        reader_ct,
        writer_ct,
        compute_ct,
        idx_bytes,
    ) = _CACHE[key]

    def cb(idx_id, n, ps=_TILE_BYTES):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx_id, data_format=ttnn.bfloat16, page_size=ps)
        return ttnn.CBDescriptor(total_size=n * ps, core_ranges=core, format_descriptors=[fmt])

    cbs = [
        cb(cb_gate, 8),
        cb(cb_up, 8),
        cb(cb_gb, 8),
        cb(cb_ub, 8),
        cb(cb_out, 4),
        cb(cb_idxr, 1, idx_bytes),
        cb(cb_idxw, 1, idx_bytes),
    ]

    rr = ttnn.RuntimeArgs()
    wr = ttnn.RuntimeArgs()
    for c, (x, y) in enumerate(core_list):
        st = c * per_core
        rr[x][y] = [raw.buffer_address(), bias.buffer_address(), idx.buffer_address(), st, per_core]
        wr[x][y] = [out.buffer_address(), idx.buffer_address(), st, per_core]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/swiglu_reader_bias.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=reader_ct,
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/swiglu_writer_final.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=writer_ct,
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/swiglu_compute_bias.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=compute_ct,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([raw, bias, idx, out], prog)
