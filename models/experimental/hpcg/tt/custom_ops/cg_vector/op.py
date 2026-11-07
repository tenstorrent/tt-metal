# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused CG vector kernels: one pass over whole fp32 vectors per call, split evenly over the cores by tiles.

  dot   bytes [64 core, 64 core + 4) of part = this core's share of a . b (the rest of its 64-byte slot zeroed);
        DotFinishOp adds the slots
  axpy  b = a + (s1 / s2) b                  (CG's p = z + beta p, beta = rtz / rtz_old)
  xr    x += (s1 / s2) p, r -= (s1 / s2) Ap, and this core's share of r . r in its slot of part   (alpha = rtz / pAp)
  zero  a = 0, b = 0: zeros written, nothing read (x - x would keep a NaN or inf)
The scalar quotient s1 / s2 is computed on chip (IEEE fp32 division in the reader), so no host round trip.
"""

import ttnn

from models.experimental.hpcg.tt import constants as C
from models.experimental.hpcg.tt.layout import TILE
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import core_ranges, logical_coords

KERNEL_ROOT = "models/experimental/hpcg/tt/custom_ops/cg_vector/kernels"
CB, VB = C.CBS["CG_VECTOR"], C.VB  # shared with xhpcg/tt/ops.cpp (tt/shared_constants.h)


_PARTS = {}


def _shared_parts(device, nc, base, rem, mcode):
    """The descriptor parts that are the same for every CgVectorOp of a vector length and mode, made once per
    process (ProgramDescriptor copies them in C++; building them from Python costs about 30 us per op, mostly the
    compute kernel's per-core argument lists): the core range, the circular buffers, the defines and the compute
    kernel."""
    key = (id(device), nc, base, rem, mcode)
    if key in _PARTS:
        return _PARTS[key]
    g = device.compute_with_storage_grid_size()
    cores = core_ranges(nc, g.x)

    def cb(idx, n=2):
        return ttnn.CBDescriptor(
            total_size=n * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=4096)],
        )

    cbs = (
        [cb(CB[k], 2 * VB) for k in ("CB_A", "CB_B", "CB_C", "CB_D")]
        + [cb(CB[k]) for k in ("CB_ACC", "CB_OUT", "CB_OUT2", "CB_OUT3")]
        + [cb(CB["CB_S"], 1)]
    )
    defs = [(k, str(v)) for k, v in CB.items()] + [("VB", str(VB))]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    m = [ttnn.UnpackToDestMode.Default] * 64
    for i in range(6):  # CB_A .. CB_ACC
        m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
    ca = [(xy, [base + (1 if i < rem else 0), mcode]) for i, xy in enumerate(logical_coords(device)[:nc])]
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[],
        defines=defs,
        runtime_args=ca,
        config=cfg,
    )
    _PARTS[key] = (cores, cbs, defs, compute)
    return _PARTS[key]


class CgVectorOp:
    """mode dot: (a, b, part). axpy: (a, b, s1, s2), writes b. xr: (a=x, b=p, c=r, d=Ap, s1, s2, part), writes x
    and r. zero: (a, b), writes zeros to both. All vectors are fp32 TILE [P / 32, 32] with the same P; s1 and s2 are one 4 KB page each, the value in
    word 0 (element (0, 0) of a tile)."""

    def __init__(self, device, mode, a, b, c=None, d=None, s1=None, s2=None, part=None):
        if mode not in ("dot", "axpy", "xr", "zero"):
            raise ValueError(f"mode must be dot, axpy, xr or zero, got {mode!r}")
        need = {"dot": (part,), "axpy": (s1, s2), "xr": (c, d, s1, s2, part), "zero": ()}[mode]
        if any(t is None for t in need):
            raise ValueError(f"mode {mode} is missing an input")
        for t in (a, b, c, d):
            if t is not None and (
                t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or tuple(t.shape) != tuple(a.shape)
            ):
                raise ValueError("vectors must be fp32 TILE tensors of one shape")
        g = device.compute_with_storage_grid_size()
        nt = int(a.shape[0]) * int(a.shape[1]) // TILE
        nc = min(g.x * g.y, nt)
        base, rem = divmod(nt, nc)
        c = c if c is not None else a
        d = d if d is not None else a
        s1 = s1 if s1 is not None else a
        s2 = s2 if s2 is not None else a
        part = part if part is not None else a
        o1 = b if mode == "axpy" else a  # axpy writes over b (p = z + beta p); xr writes x (a) and r (c)
        o2 = b if mode in ("dot", "zero") else c  # dot: the writer reads b; zero: writes a and b
        # the mode is a runtime argument (0 dot, 1 axpy, 2 xr, 3 zero): one kernel binary for all four modes
        mcode = {"dot": 0, "axpy": 1, "xr": 2, "zero": 3}[mode]
        # reader / writer: common arguments (a core derives its tile range from its logical position); the compute
        # kernel keeps its per-core tile count
        reader_common = [x.buffer_address() for x in (a, b, c, d, s1, s2)] + [base, rem, g.x, mcode]
        writer_common = [x.buffer_address() for x in (o1, o2, part)] + [base, rem, g.x, d.buffer_address(), mcode]
        cores, cbs, defs, compute = _shared_parts(device, nc, base, rem, mcode)

        def accessors(*ts):
            return [x for tt in ts for x in ttnn.TensorAccessorArgs(tt).get_compile_time_args()]

        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/reader.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(a, b, c, d, s1, s2),
                defines=defs,
                common_runtime_args=reader_common,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/writer.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(o1, o2, part, d),
                defines=defs,
                common_runtime_args=writer_common,
                config=ttnn.WriterConfigDescriptor(),
            ),
            compute,
        ]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        self.io = [x for x in (a, b, c, d, s1, s2) if x is not None] + [part if mode not in ("axpy", "zero") else o1]
        self.nc = nc

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)


_FINISH = []


def _finish_parts():
    """DotFinishOp's core (0, 0), circular buffers, defines and compute configuration: the same for every finish,
    made once per process."""
    if not _FINISH:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        cbs_ids = C.CBS["DOT_FINISH"]
        # keep_old and sqrt are runtime arguments: the three finishes of an iteration share one binary
        defs = [(k, str(v)) for k, v in cbs_ids.items()] + [("VB", str(VB))]

        def cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=4096)
                ],
            )

        cbs = [cb(cbs_ids["CB_A"], 2 * VB), cb(cbs_ids["CB_OUT"], 1), cb(cbs_ids["CB_T"], 1)]
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
        )
        m = [ttnn.UnpackToDestMode.Default] * 64
        m[cbs_ids["CB_A"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
        _FINISH.append((cores, cbs, defs, cfg))
    return _FINISH[0]


class DotFinishOp:
    """Finishes a dot product on one core: adds the per-core values in the first nparts pages of `part` (64-byte slots,
    zero elsewhere), reduces the 1024 lanes, and writes the value to word 0 of `out` (one fp32 4 KB page: element
    (0, 0) of a tile). keep_old: first copies out's current page to `old` (CG's previous r.z). sqrt: writes the
    square root (the residual norm). Replaces two ttnn.sum calls and a ttnn.copy (or ttnn.sqrt) with one launch."""

    def __init__(self, device, part, nparts, out, old=None, keep_old=False, sqrt=False):
        if keep_old and old is None:
            raise ValueError("keep_old needs the `old` tile")
        old = old if old is not None else out
        ra, wa, ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        ra[0][0] = [part.buffer_address(), nparts]
        wa[0][0] = [out.buffer_address(), old.buffer_address(), int(keep_old)]
        ca[0][0] = [nparts, int(sqrt)]
        cores, cbs, defs, cfg = _finish_parts()
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH

        def accessors(*ts):
            return [x for t in ts for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/finish_reader.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(part),
                defines=defs,
                runtime_args=ra,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/finish_writer.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(out, old),
                defines=defs,
                runtime_args=wa,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/finish_compute.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=[],
                defines=defs,
                runtime_args=ca,
                config=cfg,
            ),
        ]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        self.io = [part, old, out]

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)
