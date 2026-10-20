# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused CG vector kernels: one pass over whole fp32 vectors per call, split evenly over the cores by tiles.

  dot   bytes [64 core, 64 core + 4) of part = this core's share of a . b (the rest of its 64-byte slot zeroed);
        DotFinishOp adds the slots
  axpy  b = a + (s1 / s2) b                  (CG's p = z + beta p, beta = rtz / rtz_old)
  xr    x += (s1 / s2) p, r -= (s1 / s2) Ap, and this core's share of r . r in its slot of part   (alpha = rtz / pAp)
The scalar quotient s1 / s2 is computed on chip (IEEE fp32 division in the reader), so no host round trip.
"""

import ttnn

from models.experimental.hpcg.tt.layout import TILE
from models.experimental.hpcg.tt.custom_ops.sparse_rows.op import core_ranges

KERNEL_ROOT = "models/experimental/hpcg/tt/custom_ops/cg_vector/kernels"
CB = dict(CB_A=0, CB_B=1, CB_C=2, CB_D=3, CB_S=4, CB_ACC=5, CB_OUT=16, CB_OUT2=17, CB_OUT3=18)
VB = 2  # tiles per read batch (1-8 measured within 3%)


class CgVectorOp:
    """mode dot: (a, b, part). axpy: (a, b, s1, s2), writes b. xr: (a=x, b=p, c=r, d=Ap, s1, s2, part), writes x
    and r. All vectors are fp32 TILE [P / 32, 32] with the same P; s1 and s2 are 1x1-element fp32 tiles."""

    def __init__(self, device, mode, a, b, c=None, d=None, s1=None, s2=None, part=None):
        if mode not in ("dot", "axpy", "xr"):
            raise ValueError(f"mode must be dot, axpy or xr, got {mode!r}")
        need = {"dot": (part,), "axpy": (s1, s2), "xr": (c, d, s1, s2, part)}[mode]
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
        cores = core_ranges(nc, g.x)
        base, rem = divmod(nt, nc)
        c = c if c is not None else a
        d = d if d is not None else a
        s1 = s1 if s1 is not None else a
        s2 = s2 if s2 is not None else a
        part = part if part is not None else a
        o1 = b if mode == "axpy" else a  # axpy writes over b (p = z + beta p); xr writes x (a) and r (c)
        o2 = b if mode == "dot" else c  # dot: the writer reads b
        ra, wa, ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        t = 0
        r_head = [x.buffer_address() for x in (a, b, c, d, s1, s2)]
        w_head = [x.buffer_address() for x in (o1, o2, part)]
        ad = d.buffer_address()
        for i in range(nc):
            k = base + (1 if i < rem else 0)
            cx, cy = i % g.x, i // g.x
            ra[cx][cy] = r_head + [t, k]
            wa[cx][cy] = w_head + [t, k, i, ad]
            ca[cx][cy] = [k]
            t += k
        defs = [(k, str(v)) for k, v in CB.items()] + [("MODE_" + mode.upper(), "1"), ("VB", str(VB))]

        def accessors(*ts):
            return [x for tt in ts for x in ttnn.TensorAccessorArgs(tt).get_compile_time_args()]

        def cb(idx, n=2):
            return ttnn.CBDescriptor(
                total_size=n * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=4096)
                ],
            )

        cbs = (
            [cb(CB[k], 2 * VB) for k in ("CB_A", "CB_B", "CB_C", "CB_D")]
            + [cb(CB[k]) for k in ("CB_ACC", "CB_OUT", "CB_OUT2", "CB_OUT3")]
            + [cb(CB["CB_S"], 1)]
        )
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
        )
        m = [ttnn.UnpackToDestMode.Default] * 64
        for i in range(6):  # CB_A .. CB_ACC
            m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(m)
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/reader.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(a, b, c, d, s1, s2),
                defines=defs,
                runtime_args=ra,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/writer.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=accessors(o1, o2, part, d),
                defines=defs,
                runtime_args=wa,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_ROOT}/compute.cpp",
                source_type=src,
                core_ranges=cores,
                compile_time_args=[],
                defines=defs,
                runtime_args=ca,
                config=cfg,
            ),
        ]
        self.prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        self.io = [x for x in (a, b, c, d, s1, s2) if x is not None] + [part if mode != "axpy" else o1]
        self.nc = nc

    def __call__(self):
        return ttnn.generic_op(self.io, self.prog)


class DotFinishOp:
    """Finishes a dot product on one core: adds the per-core values in the first nparts pages of `part` (64-byte slots,
    zero elsewhere), reduces the 1024 lanes, and
    writes the value to element (0, 0) of `out` (an fp32 32x32 tile). keep_old: first copies out's current tile to
    `old` (CG's previous r.z). sqrt: writes the square root (the residual norm). Replaces two ttnn.sum calls and a
    ttnn.copy (or ttnn.sqrt) with one launch."""

    def __init__(self, device, part, nparts, out, old=None, keep_old=False, sqrt=False):
        if keep_old and old is None:
            raise ValueError("keep_old needs the `old` tile")
        old = old if old is not None else out
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        ra, wa, ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        ra[0][0] = [part.buffer_address(), nparts]
        wa[0][0] = [out.buffer_address(), old.buffer_address(), int(keep_old)]
        ca[0][0] = [nparts, int(sqrt)]
        cbs_ids = dict(CB_A=0, CB_OUT=16, CB_T=17)
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
