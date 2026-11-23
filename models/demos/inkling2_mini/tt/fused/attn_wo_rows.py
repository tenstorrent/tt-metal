# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Speculative verify: AttnWo (o-projection + post-attention short conv in one program) for R rows.

The R tokens' o-projection inputs ride in rows 0 .. R-1 of x (each row of a bank linear is bit-equal to the 1-row
call); each core then runs the conv of every row on its own columns, as conv_add's verify rows do: row r's lags
1..min(r, 3) are rows r-1.. of a, older lags the ring as it was before the pass; fp32 a of row r goes to the conv's
pending slot r (the ring is written by the commit). Output row r: a [1, 1, 1, C] fp32 row object with a 1-vector dense
twin, bit-equal to AttnWo of token r (kernels attn_wo_rows_writer.cpp / attn_wo_rows_compute.cpp)."""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attn_wo import TILE, AttnWo


class AttnWoRows(AttnWo):
    def __call__(self, x, conv, resids, R):
        """x: o-projection input, rows 0 .. R-1 = tokens. conv = (FusedShortConv, ring, phase of row 0).
        resids: R residual rows with single-vector dense twins, or None. Returns R conv outputs (dense twins only)."""
        b = self.bank
        cv, ring, phase = conv
        assert cv.c == self.cols and 1 <= R <= 4
        outs = [cv.out.get_twin((("conv", ttnn.float32), "rows", r), 1, cv.c, ttnn.float32) for r in range(R)]
        pend = [cv.pending_slot(ring, r) for r in range(R)]
        lags = [ring[(phase - k) % 4] for k in range(1, 4)]
        res_es, res = 0, [0] * R
        io = [*pend, *lags, cv.taps, *(d for _, d in outs)]
        if resids is not None:
            tws = [F.twin(t) for t in resids]
            assert all(tw is not None and tw[1] == 1 for tw in tws), "resid without a single-vector dense twin"
            res_es = F.ES[tws[0][0].dtype]
            res = [tw[0].buffer_address() for tw in tws]
            io += [tw[0] for tw in tws]
        conv_args = [p.buffer_address() for p in pend] + [g.buffer_address() for g in lags]
        conv_args += [cv.taps.buffer_address()] + res + [d.buffer_address() for _, d in outs]
        tb = ttnn.Tile([TILE, TILE]).get_tile_size(b.dtype)
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
        named = dict(kt=b.Kt, kb=b.KB, nblk=b.nblk, n_banks=b.nb, max_ncq=b.max_ncq, out_tile=otb, cols=self.cols,
                     res_es=res_es, slots=self.slots, rows=R)  # fmt: skip
        r_args, c_args, w_args = {}, {}, {}
        for c, i in b.cores:
            bank, rot, part = b.items[i]
            core = ttnn.CoreCoord(*c)
            r_args[core] = b.reader_args(i)
            c_args[core] = [rot, len(part)] + [n for n, _ in part]
            w_args[core] = [x.buffer_address(), rot, len(part)] + conv_args + [v for p_ in part for v in p_]
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=b.compute.math_fidelity, fp32_dest_acc_en=True, dst_full_sync_en=True,
            math_approx_mode=b.compute.math_approx_mode,
        )  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in [3, 4] + ([5] if res_es == 4 else []):  # fp32 conv inputs reach DST unrounded (as conv.py)
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            *self._reader(acc(b.weight), named, r_args),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/attn_wo_rows_compute.cpp", core_ranges=b.grid,
                named_compile_time_args=list(named.items()), runtime_args=[(k, v) for k, v in c_args.items()],
                config=compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/attn_wo_rows_writer.cpp", core_ranges=b.grid,
                compile_time_args=acc(x), named_compile_time_args=list(named.items()),
                runtime_args=[(k, v) for k, v in w_args.items()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]  # fmt: skip
        t32 = ttnn.TileDescriptor(ttnn.Tile([TILE, TILE]))

        def cb(i, fmt, page, n):
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=b.grid,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page, tile=t32)])  # fmt: skip

        cbs = [
            cb(0, ttnn.bfloat16, 2048, b.Kt),  # x, whole K resident
            cb(1, b.dtype, tb, 4 * b.KB * b.max_ncq),  # weight blocks (bank_linear_reader.cpp)
            cb(16, ttnn.bfloat16, otb, 8),  # output tiles of a pass
            cb(2, ttnn.bfloat16, 2048, R),  # a, page r = row r (slot layout)
            cb(3, ttnn.float32, 4096, 3),  # ring lag 1..3 (row 0's)
            cb(4, ttnn.float32, 4096, 4),  # taps lag 0..3
            cb(17, ttnn.float32, 4096, 1),  # fp32 a -> pending slot
            cb(18, ttnn.float32, 4096, 1),  # conv out
            cb(19, ttnn.bfloat16, 2048, 1),  # conv rounded to a's dtype
        ]
        if res_es:
            cbs.append(cb(5, tws[0][0].dtype, 1024 * res_es, R))
        ttnn.generic_op([x, b.weight, *io], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return [o for o, _ in outs]
