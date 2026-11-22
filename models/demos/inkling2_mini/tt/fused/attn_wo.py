# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Attention o-projection + the post-attention short conv in one program.

The o-projection is a BankLinear (tt/fused/bank_linear.py: its schedule, placement, weight and reader unchanged).
The conv (tt/fused/conv.py FusedShortConv.conv_add) is elementwise per channel, so each o-projection core runs it on
its own output columns and no core waits for another: the writer gathers row 0 of the core's output tiles into one
local dense page and reads the ring slot, taps and residual chunks of those columns while the weights stream; the
compute runs the conv kernel's op sequence on that page. Output: the conv module's own cached output tensor (its
dense twin written, no row form; a itself is not written anywhere), bit-equal to
conv_add(a, ring, phase, resid, fp32, row_out=False). (This replaces a dense twin of a for the conv to read,
the conv no longer reads a from DRAM at all.)
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

TILE = 32


class AttnWo:
    def __init__(self, bank, cols):
        """bank: the o-projection's BankLinear (bf16 out); cols: the hidden size (columns of a)."""
        assert bank.out_dtype == ttnn.bfloat16
        self.bank, self.cols = bank, cols
        # column tiles < cols per core = used slots of its dense page
        slots = [sum(1 for ncq, n0 in part for t in range(ncq) if (n0 + t) * TILE < cols) for _, _, part in bank.items]
        self.slots = max(slots)
        assert self.slots <= 32, self.slots

    def __call__(self, x, conv):
        """x: o-projection input [1,1,32,K] bf16. conv = (FusedShortConv, ring, phase, resid or None), resid with a
        single-vector dense twin. Returns the conv output (fp32 row-form object, dense twin only)."""
        b = self.bank
        cv, ring, phase, resid = conv
        assert cv.c == self.cols
        y, y_dense = cv.out.get_twin(("conv", ttnn.float32), 1, cv.c, ttnn.float32)
        slots = [ring[(phase - k) % 4] for k in range(4)]
        res_es, io = 0, [*slots, cv.taps, y_dense, y]
        if resid is not None:
            tw = F.twin(resid)
            assert tw is not None and tw[1] == 1, "resid without a single-vector dense twin"
            res_es = F.ES[tw[0].dtype]
            io.append(tw[0])
        conv_args = [s.buffer_address() for s in slots] + [cv.taps.buffer_address()]
        conv_args += [tw[0].buffer_address() if res_es else 0, y_dense.buffer_address()]
        tb = ttnn.Tile([TILE, TILE]).get_tile_size(b.dtype)
        otb = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
        named = dict(kt=b.Kt, kb=b.KB, nblk=b.nblk, n_banks=b.nb, max_ncq=b.max_ncq, out_tile=otb, cols=self.cols,
                     res_es=res_es, slots=self.slots)  # fmt: skip
        r_args, c_args, w_args = {}, {}, {}
        for c, i in b.cores:
            bank, rot, part = b.items[i]
            core = ttnn.CoreCoord(*c)
            r_args[core] = [b.weight.buffer_address(), bank, b.offsets[i], len(part)] + [n for n, _ in part]
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
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/bank_linear_reader.cpp", core_ranges=b.grid,
                compile_time_args=acc(b.weight), named_compile_time_args=list(named.items()),
                runtime_args=[(k, v) for k, v in r_args.items()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/attn_wo_compute.cpp", core_ranges=b.grid,
                named_compile_time_args=list(named.items()), runtime_args=[(k, v) for k, v in c_args.items()],
                config=compute,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/attn_wo_writer.cpp", core_ranges=b.grid,
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
            cb(2, ttnn.bfloat16, 2048, 1),  # a, this core's columns (slot layout)
            cb(3, ttnn.float32, 4096, 3),  # ring lag 1..3
            cb(4, ttnn.float32, 4096, 4),  # taps lag 0..3
            cb(17, ttnn.float32, 4096, 1),  # fp32 a -> ring slot
            cb(18, ttnn.float32, 4096, 1),  # conv out
            cb(19, ttnn.bfloat16, 2048, 1),  # conv rounded to a's dtype
        ]
        if res_es:
            cbs.append(cb(5, tw[0].dtype, 1024 * res_es, 1))
        ttnn.generic_op([x, b.weight, *io], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return y  # the conv module's cached object: its dense twin is keyed by identity
