# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary decode linear whose weight records come from the prefetched stream (stream.py) instead of DRAM.

Same records (decode_linear3.pack_records_inline) and compute (tdec3_compute.cpp, BATCH4, every record from c_1) as
TernaryDecodeLinear3; it runs on the 64 receiver cores, receiver g computing output tiles assign[g] = (first, count),
and kernels/tgcb_reader.cpp copies the records from the ring into c_1 (two records, as TernaryDecodeLinear3's c_1)."""

import os

import ttnn
from models.demos.blackhole.bonsai2.tt.prefetch.stream import BLOCK_PAGES, NUM_RECEIVERS, PAGE
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, TILE, pack_records_inline

TERNARY = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"
PREFETCH = "models/demos/blackhole/bonsai2/tt/prefetch/kernels/"


def balanced(nt, receivers=NUM_RECEIVERS, offset=0):
    """Output tiles 0..nt-1 over the receivers, receiver (offset + i) % receivers taking the i-th larger share."""
    base, extra = divmod(nt, receivers)
    counts = [0] * receivers
    for i in range(receivers):
        counts[(offset + i) % receivers] = base + (1 if i < extra else 0)
    assign, start = [], 0
    for c in counts:
        assign.append((start, c))
        start += c
    return assign


class GcbTernaryLinear:
    def __init__(self, K, N, rec, ppr, assign):
        """rec, ppr: pack_records_inline(t, s); assign[g] = (first output tile, count) of receiver g."""
        self.K, self.N, self.ppr, self.assign = K, N, ppr, assign
        self.rec = rec

    @classmethod
    def from_host(cls, t_kn, s_nb, assign=None):
        rec, ppr = pack_records_inline(t_kn, s_nb)
        K, N = t_kn.shape
        return cls(K, N, rec, ppr, assign or balanced(N // TILE))

    def pages(self):
        """Per receiver, the pages it reads (for WeightStream.add)."""
        return [self.rec[f * self.ppr : (f + n) * self.ppr] if n else None for f, n in self.assign]

    def bind(self, counts):
        """counts: WeightStream.add's (pages, blocks) per receiver."""
        self.counts = counts

    def __call__(self, x_rot, pf, inflight=8, dram=None):
        """x_rot [C, K] bf16 ROW_MAJOR (C copies, one per DRAM bank); pf: stream.Prefetch -> [1, 1, N] bf16 TILE."""
        device = x_rot.device()
        K, N, ppr = self.K, self.N, self.ppr
        Kt, KB = K // TILE, K // KBLOCK
        G = -(-KB // 32)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        cores = [c for c, (_, n) in zip(pf.receivers, self.assign)]
        crs = pf.receiver_crs
        max_n = max(n for _, n in self.assign)
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for g, (cc, (first, n), (pages, blocks)) in enumerate(zip(cores, self.assign, self.counts)):
            extra = [dram.buffer_address(), first * ppr] if dram is not None else []
            rd[cc.x][cc.y] = [x_rot.buffer_address(), g % x_rot.shape[0], pages, blocks] + extra
            wr[cc.x][cc.y] = [out.buffer_address(), out.buffer_address(), first, n, 0]
            cp[cc.x][cc.y] = [n]
        tiny = ttnn.TileDescriptor(1, 32, False)

        def cb(index, fmt, page, pages, tile=None):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page, tile=tile)
                ],
            )

        ring = (
            None
            if pf.gcb is None
            else ttnn.CBDescriptor(total_size=pf.gcb.size(), core_ranges=crs, format_descriptors=[])
        )
        if ring is not None:
            ring.remote_format_descriptors = [
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.bfloat8_b, page_size=BLOCK_PAGES * PAGE)
            ]
            ring.set_global_circular_buffer(pf.gcb)
        dbg = [(d, "1") for d in os.environ.get("BONSAI_GCB_DEFINES", "").split(",") if d]
        two_cb = ("TWO_CB", "1") in dbg
        rd_ct = [Kt, BLOCK_PAGES] + ttnn.TensorAccessorArgs(x_rot).get_compile_time_args()
        if dram is not None:
            rd_ct += ttnn.TensorAccessorArgs(dram).get_compile_time_args() + [ppr]
            dbg = dbg + [("DRAM_FEED", "1")]
        wr_ct = [ppr, inflight] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=PREFETCH + "tgcb_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    defines=dbg,
                    runtime_args=rd,
                    # BRISC on NOC 0, as the stock GCB receivers: from NCRISC the ring acks never reached the
                    # sender (watcher: sender at remote reserve_back, receivers at remote wait_front)
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=TERNARY + "tdec3_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    defines=[("GCB_STREAM", "1"), ("INLINE_SCALES", "1"), ("OUT_PER_NT", "1"), ("SPLIT_NT", "0")]
                    + [("S_PER_NT", str(G))]
                    + dbg,
                    runtime_args=wr,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=TERNARY + "tdec3_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[Kt, KB],
                    defines=[("SFPU_ITER", "10"), ("SCALE_FP32", "0"), ("SFPU_REINIT", "1"), ("SCALE_GROUP16", "1")]
                    + [("INLINE_SCALES", "1"), ("RECORD_PAGES", str(ppr * 16))]
                    + ([] if two_cb else [("ONE_WCB", "1")])
                    + ([("BATCH4", "1")] if K % 512 == 0 else [])
                    + dbg,
                    runtime_args=cp,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cb(0, ttnn.bfloat16, 64, Kt, tiny),
                *([ring] if pf.gcb is not None else []),
                *([cb(6, ttnn.bfloat8_b, 256, 2 * ppr * 16)] if two_cb else []),
                cb(2, ttnn.float32, 2048, 1),
                cb(16, ttnn.bfloat16, 2048, max(max_n, 1)),
                cb(1, ttnn.bfloat8_b, 256, 2 * ppr * 16),
            ],
        )
        return ttnn.generic_op([x_rot, out], program)
