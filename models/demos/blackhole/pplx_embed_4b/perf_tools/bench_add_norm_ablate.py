# Row-split add+RMSNorm at the model's per-batch placement (a DRAM; b and the normalised output L1; the sum L1 at bs8 /
# 16, DRAM at bs32): what bounds it. Scratch kernel variants (written to a temp dir, not the repo) skip the a / b reads,
# the sum / normalised writes, or the partial exchange (no peer writes, no semaphore wait: the compute sums whatever
# sits in CB 8), or replace the compute with tile copies that keep every CB handshake (data movement only); traced us
# per call.
# Usage: bench_add_norm_ablate.py [batch ...]   (default 8 16 32; AN_A_L1=1 puts a in L1)
import os
import re
import statistics
import sys
import tempfile
import time

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import make_add_norm_constants
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import op_split as ops

W, EPS, B8 = 2560, 1e-6, ttnn.bfloat8_b
R_BY_BS = {8: 5, 16: 5, 32: 4}
SCRATCH = os.environ.get("ABL_DIR") or tempfile.mkdtemp(prefix="addnorm_abl_")


PASS_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
// Data-movement floor: the split compute's CB handshakes, with tile copies instead of the math (a -> sum out and the
// bfp8 working copy, working copy -> normalised out; the partial tile is pushed unwritten).
void kernel_main() {
    constexpr uint32_t Wc = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    CircularBuffer ca(0), cbb(1), cg(2), csc(3), ceps(4), cs(5), part(7), parts(8), so(16), no(17);
    compute_kernel_hw_startup(0, 0, 16);
    cg.wait_front(Wc);
    csc.wait_front(1);
    ceps.wait_front(1);
    const uint32_t n_waves = get_arg_val<uint32_t>(0);
    for (uint32_t w = 0; w < n_waves; ++w) {
        ca.wait_front(Wc);
        cbb.wait_front(Wc);
        so.reserve_back(Wc);
        cs.reserve_back(Wc);
        copy_tile_to_dst_init_short(0);
        for (uint32_t t = 0; t < Wc; ++t) {
            tile_regs_acquire();
            copy_tile(0, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 16, t);
            pack_tile(0, 5, t);
            tile_regs_release();
        }
        so.push_back(Wc);
        cs.push_back(Wc);
        ca.pop_front(Wc);
        cbb.pop_front(Wc);
        part.reserve_back(1);
        part.push_back(1);
        parts.wait_front(R);
        parts.pop_front(R);
        cs.wait_front(Wc);
        no.reserve_back(Wc);
        copy_tile_to_dst_init_short(5);
        for (uint32_t t = 0; t < Wc; ++t) {
            tile_regs_acquire();
            copy_tile(5, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, 17, t);
            tile_regs_release();
        }
        no.push_back(Wc);
        cs.pop_front(Wc);
    }
}
"""


def variant(name, read=1, write=1, xchg=1):
    r = open(ops.READER_KERNEL).read()
    r = re.sub(r"noc\.async_read\(s([ab]),", r"if (ABL_READ) noc.async_read(s\1,", r)
    w = open(ops.WRITER_KERNEL).read()
    w = re.sub(r"noc\.async_write\(c([so]),", r"if (ABL_WRITE) noc.async_write(c\1,", w)
    for call in (
        "noc_async_write_one_packet_with_trid(",
        "noc_async_write(part_src",
        "noc_semaphore_inc(",
        "noc_semaphore_wait_min(",
    ):
        w = w.replace(call, "if (ABL_XCHG) " + call)
    hdr = f"#define ABL_READ {read}\n#define ABL_WRITE {write}\n#define ABL_XCHG {xchg}\n"
    rp, wp = os.path.join(SCRATCH, f"r_{name}.cpp"), os.path.join(SCRATCH, f"w_{name}.cpp")
    open(rp, "w").write(hdr + r)
    open(wp, "w").write(hdr + w)
    return rp, wp, ops.COMPUTE_KERNEL


def main():
    batches = [int(v) for v in sys.argv[1:]] or [8, 16, 32]
    reader0, writer0, compute0 = ops.READER_KERNEL, ops.WRITER_KERNEL, ops.COMPUTE_KERNEL
    pass_c = os.path.join(SCRATCH, "compute_pass.cpp")
    open(pass_c, "w").write(PASS_COMPUTE)
    variants = {
        "full": (reader0, writer0, compute0),
        "no a/b reads": variant("noread", read=0),
        "no sum/out writes": variant("nowrite", write=0),
        "no reads, no writes": variant("nordwr", read=0, write=0),
        "local exchange": variant("noxchg", xchg=0),
        "compute + handshakes": variant("none", read=0, write=0, xchg=0),
        "DM only (copy compute)": (reader0, writer0, pass_c),
        "handshakes only": variant("none", read=0, write=0, xchg=0)[:2] + (pass_c,),
    }
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    try:
        torch.manual_seed(0)
        G, SC, EP = make_add_norm_constants(torch.rand(W) * 0.5 + 0.75, EPS, D)
        for bs in batches:
            M, R = bs * 512, R_BY_BS[bs]
            smc = DR if bs == 32 else L1
            amc = L1 if os.getenv("AN_A_L1") == "1" else DR  # the post-MLP call at bs8 / 16 reads a from L1
            a = ttnn.from_torch(torch.randn(1, 1, M, W), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=amc)
            b = ttnn.from_torch(
                torch.randn(1, 1, M, W) * 0.3, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
            )
            o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), B8, ttnn.TILE_LAYOUT, D, L1)
            for vname, (rp, wp, cp) in variants.items():
                ops.READER_KERNEL, ops.WRITER_KERNEL, ops.COMPUTE_KERNEL = rp, wp, cp
                fn = lambda: ops.fused_add_rmsnorm_split(
                    a, b, G, SC, EP, R=R, memory_config=smc, out_memory_config=L1, out_tensor=o
                )[0]
                for _ in range(2):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(D)
                n = 4
                tid = ttnn.begin_trace_capture(D, cq_id=0)
                outs = [fn() for _ in range(n)]
                ttnn.end_trace_capture(D, tid, cq_id=0)
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts = []
                for _ in range(7):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / n * 1e6)
                ttnn.release_trace(D, tid)
                [ttnn.deallocate(t) for t in outs]
                print(
                    f"RES addnorm bs{bs} R={R} a={'L1' if amc is L1 else 'DRAM'} {vname:22s} {statistics.median(ts):7.1f} us/call",
                    flush=True,
                )
            for t in (a, b, o):
                ttnn.deallocate(t)
    finally:
        ops.READER_KERNEL, ops.WRITER_KERNEL, ops.COMPUTE_KERNEL = reader0, writer0, compute0
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
