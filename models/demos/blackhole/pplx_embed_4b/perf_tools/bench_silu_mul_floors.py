# silu_mul (mode 3, the single-pass SwiGLU) at the bs1 call ([512, 9728] bfp8 a / b in L1 interleaved, out L1
# interleaved): what bounds it. Scratch kernel variants (a temp dir, not the repo) skip the a / b reads, the output
# writes, or replace the SwiGLU with a copy of a's tiles that keeps every CB handshake (data movement only); traced, wall
# us per call. Run under TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> and read device us with
# device_kernel_us.py <dir> <variant names in print order>.
import os
import re
import statistics
import tempfile
import time

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.silu_mul import op

SCRATCH = tempfile.mkdtemp(prefix="silu_mul_abl_")

PASS_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
// Data-movement floor: silu_mul's CB protocol with a's tiles copied to the output instead of the SwiGLU.
void kernel_main() {
    constexpr uint32_t CH = get_compile_time_arg_val(0);
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    CircularBuffer ca(0), cbb(1), co(16);
    compute_kernel_hw_startup(0, 0, 16);
    copy_tile_init(0);
    for (uint32_t u = 0; u < n_units; ++u) {
        ca.wait_front(CH);
        cbb.wait_front(CH);
        co.reserve_back(CH);
        tile_regs_acquire();
        for (uint32_t i = 0; i < CH; ++i) {
            copy_tile(0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < CH; ++i) {
            pack_tile(i, 16, i);
        }
        tile_regs_release();
        co.push_back(CH);
        ca.pop_front(CH);
        cbb.pop_front(CH);
    }
}
"""


def patched(src, name, pattern):
    s, n = re.subn(pattern, lambda m: "if (0) " + m.group(0), open(src).read())
    assert n > 0, (src, pattern)
    p = os.path.join(SCRATCH, name)
    open(p, "w").write(s)
    return p


def main():
    r0, w0, c0 = op.READER_KERNEL, op.WRITER_KERNEL, op.COMPUTE_KERNEL
    pass_c = os.path.join(SCRATCH, "compute_pass.cpp")
    open(pass_c, "w").write(PASS_COMPUTE)
    r_no = patched(r0, "reader_noread.cpp", r"noc\.async_read\(")
    w_no = patched(w0, "writer_nowrite.cpp", r"noc\.async_write\(")
    variants = {
        "full": (r0, c0, w0),
        "compute only": (r_no, c0, w_no),
        "DM only (copy compute)": (r0, pass_c, w0),
        "no reads": (r_no, c0, w0),
        "no writes": (r0, c0, w_no),
    }
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    L1, B8 = ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b
    try:
        torch.manual_seed(0)
        a, b = (
            ttnn.from_torch(torch.randn(1, 1, 512, 9728), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
            for _ in range(2)
        )
        for vname, (r, c, w) in variants.items():
            op.READER_KERNEL, op.COMPUTE_KERNEL, op.WRITER_KERNEL = r, c, w
            fn = lambda: op.silu_mul(a, b, out_dtype=B8, memory_config=L1, mode=3)
            for _ in range(2):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(D)
            n = 8
            tid = ttnn.begin_trace_capture(D, cq_id=0)
            for _ in range(n):
                ttnn.deallocate(fn())
            ttnn.end_trace_capture(D, tid, cq_id=0)
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts.append((time.perf_counter() - t0) / n * 1e6)
            ttnn.release_trace(D, tid)
            print(f"RES silu_mul bs1 {vname:24s} {statistics.median(ts):7.1f} us/call", flush=True)
    finally:
        op.READER_KERNEL, op.COMPUTE_KERNEL, op.WRITER_KERNEL = r0, c0, w0
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
