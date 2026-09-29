# bs16 fused heads op (head split + Q/K RMSNorm + RoPE) at the model's dtypes: what bounds it.
# Scratch kernel variants (written next to the scratch dir, not the repo) skip the unit reads, the unit writes, or
# replace the compute with a copy; traced wall time per call. Usage: bench_heads_bs16_ablate.py [batch]
import os
import re
import statistics
import sys
import tempfile
import time

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.tt_transformers.tt.common import get_rot_transformation_mat

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SCRATCH = os.environ.get("ABL_DIR") or tempfile.mkdtemp(prefix="heads_abl_")

PASS_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t unit_tiles = (heads_per_group * q_heads_per_kv + 2 * heads_per_group) * Wt;
    const uint32_t n = get_arg_val<uint32_t>(0);
    CircularBuffer in(0), out(16), sc(3), ep(4), ct(11), ccos(9), csin(10);
    compute_kernel_hw_startup(0, 0, 16);
    sc.wait_front(1);
    ep.wait_front(1);
    ct.wait_front(1);
    copy_init(0);
    for (uint32_t w = 0; w < n; ++w) {
        in.wait_front(unit_tiles);
        out.reserve_back(unit_tiles);
        ccos.wait_front(Wt);
        csin.wait_front(Wt);
        for (uint32_t t0 = 0; t0 < unit_tiles; t0 += 4) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < 4; ++j) copy_tile(0, t0 + j, j);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < 4; ++j) pack_tile(j, 16, t0 + j);
            tile_regs_release();
        }
        ccos.pop_front(Wt);
        csin.pop_front(Wt);
        out.push_back(unit_tiles);
        in.pop_front(unit_tiles);
    }
}
"""


def variant(src_path, name, read=1, write=1):
    src = open(src_path).read()
    src = re.sub(r"noc\.async_read\(\s*s0,", "if (ABL_READ) noc.async_read(s0,", src)
    src = re.sub(r"noc\.async_write\(", "if (ABL_WRITE) noc.async_write(", src)
    # includes are relative to the repo's kernel dir: keep them resolvable from the scratch copy
    out = os.path.join(SCRATCH, name)
    open(out, "w").write(f"#define ABL_READ {read}\n#define ABL_WRITE {write}\n" + src)
    return out


def main():
    # QWEN_FUSED_COMPUTE_V3=1: ablate the v3 compute (op.py sizes its CBs for it and binds COMPUTE_KERNEL_BS1)
    cattr = "COMPUTE_KERNEL_BS1" if os.getenv("QWEN_FUSED_COMPUTE_V3", "0") == "1" else "COMPUTE_KERNEL"
    reader, writer, compute = hop.READER_KERNEL, hop.WRITER_KERNEL, getattr(hop, cattr)
    pass_c = os.path.join(SCRATCH, "compute_pass.cpp")
    open(pass_c, "w").write(PASS_COMPUTE)
    variants = {
        "full": (reader, compute, writer),
        "compute only (no unit read/write)": (
            variant(reader, "r_noread.cpp", read=0),
            compute,
            variant(writer, "w_nowrite.cpp", write=0),
        ),
        "data movement only (copy compute)": (reader, pass_c, writer),
        "read only": (reader, pass_c, variant(writer, "w_nowrite.cpp", write=0)),
        "write only": (variant(reader, "r_noread.cpp", read=0), pass_c, writer),
        "handshakes only": (
            variant(reader, "r_noread.cpp", read=0),
            pass_c,
            variant(writer, "w_nowrite.cpp", write=0),
        ),
    }
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
    try:
        GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
        ang = torch.rand(1, 1, S, DH) * 6.28
        DR = ttnn.DRAM_MEMORY_CONFIG
        # cos / sin / rotation tile in L1 like the model (QWEN_ROPE_PREFILL_L1=1); ABL_ROPE=DRAM for the old placement
        RM = DR if os.getenv("ABL_ROPE", "L1") == "DRAM" else ttnn.L1_MEMORY_CONFIG
        cos = ttnn.from_torch(torch.cos(ang), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=RM)
        sin = ttnn.from_torch(torch.sin(ang), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=RM)
        T = ttnn.from_torch(
            get_rot_transformation_mat(32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=RM
        )
        placements = [("DRAM in", DR)]
        if os.getenv("ABL_L1_IN", "1") == "1":
            placements.append(("L1 in", ttnn.L1_MEMORY_CONFIG))
        for pname, mc in placements:
            try:
                x = ttnn.from_torch(
                    torch.randn(B, 1, S, (NH + 2 * NKV) * DH),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=D,
                    memory_config=mc,
                )
            except Exception as e:  # noqa: BLE001
                print(f"RES {pname}: cannot place input ({str(e)[:120]})")
                continue
            for vname, (r, c, w) in variants.items():
                if pname != "DRAM in" and vname not in ("full", "compute only (no unit read/write)", "read only"):
                    continue
                hop.READER_KERNEL, hop.WRITER_KERNEL = r, w
                setattr(hop, cattr, c)

                def fn():
                    return hop.nlp_create_qkv_heads_norm_headsplit(
                        x,
                        GQ,
                        GK,
                        SC,
                        EP,
                        num_heads=NH,
                        num_kv_heads=NKV,
                        memory_config=DR,
                        rot_cos=cos,
                        rot_sin=sin,
                        trans_mat=T,
                        q_dtype=ttnn.bfloat8_b,
                        kv_dtype=ttnn.bfloat8_b,
                        norm_eps=EPS,
                    )

                for _ in range(2):
                    [ttnn.deallocate(t) for t in fn()]
                ttnn.synchronize_device(D)
                n = 8
                tid = ttnn.begin_trace_capture(D, cq_id=0)
                outs = [fn() for _ in range(n)]
                ttnn.end_trace_capture(D, tid, cq_id=0)
                ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
                    ts.append((time.perf_counter() - t0) / n * 1e6)
                ttnn.release_trace(D, tid)
                [ttnn.deallocate(t) for o in outs for t in o]
                print(f"RES B{B} {pname:8s} {vname:36s} {statistics.median(ts):7.1f} us/call", flush=True)
            ttnn.deallocate(x)
    finally:
        hop.READER_KERNEL, hop.WRITER_KERNEL = reader, writer
        setattr(hop, cattr, compute)
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
