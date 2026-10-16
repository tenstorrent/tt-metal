# The plain minimal_matmul's end-of-block epilogue (copy_and_pack_block: intermediate -> out, one DST acquire per tile):
# what it costs and what batching it saves. Variants patch compute_metal2.cpp (restored afterwards) and run
# bench_mm_ablate.py's child under the device profiler; reads / writes can be skipped as in bench_mm_gap_ladder.py.
#   full / compute only: as shipped
#   batched: the epilogue copies up to 8 tiles per DST acquire (half of DST) instead of 1
#   no epilogue: out reserved and pushed, nothing packed (upper bound; output garbage)
# Usage: bench_mm_epilogue.py <preset> <batch>   (MM_BLOCKS overrides the blocks)
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR, patch  # noqa: E402
from device_kernel_us import kernel_us  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")
OLD_LOOP = """    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_dfb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_dfb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        dfb_out.push_back(N_block_tiles);
    }
}"""
BATCHED = """    uint32_t tile_id = 0;
    constexpr uint32_t CH = 8;  // DST tiles per acquire (half of DST, bf16)
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += CH) {
            const uint32_t nt = (N_block_tiles - n0 < CH) ? (N_block_tiles - n0) : CH;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nt; j++) {
                copy_tile(in_dfb, tile_id + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nt; j++) {
                pack_tile(j, out_dfb);
            }
            tile_regs_release();
            tile_id += nt;
        }
        dfb_out.push_back(N_block_tiles);
    }
}"""
NONE = """    for (uint32_t m = 0; m < M_block_tiles; m++) {
        dfb_out.push_back(N_block_tiles);
    }
    (void)fused_act_dst_id;
}"""
VARIANTS = {  # name: (epilogue body, reads, writes)
    "full": (None, 1, 1),
    "batched epilogue": (BATCHED, 1, 1),
    "compute only": (None, 0, 0),
    "compute only, batched epilogue": (BATCHED, 0, 0),
    "compute only, no epilogue": (NONE, 0, 0),
}


def main():
    name, B = sys.argv[1], sys.argv[2]
    orig_c, orig_h = open(COMPUTE).read(), open(HDR).read()
    assert orig_c.count(OLD_LOOP) == 1
    try:
        for vname, (body, r, w) in VARIANTS.items():
            open(COMPUTE, "w").write(orig_c if body is None else orig_c.replace(OLD_LOOP, body))
            open(HDR, "w").write(patch(orig_h, r, r, w))
            d = tempfile.mkdtemp(prefix="mm_epi_")
            env = dict(
                os.environ,
                TT_METAL_CACHE=os.path.join(d, "cache"),
                TT_METAL_DEVICE_PROFILER="1",
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="40000",
                TT_METAL_PROFILER_DIR=os.path.join(d, "prof"),
            )
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", name, B, os.path.join(d, "o.pt")],
                env=env, capture_output=True, text=True, timeout=600,
            )  # fmt: skip
            us = kernel_us(os.path.join(d, "prof")) if p.returncode == 0 else {}
            if not us:
                err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "TT_FATAL" in l][:2]
                print(f"RES {name} B{B} {vname:34s} FAILED {err}", flush=True)
                continue
            per = sorted(us.values(), key=len)[-1]
            per = sorted(per)[len(per) // 2]
            print(f"RES {name} B{B} {vname:34s} {per:8.1f} us/call device", flush=True)
    finally:
        open(COMPUTE, "w").write(orig_c)
        open(HDR, "w").write(orig_h)


if __name__ == "__main__":
    main()
