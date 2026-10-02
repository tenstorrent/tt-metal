# Device-kernel-time ablation ladder for minimal_matmul at the model's calls, for splitting each matmul's gap to the
# achievable bound (profile_page.py) into categories: data movement not hidden (full - compute only), extra passes
# (fused SwiGLU's partial-sum add; the plain path's end-of-block intermediate -> out copy), SFPU not hidden (fused
# SwiGLU), and the compute kernel's skeleton (every CB handshake, init, DST acquire / release and pack kept, the
# matmul_block math removed). Each variant patches compute_metal2.cpp and matmul_dataflow_common_metal2.hpp (restored
# afterwards) and runs bench_mm_ablate.py's child under the device profiler in its own process with a fresh JIT cache;
# device_kernel_us.py reads the per-call device time.
# Usage: bench_mm_gap_ladder.py <preset> <batch>   (presets as bench_mm_ablate.py: ff13fused, qkv, ff2, wo; MM_BLOCKS
#        overrides the blocks)
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR, patch  # noqa: E402
from device_kernel_us import kernel_us  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")
SFPU_CALL = "PACK((llk_minimal_matmul_swiglu(gate, gate + 1, gate)));"
ADD_GUARD = "            if (add_partials) {"
COPY = "            copy_tile(in_dfb, tile_id, fused_act_dst_id /*dst*/);"
MM_CALL = re.compile(r"(\n\s*)matmul_block\(\n")  # the calls, not matmul_block_init
# name: (sfpu, add partials, end-of-block copy, matmul math, reads, writes)
FUSED = {
    "full": (1, 1, 1, 1, 1, 1),
    "no partials add": (1, 0, 1, 1, 1, 1),
    "no add, no SFPU": (0, 0, 1, 1, 1, 1),
    "compute only, no add, no SFPU": (0, 0, 1, 1, 0, 0),
    "skeleton": (0, 0, 1, 0, 0, 0),
}
PLAIN = {
    "full": (1, 1, 1, 1, 1, 1),
    "compute only": (1, 1, 1, 1, 0, 0),
    "compute only, no copy": (1, 1, 0, 1, 0, 0),
    "skeleton": (1, 1, 0, 0, 0, 0),
}


def patch_compute(src, sfpu, add, copy, mm):
    assert src.count(SFPU_CALL) == 1 and src.count(ADD_GUARD) == 1 and src.count(COPY) == 1
    assert len(MM_CALL.findall(src)) == 2, MM_CALL.findall(src)
    if not sfpu:
        src = src.replace(SFPU_CALL, "(void)gate;")
    if not add:
        src = src.replace(ADD_GUARD, "            if (0 && add_partials) {")
    if not copy:
        src = src.replace(COPY, "")
    if not mm:
        src = MM_CALL.sub(r"\1if (0) matmul_block(\n", src)
    return src


def main():
    name, B = sys.argv[1], sys.argv[2]
    variants = FUSED if name == "ff13fused" else PLAIN
    orig_c, orig_h = open(COMPUTE).read(), open(HDR).read()
    try:
        for vname, (sfpu, add, copy, mm, r, w) in variants.items():
            open(COMPUTE, "w").write(patch_compute(orig_c, sfpu, add, copy, mm))
            open(HDR, "w").write(patch(orig_h, r, r, w))
            d = tempfile.mkdtemp(prefix="mm_ladder_")
            env = dict(
                os.environ,
                TT_METAL_CACHE=os.path.join(d, "cache"),
                TT_METAL_DEVICE_PROFILER="1",
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="40000",
                TT_METAL_PROFILER_DIR=os.path.join(d, "prof"),
            )
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", name, B, os.path.join(d, "o.pt")],
                env=env, capture_output=True, text=True, timeout=1200,
            )  # fmt: skip
            us = kernel_us(os.path.join(d, "prof")) if p.returncode == 0 else {}
            if not us:
                err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "rror" in l][:3]
                print(f"RES {name} B{B} {vname:32s} FAILED {err}", flush=True)
                continue
            per = sorted(us.values(), key=len)[-1]  # the 8-call trace
            per = sorted(per)[len(per) // 2]
            print(f"RES {name} B{B} {vname:32s} {per:8.1f} us/call device", flush=True)
    finally:
        open(COMPUTE, "w").write(orig_c)
        open(HDR, "w").write(orig_h)


if __name__ == "__main__":
    main()
