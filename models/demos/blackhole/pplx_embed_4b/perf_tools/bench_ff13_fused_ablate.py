# Fused-SwiGLU FF1+FF3 minimal_matmul (pack-thread SwiGLU in the last K block): where does the time go against the
# plain packed matmul? Variants patch compute_metal2.cpp (skip or double the pack-thread SFPU SwiGLU, skip the last K
# block's dest-reuse add of the partial sums) and optionally matmul_dataflow_common_metal2.hpp (skip reads / writes,
# as bench_mm_ablate.py), both restored afterwards; each variant runs bench_mm_ablate.py's child (preset ff13fused:
# in0 bfp8 L1, packed w13 bfp4 DRAM interleaved, bfp8 out in DRAM) in its own process with a fresh JIT cache.
# Usage: bench_ff13_fused_ablate.py <batch>   blocks default to the model's (bs8 8,20,8 1x8, else 4,20,8 1x8);
#        MM_BLOCKS overrides; ABL_ONLY="full|no SFPU" picks variants.
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_mm_ablate import HDR, patch  # noqa: E402

COMPUTE = os.path.join(os.path.dirname(HDR), "compute_metal2.cpp")
SFPU_CALL = "PACK((llk_minimal_matmul_swiglu(gate, gate + 1, gate)));"
ADD_GUARD = "            if (add_partials) {"
VARIANTS = {  # name: (SFPU passes, add partials, read in0, read in1, write out)
    "full": (1, 1, 1, 1, 1),
    "no SFPU": (0, 1, 1, 1, 1),
    "SFPU x2": (2, 1, 1, 1, 1),
    "no partials add": (1, 0, 1, 1, 1),
    "no SFPU, no add": (0, 0, 1, 1, 1),
    "compute only": (1, 1, 0, 0, 0),
    "compute only, no SFPU": (0, 1, 0, 0, 0),
    "compute only, no SFPU, no add": (0, 0, 0, 0, 0),
}
VARIANTS = {k: v for k, v in VARIANTS.items() if not os.getenv("ABL_ONLY") or k in os.getenv("ABL_ONLY").split("|")}


def patch_compute(src, sfpu, add):
    assert src.count(SFPU_CALL) == 1 and src.count(ADD_GUARD) == 1
    src = src.replace(SFPU_CALL, " ".join([SFPU_CALL] * sfpu) or "(void)gate;")
    return src.replace(ADD_GUARD, f"            if ({add} && add_partials) {{")


def main():
    B = int(sys.argv[1])
    M, K, N = B * 512, 2560, 19456
    blocks = os.getenv("MM_BLOCKS") or ("8,20,8,1,8" if B == 8 else "4,20,8,1,8")
    print(f"RES ff13fused B{B}: M={M} K={K} N={N} blocks={blocks}", flush=True)
    orig_c, orig_h = open(COMPUTE).read(), open(HDR).read()
    ref = full_t = None
    try:
        for vname, (sfpu, add, r0, r1, w) in VARIANTS.items():
            open(COMPUTE, "w").write(patch_compute(orig_c, sfpu, add))
            open(HDR, "w").write(patch(orig_h, r0, r1, w))
            d = tempfile.mkdtemp(prefix="ff13_abl_")
            env = dict(os.environ, TT_METAL_CACHE=os.path.join(d, "cache"), MM_BLOCKS=blocks)
            out_path = os.path.join(d, "out.pt")
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_mm_ablate.py"), "--child", "ff13fused", str(B), out_path],
                env=env, capture_output=True, text=True, timeout=1200,
            )  # fmt: skip
            res = [l for l in p.stdout.splitlines() if l.startswith("CHILD")]
            if not res:
                err = [l for l in (p.stdout + p.stderr).splitlines() if "TT_THROW" in l or "rror" in l][:4]
                print(f"RES {vname:32s} FAILED {err}", flush=True)
                continue
            t = float(res[0].split()[1])
            o = torch.load(out_path)
            if vname == "full":
                ref, full_t = o, t
            pcc = torch.corrcoef(torch.stack([o, ref]))[0, 1].item() if ref is not None else float("nan")
            dt = f"{t - full_t:+7.1f}" if full_t else "      —"
            print(
                f"RES {vname:32s} {t:8.1f} us  Δ {dt}  {2 * M * K * N / t / 1e6:6.1f} TFLOP/s  pcc vs full {pcc:.3f}",
                flush=True,
            )
    finally:
        open(COMPUTE, "w").write(orig_c)
        open(HDR, "w").write(orig_h)


if __name__ == "__main__":
    main()
