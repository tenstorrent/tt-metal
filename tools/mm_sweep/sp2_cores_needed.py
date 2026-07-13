#!/usr/bin/env python3
# SP2: compute-core floor. cores_needed = min(M,N) * BW / R_percore.
# Regime A (N>=M): read in1 (shardable, contiguous) -> BW=510. Regime B (M>N): read in0
# (interleaved) -> BW=419. Compute-bound if cores_needed > GRID (110).
import sys

R = float(sys.argv[1]) if len(sys.argv) > 1 else 1.7e12  # per-core FLOP/s
GRID = 110
BW_A, BW_B = 510e9, 419e9  # measured ceilings (SP1 contiguous, SP5 interleaved)

# FLUX/LTX skinny shapes from bh_skinny_results.md
SHAPES = [
    (32, 2048, 32),
    (32, 256, 6144),
    (32, 2048, 512),
    (32, 2048, 1536),
    (1216, 4096, 32),
    (32, 2048, 2048),
    (32, 6144, 1536),
    (32, 6144, 2304),
    (32, 6144, 3072),
    (4864, 4096, 32),
    (32, 6144, 6144),
    (32, 6144, 9216),
    (64, 6144, 1536),
    (64, 15360, 1536),
    (64, 4608, 6144),
    (64, 6144, 4608),
    (64, 6144, 9216),
    (512, 6144, 128),
    (128, 6144, 768),
    (128, 15360, 768),
    (1024, 6144, 128),
    (2048, 6144, 128),
    (128, 6144, 2304),
    (128, 2304, 6144),
    (4096, 6144, 128),
    (128, 6144, 4608),
    (8192, 6144, 128),
    (16384, 6144, 128),
    (512, 6144, 1536),
]
# read-core floors (cores to reach ~peak read): Regime A contiguous 16KB bursts saturate ~8-16
# (SP1); Regime B interleaved 2KB bursts need ~32 for ~405 (SP5).
READ_CORES = {"A": 16, "B": 32}
print(
    f"R_percore = {R/1e12:.2f} TFLOP/s   compute-bound threshold min(M,N) > {GRID*R/BW_A:.0f}(A)/{GRID*R/BW_B:.0f}(B) rows\n"
)
print(f"{'shape':>20} {'reg':>4} {'minMN':>6} {'compute_c':>9} {'read_c':>6} {'bind':>5} {'verdict':>13}")
for M, K, N in SHAPES:
    regime = "A" if N >= M else "B"
    BW = BW_A if regime == "A" else BW_B
    minmn = min(M, N)
    comp = minmn * BW / R
    rd = READ_CORES[regime]
    if comp > GRID:
        verdict = "COMPUTE-bound"
        bind = GRID
    else:
        bind = max(comp, rd)
        verdict = "compute-set" if comp >= rd else "read-set"
    print(f"{f'{M}x{K}x{N}':>20} {regime:>4} {minmn:>6} {comp:>9.0f} {rd:>6} {bind:>5.0f} {verdict:>13}")
