# SP2 — per-core compute rate & compute-core floor

**Goal:** how many cores keep compute time ≤ read time (so the op stays DRAM-bound, not compute-bound)?
**Scope (fixed):** bf16 inputs, bf16 output, HiFi2, fp32 accumulation. (bf8 / LoFi are out of scope —
not explored.)

## Per-core compute rate R (measured — REVISED, higher than first pass)

**First pass (ttnn.matmul, MISLEADING):** default `ttnn.matmul` HiFi2/bf16/fp32-acc on compute-bound
squares plateaued at only ~150 TFLOP/s (49.5 % of 304, ~1.37 TF/core). But that is a **full-op /
dataflow-limited** number, NOT the compute-engine ceiling — ttnn.matmul's default program config
doesn't feed the FPU at peak. (`minimal_matmul` would be the right tool but it HANGS on this tree.)

**Compute-only microbench (the real ceiling).** Ran minimal_matmul's ACTUAL `compute.cpp` kernel on a
**single core** with DRAM/NoC stubbed out (feeder pushes garbage in0/in1 blocks, drain pops output),
sweeping block sizes. Kernel-time via device profiler. This isolates unpack+FPU+pack:

| block (mb·kb·nb, subblock) | TF/core | % of 2.76 single-core peak |
|---|---:|---:|
| 4·8·4 sb2×2 | 2.18 | 79 |
| 8·8·8 sb2×4 | 2.22 | 80 |
| 4·16·4 sb2×2 | 2.36 | 85 |
| 4·32·4 sb2×4 | 2.48 | 90 |
| **2·64·2 sb2×2** | **2.51** | **91** |

**Compute ceiling ≈ 2.5 TFLOP/s per core = ~90 % of the 2.76 TF/core theoretical peak ⇒ ~275 TFLOP/s
aggregate (90 % of 304).** The lever is **deep K-block** (kb 16→64 lifts 79 %→91 %): amortizes the
per-output-block pack + reconfig over more matmul_block calls. Small M/N blocks (to fit deep-K in L1)
are fine. This **overturns the "HiFi2 is unpack-bound at ~50 %" belief** — that was a dataflow-
confounded full-op number; the compute engine reaches ~90 % with deep-K blocking.

**Use R_percore ≈ 2.4 TFLOP/s** (achievable with deep-K blocks) for the floor.

## The floor formula
For a low-AI shape the dominant read operand is the larger of in0 (M·K) / in1 (K·N). The arithmetic
intensity **against that operand simplifies to `min(M,N)` FLOP/byte** (bf16, the 2-byte and 2-FLOP
factors cancel):  AI_dom = 2MKN / (2·K·max(M,N)) = min(M,N).

Not-compute-bound requires `compute_time ≤ read_time`, i.e.
**cores_needed = min(M,N) × BW / R_percore**, with BW = 510 GB/s (Regime A, sharded in1) or 419
(Regime B, interleaved in0). Compute-bound (can't saturate DRAM even at 110 cores) when
`min(M,N) > 110·R/BW` ≈ **518 rows (A) / 630 rows (B)** at R=2.4 TF/core.

The op's actual core budget = **max(compute_cores, read_cores)**, where read_cores ≈ 16 (Regime A,
16 KB bursts, SP1) or ≈ 32 (Regime B, 2 KB bursts, SP5).

## Result — FLUX/LTX skinny shapes (binding core count, R=2.4 TF/core)
| min(M,N) | regime | compute_c | read_c | **binding** | note |
|---:|:--:|---:|---:|---:|:--|
| 32 (M=1 tile) | A | 7 | 16 | **16** | read-set |
| 32 (M=1 tile) | B | 6 | 32 | **32** | read-set |
| 64 | A | 14 | 16 | **16** | read-set |
| 128 | B | 22 | 32 | **32** | read-set |
| 128 | A | 27 | 16 | **27** | compute-set |
| 512 (512×6144×1536) | A | 109 | 16 | **109** | read-set (barely fits!) |

**Every FLUX/LTX skinny shape needs ≤ 32 cores** (min=128 Regime-A ~27), and with the corrected
R **nothing in the set is compute-bound** — the compute-bound threshold rises to min(M,N) > ~518(A)/
630(B). **Almost all shapes are READ-SET**: compute (~90 % util, ~2.4 TF/core) is so fast that the
DRAM read is the bottleneck for essentially every low-AI shape. So compute is NOT the limiter, and
**~80–95 of the 110 cores are free** — directly relevant to the NoC-headroom / integration concern.

## Implications for the op design
1. **Right-size the core grid per shape** to `max(compute_cores, read_cores)` instead of always using
   110. Small M-shapes: ~16–32 cores. This minimizes NoC pressure (the SP5 integration concern) and
   keeps reader==consumer clean.
2. **Read is the binding constraint for essentially all low-AI shapes** — with R=2.4 TF/core, compute
   is fast enough that only min(M,N) ≳ 27-tile Regime-A shapes are even compute-set, and none in the
   FLUX/LTX set are compute-*bound*.
3. **Regime A is cheap on cores** (read_c ≈ 16, few readers saturate 510) → most cores free for
   compute if M grows. **Regime B needs ≥32 readers** and can't exceed ~419.
4. Above min(M,N) ≈ 518 (A) / 630 (B), the shape is compute-bound — outside this op's DRAM-BW mandate;
   hand off to the reuse matmul (plan constraint #13).

## Caveat / flag
- **R is measured by the compute-only microbench** (real `compute.cpp`, 1 core, stubbed DRAM/NoC) —
  the authoritative number. `ttnn.matmul` (150/49 %) is kept only as a reference for how much a
  suboptimal full-op dataflow leaves on the table; it is NOT the ceiling.
- `minimal_matmul` currently hangs on this branch, but **that is not a concern for this work** — we are
  building a functionally separate op with its own kernels and program factory, so the existing op's
  state is irrelevant. (The compute-only harness deliberately reuses only its `compute.cpp` kernel.)
- R ≈ 2.4 TF/core is the deep-K rate. For M=1-tile shapes the per-core output block is thin, so
  effective compute may dip slightly → compute_c a touch higher; but those shapes are read-set with a
  large margin (compute floor 6–7 vs read 16–32), so the conclusion holds.

## Design guideline
- The op's compute kernel must use **deep K-blocks (kb ≥ 16, ideally 32)** to reach the ~90 % ceiling;
  shallow K-blocks give only ~79 %. Small M/N blocks are fine (needed to fit deep-K in L1).
- Subblock 2×4 / 4×2 (8 tiles, needs `dst_full_sync_en`) is marginally better than 2×2; 2×2 is safe.

## Artifacts
- `tests/tt_metal/tt_metal/perf_microbenchmark/sp2_compute_only/` — **compute-only microbench**: runs
  minimal_matmul's real `compute.cpp` on 1 core, DRAM/NoC stubbed (feeder+drain). Build target
  `test_compute_only`. Run: `TT_METAL_DEVICE_PROFILER=1 test_compute_only --mb M --kb K --nb N
  --sbh h --sbw w --knum 100`; parse with `parse_kernel_bw.py <csv> <flop=2*mb*nb*kb*knum*32^3>`.
  Reusable to measure the compute ceiling across block sizes (bf16/HiFi2 fixed scope).
- `tools/mm_sweep/sp2_cores_needed.py <R_percore>` — applies the floor formula to the shape list.
- `tools/mm_sweep/sp2_compute_rate.py` / `sp2_ttnn_tuned.py` — the ttnn.matmul measurement (kept for
  reference; shows the misleading dataflow-limited ~150).
