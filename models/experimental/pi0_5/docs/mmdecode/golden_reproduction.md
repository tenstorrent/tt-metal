# Golden Baseline Reproduction — pi05 matmul_decode 3-stage (Blackhole P150)

Reproduces BOTH frozen golden baselines on the current binary and compares to the
frozen golden CSVs in `deep-work/mmd13_csvs/`.

## Provenance / Run Record

- **Host / arch**: tt-quietbox, Blackhole **P150**, single device, one device job at a time.
- **tt-metal commit**: `e4500c1fae97c103b16fc24fc7010b852992a9e6` (FORK
  `/home/ttuser/salnahari/tt-metal-matmul_decode`). Source tree clean at start and end.
- **Phase A (GOLDEN NATIVE)**: profiled on the **CLEAN** committed binary (no patch —
  native is upstream / patch-invariant). Harness
  `profile_native_explicit_sweep_stages.py` (`test_natexp_sweep_{siglip,vlm,denoise}`,
  one projection per tracy subprocess via `ONLY_PROJ`).
- **Phase B (GOLDEN-LEVER mmd)**: profiled on the **deep-plan_15-PATCHED** binary
  (`git apply deep-work/deep-plan_15_fork.patch` → 9 modified + 3 new matmul_decode
  files; rebuilt with `./build_metal.sh`, tracy ON). Harness
  `profile_unified_mmsweep_stages.py::test_profile` (the `MatmulDecodeLinear` blocked
  wrapper = the golden adopted chunked-M=32 / lever schemes), `ONLY_STAGE` + `ONLY_PROJ`.
- **op-support-count = 20000** on EVERY tracy run (the proven-safe value; only sets
  profiler buffer capacity, not timing — reproduced KERNEL numbers are identical).
- **Metric** = DEVICE KERNEL DURATION (tracy col-20) via
  `extract_perf.py METRIC=KERNEL`. Native: `EXTRACT_MODE=natexp` (min µs/call over the
  10 swept explicit configs). mmd: `EXTRACT_MODE=mmsweep MMSWEEP_OP=mmd` (min-of-N,
  N_ITERS=5).
- **Back-test gate**: `EXTRACT_MODE=mmsweep METRIC=KERNEL MMSWEEP_OP=mmd` on frozen
  `mmd_SigLIP_qkv.csv` → **`SigLIP.qkv mmd_us/fwd=45.969`** — PASSED before any device
  work. Re-confirmed on the patched binary by re-profiling SigLIP.qkv → **46.362**
  (+0.85%, calls/fwd=8.0 — scheme intact).
- All 24 tracy subprocesses returned rc=0; no device wedge, no resets needed.

Tolerance: **≤ ±3 % = reproduced**.

## Table 1 — GOLDEN NATIVE (best-explicit native, KERNEL µs/call) — CLEAN binary

| stage.proj    | frozen golden µs | reproduced µs | %Δ      | match (≤±3%) |
|---------------|------------------|---------------|---------|--------------|
| DENOISE.gate  |  34.637          |  34.623       | -0.04 % | YES          |
| DENOISE.up    |  34.633          |  34.702       | +0.20 % | YES          |
| DENOISE.down  |  35.190          |  35.206       | +0.05 % | YES          |
| SigLIP.qkv    |  55.441          |  55.754       | +0.56 % | YES          |
| SigLIP.o      |  20.433          |  20.393       | -0.20 % | YES          |
| SigLIP.fc1    |  41.367          |  41.298       | -0.17 % | YES          |
| SigLIP.fc2    |  42.085          |  42.027       | -0.14 % | YES          |
| VLM.qkv       |  38.377          |  38.376       | -0.00 % | YES          |
| VLM.o         |  37.354          |  37.366       | +0.03 % | YES          |
| VLM.gate      | 254.609          | 254.570       | -0.02 % | YES          |
| VLM.up        | 254.685          | 254.304       | -0.15 % | YES          |
| VLM.down      | 236.716          | 236.743       | +0.01 % | YES          |

**12 / 12 within ±3 %.** Max divergence +0.56 % (SigLIP.qkv). Every shape selected the
**same `best_cfg`** as golden (e.g. SigLIP.qkv `nc9nr8bw2sb1x8`, VLM.gate `nc8nr9bw8sb1x8`).

## Table 2 — GOLDEN-LEVER mmd (MatmulDecodeLinear, KERNEL µs/block-forward) — PATCHED binary

| stage.proj    | frozen golden µs | reproduced µs | %Δ      | match (≤±3%) | calls/fwd |
|---------------|------------------|---------------|---------|--------------|-----------|
| DENOISE.gate  |   9.898          |   9.909       | +0.11 % | YES          | 2         |
| DENOISE.up    |   9.832          |   9.835       | +0.03 % | YES          | 2         |
| DENOISE.down  |  22.757          |  23.031       | +1.20 % | YES          | 2         |
| SigLIP.qkv    |  45.969          |  46.362       | +0.85 % | YES          | 8         |
| SigLIP.o      |  38.578          |  38.348       | -0.60 % | YES          | 8         |
| SigLIP.fc1    |  65.150          |  64.631       | -0.80 % | YES          | 8         |
| SigLIP.fc2    | 131.445          | 131.539       | +0.07 % | YES          | 8         |
| VLM.qkv       |  56.041          |  56.354       | +0.56 % | YES          | 9         |
| VLM.o         |  56.396          |  56.908       | +0.91 % | YES          | 9         |
| VLM.gate      | 263.957          | 263.539       | -0.16 % | YES          | 9         |
| VLM.up        | 264.002          | 263.939       | -0.02 % | YES          | 9         |
| VLM.down      | 405.755          | 405.264       | -0.12 % | YES          | 18        |

**12 / 12 within ±3 %.** Max divergence +1.20 % (DENOISE.down). `calls/fwd` matches the
golden scheme exactly for every projection (the chunked-M=32 / lever block counts are
intact), confirming the deep-plan_15 lever schemes were reproduced, not merely the timing.

## Verdict

**Both golden legs reproduced within tolerance.**

- **GOLDEN NATIVE**: 12/12 within ±3 % (max |Δ| = 0.56 %). Reproduced on the clean
  committed binary, as expected (native is patch-invariant). Identical best-config
  selection per shape.
- **GOLDEN-LEVER mmd**: 12/12 within ±3 % (max |Δ| = 1.20 %, DENOISE.down). Reproduced
  on the deep-plan_15-patched + rebuilt binary, with identical lever block schemes
  (calls/fwd match) across all 12 projections.

**No shape diverges > 3 %.** All residual deltas (sub-1.2 %) are consistent with normal
run-to-run KERNEL-duration jitter on identical kernels/schemes — there is no evidence of
binary drift, scheme mismatch, or dtype change. The single largest mmd delta (DENOISE.down,
+1.20 %) is the smallest-K / smallest fixed-cost shape, where per-call jitter contributes a
larger relative share; still comfortably inside tolerance.

## Final state

- Source tree: **clean at e4500c1f, 0 dirty** (patch reverted via `git checkout -- .` +
  `git clean -fd` of the 3 patch-added files). HEAD never moved off e4500c1f.
- Binary on disk: **the deep-plan_15-PATCHED build** (`build_Release/.so` rebuilt
  2026-06-12 20:37, tracy ON). Per task allowance this is acceptable as long as it is
  clearly stated — the SOURCE tree is the clean baseline; the installed .so reflects the
  patched matmul_decode.
- Device: **idle**, no profiling processes.

Repro artifacts: `deep-work/golden_repro/nat/*.csv` (12), `deep-work/golden_repro/mmd/*.csv` (12),
per-run logs alongside; driver `deep-work/run_golden_repro.sh`.
