# Issue #59851 data: Gemma4 prefill matmul ns/token vs chunk size

Work-in-progress data, not for merge. Drop this directory before opening a PR.

Measured 2026-10-10 on bh-glx-120-b09u08 (Blackhole galaxy), branch `mmanzoor/gemma4-matmul-chunk-sweep`
(main 6ba9d172b63 + the two sweep scripts). Single chip (device 0), one CP8/TP4 device's shapes, production code path.

## Files

- `summary.csv` / `summary.txt`: one row per (op, C, attention fidelity mode), pooled over 3 independent profiler
  runs x 10 warm calls. Median/IQR/min/max device kernel ns, ns per computed token, FPU util (full grid and used cores),
  weight GB/s, program config, compute config, PCC/rel-RMSE vs torch.
- `ns_per_token.png`: per-op ns/token and FPU util vs C.
- `real_weights_summary.csv`: real TP-rank-0 weights (layer 0 sliding, layer 5 global) at C=1K..16K. Within 1.3% of
  random weights at every point, so random weights are a valid substitute.
- Raw data (not in git, the repo's large-file hook caps files at 500 KB): per-run meta.jsonl, ops_perf_results CSVs
  (per-call timings), raw.csv, run status, layer-perf manifests and the sz2048 crash log are in
  `/data/mmanzoor/matmul_chunk_sweep/bundle_59851.tar.gz` (2 MB; also unpacked next to it) on the NFS share
  10.32.13.1:/data. Full tracy dirs (26 GB) stay in /data/mmanzoor/matmul_chunk_sweep.
- `run_sweep.sh`, `run_followups.sh`, `run_fallback.sh`: exact commands used (paths assume the original box).
- `extract_tp0_weights.py`: builds tp0_weights.pt for `--weights` from the HF snapshot.

Reproduce (single chip; ~4 min per run):

    python -m tracy -r -p -o <dir> models/demos/gemma4_d_p/scripts/matmul_chunk_sweep.py \
        --meta <dir>/meta.jsonl --attn-fidelity prod lofi hifi2 --reps 10
    python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_report.py <dir> [<dir2> ...] --out <out>

## Per-device shapes (TP4, bfp8 weights, bf16 activations, rows = C/8)

sliding_qkv 5376x4096, global_qk 5376x4608, sliding_o 2048x5376, global_o 4096x5376, gate/up/down 5376x5376.

## Configuration regimes on main (all change with C, so the curves have steps)

| C | program config | attention compute | MLP compute |
|---|---|---|---|
| <= 2048 | 1D in0-mcast, in0 width-sharded in L1 (64-84 cores; down 42) | HiFi2, fp32 acc, L1 acc | LoFi, fp32 acc, L1 acc |
| 2304-3840 | explicit 2D mcast (72-120 cores) | HiFi2, fp32 acc | LoFi, fp32 acc |
| 4096-10240 | explicit 2D mcast | LoFi, fp32 acc | LoFi, fp32 acc |
| >= 12288 | ttnn default (per_core_M > 4 does not fit L1) | HiFi2, bf16 acc (ttnn default for bf16 x bfp8) | LoFi, bf16 acc |

Valid model chunks: multiple of 1024 that divides the context (1K, 2K, 4K, 8K, 16K at 256K).

## Findings so far

- C <= 2048 is weight-bandwidth bound (300-390 GB/s of weight reads); time is nearly flat, so ns/token ~ 1/C.
- 1D -> 2D switch at C=2304 is a regression for every op (e.g. up 81 -> 120 us).
- Explicit 2D keeps improving to C=10240. At C=8192 (service default) 2D uses 96 of 120 cores (32 M tiles over 10
  rows -> per_core_M=4 -> 8 rows), so 8192 ~= 7168 per token.
- At C >= 12288 attention falls back to HiFi2/bf16-acc default and gets 1.5x slower per token than 10240.
- Fixed-fidelity attention: LoFi vs HiFi2 differs by only 0-10% at C <= 2048 (bandwidth bound), 1.4-1.7x at 8K.
- Production ns/token (C=2048 / 4096 / 8192 / 10240): sliding_qkv 359/213/152/129, global_qk 369/231/155/132,
  sliding_o 194/134/97/86, global_o 309/213/145/126, gate 398/302/217/176, up 317/236/150/123, down 400/263/169/143.
- All PCC >= 0.9988 (lowest: gate under the bf16-acc default at C >= 12K, 0.9988-0.9994).

## Not done yet

1. LoFi/fp32-acc floor at C >= 12288 (`run_fallback.sh`, `--fallback-lofi`): never ran. Needed to know where the true
   low stable region is, since main's configs cannot show it past 10240.
2. Layer-time shares (`test_prefill_layer_perf_chunk_n` on 8x4, sz2048/4096/8192/16384): sz2048 aborted at close with
   profiler TT_FATAL "End marker found without a corresponding start marker" (profiler.cpp:2249); sz4096 ran but its
   tracy post-processing was cut off (empty ops CSV). 8192/16384 not run.
3. Recommendation, issue write-up and plot polish.
