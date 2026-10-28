# Prefill stage 2 (op topology, E9.2): work log

Run folder: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/02-topology/` (below: `$T`): `scripts/`,
`logs/`, `run_ab/` (harness records), `dump/` (gate replay outputs), `analysis/`, `tracy/`. Host
sjc-snva-tp100, card 0 (P150a), grid 12 x 10 (`EMBEDDING_GEMMA2_GRID=12x10`) unless a line says
13 x 10. Precision `sdpa_nofp32acc` (unchanged). Every device command ran through `$T/scripts/run.sh`
(card-free check polling every 60 s, env dump, `timeout -s KILL`, `TT_METAL_OPERATION_TIMEOUT_SECONDS=60`
(120 for Tracy), `__start` with commit and dirty files / `__end wall_s` / `__tt_exit`), one at a time.
The card was never held by another process (`$T/logs/card_wait.log` absent). Environment: as in
`../01-baseline/work_log.md` ("Environment").

## Commits

| commit | content |
|---|---|
| `b747b794a1` | commit 1, docs only: `target_contract.json` `checkout.commit` -> `ec3ca0e1c6` and `compute_grid_note` (orchestrator-approved); E9.1-r2b review minors in `01-baseline/README.md` and `work_log.md` |
| `7cef532759` | topology rewrites: `tt/masks.py`, `tt/per_layer_input.py`, `tt/pooler.py`, `tt/text_model.py` (`encode_embeds`), `tt/model.py` (uses it; `l2_normalize` from `pooler.py`); harness `tests/perf/prefill.py` (`EMBEDDING_GEMMA2_PREFILL_DUMP`, `model_env` in run_info) |
| (this docs commit) | `doc/prefill/02-topology/` |

Code identity of the device runs: every run from `smoke_new` (13:26:09) on started at HEAD
`b747b794a1` with the six files of `7cef532759` modified in the working tree (log headers). Their
mtimes are 13:24:10-13:26:02, before the first of those runs, and they were committed unchanged as
`7cef532759` (14:11) **[M: `stat`, `git diff --name-only b747b794a1 7cef532759`]**. Runs after 14:11
log HEAD `7cef532759`, clean tree.

## Commands (chronological)

| time | label (`$T/logs/<label>.log`) | command | result |
|---|---|---|---|
| (CPU) | before tables | `python3 $T/scripts/op_seq.py roofline/capture/out/text_b1_sB/ops_perf_results.csv --layers 0,5 --md $T/analysis/before/b1_sB.md --json ...` (the stage-1 12 x 10 profiles, `a903d517c6`) | 10 buckets: 645-646 ops; per layer 26 ops, 2 DM-only |
| 13:18:37 | `probe_windowed_a` | `python $T/scripts/probe_windowed.py 128:96,512:384,2048:1536,2048:2048,2048:1000,8192:6144` at `ec3ca0e1c6` | windowed SDPA bitwise equal to the dense mask on valid rows, 1.0-3.1x slower (README) |
| 13:20:20 | `probe_mask_a` | `python $T/scripts/probe_mask.py 1024,2048,8192,16384` at `ec3ca0e1c6` | v1 (fused key bias) and v2 (band + full) bitwise equal to the stage-1 build; v2 4.3x faster at 8192 / 16384 |
| 13:26:09 | `smoke_new` | `EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 ... pytest prefill.py` (quick buckets) | 6 passed |
| 13:27:37-13:56:13 | `chain_ab` (`ab_<label>`) | `$T/scripts/chain_ab.sh`: per label `EMBEDDING_GEMMA2_PREFILL_SHAPES=1 EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 EMBEDDING_GEMMA2_PREFILL_DUMP=$T/dump/<label> pytest prefill.py`; labels ctl_start, r_mask_bias, r_mask_dense, r_ple, r_pool, all_new, ctl_end (switches below) | 7 x 20 passed |
| 13:56:25 | `probe_pack` | `python $T/scripts/probe_pack.py 512,2048,8192` | gate/up packing slower; rms_norm L2 off by up to 1 % |
| 13:56:30-14:08:21 | `gates_e92_12x10_*`, `gates_e92_13x10_*` | `$T/scripts/gates.sh <tag>` (copy of the r2a gate script): text suite, `test_text_model[b1_s32768]`, demo | 153 / 1 / 2 passed at both grids |
| 14:08:23, 14:09:18 | `soak_1x2048_12x10`, `soak_1x4096_12x10` | `python $T/scripts/soak.py B 2000 100` | 2000 replays each, 0 mismatches |
| 14:10:54, 14:11:14 | `perf_text_new`, `perf_text_ctl` | `EMBEDDING_GEMMA2_PERF_SHAPES=1 EMBEDDING_GEMMA2_PERF_HF_CHECK=1 pytest perf.py -k "TestText and test_perf"` | 6 passed each |
| 14:11:33-14:32:47 | `sweep_new` | `EMBEDDING_GEMMA2_PREFILL_ALL=1 EMBEDDING_GEMMA2_PREFILL_DUMP=$T/dump/sweep_new pytest prefill.py` (HEAD `7cef532759`, clean) | 40 passed in 1270 s |
| 14:32:47-14:55:05 | `sweep_ctl` | same with the control switches | 40 passed in 1334 s |
| 14:55:06 | `tracy_b1_s128` | after profiles (`$T/scripts/tracy_after.sh`) | stopped at start by me (tmux session killed) at the person's 15:00 deadline; no after profiles |

A/B switches (code defaults = on; the control restores the stage-1 ops on the same code):

| label | `EG2_MASK_BIAS_FUSED` | `EG2_MASK_SLIDING_FROM_FULL` | `EG2_PLE_PER_LAYER_MIN_TOKENS` | `EG2_POOL_MEAN_DIVIDE` |
|---|---|---|---|---|
| ctl_start, ctl_end, sweep_ctl | 0 | 0 | 1000000000 | 1 |
| r_mask_bias | (1) | 0 | 1000000000 | 1 |
| r_mask_dense | 0 | (1) | 1000000000 | 1 |
| r_ple | 0 | 0 | (8192) | 1 |
| r_pool | 0 | 0 | 1000000000 | (0) |
| all_new, sweep_new | (1) | (1) | (8192) | (0) |

Analysis (CPU): `python3 $T/scripts/ab_table.py $T/run_ab <labels> --json $T/analysis/ab_burst.json`
(`analysis/ab_burst.md`); `python3 $T/scripts/compare_dumps.py $T/dump/ctl_start $T/dump/<label>`
(`analysis/ab_bitwise.log`); `python3 $T/scripts/build_summary.py $T perf_summary.json <commit>`.

## Decisions and findings

- **Control = stage-1 code path.** `ctl_start` reproduces the stage-1 12 x 10 gate values to every
  printed digit at all 10 buckets (e.g. 128 0.9942359516267729 / 0.9983430504798889, 65536
  0.9972403919928315 / 0.9995833039283752, as in `../01-baseline/perf_summary.json`) and the stage-1
  burst within 0.06 % (128 4.918 vs 4.918, 2048 25.345 vs 25.341, 65536 1723.46 vs 1724.45) **[M]**.
  `ctl_end` equals `ctl_start` bitwise at all 10 buckets; burst drift <= 0.03 % up to 32768 and
  +0.14 % at 65536 **[M]**. So same-session A/B differences above 0.15 % are not drift.
- **Windowed full attention rejected** (`probe_windowed_a`): `cu_window_seqlens = [0, L, S]` gives the
  dense-mask output bit for bit on the valid rows (12 / 12 cases), but SDPA is slower at every size
  measured: hd 512 2048:1536 0.581 -> 1.182 ms, 8192:6144 4.144 -> 7.606 ms; hd 256 2048:1536
  0.344 -> 0.615 ms (host-timed loops of 10 calls) **[M]**. The full-mask add it would remove costs
  33.5 us at 2048 and 26.5 ms at 65536 (profiles). Stage-4 item (the windowed mask generation, not
  the chunk skipping, is the likely cost **[I]**).
- **Mask outside the trace rejected**: refreshing a persistent mask only when the padding length
  changes moves the build to every request with a new length; it saves only for repeated lengths,
  which the harness input (one length per bucket) would overstate. Not built.
- **No-mask path not built**: the contract lists only `masked` shapes (no full-length batches); the
  skill makes it optional then.
- **Per-layer PLE threshold 8192 tokens** from the stage-1 profiles (wide matmul + 24 slices vs 24 x
  plain [S,512]x[512,512] matmuls, device us): 128 73 vs 106, 512 150 vs 141, 1024 323 vs 222, 2048
  517 vs 625, 4096 814 vs 1031, 8192 2126 vs 1519, 16384 4082 vs 2839, 32768 7999 vs 5427, 65536
  16243 vs 10746. 512 / 1024 favour per-layer by 9 / 100 us on these numbers, but the per-layer
  matmul at 1024 runs with its input in L1 while the profile's plain matmul input differs **[I]**;
  only the >= 8192 path was A/B-tested and kept. Below 8192 the wide path is unchanged (bitwise).
- **Gate / up packing rejected** (`probe_pack`): packed [512, 4096] matmul + 2 slices + gelu-multiply
  vs the current two matmuls + multiply: M 512 0.100 vs 0.074 ms, 2048 0.266 vs 0.192, 8192 1.008 vs
  0.848 ms **[M]** (DRAM operands in the probe). No stock op consumes the packed output without the
  split: `ttnn.geglu` is a composite (split + gelu + multiply, `unary_composite_op.cpp:280`, erf GELU by
  default); `dit_minimal_matmul_addcmul_fused` multiplies by a [1, N] row-broadcast tensor only and needs
  the weight dtype equal to the input dtype.
- **Pooler L2 normalize as `rms_norm` rejected** (`probe_pack`): rms_norm(x) * 768^-0.5 is 3.3x faster
  per call (host-timed 0.033 vs 0.109 ms) but the output norm is 0.990 / 0.996 / 1.007 instead of 1.0
  (max element error 1.4e-3 vs 6e-9 for the current chain) **[M]**. Adaptation not tried: a fp32
  dest-accumulation compute config for the norm. Cost left: 7 small ops, about 28 us at 2048 (profile).
- **Soaks** (KNOWN HAZARDS): no change touches the K/V rms_norm dtype or placement; masks, the PLE
  source and the pooler temporaries are DRAM tensors. The soaks were run anyway: 2000 replays at
  1 x 2048 (50.2 s) and 1 x 4096 (87.9 s), output bitwise equal to the first replay, 0 mismatches **[M]**.
- **b8 x 512, b32 x 128** (`perf_text_new` / `perf_text_ctl`, forward mean ms): 34.613 vs 34.612 and
  32.332 vs 32.348; b1 x 128 / 512 / 2048 / 8192: 4.909 / 7.227 / 25.102 / 100.224 vs 4.960 / 7.274 /
  25.340 / 103.751 **[M]**. At S <= 513 there is no band, so only the key-bias and pooler rewrites
  apply. Accuracy of padded batches: `test_text_model` b8 (512 ... 7), b3 (1025, 513, 31),
  b2 (512, 33) and `test_trace` pass at both grids (gates). b32 x 128 has no HF check (perf.py checks
  b1 x 512 only: cosine 0.999106 new and control) **[I: same code path as b8 x 512]**.

## Anomaly ledger

| observation | evidence | control | resolution |
|---|---|---|---|
| r_pool +0.07 / +0.13 % burst at 32768 / 65536 although it removes about 0.6 ms of device work at 65536 | `analysis/ab_burst.md` | ctl_end vs ctl_start +0.01 / +0.14 % at the same buckets | within the measured drift; the pooler win is claimed only up to 4096 |
| r_mask_bias +0.01 % at >= 16384 | same | same | within drift; claimed only at <= 512 |

## Self-check against prefill-stage-review (not independent)

- Before / after for all 10 buckets in burst and sustained, same session, same commit, same power
  limit; headline sustained; no bucket regresses; gates 10 / 10 at 12 x 10.
- Each kept rewrite: bitwise equal last hidden (data movement / mask topology) or embedding within
  3e-8 (pooler); burst win measured per rewrite (wins only at the buckets claimed); sustained
  confirmed for the kept set together.
- Every catalogue class assessed (README rewrite table); rejections carry a measured cost or the
  validation line.
- Gaps: no after Tracy profile (after op counts are from code, labelled [I]); no independent review
  yet; rms_norm L2 fp32 adaptation not tried; b32 x 128 has no HF accuracy check.
- Stage closure stopped at the person's 15:00 PDT deadline.
