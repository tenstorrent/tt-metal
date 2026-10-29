# Prefill stage 4 (attention, E9.4): work log

Run folder: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e9/04-attention/` (below: `$A`): `scripts/`,
`logs/`, `run_ab/` (harness records), `search/` (SDPA search records), `analysis/`, `tracy/`. Host
sjc-snva-tp100, card 0 (P150a, default 150 W TDP), grid 12 x 10 (`EMBEDDING_GEMMA2_GRID=12x10`, the
harness default) unless a line says 13 x 10. Precision `sdpa_nofp32acc` (unchanged). Every device command
ran through `$A/scripts/run.sh` (card-free check polling every 60 s, env dump, `timeout -s KILL`,
`TT_METAL_OPERATION_TIMEOUT_SECONDS=60` (120 for sweeps and Tracy), `__start` with commit and dirty files /
`__end wall_s` / `__tt_exit`), one at a time, chained in tmux sessions `tt-bringup_embeddinggemma2_e904_*`.
The card was never held by another process (`$A/logs/card_wait.log` absent). Batch 1 only (person's
decision 2026-10-29); the batched suite cases still run in the gates.

## Commits

| commit | content |
|---|---|
| `441e57eb47` | commit 1, E9.3 review remediation: five `TUNED_CONFIGS[(12, 10)]` entries dropped (`tt/linear.py`), CPU identity test `tests/perf/test_tuned_config_identity.py`, empty SDPA table hook in `tt/attention.py`, 03-matmul doc amendments |
| `7f868fe143` | SDPA configs for 12 x 10 (`tt/attention.py` `SDPA_TUNED`, `sliding_map_policy`), padding-aware sliding block map (`tt/masks.py`, `tt/text_model.py`, `tt/encoder_layer.py`), identity test for the map policy |
| (this docs commit) | `doc/prefill/04-attention/` |

## Commands (chronological)

| time | label (`$A/logs/<label>.log`) | command | result |
|---|---|---|---|
| 12:15-12:19 | `c1_ctl` | `$A/scripts/c1_ab.sh ctl` = `EMBEDDING_GEMMA2_PREFILL_BUCKETS=128,256,1024,4096,8192 ..._HF_CHECK=1 ..._SUSTAINED=1 ..._TELEMETRY=1 pytest prefill.py` at `05a8aa15d7` | 15 passed |
| 12:20-12:24 | `c1_new` | same, five tuned entries dropped (uncommitted, = `441e57eb47`) | 15 passed; `c1_ab_table.md`: +0.05 ... +0.27 %, 256 PCC 0.99336 -> 0.99432 (03-matmul README "E9.4 amendments") |
| (CPU) | `c1_profile_clock_attr_65536` | `python3 $A/scripts/profile_clock_attr.py <03-matmul tracy before / after 65536 csv>` | capture clocks 1331.0 / 1293.0 MHz; -55.2 Mcyc per forward = ple_gate |
| (CPU) | `c1_identity_planted`, `c1_identity_clean` | `pytest tests/perf/test_tuned_config_identity.py` with a planted `TUNED_CONFIGS[(13, 10)]` entry, then reverted | 1 failed (`down@b1_s4096`), then 6 passed |
| 12:24-12:38 | `sweep_r1_128-256-512-1024` | `$A/scripts/sweep.sh r1 128,256,512,1024` (`sdpa_sweep.py`, set `r1`: per layer kind q / k / grid one at a time) | 71 candidates, 8 illegal (L1) |
| 12:38-12:47 | `sweep_r1_2048-4096-8192` | `sweep.sh r1 2048,4096,8192` | 45 candidates, 11 illegal (L1) |
| 12:47-12:49 | `sweep_wave_2048-4096` | `sweep.sh wave 2048,4096` (non power-of-two q, sliding map at 2048) | 13 candidates; 3 block-map shape errors (my floor bug, fixed in `tt/masks.py` `band_block_map`), 1 L1 clash |
| 12:49-13:07 | `sweep_r1_{16384,32768,65536}` | `sweep.sh r1 <S> <S>` (one session per bucket) | 14 each, 4 illegal each |
| 13:07-13:13 | `sweep_r2_small_128-256-512-1024` | `sweep.sh r2_small ...` (smaller grids, combinations) | 26 candidates |
| 13:13-13:18 | `sweep_r2_mid_2048-4096-8192` | `sweep.sh r2_mid ...` (whole-wave q, block map at 2048, combinations) | 22 candidates; 3 illegal (q 144 not a tile multiple) |
| 13:18-13:20 | `sweep_padmap_2048-4096-8192` | `sweep.sh padmap ...` (band map with padding-only key chunks cleared, host-built probe) | -3.95 / -1.99 / -2.10 %, bitwise |
| 13:20 | `padmap_check` | `python $A/scripts/padmap_check.py 2048,4096,8192` (device-built `padded_block_map` vs host reference, 4 keep patterns; SDPA valid rows band vs band+pad) | 12 / 12 maps equal, SDPA bitwise equal at 3 buckets, `fails 0` |
| 13:20-13:29 | `sweep_r2_long_{16384,32768,65536}` | `sweep.sh r2_long <S> <S>` | 15 candidates, 4 illegal (q 112 / 144 not tile multiples); q 96 slower; pad map -2.08 / -1.67 / -1.28 % |
| 13:29-13:37 | `sweep_r3_small_...`, `sweep_r3_mid_...` | `sweep.sh r3_small ...`, `sweep.sh r3_mid ...` | 16 + 13 candidates, all legal: final combinations at 128-8192 |
| 13:37 | `sweep_r3_long_16384_killed` | started with the final table being written into `tt/attention.py`; killed (SIGINT, device closed) before any result could use the new code: its process had loaded weights at 13:37:08, the edit landed at 13:37:28; records moved to `search/r3_long_16384_killed.jsonl`, not used | - |
| 13:38-13:43 | `sweep_r3_long_{16384,32768,65536}` | `EG2_SDPA_TUNED=0 sweep.sh r3_long <S> <S>` (full q 160, + pad map) | -0.94 / -2.61 / -2.50 % (q 160), -3.00 / -4.32 / -3.88 % (+ pad) |
| 13:44-13:46 | `smoke_new` | `EMBEDDING_GEMMA2_PREFILL_BUCKETS=128,...,8192 EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 pytest prefill.py` (final table, uncommitted = `7f868fe143`) | 14 passed, 7 / 7 gates pass |
| 13:46-14:07 | `sweep_new` | `$A/scripts/closure.sh` (all closure lines at `7f868fe143`, clean): `EMBEDDING_GEMMA2_PREFILL_ALL=1 pytest prefill.py` | 40 passed in 1234 s |
| 14:07-14:28 | `sweep_ctl` | same with `EG2_SDPA_TUNED=0` | 40 passed in 1263 s; README before / after |
| 14:28 | `gate_track` | `TT_METAL_TRACE_ALLOC_TRACKING=1 EMBEDDING_GEMMA2_PREFILL_SHAPES=1 EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1 pytest prefill.py -k gate` | 10 passed, no allocation warning |
| 14:29-14:31 | `soak_1x2048_12x10`, `soak_1x4096_12x10` | `python e9/02-topology/scripts/soak.py B 2000 100` | 2000 replays each, 0 mismatches |
| 14:31-14:43 | `gates_e94_{12x10,13x10}_{text,s32768,demo}` | `$A/scripts/gates.sh <tag>` (copy of the 03-matmul gate script) | 153 / 1 / 2 passed at both grids |
| 14:43-14:46 | `tracy_{before,after}_b1_s{256,2048,8192,65536}` | `[EG2_SDPA_TUNED=0] $A/scripts/tracy.sh <tag> 256 2048 8192 65536` (`perf.py::test_tracy_forward`, all-valid input) | 8 passed; `$A/analysis/op_table_*`, `sdpa_table.md` |
| 14:46-14:48 | `sweep_r4_long_{16384,32768}` | `EG2_SDPA_TUNED=0 sweep.sh r4_long <S> <S>` (full q 192) | q 192 illegal (1698176 B static CBs); q 160 -0.95 / -2.72 % (repeat of `r3_long`) |
| (CPU) | - | `$A/scripts/build_summary.py`, `sdpa_table.py`, `search_table_compact.py`, `prefill_op_table` | `perf_summary.json`, tables |

## Anomaly ledger

| observation | evidence | control | resolution |
|---|---|---|---|
| E9.3 drops cost more than their stage-3 single-entry wins at 128 / 1024 (+0.27 / +0.16 % vs -0.08 / -0.07 %) | `c1_ab_table.md` vs `../03-matmul/final_sweep.md` | back-to-back control, same code otherwise | recorded in the 03-matmul README; the drops stand as the review decided; restoring `ple_gate@128` is the orchestrator's call |
| 4096 sliding candidates with q 96 / 160 / 192 failed block-map validation in `wave` | `sdpa_device_operation.cpp:210`: map [1, 1, 42, 32] for 43 chunks | my `band_block_map` used floor(S / q) | fixed (ceil, clipped last block; equal maps for divisible chunks, brute-force check for 2048 / q 96); the same chunks re-measured legal (and slower) in `r2_mid` |
| q 112 / 144 illegal | `sdpa_device_operation.cpp:120` (not a multiple of 32) | - | my arithmetic slip; tile multiples 96 / 160 / 192 measured instead |
| `r3_long_16384` started while the final table was being written | log header 13:37:07, weights loaded 13:37:08, edit 13:37:28 | - | killed, results discarded (`*_killed`), re-run with `EG2_SDPA_TUNED=0` |
| Burst drift 0.5-0.8 % at 65536 | `r1_65536`, `r2_long_65536`, `r3_long_65536` controls | same session | kept 65536 change -3.88 % (search) exceeds it; closure sustained A/B is the claim |

## Self-check against prefill-stage-review (not independent)

| check | status |
|---|---|
| before / after in the stage-1 regime, burst and sustained, same commit, all 10 buckets | `sweep_new` / `sweep_ctl` (`7f868fe143`, `EG2_SDPA_TUNED=0` control), back to back |
| every kept config improves full-forward burst wall time, alone and together | search rounds (`search_tables.md`; 2048's final grid + padding map together only in the closure A/B) |
| no shape regresses beyond noise | README before / after |
| configs gated on geometry; other geometry unchanged | `SDPA_TUNED[(12, 10)]`, map policy per grid; CPU identity test; 13 x 10 gates |
| masked / no-mask | only the masked path exists (contract); one chunk plan; padding map equals the band map on all-valid input |
| precision held | no dtype / fidelity / exp change; k-chunk changes alter accumulation order (PCC vs control 0.996-0.999), gates pass |
| PCC gates 10 / 10, thresholds and tests unchanged | README |
| trace allocation tracking on replay | `gate_track` |
| failed candidates with exact errors | `search_tables.md` (42) |
| K/V norm / L1 hazard soak | `soak_1x{2048,4096}_12x10` |
| gaps | full-layer padding skip blocked by the kernel's chain rule (inferred, not measured); independent review pending |
