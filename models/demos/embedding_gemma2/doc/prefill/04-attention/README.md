# Prefill stage 4: attention (E9.4)

Text-encoder prefill of `google/embeddinggemma-2`, batch 1, masked path, 12 x 10 grid (120 cores),
precision `sdpa_nofp32acc` (unchanged: SDPA HiFi2, no fp32 accumulation, exact exp; bf16 Q / K / V;
bfloat4_b masks), P150a at the default 150 W TDP. Labels: **[M]** measured (run named), **[I]** inferred.
Run folder `tt-ops/runs/embeddinggemma-2/e9/04-attention/` (`$A`); commands in `work_log.md`. Scope: batch 1
only (person's decision, 2026-10-29); batched calls keep the stage-3 SDPA calls.

## Before / after (latency_ms per traced warmed forward, 12 x 10) [M]

Before = same-session control on the same commit with `EG2_SDPA_TUNED=0` (issues the stage-3 SDPA calls:
`tests/perf/test_tuned_config_identity.py`; band block map from 4096, no padding map). After = code
defaults. Both `EMBEDDING_GEMMA2_PREFILL_ALL=1` (gate, burst, host, sustained with telemetry), back to back
(`sweep_new`, then `sweep_ctl`), records `$A/run_ab/sweep_{new,ctl}/prefill.jsonl`, summary
`perf_summary.json`. Headline: sustained. Code `7f868fe143` (clean tree).

| bucket | burst before | burst after | change | sustained before (AICLK MHz) | sustained after (AICLK MHz) | change | gate PCC / cos after (before) |
|---|---|---|---|---|---|---|---|
| 128 | 4.853 | 4.841 | -0.24 % | 4.848 (1350) | 4.837 (1350) | **-0.24 %** | 0.99424 / 0.99834 (0.99424 / 0.99834) pass |
| 256 | 5.866 | 5.516 | -5.97 % | 5.864 (1350) | 5.512 (1350) | **-6.00 %** | 0.99432 / 0.99861 (0.99432 / 0.99861) pass |
| 512 | 7.215 | 7.133 | -1.13 % | 7.211 (1350) | 7.130 (1350) | **-1.13 %** | 0.99537 / 0.99912 (0.99537 / 0.99912) pass |
| 1024 | 11.529 | 11.282 | -2.15 % | 11.524 (1350) | 11.279 (1350) | **-2.12 %** | 0.99588 / 0.99940 (0.99577 / 0.99947) pass |
| 2048 | 23.815 | 19.233 | -19.24 % | 23.810 (1350) | 19.235 (1350) | **-19.22 %** | 0.99492 / 0.99953 (0.99492 / 0.99953) pass |
| 4096 | 41.636 | 39.631 | -4.82 % | 41.717 (1343) | 39.701 (1343) | **-4.83 %** | 0.99570 / 0.99959 (0.99537 / 0.99952) pass |
| 8192 | 94.985 | 87.912 | -7.45 % | 96.972 (1306) | 90.479 (1293) | **-6.70 %** | 0.99695 / 0.99967 (0.99676 / 0.99966) pass |
| 16384 | 201.353 | 195.303 | -3.00 % | 214.721 (1231) | 207.514 (1237) | **-3.36 %** | 0.99705 / 0.99967 (0.99711 / 0.99966) pass |
| 32768 | 493.570 | 471.979 | -4.37 % | 548.900 (1143) | 526.523 (1143) | **-4.08 %** | 0.99714 / 0.99965 (0.99709 / 0.99965) pass |
| 65536 | 1537.284 | 1473.386 | -4.16 % | 1634.119 (1100) | 1577.429 (1087) | **-3.47 %** | 0.99723 / 0.99959 (0.99729 / 0.99958) pass |

**Input padding.** The harness input is 3/4 valid: logical length is 3/4 of the bucket
(`prefill_refs.logical_len`), so a quarter of the keys are padding. From 2048 up, part of the gain above
comes from the padding-aware sliding block map, which skips key blocks that hold only padding. An all-valid
input does not get that part. `perf_summary.json` records `input_valid_fraction` 0.75.

**Run order and clocks.** Control and candidate ran one after the other, not interleaved: `sweep_new` first,
then `sweep_ctl`. The sustained AICLK differs between the two runs at 8192 (ctl 1306 / new 1293 MHz), 16384
(1231 / 1237) and 65536 (1100 / 1087). The clocks are equal at the other buckets (1350 up to 2048, 1343 at
4096, 1143 at 32768).

### All-valid input A/B (E9.5 commit 1) [M, partial]

Settings: `EMBEDDING_GEMMA2_PREFILL_FULL_INPUT=1` (logical length = bucket, `valid_fraction` 1.0) with
`EG2_SDPA_TUNED=0` (before) vs defaults (after). 12 x 10, burst + sustained + telemetry, same tree
(`195b2ce22a` + the harness switch). The run order was `allvalid_new_short`, then `allvalid_ctl_short`, back to
back. Records are in `tt-ops/runs/embeddinggemma-2/e9/05-memory/run_ab/allvalid_{new,ctl}/prefill.jsonl`, and the
script is `e9/05-memory/scripts/allvalid_ab.sh`.

**Partial.** The chain was stopped at the person's request (2026-10-29, after 16:08). The control's last
record was the 2048 sustained window at 16:08:35, and the run was killed during 4096 sustained:
- Measured: burst for 128-8192, and sustained for both arms at 128-2048.
- Not measured: sustained for the control at 4096 and 8192, and every bucket from 16384 to 65536.

| bucket | burst before | burst after | change | (3/4-valid change) | sustained before (AICLK) | sustained after (AICLK) | change |
|---|---|---|---|---|---|---|---|
| 128 | 4.852 | 4.840 | -0.25 % | -0.24 % | 4.848 (1350) | 4.837 (1350) | -0.24 % |
| 256 | 5.867 | 5.516 | -5.99 % | -5.97 % | 5.864 (1350) | 5.512 (1350) | -6.00 % |
| 512 | 7.215 | 7.134 | -1.13 % | -1.13 % | 7.212 (1350) | 7.129 (1350) | -1.14 % |
| 1024 | 11.524 | 11.280 | -2.11 % | -2.15 % | 11.520 (1350) | 11.277 (1350) | -2.11 % |
| 2048 | 23.807 | 19.346 | -18.74 % | -19.24 % | 23.802 (1350) | 19.348 (1350) | -18.71 % |
| 4096 | 41.619 | 40.474 | -2.75 % | -4.82 % | not measured | 40.570 (1343) | - |
| 8192 | 94.953 | 90.039 | -5.18 % | -7.45 % | not measured | 93.174 (1284) | - |
| 16384-65536 | not measured | | | | | | |

Up to 1024 the all-valid result equals the 3/4-valid result, because there is no block map there. At 2048,
4096 and 8192 the all-valid gain is smaller by 0.5, 2.1 and 2.3 points. That difference is the padding-aware
map's share at the harness fill. It agrees with the bound in "Open / not done" (0.3 % at 2048, 2.0-2.1 % at
4096-16384).

No bucket regresses. Burst and sustained agree within 0.4 % up to 4096; from 8192 the sustained clock is
power-limited (1087-1306 MHz) and moves by up to 13 MHz between the two runs. The control reproduces the
E9.4 commit-1 numbers (128 / 256 / 1024 / 4096 burst 4.853 / 5.866 / 11.529 / 41.636 ms here vs 4.853 /
5.868 / 11.528 / 41.657 ms in `c1_new`) **[M]**. 128, 256, 512 and 2048 keep the last-hidden PCC to 5 digits
(their changes are bitwise: grids, q chunks with the k chunk kept, block maps); the k-chunk changes at 1024
and the full-layer q 160 from 4096 move it by -0.00006 ... +0.00033.

Cumulative against the stage-1 baseline at 12 x 10 (`../01-baseline/perf_summary.json`, a different session
and commit; sustained, then burst): 128 -1.6 / -1.6 %, 256 -6.8 / -6.8 %, 512 -1.8 / -1.8 %, 1024 -3.9 / -3.9 %,
2048 -24.1 / -24.1 %, 4096 -11.2 / -11.4 %, 8192 -13.0 / -15.4 %, 16384 -10.2 / -13.1 %, 32768 -12.0 / -16.9 %,
65536 -11.6 / -14.6 % **[M, cross-session]**.

## What changed

All of it is keyed to the 12 x 10 grid and batch 1 (`tt/attention.py`), behind `EG2_SDPA_TUNED` (default
on; `0` issues the stage-3 calls: the same-session control above). 13 x 10 issues the stage-3 calls
(`tests/perf/test_tuned_config_identity.py`, CPU: SDPA configs and the block-map policy identical to tuned-off
at 13 x 10, 12 x 10 differs in exactly the listed entries).

1. **Per-bucket SDPA chunks and grid** (`SDPA_TUNED[(12, 10)]`, 15 entries): fewer, larger work units at
   256-1024 on 8 x 8 / 4 x 8 / 4 x 2 grids; at 2048 sliding q 128 on 8 x 8 (64 units, one wave) and full q 96
   (88 units, one wave on 120 cores instead of 128 units in two); full q 160 from 4096 (104 / 208 / 412 / 820 /
   1640 units = 1 / 2 / 4 / 7 / 14 waves instead of 2 / 3 / 5 / 9 / 18 waves of q 128).
2. **Sliding block map from 2048** (was 4096): at 2048 the band (|q - k| <= 512) reaches 124 of the 256
   (q 128, k 128) blocks, so SDPA skips about half the K/V blocks of the 20 sliding layers (below 60 %
   visible, so the kernel honours the map; see "Masks and paths").
3. **Padding-aware sliding block map** (`AttentionMasks.padded_block_map`, from 2048): once per forward, 6
   small device ops build `band map x (key chunk holds a valid token)`, so SDPA also skips blocks whose keys
   are all padding. Bitwise equal outputs on valid rows (below); the gain depends on the input's padding
   (the harness input is 3/4 valid; an all-valid input skips nothing extra).
4. `band_block_map` takes chunks that do not divide S (ceil, clipped last block); before, a non power-of-two
   q chunk with a block map failed validation (`search_tables.md`, `wave` at 4096).

## Per-shape SDPA table

Per bucket and layer kind (20 sliding layers: head_dim 256, 4 Q / 2 KV heads, band + padding mask; 4 full
layers: head_dim 512, 4 Q / 1 KV head, padding mask), 12 x 10, batch 1. Work units = 4 heads x ceil(S / q);
cores = cores that get a unit; units / core = the busiest core. dtype for every row: Q / K / V bf16, mask
bfloat4_b, output bf16 (heads concatenated), HiFi2, fp32 accumulation off, exact exp (`sdpa_nofp32acc`,
unchanged). Burst / sustained / PCC are the forward's (`sweep_ctl` -> `sweep_new`, table above). SDPA us per
call: Tracy device kernel time of one warmed eager forward, before (`EG2_SDPA_TUNED=0`) and after, at 256 /
2048 / 8192 / 65536 (`$A/tracy/{before,after}/`, `$A/analysis/op_table_*`); those profiles run
`perf.py::test_tracy_forward`, whose input is all-valid (no padding), so they show the chunk / grid / band-map
changes and not the padding map's skips. Table generated by `$A/scripts/sdpa_table.py`.

| bucket | layer kind (calls) | q / k before -> after | grid | units | cores | units / core | block map | SDPA us / call before -> after | forward burst ms before -> after | sustained ms before -> after | PCC / cos after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | sliding (20) | 32 / 128 -> 32 / 128 | 8x8 | 16 | 16 | 1 | - | - | 4.853 -> 4.841 | 4.848 -> 4.837 | 0.99424 / 0.99834 |
| 128 | full (4) | 64 / 128 -> 64 / 128 | 8x8 -> 4x2 | 8 | 8 | 1 | - | - |  |  |  |
| 256 | sliding (20) | 32 / 128 -> 64 / 128 | 8x8 | 32 -> 16 | 32 -> 16 | 1 | - | 31 -> 20 | 5.866 -> 5.516 | 5.864 -> 5.512 | 0.99432 / 0.99861 |
| 256 | full (4) | 32 / 128 -> 128 / 128 | 8x8 | 32 -> 8 | 32 -> 8 | 1 | - | 82 -> 49 |  |  |  |
| 512 | sliding (20) | 32 / 64 -> 64 / 64 | 8x8 -> 4x8 | 64 -> 32 | 64 -> 32 | 1 | - | - | 7.215 -> 7.133 | 7.211 -> 7.130 | 0.99537 / 0.99912 |
| 512 | full (4) | 32 / 32 -> 64 / 32 | 8x8 | 64 -> 32 | 64 -> 32 | 1 | - | - |  |  |  |
| 1024 | sliding (20) | 64 / 128 -> 64 / 64 | 8x8 | 64 | 64 | 1 | - | - | 11.529 -> 11.282 | 11.524 -> 11.279 | 0.99588 / 0.99940 |
| 1024 | full (4) | 64 / 128 -> 64 / 32 | 8x8 | 64 | 64 | 1 | - | - |  |  |  |
| 2048 | sliding (20) | 64 / 128 -> 128 / 128 | 12x10 -> 8x8 | 128 -> 64 | 120 -> 64 | 2 -> 1 | - -> band+pad | 319 -> 123 | 23.815 -> 19.233 | 23.810 -> 19.235 | 0.99492 / 0.99953 |
| 2048 | full (4) | 64 / 64 -> 96 / 64 | 12x10 | 128 -> 88 | 120 -> 88 | 2 -> 1 | - | 566 -> 417 |  |  |  |
| 4096 | sliding (20) | 256 / 128 -> 256 / 128 | 8x8 | 64 | 64 | 1 | band -> band+pad | - | 41.636 -> 39.631 | 41.717 -> 39.701 | 0.99570 / 0.99959 |
| 4096 | full (4) | 128 / 128 -> 160 / 128 | 12x10 | 128 -> 104 | 120 -> 104 | 2 -> 1 | - | - |  |  |  |
| 8192 | sliding (20) | 256 / 128 -> 256 / 128 | 12x10 -> 8x8 | 128 | 120 -> 64 | 2 | band -> band+pad | 750 -> 632 | 94.985 -> 87.912 | 96.972 -> 90.479 | 0.99695 / 0.99967 |
| 8192 | full (4) | 128 / 128 -> 160 / 128 | 12x10 | 256 -> 208 | 120 | 3 -> 2 | - | 4122 -> 3482 |  |  |  |
| 16384 | sliding (20) | 256 / 128 -> 256 / 128 | 12x10 | 256 | 120 | 3 | band -> band+pad | - | 201.353 -> 195.303 | 214.721 -> 207.514 | 0.99705 / 0.99967 |
| 16384 | full (4) | 128 / 128 -> 160 / 128 | 12x10 | 512 -> 412 | 120 | 5 -> 4 | - | - |  |  |  |
| 32768 | sliding (20) | 256 / 128 -> 256 / 128 | 12x10 | 512 | 120 | 5 | band -> band+pad | - | 493.570 -> 471.979 | 548.900 -> 526.523 | 0.99714 / 0.99965 |
| 32768 | full (4) | 128 / 128 -> 160 / 128 | 12x10 | 1024 -> 820 | 120 | 9 -> 7 | - | - |  |  |  |
| 65536 | sliding (20) | 256 / 128 -> 256 / 128 | 12x10 | 1024 | 120 | 9 | band -> band+pad | 4548 -> 4552 | 1537.284 -> 1473.386 | 1634.119 -> 1577.429 | 0.99723 / 0.99959 |
| 65536 | full (4) | 128 / 128 -> 160 / 128 | 12x10 | 2048 -> 1640 | 120 | 18 -> 14 | - | 203640 -> 191587 |  |  |  |

SDPA share of the forward's device time (op tables, before -> after): 2048 sliding 27.1 -> 12.9 %, full 9.6 ->
8.7 %; 8192 full 17.5 -> 15.6 %, sliding 15.9 -> 14.1 %; 65536 full 57.2 -> 55.6 % (1.92x -> 1.81x the
HiFi2 compute floor of the op-table model), sliding 6.4 -> 6.6 % (at the model's DRAM floor). The 65536
profiles ran at a 1287 MHz capture clock (both), the others at 1350 MHz.

## Search (in the full model) and what was kept

`search_tables.md`: 277 candidates over 10 buckets in 19 sessions (`$A/search/*.jsonl`), 44 illegal with
their exact errors. The real model and the harness input, one candidate at a time, burst median vs a control
(the stage-3 calls) at the start and end of each bucket (drift <= 0.04 % up to 16384, 0.13-0.20 % at 32768,
0.54-0.82 % at 65536); last hidden of the replay vs the control's. Knobs: q chunk, k chunk, grid (the
core limit of this op: `SDPAProgramConfig.max_cores_per_head_batch` is read only by decode SDPA and
`k_splits` only by ring-joint SDPA, so the program-config grid is the one core-count control for this call:
2 x 2 to 12 x 10 swept), the sliding block map per bucket and the padding map. `exp_approx_mode` and the
compute kernel config belong to the precision policy (stage 7) and were not changed.

| bucket | kept (sliding hd 256 / full hd 512: q / k, grid) | alone (round) | together (round) | not kept, best alternatives |
|---|---|---|---|---|
| 128 | full 64 / 128 on 4 x 2 | -0.24 % (`r2_small`), bitwise | - | sliding: all 13 candidates +0.7 ... +7.8 % (more cores cost more at 16 units); full 32 / 128 on 4 x 4 -0.20 % |
| 256 | sliding 64 / 128 on 8 x 8; full 128 / 128 on 8 x 8 | -3.78 % / -2.12 % (`r1`) | -6.00 % (`r2_small`), bitwise | k 256 combinations -6.34 / -6.54 % at last-hidden PCC vs control 0.998 / 0.995: 0.3-0.5 % more for a changed accumulation order; the bitwise pair is kept |
| 512 | sliding 64 / 64 on 4 x 8; full 64 / 32 on 8 x 8 | -1.04 % (`r2_small`) / -0.15 % (`r1`) | -1.11 % (`r3_small`), bitwise | sliding on 4 x 4 +7.0 %, 8 x 4 -0.59 % |
| 1024 | sliding 64 / 64 on 8 x 8; full 64 / 32 on 8 x 8 | -0.65 % (`r1`) / -1.44 % (`r2_small`) | -2.10 % (`r3_small`), PCC vs control 0.99581 | no bitwise candidate beats -0.25 %; 4 x 8 grids +2.9 / +8.5 % |
| 2048 | sliding 128 / 128 on 8 x 8 + band map + padding map; full 96 / 64 on 12 x 10 | sliding q 128 -4.56 % (`r1`), + map -13.36 % (`r2_mid`); full q 96 -2.53 % (`wave`) | -18.86 % with the band map (`r3_mid`); padding map +0.28 % more on the 12 x 10 variant (-16.07 vs -15.79 %); kept set in the closure A/B | sliding q 96 -5.18 % alone but -14.85 % vs -15.79 % in the combination; sliding k 64 / 256 with the map -15.40 / -14.41 % (not bitwise) |
| 4096 | sliding unchanged (256 / 128 on 8 x 8, band map) + padding map; full 160 / 128 | full -2.81 % (`r3_mid`), padding map -1.99 % (`padmap`) | -4.79 % (`r3_mid`) | sliding on 12 x 10 with the rest -4.59 %; sliding q 160-224 +0.8 ... +3.3 %; full q 192 L1 clash; full q 96 +0.08 % |
| 8192 | sliding 256 / 128 on 8 x 8 + padding map; full 160 / 128 | -2.39 % (`r1`) / -2.73 % (`r2_mid`); padding map -2.10 % | -5.11 % (`r2_mid`), with the padding map -7.21 % (`r3_mid`) | sliding 8 x 10 -1.97 %, 8 x 6 -1.05 %, 4 x 8 +2.34 %; sliding q 160 / 192 +0.75 / -0.59 % |
| 16384 | padding map; full 160 / 128 | -2.08 % (`r2_long`) / -0.94 % (`r3_long`) | -3.00 % (`r3_long`) | full q 96 +2.71 % (more K/V passes outweigh the wave fill); no sliding candidate beats the control (q 256 k 256 -0.25 % not bitwise; 8 x 8 +0.04 %) |
| 32768 | padding map; full 160 / 128 | -1.67 % / -2.61 % | -4.32 % | full q 96 +8.44 %; sliding 8 x 8 +0.98 %, q 224 +1.19 % |
| 65536 | padding map; full 160 / 128 | -1.28 % / -2.50 % (drift 0.54 / 0.82 %) | -3.88 % (drift 0.82 %) | full q 96 +7.66 %; sliding 8 x 8 +0.86 %, q 224 +1.11 % |

Findings that generalize **[M]**: at 128-1024 the cost is per-unit overhead, not balance: fewer, larger units
on fewer cores win (256: 16 + 8 units on 64 cores beat 32 + 32). From 2048 the full layers want whole
waves on 120 cores with the largest q that fits L1: q 160 for head_dim 512 (q 192 clashes with the L1
activations at 4096 and needs 1698176 B of static CBs at 16384, `r4_long`; q 256 does not fit); q 96 fills
waves too but rereads K/V 33 % more often and loses from 4096 on. The sliding
layers with a block map read only their band; they gain from fewer cores (8 x 8) at 2048 / 8192, not at
16384-65536. k chunks above 128 (hd 512) or 256 (hd 256) do not fit L1 at any q that helps.

## Known targets from the brief

- **2048 full SDPA: chunk 64, 128 units, two waves on 120 cores.** Fixed: q 96 gives 88 units in one wave
  (-2.53 % of the forward alone, `wave`, bitwise); q 128 (64 units) is -1.61 / -1.41 % (k 64 / 128), q 160 /
  192 -0.09 / +0.79 %. The sliding call at
  2048 had the same two-wave split (q 64, 128 units) and is now q 128 on 64 cores with the band map.
- **Sliding and full layers swept separately** (every round names `hd256` / `hd512`; they are different SDPA
  calls with different head_dim, heads and masks). Sliding layers use the dense band + padding bfp4 mask with
  the block map from 2048 now (was 4096); the map is built per bucket from the tuned chunks.
- **head_dim 512 is L1-bound** (exact errors in `search_tables.md`): q 256 needs 1744256 B (q 256 / k 128) to
  2006400 B of static CBs (> 1572864 B L1) at every bucket; k 256 needs 1631616 B (q 64) / 1965440 B (q 128);
  k 512 2.4-3.1 MB; q 192 / k 128 at 4096 clashes with an L1 activation at 1462272 B (CB region to 1501568 B).
  head_dim 256: k 512 does not fit (2117120-2248640 B) or clashes (1024, 2048).
- **Head split (NlpCreateHeads)**: re-measured below ("Head layouts"); stays, stage-6 candidate.
- **Configs keyed by grid and bucket at batch 1; 13 x 10 unchanged**: `SDPA_TUNED[(12, 10)]`,
  `SLIDING_MAP_MIN_SEQ[(12, 10)]`, `SLIDING_PAD_MAP[(12, 10)]`; the CPU identity test from commit 1 extended to
  SDPA configs and the map policy (7 tests, each with a negative control where it applies).

## Masks and paths

- **One path, one chunk plan.** The contract has only the masked path (`target_contract.json`
  `paths: ["masked"]`; stage 2 built no no-mask path), so "masked and no-mask paths share one chunk plan and
  match on all-valid input" reduces to: one chunk plan per (layer kind, bucket) and the mask prepared once
  per forward. `sdpa_program_config` is the only source of chunk sizes; the sliding block maps are built
  from the same call (`AttentionMasks.prepare`). On all-valid input the padding-aware map equals the band
  map (`padmap_check`: "all valid" case, 148 / 148 blocks at 4096, 308 / 308 at 8192), so all-valid input
  runs exactly the band-map plan.
- **Mask preparation runs once per forward**: `AttentionMasks.forward` (2 device ops for the full and sliding
  masks, stage 2) + the padding-aware map (6 small ops from 2048 tokens) before the first layer; the 24 layers
  read them.
- **Mask dtype bfloat4_b, not the bf16 score dtype.** Kept from E5a (`doc/optimized_text/`): SDPA reads it at
  bf16 speed with identical outputs (sweep 512-8192), and it is 3.6x smaller in DRAM, which bounds 65536
  (a bf16 full + sliding pair there is 17 GB). In bfp4, 0 stays 0 and -1e9 stays fully masking. Not
  re-opened here.
- **Padding-aware sliding map** (new): the skipped blocks hold only keys the dense mask sets to -1e9, so
  valid rows do not change: device map equal to the host reference for 3/4-valid, all-valid, 1-token and
  holed `keep` at 2048 / 4096 / 8192 (12 / 12), SDPA output on valid rows bitwise equal to the band-only map
  at all three buckets (`$A/logs/padmap_check.log`), and the full-model replay with it is bitwise equal to the
  band-only map in the search (`padmap`, `r2_long`: PCC 1.0, bitwise True at 2048-65536). A query chunk whose
  band holds only padding keys runs one K chunk (the kernel's minimum) instead of its band: a padded row,
  never read and masked as a key for every valid row.
- **Full layers: no block map.** A padding-only map would skip 1/4 of the key blocks at the harness fill, but
  the SDPA reader ignores a block map for K/V-chain participants when more than 60 % of the sampled blocks
  are visible (`ttnn/.../sdpa/device/kernels/dataflow/reader_interleaved.cpp:310-338`, "chains win below about
  40 percent masked"), and non-causal calls always chain (`sdpa_program_factory.cpp` `kv_chain_mode_for`).
  Not measured **[I]**. Windowed SDPA (`cu_window_seqlens`, on-device padding) was 1.4-3.1x slower in stage 2
  (`../02-topology/README.md`, rewrite row 6); not re-probed with the new chunks.

## Head layouts

- **Out**: SDPA writes `[B, 1, S, H*hd]` (`output_concat_heads=True`, before stage 2); there is no concat op.
- **In**: `nlp_create_qkv_heads` (NlpCreateHeads, one per layer) stays. Re-measured cost, after profiles
  (all-valid input, device kernel time per forward) **[M]**: 0.101 ms (1.9 %) at 256, 0.681 ms (3.6 %) at
  2048, 3.917 ms (4.4 %) at 8192, 30.33 ms (2.2 %) at 65536. Blocker (stage 2 rejection, unchanged): the
  per-head q / k / v RMSNorm and the rotary embedding need `[B, H, S, hd]`, and no stock SDPA entry point
  reads heads from the fused QKV output. **Stage-6 candidate**: a split that writes the normed heads (fold
  the per-head RMSNorm into the split, or read heads by stride in the norm / rope kernels), worth up to the
  cost above.
- The padding-aware map adds 6 small ops per forward (reshape, reduce max, transpose, multiply, typecast,
  untilize; the untilize adds a fill-pad and a clone at 2048, 8 device ops there): 27 us at 2048, about 28 us at 8192, about 38 us at 65536 in the
  after profiles (ops present only after; the multiply is lost in the run-to-run spread of the large mask
  adds at 8192 / 65536) **[M]**.

## Accuracy and gates at closure [M]

All at `7f868fe143`, clean tree, `$A/scripts/closure.sh` (`$A/logs/closure_summary.log`, `gates_summary.log`).

- Contract gate at 12 x 10 (traced replay vs HF fp32, bars 0.99 / 0.99): 10 / 10 pass in `sweep_new` (values
  in the table above) and in `gate_track` (`TT_METAL_TRACE_ALLOC_TRACKING=1`, 10 passed, no allocation
  warning in the log; the per-forward padding map allocates inside the captured forward like the masks);
  replay == eager bitwise at every bucket. Lowest: 256, last hidden PCC 0.99432 (unchanged). Bars unchanged.
- Text suite + full model + trace + generator_vllm + precision_config (153 passed, 1 deselected = s32768),
  `test_text_model[b1_s32768]` (1 passed) and the demo (2 passed) at 12 x 10 and at 13 x 10
  (`gates_e94_{12x10,13x10}_*`). The suite is unchanged (153 cases); its batched cases (b8 x 512, b4 x 2048,
  ...) run the stage-3 SDPA calls.
- Soak 2000 replays at 1 x 2048 and 1 x 4096, 12 x 10: 0 mismatches (`soak_1x{2048,4096}_12x10`). The SDPA
  CBs at both buckets changed (2048: new chunks on 8 x 8 and q 96; 4096: q 160), which moves the L1 layout
  next to the K/V rms_norm (KNOWN HAZARD).
- Matmul L1 fit after the attention change: the stage-3 matmul configs (in0_block_w 16 2D at 1024-4096, the
  largest static CBs) run in the same forwards at every bucket in `sweep_new`, the gates and the soaks with
  no CB / L1 clash error; the search recorded every SDPA candidate that clashed instead
  (`program.cpp:2590`, `search_tables.md`).
- 13 x 10 path: same SDPA calls and block-map policy as stage 3 (CPU identity test), gates above.
- Padding-aware map correctness: `padmap_check` (device map vs host reference, 12 / 12; SDPA valid rows bitwise
  equal to the band-only map at 2048 / 4096 / 8192).

## Open / not done

- **Full-layer SDPA at 16384-65536** stays the largest single cost (55.6 % of the 65536 forward's device
  time in the after profile, 1.81x the op-table model's HiFi2 compute floor). Every legal q / k / grid was swept; the remaining knobs are outside this stage: the
  kernel's chain rule that ignores a block map above 60 % visible (a padding map for the full layers would
  skip 25 % of their key blocks at the harness fill; a kernel change, stage 6), precision of the QK / PV
  matmuls (`exp_approx`, fidelity: stage 7), and L1 headroom for k 256 at head_dim 512 (stage 5 / 6).
- **Head split** stays (cost in "Head layouts"); stage-6 candidate: read Q / K / V heads from the fused QKV output in the
  norm + rope kernels, or fold the per-head norm into the split.
- **Input-dependent gain**: the padding-aware map's share depends on how much of the bucket is padding.
  The harness input is 3/4 valid (contract regime); a full-length input gets the chunk / grid / band-map
  part only. The search's band-map-only numbers (`padmap` rows vs their round's other rows) bound the
  difference per bucket: 0.3 % at 2048, 2.0-2.1 % at 4096-16384, 1.3-1.7 % at 32768-65536.
- 65536 burst drift between controls is 0.5-0.8 % (power-limited clock); the kept 65536 change (-3.88 %
  in the search) is above it.
- Batched shapes (b8 x 512, b32 x 128) were not re-measured (batch-1-only decision); they issue the
  stage-3 SDPA calls (CPU identity test) and pass the suite.
- Independent stage review: not done by me (the orchestrator's reviewer runs it).
