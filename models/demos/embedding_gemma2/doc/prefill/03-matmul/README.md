# Prefill stage 3: matmul configs (E9.3)

Text-encoder prefill of `google/embeddinggemma-2`, batch 1, masked path, 12 x 10 grid (120 cores),
precision `sdpa_nofp32acc` (unchanged: bfp8 weights, HiFi2, fp32 accumulation, bf16 out for every
projection), P150a at the default 150 W TDP. Labels: **[M]** measured (run named), **[I]** inferred.
Run folder `tt-ops/runs/embeddinggemma-2/e9/03-matmul/` (`$U`); commands in `work_log.md`. Scope: batch 1
only (person's decision, 2026-10-29); batched calls keep the stage-2 configs.

## Before / after (latency_ms per traced warmed forward, 12 x 10) [M]

Before = same-session control on the same commit with the tuned table off (`EG2_MM_TUNED=0`, which issues
exactly the `a12d330bb3` matmul calls: `$U/scripts/config_dump.py --tuned-off`, 0 of 168 differ). After =
code defaults. Both `EMBEDDING_GEMMA2_PREFILL_ALL=1` (gate, burst, host, sustained with telemetry), back
to back (`sweep_new`, then `sweep_ctl`), records `$U/run_ab/sweep_{new,ctl}/prefill.jsonl`, summary
`perf_summary.json`. Headline: sustained. Code `5207e7e7b5`.

| bucket | burst before | burst after | change | sustained before (AICLK MHz) | sustained after (AICLK MHz) | change | gate PCC / cos after (before) |
|---|---|---|---|---|---|---|---|
| 128 | 4.878 | 4.840 | -0.78 % | 4.874 (1350) | **4.836** (1350) | **-0.79 %** | 0.99424 / 0.99834 (0.99424 / 0.99834) pass |
| 256 | 5.874 | 5.855 | -0.34 % | 5.871 (1350) | **5.851** (1350) | **-0.34 %** | 0.99336 / 0.99879 (0.99432 / 0.99861) pass |
| 512 | 7.215 | 7.214 | -0.01 % | 7.211 (1350) | **7.211** (1350) | **+0.00 %** | 0.99537 / 0.99912 (0.99537 / 0.99912) pass |
| 1024 | 11.647 | 11.512 | -1.16 % | 11.643 (1350) | **11.506** (1350) | **-1.18 %** | 0.99577 / 0.99947 (0.99577 / 0.99939) pass |
| 2048 | 25.111 | 23.815 | -5.16 % | 25.100 (1350) | **23.810** (1350) | **-5.14 %** | 0.99492 / 0.99953 (0.99574 / 0.99952) pass |
| 4096 | 43.919 | 41.609 | -5.26 % | 43.936 (1350) | **41.634** (1350) | **-5.24 %** | 0.99537 / 0.99952 (0.99581 / 0.99961) pass |
| 8192 | 100.343 | 94.888 | -5.44 % | 101.149 (1331) | **96.305** (1318) | **-4.79 %** | 0.99676 / 0.99966 (0.99628 / 0.99965) pass |
| 16384 | 211.755 | 201.340 | -4.92 % | 222.041 (1256) | **213.810** (1237) | **-3.71 %** | 0.99711 / 0.99966 (0.99706 / 0.99967) pass |
| 32768 | 520.911 | 493.490 | -5.26 % | 564.480 (1181) | **544.670** (1156) | **-3.51 %** | 0.99709 / 0.99965 (0.99693 / 0.99965) pass |
| 65536 | 1569.359 | 1534.655 | -2.21 % | 1661.942 (1112) | **1630.397** (1100) | **-1.90 %** | 0.99729 / 0.99958 (0.99724 / 0.99958) pass |

No bucket regresses. 512 has no kept entry and issues the same calls in both runs (+0.00 %). From 8192 up
the sustained AICLK is 12-25 MHz lower after the change than before (same pattern as stage 2: less idle
time between heavy ops at the same power limit **[I]**), so the sustained gain there is smaller than the
burst gain. The control matches the stage-2 after numbers within run-to-run variation (stage-2
`sweep_new` sustained 4.875 / 25.099 / 100.553 / 1643.468 ms vs 4.874 / 25.100 / 101.149 / 1661.942 ms here
at 128 / 2048 / 8192 / 65536; the 8192 and 65536 differences follow the sustained clock, 1343 / 1125 MHz
then vs 1331 / 1112 MHz now) **[M]**. The per-bucket "all kept" burst of the final sweep agrees with
`sweep_new` (2048: -5.14 % vs -5.16 %; 8192: -5.46 % vs -5.44 %).

## E9.4 amendments (stage-3 review remediation, E9.4 commit 1) [M]

The independent review (verdicts.md row E9.3) asked to drop five entries. Dropped from
`TUNED_CONFIGS[(12, 10)]`: `gate@256` (its -0.17 % came with a last-hidden PCC drop 0.99432 -> 0.99336)
and the four entries whose final-sweep wins were <= 0.1 %: `ple_gate@128`, `qkv_sliding@1024`,
`ple_layer@8192`, `ple_gate@4096`. Each changed bucket has exactly one dropped entry, so each row
below is that entry's own A/B: control = the table above (`7febde8744` code), candidate = the
table without the entry, back to back in one tmux job (`c1_ctl` then `c1_new`, gate + burst +
sustained + telemetry; `tt-ops/runs/embeddinggemma-2/e9/04-attention/run_ab/c1_{ctl,new}/prefill.jsonl`,
`logs/c1_ab_table.md`).

| bucket | dropped entry | burst ctl -> new (ms) | change | sustained ctl -> new (ms, AICLK) | change | last hidden PCC / cos new (ctl) |
|---|---|---|---|---|---|---|
| 128 | ple_gate@128 | 4.839 -> 4.853 | +0.27 % | 4.836 -> 4.848 (1350) | +0.26 % | 0.99424 / 0.99834 (same) |
| 256 | gate@256 | 5.855 -> 5.868 | +0.22 % | 5.852 -> 5.864 (1350) | +0.21 % | **0.99432** / 0.99861 (0.99336 / 0.99879) |
| 1024 | qkv_sliding@1024 | 11.510 -> 11.528 | +0.16 % | 11.506 -> 11.523 (1350) | +0.15 % | 0.99577 / 0.99947 (same) |
| 4096 | ple_gate@4096 | 41.625 -> 41.657 | +0.08 % | 41.669 -> 41.690 (1350) | +0.05 % | 0.99537 / 0.99952 (same) |
| 8192 | ple_layer@8192 | 94.849 -> 94.960 | +0.12 % | 96.608 -> 96.544 (1312) | -0.07 % | 0.99676 / 0.99966 (same) |

Every drop costs 0.05-0.27 % (8192 sustained is clock-bound and moves -0.07 %). The 256 gate PCC is back
to 0.99432 as expected. 128 and 1024 cost more than their stage-3 single-entry wins (-0.08 % / -0.07 %
in `final_sweep.md`): the entries were worth more in the full table than alone **[I]**. The drops stand as
the review decided; restoring `ple_gate@128` (+0.27 %) is the orchestrator's call. The headline table
above is the stage-3 closure and is not re-derived; stage 4 (`../04-attention/`) re-measures all 10 buckets
on the amended table.

## Matmul roles per shape (before)

Every matmul of the forward, by role, from the stage-2 code's Tracy profiles at 12 x 10 (one warmed
eager forward, `perf.py::test_tracy_forward`; 128 / 2048 / 8192 / 65536 = `../02-topology/profiles_after/`,
the other six `profiles_before/`): `roles_before.md` (all 10 buckets: role, M x K x N, calls, device us
and per call, share of the forward's device kernel sum, cores, family, memory, bound class and floor gap
of the `prefill_op_table` roofline model). After: `roles_after.md` (`$U/tracy/after/`). Matmuls are 31 %
of device time at 128, 26 % at 512, 33 % at 2048 / 4096, 32 % at 8192, 24 % at 32768, 17 % at 65536.

Material roles (>= 1 % of the forward at some bucket) and their largest share (before):

| role | M x K x N (per bucket M = S) | largest share (bucket) | bound class (model) | gap to the model floor (2048 / 8192 / 65536) |
|---|---|---|---|---|
| gate (MLP, + gelu_tanh) | S x 512 x 2048 | 10.0 % (8192) | memory | 4.1x / 4.9x / 4.1x |
| up (MLP) | S x 512 x 2048 | 5.3 % (4096) | memory | 2.6x / 2.2x / 2.1x |
| qkv_sliding | S x 512 x 2048 | 4.4 % (4096) | memory | 2.6x / 2.2x / 2.1x |
| down (MLP) | S x 2048 x 512 | 6.5 % (128) | memory (compute at 512 / 1024) | 2.7x / 1.7x / 1.6x |
| o_sliding | S x 1024 x 512 | 3.1 % (128) | compute at 2048, memory from 4096 | 5.7x / 1.7x / 1.3x |
| ple_gate (+ gelu_tanh) | S x 512 x 512 | 5.2 % (32768) | compute at 2048, memory from 8192 | 9.4x / 4.2x / 8.4x |
| ple_proj | S x 512 x 512 | 2.5 % (2048) | compute / memory | 8.0x / 1.9x / 1.6x |
| ple_layer (>= 8192) | S x 512 x 512 | 1.5 % (8192) | memory | - / 1.9x / 1.6x |
| qkv_full | S x 512 x 3072 | 1.2 % (4096) | memory | 2.5x / 2.3x / 2.2x |
| ple_wide (< 8192) | S x 512 x 12288 | 1.6 % (1024) | memory | 2.6x / - / - |
| o_full | S x 2048 x 512 | 1.1 % (128; <= 0.8 % elsewhere) | memory | 2.7x / 1.7x / 1.6x |
| final, pooler | | <= 0.1 % | | not material |

## What was searched, kept and why

In-model search (`$U/scripts/mm_sweep.py`): the real model, the harness input, one candidate (role, bucket)
at a time in one device session, burst median vs a control at the start and end of the session (drift
<= 0.04 % up to 16384, 0.19 % at 32768, 0.2-0.7 % at 65536). 366 single-role candidates over 74 (role,
bucket) plus 24 combinations, including 9 illegal candidates (static circular buffers over the 1.57 MB
L1, or clashing with live L1 buffers) with their exact errors:
`search_tables.md` (per role and bucket: kernel family, grid, in0_block_w, subblocks, per-core / minimal
blocks, M fold, output memory, gelu placement, legality, burst, PCC vs control, kept). Then `final_sweep.md`:
each kept config alone and all together per bucket, against the stage-2 control. Final config table:
`config_table.md`.

| finding | evidence [M] | kept |
|---|---|---|
| **gelu_tanh fused into `minimal_matmul` doubles the call** (gate at 8192: 416 us vs 188 us plain). The stage-2 rule sends the gelu gate to `minimal_matmul` from 8192 tokens because `mcast2d_config` returns None: its static-CB estimate exceeds the 900 KB budget at per_core_M = 26 even with in0_block_w 1 (1077 KB). Plain `minimal_matmul` + gelu as the multiply's SFPU pre-activation is the fastest fix; folding M into 4096-row chunks keeps the gelu on the 2D fast path and is second (-3.30 % at 8192) | gate alone -3.50 / -2.90 / -1.54 % (8192 / 16384 / 32768); ple_gate (same issue from 32768; 2D on 80 cores below) -1.35 / -1.55 / -3.72 / -1.94 % (8192 ... 65536) | gate 8192-32768, ple_gate 8192-65536 |
| **in0_block_w 16** (K = 512 in one block) on the K = 512 2D roles | up -0.57 / -0.95 %, qkv_sliding -0.47 / -0.79 %, qkv_full -0.11 / -0.18 % at 2048 / 4096 (bitwise); 1024 and 256 smaller | 256-4096 per role |
| gelu placement at 1024-4096: with bw 16, gelu in the multiply wins at 2048 / 4096 (-1.15 / -1.29 %), fused in the matmul wins at 256 / 1024 (-0.18 / -0.85 %) | final sweep | per bucket |
| N = 512 roles: 2D (8 of 12 columns, per_core_N 2) beats `minimal_matmul` for o_sliding / ple_proj at 2048 (-1.18 / -1.23 %) and ple_proj at 4096 (-0.94 %); from 4096 / 8192 smaller `minimal_matmul` blocks (4,8,2) / (4,8,4) beat the default (8,8,8) by 0.1-0.6 % | final sweep | per bucket |
| 128 tokens: 1D in0-multicast over up to 120 cores for qkv_sliding (8 x 8), qkv_full, ple_wide (2D used 44-48 cores) | -0.13 / -0.16 / -0.18 %; all four 128 entries -0.78 % | yes |
| 512 tokens: no candidate beat the control (bw 16 +0.1-0.2 %, 1D +0.3-19 %, transpose / minimal slower) | `search_tables.md` | rule kept |
| 8192 / 16384 up: 2D per 4096-row chunk (fuse_batch off) instead of `minimal_matmul` | -0.27 / -0.17 % | yes |
| down / ple_layer at 32768 / 65536, gate at 65536 | gains inside the session drift (-0.13 % at 0.19 % drift; +0.18 % and -0.67 % at 0.71 % drift) | **not kept** |

### Known targets from the brief

- **2D matmuls on 110 cores at 12 x 10.** Cause: `mcast2d_config` sets per_core_N = ceil(N_tiles / 12);
  for N = 2048 (64 tiles) that is 6, so 11 columns are used (and per_core_M = ceil(M_tiles / 10)). For
  rectangular program configs (2D / 1D multicast) it is not fixable at tile granularity: any rectangular
  split of a 64 x 64-tile output (2048 tokens) on 12 x 10 has at least 42 tiles on the busiest core (6 x 7);
  using all 120 cores needs 5.33 N tiles per core. The non-rectangular `minimal_matmul` does run on all
  120 cores and was measured +0.66 % slower (below). The alternatives were measured and are slower: grid 12 x 8 +0.21 %, 8 x 10 +0.54 %, 8 x 8 +1.22 %,
  transpose_mcast +1.08 %, `minimal_matmul` on all 120 cores (pads N to 70 tiles) +0.66 %, subblock 1 x 2
  -0.04 % (2048). Kept instead: in0_block_w 16 on the same grid. Blocker recorded: tile granularity.
- **gelu gate on `minimal_matmul` from 8192 tokens.** Selection rule and fix in the first finding above.
  At 12 x 10 all 24 layers take the same path at every bucket (24 calls per row in every profile); the
  "layer 0 at smaller buckets" observation of the 13 x 10 baseline does not occur at 12 x 10.
- **Gate x up multiply with an input in DRAM at 2048** (L1 rule `tt/config.py` `L1_ACT_BYTES_PER_BANK`:
  a bf16 [2048, 2048] tensor is 8 MB, over 64 KB x 120). Recorded, not kept (placement is stage 5): with the
  up and gate outputs in L1 (`out_mem="L1"`) the forward is -4.10 % at 2048 and -5.13 % at 4096 with the
  gelu fused, -4.56 / -5.55 % with the gelu in the multiply; up alone in L1 -2.99 / -3.78 % (bitwise).
- **Roles ranked by share** before the search: `roles_before.md`; QKV, O, gate, up, down and the
  per-layer-input projections are the material roles (table above).
- **13 x 10 unchanged.** `TUNED_CONFIGS` has only a (12, 10) key and is consulted at batch 1 only.
  `config_dump.py` (`$U/logs/config_dump.log`): 13 x 10 issues the same 168 (role, shape) calls as
  `a12d330bb3`, including the batched shapes; 12 x 10 differs in exactly the 39 kept entries. Gates pass
  at 13 x 10 (below). E9.4 moved this proof into the repo as a CPU-only test,
  `tests/perf/test_tuned_config_identity.py` (table on vs `TUNED_ON` off: 13 x 10 identical, 12 x 10
  differs in exactly the listed entries, batched shapes included); with a planted 13 x 10 entry
  (`down@4096`) it fails, reverted it passes (`04-attention/logs/c1_identity_{planted,clean}.log`).
  After the E9.4 drops the table has 34 entries.

### Floors and blockers per material role

| role | after (best kept) | remaining gap and blocker |
|---|---|---|
| gate | gelu moved out of `minimal_matmul` (8192-32768); bw 16 (2048 / 4096) | SFPU gelu_tanh in exact mode (`math_approx_mode` False): precision policy, stage 7. Output (bf16, 8-268 MB) in DRAM: placement, stage 5 (L1 output -4 to -6 % measured at 2048 / 4096). 64 N tiles on 12 columns (tile granularity). 65536: no candidate beat the 0.7 % drift of the power-limited clock |
| up, qkv_sliding, qkv_full | bw 16 (1024-4096), 2D per 4096-row chunk (8192 / 16384), 1D at 128 | DRAM output (stage 5); tile granularity on 12 columns; `minimal_matmul` block variants all slower or illegal (static CBs 1.70 / 2.49 MB > 1.57 MB L1) at >= 8192 |
| down, o_sliding | minimal (4,8,2) / (4,8,4); 2D for o_sliding at 2048 | input h in DRAM from 2048 (placement); at 128-1024 latency-bound (2D on 32-80 cores; 1D, transposed and `minimal_matmul` all 0.7-24 % slower) |
| ple_gate, ple_proj, ple_layer | gelu in the multiply, 2D / minimal per bucket | N = 16 tiles: at most 8 columns in 2D; small per-call work (26-64 us at 2048-8192) |
| ple_wide | 1D at 128 | below 8192 only; DRAM output [S, 12288]; at 1024 every 1D / transposed variant is illegal (static CBs 1.6-2.4 MB > 1.57 MB) |
| o_full | rule (searched at 2048-8192: transposed 2D and minimal blocks within +0.9 / -0.04 %) | material only at 128 (4 calls, 49 us); same shape class as down there, whose 1D / transposed / minimal variants are all slower; not searched separately at 128 |
| all, 128-1024 | small per-bucket gains | per-call latency (4-15 us per call, model floor 1-3 us); dispatch is not in the device time |

## Accuracy and gates at closure [M]

- Contract gate at 12 x 10 (traced replay vs HF fp32, bars 0.99 / 0.99): 10 / 10 pass in `sweep_new` and
  in `gate_track` (same values; `gate_track` ran with `TT_METAL_TRACE_ALLOC_TRACKING=1`, 20 passed, no
  unsafe allocation reached `execute_trace`); replay == eager bitwise at every bucket. Values in the table
  above. Lowest: 256, last hidden PCC 0.99336 (before 0.99432); the gelu placement and family changes lower
  the last-hidden PCC by 0.0004-0.0010 at 256 / 2048 / 4096 and raise it by 0.00005-0.0005 at 8192-65536;
  128, 512 and 1024 last hidden unchanged to 5 digits. Bars unchanged.
- Text suite + full model + trace + generator_vllm + precision_config (153 passed, 1 deselected =
  s32768), `test_text_model[b1_s32768]` (1 passed) and the demo (2 passed) at 12 x 10 and at 13 x 10
  (`gates_e93_{12x10,13x10}_*`, HEAD `5207e7e7b5`). The suite is unchanged (153 cases).
- Soak 2000 replays at 1 x 2048 and 1 x 4096, 12 x 10: 0 mismatches (`soak_1x{2048,4096}_12x10`; the kept
  2D configs grow static CBs at these buckets, which moves the L1 layout around the K/V norm). They ran at
  `50ad98eee4`, whose 2048 / 4096 entries equal `5207e7e7b5`'s (the amends only removed 32768 / 65536
  entries).
- 13 x 10 path: same calls as `a12d330bb3` (`config_dump.py`), gates above.

## Open / not done

- 65536: the gate role keeps the stage-2 config (no candidate beat the 0.2-0.7 % burst drift of the
  power-limited clock; the profile predicts about 1.3 ms per layer less device time). Needs a quieter
  measurement (longer burst, or sustained A/B) to decide.
- Output placement: up / gate in L1 at 2048 / 4096 is worth -4 to -6 % (measured); stage 5.
- gelu_tanh exact-mode SFPU cost (the remaining gate gap): precision, stage 7.
- After profiles at 6 buckets (128, 2048, 4096, 8192, 16384, 65536: `profiles_after/`, `roles_after.md`).
  The 65536 after profile's kernel sum (1418.9 ms) is almost equal to before (1420.5 ms) although ple_gate
  drops 2206 -> 433 us per call. **Attributed in E9.4 [M]:** the two captures convert device cycles to ns
  at different clocks (DEVICE FW END - START CYCLE over DEVICE FW DURATION: 1331.0 MHz before, 1293.0 MHz
  after; the profiler reads AICLK once per capture, and `op_table.json`'s `clock_mhz` 1350 is the nominal
  value, not the capture's). In cycles every unchanged op group is within -1.9 / +1.6 % (full SDPA
  -0.04 %, MLP matmuls +0.01 / +0.16 %), and the kernel sum per forward drops 1901.8 -> 1846.7 Mcyc
  (-55.2 Mcyc = -41.5 ms at 1331 MHz): ple_gate's matmul -70.5 Mcyc, its plain minimal_matmul calls +13.5,
  the gelu moved into the multiply +19.8, the other multiplies -15.2. The "other roles 2-3 % slower" was
  the 2.9 % clock ratio (1331 / 1293). Script and output:
  `tt-ops/runs/embeddinggemma-2/e9/04-attention/scripts/profile_clock_attr.py`,
  `logs/c1_profile_clock_attr_65536.log`. Wall time (burst -2.21 %, sustained -1.90 %) remains the claim.
- Batched shapes (b8 x 512, b32 x 128) were not re-measured (batch-1-only decision); they issue the
  stage-2 calls.
- Independent stage review: not done by me (the orchestrator's reviewer runs it).
