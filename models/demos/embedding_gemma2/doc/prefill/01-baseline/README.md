# Prefill stage 1: baseline and target contract (E9)

**Reproduce at 12 x 10** (environment in `work_log.md`, "Environment"; `EMBEDDING_GEMMA2_GRID=12x10` is
the harness default): quick default `pytest models/demos/embedding_gemma2/tests/perf/prefill.py -s -q`
(buckets 128 / 1024 / 8192, burst; 3 passed in 53 s, `ctl_start`) and the full stage-1 sweep
`EMBEDDING_GEMMA2_PREFILL_ALL=1 pytest models/demos/embedding_gemma2/tests/perf/prefill.py -s -q`
(10 buckets, gate + burst + sustained + telemetry + host; 40 passed in 1339 s, `sweep_all`). Timings
from the E9.1-r2b table in `work_log.md`.

**This is the stage-1 baseline at the contract geometry: 12 x 10 compute grid (120 cores, Galaxy
target), text precision `sdpa_nofp32acc`** (E9.1-r2b re-baseline, 2026-10-28). Text-encoder prefill of
`google/embeddinggemma-2` on one Blackhole P150a (12 x 10 via `EMBEDDING_GEMMA2_GRID=12x10`, every
record carries `compute_grid` [12, 10]), batch 1, buckets 128 to 65536, masked path. Contract:
`../target_contract.json`. Labels: **[M]** measured (command and record named), **[I]** inferred.
No model code changed (HEAD `637e2b5292` during the runs; model code as at `a903d517c6`). The 13 x 10
`p3_qkv_gu` baseline of the morning is kept below, labelled as the superseded geometry.

## Headline at 12 x 10 (latency_ms per traced warmed forward) [M]

Sweep run `20261028-122505` (`EMBEDDING_GEMMA2_PREFILL_ALL=1`, `logs/sweep_all.log`, 40 passed),
records `tt-ops/runs/embeddinggemma-2/e9/01r2b/run/prefill.jsonl`, summary `perf_summary.json` (top
level; the 13 x 10 summary is nested unchanged under `superseded_13x10_p3_qkv_gu`). Power limit 150 W
TDP (default), AICLK max 1350 MHz, fw bundle 19.8.0.0, KMD 2.11.0, PCIe x16 Gen5. Telemetry: median
(min or max) over the last third of the sustained run, sysfs every 100 ms. Gate: traced replay output
vs cached HF fp32 (bars 0.99 / 0.99), replay == eager bitwise on all 10. % of roofline = roofline ms
(roofline session model, `sdpa_nofp32acc`, 120 cores, 1.35 GHz; see Roofline) / sustained ms.
Owner stages: 02 topology, 03 matmul, 04 attention, 05 memory.

| bucket (logical) | burst median ms (avg / best, n) | **sustained ms** | sust. / burst | AICLK MHz median (min) | power W median (max) | gate PCC / cos | % of roofline (roofline ms) | top-3 costs (device ms, owner stage) |
|---|---|---|---|---|---|---|---|---|
| 128 (96) | 4.918 (4.918 / 4.916, 20x5) | **4.914** | 0.999 | 1350 (1350) | 77 (81) | 0.99424 / 0.99834 pass | 6.6 % (0.323) | RMSNorm 1.73 (05); matmul 1.41 (03); rotary 0.60 (04 / 05) |
| 256 (192) | 5.917 (5.917 / 5.916, 20x5) | **5.913** | 0.999 | 1350 (1350) | 84 (94) | 0.99432 / 0.99861 pass | 5.5 % (0.323) | RMSNorm 1.76 (05); matmul 1.46 (03); SDPA 0.95 (04) |
| 512 (384) | 7.262 (7.262 / 7.260, 20x5) | **7.257** | 0.999 | 1350 (1343) | 98 (115) | 0.99537 / 0.99912 pass | 9.5 % (0.693) | RMSNorm 1.84 (05); matmul 1.80 (03); SDPA 1.49 (04) |
| 1024 (768) | 11.740 (11.741 / 11.739, 18x5) | **11.734** | 0.999 | 1350 (1343) | 106 (155) | 0.99577 / 0.99939 pass | 13.1 % (1.535) | SDPA 3.05 (04); matmul 2.94 (03); RMSNorm 2.06 (05) |
| 2048 (1536) | 25.341 (25.338 / 25.330, 8x5) | **25.335** | 1.000 | 1350 (1343) | 115 (173) | 0.99574 / 0.99952 pass | 13.4 % (3.404) | SDPA 8.66 (04); matmul 8.11 (03); RMSNorm 2.57 (05) |
| 4096 (3072) | 44.713 (44.712 / 44.705, 5x5) | **44.714** | 1.000 | 1350 (1343) | 125 (186) | 0.99581 / 0.99961 pass | 17.6 % (7.886) | matmul 14.78 (03); SDPA 11.83 (04); rotary 5.49 (04 / 05) |
| 8192 (6144) | 103.900 (103.911 / 103.888, 3x5) | **103.961** | 1.001 | 1350 (1343) | 109 (223) | 0.99628 / 0.99965 pass | 19.1 % (19.821) | SDPA 31.44 (04); matmul 31.29 (03); RMSNorm 11.69 (05) |
| 16384 (12288) | 224.728 (224.737 / 224.695, 3x5) | **231.122** | 1.028 | 1293 (1287) | 118 (207) | 0.99706 / 0.99967 pass | 24.0 % (55.573) | SDPA 79.90 (04); matmul 59.50 (03); RMSNorm 22.17 (05) |
| 32768 (24576) | 568.084 (568.075 / 567.433, 3x5) | **598.529** | 1.054 | 1237 (1231) | 121 (192) | 0.99693 / 0.99965 pass | 29.2 % (174.610) | SDPA 242.14 (04); matmul 123.90 (03); mask build 58.33 (02 / 05) |
| 65536 (49152) | 1724.449 (1723.010 / 1718.434, 3x5) | **1785.044** | 1.035 | 1193 (1150) | 134 (188) | 0.99724 / 0.99958 pass | 33.8 % (602.815) | SDPA 893.60 (04); matmul 250.43 (03); mask build 242.01 (02 / 05) |

- Gates: 10 / 10 pass, values identical to the E9.1-r2a gate run (`01r2a/run_gate_12x10`, which ran
  with `TT_METAL_TRACE_ALLOC_TRACKING=1`; no allocation flagged) **[M]**. Minimum 0.99424 / 0.99834 (128).
- Up to 8192 the sustained clock holds 1350 MHz and sustained = burst within 0.1 %. From 16384 up the
  firmware lowers AICLK to 1293 / 1237 / 1193 MHz (minimum 1150) at 118-134 W median, and sustained
  is 2.8 / 5.4 / 3.5 % above burst **[M]**. The drop is larger than in the 13 x 10 `p3_qkv_gu`
  baseline (1325 / 1312 / 1306 MHz). The grid is excluded: at 32768 the 13 x 10 control on the same
  code gives the same clock and time (Controls) **[M]**. The cause is unattributed: only the grid was
  controlled. Sustained power and Vcore are lower in the 12 x 10 sweep than in the 13 x 10
  `p3_qkv_gu` baseline (32768 / 65536 power median 121 / 134 W vs 130 / 138 W; Vcore at 32768 0.767 V
  vs 0.785 V; `perf_summary.json` `sustained_telemetry`, top level vs `superseded_13x10_p3_qkv_gu`)
  **[M]**. Power stays under the 150 W TDP and current under the 200 A TDC as medians; which limit the
  firmware applies is not exposed in sysfs **[I]**.
- Burst at >= 16384 is not a full-clock figure: the clock starts dropping inside the first windows
  (as in the 13 x 10 baseline).

## Controls in the same session (E9.1-r2b) [M]

Order: `ctl_start`, sweep, `ctl13`, `ctl12`, `ctl_end`, then `tracy_b1_s2048_13x10`, `ctl13_32768`,
`ctl12_32768` (`e9/01r2b/scripts/chain.sh`, `chain2.sh`, `logs/chain.log`). Same code, same precision;
only `EMBEDDING_GEMMA2_GRID` differs.

| bucket | 13 x 10 burst / sustained ms | 12 x 10 burst / sustained ms (ctl12; sweep sustained) | 12 x 10 cost (sustained) | sustained AICLK 13 x 10 / 12 x 10 |
|---|---|---|---|---|
| 512 | 7.054 / 7.050 | 7.262 / 7.258 (7.257) | +2.9 % | 1350 / 1350 |
| 2048 | 20.069 / 20.067 | 25.338 / 25.337 (25.335) | **+26.3 %** | 1350 / 1350 |
| 8192 | 95.610 / 96.896 | 103.768 / 103.877 (103.961) | +7.2 % (burst +8.5 %) | 1318 / 1350 |
| 32768 | 567.833 / 599.322 | 567.598 / 599.656 (598.529) | +0.1 % (noise) | 1237 / 1237 |

- The earlier uncontrolled 1 x 2048 pair (27.9 ms at 12 x 10 vs 20.1 ms at 13 x 10) mixed grid and
  precision (`p3_qkv_gu` at 12 x 10, a different session). Controlled, the grid alone costs 5.27 ms at
  2048 with `sdpa_nofp32acc`.
- At 8192 the 13 x 10 run is power-throttled in sustained (1318 MHz) and 12 x 10 is not (1350 MHz),
  so the sustained cost (+7.2 %) is smaller than the burst cost (+8.5 %).
- Drift (unchanged code, plain default run at start and end, burst ms): 128 4.918 / 4.917, 1024
  11.739 / 11.741, 8192 103.748 / 103.753 (<= 0.02 %). Sweep vs `ctl12` repeat: 512 7.262 / 7.262,
  2048 25.341 / 25.338, 8192 103.900 / 103.768 burst.
- 13 x 10 gates on the same inputs: 512 0.99537, 2048 0.99574 (the same values as at 12 x 10),
  8192 0.99674 (12 x 10: 0.99628; the 8192 gate projection runs another kernel at 12 x 10, see below).

## Profiles and per-op tables at 12 x 10

Text profiles were not recaptured. The roofline session captured them at `a903d517c6` through the
stage-1 harness path (`perf.py::test_tracy_forward[b1_s<bucket>]`, one warmed eager forward between
`PERF_TEXT` / `PERF_TEXT_END`) into `tt-ops/runs/embeddinggemma-2/roofline/capture/out/text_b1_s<bucket>/`.
Checked per bucket before use (`e9/01r2b/scripts/verify_captures.py`, `logs/verify_captures.log`)
**[M]**: commit `a903d517c6` in `device_info.json` and `run_info.json`, clean tree, exit 0;
`EMBEDDING_GEMMA2_GRID=12x10`, every forward op `AVAILABLE WORKER CORE COUNT` 120, max `CORE COUNT`
120; `precision_config_id` `sdpa_nofp32acc`, sha256 of the precision file equal to the committed
`selected_precision_config.json`; one `PERF_TEXT` and one `PERF_TEXT_END` signpost, 645 / 646 forward
ops; `MAX_SEQ_LEN` 32768 (65536 for 65536). Between `a903d517c6` and the measured HEAD only docs and
`tests/perf` tooling changed. CSV sha256 per bucket: `logs/op_tables_12x10.log`.

Tables: `profiles_12x10/b1_s<bucket>/op_table.{md,json}`
(`prefill_op_table <csv> --bucket B --grid-cores 120`; the header says 120 cores). The 13 x 10 tables
stay in `profiles/` (130 cores, `p3_qkv_gu`).

### Expected costs from the roofline work, checked against the 12 x 10 tables

| expectation | verdict | evidence [M] |
|---|---|---|
| SDPA at 2048: 13.8 ms at 12 x 10 vs 7.95 ms at 13 x 10, from q_chunk 64 = 128 work units (2 waves on 120 cores, 1 on 130); stage 4 | **cause confirmed, size refuted for the current precision.** 13.8 ms is the `p3_qkv_gu` (fp32-accumulating SDPA) 12 x 10 profile. With `sdpa_nofp32acc` SDPA at 2048 is 8.66 ms at 12 x 10 vs 6.85 ms at 13 x 10 (+1.81 ms, +26 %). Both SDPA kinds run q_chunk 64 at 2048 (sliding k_chunk 128, full k_chunk 64; CSV attributes), so 4 x 2048 / 64 = 128 units. Owner 04-attention (`_SDPA_TABLE` chunks were tuned at 13 x 10) | `profiles_12x10/b1_s2048`; 13 x 10 control profile at the same code `e9/01r2b/tracy/b1_s2048_13x10/op_table.json` |
| 2D matmuls run on 110 cores at 12 x 10; stage 3 | **confirmed at 2048 and 4096 only**: the N = 2048 gate / up / PLE projections (2048x512x2048: 44 plain + 24 gelu calls, 3.99 ms at 2048; 7.75 ms at 4096) run on 110 cores (per_core_N = ceil(64 / 12) = 6 tiles, so 11 of 12 columns **[I]**). At <= 1024 the matmuls run on 24-96 cores, at >= 8192 they are `minimal_matmul` on 120. At 2048 they cost +1.18 ms vs 13 x 10 (2.81 ms on 130 cores) | `profiles_12x10/b1_s{2048,4096}`, 2048 control |
| layer-0 gelu gate falls back to `minimal_matmul`; stage 3 | **confirmed, wider than stated**: the gate projection with fused gelu (Mx512x2048) is `minimal_matmul` in all 24 layers at >= 8192 (2D `linear` at 13 x 10; layer 0 is only where outputs first differ). At 8192 it costs 9.99 ms (416 us a call) vs 8.26 ms for 44 plain `minimal_matmul` calls of the same shape (188 us a call): the fused activation doubles the call. From 32768 the Mx512x512 projection with fused gelu also moves to `minimal_matmul`: 26.5 ms at 32768 vs 7.15 ms as an 80-core `matmul` at 16384 (3.7x for 2x the tokens) | `profiles_12x10/b1_s{8192,16384,32768,65536}` |
| per-forward mask build at 32K / 65K; stage 2 / 5 | **confirmed**: the two bfp4 [S, S] mask adds take 51.9 + 6.5 = 58.3 ms at 32768 (third cost) and 215.5 + 26.5 = 242.0 ms at 65536 (third cost, 8 ms behind matmul); 3.7 ms at 8192 | `profiles_12x10/b1_s{32768,65536}` |

Grid cost at 2048 by op family (13 x 10 vs 12 x 10 device ms, same code; kernel sums 19.89 vs 25.10,
+5.21 ms, matching the +5.27 ms wall difference) **[M]**: SDPA 6.85 -> 8.66 (+1.81), matmul 6.75 ->
8.11 (+1.36), eltwise 1.40 -> 2.36 (+0.96), rotary 1.80 -> 2.43 (+0.63), norms 2.27 -> 2.58 (+0.30),
other +0.15. Part of the eltwise / rotary / norm cost is placement, not cores: at 12 x 10 the MLP
gate x up multiply ([1, 1, 2048, 2048], 24 calls, 1.41 ms) and the full-attention layers' rotary and
Q/K norms run on DRAM-interleaved tensors; at 13 x 10 the same ops run in L1 (op-table memory column;
the 13 x 10 control has the multiply in L1 at 0.69 ms for 48 calls) **[M: placement; I: share of the
cost]**. The activation memory config depends on the grid (`tt/config.act_memory_config`): a
stage-5 item.

## Reconciliation at 12 x 10: wall vs device kernel sum, gaps and host work [M]

| bucket | ops | burst wall ms | device kernel sum ms | traced residual ms (share, per op) | eager op-to-op gaps ms | host work ms (round trip - replay) | round trip ms | roofline ms (+SFPU) | % of roofline vs device |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 645 | 4.918 | 4.573 | 0.346 (7.0 %, 0.54 us) | 12.456 | 0.082 | 5.036 | 0.323 (0.323) | 7.1 % |
| 256 | 645 | 5.917 | 5.584 | 0.333 (5.6 %, 0.52 us) | 11.854 | 0.084 | 6.035 | 0.323 (0.459) | 5.8 % |
| 512 | 645 | 7.262 | 6.968 | 0.294 (4.1 %, 0.46 us) | 10.684 | 0.047 | 7.346 | 0.693 (1.014) | 9.9 % |
| 1024 | 646 | 11.740 | 11.554 | 0.187 (1.6 %, 0.29 us) | 6.165 | 0.087 | 11.862 | 1.535 (2.271) | 13.3 % |
| 2048 | 646 | 25.341 | 25.095 | 0.246 (1.0 %, 0.38 us) | 0.448 | 0.090 | 25.455 | 3.404 (5.050) | 13.6 % |
| 4096 | 646 | 44.713 | 44.474 | 0.240 (0.5 %, 0.37 us) | 0.568 | 0.119 | 44.861 | 7.886 (11.660) | 17.7 % |
| 8192 | 646 | 103.900 | 103.377 | 0.524 (0.5 %, 0.81 us) | 0.525 | 0.339 | 104.280 | 19.821 (29.095) | 19.2 % |
| 16384 | 646 | 224.728 | 224.524 | 0.205 (0.1 %, 0.32 us) | 0.545 | -1.275 (noise) | 226.930 | 55.573 (80.819) | 24.8 % |
| 32768 | 646 | 568.084 | 561.992 | 6.092 (1.1 %) | 0.532 | -12.190 (noise) | 578.638 | 174.610 (251.683) | 31.1 % |
| 65536 | 646 | 1724.449 | 1661.090 | 63.358 (3.7 %) | 0.528 | -27.847 (noise) | 1755.799 | 602.815 (863.078) | 36.3 % |

- wall = device kernel sum + traced residual. Kernel sum and eager gaps come from the eager profile;
  the residual is wall minus kernel sum. Up to 16384 it is 0.2-0.5 ms (0.3-0.8 us per op, dispatch).
  At 32768 / 65536 it follows the clock: the profile is one forward at 1350 MHz, and the burst windows
  run while AICLK falls **[I, as in the 13 x 10 baseline]**.
- Host work per request (round trip minus replay + sync) is 0.05-0.34 ms up to 8192. From 16384 up
  it is negative: the two are timed one after the other while the clock drifts, at N = 14 / 6 / 5 **[M; noise]**.
- Eager op-to-op gaps fall from 12.5 ms at 128 to about 0.5 ms from 2048 up; trace removes them.

## Host round trip, H2D, D2H at 12 x 10 (`EMBEDDING_GEMMA2_PREFILL_HOST=1`) [M]

Mean ms; N is the round-trip count.

| bucket | host prep | H2D (bytes) | D2H embedding | D2H last hidden (bytes) | round trip | replay + sync | N |
|---|---|---|---|---|---|---|---|
| 128 | 0.078 | 0.019 (9 KB) | 0.044 | 0.129 (0.2 MB) | 5.036 | 4.954 | 20 |
| 256 | 0.083 | 0.017 (17 KB) | 0.042 | 0.128 (0.4 MB) | 6.035 | 5.952 | 20 |
| 512 | 0.104 | 0.020 (35 KB) | 0.036 | 0.199 (0.8 MB) | 7.346 | 7.299 | 20 |
| 1024 | 0.115 | 0.026 (70 KB) | 0.038 | 0.414 (1.6 MB) | 11.862 | 11.775 | 20 |
| 2048 | 0.171 | 0.028 (139 KB) | 0.036 | 0.824 (3.1 MB) | 25.455 | 25.366 | 20 |
| 4096 | 0.276 | 0.038 (279 KB) | 0.038 | 1.940 (6.3 MB) | 44.861 | 44.743 | 20 |
| 8192 | 0.487 | 0.048 (557 KB) | 0.038 | 4.455 (12.6 MB) | 104.280 | 103.942 | 20 |
| 16384 | 0.937 | 0.062 (1114 KB) | 0.037 | 12.235 (25.2 MB) | 226.930 | 228.205 | 14 |
| 32768 | 1.793 | 0.099 (2228 KB) | 0.038 | 37.154 (50.3 MB) | 578.638 | 590.827 | 6 |
| 65536 | 3.486 | 0.231 (4456 KB) | 0.043 | 116.022 (100.7 MB) | 1755.799 | 1783.646 | 5 |

The customer output is the embedding (0.04 ms D2H). Round trip = H2D + replay + embedding readback.

## Top three costs per bucket at 12 x 10 and owning stage [M]

Device ms per op family from `profiles_12x10` (share of burst wall, family floor from the op table),
grouping as in the 13 x 10 analysis (`e9/01r2b/scripts/analyze.py`, `analysis/r2b.json`):

| bucket | 1 | 2 | 3 |
|---|---|---|---|
| 128 | RMSNorm 1.73 ms (35 %): 05-memory | matmul 1.41 ms (29 %): 03-matmul | rotary 0.60 ms (12 %): 04-attention / 05-memory |
| 256 | RMSNorm 1.76 ms (30 %): 05-memory | matmul 1.46 ms (25 %): 03-matmul | SDPA 0.95 ms (16 %): 04-attention |
| 512 | RMSNorm 1.84 ms (25 %): 05-memory | matmul 1.80 ms (25 %): 03-matmul | SDPA 1.49 ms (20 %): 04-attention |
| 1024 | SDPA 3.05 ms (26 %): 04-attention | matmul 2.94 ms (25 %): 03-matmul | RMSNorm 2.06 ms (18 %): 05-memory |
| 2048 | SDPA 8.66 ms (34 %): 04-attention (2 waves at q_chunk 64) | matmul 8.11 ms (32 %): 03-matmul (110-core 2D) | RMSNorm 2.57 ms (10 %): 05-memory |
| 4096 | matmul 14.78 ms (33 %): 03-matmul (110-core 2D) | SDPA 11.83 ms (26 %): 04-attention | rotary 5.49 ms (12 %): 04-attention / 05-memory |
| 8192 | SDPA 31.44 ms (30 %): 04-attention | matmul 31.29 ms (30 %): 03-matmul (gelu gate on `minimal_matmul`) | RMSNorm 11.69 ms (11 %): 05-memory |
| 16384 | SDPA 79.90 ms (36 %): 04-attention | matmul 59.50 ms (26 %): 03-matmul | RMSNorm 22.17 ms (10 %): 05-memory |
| 32768 | SDPA 242.14 ms (43 %): 04-attention | matmul 123.90 ms (22 %): 03-matmul | mask build 58.33 ms (10 %): 02-topology / 05-memory |
| 65536 | SDPA 893.60 ms (52 %): 04-attention | matmul 250.43 ms (15 %): 03-matmul | mask build 242.01 ms (14 %): 02-topology / 05-memory |

The traced dispatch residual is never in the top three (<= 7.0 % of wall).

## Roofline (roofline session, not duplicated here)

Model, method and tables: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/roofline/` (`README.md`,
`roofline.py`, `out/table_text.md`, page linked in its README). The % of roofline column uses
`out/roofline_sdpa_nofp32acc.json` (sha256 `58935ff3ee4f...`, written 12:19:26, read unchanged after the
runs): roofline = max(FPU, SFPU, DRAM) for the text tower at 120 cores and 1.35 GHz, cold, every
activation on chip (optimistic from 16384, where activations exceed L1). "+SFPU" serializes the SFPU
term. The FPU term assumes 89 % of peak matmul throughput, and % of roofline is computed against
1.35 GHz while the long buckets ran at about 1.19-1.29 GHz sustained (at 65536, 1193 MHz: about 38 %
of a roofline scaled to the clock actually run, vs 33.8 % at 1.35 GHz) **[I: linear clock scaling]**. The op-table floor in `profiles_12x10` is a separate, simpler model (per op, padded tensors
once); the headline uses the roofline session's number.

## Known open item (not fixed here)

The E5a fallback text precision (`baseline_precision_config.json`) crashes at 12 x 10 in multi-bucket
device sessions <= 8192 (static CBs clash with L1 buffers at the 1 x 4096 full-attention SDPA).
Stage 5 owns it (`target_contract.json` `accuracy.open_items.baseline_precision_12x10`). The default
`sdpa_nofp32acc` is not affected (sweep: buckets <= 8192 in one session, 40 passed).

# Superseded geometry: stage 1 at 13 x 10 with `p3_qkv_gu` (2026-10-28 morning)

Everything below was measured on the P150a's full 13 x 10 grid (130 cores) with the earlier text
precision `p3_qkv_gu`. It is not comparable with the 12 x 10 contract numbers above (other grid,
other precision). Kept as measured; the harness and telemetry sections still describe the harness.

## Headline (latency_ms per traced warmed forward)

Sweep run `20261028-072751` (65536 burst and sustained: run `20261028-074659`, see Runs), records
`tt-ops/runs/embeddinggemma-2/e9/01-baseline/run/prefill.jsonl`, summary `perf_summary.json` **[M]**.
Power limit 150 W TDP (default), AICLK max 1350 MHz. Telemetry: median (min or max) over the last
third of the sustained run, sampled every 100 ms.

| bucket (logical) | burst median ms | burst avg / best | burst n | **sustained ms** | sust. first window | replays / s | sust. / burst | AICLK MHz (min) | power W (max) | current A | temp max C | tok/s/chip (sust.) | gate: last-hidden PCC / emb. cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 (96) | 4.813 | 4.814 / 4.813 | 20x5 | **4.809** | 4.810 | 6240 / 30 | 0.999 | 1350 (1350) | 77 (81) | 96 | 67.5 | 26,614 | 0.99170 / 0.99747 pass |
| 256 (192) | 5.474 | 5.474 / 5.474 | 20x5 | **5.470** | 5.471 | 5520 / 30 | 0.999 | 1350 (1350) | 86 (92) | 107 | 68.9 | 46,801 | **0.98872** / 0.99763 **FAIL** |
| 512 (384) | 6.834 | 6.834 / 6.833 | 20x5 | **6.829** | 6.830 | 4440 / 30 | 0.999 | 1350 (1343) | 96 (115) | 120 | 71.1 | 74,972 | 0.99366 / 0.99882 pass |
| 1024 (768) | 10.758 | 10.758 / 10.758 | 19x5 | **10.754** | 10.756 | 2820 / 30 | 1.000 | 1350 (1343) | 105 (123) | 132 | 73.0 | 95,222 | 0.99233 / 0.99911 pass |
| 2048 (1536) | 20.066 | 20.066 / 20.061 | 10x5 | **20.065** | 20.059 | 1500 / 30 | 1.000 | 1350 (1343) | 127 (157) | 160 | 76.4 | 102,067 | 0.99321 / 0.99927 pass |
| 4096 (3072) | 34.048 | 34.047 / 34.038 | 6x5 | **34.389** | 34.037 | 885 / 30 | 1.010 | 1337 (1325) | 135 (175) | 172 | 76.2 | 119,107 | 0.99387 / 0.99947 pass |
| 8192 (6144) | 92.970 | 92.977 / 92.918 | 3x5 | **93.568** | 92.956 | 324 / 30 | 1.006 | 1337 (1337) | 131 (195) | 165 | 73.8 | 87,551 | 0.99446 / 0.99954 pass |
| 16384 (12288) | 223.374 | 223.349 / 223.246 | 3x5 | **226.653** | 223.375 | 300 / 68 | 1.015 | 1325 (1312) | 131 (218) | 166 | 73.5 | 72,287 | 0.99496 / 0.99953 pass |
| 32768 (24576) | 639.600 | 639.528 / 638.980 | 3x5 | **651.746** | 637.209 | 300 / 196 | 1.019 | 1312 (1293) | 130 (238) | 166 | 75.1 | 50,277 | 0.99453 / 0.99945 pass |
| 65536 (49152) | 2049.762 | 2048.830 / 2041.619 | 3x5 | **2079.677** | 2046.885 | 300 / 624 | 1.015 | 1306 (1268) | 138 (233) | 176 | 77.0 | 31,513 | 0.99420 / 0.99937 pass |

- Gates are on the traced replay output after loading the real input, vs HF fp32 (bars 0.99 / 0.99)
  **[M]**. Replay equals eager bit for bit on every bucket (`replay_equals_eager_bitwise`).
- **b1_s256 fails the hard last-hidden PCC gate** (0.98872). See "Gate failure at b1_s256"; the
  gate was not changed and no override is applied.
- Short buckets (<= 2048) hold 1350 MHz and do not drift. From 4096 up the firmware lowers AICLK
  (medians 1306-1337 MHz, minimum 1268 MHz) with power around 130-138 W median (150 W TDP limit, 200 A TDC limit) and
  sustained is 0.6-1.9 % slower than burst **[M]**. Power samples peak above 150 W (to 238 W)
  for single 100 ms reads; the limit is enforced on an average **[I]**.
- The 65536 "burst" is not a full-clock figure: the clock drops inside the first window **[M]**.
  The sustained telemetry, which starts after the same 3 s idle as a burst window
  (`run/telemetry_20261028-072751_b1_s65536.csv`, `run/telemetry_20261028-074659_b1_s65536.csv`),
  reads 1350 MHz / 68-69 W at the start, 1343 MHz at 1 s and 1287 MHz at 2 s, within about one
  forward (2.05 s). The burst windows of run `20261028-074659` (3 forwards each) rise from 2041.6 to
  2055.2 ms (2041.6, 2046.3, 2049.8, 2051.3, 2055.2) **[M]**. The burst record's `telemetry_start`
  (1287 / 1306 MHz) is sampled after warm-up and before the first 3 s idle, so it is not the clock
  at window start.

## Harness

`tests/perf/prefill.py` (new). Quick default (`pytest .../prefill.py`): buckets 128 / 1024 / 8192,
burst only, 53 s pytest time with a warm cache **[M]** (`ctl_quick_end`). Opt-in switches (`"1"` = on):

| env | adds | cost **[M]** unless noted |
|---|---|---|
| (none) | buckets 128, 1024, 8192; burst | ~1 min |
| `EMBEDDING_GEMMA2_PREFILL_SHAPES=1` | all 10 contract buckets; 16384, 32768, 65536 each in its own device session | +~4 min with burst only **[I]** |
| `EMBEDDING_GEMMA2_PREFILL_HF_CHECK=1` | gate per bucket on the traced replay output vs cached HF fp32; replay vs eager | +~1 min with cached refs (gate run: all buckets, burst + gate, 261 s) |
| `EMBEDDING_GEMMA2_PREFILL_HOST=1` | host prep, H2D, D2H (embedding, last hidden), round trip | +~1 min **[I]** |
| `EMBEDDING_GEMMA2_PREFILL_SUSTAINED=1` | max(30 s, 300 replays) per bucket; 65536 alone 10.4 min | +~3 min (quick buckets) / +~25 min (all) **[I]** |
| `EMBEDDING_GEMMA2_PREFILL_TELEMETRY=1` | AICLK / power / current / vcore / temperature every 100 ms during sustained runs | ~0 |
| `EMBEDDING_GEMMA2_PREFILL_ALL=1` | all of the above | 18.7 min + 11.9 min for the 65536 rerun **[M]**; ~30 min in one run **[I]** |

Other env: `EMBEDDING_GEMMA2_PREFILL_BUCKETS` (comma list), `EMBEDDING_GEMMA2_PREFILL_REFS`
(reference cache), `EMBEDDING_GEMMA2_PREFILL_OUT` (records; default `generated/embedding_gemma2_prefill`),
`EMBEDDING_GEMMA2_PREFILL_TRACE_MB` (128), `EMBEDDING_GEMMA2_SUSTAINED_S` (30),
`EMBEDDING_GEMMA2_PREFILL_SUSTAINED_REPLAYS` (300), `EMBEDDING_GEMMA2_PREFILL_BURST_IDLE_S` (3).
Supporting files: `prefill_refs.py` (HF fp32 references, own process, optional RSS cap),
`prefill_telemetry.py` (sysfs reader + sampler process), `prefill_op_table.py` (per-op table with
roofline), `prefill_summary.py` (`perf_summary.json`). Profiling: `perf.py::test_tracy_forward`
now covers `b1_s<bucket>` for every contract bucket under the profiler (collected only under the
profiler, so `EMBEDDING_GEMMA2_PERF_ALL=1` still collects 65 ids **[M]**).

Definitions. Inputs: `tests/common.make_batch` corpus at offset 0, logical length 3/4 of the bucket,
right-padded into the bucket. One trace per bucket (`TextEncoder.capture_all`), device sessions:
buckets <= 8192 share one (max_seq_len 8192); 16384, 32768, 65536 have their own. Phase order per
session: gate, burst, host, sustained. Burst: 5 windows, each after 3 s idle, of n back-to-back
`execute_trace` replays and one sync (n = 200 ms / forward, clamped 3-20). Sustained: back-to-back
replays in ~500 ms windows (>= 1 replay) with one sync each for max(30 s, 300 replays); median of
the windows in the last third.

Tracing evidence **[M]**: every bucket captured in a 128 MiB trace region, 65536 included (the
65536 trace fits; no blocker). Allocation tracking: the gate run used
`TT_METAL_TRACE_ALLOC_TRACKING=1` on all 10 buckets (`run_gate/prefill.jsonl`, `logs/gate_alloc_tracking.log`);
no unsafe allocation was flagged at `execute_trace` (which fails when one survives), and every
gate value is exactly equal (float equality) to the untracked sweep's.

## Telemetry (sysfs)

`/sys/class/tenstorrent/tenstorrent!0/tt_aiclk` and the driver's hwmon (`power1_input`,
`curr1_input`, `in0_input`, `temp1_input`) are plain file reads: they work while the benchmark holds
the card (read at 1350 MHz / 115 W during a 1024 burst, smoke run) **[M]**. Safety A/B at 1x2048,
sustained, same session order, sampler off then on (`run_controls`, runs `20261028-080501` /
`-080558`): burst 20.065 / 20.065 ms, sustained 20.059 / 20.062 ms (+0.015 %) **[M]**: no measurable
effect. tt-smi was not used during runs (not needed); its snapshot before the stage gives firmware and
limits (`logs/tt_smi_snapshot_pre.json`).

## Profiles and per-op tables

One warmed eager forward per bucket between `PERF_TEXT` / `PERF_TEXT_END` signposts under Tracy
(`perf.py::test_tracy_forward[b1_s<bucket>]`, separate process per bucket; full-length rows: the
device work of a bucket does not depend on the logical length, since shapes, dense masks and the
block map are static **[I: from tt/masks.py]**). Tables:
`profiles/b1_s<bucket>/op_table.md` (+ `.json`, tt-perf-report `forward_perf_report.csv`); raw ops
CSVs (2.5 MB each) in `tt-ops/runs/embeddinggemma-2/e9/01-baseline/tracy/b1_s<bucket>/`.
Roofline columns are a model **[I]** (definitions in `tests/perf/prefill_op_table.py`): 130 cores,
4096 / 2048 / 1024 FLOP per cycle per core for LoFi / HiFi2 / HiFi4
(`tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`), 512 GB/s DRAM (8 x 32-bit GDDR6 at 16 GT/s), 64 B per cycle
per core for L1 operands, 1350 MHz, padded tensors moved once; SDPA counts the needed band
(1025 keys) for sliding layers. Profile model build: `EMBEDDING_GEMMA2_MAX_SEQ_LEN` 32768 (65536 for
the 65536 bucket); the timed sessions use 8192 for buckets <= 8192. Only RoPE table size differs;
the rotary op reads only its rows **[I]**, and kernel sums sit 1-7 % under the traced wall (below).

## Reconciliation: wall vs device kernel sum, gaps and host work

| bucket | ops | burst wall ms | device kernel sum ms | traced residual ms (share, per op) | eager op-to-op gaps ms | host work ms (round trip - replay) | round trip ms | roofline floor ms | wall / floor |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 645 | 4.813 | 4.454 | 0.359 (7.5 %, 0.56 us) | 12.995 | 0.080 | 4.935 | 0.319 | 15.1 |
| 256 | 645 | 5.474 | 5.120 | 0.354 (6.5 %, 0.55 us) | 10.744 | 0.083 | 5.596 | 0.362 | 15.1 |
| 512 | 645 | 6.834 | 6.519 | 0.315 (4.6 %, 0.49 us) | 9.508 | 0.095 | 6.968 | 0.558 | 12.2 |
| 1024 | 646 | 10.758 | 10.536 | 0.223 (2.1 %, 0.34 us) | 5.530 | 0.082 | 10.877 | 1.380 | 7.8 |
| 2048 | 646 | 20.066 | 19.827 | 0.238 (1.2 %, 0.37 us) | 0.457 | 0.109 | 20.215 | 2.991 | 6.7 |
| 4096 | 646 | 34.048 | 33.570 | 0.478 (1.4 %, 0.74 us) | 0.616 | 0.062 | 34.200 | 8.752 | 3.9 |
| 8192 | 646 | 92.970 | 92.333 | 0.637 (0.7 %, 0.99 us) | 0.602 | 0.430 | 93.477 | 35.807 | 2.6 |
| 16384 | 646 | 223.374 | 222.101 | 1.273 (0.6 %) | 0.609 | -0.10 (noise) | 224.291 | 86.967 | 2.6 |
| 32768 | 646 | 639.600 | 636.357 | 3.244 (0.5 %) | 0.571 | -1.50 (noise) | 641.832 | 236.172 | 2.7 |
| 65536 | 646 | 2049.762 | 2013.640 | 36.123 (1.8 %) | 0.560 | -3.70 (noise) | 2074.349 | 722.115 | 2.8 |

- wall = device kernel sum + traced residual. Kernel sum and eager gaps are from the eager Tracy
  forward **[M]**; the traced residual is wall minus kernel sum **[M by difference]**. Traced
  op-to-op gaps are not profiled (no profile of a trace replay), so the residual holds dispatch gaps,
  the one sync per window, and any clock difference between the cold profiled forward and the timed
  window **[I]**. At <= 8192 it is 0.3-1.0 us per op, i.e. dispatch; at 65536 the 36 ms follows the
  lower clock (AICLK falls from 1350 to about 1287 MHz within the first forward of a window, per the
  sustained telemetry after the same idle, vs a cold profile) **[I]**.
- Eager op-to-op gaps (host dispatch in eager mode) are 13 ms at 128 and fall under 1 ms at >= 2048,
  where the device is the bottleneck; trace removes them (traced residual 0.36 ms at 128) **[M]**.
- Host work per request after host prep (round trip minus replay+sync) is 0.06-0.43 ms up to 8192;
  above 8192 it is inside run-to-run noise (negative) **[M]**.

## Host round trip, H2D, D2H (`EMBEDDING_GEMMA2_PREFILL_HOST=1`) [M]

Mean ms, N = 20 (round trip / replay: N = 14 at 16384, 5 at 32768 and 65536).

| bucket | host prep | H2D (bytes) | D2H embedding [1, 768] | D2H last hidden (bytes) | round trip | replay + sync |
|---|---|---|---|---|---|---|
| 128 | 0.087 | 0.020 (9 KB) | 0.044 | 0.093 (0.2 MB) | 4.935 | 4.855 |
| 256 | 0.078 | 0.017 (17 KB) | 0.038 | 0.123 (0.4 MB) | 5.596 | 5.513 |
| 512 | 0.098 | 0.021 (35 KB) | 0.036 | 0.683 (0.8 MB) | 6.968 | 6.873 |
| 1024 | 0.134 | 0.025 (70 KB) | 0.036 | 0.355 (1.6 MB) | 10.877 | 10.795 |
| 2048 | 0.178 | 0.027 (139 KB) | 0.037 | 0.792 (3.1 MB) | 20.215 | 20.106 |
| 4096 | 0.282 | 0.036 (279 KB) | 0.036 | 1.706 (6.3 MB) | 34.200 | 34.138 |
| 8192 | 0.528 | 0.046 (557 KB) | 0.036 | 3.704 (12.6 MB) | 93.477 | 93.047 |
| 16384 | 0.879 | 0.063 (1.1 MB) | 0.038 | 11.396 (25.2 MB) | 224.291 | 224.394 |
| 32768 | 1.727 | 0.098 (2.2 MB) | 0.043 | 51.855 (50.3 MB) | 641.832 | 643.329 |
| 65536 | 3.478 | 0.182 (4.5 MB) | 0.043 | 100.403 (100.7 MB) | 2074.349 | 2078.049 |

H2D is ids (uint32) + keep mask (bf16, tile-padded to 32 rows); round trip = H2D + replay + embedding
readback. The customer output is the embedding (0.04 ms); reading the full last hidden state costs up
to 100 ms at 65536 (5 % of the forward). PCIe x16 Gen5 on this card.

## Top three costs per bucket and owning stage

From the per-op tables grouped by op family (device ms, share of burst wall, family roofline floor)
**[M]**, owners per the prefill-optimization stage table:

| bucket | 1 | 2 | 3 |
|---|---|---|---|
| 128 | RMSNorm 1.73 ms (36 %), floor 0.01: 05-memory | matmul 1.33 ms (28 %), floor 0.30: 03-matmul | rotary 0.59 ms (12 %), floor 0.00: 04-attention (memory-bound op, 05-memory) |
| 256 | RMSNorm 1.75 ms (32 %): 05-memory | matmul 1.38 ms (25 %): 03-matmul | SDPA 0.81 ms (15 %), floor 0.02: 04-attention |
| 512 | RMSNorm 1.83 ms (27 %): 05-memory | matmul 1.65 ms (24 %): 03-matmul | SDPA 1.54 ms (23 %): 04-attention |
| 1024 | SDPA 3.11 ms (29 %), floor 0.33: 04-attention | matmul 2.61 ms (24 %), floor 0.81: 03-matmul | RMSNorm 1.94 ms (18 %): 05-memory |
| 2048 | SDPA 7.95 ms (40 %), floor 0.86: 04-attention | matmul 6.19 ms (31 %), floor 1.61: 03-matmul | RMSNorm 2.17 ms (11 %): 05-memory |
| 4096 | SDPA 11.62 ms (34 %), floor 2.49: 04-attention | matmul 11.13 ms (33 %), floor 3.21: 03-matmul | rotary 3.05 ms (9 %), floor 0.66: 04-attention / 05-memory |
| 8192 | SDPA 34.25 ms (37 %), floor 8.64: 04-attention | matmul 24.89 ms (27 %), floor 8.20: 03-matmul | eltwise 9.91 ms (11 %), floor 7.23: 05-memory (fusion: 02-topology) |
| 16384 | SDPA 106.78 ms (48 %), floor 32.46: 04-attention | matmul 46.97 ms (21 %), floor 16.12: 03-matmul | RMSNorm 17.86 ms (8 %), floor 13.12: 05-memory |
| 32768 | SDPA 372.76 ms (58 %), floor 125.66: 04-attention | matmul 101.75 ms (16 %), floor 31.98: 03-matmul | mask build 54.25 ms (8 %), floor 3.55: 04-attention (mask-free padding) / 05-memory |
| 65536 | SDPA 1388.11 ms (68 %), floor 494.29: 04-attention | mask build 216.82 ms (11 %), floor 14.18: 04-attention / 05-memory | matmul 198.96 ms (10 %), floor 63.69: 03-matmul |

- Short buckets (<= 512) are norm-shaped: at 128 the RMSNorm rows run on 4-16 cores at 176-626x
  their modelled floor; 645 ops in 4.8 ms.
- SDPA dominates from 1024 up; at 65536 the 4 full-attention layers (head dim 512) take 1289 ms of
  the 1388 ms **[M]**.
- "mask build" is the two per-forward bfp4 mask adds (`masks.py`: sliding = band + key bias, full =
  zeros + key bias), [S, S] each: 191.5 + 25.3 ms at 65536 **[M]**. It scales as S^2 and is the
  second cost at 65536.
- The traced dispatch residual is never in the top three (<= 7.5 % of wall).

## Gate failure at b1_s256 (open; orchestrator decision)

Observation **[M]**: last-hidden PCC 0.98872 < 0.99 at b1_s256 (logical 192, corpus offset 0);
embedding cosine 0.99763 passes. Same value in the tracked gate run and the sweep; replay equals
eager bitwise, so not a trace issue.

Investigation (`tt-ops/.../scripts/gate256_probe.py`, eager vs HF fp32, `gate256_probe.jsonl`)
**[M]**:
- Selected `p3_qkv_gu`, offset 0: L 160 0.99020, 176 0.99133, 192 0.98872, 200 0.98836, 208 0.99170,
  224 0.99089; offsets 997 and 5003 at the same lengths: >= 0.9955. The corpus start (model card)
  is the hard input; worst-token cosine there is 0.35-0.90 (selected) vs 0.71-0.96 (baseline).
- Control: the E5a baseline precision (`baseline_precision_config.json`, bf16 activations) on the
  same inputs: >= 0.99677 everywhere (192: 0.99692).
- Cause: the selected precision (bfp8 QKV and gate/up outputs) on this input **[I: the controls
  isolate precision; the per-op source was not bisected]**. 0.98872 is 2.3e-3 below the E6a probe
  minimum for this config (0.99106), so the E6a probe never covered this input.
- Cost of the control config, burst, same card (`run_baseline_precision`, run `20261028-080654`):
  128 4.924 (+2.3 %), 256 5.837 (+6.6 %), 512 7.313 (+7.0 %), 1024 12.372 (+15.0 %), 2048 22.057
  (+9.9 %), 4096 39.527 (+16.1 %), 8192 105.575 (+13.6 %) ms; all its gates pass (0.99677-0.99799).

Options (not taken here: stage 1 changes no model code, precision is stage 7, the contract gate is
the orchestrator's): (a) a stage-7 precision change (per-bucket or per-layer exception) that passes
256 and is re-probed; (b) revert the text precision to the baseline at the measured cost; (c) a
`pcc_gate_override` for b1_s256 with this measured reason. `target_contract.json` carries no override.

## HF fp32 references: host RAM and time [M]

`prefill_refs.py`, one process per bucket under `/usr/bin/time -v` (`logs/ref_s<bucket>.log`),
12 torch threads, cached in `tt-ops/runs/embeddinggemma-2/e9/refs/`.

| bucket (logical) | forward s | peak RSS GB |
|---|---|---|
| 128-4096 (96-3072) | 0.07-2.04 | 3.2 |
| 8192 (6144) | 6.4 | 3.2 |
| 16384 (12288) | 20.3 | 5.4 |
| 32768 (24576) | 69.4 | 13.7 |
| 65536 (49152) | 248.3 (4 min 15 s wall) | 44.5 |
| 65536 full length (65536; not a contract input) | stopped at 25.5 s by the 52 GB RSS cap | > 52.4 |

The contract input at 65536 (49152 tokens) has an HF fp32 reference on this 61 GB host, so no gate
override is needed there. A full 65536-token reference exceeds a 52 GB cap (E5a: 24 GB at 32768
tokens); it would need more host RAM or a streaming-attention reference **[M / I]**.

## Controls and drift

- End-of-session control, plain default run (`ctl_quick_end`, run `20261028-080858`): 128 4.813,
  1024 10.759, 8192 92.944 ms vs sweep 4.813 / 10.758 / 92.970: within 0.03 % **[M]**.
- The gate run (alloc tracking on, before the sweep) burst numbers match the sweep within 0.015 %
  up to 16384, -0.14 % at 32768 and -0.03 % at 65536 **[M]** (`run_gate/prefill.jsonl`).
- 65536 burst in two runs: 2057.2 and 2049.8 ms (0.4 %) **[M]**; in both the clock drops inside
  the first window (see Headline) **[I: from the sustained telemetry after the same idle]**.

## Runs

| run | what | result |
|---|---|---|
| `20261028-072117` | smoke: quick buckets, gate + host | 9 passed |
| `20261028-072237` (`gate_alloc_tracking`) | all buckets, gate + burst, `TT_METAL_TRACE_ALLOC_TRACKING=1` | 19 passed, 1 failed (b1_s256 gate) |
| `20261028-072751` (`sweep_all`) | `EMBEDDING_GEMMA2_PREFILL_ALL=1` | 38 passed, 2 failed: b1_s256 gate; b1_s65536 sustained hit the repo's 300 s pytest-timeout (harness limit, not a device hang; the device closed cleanly) |
| `20261028-074659` (`sweep_s65536_sustained`) | 65536 burst + sustained after adding `@pytest.mark.timeout(3600)` | 2 passed |
| tracy `b1_s128` ... `b1_s65536` | one warmed eager forward per bucket | 10 x exit 0 |
| `20261028-080501` / `-080558` | telemetry A/B at 2048 | 2 + 2 passed |
| `20261028-080654` | baseline precision, buckets <= 8192, gate + burst | 14 passed |
| `20261028-080858` | end control, plain default | 3 passed |
| regression | `perf.py` default 4 passed; `test_trace.py` + `test_full_model_trace.py` 17 passed; `pytest tests --collect-only` 362 collected | exit 0 |

Exact commands, logs and scripts: `work_log.md`.
