# Prefill stage 1: baseline and target contract (E9)

Text-encoder prefill of `google/embeddinggemma-2` on one Blackhole P150a, batch 1, buckets 128 to
65536, masked path, selected precision `p3_qkv_gu`. Contract: `../target_contract.json`. Labels:
**[M]** measured (command and record named), **[I]** inferred. No model code changed in this stage.

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
- At 65536 the clock is already at 1287-1306 MHz when the burst windows start (telemetry at burst
  start), so its "burst" is not a full-clock number **[M]**; one forward (2 s) is longer than the
  3 s idle window lets the clock recover **[I]**.

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
  lower clock (1287-1306 MHz during that burst vs a cold profile) **[I]**.
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
- Cause: the selected precision (bfp8 QKV and gate/up outputs) on this input, inside the 1.1e-3
  margin the E6a probe measured for it (min 0.99106) **[I: the controls isolate precision; the
  per-op source was not bisected]**.
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
- 65536 burst in two runs: 2057.2 and 2049.8 ms (0.4 %), clock already lowered in both **[M]**.

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
