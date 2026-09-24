# pplx-embed-v1-4B on Blackhole P150 — what landed and what it bought

Companion to `NEGATIVE_RESULTS.md` (what was tried and failed, with numbers). Branch
`arg/pplx-embed-upstream` in `tt-metal/`; model code `models/demos/blackhole/pplx_embed_4b/`.
All numbers: ISL 512, demo extended-trace timing (forward + pooling + I/O in one traced replay),
best of 10 iterations, same-chip sequential A/B for every landing (chips 4 / 7 / 8 / 6 for bs1 / 8 / 16 / 32).

## Where it stands (2026-09-24)

| batch | H200 | start of effort | cold best (best of 10) | × H200 | sustained (median of iterations 5–9) | × H200 | 3× target |
|---|---|---|---|---|---|---|---|
| 1 | 5.437 ms | 25.9 ms | **17.6** | 3.24× | 18.3 | 3.37× | 16.3 |
| 8 | 33.081 | 156.4 | **115.3** | 3.49× | 121.0 | 3.66× | 99.2 |
| 16 | 67.225 | 290.9 | **221.0** | 3.29× | 228.2 | 3.39× | 201.7 |
| 32 | 139.150 | 557.8 | **425.5** | 3.06× | 451.1 | 3.24× | 417.5 |

(Measured 2026-09-24 with the shipped defaults after the concat-free SDPA landing, one 30-iteration run per
batch size on chips 4 / 7 / 8 / 6; sustained here is the median of iterations 15–29.)

"Cold best" is the first iterations on a cool chip at the full 1.35 GHz AICLK; "sustained" is after the
board's power manager settles the clock at ≈1.1–1.2 GHz (see the note below). The H200 figures are
taken as steady-state numbers, so the sustained column is the like-for-like comparison unless the
H200 harness also reported a cold first iteration.

Accuracy: STS-B Spearman 0.8135 → **0.8161** on the bs1 path (`eval_accuracy_tt.py`); through the
batched paths (`eval_accuracy_batched.py`, fixed ISL 512, masked mean) bs1 0.8121 / bs8 0.8123 /
bs16 0.8140 / bs32 0.8159. Every landing was re-checked on STS-B or per call against the stock ops.

¹ ² **Cold vs sustained.** "Best of 10" is the cold-chip number: tt-smi sampled during bs16 runs shows
AICLK at 1281–1350 MHz for iterations that read 216.7 ms and 1112–1162 MHz for iterations that read
232–243 ms; the board's power manager pulls the clock down ≈ 0.6 s into a sustained load (iteration 3
of a cold run) and deeper as the chip warms. No power or clock setting was changed (`AICLK_LIMIT_MAX`
1350, board defaults). bs1 barely throttles. Sustained with the shipped defaults: bs1 18.3, bs8 121.0, bs16 228.2, bs32 451.1 ms (30-iteration runs); before today's bs>1 rounds 18.0 / 126.7 / 235 / 452.3 — so at bs32 the new defaults only help the cold iteration (the chip is power-bound at the
sustained clock), at bs8 they hold −2.3% sustained, at bs16 ≈ −1.5%.

## Landings, in order (e2e ms bs1 / bs8 / bs16 / bs32 after each)

| commit | change | mechanism | after |
|---|---|---|---|
| 08fd9ea, fb45b51 | prefill config baseline (09-22) | `minimal_matmul` subblock 1×8 (bs8 −12%, bs16 −13%, bs32 −18%); `in0_block_w` cap 8 → 38 for the bs1 legacy matmuls (FF2 35% → 70% of roofline, −4.4 ms); block-sharded LayerNorm at bs1 (−1.9 ms); SDPA chunks 512/256 (−2…4% batched) | 25.9 / 156.4 / 290.9 / 557.8 |
| e02bd99, 81691379 | head-split QKV / concat as model-local `generic_op`s | reader/writer kernels scatter the fused QKV activation straight into `[B, H, S, d]` heads (bs1 −2.7 ms, bs32 −8.8 ms) | (included above) |
| d9cb8ee | time the extended trace | the timed loop had fallen back to the Generator path; pooling + I/O now inside one replay | 25.2 / 155.6 / 288.7 / 543.5 |
| 355d90c | fused head-split + Q/K RMSNorm | one `generic_op` replaces create_heads + q_norm + k_norm (three DRAM passes → one) | 25.0 / 144.7 / 276.8 / 519.2 |
| 36b3972 | RoPE fused into the same op | cos/sin/transform tiles applied per head inside the kernel; two `rotary_embedding_llama` ops gone | 23.9 / 143.4 / 263.5 / 495.0 |
| 48269f7 | fused op emits Q and K/V in bfp8 | packer converts on the way out; the Typecast before SDPA is deleted, SDPA reads half the bytes | 23.7 / 135.3 / 250.4 / 474.3 |
| b711bee | QKV projection writes bfp8 | the matmul output feeding the fused op is bfp8 too (26 MB less traffic per layer at bs8) | 23.7 / 127.5 / 240.9 / 456.7 |
| 0502099 | merged core ranges, fewest-cores split | per-core `CoreRange`s cost ≈ 0.4 µs/core per launch; rectangles + the smallest core set with the same per-core max | 23.4 / 126.7 / 240.6 / 455.8 |
| 755482c | batched SDPA 12×10, k-chunk 512 | all 120 workers, one K chunk (bs32 932 → 806 µs/call) | 23.3 / 126.2 / 239.6 / 450.6 |
| 980976c | fused residual add + RMSNorm (bs16+) | one `generic_op` emits the sum (residual stream) and the normalised tensor: 4 DRAM passes → 2 | 23.3 / 126.2 / 234.8 / 443.5 |
| e9b0500, 977850 | bs16 fused-SwiGLU blocks 4,20,8 / 1×4 | in-model block sweep of the packed-w13 `minimal_matmul` | 23.3 / 126.2 / 228.3 / 443.5 |
| 308438e | bs8 blocks FF2 16,8,8 · QKV 8,4,8 · WO 16,8,8 | same sweep for the plain matmuls | 23.3 / 123.4 / 228.3 / 443.5 |
| d5012e1 | `silu_mul` at bs32 | SwiGLU product as one `generic_op` (dest-reuse multiply after `silu_tile`), −14% on the op | 23.3 / 123.4 / 228.3 / 438.1 |
| 213530d | bs1: 12×8 matmul grids, coalesced weight reads, SDPA q256 | see below | **17.7** / 123.4 / 228.3 / 438.1 |
| 9441a6c | bs>1: row-split add+RMSNorm, SDPA 12×8 at bs8, interleaved weights at bs32 | see below | 17.7 / **118.5** / **216.7** / **428.1** |
| sdpa concat-out | SDPA writes `[B, 1, S, H·d]` directly (`output_heads_concat`), concat pass gone at bs>1 | tile-id remap in the SDPA writer; bit-identical; SDPA +3–7% vs concat −68…−350 µs | 17.7 / **115.4** / **221.9** / **430.1** (same-chip A/B arms) |

## The optimizations, by mechanism

**Fusions (model-local `generic_op`s under `tt/custom_ops/`).**
- `fused_qkv_heads_norm`: head split + per-head Q/K RMSNorm + RoPE in one pass, Q/K/V emitted in bfp8. Replaced five ops per layer.
- `fused_add_rmsnorm`: residual add + RMSNorm emitting both outputs. Row-granular kernel at first (one tile-row per core; bs16/32 only), then the row-split kernel for every batch: each row is split over R cores that exchange partial mean-squares over the NoC (fixed peer groups, one CB slot per wave, monotonic semaphore). Stock add + rms_norm → split: 184 → 141 µs (M=4096), 320 → 242 (M=8192), 597 → 459 (M=16384).
- `silu_mul`: `silu(gate) * up` as one op at bs32, where the fused-SwiGLU matmul loses more than the multiply costs.
- `fused_concat_heads`: SDPA output back to `[B, S, H·d]` in one pass.

**Data types.** Weights bfp4 (LoFi), residual stream and activations bfp8, Q/K/V bfp8 straight out of the fused op, QKV projection output bfp8. STS-B went up, not down, through these.

**Matmul configuration.**
- Batched (M ≥ 4096): `minimal_matmul` on 12×10 with per-projection blocks from in-model sweeps; fused-SwiGLU packed w13 at bs8/16; interleaved weights for QKV/WO/W1/W3 at bs32 (2–3% faster than width-sharded from M=8192 up).
- bs1 (M=512): the legacy 2D-multicast kernel on **12×8 = 96 cores**. It had been pinned to 8×8 because grids with more columns than the 8 DRAM banks returned inf — a program-factory bug (a column took a whole bank stripe when its `per_core_N` was smaller); the one-line fix makes 12×8 bit-identical to 8×8: QKV 67.5 → 49.8 µs, WO 48.7 → 40.1, FF1/FF3 110.9 → 79.5, FF2 101.8 → 78.8. The DRAM-sharded in1 reader also now issues one multi-burst NoC read per block-row segment instead of one 576-byte request per tile (QKV 70.3 → 67.8, FF2 111.6 → 102.8 at 8×8). bs1 21.4 → 17.7 ms from the grids alone.

**SDPA.** Streaming kernel (fp32 acc off), LoFi, exp approximation, bfp8 operands, non-causal GQA. bs1: q256/k256 on 8×8 fills the grid (79 → 55 µs/call). bs8: 12×8 q512/k512 (272 → 234 µs). bs16/32: 12×10 q512/k512.

**Launch cost.** Merged rectangular core ranges and the fewest-cores split for the generic ops (≈ 0.4 µs per extra core per launch).

**Measurement fixes that changed conclusions** (details in `NEGATIVE_RESULTS.md` §0, §16, §17, §46): the timed path had silently fallen back to the Generator; `DEVICE FW DURATION` overstated op costs; a loaded P150 runs at 1.09–1.14 GHz so profiler µs are ~20% optimistic for batched runs; kernel-source edits are picked up by any process started later; bs16 is bimodal per launch; bs32's best-of-10 is the cold first iteration.

## Sustained clock: how to read and how to measure

The board's power manager holds AICLK at ≈1.1 GHz (bs32), ≈1.19 GHz (bs8/16) under sustained load
against 1.35 GHz cold; power sits at ≈160 W with 176–185 W peaks. The bs32 iteration slows only 4.4%
for a 21% clock drop, so ≈ 80% of it is DRAM-traffic-bound and not helped by core efficiency or by
using fewer cores (every fewer-cores probe lost, §48 of the negatives file). What carries over to
the sustained number: the fused add+RMSNorm (bs16 −2.1% sustained), the bs8 changes (−2.3%
sustained), bs1 everything. What does not: the bs32 weight layout and SDPA grid (cold-only).
Measure with `sustained_run.sh <bs> <chip> 30 <tag> "<ENV>"` (median of iterations 15–29 plus the
sampled clock and power) and compare arms on one chip.

## Where the gates live

`demo/_common.py::apply_workload_env(batch_size, seq_len)` sets every per-batch default with
`os.environ.setdefault`, so any knob can be overridden from the shell for an A/B:
bs1 → `QWEN_SDPA_Q_CHUNK/K_CHUNK=256`, `QWEN_QKV_GRID_X=12`, `QWEN_LEGACY_GRID_{FF13,FF2,WO}=12,8`,
`QWEN_LEGACY_TIGHT_PER_CORE_N=1`, per-shape `QWEN_LEGACY_SUBBLOCK_K<k>_N<n>`;
bs8 → `QWEN_SDPA_GRID=12,8`, `QWEN_SDPA_Q_CHUNK=512`, `QWEN_MM_BLOCK_{FF2,QKV,WO}`,
`QWEN_FUSED_ADD_NORM_MIN_ROWS=4096`, `QWEN_FUSED_ADD_NORM_R=5`;
bs16 → `QWEN_MM_BLOCK_FF13=4,20,8`, `QWEN_MM_SUBBLOCK_FF13=1,4`, `QWEN_FUSED_ADD_NORM_R=5`;
bs32 → `QWEN_FUSED_ADD_NORM_R=4`, `QWEN_WEIGHT_INTERLEAVED_K{2560_N6144,4096_N2560,2560_N9728}=1`;
all batches → `QWEN_FUSED_HEADS_NORM=1`, `QWEN_FUSED_ROTARY=1`, `QWEN_FUSED_Q_BFP8=force`,
`QWEN_FUSED_KV_BFP8=1`, `QWEN_QKV_OUT_BFP8=1`, `QWEN_FUSED_ADD_NORM=1`, `QWEN_SILU_MUL=1` (bs32 rows).

## What is left, and who has it

- `minimal_matmul` (K=2560, N=9728) at 332 TFLOP/s vs 473 for (K=9728, N=2560): ≈ −53 ms at bs32, −6 ms at bs8 — tenstorrent/tt-metal#57626 (Sankar Manoj).
- SwiGLU epilogue without halving the accumulation subblock: ≈ −13 ms at bs32 — #57627 (Sankar Manoj).
- SDPA: bs1 fills 64 of 120 cores (≈ −0.5…−0.9 ms); batched static CBs block L1-resident activations — #57628 (Chris Maryan).
- Interleaved `rms_norm` at ~50% of DRAM bandwidth (the remaining half of the LayerNorm cost) — #57629 (Chris Maryan).
- Upstreaming the 2D matmul factory fix and coalesced reads — #57630 (Chris Maryan).
- Sustained regime (§48–§49 of the negatives file): SDPA writing `[B, S, H·d]` directly is **done** (§49, −0.9…−2.0% cold, −1.2% sustained at bs32); the SwiGLU product in FF2's in0 path was retracted (12× redundant SFPU work per row group, see #57627); head-major output / in0 tile-id remap in `minimal_matmul` (≈ −10 ms bs32) — #57722 (Sankar Manoj), only worth it together with SDPA head offsets.
- Model-side: fused heads + concat ops are 7.7% of bs8; the bs16 two-mode band needs an explanation before bs16 gains can be claimed.

Tools: `github_issues/` (issue bodies with repro scripts), scratch `ab_one.sh` /
`ab_multi.sh` (same-chip A/B, alternating multi-launch for bs16), `eval_accuracy_batched.py`.
