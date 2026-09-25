# pplx-embed-v1-4B on Blackhole P150 — what landed and what it bought

Companion to `NEGATIVE_RESULTS.md` (what was tried and failed, with numbers). Branch
`arg/pplx-embed-upstream` in `tt-metal/`; model code `models/demos/blackhole/pplx_embed_4b/`.
All numbers: ISL 512, demo extended-trace timing (forward + pooling + I/O in one traced replay),
best of 10 iterations, same-chip sequential A/B for every landing (chips 4 / 7 / 8 / 6 for bs1 / 8 / 16 / 32).

## Where it stands (2026-09-24)

| batch | H200 | start of effort | cold best (best of 10) | × H200 | sustained (median of iterations 5–9) | × H200 | 3× target |
|---|---|---|---|---|---|---|---|
| 1 | 5.437 ms | 25.9 ms | **17.5** | 3.22× | 17.7 | 3.26× | 16.3 |
| 8 | 33.081 | 156.4 | **115.3** | 3.49× | 121.0 | 3.66× | 99.2 |
| 16 | 67.225 | 290.9 | **221.0** | 3.29× | 228.2 | 3.39× | 201.7 |
| 32 | 139.150 | 557.8 | **425.5** | 3.06× | 451.1 | 3.24× | 417.5 |

(Measured 2026-09-24 with the shipped defaults after the concat-free SDPA landing — bs1 after the bs1 op-count landing below, same method — one 30-iteration run per
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
1350, board defaults). bs1 barely throttles. Sustained with the shipped defaults: bs1 17.7, bs8 121.0, bs16 228.2, bs32 451.1 ms (30-iteration runs); before today's bs>1 rounds 18.0 / 126.7 / 235 / 452.3 — so at bs32 the new defaults only help the cold iteration (the chip is power-bound at the
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
| bs1 op-count (09-24) | SDPA `output_heads_concat` at bs1 too (SDPA 57.3 → 54.1 µs standalone, the 4.6 µs model-local concat op gone) + the residual adds written in the norm's 10×8 block-shard layout (the 72 I2S ops become no-ops) | from Gio's B1 shard-layout idea; bit-identical; the residual item is neutral alone, −0.3 ms median together (NEGATIVE_RESULTS §50) | **17.5** / 115.3 / 221.0 / 425.5 |
| bs1 resid-sharded on every layer (09-24) | the §50 residual-shard item only applied to even layers: `supported(x, x)` in `decoder_fusion.py` rejects the sharded input the previous layer handed over, so odd layers fell back to interleaved adds + a real I2S before each norm (36 of 72 left). Now every layer; I2S 36 → 1 | bit-identical, STS-B 0.8161; device replay 16.97 → 16.80 ms; e2e bs1 −0.07 ms (mean of 3 alternating 30-iteration A/B pairs, 17.285 → 17.218; 8× p150b host, chip 0); bs>1 untouched | bs1 −0.4% |
| bs1 SDPA GQA packing (09-24) | new SDPA arg `pack_gqa_heads`: the 4 Q heads sharing a KV head are scheduled as one head of 4·S rows (same memory), so each KV head's K/V streams once down one 8-core chain instead of 4× over 2-core chains; k_chunk 256 → 512 (one K chunk now wins). Default at bs1 (`QWEN_SDPA_GQA_PACK=1`) | SDPA 49.7 → 35.9 µs/call in-model; device replay −0.49 ms; e2e 17.19 → 16.70 ms (3 alternating 30-it A/B pairs, 8× p150b host, chip 0); fixed-512 STS-B 0.8121 → 0.8133, bucketed 0.8161; packed at k256 is bit-identical to unpacked. Same 64-core grid (one Q chunk per core); the gain is the removed K/V re-streaming | bs1 −2.8% |
| bs1 resident heads-op constants (09-24) | the fused heads op takes cos/sin, the rotation tile, scaler and eps from a per-core L1 shard shared by all layers (CBs aliased via `cb_descriptor_from_sharded_tensor`) and reads gamma after the first unit, so compute starts its first unit without waiting on those reads (`QWEN_FUSED_RESIDENT_CONSTS=1`) | bit-identical; heads op 45.0 → 41.4 µs median in-model (46.5 → ~41 standalone); e2e 16.691 → 16.587 ms (3 alternating 30-it A/B pairs, 8× p150b host, chip 0). Preloading only the per-call constants, or only cos/sin, measured ≈0: the gain needs both off the first unit's path | bs1 −0.6% |
| bs1 heads-op compute v3 (09-24) | `compute_qkv_heads_norm_bs1.cpp` (was `_v3.cpp`; `QWEN_FUSED_COMPUTE_V3=1`, bs1 only): each norm / RoPE phase runs once per unit over its 4 Q + 1 K heads (contiguous in the unit) instead of once per head, so reconfig / init / CB handshakes are paid 9× per unit instead of 45×; only the gamma multiply and the last RoPE add split into a Q and a K loop. Compute binary 24.4 → 21.6 KB | bit-identical to v1 (bs1 resident/non-resident, q_split 2, bs8); heads op 41.7 → 28.7 µs in-model; e2e 16.474 → 16.135 ms (3 alternating 30-it A/B pairs, 8× p150b host, chip 0); cold/sustained 16.5/16.6 → 16.0/16.2 (NEGATIVE_RESULTS §51) | bs1 −2.1% |
| bs1 SDPA q192 on 88 cores (09-25) | packed SDPA calls take q_chunk 192 on an 11×8 grid (`QWEN_SDPA_GQA_PACK_Q_CHUNK=192`, `QWEN_SDPA_GQA_PACK_GRID=11,8`): each KV head's 64 packed row tiles split into 11 chunks, one per core of one grid row, so the K/V row multicast stays and each core computes 6 row tiles instead of 8. A chunk can now run into the next query head of its group: the concat-output writer wraps those rows (`head_wrap_tile_offset`), and the op no longer requires q_chunk to divide Sq. Unpacked calls keep q256 on 8×8 | SDPA 35.9 → 29.3 µs/call in-model (device replay 16.97 → 15.65 ms vs the 09-24 profile); e2e 16.15 → 15.88 ms (3 alternating 30-it A/B pairs, 8× p150b host, chip 0); cold/sustained 16.0/16.1 → 15.75/15.9; fixed-512 STS-B 0.8140, bucketed 0.8161; concat output bit-identical to `nlp_concat_heads` at q192/q160 (NEGATIVE_RESULTS §52) | bs1 −1.7% |
| bs1 norm output stays block-sharded (09-25) | both prefill RMSNorms keep their 10×8 block-shard output (`DistributedNorm.keep_sharded_out`) and QKV / FF1 / FF3 read it as a block-sharded in0 (`QWEN_BS1_NORM_SHARDED_OUT=1`): the 2D multicast factory takes its in0 senders from the shard grid's width, so 10 shard columns feed the 12-column matmul once in0_block_w divides the 8-tile shard (8 instead of the model's 10) and fuse_batch is set. The 72 ShardedToInterleaved ops per forward go away | S2I 72 → 0 calls; QKV / FF1 / FF3 kernel time unchanged (42.3 / 69.7 µs); device replay 15.65 → 15.33 ms; e2e 15.91 → 15.74 ms (3 alternating 30-it A/B pairs, chip 0); cold/sustained 15.75/15.9 → 15.6/15.75; STS-B fixed-512 0.8111 (0.8140 before; in0_block_w 8 changes the accumulation order), bucketed 0.8169 (NEGATIVE_RESULTS §51 correction) | bs1 −1.1% |
| batched add+norm operands in L1 (09-25) | at bs8/16/32 (ISL 512) the fused add+RMSNorm's short-lived tensors live in L1 interleaved instead of DRAM: its b operand (WO / FF2 outputs, `TT_PREFILL_WO_L1=1 TT_PREFILL_FF2_L1=1`), both norms' outputs (`QWEN_FUSED_ADD_NORM_OUT_L1=1`) and at bs8/16 the post-attention residual sum (`QWEN_FUSED_ADD_NORM_SUM1_L1=1`); the decoder's move of the attention output to the residual's DRAM config is skipped in the fused path (it had become an L1 → DRAM copy per layer). Model-level only; the op gains an `out_memory_config`. Opt out: `QWEN_BATCHED_L1_INTERMEDIATES=0` | fused op 434.9 → 255.0 µs/call at bs32; sustained_run.sh, 3 alternating rounds per chip (8× p150b host, chips 0-2 concurrently), cold / sustained: bs8 108.2 / 121.0 → 102.7 / 117.0, bs16 202.8 / 239.6 → 194.6 / 232.7, bs32 396.4 / 458.5 → 382.8 / 447.5 ms; batched STS-B 0.8123 / 0.8140 / 0.8159 unchanged; bs1 unaffected (NEGATIVE_RESULTS §54) | bs8 −5.1 / −3.3%, bs16 −4.0 / −2.9%, bs32 −3.4 / −2.4% (cold / sustained) |

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

## Qwen3-Embedding-4B through the same stack (2026-09-24)

Same backbone (2560 / 9728 / 36 layers / 32 Q + 8 KV heads / d 128), so the stack runs it unchanged with
`HF_MODEL=Qwen/Qwen3-Embedding-4B`; only the recipe switches follow the checkpoint (causal attention via
`QWEN_SDPA_CAUSAL=1`, set automatically for non-pplx checkpoints; last-token pooling with EOS). Nothing in
the pplx path changed (bs1 re-measured at 17.7 ms with the switch in place). Run instructions:
`PERF_GUIDE.md` §9.

| batch | H200 | cold best (best of 10) | × H200 | sustained (median of it 15–29) | × H200 | 3× target | previous Qwen demo (`../qwen3_embedding_4b`) |
|---|---|---|---|---|---|---|---|
| 1 | 5.437 ms | **18.1** | 3.33× | 18.3 | 3.37× | 16.3 | 32.3 |
| 8 | 33.081 | **115.7** | 3.50× | 121.9 | 3.68× | 99.2 | — |
| 16 | 67.225 | **217.6** | 3.24× | 228.3 | 3.40× | 201.7 | — |
| 32 | 139.150 | **426.9** | 3.07× | 445.5 | 3.20× | 417.5 | 725 |

Within ±1–2% of pplx-embed-4B at every batch (run-to-run spread), as expected for identical compute; AICLK
settled at 1328 / 1206 / 1190 / 1116 MHz. STS-B Spearman (last token + EOS,
`eval_accuracy_batched.py --pool last --eos`, fixed ISL 512): bs1 0.8190, bs8 0.8095, bs16 0.8076,
bs32 0.8073. The SDPA sweeps under causal attention pick the same grids and chunks as the bidirectional
ones. One bs8 run died with a segfault while loading the tensor cache (host side, before any iteration);
the rerun was clean — treat such a crash as transient and rerun before debugging.

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
