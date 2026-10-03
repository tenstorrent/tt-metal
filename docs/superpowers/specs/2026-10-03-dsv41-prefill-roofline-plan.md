# DeepSeek-V4.1-Flash prefill: roofline, profile and plan (4x8 Blackhole Galaxy)

Author: h45p (prefill roofline agent), 2026-10-03. Package `models/demos/blackhole/deepseek_v41_flash`. Every number is tagged
MEASURED (run on a 4x8 BH Galaxy, logs in `/mnt/tt-data/ssinghal/dsv4-logs/h45p_*`), DOC (quoted from a repo document) or ESTIMATE (arithmetic or assumption).
No run above 4096 tokens/row (U=4 -> 16 x 1024 prompts) was made here; 64k+ and 1M figures are extrapolations.

## 0. Summary

* Prefill is ~80-100x from the compute roofline. The cause is NOT the chunk size: the per-layer device time per token is the same at 512 and
  4096 tokens per mesh row (12.7 vs 12.3 us/token/layer MEASURED) because every token-wise block (mHC, router, `moe_compute`, shared expert,
  expand) runs on a **32-token slice** and the kernels are launch/weight-sweep bound at that size. Bigger chunks only help once the 32-token granularity is removed.
* One layer at R = 4096 tokens/row (16k tokens/layer-run): 201.6 ms device-kernel time, of which MoE (gate+dispatch+`moe_compute`+tail+allgather) 52%,
  mHC (3 phases) 33%, attention 8%, shared expert 7%.
* The 8 mesh columns execute all token-wise work **redundantly** (tokens are replicated over the columns; only the experts and the attention heads are split). mHC + shared expert + expand
  are ~49% of the layer after the MoE fix, 8x redundant and 10x above their DRAM floor.
* Implemented (bit-identical, measured): grouping G x 32 tokens into one `moe_compute` call (T up to 256 tokens/device), and a big-M shared expert.
  MoE block 5.9 -> 2.3 us/token (2.6x); traced prefill of 16 x 1024 tokens on 2 layers 0.496 -> 0.379 s (1.31x), layer kernel time 201.6 -> 157.4 ms.
* 1M-token path: traced (not eager) + sequence-parallel over the 4 mesh rows for one long user + column-split of token-wise work + bigger-T mHC + two-level indexer top-k;
  ESTIMATE 2-4 minutes for 1M tokens (section 6), today ~10 min traced-at-best, ~5 h eager for a single user (coordinator measurement).

## 1. Inputs verified

| quantity | value | source |
|---|---|---|
| model | D 5120, 40 layers, 64 heads x 512, q_lora 1280, o_groups 8 x o_lora 1024, 384 routed experts top-6 + 1 shared, expert FF 2304, hc_mult 4, window 128, compress ratio 2 (layers 2-19) / 1 (20-39), 8 index layers, topk 512 | `/mnt/tt-data/ssinghal/deepseek-v41-flash/inference/config.json` |
| BH matmul engine | 8x16x16x16 MAC/cycle, 1.35 GHz = 5.4 TFLOP/s per engine at LoFi, 2.7 HiFi2, 1.35 HiFi4 | DOC `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:52-70` |
| BH cores | 130 compute cores (13x10); the device exposes 120 L1 banks (12x10) to ttnn | DOC BringUp guide; MEASURED (TT_FATAL "120 of them") |
| chip peak | LoFi 5.4 x 130 = 702, HiFi2 351, HiFi4 175 TFLOP/s (theoretical); best measured GEMM ~580 TFLOP/s (BFP4 LoFi) | DOC `GEMM_FLOPS.md:86` + arithmetic |
| DRAM | 8 banks x 4 GB; effective ~450 GB/s per chip (MoE streams an expert in 83 us) | DOC `roofline_model.py`, `dram-overlap-research.md` |
| fabric | no BH-specific numbers in the repo; DSv3 tests assume 200 Gbps/direction/link on Galaxy | DOC `test_sparse_mla_ccl_perf.py:46-49` |

The existing decode roofline assumed 100 TFLOP/s/chip. With bfp8 weights at HiFi2 the verified peak is 351, at LoFi 702 (the moe_compute kernel runs LoFi, fp32 accumulate).
Below "floor" = 50% of the HiFi2 peak = 175 TFLOP/s/chip, 5.6 PFLOP/s for 32 chips (ASSUMED achievable efficiency; DSv3 prefill blocks on the same Galaxy are far below it).

## 2. Roofline per token (ISL 4k, whole mesh = 32 chips) - `tests/prefill_roofline_model.py`

FLOPs per token per layer (2 x MACs): routed top-6 + shared 0.495 GFLOP (6 x 3 x 5120 x 2304 x 2 + shared 2 x 5120 x 4608... = 0.495), attention projections 0.254
(wqkv 9.2M + wq_b 42M + wo_a 33.5M + wo_b 42M MACs), router 0.004, mHC 0.002, SDPA core (window 128 + <=512 selected, 64 heads, d 512, QK+PV) 0.08, compressors ~0.007,
indexer scoring (8 layers only) see section 5. Whole model at 4k context: **34.0 GFLOP/token** (MoE 58%, projections 30%, SDPA 9.5%, rest 3%). The coordinator's 39.6 GFLOP/token includes a longer context.

| component | GFLOP/token (40 L) | floor us/token (32 chips, 175 TF/chip) | DRAM bytes/chip/layer per weight pass | MEASURED us/token (40 L), G=1, R=4096 | gap |
|---|---|---|---|---|---|
| routed + shared MoE | 19.8 | 3.5 | 451 MB routed (bfp8) + 37 MB shared | 255 (MoE+allgather) + 34 (shared) = 289 | 80x |
| attention projections + SDPA | 13.4 | 2.4 | ~25 MB | 41 (attention phase) | 17x |
| mHC (mixes, collapse, expand x2) + residual | 0.08 | 0.01 (compute) / 0.36 (DRAM: ~8 fp32 passes of the 82 KB stream, 8x column redundancy) | 5 MB | 162 (expand_collapse 86 + mhc_attn 44 + expand_out 32) | 450x vs DRAM floor |
| Engram dev (2 layers) | 0.13 | 0.02 | ~120 MB x2 | not in the 2-layer profile; eager 4.1 s of 26 s layer time at S=512 x 4 (3x a normal layer) | - |
| head / embedding | last token only (verified in `forward_device`: only the 32-row chunks holding each user's last token go through the head) | ~0 | 1.3 GB (vocab 129k x 5120 bf16/8 cols) once per prefill | eager 2.7 s / 1.55 s are 64 separate launches; negligible in a trace | - |
| **total** | **34.0** | **6.1 us -> 164k tok/s** (HiFi2 peak: 3.0 us, 328k tok/s; LoFi 50%: 3.2 us) | ~520 MB/layer = 20.8 GB per full pass = 46 ms at 450 GB/s | **492 us/token (2.0k tok/s kernel-only; 1.65k traced end-to-end)** | **~80x** |

Floors for 16 x 4096 = 65,536 tokens: 2.2 PFLOP -> 0.40 s at 175 TF/chip (0.20 s at the HiFi2 peak, 0.08 s LoFi peak); the coordinator's 100/200 TF/chip numbers give 0.81/0.41 s. Weight DRAM floor: one pass of ~20.8 GB/chip
= 46 ms; a chunk of 1024 tokens/row in 4 chunks pays 4 passes = 0.19 s, which is below the compute floor, i.e. at >=1k tokens per call the DRAM sweep is not binding if every call is large.
**Today every 32-token slice is a separate pass over the routed weights**: 128 slices/layer x 0.75 ms.

MoE call arithmetic (ESTIMATE, `prefill_roofline_model.py`): the routed weights are swept once per call (1.0 ms/chip/layer at 450 GB/s). Compute at 50% of HiFi2 peak crosses the sweep time at ~16k tokens per call (global),
i.e. ~256 tokens per expert; the 32-row tile padding makes the kernel compute-inefficient below ~32 tokens/expert (2048 tokens/call). `moe_compute` today takes 4 rows x 32 = 128 tokens (2 tokens/expert): sweep-bound.

## 3. Measured profile of one prefill layer (layers 2-3, U=4, whole prompt in one chunk, eager + device profiler, mean over 32 devices)

Tool: `tests/test_prefill_scen_device.py` with `DSV41_PROF_EVERY=2 DSV41_PROF_MARKS=<dir>/marks.json` + `tests/prefill_prof_summary.py` (phases = device-operation-id ranges recorded by `prefill_layer._mark`).
The CSV has no op names in this build, so phases are the unit, not op codes. Kernel time only (op-to-op gaps excluded); the trace adds ~20-25%.

| phase (ms per layer) | R=512 tokens/row (16 x 128) | R=4096 (16 x 1024) | R=4096, G=8 + big shared | ops/layer/dev (R=4096) |
|---|---|---|---|---|
| MoE: gate + dispatch + moe_compute + tail + allgather | 12.97 (50%) | 104.3 (51.7%) | 62.8 (39.9%) | 1664 -> 848 |
| mHC expand + mixes(ffn) + collapse(ffn) | 4.49 | 35.4 | 35.0 | 645 |
| mHC mixes + collapse (attn) | 2.29 | 17.8 | 17.9 | 389 |
| attention (all R tokens, SDPA d=512, projections, state writes) | 2.81 | 16.8 | 16.9 | 148 |
| shared expert | 1.77 | 14.2 | 0.47 (+11.3 for the big matmul, booked under "slice") | 640 |
| expand out + to DRAM | 1.65 | 13.2 | 12.9 | 128 |
| **kernel total per layer** | **26.0** | **201.6** | **157.4** | |
| per token per layer | 12.7 us | 12.3 us | 9.6 us | |
| traced replay, 2 layers + embed + head (`DSV41_SCEN=...zt`) | G=1: 65 ms; G=8: 50 ms | G=1: 0.496 s; G=8: 0.379 s | | |

Conclusions:
1. us/token is flat from R=512 to R=4096: the system runs at the speed of 128 x 32-token slices per layer; chunk size is not the lever (attention grows only from 8% to 8%).
2. Slice cost: MoE 0.81 ms per 128 tokens (6.4 us/token), mHC phases 5.3 ops/slice at ~54 us avg although each moves only ~2.6 MB (6 us of DRAM): these ops are latency/launch bound.
3. The shared expert at M=32 re-reads 37 MB per slice (1.77 ms/layer at R=512 = 16 slices x 83 us x ... = DRAM sweep bound); a single M=4096 matmul costs 11.3 ms = 26 TFLOP/s/chip because every column computes all 4096 tokens (HiFi4, 8x redundant).
4. Extrapolated 40-layer traced prefill from the 2-layer trace: 40 x 0.248 = 9.9 s for 16384 tokens -> 1.65k tok/s, matching the reported ~1.5k. With G=8: 40 x 0.1895 = 7.6 s -> 2.16k tok/s (ESTIMATE from 2 layers; embedding/head are fixed costs and included in both).
5. Fixed host cost per prefill: logits readback 0.10 s (16 x 129280 fp32 gather) regardless of S.

## 4. The MoE 128-token cap, measured (`tests/test_moe_tokens_scaling.py`, one layer, random routing, 4 rows x T tokens per call, trace-timed)

| T tokens/device | tokens per call | ms per call | us/token | status |
|---|---|---|---|---|
| 32 (today) | 128 | 0.754 | 5.89 | |
| 64 | 256 | 1.15 | 4.5 | bit-identical to T=32 slices |
| 128 | 512 | 1.50 (L1 scores) / 1.66 (sharded) | 2.9-3.2 | bit-identical (PCC 1.000000, max |diff| 0) |
| 256 | 1024 | 2.33 | 2.27 | bit-identical; needs `compute.output_height_shard_dim=8` |
| 512 | 2048 | - | - | fails: moe_core_placement cannot place 64 combine + 4 tilize + 8 ring cores on the 12x10 grid |

Blockers removed (all in the package, no ttnn edits): (a) the stock scores shard config puts one token per core (T=128 needs 16x8 cores > 120): `big_batch_scores_config` uses 64 cores, T/64 tokens each (or L1 interleaved, accepted by dispatch, slightly faster);
(b) `moe_compute` capacity is 32 x output_height_shard_dim x data_parallel_cores tokens: raised via `with_height_shard` (4 -> 8 at T=256); (c) the router kernels (`router_select.py`, `moe_tail.py`) assert T <= 32: the router runs per 32-token slice and the results are concatenated.
The marginal cost is 1.7 us/token (T 128 -> 256), the fixed cost ~0.4 ms: the kernel is still ~5x above the compute floor (2.27 us vs ~0.5 us/token), it is no longer weight-sweep bound; remaining cost is tile padding (M=32 per expert, ~16 tokens/expert at T=256) and dispatch/combine. A T=512 kernel would need a different core placement (owner of `moe_compute`).

In the layer: `DSV41PrefillMoE` / `DSV41PrefillLayer.forward` group G consecutive 32-token chunks (`DSV41_MOE_G`, scenario test builds `DSV41PrefillMoE(T=32*G)`), run the router per slice, one `moe_compute`, one allgather, slice back.
Correctness MEASURED: logits/hidden PCC printed by the scenario test identical to 5 digits for G=1 and G=8 (layers 2-3, 16 x 128 tokens; the PCC against the CPU dump is 0.49 only because layers 0-1 were skipped), traced logits vs eager 1.00007, argmax 16/16.

## 5. Other limits (ESTIMATES unless marked)

* **Attention core / indexer.** Per query token: window 128 + min(N, 512) keys. SDPA core 3.2 GFLOP/token (9% at 4k context). Indexer scoring (32 heads x 128 over N = ctx/ratio keys; 8 index layers, 3 with ratio 2, 5 with ratio 1):
  avg per token 0.2 GFLOP at 4k ctx, 1.8 at 64k, 7.1 at 256k, 28 at 1M (45% of 61.8 GFLOP/token at 1M). Compute-only time at 175 TF x 32: 1M tokens x 28 GFLOP = 2.8e16 -> 5 s. The decode measurements (`kv-paged-capacity-design.md` 5.2, MEASURED for decode): scoring 0.9 ns/entry per 32-row tile (memory bound), top-k 4.4 ns/entry per 32-row call.
  Prefill top-k over every query: sum over queries of N = ctx^2/2 entries per layer: 1M, ratio 1: 5e11 entries x 4.4 ns / 32 rows = 69 s per layer on one chip = 2.2 s on 32 chips; 5 ratio-1 layers + 3 ratio-2 layers ~ 14 s. A two-level top-k (candidate blocks of 8, designed for layer 20) cuts it ~12x. At 64k: 0.35 s total. Not a bottleneck below 256k; at 1M it is ~15-20% of the floor.
* **MoE all-to-all.** DSv3 prefill on the same Galaxy: dispatch 0.55-0.87 ms + combine 0.75-1.43 ms per layer per 5120-token chunk (DOC `test_dispatch_combine_perf.py`) = 0.25-0.45 us/token/layer; x40 = 10-18 us/token = 55-100k tok/s cap. Our dispatch is a 4-device linear along the mesh rows with T tokens/device: the grouped call above already includes it (2.27 us/token total at T=256). Load imbalance: uniform random routing was measured here (T=256: 12 experts/device all active); real routing skew at 16k tokens/call is not measured (expect 1.2-1.5x on the busiest device, ESTIMATE).
* **Engram, host.** MEASURED by h44p: gather + dequant 0.023 ms/token + tilize 0.014 ms/token with RAM tables = 26k tok/s per thread, now overlapped by a 1-thread prefetch; 12 KB/token/layer rows, 25 MB per layer for 2048 tokens. At 1M tokens: 25 GB of rows per Engram layer, 50 GB total, uploaded replicated over the 8 columns (x8 on the wire = 400 GB, ~15-20 s at PCIe rates, ESTIMATE): upload each row once per mesh row and all-gather on the device, or use 4 gather threads. Engram on device: eager 2 s per layer vs 0.7 s for a normal layer (3x); the wkv matmul runs at T=32 and re-reads its weights per slice - same fix as the MoE (group slices).
* **Embedding / head.** Head runs only on the chunks holding the last token (verified). Embedding is one lookup per 32 tokens (64 launches at S=512 x 4), negligible in a trace; the eager 1.55 s / 2.7 s are launch + sync artefacts of `profile=True`.
* **Eager vs traced.** Coordinator measurements (eager, 4 users = 1 per row, chunk 1-2k): 8k 189 s, 16k 347 s, 32k 624 s (210 tok/s aggregate); traced (MEASURED here, 16 users) is ~8x faster (1.65k tok/s). All long-prompt runs must be traced: one trace per chunk shape with runtime position tensors (as GPT-OSS does for 32k, as h44p notes, whole-layer traces need care with eager temporaries).

## 6. How the references split the sequence, and what that means for us

| | layout | tokens/chip/call | chunk |
|---|---|---|---|
| `deepseek_v3_d_p` (V3/Kimi/GLM, MEASURED DOC) | 8x4: SP on 8 rows (zigzag / block-cyclic, `common/prefill/chunk_layout.py`), TP on 4 cols; EP over the SP axis (32 experts/chip), `dispatch`/`combine`/`unified_routed_expert_moe` | 640 | 5120 tokens (`PREFILL_CHUNK_SIZE`), 11 chunks = 56k default; traced: 0.40-0.72 s per 5120-token chunk for 61 layers (Kimi) = ~12.7k tok/s; MoE layer 13.7 ms / dense layer 5.4 ms per chunk |
| GPT-OSS | row-sharded batching: `users_per_row` users packed in the seq dim per mesh row (like ours); MoE `deepseek_prefill` dispatch/combine with `seq_len_per_chip=1024` chunks, capacity factor 2; long prompts one user at a time in chunks up to 128k | 1024 | |
| ours today | users across the 4 rows, 8 columns replicate the tokens (TP heads, EP experts), token-wise blocks on 32-token slices | R = U x C per row, 32 per slice | 128-1024 per user |

* Same-galaxy yardstick: V3 prefill at 5120 tokens/chunk reaches ~12.7k tok/s with ~74 GFLOP/token (37B active); at equal efficiency our 34 GFLOP/token model would reach ~25-30k tok/s (ESTIMATE, not measured), i.e. a realistic end target is ~10x above today's 2.2k tok/s.
* The V3 MoE path needs dense per-expert weights (second copy, ~9.6 GB/chip at bfp4 or 18 GB at bfp8) which does not fit next to the resident decode ring (~6 GB free, `device-prefill-design.md` 2.1). This is why we stay on `moe_compute` and enlarge its token count.
* **Single long prompt (the 1M case).** Today one user occupies one mesh row (8 of 32 chips); the other three rows idle for a single prompt. Sequence-parallel over the 4 rows is natural here because the attention state is tiny: split every chunk C into 4 row-slices of C/4; per layer the only cross-row traffic is (a) the 128-token window halo from the previous slice, (b) all-gather of the new compressed latents and index keys (C/ratio x 512 B + 256 B per entry; 16k chunk: ~8 MB), (c) the existing MoE dispatch along the rows. Everything else is unchanged (mHC, router, experts, projections). Contiguous slices are fine (attention cost per query is window + top-k; only indexer scoring grows with position, by <C/N within a chunk). This is a 4x for a single user and needs no new ops.
  Column split (token-wise work) would give another up to 8x on 49% of the layer, see ranked list.

## 7. Ranked fixes (gain = on the measured 12.3 us/token/layer, effort for one engineer)

| # | fix | expected gain | effort | status |
|---|---|---|---|---|
| 1 | Trace everything (chunk-shaped trace, runtime position/chunk tensors); never run eager for long prompts | ~8x vs eager (coordinator 32k: 210 tok/s vs 1.65k traced) | medium (owner h44p; whole-layer traces clobber eager temporaries) | not mine |
| 2 | Group G x 32 slices into one `moe_compute` (T up to 256) + router per slice | MoE 6.4 -> ~3.9 us/token (block 5.9 -> 2.27); layer kernel 201.6 -> 157.4 ms; traced 0.496 -> 0.379 s | done | **implemented, bit-identical** |
| 3 | Big-M shared expert (and Engram wkv) on grouped slices | removes the 37 MB per-slice weight re-read: 14.2 -> 11.3 ms at R=4096 (compute-bound: 8x column redundancy), 1.8 -> ~0.5 at small R | done (shared); Engram TBD | implemented |
| 4 | Column-split of token-wise work: each of the 8 columns runs mHC / shared / expand / norms on R/8 tokens; reduce-scatter instead of all-reduce after `wo_b`, all-gather the collapse output (attention input) and the MoE input (`T=32 x 8 cols = 256` tokens/device: exactly the grouped call), all-to-all of the MoE output back to the owning column | mHC + shared + expand = 49% of the layer -> /8 + ~3 gathers of 42 MB (~1-2 ms each, ESTIMATE): layer 157 -> ~80-90 ms (1.8x) | high (new CCL placement; per-device-different slices) | design only |
| 5 | mHC kernels for T > 32 (or a plain-ttnn big-M mHC on a compact `[R, 4D]` stream): 5.3 ops/slice at ~54 us for 2.6 MB of traffic | mHC 66 ms/layer -> DRAM floor ~6 ms at R=4096; combined with #4 ~10x | high (kernels hard-assert T <= 32 in mhc_collapse/expand/mixes) | not started |
| 6 | Row sequence-parallel for a single user (section 6) | 4x for a single prompt | medium | not started |
| 7 | Engram: 4 gather threads or per-row single upload + device all-gather; group slices for the wkv matmul | ~3x on the 2 Engram layers (14% of eager layer time); removes 400 GB of PCIe at 1M | low-medium | partly (h44p: prefetch) |
| 8 | Two-level indexer top-k + sparse_sdpa for > 512 compressed entries | at 1M: ~14 s -> ~2 s of top-k; needed for correctness beyond 1k tokens | medium (KV agent) | not mine |
| 9 | MoE T=512 (kernel core placement) / bigger expert tile M | MoE 2.27 -> ~1.5 us/token | high (kernel) | blocked, see section 4 |

Projection on the traced aggregate rate (ESTIMATES): 1.65k tok/s today; #2+#3: 2.2k (measured on 2 layers); +#5: ~3.5k; +#4: ~6-7k; +#9/attention tuning: ~8-10k tok/s.
With #6 a single user gets the aggregate: **1M tokens = 1M / 2.2k = 7.6 min now-with-#2/#3 (+6 s indexer top-k + ~15-20 s Engram upload), ~2.5-3 min after #4+#5, ~2 min with the rest; the compute floor is 12-25 s** (62 GFLOP/token at 1M incl. indexer, 175 TF x 32 chips: 11 s; HiFi2 peak: 5.5 s).
Compared with eager one-row-per-user (coordinator: ~5 h): 40-150x.

## 8. Files (overlay; `changes.diff` against the h44p prefill snapshot)

* `tt/prefill_layer.py`: grouped MoE (`DSV41_MOE_G`), router per 32-token slice, `shared_big` (`DSV41_SHARED_BIG`), profiling marks (`DSV41_PROF_EVERY`, `DSV41_PROF_MARKS`).
* `tt/moe_block.py`: `big_batch_scores_config`, `with_height_shard` (`DSV41_OHSD`, `DSV41_SCORES_CFG`), `gate_batch` (router in slices).
* `tests/test_moe_tokens_scaling.py` (T scaling + equality vs T=32), `tests/prefill_roofline_model.py` (CPU roofline), `tests/prefill_prof_summary.py` (phase table), `tests/test_prefill_scen_device.py` (`DSV41_MOE_G`).
* Usage: `DSV41_MOE_G=8 DSV41_SCEN=1024:0::zt DSV41_LAYERS=2-3 DSV41_U=4 pytest tests/test_prefill_scen_device.py` (G must divide the number of 32-token chunks per row; `output_height_shard_dim` follows automatically: 4 for T <= 128, 8 for T = 256).
