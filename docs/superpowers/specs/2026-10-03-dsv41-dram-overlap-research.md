# DSV4.1-Flash decode: overlapping weight streaming with latency-bound phases (research, no device runs)

Tags: **[V]** verified (read in source / log / measured data), **[I]** inferred or from model knowledge, **[W]** web (search summary, not deeply verified).

## 0. Bottom line

* Prefetch can only hide the DRAM time of the *weight matmuls*. It does not shrink the latency-bound non-weight ops (SDPA, norms, rope, topk, mHC post, collectives, ~700 us/layer). Gap to roofline is 58 - 26.9 = 31 ms/token; prefetch ideal ceiling is ~6 ms/token, realistic 2-4 ms. The rest of the gap needs fusion / persistent kernels / MoE straggler work.
* A working Blackhole mechanism exists in-tree (DRISC "Tensor prefetcher" + GCB + ring matmul, trace-capturable), but it has never been used on BH Galaxy in the Llama path and carries heavy L1 and layout constraints.
* Micro-batching (c) is a net loss. Cross-layer pipelining (d) has no concurrency in one command queue without sub-devices.
* Recommended order: (1) DRAM-sharded program configs for attention matmuls (+1.2-1.6 ms, low risk), (2) shared-expert overlap with moe_compute (already ranked, ~4 ms), (3) prototype Tensor-prefetcher feeding the shared expert (2-3 ms more).

## 1. Where the time goes (measured, layer 2 op table [V], docs/superpowers/specs/2026-10-02-dsv41-layer-op-table.md)

Layer traced 1502 us (58 ms/40 = 1450). Weight-streaming matmul kernels in the trace, with achieved BW (bytes from roofline_model.py shapes, bfp8 1.0625 B/elem) [V time, I arithmetic]:

| matmul | bytes/chip | kernel us | GB/s |
|---|---|---|---|
| attn wqkv 5120x1792 | 9.7 MB | 34.2 | ~285 |
| attn wq_b 1280x4096 | 5.6 MB | 21.0 | ~265 |
| attn wo_a 4608x1024 | 5.0 MB | 23.6 | ~213 |
| attn wo_b 1024x5120 | 5.6 MB | 21.6 | ~258 |
| shared gate+up 5120x9216 | 25 MB | 65 | ~377 (already DRAM-sharded config, shared_expert_v2) |
| shared down 2304x5120 | 12.5 MB | 38 | ~311 |
| router matmul | 3.9 MB | 24 | ~160 |
| mHC proj x2 | ~2x2 MB | 2x21.7 | ~100-190 |

Sum of weight matmuls ~270 us/layer (+35 us compressor on L2) vs ~165 us at 450 GB/s. The pair-swap rope matmuls (17.5 us x2) read only 0.5 MB: latency, not weights. Everything else (SDPA 38, topk 63 on 1 core, mHC post 46 on 1 core x2, norms, slices, CCL ~200 us in moe) is non-weight latency.

Upper bound of prefetch: weight matmuls from ~270 us to ~60-80 us (L1-resident, M=4 tiles) = ~190-210 us/layer = 7.6-8.4 ms/token. L1-capacity and ordering limits make 50% of that realistic: **~3-4 ms/token**.

## 2. (a) Prefetcher machinery in tt-metal

**Two generations [V, tt_metal/impl/buffers/prefetcher_matmul_design.md]:**
1. Worker-core `ttnn.dram_prefetcher` (reader+writer kernels on worker next to DRAM bank, triple-buffered local CB, global CB to ring-matmul receivers). Used by Llama-70B galaxy on Wormhole.
2. DRISC "Tensor prefetcher" (`ttnn.experimental.start_tensor_prefetcher / queue_tensor_prefetcher_request / stop_...`, ttnn/cpp/ttnn/operations/experimental/tensor_prefetcher): long-running kernel on the programmable DRAM cores; Blackhole only, firmware >= 19.12.0.0, off when streaming profiler is on [V, tensor_prefetcher.hpp, prefetcher_support.py]. No worker cores or worker L1 for the producer (DRISC L1 only ~70-80 KB staging). Queue API is non-blocking, takes a flat list of (tensor, block_count[, rotation]) streamed in order, can address several GCBs, and `capture_into_trace=True` re-sends the request on every `execute_trace` [V]. Newer PrefetcherPipe delivery (#55285) and MPFE (GDDR arbitration) weights are tunable [V].

**Hard constraints [V]:**
* Weights must be DRAM, TILE, width-sharded across all DRAM banks, receiver-contiguous layout, K/N tile aligned. Consumer must be the gather-in0 ring matmul (`bmm_large_block_zm_fused_bias_activation_gathered` + `reader_bmm_tile_layout_in1_ring_all_gather`); the standard ttnn.linear and moe_compute cannot consume a GCB. Ring size = banks x receivers per bank (BH 8 x 3 = 24 in Llama).
* The GCB is a FIFO in receiver L1: stream order must equal consumption order, and producer blocks when full. Run-ahead depth = GCB size, not DRAM time. Llama BH: 656 tiles x 1088 B = 0.71 MB/receiver (trimmed from 728 because it clashed with matmul CBs and persistent buffers) [V, llama3_70b_galaxy/tt/prefetcher_common.py]. 24 receivers = ~17 MB; 64 receivers = ~45 MB. Theory 1.5 MB x ~130 cores = ~195 MB, practically far less (dispatch, CCL persistent buffers, matmul/other CBs, moe_compute rings, trace-resident programs all need L1; Qwen PR #52723 hit "persistent L1 clash" with only 656 tiles [W]).
* Our L1 headroom is unmeasured here; tests/test_l1_probe.py exists to measure largest contiguous free L1 [V it exists, I result unknown].

**What was gained elsewhere:**
* Llama-70B WH Galaxy: the prefetcher was a core part of the design but I found no isolated A/B in repo [V absent]. On **Blackhole Galaxy the Llama path currently disables it**: `self.use_prefetcher = not self.is_blackhole` (model_config.py:608) [V].
* Qwen3-32B BH Galaxy PR #52723: 44.8 -> 56.9 tok/s/user total; prefetcher bring-up gave 44.8 -> 47.6 (~+6%), fused CCL gave the rest [W]. These are dense models where weight matmuls are the whole layer; expect no better for us.
* Blackhole prefetcher benches exist (tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py, _bench.py) and print GB/s; no numbers were available offline. Web: ring matmuls paced by prefetcher stream at ~134 GB/s per ... vs >500 GB/s DRAM (unclear scope, unverified) [W]. If true, streaming rate (not DRAM) would limit it, which matters since our target is hiding behind 100s of us.
* deepseek_v3_b1 [V]: no GCB prefetcher. It overlaps in-kernel: moe_kernel.cpp / decoder_block_kernel.cpp reorder blocks "so NCRISC on streamer cores can prefetch DRAM weights into cb_in1 while TRISC is still busy" with the SRAM chain; dram_streaming_matmul, dram_streaming_experts_matmul, persistent_loop micro-ops. That is the megakernel route: overlap happens inside one persistent program with RISC-level concurrency, tied to the 7168 / 130-core layout.

**Applying to our weights (per chip, bfp8) [I]:**
| set | MB | fits in GCB? | idle window to hide it |
|---|---|---|---|
| shared expert | 37.6 | needs ~52 receivers at 0.71 MB | attention + mHC_ffn + router (~600 us, DRAM idle) |
| attention (next layer) | 27.5 | ~39 receivers | shared expert tail + mHC_attn + glue |
| router + mHC fn | ~8 | yes | any |
Both cannot be resident together (65 MB), so schedule shared expert fill during attention/router and attention(L+1) fill during shared/mHC. moe_compute DRAM-saturates ~440 us, so nothing can stream then; and the shared expert must sit in L1 across the MoE, on cores moe_compute must not use.

**Expected gain [I]:** shared expert 103 -> ~30 us (-70), attention weight matmuls 100 -> ~45 (-55), router/mHC -15: ~140 us/layer = **~5.6 ms ideal, 2-4 ms realistic** after ring-matmul inefficiency at M=4, extra ops (in0 resharding to ring, ring output gather/untilize) and the +gaps seen in the sub-device spike.

**Routed experts (partial prefetch):** routing is data-dependent [I].
* moe_compute dm0 reads weights only after `metadata_ready_sem.wait_min(1)` (dm0.cpp:217) [V]; the weight stream cannot start before dispatch/tilize metadata. Idea: let dm0 start the first K blocks of weights for the most likely expert (or the first local expert in index order) before metadata, discard if idle. 3.8 of 12 local experts active on the busiest device [V roofline], so naive guess hit rate ~30% [I]. Gain bounded by dispatch + tilize latency (~50-100 us?, not measured), likely <=30 us/layer.
* Literature: cross-layer-gate prediction (Fate), Pre-gated MoE, ProMoE, SP-MoE etc. get 70-90% top-k accuracy at layer-ahead on offloading setups, but only 44-55% real cache hit rates [W]; those hide PCIe/host latency, not our case where the miss is just a wasted read of ~20 MB (44 us at 450 GB/s) competing with the real stream. Expected net <=0 at batch 16 [I]. Not recommended.
* GPU stacks [W/I, from docs/superpowers/specs/2026-10-03-dsv41-gpu-fusion-research.md]: DeepEP hook-based comm/compute overlap, Mega-MoE (dispatch + GEMM1 + SwiGLU + GEMM2 + combine in ONE kernel, wave scheduling; 1.50-1.73x), aux-stream shared expert overlap; on NVIDIA also PDL / programmatic dependent launch lets the next kernel prefetch weights into L2 before the previous finishes. The common ingredient: **concurrency inside a kernel or on a second stream**, same as our need.

## 3. (b) Sub-devices / GCB inside a captured trace on Blackhole

Verified in logs (/mnt/tt-data/ssinghal/dsv4-logs/subdev_agent_run2..5.log, tests/test_subdevice_overlap.py) [V]:
* Segmented traces with manager load/clear between segments (router + shared expert overlap, layer-2 weights, one rep): single 336.6 us, seg3 (no sd) 348 us, sd 328.7-344.8 us depending on order. Each trace boundary costs ~5.5-7.5 us. Best sd order (sr / S1R) saved ~8 us vs single but only ~19 us vs equal-segmentation; overall consistent with the earlier "+9 us/layer slower" in the full flow, since topk/sum/to_layout must be run after clearing the manager.
* topk, sum, to_layout (unconfined, under loaded 2-sub-device manager) fail: `TT_FATAL sub_device_ids.size() == 1: Programs must be executed on a single sub-device` (subdev_agent_probe46b.log) [V]. Any op that spans both sub-devices cannot run while loaded; this includes ops needed for the router and likely the moe/CCL ops.
* "whole" mode (manager loaded once before capture and kept; run3/run4) did work in-trace: 161.8 us vs 200 us for the same ops un-overlapped (no router stage 2) = -38 us; run2 failed (`!is_capturing_trace`, `get_bypass_mode`) so load order matters [V]. This shows genuine concurrency is possible within one trace when every op in the trace is sub-device-confined, but the full layer has unconfinable ops.
* Tensor prefetcher: has explicit trace-capture tests (test_prefetcher_BH_tensor_large.py "Trace capture / replay") and a `capture_into_trace` flag; requests bypass the command queue (socket + host worker thread; re-sent per execute_trace) [V]. It needs no worker sub-device for the producer (DRISC), so the main sub-device blocker (confining every op) might not apply, but the receiver cores' L1 and the GCB-fed matmul programs do coexist with every other op in the trace, hence the "static CB clash with L1 buffers" failure mode seen in Llama BH [V]. A request replays reading weights at replay time, not capture time [V].
* Hazard to check [I]: the prefetch request is not ordered against device work other than through the GCB (and `wait_for_cq_on_tensor_prefetcher`); it free-runs ahead until the GCB is full, so it will compete with moe_compute's DRAM stream unless the GCB is full at that time (stream order guarantees it blocks, but only if the fill is complete before the MoE; with 37.6 MB sat in L1 during MoE this is the design).

## 4. (c) Alternatives without prefetch

**DRAM-sharded matmul (single program):** measured attention linears reach 213-285 GB/s (47-63% of 450) [V time, I arithmetic]; shared gate/up is at 377 GB/s with the existing DRAM-sharded config. Only shared_expert_v2 uses `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` [V grep]. Moving wqkv/wq_b/wo_a/wo_b/router to DRAM-sharded layouts could approach ~380-400 GB/s: attention weight matmuls 100 -> ~70 us, router 24 -> ~12, shared down 38 -> ~32: **~40-50 us/layer = 1.6-2 ms/token** [I]. tests/test_attn_probe_matmul.py / test_attention_ab.py probed program configs for attention matmuls; I did not read their results (check before redoing) [unverified]. Constraint: the sharded weight must feed from replicated-in0 L1 and outputs resharded; each adds an op (~2-4 us) [V op table].

**Micro-batching 2x8 [I, arithmetic]:** NEGATIVE.
* Latency-bound ops cost the same at M=2 users as at M=4 (op table: 1-3 us launch-bound ops, SDPA 38 us, topk 63 us): two halves = 2x the latency-bound time unless executed concurrently.
* One in-order command queue gives no overlap between half A's small ops and half B's weight matmul; concurrency would need the sub-device or second-CQ machinery which (b) shows is not usable for the whole layer.
* Re-read: attention 27.5 + shared 37.6 + router/mHC 8 = ~73 MB read twice (+73 MB = +163 us DRAM per layer); expert bytes rise from 3.8 busiest-device experts to ~5 (union of two half-batches at 2.5 each; uniform-random model) = +24 MB (+55 us). Total +220 us DRAM/layer plus ~+300 us of duplicated latency ops: ~+10 ms/token worse, minus an overlap gain that cannot be realized. Reject. (Only fused weight-stationary variants, e.g. one matmul kernel reading weights once for both halves, avoid the re-read, but that equals batch 16 already.)

## 5. (d) Cross-layer software pipelining at trace level

* The trace is a serial command stream [V]; ops from layer L+1 cannot be hoisted before layer L's last op because of the residual dependency, except weight-only work. Legal reorderings are tiny: mHC mixing coefficients (post/comb, Sinkhorn) are only needed at expand [I, LMSYS blog via gpu-fusion-research.md], so the 46 us single-core mHC post x2 per layer could run concurrently with attention/FFN if it had a concurrent engine, saving up to ~90 us/layer (3.6 ms/token) [I]. Needs the same sub-device / persistent-kernel concurrency and runs into "unconfinable ops" for the other branch.
* Practical form: fuse (expand L) + (proj L+1) + post as one kernel (rank 6 in op table, 55 us/layer = 2.2 ms/token, [V estimate in table]); that is fusion, not pipelining.
* A persistent-kernel (deepseek_v3_b1-style) layer is the structural answer: overlap weight loads and small-op latency within one program via separate RISCs [V mechanism in b1 kernels; I applicability]. Cost: large (new kernels at our 5120/4x8 shapes), but it also removes the ~60 us/layer of op gaps and per-op launch costs.

## 6. Feasibility / gain / risk summary (per token, 40 layers)

| option | feasibility | expected gain | risk |
|---|---|---|---|
| A. DRAM-sharded attention/router/shared-down matmuls | high, in-tree | 1.6-2 ms | low; adds reshard ops; verify with existing probe logs |
| B. Shared expert concurrent with moe_compute (sub-device whole-trace or in-moe-expert) | medium-low (topk/CCL unconfinable; seg boundary 6 us) | up to ~4 ms ideal, spike showed ~0 or negative in segmented form, -38 us in whole mode | high |
| C. Tensor prefetcher -> shared expert (then attention) through GCB + ring matmul | medium (new BH-only path; L1 headroom unknown; fixed layouts) | 2-4 ms (ceiling ~5.6) | high: L1 clash, DRISC firmware, moe_compute DRAM contention, replicate-weights layout for bfp8 |
| D. Expert speculative prefetch | low | <=0.5 ms, may regress | medium |
| E. 2x8 micro-batching | reject | negative (~ -10 ms) | n/a |
| F. Cross-layer fusion (expand+proj+post) | high | ~2 ms | medium |
| G. Persistent-kernel layer (b1 style) | long term | most of the remaining ~15+ ms | very high effort |

Combined A + C + F realistic: ~6-8 ms (58 -> ~50-52 ms). Reaching the 27 ms floor requires G plus fixing the MoE straggler (expert imbalance cost ~100-150 us/layer per the op table [V], ~4-6 ms/token), which is bigger than C.

## 7. Recommended first experiment (single device, on .45, test only)

Goal: de-risk C in <1 day without touching the model.
1. `ttnn.experimental.is_tensor_prefetcher_supported(device)` on the .45 BH Galaxy (firmware >= 19.12.0.0) and `test_prefetcher_BH_tensor_large.py` trace tests, to confirm the DRISC path works on this box.
2. Microbench in the style of tests/test_attn_probe_matmul.py: shared-expert gate/up (K=5120, N=4608 or 9216 fused, bfp8, width-sharded across 8 banks, receiver-contiguous), ring matmul fed from a GCB (e.g. 24 and 48 receivers, 656 tiles), producer queued with `capture_into_trace=True` *before* a ~300 us dummy chain (the real attention ops), then the consumer. Measure (t(3)-t(1))/2 in-trace consumer time (target <35 us vs 65+38 now) and the dummy chain's own time with the prefetcher running (must not slow; checks NoC/DRAM interference with the attention ops).
3. In parallel, run `get_memory_view` after loading a full layer (test_l1_probe.py) to learn the largest contiguous free L1 per receiver core; this decides 24 vs 48 vs 64 receivers.
Go/no-go: consumer <= 40 us and chain slowdown <= 5 us => implement C for shared expert; otherwise do A and F instead.

## 8. Open items
* Real BH prefetcher GB/s from the in-tree benches (not run here).
* Result of the existing attention matmul probes and whether DRAM-sharded configs were already tried there.
* Free L1 per core in a full-layer trace (CCL persistent buffers, moe_compute rings).
