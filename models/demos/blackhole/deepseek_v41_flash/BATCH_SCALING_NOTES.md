# DeepSeek-V4.1-Flash decode: is the slowdown with batch size expected?  (BH Galaxy 4x8, ISL ~100, Engram on, traced)

Host .44, overlay /mnt/tt-data/ssinghal/wt/h46s, raw data /mnt/tt-data/ssinghal/dsv4-logs/bs_{A16,A32,A64,A128,B32,B64,B128}_h44 and bs_smoke16c (= B16 profile).
Labels: **[M]** measured, **[I]** inferred/modelled. Prompts: GSM8K (128 distinct), greedy, 40 layers, DSV41_ENGRAM_RAM=1.

## Answer in one paragraph
Partly expected, mostly not. Device time per token grows 43.3 -> 80.8 ms (B=16 -> 128) [M]. The MoE weight-streaming effect (more distinct experts per device)
is real but SMALL with real routing: the busiest device sees 2.4 -> 3.6 distinct experts (mean over 40 layers) [M], i.e. ~+1.1 experts x ~83-120 us x 40 layers = ~4-6 ms/token
(~10-15 % of the growth) [I]. The dominant growth is **mHC (+390 us/layer = ~15.6 ms/token, ~40 %)**, which scales ~linearly with tokens per mesh row although the work per token is tiny:
that is unexpected. Attention (+44 us/layer), dispatch/combine (+15-20 us/layer) grow modestly (expected-ish). Shared expert, router and CCL are flat (expected).
CORRECTION from the 40-layer A runs (layer traced without profiler, real routing, [M]): layer 5 grows 1.044 -> 2.114 ms and layer 0 1.201 -> 2.023 ms (B=16 -> 128),
i.e. +0.82..1.07 ms/layer x 40 = 33-43 ms ~= the whole 37.5 ms device growth. So essentially ALL growth is inside the decoder layers and the head/embedding/Engram-device part is small (not separately measured).
The profiled per-op sum (7-layer model, profiler on) explains only +735 us (L5) of the +1070 us: ~+335 us/layer at B=128 is NOT attributed to a group (hypothesis [I]: real-routing
expert imbalance/slowest-device wait in moe_compute+reduce_scatter, which the garbage-token B runs do not reproduce; or profiler-on timing). Host Engram prep adds +3 ms [M].

## Per-batch table
Step wall (decode_forward, traced replay + token readback, mean of 21 steady steps) and its parts [M]:

| batch (U/row) | ms/token wall | device wait (decode_device_read) | host prep (Engram rows etc.) |
|---|---|---|---|
| 16 (4)   | 44.6 | 43.3 | 1.5 |
| 32 (8)   | 53.2 | 52.7 | 2.4 |
| 64 (16)  | 63.9 | 62.8 | 3.3 |
| 128 (32) | 83.4 | 80.8 | 4.5 |

(Within-process closed loop; the earlier GSM8K demo numbers 54.6/68.7 include longer prompts/ragged users and the 16k-ISL/1.6GB-trace config.)

Real routing from the 4 eager decode steps after prefill (all 40 layers, mean over layers and steps) [M]:

| batch | distinct experts / layer (of 384) | distinct per device: mean | **max device** (mean over layers) | (token,expert) pairs on busiest device | mean pairs/device |
|---|---|---|---|---|---|
| 16  | 20.0 | 0.62 | 2.42 | 19  | 3 |
| 32  | 24.4 | 0.76 | 2.73 | 39  | 6 |
| 64  | 31.6 | 0.99 | 3.25 | 77  | 12 |
| 128 | 37.6 | 1.17 | 3.56 | 154 | 24 |

NOTE: the hypothesis numbers (3.8 -> 12 active experts/device) are NOT what real GSM8K routing gives: 128 users generating real text repeat tokens, and only ~38 of 384 experts
are hit per layer. Per layer it ranges ~13..60 distinct (late layers 23-36 are the most spread, devmax up to 5). Pair imbalance is large: busiest device handles 6.4x the mean pairs at B=128.
Caveat: only the first 4 decode steps (positions right after the prompt); steady-state diversity may be higher. The B profile runs (layers 0-6 only) generate garbage tokens
(7 layers cannot produce text) and therefore route to 2-5x more experts (distinct/layer 105 at B=128, devmax 6.4): their moe_compute times OVERSTATE realistic cost, mHC/attention/shared/router/CCL are token-independent and valid.

Per-layer kernel time, traced, us. Layer 5 = MoE layer with ratio-2 compressed attention, layer 0 = first (ratio-0, hash-routed) layer. "med" = median over 32 devices, "max" = slowest device.
Profiled with 7-layer model (real activations through layers 0-6), device profiler, trace replay sessions [M]. L5:

| group (us) | B16 | B32 | B64 | B128 | growth 16->128 |
|---|---|---|---|---|---|
| layer traced total | 1059 | 1253 | 1383 | 1794 | +735 |
| mHC (8-10 programs) | 113 | 204 | 268 | 504 | **+391 (unexpected)** |
| attention (23 ops) | 247 | 247 | 267 | 291 | +44 (expected-ish) |
| router (4) | 41 | 42 | 42 | 42 | 0 (expected) |
| dispatch | 23 | 19 | 34 | 36 | +13 |
| moe_compute med / max | 237 / 372 | 359 / 448 | 443 / 464 | 500 / 634 | +263 / +262 (routing artefact of B runs, see below) |
| combine (tilize+fast-reduce+reduce_scatter) | 173 | 135 | 102 | 193 | +20 (noisy: reduce_scatter kernel time includes waiting for the slowest device) |
| shared expert | 111 | 111 | 112 | 112 | 0 (expected, weight bound) |
| CCL attention (reduce_scatter+all_gather) | 47 | 46 | 46 | 44 | 0 |
| CCL moe all_gather | 26 | 25 | 25 | 25 | 0 |

Layer 0: total 1296 / 1568 / 1690 / 2146 us; moe_compute med/max 358/616, 442/785, 577/788, 750/993; mHC 113/204/266/500; attention 233/239/253/276; everything else flat.
Gap between programs in the trace is only ~30-35 us per layer (op-to-op), so the trace is kernel-bound, not launch-bound.
mHC programs at L5 (us, med): expand2 32/63/101/198, collapse_norm 27/-/67/122, mixes_proj2 26/33/54/115, mixes_post2 27/33/46/69 (B32 takes a different, unfused collapse path = 75 us for collapse+norm_apply).
Attention growth is the paged-attend permutes (6->23, 9->23 us), nlp_concat_heads (3->14); sparse_sdpa itself is flat 34-39 us.

## Expected scaling from weight bytes
Per expert: 3 x 5120 x 2048 x ~1.06 B/param (bfp8) ~ 33-38 MB (user figure 37.6 MB) / 453 GB/s = ~83 us per ACTIVE expert on the busiest device (tile cost independent of tokens at these batch sizes) [I].
- Expected moe_compute growth 16 -> 128 with REAL routing: (3.56 - 2.42) x 83 = +95 us/layer = **+3.8 ms/token** (+5.6 ms/token at the measured ~123 us/expert).
- Measured slope from the B profile runs, moe_compute slowest device vs busiest-device distinct experts of the same layer: layer 5: 123 us/expert, layer 0: 99 us/expert (4 points each, one profiled step, noisy; intercepts unreliable) [M/I]
  -> ~1.2-1.5x the pure weight-stream bound: some extra per-token / imbalance cost, but the order of magnitude is as expected. Median-device moe_compute rises ~140 us per mean expert.
- The slowest device is 1.3-1.6x the median at every batch (e.g. L5 B16 372 vs 237): imbalance costs ~100-150 us/layer at all batches (it shows up in combine/reduce_scatter wait too); it is not a batch-scaling effect but is the main reason B=16 is not faster.
- Total layer-growth budget: measured 37.5 ms device growth (43.3 -> 80.8). Profiled groups with realistic MoE: (391 mHC + 44 attn + 13 disp + 20 combine + ~95..125 moe_compute) = ~565-595 us x 40 = ~23 ms [I];
  the 40-layer A-run layer traces (+0.82..1.07 ms/layer) cover the full 37.5 ms, leaving ~13 ms/token unattributed inside the layers (see correction above).
- Host: decode_host_prep 1.5 -> 4.5 ms (+3.0 ms) = Engram hash+row gather 24 rows x 2 layers per user, scales linearly with batch [M]; fully serial with the replay in this driver.

## Classification
| group | growth 16 -> 128 | verdict |
|---|---|---|
| moe_compute | +95..125 us/layer real routing (+3.8..5.6 ms/token) | EXPECTED (distinct experts x weight stream; ~1.2-1.5x of the bound) |
| mHC | +391 us/layer = **+15.6 ms/token** | **UNEXPECTED #1**: 4.5x for 8x tokens; data per token is only 4x5120 fp32; each program 3-6x slower; unfused/different paths at U=8 |
| unattributed layer growth (A-run layer trace minus profiled groups) | ~+335 us/layer at B=128 (~13 ms/token) [I] | **UNEXPECTED #2 (not decomposed; suspect real-routing imbalance in moe_compute/reduce_scatter)** |
| host Engram prep | +3.0 ms | expected per-row cost, but serial: avoidable (overlap) |
| attention | +44 us/layer = +1.8 ms/token | expected-ish (per-user permutes/concat_heads scale with users; SDPA flat) - small |
| dispatch + combine | +33 us/layer = +1.3 ms/token | expected (per-row), combine noisy |
| shared expert, router, CCL | 0 | expected (flat; weight/latency bound) |
| moe imbalance (slowest device 1.3-1.6x median) | constant, 100-150 us/layer | unexpected but not batch dependent (cost at every batch) |

## Ranked fixes (gain at B=128 unless stated) [I]
1. **mHC kernels parallel over tokens** (expand2, collapse_norm, mixes_proj2/post2: use all 32+ cores per token block instead of per-row loops; keep one code path for U=8/16/32 - U=8 currently falls to unfused collapse/norm_apply): target ~150 us/layer at B=128 vs 504 = **-14 ms/token** (B=64: -4 ms, B=32: -3 ms).
2. **Decompose the unattributed ~+335 us/layer** (profile a 40-layer model with real tokens, i.e. fix the 7-layer garbage-token problem, e.g. dump real hidden states / force captured routing ids into the profiled layer via forced_routing): suspect moe_compute slowest-device + reduce_scatter wait from imbalance; potential up to -10 ms/token at B=128. The head/embedding part looks small.
3. **Overlap host Engram prep with the previous replay** (it is serial now): -3 ms/token at B=128, -1.5 at B=16 (the user wants Engram on: keep it, just overlap/vectorise).
4. **Expert-load imbalance**: slowest device 1.3-1.6x median in moe_compute; user already dropped placement balancing, but a cheaper lever is the per-device max active expert count (e.g. replicate hot experts / token-dropping order) - gain ~100 us/layer = up to -4 ms/token at all batches; low confidence.
5. Attention paged_attend permutes + nlp_concat_heads (+30 us/layer): fuse into SDPA layout, -1.2 ms/token at B=128.
6. bfp4 experts (-42 us/expert x ~3.5 = ~-150 us/layer = -6 ms/token) - rejected earlier for accuracy; listed only for completeness.
Not worth it: shared expert, router, CCL (flat).

## Small batch B=4 (U=1) [I - no profile run available]
Routing trend: 24 (token,expert) slots over 384 experts -> ~9-11 distinct experts/layer, busiest device ~1.5-1.8 distinct (>=1), vs 2.4 at B=16.
Expected moe_compute saving vs B=16 ~ (2.42-1.6) x 83-123 us = 70-100 us/layer = ~3-4 ms/token. mHC/attention/shared/router/CCL are flat or smaller (mHC at T=1 padded to 4 rows costs the same as U=4: ~113 us/layer),
so B=4 ~ 40-41 ms/token (A-run layer trend: the layer cost at B=16 is 1.04-1.2 ms, the floor is lower) (~24-25 tok/s/user) vs 44.6 at B=16. The floor is ~35-38 ms (non-MoE ~ 0.75 ms/layer x 40 + head/Engram). The B=4 agent (aafe18fd474c87cbe) could not be profiled here.

## Method and caveats
- test_batch_scaling.py: eager decode with route capture (host copy of router indices per MoE layer; hook in tt/moe_block.py active only on eager steps), traced decode for ms/token,
  per-op table (OpRecorder + device profiler, trace replay sessions, programs mapped to callers) for layers 5 and 0 with the real per-layer inputs and paged step state. bs_analyze.py groups ops by caller.
- Profile runs use a 7-layer model (layers 0-6) because the device profiler distorts ms/token and writes ~18 GB; per-op numbers of token-independent groups are valid for the 40-layer model; layers 20-39 (ratio-1 attention) were not profiled.
- Layer total traced time in the full 40-layer A runs (layers 5 / 0, no profiler, real routing) [M]: B16 1.044/1.201, B32 1.324/1.312, B64 1.445/1.686, B128 2.114/2.023 ms (bs_A*/layer_*/raw.json traced_ms; A runs have no per-op data).
- 4 eager routing steps only; one profiled step per layer; fits over 4 points. mean-over-layers distinct counts include both hash-routed and learned-router layers.
- ISL ~100 only; no indexer; long-context pool reads not covered.

## Hang record (.42)
Smoke run on .42 hung at the first device op (main thread in FDMeshCommandQueue::wait_for_outstanding_reads via GlobalSemaphore::reset_semaphore_value; completion-queue reader spinning; same signature as the earlier .42/.43 notes).
hangwatch triage: /mnt/tt-data/ssinghal/dsv4-logs/triage/hang_42_1085121_0923.txt (+ .console); one `tt-smi -glx_reset` done by hangwatch (rc=0); health after reset not re-tested. Remaining runs were done on .44 (no hangs).

## Diff
changes.diff (rebased on main 9411b8398a4, `git apply --check` OK, 3 files added/changed, +321 lines, no deletions): tt/moe_block.py (route-capture hook, off by default),
tests/test_batch_scaling.py, tests/bs_analyze.py. Driver scripts (outside repo): /mnt/tt-data/ssinghal/wt/h46s/{launch.sh,drive.sh,run4x.sh}.
