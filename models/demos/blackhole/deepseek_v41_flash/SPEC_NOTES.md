# DSpark speculative decoding, final increment (against main db0a20b75e0)

changes.diff: `git apply --check` passes on main db0a20b75e0. Touches 3 existing files + 1 new:
* tt/mhc.py: `DSV41MHC.mixes` pads the token dim to a multiple of 8 when T % 8 != 0 and T > 8 (the packed mixes-v2 kernel silently returns WRONG mixes for
  T = 12 and 20, e.g. spec verify with 4 users/row x n = 3 or 5; verify PCC of rows >= 1 dropped to 0.83-0.95; with the pad 0.9995+). T = 4 / 8 / 16 / 24 / 32 unchanged.
* tt/spec_decoder.py: profile hook, `keff` (padded verify blocks), top-2 logits per row (near-tie evidence), `stop_after` timing modes.
* tests/test_spec_loop.py: prompt fed through the device, ROUND_BREAKDOWN / PROFILE / near-tie gap logging, snapshot buffers allocated before the trace, finished users kept inside the cache.
* reference/ref_spec_exact.py: offline exactness report (divergence + top1-top2 gap).
Default decode is unchanged (nothing imports the spec code).

## Same-build k sweep (16 users = 4/mesh row, 40 new tokens/user, GSM8K prompts, spec4_*.log on .46; T = 4 x n rows per mesh row)
| k (drafts verified) | rows T | accepted/round (CPU ref A_k) | tok/round | verify ms | draft+commit ms | device round | host | wall | tok/s/user |
|---|---|---|---|---|---|---|---|---|---|
| plain (k=0) | 4 | - | 1 | 46.0 device total | - | 46.0 | 4.6 | 50.6 | 19.8 |
| 1 | 8 | 0.857 (0.90) | 1.86 | 54.7 | 10.3 | 65.0 | 5.6 | 70.6 | 26.3 |
| 2 | 12 | 1.547 (1.67) | 2.55 | 74.4 | 10.4 | 84.8 | 8.1 | 92.9 | 27.4 |
| 3 | 16 | 2.196 (2.30) | 3.20 | 64.0 | 10.5 | 74.6 | 8.8 | 83.4 | 38.3 |
| 4 | 20 | 2.303 (2.81) | 3.30 | 92.9 | 10.6 | 103.6 | 9.3 | 112.9 | 29.2 |
| 5 | 24 | 2.344 (3.22) | 3.34 | 82.6 | 10.6 | 93.4 | 5.7 | 99.1 | 33.7 |
| 2 padded to T=16 (k=3 block, accept clipped at 2) | 16 | 1.613 | 2.61 | 66.4 | 10.4 | 77.0 | 5.9 | 82.9 | 31.5 |
| 4 padded to T=32 (n=8, clipped at 4) | 32 | 2.553 | 3.55 | 86.6 | 10.7 | 97.6 | 6.5 | 104.0 | 34.2 |
| 5 padded to T=32 (n=8, clipped at 5) | 32 | 2.787 | 3.79 | 86.4 | 10.8 | 97.4 | 7.0 | 104.3 | 36.3 |
Spec numbers within +-10% run-to-run across hosts (committed run: k=3 on .47, 73.6 ms wall, 44.8 tok/s/user, verify 57.5; same code on .46 now 83.4 ms / 38.3).
The committed k=3 row and the table are the SAME code; the difference is host/run (the plain baseline also moved 45.3 ms on .45 -> 50.6 ms on .46). Use ratios: k=3 is ~1.9-2.0x plain.
Acceptance per position (k=5, free running 16 users x ~17 rounds) is within sampling noise of the CPU reference for k <= 3 (0.912/0.696/0.588 vs 0.905/0.796/0.693 conditional-free
leading-prefix probabilities 0.905,0.72,0.54...), lower for k=4,5 because users hit the position-127 cache limit (only 40 positions after the 80-token prompt; late rounds are cut short).
Teacher-forced (device hidden states, CPU stream tokens) drafter match d1..d5 = 0.90/0.75-0.79/0.56-0.65/0.46-0.55/0.31-0.50 vs CPU 0.905/0.796/0.693/0.598/0.489 (n ~ 108-126 per position).

### Why k=2/4/5 (T=12/20/24) are slow: confirmed by per-section profile (PROFILE lines, eager + device sync, ms summed over 40 layers)
T=16: attention 155, moe 152, mhc mixes+expand 98 + 57, collapse+norm 33+28 (total 614). T=20: attention 178, moe 154, mixes 131+93, collapse+norm 66+63 (total 778).
T=24: 184/157/112+75/66+63 (753). T=4 plain: moe 137, attention 117 (488). The mHC kernels have fast paths only for T = 4/16/32 (collapse+norm fused) and T % 8 (mixes v2): other T take the
composite fallback (+25..40 ms per round). Padding the block to the fast T (k=2 -> T=16, k=4/5 -> T=32) recovers 3-5 tok/s/user but wastes rows.
Adding native T=12/20/24 support to the mHC kernels is worth it only for k=2 (T=12) and k=4/5: expected gain = the mHC fallback delta (T=20 vs 16: ~+40 ms of mixes/collapse in the profile sum; 15-30 ms per round real),
which would put k=4 at ~75 ms (T=20 would then cost about T=16 + 25% rows) -> ~44 tok/s/user; k=3 stays the sweet spot. The remaining big levers are attention (155 ms-sum, per-j paged_update_cache launches + SDPA per virtual user),
moe (152, expert traffic), and mHC at T=16 (~220 ms-sum).

## Exactness vs plain greedy (near-tie evidence, 16 users = 8 prompts x 2, 41 tokens)
k=3 vs plain (same code, tests spec4_k3b vs spec4_k0): 8/16 identical; the other 4 distinct prompts diverge at generated token 18 / 3 / 7 / 36 with top1-top2 logit gaps (plain / spec) 0.18/0.016, 0.028/0.005,
0.14/0.31, 0.25/0.23 (logits are O(15-30)): all are near-ties (< 0.3). Plain decode itself is not reproducible across builds: two plain runs (before/after the T=4 mixes pad, different hosts) differ on 3/8 prompts (tokens 18, 20, 7).
So divergences are bf16/bfp8 batch-shape numerics at near-ties, not a state/position/mask bug (verify rows PCC 0.9995+ vs the single-token path; teacher-forced argmax vs CPU 0.95-1.0; accepted drafts match CPU statistics).
Not verified: >= 64 tokens (the 127-position cache window after an 80-token prompt limits to ~40 new tokens), more than 8 distinct prompts.

## Root causes found
* Verify rows at T=12/20 wrong: mHC mixes-v2 kernel (above). Fixed in mhc.py.
* Drafter static-CB clash (TT_THROW): moe_compute's semaphore landed under the 655 KB L1 outputs (2 KB warmup hole consumed); DraftMoEBlock.warmup uses a 32 KB hole (in main).
* "0.12 accepted drafts" : inconsistent seeding (CPU chain dump), fixed by feeding the prompt through the device (in main).
* Hangs (MoE a2a dispatch + reduce_scatter, 5 occurrences at the first free-running rounds, k=4/5/7): the test loop kept running FINISHED users in the batch with positions up to base+n-1 > 127 (cache / compress slots end at 127);
  out-of-range positions corrupt routing/dispatch. Fix: finished users are clamped to MAXPOS-n+1. With the fix k=7(n=8,T=32) runs complete (14 rounds). The post-trace allocation of snapshot buffers (allocator warning) was also removed
  (snapshots now allocated before the trace and refreshed in place). Spurious hangwatch/lock deadlocks were launcher issues, not device problems.
* "Compile-pass race" (garbage first tokens once, k3p): NOT reproduced in 12+ later full runs (all first tokens == CPU token). Probable cause: the post-trace snapshot allocation overlapping trace memory (now removed); the test asserts the
  first token against the CPU reference (printed, "device first token ... match"), keep that as a gate.

## Plugging into the paged pools (not implemented, design)
The spec path owns per-user paged caches ([U pages, 1, page, 512], page table rows shared by the n virtual rows of a user, dense paged SDPA). The generator's paged decode uses the shared pool + sparse_sdpa over
per-row index lists built by paged_kv_step. To plug in: treat each (user, j) row as a virtual user with position p+j sharing the user's page-table row; write the n K/V rows (per-j writes into distinct slots) BEFORE
building the n index lists (row j sees positions <= p+j); the window ring of compressed layers needs RING >= 128 + k slots (rejected writes overwrite entries still needed), i.e. 160-192 ring rows per user instead of 128 in the pool
(+ ~4% capacity), compressed-latent slots of the ratio-1/2 owners are rewritten idempotently. The drafter's 3 ring caches (418 KB/user/chip) stay private. Prefill -> decode handoff only needs to seed the drafter rings from the
layers 37-39 attention-input hidden of the last 128 prompt positions (today the prompt is replayed through the verify step, 1.5 s for 80 tokens). Not verified on device.

## Verified vs not verified
Verified (device): drafter == CPU draft tokens given CPU hidden; verify rows PCC; full 40-layer loop k=1..5 (+padded) with device-made state, acceptance vs CPU, exactness near-tie evidence, timing breakdown.
Not verified: paged-pool integration, >64 generated tokens, >8 prompts, exact tok/s target (best 44.8 committed / 38-41 same-host; 50 needs verify cost reduction).
# Verify-cost investigation (against main 450e1196f7c) — no 50 tok/s/user lever found cheaply; pool plug-in NOT implemented

changes.diff (68 lines, `git apply --check` OK): drafter timing hooks only (`DSparkDrafter.stop`, `MTP_BREAKDOWN` in tests/test_spec_mtp.py). No behaviour change.

## Measurements (all on .46, 4 users/mesh row, same build)
* Drafter (U=4, write_main n=3 + draft), traced, cumulative ms: write_main 0.87 | embed 0.44 | stage0 2.48 | stage1 4.30 | stage2 6.47 | head 6.93 | markov x5 + conf 9.01 (total 9.9 with write_main).
  ~2.0 ms per stage (backbone layer ~1.15 ms), markov 2.1 ms (5 sequential embed + matmul + 2 allgathers + argmax). Possible saving ~1.5-2 ms (one packed allgather in sample_global, fused router_select in DraftGate) = 2% of a round: not done.
* Verify cost, 12 layers (2-13, T=16 vs plain T=4), pure device replay: plain 18.1 ms, k=3 25.4 ms (+7.3 ms = +0.6 ms/layer, ~+24 ms over 40 layers incl. the sync-free overlap; full model measured +18 ms).
  Ablations at T=16: per-block-index paged_update_cache calls reduced to ONE (wrong results, timing only): 24.7 ms (-0.7 ms/12 layers = ~2.3 ms of 64) -> the n-launch write is NOT the bottleneck;
  DSV41_MHC_EP=1: 25.5 (no change); DSV41_ATTN_SDPA_FID=HiFi2: 25.0 (-0.4); (DSV41_MHC_CN_FUSED=0 run: see log spec5_abl_cnf0.log).
* Profile sums (T=16 vs T=4, ms over 40 layers, eager+sync): attention 155 vs 117, moe 152 vs 137, mHC (mixes+expand+collapse) 259 vs 189, shared 30 vs 29. The extra cost is spread (rows x 4 in every mHC/attention op), no single outlier at T=16.
* Native T=12/20/24 mHC fast path: the pad-to-8 fix already makes them correct; profile T=20 vs T=16 mixes+collapse = +~110 ms-sum, i.e. a fused T=12/20/24 path would at best bring k=2/4/5 to the k=3 curve (k=4 ~44 tok/s/user est.), not above k=3.

## Where 50 tok/s/user would have to come from (k=3, tokens/round 3.2 => round <= 64 ms)
now: verify 64 + draft 10.5 + host/readback 7-9 = 83 wall (74.6 device). Plain decode already costs 46 device; so verify extra (18) + draft (10.5) + host (8) must all drop to ~10 total.
Biggest remaining levers, all outside this overlay's scope: (a) mHC/attention/MoE kernels scaling sub-linearly with T (they are ~linear now), (b) Engram on device so the host leaves the critical path (-8 ms), (c) fusing drafter stages + packed sampling (-2..3 ms).

## Paged-pool plug-in (virtual-user rows) — not implemented; findings
Feasible pieces already in main: `paged_kv_step` has `nq` (rows of one user consecutive, user = row // nq), `ring_rows` is a parameter (160 for spec slack, note in paged_ops.py), `sparse_sdpa` takes per-row index lists, so causality inside the block needs no mask.
Blockers found (each is real work, together several days, device-verification at ISL 2k needs the indexer):
1. Compressed layers at ISL 2k have >512 compressed entries, so every virtual row needs its own indexer top-512: `DSV41DecodeIndexer` is batched over USERS with one key slab per user and a per-user valid length; n rows per user need either the key slab repeated (n x memory) or an indexer variant taking several queries per key slab; key append/valid length per virtual row (entries completed inside the block).
2. Ratio-2/ratio-1 block compression (prev_cs chain, odd-position-only latent write, commit(m)) exists only in SpecCompressedAttention (non-paged); the paged class writes one latent per user per step.
3. The pool's ring region is [slot][user][ring_rows] sized at pool build: spec needs ring_rows >= 128+k (160) for every handed-off user; prefill->decode handoff (`prefill_handoff.py`) stages 128-row rings.
4. The drafter's 3 stage rings must be seeded from layers 37-39 hidden of the prompt (today the loop re-feeds the prompt through the verify step, ~1.5 s for 80 tokens, impossible at ISL 2k: 2k tokens x 40 layers).
I can do (1)-(4) incrementally if prioritised; the first deliverable would be window-only layers + compressed layers with <=512 entries (ISL <= 1k) on the pool, then the multi-query indexer.


## Stage 1 (spec via paged pool)
# Paged-pool spec verify, stage 1 (against main bdde8dcb6e0): window layers + compressed layers with <= 512 selected entries (no indexer)

changes.diff (= changes_stage1.diff, `git apply --check` OK). Opt-in: `DSV41_SPEC_PAGED=1` in tests/test_spec_loop.py; nothing in the default decode path changes.

## What was built
* `tt/spec_paged.py`: `SpecPagedWindowAttention`, `SpecPagedCompressedAttention` (owner + reader), `SpecPagedChain`. Every (user, j) verify row is a virtual user of `paged_kv_step` with `nq = n`
  (user = row // n): ONE op writes the n ring rows (ring_rows = 160 = 128 + k slack, set at pool build), the group-completing latents (owners), and builds one `sparse_sdpa` index row per virtual row
  (window rows <= pos + compressed entries <= pos: causal inside the block, no mask). Block compression of ratio-2 (prev_cs chain, commit(m)) is reused from SpecCompressedAttention.
* Shared-file edits (all backward compatible for nq = 1 / single-token decode):
  - `tt/paged_kernels/paged_kv_step.cpp`: (a) KV_MODE 0 reads the q tile of the ROW (`row*16`, was `user*16`; equal when nq = 1); (b) the latent is written only by the group-COMPLETING position
    (`(pos+1) % RATIO == 0`; ratio 1 unchanged): even positions of ratio-2 layers used to write a junk entry that the odd position finalises, which races when both rows are in one block.
  - `tt/paged_attention.py`: `_paged_attend` passes `nq=getattr(self, "nq", 1)`.
* `tt/mtp.py`: `DSparkDrafter(max_pos=...)` (rope / mask tables > 256 positions). `tt/spec_decoder.py`: tap fallback for partial-layer debug runs.
* Tests: `tests/test_spec_paged_attn.py` (layer level), `tests/test_spec_loop.py` with `DSV41_SPEC_PAGED=1 DSV41_CTX=<positions>` (full 40-layer loop, prompt fed through the device).

## Verified on device (.46)
1. Layer level, paged vs ORIGINAL single-token attention (random inputs, empty state, n = 4 / 6, up to 12 / 30 blocks): window layer 0.99998; ratio-2 owner (2) 0.9976 (n=4, 12 blocks), 0.9959 (n=6, 30 blocks = positions to 180);
   ratio-1 owner (20) 0.9938 (n=4); paged vs non-paged spec attention 0.9997+ (n = 3, 4) incl. readers (3, 21). Reader layers vs an ORIGINAL reader in this synthetic test are not comparable (0.2-0.9: the test's original reader copies an empty owner cache);
   paged == non-paged spec on readers (0.9997).
2. FINDING: the NON-paged spec attention is WRONG beyond position 127 for ratio-2 layers (PCC vs original 0.97 -> 0.7 for the odd rows from position 131; paged stays 0.996). This is why the non-paged loop was limited to 127 positions (more than 64 compressed entries => second k-chunk bug in the composite SDPA path).
3. Full 40-layer loop, paged, k=3, 16 users, 40 tokens (spec6_pg_k3_g40.log): first token == CPU 16/16, acceptance 2.214 accepted/round (non-paged 2.196, CPU ref 2.30), 3.21 tokens/round, device round 75.4 ms (non-paged 74.6), wall 82.4 ms vs 83.4 =>
   no speed penalty; teacher-forced drafter d1..d5 0.905/0.778/0.667/0.524/0.365; stream vs the non-paged k=3 stream: 10/16 identical (divergences at 18, 3, 4, 19, 7 = the same near-tie prompts as plain-vs-spec).
4. Long run, paged, 200 new tokens (positions to 280, impossible before): plain (k=0) 53.2 ms/token wall, 18.8 tok/s/user, 0.919 token agreement with the CPU reference over the first 64 tokens; k=3: 68 rounds, 2.243 accepted/round, 3.24 tokens/round, 85.5 ms wall -> 37.9 tok/s/user (2.0x plain).
   Exactness vs paged plain over 200 tokens: 0/16 identical (first divergence 18-193); near-tie evidence (top1-top2 logit gap at the divergence, plain / spec): 0.002/0.214, 0.038/0.010, 0.126/0.232, **2.134/0.244 (user 3, token 165)**, 0.106/0.038, 0.029/0.088, 0.231/0.243, 0.216/0.199.
   Seven of eight are near-ties; user 3 (position ~245) is a 2.1-logit shift between the 1-row and 4-row paths and is NOT explained by a tie: open item (could be bf16 accumulation at long context or a small long-position error in the spec path; layer-level PCC at position 180 is 0.996).

## Not verified / next
* ISL > 512 for ratio-1 layers and > 1024 for ratio 2 needs the indexer (stage 2), ring 160 in pool/handoff (stage 3: today `SpecPagedChain` builds its own pool with ring_rows = 160 and starts from an empty cache, the prompt is replayed through the device).
* Pool is bf16 (fp8 pool untested with spec). Engram / drafter rings unchanged.


## Stages 2 and 3 (multi-query indexer, ring 160, drafter seeding)
# Paged spec verify: user-3 outlier, stage 2 (multi-query indexer, ISL 2k), stage 3 (ring 160, drafter seeding) — against main f06259db6f4

changes.diff (`git apply --check` OK): `tt/spec_paged.py` (SpecIndexer + indexer wiring in the chain/attention), `tests/test_spec_loop.py` (long prompts, teacher-forced replay of the plain stream, tail-replay check), `tt/dsv41_model.py` (one line: `ring_rows` from `DSV41_RING_ROWS`, default 128 = unchanged).

## 0. User-3 token-165 outlier (bounded investigation)
Teacher-forced replay of the paged PLAIN stream through the k=3 verify step (`DSV41_TEACHER_FULL=1`: every verify row sees exactly the plain history; 8 distinct prompts x 200 positions = 1600 rows, spec4/spec7_tf_k3b.log):
14 argmax mismatches (0.9%). 13 are near-ties (plain top1-top2 gap 0.002-0.5, spec gap 0.03-0.58). User 3, position 244 (block row 0): plain gap 2.134, spec argmax differs with spec gap 0.819, and user 3 mismatches again at position 262 (gap 0.265 / 0.075).
It reproduces under teacher forcing, so it is not trajectory drift; it is not at a ring wrap (slot 84) or a ratio-2 group boundary (even position, block row 0). Layer-level paged-attention PCC vs the original is 0.996+ out to position 180, the paged_kv_step edit is exercised by every other row. Most likely a MoE routing near-tie flip
(1-row vs 4-row gate numerics change an expert -> ~1-2 logit shift) but not isolated (needs per-layer routing capture of the two paths at that position). Not a systematic attention error: 99.1% of rows agree.

## 1. Stage 2: multi-query indexer (ISL 2k)
`SpecIndexer(DSV41DecodeIndexer)` (matmul backend): the n rows of a user are stacked on the head axis ([U,1,32n,128] @ K_u^T), per-row head weights, selection with per-ROW valid counts (causal inside the block); keys of group-completing rows are written
first (one `paged_update_cache` per block index, others -1); ratio-2 keys only from odd positions. Index-source layers 2/8/14/20 own key slabs (bfp8, [U,1,n_alloc,128]), 24/28/32/36 score against layer 20's slab, other layers reuse `st["topk_ids"]`.
Enabled automatically when `DSV41_CTX > 512`. `tests/test_spec_loop.py DSV41_PROMPT_DIR=/mnt/tt-data/ssinghal/dsv4-prefill-s2048b1f` feeds the REAL 2048-token prompt through the device (tiled over 16 users) and checks the first token against the CPU reference.
Results (4 users/mesh row, 16 users, .46):
* plain (k=0, n=1 = the indexer at n=1): prompt replay 2048 tokens in 102 s, **first token 223 == CPU reference**, 50.2 ms/token wall, 19.9 tok/s/user at ISL 2048 (spec7_isl2k_k0.log).
* k=3: prompt replay 512 blocks in 41 s, **first token == CPU reference**, 22 rounds: 2.045 accepted/round (P(m>=1,2,3) = 0.955/0.545/0.545), 3.05 tokens/round, round wall 79.8 ms = verify 61.6 + draft/commit 10.4 + host 7.7 => **38.1 tok/s/user (1.9x plain)** (spec7_isl2k_k3b.log).
* Exactness vs plain at ISL 2k: first 25 generated tokens identical for all users, then a tie (plain top1-top2 gap 0.001 vs spec 0.266).
Not verified: acceptance statistics at ISL 2k over many prompts (one real prompt, 22 rounds), ISL > 2k, fp8 pool, the per-row indexer top-512 vs a host top-512 reference (only end-to-end first token + stream equality).

## 2. Stage 3
* Ring 160 in the pool / hand-off: `PagedStateSink` and the decode attention take `ring_rows` from the pool (already parametrised); the generator's pool now reads `DSV41_RING_ROWS` (default 128). Full device prefill -> paged hand-off -> decode e2e test at S=128 with `DSV41_RING_ROWS=160` PASSES with numbers identical to the ring-128 baseline
  (first token PCC 0.97198, argmax 14/16; teacher-forced steps 0.9833/0.9753/0.9796; ring PCC min 0.928; spec8_e2e_ring160.log vs h47i_full_s128.log): ring 160 is a numerical no-op for non-spec decode. Cost: 40 layers x 4 users x 32 extra rows x 1 KiB = 5 MiB/chip.
* Drafter seeding at hand-off WITHOUT touching prefill ("tail replay"): after prefill the pool / compressor state is complete but the drafter's three main_kv rings are empty and `prev_cs` is only valid for the last prompt token. Replay the LAST 128 prompt tokens through the verify step from an EVEN start position
  p0 = ((S-128)//2)*2 with the accept count forced (the rewrite of the already-written positions is idempotent, even rows write no latent so `prev_cs` at the start is never read). Verified emulation at ISL 2048 (`DSV41_TAIL_CHECK=1`: after the full replay, prev_cs of all ratio-2 owners is ZEROED, then [1920, 2048) is replayed in 32 blocks):
  **first token identical**; drafts d1,d2 identical, d3..d5 differ (0.40 of entries equal; the verify/MoE path is not bit-reproducible run to run, plain-vs-plain streams already differ across builds) and the loop after the tail replay behaves normally (2.71 accepted/round, 47.2 tok/s/user over 7 rounds, spec8_tail_2k.log).
  Cost ~2.6 s per batch (32 blocks x 80 ms). Not verified: seeding after a REAL device prefill + hand-off (this container of tests starts from the replay path), ragged per-user prompt lengths (the verify step takes per-user positions, so per-user p0 works in principle).
* Direct tapping of layers 37-39 hidden inside the prefill chunks would avoid the 2.6 s but changes prefill_model (not done).
