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
