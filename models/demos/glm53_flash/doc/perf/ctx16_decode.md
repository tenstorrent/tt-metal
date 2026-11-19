# 16-user decode at long context: where the step goes

Status 2026-11-13: measurement plan and fix design. Nothing new has run on the device yet; the chain is
`_logs/mesh/chain_ctx_1.sh`.

## The question

The AgentX model (`agentx_scaling_model.md`) puts the 16-user ceiling on the full-width decode step at long context. Two
numbers were set side by side: the served step at 60-500 K context (38.6 ms, F10, a mean of about 6 live rows on the
16-row kernel) and the metal step at 4 K x 16 (31.3 ms). Because one is served and the other metal, the ~7 ms gap mixes
host overhead with context cost. This note isolates the context term.

## What the existing probes show

All rows below are traced decode steps with B 16 native, 16 live users, with the pooled-key stores **zero-filled**.

| Probe | ISL (per user) | OSL (per user) | Concurrency | Step |
|---|---|---|---|---|
| full 45 layers (11-12, older tree) | 4,096 | 20 steps | 16 | 29.4 ms |
| full 45 layers (11-12, older tree) | 1,048,000 | 20 steps | 16 | 57.1 ms |
| layers 0-3 (1 sparse_dsa layer) | 4,096 | 20 steps | 16 | 2.80 ms |
| layers 0-3 (1 sparse_dsa layer) | 1,048,000 | 20 steps | 16 | 5.39 ms |

All of the context cost is in the sparse_dsa layers: +2.54 ms per DSA layer, × 11 ≈ the +27.7 ms whole-model delta. The
KDA layers are state-based and context-free.

Phase split of DSA layer 3 on chip (0, 0) (traced per-chip shard marks, µs):

| Phase | 1 user, 4 K | 1 user, 1 M | 16 users, 4 K | 16 users, 1 M |
|---|---|---|---|---|
| index scoring → first radix round | (LOCAL_ALL) | 142 | (LOCAL_ALL) | 230 |
| local radix rounds (R8, 4 + tie) | 0 | 65 | 0 | 149 |
| → local coordinator (tie, record write) | 74 | 1 | 87 | 224 |
| → all-gather landed | 91 | 42 | 218 | 198 |
| → global merge done | 47 | 127 | 89 | **705** |
| attention (att_start → end, median chip) | 305 | 498 | 696 | 2,221 |

## Why these 1 M numbers overstate the cost

With zero keys every index score is equal:
- The radix rounds never compact, so every round scans every key.
- The tie round has to resolve every key.
- Ties resolve to the lowest pool ids, so a few chips own every user's selected pools. That chip's attention runs
  over all 16 users while the rest wait on the reduce-scatter: at 1 M, `att lsel` max is 769 µs against a median of
  56 µs.

At one user, the same probe gives +348 µs per DSA layer at 1 M, while the real e2e decode step grows only +118 µs per DSA
layer (19.6 → 20.9 ms over 11 layers). The zero-key probe inflates the context term about 3x. A real 16-user × 1 M e2e
costs 16 × 282 s of prefill, which we can't afford.

`GLM53_MK_PROBE_RANDKEYS=1` (this branch, `tests/test_mk_backend_probe.py`) fills every DSA layer's pooled-key store
with random keys that differ on every chip. Scores are then distinct and spread over the chips, which fixes the timing;
the decoded values stay garbage.

## The costs that scale with users × context

Each user's pooled keys are sharded round-robin by 128-token units over the 32 chips. At context c a chip holds
c / 4 / 32 pools per user: 32 at 4 K, 256 at 32 K, 8,192 at 1 M.

- **Index scoring** reads every local pooled key: 16 users × 8,192 pools × 128 × 2 B = 33.5 MB per chip per layer at
  1 M. This is data-independent and real: 230 µs at 1 M against an ~65-100 µs DRAM floor.
- **Local select** only runs past 512 local pools (LOCAL_ALL below ~64 K): R8 rounds over 8 K keys per user.
- **Global merge** takes each chip's exact local top-512 list (`CAND_MAX` = 512 records). Every chip merges all 32 lists
  of every user to the exact global top-512 (threshold T_g, ties in (chip, core, local) order). That is 32 × min(512,
  local pools) candidates per user: 1 K at 4 K, 8 K at 32 K, 16 K at ≥ 64 K, the same 16 K for any AgentX context
  (60-500 K). At t = 16 with the galaxy defaults (KPH auto → 8 lanes, 2 users per lane, 4 lists per merge helper),
  every radix round scans 16 × 16 K keys on the 64 chunk cores.
- **Attention** over the 512 selected pools (2,048 tokens) per user is spread over the chips by the round-robin units.
  Real selections are not hot-chip, so this should stay close to the 4 K cost.

## Fix candidates, ranked

All expected savings are estimates until `chain_ctx_1` measures them.

1. **Exact merge pre-filter: K16 lower bound.**
   - **Rule.** Let K16_q be the 16th largest key of chip q's list (512 / 32 = 16). Every chip has at least 16 keys
     ≥ K16_q ≥ min_q K16_q, so the union holds at least 512 keys ≥ L = min_q K16_q. Therefore T_g ≥ L: every key
     < L is outside the global top-512, and every key equal to T_g is ≥ L.
   - **Procedure.** The merge helpers compute K16 of each of their lists: two digit passes over 512 keys, or a
     16-entry min-heap. They report the minimum to the lane coordinator, the coordinator broadcasts L (one extra
     handshake round, the same message shape as a radix round), and the helpers compact their lists to keys ≥ L in
     one scan.
   - **Cost after the filter.** The R8 rounds and the tie round then see about 512 + O(32) survivors per user instead
     of 16 K, and the selection and its tie order are unchanged, so ids stay identical.
   - **Cost of the filter itself.** One heap pass + one compaction pass + one handshake, instead of 4-5 full scans and
     a tie round over 16 K keys.
   - **Expected saving.** About 0.3-0.55 ms per DSA layer at ≥ 64 K with 16 users, ≈ 3-6 ms per step (11 layers), or
     +10-20% tok/s at 16 users × long context. Half of that at 32 K (8 K candidates per user).
   - **Where.** The kernel work is in `dsa_shard.hpp` / `ix_dm0.cpp`: the merge helpers and the coordinator, a new
     mailbox message, gated by a define, e.g. `MLA_DSA_SHARD_MERGE_K16`. It is not contained, and the hang risk sits in
     the new handshake. Do it only if the random-key probe shows the merge is the leading term.
2. **16 token lanes at t = 16** (`GLM53_MLA_DSA_LANES=16`, knob only).
   - **Change.** One user per lane coordinator instead of two; KPH auto → 8, with 4 helpers per lane each holding 8
     lists. The per-core scan work is unchanged.
   - **Saving.** It halves the coordinators' per-user handshake latency in the local rounds and the merge. The TP2 card
     test showed 699 vs 721 µs at t = 16 (-3%). Expected 0.05-0.15 ms per DSA layer.
   - **Requirement.** 16 free cores for the lane coordinators; the build prints `MK_SHARD_LANES capped` otherwise.
     `chain_ctx_1` runs the A/B.
3. **bfp8 pooled keys.** This halves the scoring read (33.5 → 17 MB per chip per layer at 16 × 1 M), ~0.1 ms per DSA
   layer. The prefill writer, the decode reader and the ring all change, and accuracy needs the real-weight check (the
   reference indexer uses fp8 keys). Later.

## Measurement plan (`chain_ctx_1`)

- **Probes, layers 0-7** (2 DSA layers), B 16 native, 16 live users, 20 traced steps, random keys, at pos 4 K / 32 K /
  1 M. Plus a zero-key control at 1 M, and random + 16 lanes at 1 M.
- **Full 45-layer probes,** random keys, at pos 4 K / 32 K / 1 M.
- **Real-data e2e** at 4 K x 16 and 32 K x 16 (OSL 128). A 1 M x 16 e2e is not run (75 min of prefill).
- **Conditional row:** e2e 32 K x 16 with 16 lanes, only if the 1 M lanes A/B wins by more than 2%.

Every booked line carries ISL (per user), OSL (per user) and concurrency. The probes book the traced per-chip phase
split (min / median / max µs) for DSA layer 3, so the merge, local select, scoring and attention terms read directly
against the table above.

## F19 (2026-11-14): the exact merge pre-filter, built (`GLM53_DSA_SHARD_PREFILTER`, mk_mla rev 74)

The user's target is ISL ~1 M at 1 / 2 / 4 / 8 / 16 users. Every DSA layer's global merge there runs on full lists: each
of the 32 chips' local top-512 lists is full, so the merge helpers histogram and compact 32 x 512 = 16,384 keys per token.

**Bound.** K_q = the 16th-largest key of chip q's list (16 = TOPK / NCHIPS); L = min_q K_q. Each chip has >= 16 keys
>= K_q >= L, so the union has >= 512 keys >= L and T_g >= L. Every key < L is outside the global top-512; every key
>= T_g, the ties included, is >= L. (The bound uses the MIN over the chips; the max would not be safe.)

**Protocol** (all on-chip, after the all-gather; no new semaphore, fabric phase or L1 buffer):
1. The coordinator sends start (merge slot 0, as before).
2. Each helper computes, for each of its KPH lists, the 16th-largest key with a 16-entry sorted insert (one compare for
   most keys), and reports (min over its lists, its key count) through the existing H_SEM report.
3. The coordinator takes L = the min and the union's total, and sends L in merge slot 0. No helper reads the start
   slot's content. L = 0 (no filter) when a list holds < 16 keys or the union holds <= 512.
4. The helpers keep their keys >= L, stable per list:
   - KPH 1: into `mcand`. `mkeys` stays whole for helper rank's per-core tie count.
   - KPH > 1: in place, `len[i]` updated.
5. The 4 R8 rounds and the tie round then run unchanged on the survivors. Round 0's "<= TOPK: done" check reads the
   union's total, not the survivors.

**Why the selection is identical.** Every dropped key is strictly below T_g. In each round, the buckets above T_g's
digit lose no key, and T_g's own bucket keeps every key >= T_g. So each digit pick, k_rem, the per-list counts of keys
== T_g (list order kept), the per-chip quotas (chip order) and helper rank's per-core quotas (re-read from this chip's
x_ag records) are the rev-73 integers. Emulated by `shard_plan.helper_merge(prefilter=True)`;
`test_shard_plan_8x4.py::test_helper_merge_prefilter` holds it equal to the unfiltered and the serial merge on 11 cases
(random, ties, few, empty, one_chip, all_empty, lengths, dups, adversarial, boundary, full), at KPH 1 / 2 / 4 and 4 / 8-bit
digits.

**Survivors** (emulator, 32 full lists):

| Lists | Survivors |
|---|---|
| uniform | 667-779 of 16,384 |
| one exponent bin (24 shared bits) | 871 |
| chips with very different score scales | 14,347 (the bound is weak; the round-robin 128-token sharding makes this unlikely) |

**Kernel check (no device).** `-fsyntax-only` with the 8x4 JIT flags at KPH 1 / 2 / 4: error sets identical with and
without the define (only the harness's placeholder compile-time args fail). A marker `static_assert` in each pre-filter
block fires only with the define.

**Cost model (estimates, not measured).** From the zero-key B 16 phase split (merge 705 us per DSA layer =
2 tokens x 9 full passes over 2,048 keys + handshakes): about 18 ns per key-pass, about 5 us per handshake.
- Base: random keys need about 1 heavy round (hist + compaction); real index scores, whose order keys concentrate in a
  few exponent bins, need about 2.
- Pre-filter: a K16 pass (about 6 ns per key) + one compaction pass + one handshake, then rounds on about 100 keys per
  helper.

| ISL (per user) | OSL | Concurrency | Width / KPH | Merge saving per DSA layer | Per step (11 layers) |
|---|---|---|---|---|---|
| 1,048,000 | -- | 16 | 16 / 4, 2 tokens per lane | ~70 (random keys) - 215 (real) us | ~0.8 - 2.4 ms |
| 1,048,000 | -- | 8 | 16 / 4, 1 token per lane | ~35 - 107 us | ~0.4 - 1.2 ms |
| 1,048,000 | -- | 1 / 2 / 4 | 4 (narrow) / 2 | ~13 - 50 us | ~0.15 - 0.55 ms |

F16's -3 to -6 ms at 16 users was scaled from the inflated zero-key merge. The probe matrix measures it.

**Follow-ups.**
- After the filter about 800 keys survive. Collapsing the 4 radix-round handshakes into one survivor gather on the
  coordinator would save another about 20 us per token.
- Width 8 for 5-8 users (wt_w8 0e1a6b024bc) once both are in one tree.

**Chain.** `_logs/mesh/chain_pf2_1.sh` (runner `run_mk_pf2.sh`, shadow root `tt-metal-pf2`):
1. A hang-safe layers 0-7 probe at 1 M x 16.
2. The ids gates: 32 K x 16 and ~1 M x 1, base vs pre-filter.
3. The full-model 1 M probe matrix at N 16 / 8 / 4 / 2 / 1, plus N 1 on the B 4 build. Random keys; the b16 shape is
   narrow width 4 for N <= 4 and width 16 for N 8 / 16, via the probe's new `GLM53_MK_PROBE_WIDTH` /
   `GLM53_MK_PROBE_LIVE`.
4. ~1 M x 4 e2e.
