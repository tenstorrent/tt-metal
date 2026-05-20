# Step #1 implementation findings (PASSED) — extended with steps #3 and #4 (PASSED)

Companion to [chunked_prefill_test_plan.md](chunked_prefill_test_plan.md). Captures
what step #1 actually delivers, what it doesn't, and what to watch out for next.

**Status update (step #3 landed, all four chunks now ≥ 0.99 PCC):**

| Chunk | logical_n | step #1 | step #3 |
|------:|----------:|--------:|--------:|
|     0 |      5120 |  0.0002 |  0.9997 |
|     1 |     10240 | −0.0000 |  0.9996 |
|     2 |     15360 |  0.6722 |  0.9994 |
|     3 |     20480 |  0.5677 |  0.9993 |

See "Step #3 implementation findings" at the bottom of this doc.

## What's done

**Q-vs-K split applied across program factory + 3 kernels.** Net CT-arg count
unchanged (critical — see "Binary-size landmine" below):

- **Slot 7** repurposed: was `local_padded_N` (rows), now `q_local_padded_Nt` (tiles).
- **Slot 8** unchanged in position, renamed: `local_padded_Nt` → `kv_local_padded_Nt`.
- `kv_local_padded_N` (rows) is derived in-kernel as
  `kv_local_padded_Nt * tt::constants::TILE_HEIGHT` only where needed
  (writer/compute's `global_n_within_ring_iter`). Folds at compile time.
- Host side: `q_local_padded_N = q_shape[2]`, `kv_local_padded_N =
  input_k.logical_shape()[2]`. `num_local_q_chunks` derives from `q_*`,
  `num_local_k_chunks` from `kv_*`. K-share path (`NHK == 1`) untouched.

**Boundary-of-logical_n fix** in `ring_iter_does_work` and `find_last_active_ring_iter`:

- Old: `ring_iter_kv_start_tile <= logical_n / TILE_H`. Off-by-one: when
  `logical_n` is an exact tile multiple, the iter whose start row equals
  `logical_n` was wrongly counted as having work — entering with
  `per_q_valid_kv == 0`, which deadlocks the writer waiting on `cb_signal`.
- New: `<= (logical_n - 1) / TILE_H`. Strict in the exact-multiple case,
  identical in the partial-tile case. Function signature unchanged — callers
  just pass the shifted RHS. (Pre-condition: `logical_n >= 1`.)

**Status:**

- `test_ring_joint_attention_sdpa_accuracy[mla_100k-q160-k256]` (the regression
  canary): **PASS**.
- `test_ring_joint_attention_sdpa_chunked_accuracy`: runs all 4 chunks
  end-to-end, no hang. `num_program_cache_entries == 4` matches expectation.
- PCCs (mla_100k, sp=4, q=k=160, 20K dev shape):

  | Chunk | logical_n | PCC (after step #1) | PCC (before) |
  |------:|----------:|--------------------:|-------------:|
  |     0 |      5120 |              0.0002 |       0.6117 |
  |     1 |     10240 |             −0.0000 |       0.3533 |
  |     2 |     15360 |              0.6722 |       0.2470 |
  |     3 |     20480 |              0.5677 |       0.1873 |

## The problem step #1 alone doesn't fix

**Chunk 0 went down, not up.** The plan predicted ≥ 0.99 after step #1; we got
0.0002. Root cause is structural, not a step-#1 bug:

- Chunked-prefill test runs `is_causal=True` + `is_balanced=False` + sp=4. For
  chunk 0, only device 0's K-shard has real data; devices 1-3 hold padding.
- After AllGather, device 1 (Q rows [1280, 2560)) receives device 0's K-shard
  (positions [0, 5120)) on the ring iter where `ring_id == 0`. That iter has
  `ring_iter > 0`, so `is_causal_iter == false` — **kernel applies no causal
  mask**.
- Device 1 then attends Q[1280..2559] to **all** of K[0..5119]. Causally it
  should mask K[2560..5119]. Without `q_start_idx`, the kernel doesn't know
  device 1's absolute Q positions, so it can't apply the cross-device cutoff.
- Devices 2 and 3 fail the same way at higher Q positions. ~75% of the output
  rows are wrong; PCC ≈ 0.

**Why this doesn't bite non-chunked.** Per-device Q-shard size == K-shard size,
so any K-shard a device receives is *entirely* before that device's Q
positions (no per-row mask needed) or *entirely* after (skipped via `is_causal
&& ring_index < ring_id && !is_balanced`). In chunked, K-shard > Q-shard, so a
K-shard partially extends past a device's Q range — that's the new case.

**Why pre-step-#1 chunk 0 was "more right by accident."** The broken K
traversal only walked the first `Sq_chunk_t * ring_id` slice of K. For ring_id
0 on device 0, that slice happened to coincide with the causally-valid range
— buggy but partially aligned. After step #1 the kernel walks the *full* K,
exposing the cross-device causal gap in full.

## How to solve it

**Step #3 of the plan is the load-bearing fix.** Wire `q_start_idx` to the
kernel + add the chunked-prefill mask mode (non-causal builder, gated by
`q_start_idx.has_value()`). With absolute Q positions available, the
chunked-prefill mask becomes:

- K cols `[0, q_start_idx)`: dense (attend all).
- K cols `[q_start_idx, q_start_idx + Sq)`: standard causal triangle.
- K cols `[q_start_idx + Sq, logical_n)`: already handled by existing
  `logical_n` cutoff.

Mask is `Sq × Sq` and independent of K-shard layout / ring rotation — the
exact property the plan calls out.

**Implication for the plan's step #1 acceptance criteria.** "Chunk 0 PCC →
oracle (≥ 0.99)" was wrong as a step-#1 signal. The realistic step-#1 signals
are:

1. Non-chunked regression preserved (✓).
2. Chunked test runs to completion without hang (✓).
3. Program-cache entries match `n_chunks` (✓).
4. PCC degradation pattern matches the "kernel walks full K, no cross-device
   causal" prediction — i.e., chunk 0 worse than baseline (full K with
   no mask attends to more garbage), chunks 2-3 better than baseline (more
   real K reached in causal range). Observed (✓).

Chunk 0 ≥ 0.99 becomes step #3's acceptance criterion, not step #1's.

## Landmines for future PRs

**Binary-size budget on TENSIX is tight.** A naive Q/K split that adds even
*one* CT-arg slot to reader+writer trips `Program size (...) too large for
kernel config buffer` on `mla_100k-q160-k256`. The slot-7-repurpose trick
above kept the count flat. If you add CT args here, run that test as a canary
before assuming the diff is free.

**`find_last_active_ring_iter` and the in-kernel `ring_iter_does_work` check
must agree.** They share the `kv_start ≤ tile_id` comparison; if you fix one
boundary without the other, the writer enters a no-work iter and waits on
`cb_signal` that compute never pushes — clean device hang. Fix both, or
neither.

**Comparison form matters.** `ring_id * kv_local_padded_N < logical_n` is the
correct row-aligned semantics but requires a separate `kv_local_padded_N`
constant and a new multiplication; the previous attempt at this form bloated
the kernel binary by exactly the 272 bytes that cause the TENSIX overflow.
Prefer the tile-aligned form `kv_start <= (logical_n - 1) / TILE_H` — it
preserves the baseline arithmetic shape and folds at compile time.

## Next step

Proceed to step #3 (`q_start_idx` plumbing + chunked-prefill mask mode). Step
#2 (padded-K-tail probe) is meaningless until chunk 0 itself works, so defer.

---

## Step #3 implementation findings

**What's done.** Wired `q_start_idx` to the kernel and added a chunked-prefill
mask mode driven by the lightweight diag-tile machinery. Test now uses
`is_causal=False` + `q_start_idx=i*chunk_size`, all 4 chunks pass PCC ≥ 0.99.

**CT/RT plumbing.**

- CT arg `chunked_prefill_enabled` added to writer (slot 29) and compute
  (slot 40); reader untouched. Reader's existing skip logic (`is_causal &&
  ring_index < ring_id && !is_balanced`) doesn't fire for chunked
  (is_causal=False) so its skip predicate is correct as-is.
- RT arg `q_start_idx_t = q_start_idx / TILE_HEIGHT` added to compute
  (slot 2, after global_q_start/global_q_end). Writer doesn't need it —
  diag stamping is compute-only.
- Tile-aligned `q_start_idx` enforced via host TT_FATAL.
- `chunked_prefill_enabled` shares the diag-tile CB slot with `is_causal`:
  `diag_tile_enabled = is_causal || chunked_prefill_enabled` gates the
  diag tile generation in `generate_lightweight_mask_tiles` and the mask
  CB layout indices in the compute kernel.

**Mask mode.** `apply_causal_mask_lightweight` works as-is with absolute Q/K
tile coordinates. `sdpa_ring_v2` gets a new template param
`chunked_prefill_enabled` plus runtime params `q_start_idx_t`, `ring_index`,
`q_local_padded_Nt`. When chunked:

- `q_start_tile = q_start_idx_t + ring_index * q_local_padded_Nt + q_chunk * Sq_chunk_t` (absolute).
- `k_start_tile = ring_id * local_padded_Nt + k_chunk * Sk_chunk_t` (absolute).
- `apply_causal = is_causal_iter` is true for every iter (vs. `ring_iter == 0`
  for is_causal_sdpa).
- `causal_k_limit`, `try_skip_causal_above_diag`, and the causal `active_Sk`
  narrowing are all `if constexpr (is_causal_sdpa)` so they don't fire for
  chunked. Instead, the diag stamp's `diag_col < 0 → all neginf` branch
  handles fully-masked K chunks; `diag_col >= num_cols → no mask` handles
  fully-attendable K chunks.

**Streaming-compute requirement.** Host-side TT_FATAL: chunked-prefill
requires the streaming compute path. The legacy `sdpa_ring` /
`sdpa_inner_loop` builds `q_start_tile` and `k_start_tile` internally in the
local frame, and threading absolute coords through would touch more
surface than is justified pre-streaming-only. For mla_100k Sq=Sk=5 the
selector picks (sbh, sbw) = (1, 5) → `qk_in0_num_subblocks = 5 > 1` →
streaming engaged.

**The "first non-skipped iter" bug** (pre-existing, surfaced by step #3).
The compute kernel and writer both used `(ring_iter == 0)` to mean "first
ring iteration", which assumed every ring iter has real K. Chunked prefill
breaks that assumption: for chunk 0 with `logical_n=5120` and `sp=4`, three
of the four ring iters carry all-padding K and are skipped via
`ring_iter_processes_KV_chunks`. The next active iter would then be treated
as `ring_iter > 0` and try to restore non-existent prior accumulators —
producing PCC ≈ 0 on chunk 0 and tapering up as fewer iters are skipped.
Pre-step-#3, the K-walk-full bug accidentally hid this on chunk 0 by
restricting the K walk to a range where most devices read zeros anyway.

The fix: track an `active_iter_idx` in both compute and writer outer
loops; `is_first_active_iter = (active_iter_idx == 0)` is passed down and
used in place of `ring_iter == 0` for:

- Compute (`sdpa_ring_v2`): `is_first_kv_for_this_q` and
  `restore_from_staging = q_per_core > 1 && !is_first_active_iter`.
- Writer: gating `complete_restore` and the intra-ring prefetch.

The cross-ring prefetch at end-of-iter is unaffected: data prefetched at end
of iter N for iter N+1's Q[0] sits in CBs through any skipped iters and is
consumed by iter N+2's `complete_restore`. Pop counts still match.

**Mask-CB validation relaxation.** Two `!args.is_causal` host validations
(input-dtype equality, head-dim equality) were narrowed to
`!is_causal && !q_start_idx.has_value()`. Chunked-prefill rides on
`is_causal=False` but targets MLA, which needs the same relaxed checks
as is_causal=True (BF16 Q + BF8_B KV; K head_dim == Q != V).

**Status.**

- Chunked test, mla_100k 20K, q_start_idx=0/5K/10K/15K: PCC 0.9997, 0.9996,
  0.9994, 0.9993 — all ≥ threshold 0.99.
- Chunked test, mla_128k 20K (same q/k chunk sizes): equivalent PCCs, all PASS.
- Non-chunked canary `mla_100k-q160-k256`: PCC 0.9996 (was 0.9996 pre-step-#3
  — no regression).
- All three mla_100k non-chunked configs (k=160, k=256, k=320) pass.
- `num_program_cache_entries == n_chunks` still (logical_n on CT key); step
  #4 will collapse it.

## Step #3 landmines for future PRs

**Streaming-only constraint.** Adding chunked-prefill to the legacy
`sdpa_inner_loop` requires extending `lw_mask.apply()` to accept absolute
`k_start_tile` (or a `k_chunk_offset_tile` ride-along) and threading
absolute Q-offset params through `sdpa_ring` → `sdpa_inner_loop`. The
host-side TT_FATAL keeps us honest until then.

**The `is_first_active_iter` mirror.** Compute and writer must agree on
which ring iters are "active". Both compute the same skip predicate from
the same CT-arg-derived quantities (logical_n, kv_local_padded_Nt, ring_id
sequence via the fused-op indexer / receiver). Diverging the two would
desync the writer's signal-wait against compute's signal-push and deadlock.

**CT-arg slot budget.** Writer slot 29 and compute slot 40 are now taken.
The `mla_100k-q160-k256` canary still compiles (no binary-size overflow),
but each new CT bit chips at the margin. Keep an eye on it.

## Next step

Proceed to step #4: move `logical_n` from CT to RT so the chunked loop
shares one program-cache entry instead of `n_chunks`. Step #2 (padded-K
sweep) is now meaningful — chunk 0 works — so it can run as a verification
step after #4.

---

## Step #4 implementation findings (PASSED)

**Result.** `num_program_cache_entries` collapsed from `n_chunks` (4) to **1**
for the chunked-prefill test. mla_100k and mla_128k both confirm. Asserted in
the test (`assert cache_entries == 1`). All 4 chunks PCC ≥ 0.999.

**Mechanism — what the program hash actually keys on.** Three pieces had to
match:

1. `RingJointSDPAParams::attributes()` — used by the default program-hash for
   most ops.
2. `RingJointSDPADeviceOperation::compute_program_hash` — the op overrides it
   with a custom hash that lists fields explicitly.
3. The kernel CT args — anything baked at compile time can't vary across
   calls that share a cached program.

`logical_n` appeared in all three. Dropped from (1) and (2) only when
`q_start_idx.has_value()` (chunked); for non-chunked we keep it in the hash
so each `logical_n` builds a fresh program (preserves the existing
`global_n_partial_col` CT-arg correctness path). For (3), `logical_n` (slot
10) and `logical_nt` (slot 11) are kept as CT args repurposed as
**layout hints** — they drive constexpr derivations of mask-CB size
(`global_n_partial_col`, `total_mask_tiles`, `needs_lightweight_mask`) only.
Runtime decisions in the kernel use the RT `logical_nt` instead, which is
refreshed per call via `override_runtime_arguments`.

The chunked-prefill TT_FATAL on `q_start_idx % TILE_HEIGHT == 0` keeps
`logical_n` tile-aligned across all chunks, so the CT layout hint (baked at
first-chunk values) and the RT `logical_nt` stay layout-consistent: same
partial-col=0, same `needs_lightweight_mask`, same mask CB tile count.

**RT slot layout — gated on chunked.** The plan said "RT arg
`q_start_idx_value` (0 when disabled)" but unconditional RT pushes tripped
the kernel-config buffer budget on the `mla_100k-q160-k256` canary (program
size 71552 > max 70656). Solution: only push the chunked-specific RT args
(`q_start_idx_t`, `logical_nt`) when `q_start_idx.has_value()` — both in
`create_at` and the kernel reads (gated by `if constexpr
(chunked_prefill_enabled)`). Non-chunked builds keep step #3's RT layout
exactly, so the canary stays under budget. Chunked builds pay the extra
RT-arg cost (slot count and a kernel-side load) but their binary has the
chunked path active anyway.

**Reader's `chunked_prefill_enabled` slot.** Reader didn't have this CT
arg in step #3 (only writer and compute did). Step #4 repurposes reader's
slot 24 — currently `use_streaming_compute` in the host arg vector but
unread by the reader kernel — to carry `chunked_prefill_enabled`. Writer
and compute keep `use_streaming_compute` in their respective slots
(writer slot 24, compute slot 33). The slot has different meaning across
kernels; the host wires it accordingly. Saves one new CT slot vs. inserting
a fresh one and shifting `TensorAccessorArgs` offsets.

**Tile-units rewrite of the runtime path.** Several runtime expressions
used `logical_n` in rows (`(logical_n - 1) / TILE_HEIGHT`,
`logical_n - ring_id * kv_local_padded_N`, `logical_n % (Sk_chunk_t *
TILE_H)`). For tile-aligned `logical_n` they're equivalent to tile-unit
forms (`logical_nt - 1`, `(logical_nt - ring_id * kv_local_padded_Nt) *
TILE_H`, `(logical_nt % Sk_chunk_t) * TILE_H`). Rewrote them in compute,
writer, and reader to eliminate the row→tile conversion runtime divides.
Compute additionally drops the intermediate row-units variable
(`global_n_within_ring_iter`) and works directly on
`global_nt_within_ring_iter` in tiles, saving a tile-height multiply per
iter.

**Mirror the "tile-aligned" assumption — and *only* there.** The tile-unit
rewrite is valid because the chunked-prefill TT_FATAL on `q_start_idx %
TILE_HEIGHT == 0` keeps logical_n tile-aligned. The kernel-side rewrite
also benefits non-chunked configs whose `seq_len` is tile-aligned (all the
existing models in `test_ring_joint_sdpa.py` are, including `mla_100k`,
`mla_128k`, `sdxl_1024_v2`, the WAN variants). If a future non-chunked
config introduces non-tile-aligned `logical_n`, `(logical_n - 1) /
TILE_H == logical_nt - 1` still holds (the formula's exact for any
`logical_n > 0`), but `logical_n - ring_id * kv_local_padded_N`'s row-tile
rewrite breaks. Audit there before adding such a config.

## Step #4 landmines for future PRs

**RT-arg layout is now chunked-conditional.** `compute_args` and
`reader_args` and `writer_args` have different slot counts depending on
`chunked_prefill_enabled`. `override_runtime_arguments` mirrors this with
an `if (chunked_prefill_enabled) { reader_args[10] = ... }` block. If
someone adds a new RT slot that's *always* present, place it before any
chunked-conditional slots (otherwise its index drifts).

**The layout-hint vs RT split is a soft contract.** The CT slot 11
(`logical_nt_layout_hint`) drives constexpr-folded mask-CB layout, but the
RT `logical_nt` drives runtime skip predicates. For chunked-prefill they
match because of the tile-aligned TT_FATAL. Changing the host TT_FATAL
(e.g., to allow non-tile-aligned `q_start_idx`) would silently desync the
two — mask CB allocated for "no partial" while the kernel needs partial
masking. Document any such change in this file.

**Program-cache hash has two sources of truth.** `attributes()` and
`compute_program_hash()` both contribute to the program-cache key (the op
overrides the hash function). Conditional drops of `logical_n` had to land
in both — missing one would let the cache key drift between them. Keep
them in sync if you add more conditional attributes.

## Next step

Proceed to step #5 (chain-bounds audit with variable logical_n) only if
hangs surface under stress. The current chunked test runs cleanly at
sp=4 with `logical_n=5120` (chunk 0 has 3 of 4 ring shards as all-padding)
— no hangs. Step #5 is the conditional safety-net.
