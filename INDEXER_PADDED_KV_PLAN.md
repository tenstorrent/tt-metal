# Padded-KV / Multiturn Support for `indexer_score` — Plan (rewritten against `main`)

Porting the idea behind PR #45729 ("KV pad-aware rotation for ring-joint SDPA") to the
`indexer_score` op. This rewrite is grounded in the **current `main` branch** as it actually exists
on disk — verified by reading every op/kernel file, the device op, the hash machinery, the tests, and
the PR itself (and by diffing the two unmerged `chunk_start` branches against main).

**Headline correction over the previous draft.** The previous plan assumed the cache-stable
runtime-scalar machinery for `chunk_start` "already exists here." **It does not exist on `main`.** It
exists only on the unmerged branch `skrstic/indexer-chunk-start-coord`. On `main`:

- `chunk_start_idx` is a **compile-time** arg — `chunk_t = args.chunk_start_idx / TILE_WIDTH` is baked
  into `common_ct` (`indexer_score_program_factory.cpp:123,272`) and read as CT arg slot 4
  (`indexer_score_common.hpp:23`, `num_dim_args = 8`).
- There is **no custom `compute_program_hash`** (`indexer_score_device_operation.hpp:25-27`
  explicitly documents this). The default reflection hash keys on `operation_attributes_t` — i.e. on
  `chunk_start_idx` **and** every config field. **So every distinct `chunk_start` already
  recompiles today.**
- `override_runtime_arguments` patches **only buffer addresses** (`indexer_score_program_factory.cpp:389-401`).
- No `device_index`, no mesh-coordinate logic, no runtime chunk_start. Neither
  `skrstic/indexer-chunk-start-coord` nor `skrstic/indexer-chunk-start-runtime` is merged
  (`git merge-base --is-ancestor … main` → false for both).

This single fact reshapes the design (see §4). The previous draft's correctness analysis and PR-transfer
analysis survive intact and are kept below.

---

## 1. What PR #45729 actually does (ring-joint SDPA) — verified

Op: `ttnn/cpp/ttnn/operations/transformer/sdpa/` (`RingJointSDPADeviceOperation`). New param
`std::optional<uint32_t> kv_actual_isl` = count of valid KV tokens already in the cache before this
chunk (`…_device_operation_types.hpp:34`, `has_kv_pad_rotation()` at `:106`). Five mechanisms, by how
well they transfer to `indexer_score`:

1. **Cache-stable runtime scalar (TRANSFERS — the heart of it). CONFIRMED.**
   `compute_program_hash` hashes only the *bool* `kv_pad_rotation_enabled` and a **zeroed**
   `cache_key_logical_n`, excluding the real `logical_n` and `kv_actual_isl` entirely
   (`ring_joint_sdpa_device_operation.cpp:583-584,608,614`). The changing scalars are re-patched as
   runtime args every dispatch and re-checked by `validate_runtime_patched_scalars` on **both** cache
   miss (`:335`) and hit (`:536`). → one compiled program serves every multiturn step.

2. **Compile-for-capacity, skip-at-runtime (TRANSFERS — the design pattern). CONFIRMED.**
   With rotation on, compile-time scalars/masks/mappings are **zeroed** (`program_factory.cpp:824-826,
   1116-1119`) and live values are written later by `apply_ring_joint_scalar_runtime_args`
   (`:469-621`, invoked on the cache-hit override path `:2293`). Kernels keep the grid fixed and skip
   work at runtime; dead cores are detected and skipped from patching, not recompiled.

3. **`RingIdSequencer` shared host/device header (CONCEPT ONLY).**
   `device/ring_id_sequencer.hpp` — a pure ring-iteration state machine, replayed on host
   (`build_ring_work_plan_impl`, `program_factory.cpp:186-239`) with a no-op sync callback to build
   `active_ring_iter_mask` / `single_valid_kv_chunk_mask` so host-planned skips line up exactly with
   the kernel's ring loop. **`indexer_score` has no ring loop**, so the *struct* doesn't port — only
   the idea "plan the skip on host, skip in kernel, keep sync alive across skipped iterations."

4. **`KVPadQMapping` pre/post-wrap Q remap + `compute_gather_valid_Ht` (DOES NOT TRANSFER). CONFIRMED.**
   These exist because ring's KV cache is **block-cyclic / slab-major** across devices and the current
   Q can straddle a chunk-group boundary (`build_kv_pad_q_mapping`, `program_factory.cpp:251-299`), and
   because a fused all-gather must be bounded to the valid prefix (`compute_gather_valid_Ht`,
   `:457-467`). `indexer_score`'s K is a **plain contiguous `[B,1,T,D]`** sequence with no gather and
   no wrap → none of this applies. It collapses to a single scalar "valid KV length."

5. **Kernel changes (PARTIAL transfer). CONFIRMED.** Reader/writer/compute gained zeroed compile-time
   mask args + live runtime args, and the **"sync-then-`continue`"** discipline: call
   `get_next_ring_id_and_sync()` (the semaphore sync) **first**, then `if (!(mask>>iter & 1)) continue;`
   (`ring_joint_reader.cpp:405-412`, `ring_joint_writer.cpp:533-539`, `ring_joint_sdpa.cpp:202-208`).
   The "skip the work but keep the sync alive" discipline transfers; the ring-iteration mask does not.

**Transferable core for `indexer_score`:** mechanism #1 (cache-stable hash + validate-on-both-paths) +
#2 (compile-zeroed / runtime-patched scalar bound) + the optional-param plumbing of #6/`kv_actual_isl`.
Everything ring/gather/slab-specific (#3, #4, the ring half of #5) collapses to one contiguous
`valid_k_tiles` scalar.

Tests (`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:2740-2767`): 4 cases —
`cold_start_one_slot`, `partial_old_pad`, `no_old_pad`, `multi_slab_wrap` — vs a
`q_abs_offset = kv_actual_isl` causal reference (`:1888-1902,2202`).

---

## 2. How this maps onto `indexer_score` (main)

`indexer_score(q[B,Hi,Sq,D], k[B,1,T,D], weights[B,Hi,Sq,1])` → `score[B,1,Sq,T]`. Causality:
key `t` visible to query `s` iff `t <= chunk_start + s` (`indexer_score_device_operation.hpp:56`). Today
`chunk_start_idx + Sq <= T` is enforced at validate (`…device_operation.cpp:98`).

**Correctness is already handled.** The dense schedule deals every q-row-group the full `[0, Tt)`
rectangle and masks the causal upper-triangle in-band. `stamp_mask_tile`
(`compute_indexer_score.cpp:191-203`) OVERWRITES every column at/past the diagonal — *including all pad
cols `>= Tt`* — with full `-inf` (mask tile index 1, `indexer_score_cb.hpp:29-30`); it is invoked from
`stamp_masked_suffix` (`compute_indexer_score.cpp:272-279`). The writer additionally clips output to
`valid_w = span.k_tiles()` so columns `>= Tt` are never written (`writer_indexer_score.cpp:24-32,61-64`).
**A device's max diagonal is `chunk_start_tiles + Sqt`, so every K tile beyond that — the entire padded
tail — is already `-inf`. No new masking is needed for correctness (G2).**

**The real problem is wasted work.** The op computes `relu(q·kᵀ)`, gate-mul, and untilize over the
full physical `Tt` (`= k_len_tiles`, baked from the K tensor's padded shape) every turn, even when only
`chunk_start_tiles + Sqt` tiles are non-`-inf`. In multiturn the K buffer is allocated once at capacity
`T_cap` and grows; today every turn is `O(T_cap)` — we want `O(valid)`. `total_units = groups *
units_in_group(KC, Tt)` (`indexer_score_program_factory.cpp:150-151`), and `units_in_group =
ceil(Tt/KC)` (`indexer_score_work_split.hpp:32-34`) — the pad tail is dealt as full work units.

---

## 3. The decisive consequence of "main is compile-time"

The PR's whole value is **mechanism #1 + #2: one cached program reused across every multiturn turn**.
That only buys anything if the program is *not* recompiled per turn. On `main`, `chunk_start` is
compile-time, so **the program is already recompiled every turn anyway.** That has a blunt corollary:

> **If you are recompiling per turn regardless, the cheapest "stop wasting work" fix is not a runtime
> skip at all — it is to shrink the work-split deal to the valid width at compile time.** The host
> already knows `chunk_start` (it's a compile-time attribute), so it can compute
> `valid_k_tiles = min(Tt, chunk_start_tiles + Sqt)` and build `total_units` / the grid from
> `valid_k_tiles` instead of `Tt`. Fewer units, smaller grid, zero kernel changes, zero idle cores.
> The "compile-for-capacity + skip-at-runtime" machinery (PR #2) earns **nothing** here, because we're
> not sharing a program across turns.

So the work splits cleanly into two scopes, and the choice of scope is the central decision (§6):

- **Scope A — true cache-stable multiturn (the real PR port).** First convert the per-turn-varying
  scalars (`chunk_start`, and the new `valid_k_tiles`) to **runtime, hash-excluded** args with a custom
  `compute_program_hash`, `validate_on_program_cache_hit`, and `override_runtime_arguments` patching —
  i.e. land the idea of branch `skrstic/indexer-chunk-start-coord` on main — **then** add the in-kernel
  unit-skip (PR #2). Result: **one program** serves all turns at fixed `T_cap`, each turn costs
  `O(valid)`. This is the only scope that delivers the original G1.
- **Scope B — perf-only, stays compile-time.** Leave `chunk_start` compile-time; just build
  `total_units`/grid from `valid_k_tiles` instead of `Tt`. Tiny change (host-side work-split only, no
  kernel skip, no hash work). Each turn still recompiles, but no longer wastes compute on the pad tail.
  Does **not** give a shared program — only removes wasted work within each turn's own program.

**Recommendation: Scope A**, because cache-stable reuse across turns is the entire point of porting the
PR (and recompiling a 14×10-grid program every turn is itself a real multiturn cost). Scope B is a
legitimate, near-free stepping stone if cross-turn program reuse is explicitly out of scope.

---

## 4. Goals

- **G1 (primary, the PR port):** With K allocated at fixed capacity `T_cap`, make a turn cost
  `O(valid_k_tiles)` matmul/gate/DMA instead of `O(T_cap)`, while keeping **one cached program** across
  all turns (hash unchanged across turns that differ only in `chunk_start` / valid length). *Requires
  Scope A.*
- **G2 (correctness, already free — must not regress):** Output columns in `[valid, T_cap)` stay `-inf`.
- **G3 (no new API if possible):** `valid_k_tiles = min(Tt, chunk_start_tiles + Sqt)` is derivable from
  `chunk_start` + hashed dims. Only add an explicit `kv_actual` param if a use case needs valid-length
  *decoupled* from `chunk_start + Sq` (see §6, Decision 1).

---

## 5. Recommended design (Scope A)

Two layers. Layer 1 is the prerequisite the previous plan wrongly assumed existed; Layer 2 is the skip.

### Layer 1 — make the per-turn scalars runtime + hash-excluded (the real prerequisite)

Mirror branch `skrstic/indexer-chunk-start-coord`'s structure (verified to compile/pass its qb test):

- **Move `chunk_t` from compile-time to a per-core compute runtime arg.** Drop `chunk_t` from
  `common_ct` (`indexer_score_program_factory.cpp:272`) → `common_ct = {Hi, Sqt, Tt, Dt, QC, KC, HB}`
  (`num_dim_args` 8→7 in `indexer_score_common.hpp:27`, and `chunk_start_tiles` becomes a function
  parameter threaded through `row_valid_prefix`/`stamp_masked_suffix` instead of a `constexpr`). Pass
  `chunk_t` as compute runtime slot 2 (after `flat`,`count`); read via `get_arg_val<uint32_t>(2)`.
- **Add the derived `valid_k_tiles` as compute runtime slot 3** (and to reader, see Layer 2):
  `valid_k_tiles = min(Tt, chunk_t + Sqt)`. Pure function of (now-excluded) `chunk_t` and hashed `Sqt`.
- **Add a custom `compute_program_hash`** that excludes `chunk_start_idx` (and `valid_k_tiles`, which is
  derived from it) but includes every config/dtype/spec field — replacing the default reflection hash
  (delete/override the "no custom hash" note at `indexer_score_device_operation.hpp:25-27`). *Be
  careful:* the default hash currently keys on `chunk_start_idx`; dropping it from the hash is exactly
  what makes turns share a program, and is the load-bearing change.
- **Add `validate_on_program_cache_hit`** → re-run the `chunk_start` alignment + `chunk_start+Sq<=T`
  window check (`…device_operation.cpp:93-103`) on every hit, since those scalars are no longer hashed.
- **Patch in `override_runtime_arguments`** (`indexer_score_program_factory.cpp:389-401`): recompute
  `chunk_t` and `valid_k_tiles` and patch the compute (and reader) slots, alongside the existing
  address patches. (Single-device only for v1; the coord branch's mesh-coordinate / `cluster_axis` /
  `device_index` derivation can be folded in later if multi-device per-rank `chunk_start` is needed —
  it is orthogonal to the skip.)

### Layer 2 — skip fully-padded units in-kernel (PR mechanism #2)

Keep compile-time `k_len_tiles = T_cap` (grid/units/hash stable across turns). For each `WorkUnitSpan`,
classify by `span.k_tile_start()` vs runtime `valid_k_tiles`:

- `k_tile_start >= valid_k_tiles` → **fully-padded unit:** reader skips K/Q/W payload DMA; compute
  emits a cheap full-`-inf` strip (mask tile 1 over all `k_tiles_per_unit`, then untilize — no
  matmul/gate); writer writes it (correctness). **But the mcast handshake must still complete** — see
  Risk 1.
- `k_tile_start < valid_k_tiles <= k_tile_start + KC` → **boundary unit:** normal compute; the existing
  `stamp_masked_suffix` already `-inf`s the tail columns (past the diagonal).
- else → **fully-valid unit:** unchanged.

All three groups are uniform-per-group within a single device's program (one `chunk_start` per program),
so the grid-aligned multicast precondition (`grid_aligned` requires `rem==0` and uniform deal,
`indexer_score_program_factory.cpp:62-64`) is preserved.

---

## 6. Exact steps (main line refs)

0. **Decide scope (§3).** If Scope B: do only step 1's host arithmetic against `valid_k_tiles` in the
   work split (build `total_units` from `valid_k_tiles`, keep everything else compile-time) and stop;
   skip steps 2–6. The rest of this list is Scope A.

1. **`device/indexer_score_program_factory.cpp`**
   - Add `valid_k_tiles = std::min(Tt, chunk_t + Sqt)` near `chunk_t` (`:123`).
   - Remove `chunk_t` from `common_ct` (`:272`); push `chunk_t` and `valid_k_tiles` as compute runtime
     slots 2,3 in `SetRuntimeArgs(..., compute_id, ..., {flat, count, chunk_t, valid_k_tiles})` (`:375`).
   - Add `valid_k_tiles` (and `chunk_t` if the reader needs it for the skip) to the reader runtime args,
     after `flat`,`count` (slots 3,4 → shift the mcast 8-tuples; update `read_mcast_dir` bases in
     `reader_indexer_score.cpp:237-238`).
   - In `override_runtime_arguments` (`:389-401`): recompute and patch `chunk_t` + `valid_k_tiles` for
     compute/reader, alongside the q/k/w/out address patches. Add `rt_arg::*` slot constants for them.

2. **`device/kernels/indexer_score_common.hpp`** — `num_dim_args` 8→7; `chunk_start_tiles` is no longer
   CT arg 4. Make `row_valid_prefix` take `chunk_start_tiles` as a param (mirror coord branch). Document
   the new runtime slots.

3. **`device/kernels/indexer_score_work_split.hpp`** — add a pure helper
   `unit_is_fully_padded(k_tile_start, valid_k_tiles) { return k_tile_start >= valid_k_tiles; }`
   (host + device, alongside `valid_prefix_tiles`/`units_in_group`).

4. **`device/kernels/compute_indexer_score.cpp`**
   - Read `chunk_start_tiles = get_arg_val<uint32_t>(2)` and `valid_k_tiles = get_arg_val<uint32_t>(3)`
     (after `flat_start`/`flat_count` at slots 0,1, `:282-283`).
   - In the per-unit loop (`:308`): if `unit_is_fully_padded(span.k_tile_start(), valid_k_tiles)`, still
     `k.wait_front`/`pop_front` and `span.advance()` (keep lockstep), emit a full-`-inf` strip via
     `stamp_mask_tile` over all `k_tiles_per_unit` + the existing `untilize<...>` (`:339`), and skip
     `matmul_phase`/`mul_phase`. Otherwise unchanged (boundary handled by `stamp_masked_suffix`).

5. **`device/kernels/reader_indexer_score.cpp`** — read `valid_k_tiles`; for fully-padded units skip
   `read_k_chunk`/`read_q_rows`/`read_w_group` payloads (`:258-278`). **Sync caveat (Risk 1):** the
   per-unit/per-row mcast handshake (`mcast_send` blocks on `send_sem.wait(ndst)` at `:69`; `mcast_recv`
   blocks on `recv.wait(1)` at `:92`) must still complete on skipped cores, or peers hang. Either keep
   the handshake while skipping only the payload, or gate the skip on `mcast off` for v1 (see Risk 1).

6. **`device/kernels/writer_indexer_score.cpp`** — likely unchanged: it already writes whatever strip
   compute produced (`valid_w = span.k_tiles()`, `:61`); for a fully-padded unit that's a full `-inf`
   strip. Confirm it doesn't assume `valid_w > 0`.

7. **Device op (`indexer_score_device_operation.{hpp,cpp,_types.hpp}`)** — add custom
   `compute_program_hash` excluding `chunk_start_idx`; add `validate_on_program_cache_hit` re-running the
   `chunk_start` alignment/window checks (`:93-103`). For the explicit-param variant (Decision 1), add
   `std::optional<uint32_t> kv_actual` to `operation_attributes_t` (default `nullopt`, hash-excluded),
   used in place of `chunk_t + Sqt` when present.

8. **API (`indexer_score_nanobind.cpp`/`.hpp`, `indexer_score.hpp`)** — none for G3 (derived). For the
   explicit-param variant, add `std::optional<uint32_t> kv_actual = nb::none()` (`nanobind.cpp:52`
   sits next to `chunk_start_idx`), threaded through `invoke` (`device_operation.cpp:207-244`).

9. **Tests** — `tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py` (reuse
   `indexer_score_ref`, `:47-56`): allocate K at `T_cap > chunk_start+Sq`; assert (a) numerics match the
   unpadded reference on `[0, valid)`, (b) output `[valid, T_cap)` is `-inf`, (c) **program hash
   identical across two turns** with different `chunk_start` but same `T_cap` (the Scope-A cacheability
   guarantee — this is the test that proves G1, and the one that would *fail today*). Use a production
   config (`glx_config`: QC=64/2t, KC=512(h8)/256(h16) = 16t/8t, HB=0).

---

## 7. Decisions for you & risks

- **Decision 0 — Scope A vs B (§3). THE key decision.** A = true cache-stable multiturn (requires the
  runtime-scalar conversion in Layer 1; delivers G1). B = perf-only, stays compile-time, near-free, no
  shared program. Recommend A; B is a valid stepping stone if cross-turn reuse is out of scope.
- **Decision 0b — build on `skrstic/indexer-chunk-start-coord` or reimplement on main?** Layer 1 *is*
  essentially that branch's work (verified passing its qb test). Cherry-picking/rebasing it onto main as
  the foundation, then adding Layer 2, is lower-risk than re-deriving the hash/override/validate
  plumbing from scratch. (The branch also carries multi-device mesh-coordinate `chunk_start`, which is
  orthogonal but harmless.) Worth confirming you want the rewrite to *land that branch first*.
- **Decision 1 — derived vs explicit valid length.** G3 assumes `valid = chunk_start + Sq` (holds for
  standard causal multiturn: queries are the newest tokens). If valid KV length must be *decoupled* from
  query position (history padding, queries not at the tail), port an explicit `kv_actual` param 1:1 from
  the PR's `kv_actual_isl`. **Which is it?**
- **Risk 1 — multicast + skip deadlock (the biggest real risk).** Skipping a core's reads while peers
  wait on its mcast semaphore hangs the device — the sender blocks at `reader_indexer_score.cpp:69`, a
  receiver at `:92`. Mitigation (matches the PR's "keep sync alive across skipped iterations"): keep the
  handshake, skip only the payload DMA. **Note:** the production config is grid-aligned mcast
  (`glx_config`, QC2/KC16/hg0), so the non-mcast fallback gives **zero** speedup in production — the
  handshake-preserving version is the only one that matters there. Restricting v1 to the non-mcast path
  is a correctness-first stepping stone only.
- **Risk 2 — idle cores / load imbalance.** Cores assigned to fully-padded units go idle (same as the
  PR). Acceptable: the win is skipping the matmul, not perfect balance. A host re-deal of only valid
  units would shrink the grid → that's Scope B's recompile, rejected under Scope A (kills the shared
  program).
- **Non-risk — tail correctness.** Already guaranteed by the existing causal mask
  (`compute_indexer_score.cpp:191-203`); the skip just substitutes a cheaper way to produce the same
  `-inf`.

---

## 8. One-paragraph summary

The op-specific hard parts of PR #45729 (block-cyclic cache, `KVPadQMapping` wrap, ring sequencer,
gather bounding) **do not apply** to `indexer_score`'s plain contiguous K, and the part that would be
hard here — masking the pad tail — is **already done** by the existing causal mask. What's left is the
PR's reusable spine: carry the valid length as a hash-excluded runtime scalar and skip padded work
in-kernel while keeping multicast sync intact. **The correction this rewrite makes:** that spine is
*not* already present on `main` — `chunk_start` is compile-time and recompiles every turn, so the skip
pattern only pays off **after** converting the per-turn scalars (`chunk_start`, `valid_k_tiles`) to
runtime + hash-excluded (Layer 1, ≈ landing branch `skrstic/indexer-chunk-start-coord`). Net new
surface for the full port: that runtime-scalar conversion + custom hash + validate-on-hit, one derived
runtime scalar `valid_k_tiles`, a unit-skip branch in reader+compute that preserves the mcast handshake,
and a test asserting cache-stability across turns + tail-`-inf`.
