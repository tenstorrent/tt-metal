# Investigation — `dual_4x16` strict-validation deadlock + inter-mesh control-plane audit

Owner/handoff-from: debugging session, 2026-07-17
Branch: `riddy21/fatal-intermesh-routing-validation`
Related: [`intermesh_port_pairing_redesign.md`](./intermesh_port_pairing_redesign.md), [`intermesh_contention_findings_handoff.md`](./intermesh_contention_findings_handoff.md), `TODO(#50162)` (single/multi-host allocator path unification)

---

## TL;DR (corrected)

The full CPU-only fabric suite hangs in `bh-subtorus` on `dual_4x16_blitz_test`. The boundary
**M0↔M1** (`count: 4 policy: STRICT`, **not** `assign_z`) is placed by the allocator with 4
channels, and **all 4 channels bind on *both* sides** (instrumented: total `applied = 4` for
`M0->M1` and `4` for `M1->M0`). So there is **no physical/bind shortfall and no allocator
asymmetry** here.

The failure is a **resolved-count bug**: `num_resolved_between` returns a **rank-divergent,
unit-inconsistent** value (some ranks compute 3, others 4 for the same boundary), because the
per-channel exit/peer pairs contain duplicates that the cross-host merge dedups **inconsistently**.
The strict per-mesh-pair validation then throws on the ranks that compute 3 but not on those that
compute 4 → a `TT_THROW` on a **subset of ranks** → the surviving ranks block at the next MPI
collective → the whole job **deadlocks** (and, with no per-line timeout, hangs the suite).

> Correction to the first draft of this doc: the root cause is **not** `place_link` choosing an
> unbindable dst port. Summing `[apply-dbg] applied` across *all* ranks shows every channel binds on
> both meshes. The earlier "M1 skips 2 as no-PSD-cable" reading was wrong — that `skip_nocable` is
> **benign per-rank pruning** (each rank only applies connections for the exit chips it owns and
> skips the rest). The real defect is the counting/merge (Finding A below).

This is **not** the `assign_z` Z-only change (M0↔M1 is unmarked), and it is exposed — not caused —
by this branch's new per-mesh-pair strict validation (`main` has no such check).

---

## Reproduction

**Test case that reproduces it:** `dual_4x16_blitz_test` — MGD
`tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x16_blitz_test.textproto` (two 4×16 meshes;
the M0↔M1 connection is `channels { count: 4 policy: STRICT }`). Any `ControlPlaneFixture` test that
builds the control plane over it trips the bug during init; `TestGalaxyCornerPins` is the smallest.

**Exact steps** (this dev box; neutralize SLURM so PRTE forks ranks locally):
```bash
cd /data/rsong/tt-metal
unset $(env | grep -oE '^SLURM[^=]*')
unset PRTE_MCA_plm_slurm_args OMPI_MCA_plm_slurm_args HYDRA_BOOTSTRAP I_MPI_HYDRA_BOOTSTRAP
export TT_METAL_HOME="$PWD" PATH="$PWD/python_env/bin:$PATH"
rm -rf generated/ttrun generated/fabric

# TT_INTERMESH_DEBUG=1 prints the [intermesh-dbg]/[apply-dbg] tallies (needs the apply-dbg build).
# It WILL deadlock after the throw, so cap it with `timeout` and clean strays afterwards.
timeout 200 env TT_METAL_SLOW_DISPATCH_MODE=1 TT_INTERMESH_DEBUG=1 tt-run \
  --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x16_blitz_test.textproto \
  --mock-cluster-rank-binding tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml \
  --mpi-args "--allow-run-as-root --oversubscribe" --force-rediscovery \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="ControlPlaneFixture.TestGalaxyCornerPins"
pkill -9 -x prterun; pkill -9 -x prted; pkill -9 -f 'fabric[_]unit_tests'; pkill -9 -f 'mpirun[-]ulfm'
```

**Expected output** (the signature of the bug):
```
[intermesh-dbg] M0<->M1: placed_channels=4 budget=4          # allocator placed all 4
# summing [apply-dbg] applied across ALL ranks:
#   M0->M1 applied = 4   and   M1->M0 applied = 4            # every channel binds on BOTH sides
TT_THROW @ control_plane.cpp:3153: Requested 4 channels between 1 and 0, but only 3 were resolved.
# ...fires on the M1-owning ranks (e.g. rank22/23) only -> the rest block in the next collective -> HANG
```
The tell-tale: **both sides bind 4** (`applied=4` each direction) yet validation reports **3** on a
subset of ranks — i.e. a *counting* divergence, not a bind shortfall. (Runs on `main` don't throw:
`main` has no per-mesh-pair validation, so the miscount stays silent.)

> To see the numbers without a deadlock, run against a **relaxed** copy of the MGD (change the M0↔M1
> `policy: STRICT` → `RELAXED`): validation then only *warns* (`... but only 3 were resolved ...`) and
> the job completes, so you can inspect `[apply-dbg]` cleanly.

---

## Finding A (root cause here + HIGH severity generally) — rank-divergent `resolved` from per-channel duplicate pairs + inconsistent merge dedup

- The apply path pushes **one pair per channel, no dedup**:
  `intermesh_exit_peer_fabric_node_id_pairs_[a][b].emplace_back(my_fn, info.peer_fn)`
  (`control_plane.cpp:~3820-3821`; single-host path likewise ~4049-4052). A cable is *N* channels,
  so an *N*-channel cable between the same `(exit_chip, peer_chip)` pushes the **identical** pair
  *N* times.
- The cross-host merge **dedups, but only remote contributions**:
  - `merge_pair_vectors` (`~2851`): `if (std::find(into, p) == end) into.push_back(p)` — drops
    duplicates coming from a remote rank.
  - `merge_from_serialized` (`~2896-2901`): when the local slot is empty it `std::move`s the first
    remote root's vector in **with its duplicates**; otherwise it dedups.
- `num_resolved_between` (`~3097`) returns the raw `dst_it->second.size()`.
- Strict validation compares that size to a **channel sum** (`requested += std::get<2>(port_spec)`,
  `~3104-3117`); relaxed compares to `requested_channels` (`~3126-3155`).

So `resolved` is neither a clean channel count nor a clean unique-cable count — it's local
channels-with-dups plus remote unique-pairs, and its exact value depends on **which rank** and on
**merge/emptiness ordering**. Meanwhile `requested` is a channel count. For any boundary whose
cables carry >1 channel and whose two meshes are split across hosts, different ranks get different
`resolved`, and the strict check (`==`) or the relaxed under-resolve check (`>`) can fire on a
**subset** of ranks → MPI deadlock. `dual_4x16` M0↔M1 (2 cables × 2 channels, meshes split across
ranks) is exactly this shape: some ranks see 3, some 4.

**Fix direction:** make `resolved` and `requested` the same unit, computed identically on every
rank. Options: (a) count **channels** consistently — never dedup channel entries (key the vector by
channel, or store counts), or (b) count **unique cables/exit-peer pairs** consistently — dedup
uniformly (local + remote + the moved root) and express `requested` in the same unit. Whatever the
choice, the merge must be order-independent and every rank must arrive at the same number.

### Instrumented confirmation (2026-07-17 re-run, `TT_INTERMESH_DEBUG=1`)

Added three gated logs — `[place-dbg]` (rank-0 allocator placement), an enhanced `[apply-dbg]` that
prints `BIND my_fn <-> peer_fn (hash)` and `SKIP_NOCABLE … this rank's PSD exit chips are …`, and a
rank-0 `[resolved-dbg]` dump of the final merged pair set — then re-ran the reproducer. Result:

```
[place-dbg] src (M0,D0) <-> dst (M1,D15) nch=2   hash0=…860       # link 1 (2 channels)
[place-dbg] src (M0,D1) <-> dst (M1,D14) nch=2   hash0=…894       # link 2 (2 channels)

rank0  BIND M0->M1: (M0,D0)<->(M1,D15) x2  ,  (M0,D1)<->(M1,D14) x2      # all 4 bind
rank23 BIND M1->M0: (M1,D15)<->(M0,D0) x2                                 # link 1 binds (D15 lives on rank23)
rank22 BIND M1->M0: (M1,D14)<->(M0,D1) x2                                 # link 2 binds (D14 lives on rank22)
rank22 SKIP_NOCABLE (M1,D15): this rank's PSD exit chips are D30 D14      # benign: D15 is rank23's chip
rank23 SKIP_NOCABLE (M1,D14): this rank's PSD exit chips are D31 D15      # benign: D14 is rank22's chip

[resolved-dbg] rank0 M1->M0: 4 pairs: (14->1)(14->1)(15->0)(15->0)       # rank0 computes 4 -> PASSES
TT_THROW on rank22 AND rank23 only: "… only 3 were resolved"             # they compute 3 -> THROW -> hang
```

This is conclusive: **every one of the 4 channels binds to its true PSD peer on both meshes** (no
allocator/apply asymmetry, the `SKIP_NOCABLE`s are the other host's chips). The `(14->1)(14->1)
(15->0)(15->0)` pattern is the **per-channel duplicate**; the throw fires only on the two M1-owning
ranks because their local-vector-plus-deduped-remote arithmetic lands on **3** while rank 0 lands on
**4** — the order-dependent count divergence, exactly as Finding A predicts.

The mechanics of "3": an M1 rank pushes its own link's 2 channels as the identical pair twice
(`(14->1)(14->1)`), then `merge_pair_vectors` folds in rank 0's remote vector but **dedups** rank 0's
copy of `(14->1)` and keeps only one new `(15->0)` → `{(14->1),(14->1),(15->0)}` = **3**. Rank 0,
which owned both links locally, keeps all four of its own un-deduped entries → **4**.

---

## Other audit findings (independently verified)

### Finding B — MED/HIGH — strict per-src-node budget only reads the canonical `min→max` slot → strict MGDs authored `high→low` place nothing → guaranteed fatal
`control_plane.cpp:~3565-3575` looks up `requested_intermesh_ports[boundary.first][boundary.second]`
with `boundary.first = min(mesh)`. But `requested_intermesh_ports` is stored in the MGD's authored
direction and is **not** canonicalized/bidirectional (`mesh_graph.cpp:~286`). The relaxed budget
right above (`~3563`) already guards this with `max(requested_count(a,b), requested_count(b,a))`.
So a device-level (strict) connection authored `M3→M1` leaves `ports[1][3]` empty → `budget_per_src_node`
gets nothing → `within_budget` false for every link → 0 placed → strict `TT_FATAL` (resolved 0).
Also the budget would be keyed on the low mesh while the exit chips are on the high mesh.
Verified: the validation itself (`~3102`) is order-agnostic, so only the allocator budget is wrong.

### Finding C — MED — golden all-mesh aggregation **overwrites** instead of concatenating when a mesh spans multiple host ranks
`tests/tt_metal/tt_fabric/common/utils.cpp:~588-591`: `merged_intermesh[key] = it->second;`
(assignment). The comment assumes "each rank owns a distinct local mesh, so keys never collide,"
which is false when one mesh spans several ranks — each writes the same `"M{a}->M{b}"` key with only
its own chips' ports, and the last file read wins. Result: the combined golden under-captures the
assignment (dropped exit chips) → spurious mismatch, or an incomplete golden if regoldened. Inner
sequences should be **concatenated** (and sorted), not assigned. (Introduced by my golden-aggregation
change — worth fixing there.)

### Finding D — MED — additional subset-of-ranks throw sites that deadlock MPI
1. `pair_logical_intermesh_ports` runs **only on rank 0** (`~3729`); its `fatal_assign_z_failure`
   `TT_FATAL(false, …)` (`~3476`, called `~3618`) fires on rank 0 alone, while all other ranks are
   blocked in `forward_intermesh_connections_from_controller`'s `recv(Rank{0})` (`~3292-3304`) →
   deadlock. Same failure class as the `dual_4x16` hang, different trigger (an assign_z Z-lane
   conflict). The relaxed Phase-1a "fatal only on zero Z" still fires here.
2. `get_requested_exit_nodes` `TT_FATAL(num_physical_channels_found >= num_channels_requested, …)`
   (`~3183`) runs per-rank, filtered to local exit nodes — one host short on physical channels
   aborts alone while peers proceed → deadlock.

General mitigation: intermesh-init fatals that can fire on a subset of ranks should be turned into a
collective decision (all-reduce the failure, then all ranks throw together) so failures surface as a
clean error instead of a hang. Also add a per-line `timeout` to the plain `bh-subtorus` suite
entries (they lack `TT_METAL_OPERATION_TIMEOUT_SECONDS`).

### Finding E — LOW/MED — `sort_intermesh_exit_peer_fabric_node_id_pairs` sorts on `.first` only → nondeterministic peer order
`control_plane.cpp:~2358-2366` sorts pairs by `a.first < b.first` (ignores `.second`, non-stable).
Among entries sharing an exit chip (a chip with multiple peers) the order is unspecified and the
pre-sort order came from `unordered_map` merge iteration. Downstream consumers
(`get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes`, e.g. blitz-decode pipeline gen) can
then get host-/run-dependent peer assignments — the exact "became host-dependent" symptom the golden
guards against. Fix: sort on the full `(first, second)` pair.

### Audited clean (no issue found)
Round-robin fairness / wrap-around / Phase-1a fatal-only-on-genuine-failure / assign_z Z-only
exclusion / `dir_owner` one-neighbor-per-`(node,dir)` — internally consistent. Allocator outputs are
deterministic (`std::map`/`std::set` throughout; boundaries in a `std::set`) despite
`PortDescriptorTable` being unordered. Golden rank-file indexing is consistent (`rank+1` writer vs
`1..world_size` reader; rank 0 included).

---

## Relationship to this PR's changes
- **Not** the `assign_z` Z-only change (unmarked boundary).
- Finding A and Finding D.1 interact with this branch's **new strict per-mesh-pair validation**
  (`e2ca525`) — the validation is correct, but it turns latent count/parity bugs into hard,
  subset-of-ranks failures that deadlock. `main` lacked the check so these stayed silent.
- Finding C is in **my** golden-aggregation change (`59c5f54`) and should be fixed there.
- Finding B and Finding E predate the allocator rewrite conceptually but live in the current code.

## How the findings relate (one subsystem, one failure family)

All of this lives in **inter-mesh control-plane connectivity** (`generate_intermesh_connectivity` →
gather → rank-0 `pair_logical_intermesh_ports` → broadcast → per-rank
`convert_port_descriptors_to_intermesh_connections` → merge → `validate_requested_intermesh_connections`),
and the pieces are sequential, not independent:

- The **contention handoff** (`intermesh_contention_findings_handoff.md`, SC16 62↔63 strands zero)
  and the **redesign** (`intermesh_port_pairing_redesign.md`) are the *same* issue — greedy
  per-neighbor port exhaustion — and are **fixed** by the implemented round-robin both-sides
  allocator. Confirmed still-working here: on `dual_4x16` the allocator places the M0↔M1 boundary
  fully (`placed_channels=4`, no strand) and both sides bind. (The SC16 repro itself is now too slow
  to finish in a Debug build — topology mapping alone ran ~3.5 min before the rank-binding phase was
  timed out — but the allocator behavior it exercised is what `dual_4x16` also exercises.)
- Fixing contention **moved allocation onto rank 0** and this branch **added strict per-mesh-pair
  validation**. That combination is what *exposes* the latent defects here: **Finding A** (count
  divergence) and **Finding D.1** (rank-0-only allocator fatal) both become **subset-of-ranks
  throws** that deadlock MPI. They were silent on `main`.
- **Findings B/E** (strict budget canonicalization, non-total peer sort) and **Finding C** (golden
  aggregation overwrite) predate or sit beside the rewrite but live in the same code and matter for
  strict MGDs, determinism, and goldens.
- The handoff's **issue #2** (`should_assign_z_direction` under-marking the SC16 wrap) is the one
  genuinely **separate, still-open** item — a classifier fix, orthogonal to everything above. It is
  not required to unblock `dual_4x16`.

**Unifying theme of the deadlock-class bugs (A, D.1, D.2):** a per-rank decision or count that is
allowed to *differ across ranks* turns into a `TT_THROW`/`TT_FATAL` on only some ranks, and the
survivors hang at the next collective. The durable fix is not just each local bug but the **pattern**:
init-time inter-mesh failures and counts must be **collective** (all ranks agree, then all act).

---

## Full fix plan

Ordered by priority. P0 unblocks CI; P1 fixes the real correctness bugs; P2 hardens the class so a
future regression fails fast instead of hanging.

### P0 — stop the hang (mechanical, no allocator change)
1. **Keep `dual_4x16_blitz_test` out of the blocking suite path until Finding A lands.** It is
   already `--grep-exclude`-d; leave it excluded (or flip its M0↔M1 `policy: STRICT` → `RELAXED` so a
   miscount only *warns*). This lets the rest of `bh-subtorus` (incl. the real SC16/#49629 rings)
   complete.
2. **Add `TT_METAL_OPERATION_TIMEOUT_SECONDS` to the plain `bh-subtorus` entries** (they lack it,
   unlike `bh-subtorus-sc20`). A future subset-of-ranks throw then fails fast instead of hanging the
   whole suite (Finding D mitigation).

### P1 — fix the correctness bugs
3. **Finding A — make `resolved` rank-consistent and in channel units (the real fix).**
   - Compute the validation count from the **broadcast `intermesh_connections`** (identical on every
     rank), counting **unique `connection_hash` per unordered mesh pair** — inherently
     order-independent and already in *channels*, matching the MGD `count`. Concretely, in
     `validate_requested_intermesh_connections`, replace `num_resolved_between` (which reads
     `intermesh_exit_peer_fabric_node_id_pairs_.size()`) with a helper that tallies distinct
     `std::get<2>(conn)` (the hash) over `intermesh_connections` grouped by
     `{min(mesh),max(mesh)}`.
   - Separately, make `intermesh_exit_peer_fabric_node_id_pairs_` **consistent across ranks** for its
     downstream consumers: dedup uniformly (local + moved-root remote + subsequent remotes) so every
     rank ends with the same set. Do this in `merge_from_serialized` (dedup the moved-in first root
     too) and/or dedup once after `collect_and_merge_…`. This removes the per-channel-duplicate
     nondeterminism regardless of which downstream reads it.
   - Regression guard: the new `[resolved-dbg]` dump plus an all-gather assertion (see test #2 below).
4. **Finding B — canonicalize the strict per-src-node budget lookup.** In
   `pair_logical_intermesh_ports`, look up `requested_intermesh_ports` in **both** authored
   directions (mirror the relaxed branch's `max(requested_count(a,b), requested_count(b,a))`), and
   key `budget_per_src_node` by the actual exit-chip mesh, not `boundary.first`. Without this a
   strict MGD authored high→low resolves 0 and hard-fatals.
5. **Finding C — concatenate, don't overwrite, in golden aggregation.** In
   `tests/tt_metal/tt_fabric/common/utils.cpp` (~590), when a mesh key already exists, **merge the
   inner sequences** (append + sort by port) instead of `merged_intermesh[key] = it->second`. Fixes
   under-captured goldens when a mesh spans multiple ranks. (This is in the golden-aggregation change
   `59c5f54`.)

### P2 — harden the failure class (make init failures collective)
6. **Finding D.1 — don't fatal only on rank 0 inside the allocator.** `pair_logical_intermesh_ports`
   runs on rank 0 only; a `fatal_assign_z_failure`/`TT_FATAL` there strands every other rank in
   `forward_intermesh_connections_from_controller`'s `recv(Rank{0})`. Have rank 0 broadcast a
   success/failure status (and the message) *before* the blocking send, so all ranks throw together
   with the same diagnostic. Equivalent: compute the allocation result, all-reduce an error flag,
   then throw collectively.
7. **Finding D.2 — make `get_requested_exit_nodes`' per-rank `TT_FATAL` collective** (or defer it to
   the collective validation), so one host short on physical channels doesn't abort alone.
8. **Finding E — sort exit-peer pairs on the full `(first, second)`** (and make it stable), in
   `sort_intermesh_exit_peer_fabric_node_id_pairs`. Removes host-/run-dependent peer ordering that
   downstream pipeline generation can otherwise inherit.
9. **General:** audit remaining intermesh-init `TT_THROW`/`TT_FATAL` sites for "can fire on a subset
   of ranks" and route them through a collective decision.

### Separate track (not blocking `dual_4x16`)
10. **Handoff issue #2 — `should_assign_z_direction` under-marks the SC16 wrap.** Investigate how the
    derived classifier decides (the MGD has no explicit `assign_z` for the wrap) so the QSFP Z-lane
    wrap is marked Z and frees NESW for the ring hop — the *correct* placement, versus the allocator
    merely avoiding the strand. Track under the Z-contention epic (#49628).

### Rollout / validation
- Land P1.3 (Finding A) first with the guard test (#2 below); confirm `dual_4x16` strict resolves
  4/4 on **every** rank and no longer deadlocks. Then re-enable the reproducer in the suite.
- Diff SC16/SC20 ring goldens before/after — they should be unchanged (rings never tripped A/B).
- Remove the `TT_INTERMESH_DEBUG` diagnostics (`[intermesh-dbg]`, `[apply-dbg]`, `[place-dbg]`,
  `[resolved-dbg]`) as the final step, tracked by `#50162`.

---

## Scope confirmation (ring topologies are clean)
SC16 superpod (128 boundaries) and SC20 supercluster (160 boundaries), instrumented: **0
`skip_nocable`**, and SC20's 144 relaxed "drops" are legitimate budget/contention, not asymmetry.
The count bug (Finding A) only bites boundaries with multi-channel cables split across hosts *and*
strict/relaxed-nonrelaxed policy; the pod rings in those runs did not trip it (their strict count=2
test resolves 2/boundary consistently).

## Test cases to guard this

1. **Existing reproducer (already in-repo):** `dual_4x16_blitz_test` run through any
   `ControlPlaneFixture` control-plane build. It *does* reproduce Finding A, but as a **hard MPI
   deadlock** — unsuitable for CI as-is (it hangs rather than failing). It is currently
   `--grep-exclude`-d from the suite. Once Finding A is fixed, re-enable it (and give the
   `bh-subtorus` suite entries a `TT_METAL_OPERATION_TIMEOUT_SECONDS` so any future regression fails
   fast instead of hanging — see Finding D).

2. **Cross-rank consistency guard — IMPLEMENTED as
   `expect_intermesh_resolved_pairs_consistent_across_ranks(...)`
   (`tests/tt_metal/tt_fabric/fabric_router/test_multi_host.cpp`), now wired into both
   `MultiHost.SC20RelaxedChannelRules` and `MultiHost.SC20Strict2Connections`.** It runs on **every**
   rank (before the rank-0-only tally), and for each requested boundary (walked in a sorted,
   rank-independent order) it computes
   `resolved_r = get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src, dst).size()` plus an
   order-sensitive FNV checksum of the pair list, then `all_reduce` MIN/MAX across ranks and asserts:
   - **cross-rank count agreement:** `min(resolved_r) == max(resolved_r)` per boundary — the direct
     Finding-A check (today `dual_4x16` M0↔M1 is 3 on some ranks, 4 on others; SC20 rings agree, so
     these two tests **pass** and act as regression guards);
   - **cross-rank content/order agreement:** the checksums match — catches a same-count-different-set
     divergence and nondeterministic peer order (Finding A residual + Finding E across ranks);
   - **local determinism:** each boundary's pair list `std::is_sorted` on the **full** `(exit, peer)`
     pair (Finding E — a `.first`-only sort leaves peer order unspecified).

   Because SC20 is clean, these assertions pass there; the *same* helper fails on `dual_4x16` (min=3,
   max=4), which is why a **relaxed `dual_4x16` MGD variant** running the same helper is the ideal
   dedicated reproducer — it hits the failing boundary without the STRICT-policy deadlock (relaxed only
   *warns* on under-resolve) and without the ring's single-channel cables masking the multi-channel
   duplicate. (Not yet added; blocked only on authoring the relaxed MGD copy.)

3. **Cheapest signal:** promote the silent `log_debug` "no PSD inter-mesh cable" (`~3777`) and add a
   post-init assert that `sum_over_ranks(applied) == resolved` for each boundary — turns a silent
   miscount into an immediate, localized failure.

## Full CPU-only suite survey (2026-07-17) — every real failure traces to Finding A

Ran all 16 groups (per-group timeout + stray-cleanup; `dual_4x16` excluded). Result classes:

| bucket | groups / tests | verdict |
|---|---|---|
| **Clean pass** | first 9 groups (unit…bh-dual-galaxy), `bh-subtorus`, `bh-subtorus-sc16` (incl. **#49629** SC16 64-stage), `bh-ring-stress`, `bh-misc` | ✅ real pass |
| **Finding A failures** | `bh-pod-pipeline`: `TestLlama8b1x2PodControlPlaneInit` + `2x1` across all 4 SC4 mocks | strict `TT_THROW` M0↔M1 with **varying** resolved: `8→6`, `8→5`, `16→12/10/9`. No `assign_z` in the MGD → not the Z-only change; the varying count is the Finding-A rank-divergence signature. Same root cause as `dual_4x16`. |
| **Resource/infra** | `bh-sp4-glx`: `Test32x4QuadGalaxyFabric1DSanity` (SIGBUS), `TestTriplePod32x4Quad…` (SIGKILL/OOM) | 128-rank/triple-pod jobs crash under memory pressure (2× Bus error, 1× Killed). 12 other tests pass. Needs isolation to confirm infra vs real. |
| **Runner budget timeout (not a hang)** | `bh-subtorus-sc20`, `bh-blitz-decode` | all tests that ran PASSED (520 / 540 ×"PASSED 2 tests", 0 FAILED); groups just exceeded the 1500s per-group cap I set. Raise the cap; not a code issue. |

**Bottom line:** across the whole suite, the *only* correctness failures are the strict-validation
shortfalls, and they all trace to **Finding A** (rank-divergent resolved count) — `dual_4x16` (deadlock)
and the Llama8b pods (clean fail). Fixing Finding A should clear both. The 128-rank crashes are a
separate resource concern; the two "timeouts" are false alarms from my runner's budget.

> Confirmation still owed for the Llama8b pods: an `[apply-dbg]` run to verify all 8/16 channels bind
> (pure count bug) vs a genuine under-resolution. The varying-count signature strongly indicates the
> former, matching `dual_4x16`.

## Diagnostics left in place
`TT_INTERMESH_DEBUG`-gated, all uncommitted; keep while fixing, remove before final commit (`#50162`):
- `[intermesh-dbg]` — per-boundary allocator summary + Z-lane owners, in `pair_logical_intermesh_ports`.
- `[place-dbg]` — per-placed-link `src <-> dst` (dirs, nch, hash), in `place_link` (added 2026-07-17).
- `[apply-dbg]` — per-rank per-boundary `applied/skip_nocable/skip_hash` **plus** per-connection
  `BIND my_fn <-> peer_fn (hash)` and `SKIP_NOCABLE … this rank's PSD exit chips are …`, in
  `convert_port_descriptors_to_intermesh_connections` (the BIND/SKIP detail added 2026-07-17).
- `[resolved-dbg]` — rank-0 dump of the final merged `intermesh_exit_peer_fabric_node_id_pairs_`
  (unique exit→peer chip pairs per boundary), after the cross-host merge in
  `generate_intermesh_connectivity` (added 2026-07-17). This is the log that shows rank 0 computing 4
  while the M1 ranks throw 3.
