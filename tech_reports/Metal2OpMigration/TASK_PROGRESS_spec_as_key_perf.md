# TASK PROGRESS — Spec-as-key host-dispatch perf (remeasure + improve)

**Owner agent goal:** Remeasure where the spec-as-key perf work stands on a FRESH build, then find
the next improvement. Diego cares about the optimization work specifically.

_Last updated: 2026-06-26 (mid-task). Keep this current so another agent can resume._

## TL;DR state right now
- Working worktree: `/home/diego/tmp-metal/wt-spec`, branch **`dgomez/metal2-output-alloc-memo`**
  (= runargs-builders tip + 2 new commits: i2s builder-wiring + BufferDistributionSpec memo).
- A `./build_metal.sh --build-tests` ran; **libs (`_ttnncpp.so`, `_ttnn.so`) relinked 10:17 today**,
  test-binary compile still finishing as of last check (does NOT affect python measurement).
- **Preliminary** i2s measurement (taken UNDER build CPU contention — re-measure clean!):
  - quasar i2s = 38.3 µs median / 37.07 µs min; legacy = 16.4 µs → **2.34×**, PCC 1.0.
  - Shape: 1024×1024 bf16 tile, interleaved DRAM → HEIGHT-sharded over 4×8=32 cores (shard 32×1024).

## The branches / PRs (all pushed)
- #48250 (draft) `dgomez/metal2-spec-runargs-builders` — builders + transpose/i2s migrations + doc.
- #48252 (draft, base=48250) `dgomez/metal2-output-alloc-memo` — **THIS branch**: i2s builder wiring
  (commit 46bbffa) + BufferDistributionSpec::from_shard_spec memo (commit 236da12, WIP/footgun).
- #48071 (open) RtaName −17%; #48138 (open) skip-validate −29%; #48060 (MERGED) small-vector.

## HOW TO MEASURE (critical setup — see lessons doc)
- ttnn is an **editable install pinned to `/home/diego/tt-metal` (main repo)** via a MetaPathFinder
  + `ttnn-custom.pth`. To load wt-spec's build you MUST override BOTH. The harnesses
  `/home/diego/i2s_perf.py` and `/home/diego/tp_perf_wt.py` already bake in the preamble:
  strip the 3 `ttnn-custom.pth` entries from sys.path, prepend wt-spec paths, patch the finder
  MAPPING. They `assert ttnn.__file__.startswith(WT)`.
- Run: `TT_METAL_INSPECTOR=0 MM_ITERS=4000 MM_WARM=300 python3 /home/diego/i2s_perf.py`
  (transpose: `DIM=1024 python3 /home/diego/tp_perf_wt.py`).
- Methodology (LOCKED): DIM 1024, INSPECTOR=0, warm cache, min/median µs/op. **Measure only when
  NO build/ninja is running** (CPU contention inflates timings). Verify `_ttnncpp.so` mtime fresh.

## CLEAN MEASUREMENTS (build idle, fresh wt-spec binary, INSPECTOR=0, PCC 1.0)
- **i2s** quasar 38.4 µs / legacy 15.4 µs → **2.50×** (1024², 32-core height-shard).
- **transpose** quasar 33.8 µs / legacy 14.0 µs → **2.41×** (1024²).
- NOTE: doc/PR ratios (1.34×, 2.0×) used a legacy baseline ~2× higher than measured now; quasar
  absolutes match. Ratios in the doc need correcting; the gap is bigger than advertised.

## PROFILE — quasar i2s dispatch (py-spy --native, 4999 samples). Inclusive % of samples:
- MeshWorkload enqueue path 62.9% (umbrella)
- **create_program_spec 23.7%** ← dominant quasar-only cost (per-dispatch spec REBUILD)
- output alloc: allocate 20.7% / create_output 17.6% / Buffer:: 12.3% ← ~shared floor w/ legacy
- enqueue_mesh_workload 20.0% (command writing)
- **CoreRangeSet 10.8% / merge 7.4%** ← from `CoreRangeSet(Span<CoreCoord>)` ctor at factory
  **line 87** (`all_cores` built from per-core list each dispatch). NOT from_shard_spec.
- KernelSpec ctors 9.6%; hash 8.2% (reflect 2.9% + _Hash_bytes 2.8%); filesystem::path 4.2%
- **from_shard_spec 1.6%** ← what the pushed memo (#48252) actually covers = MIS-TARGETED for i2s.

## KEY CONCLUSIONS
1. The #48252 memo targets the wrong CoreRangeSet cost (from_shard_spec 1.6%, not the merge 7.4%).
   The merge is in the factory's `all_cores` construction inside create_program_spec.
2. The real lever is **create_program_spec (23.7%)** — the per-dispatch spec rebuild. Sub-levers:
   all_cores CoreRangeSet construction (line 87), KernelSpec allocation, filesystem::path (4.2%).
3. Output alloc (~20%) is the shared floor (legacy pays it too) — limited headroom.
4. Biggest architectural win = memoize the built ProgramSpec on a cheap key (spec is rebuilt only to
   be hashed). This is a DESIGN decision → discuss with Diego (feedback_discuss_infra_changes).

## NEXT STEPS (in order)
1. **Wait for `--build-tests` ninja to finish**, then RE-MEASURE i2s clean (no contention).
2. **Isolate the memo's contribution.** Plan: env-gate the memo in `buffer_distribution_spec.cpp`
   (e.g. `if (getenv("TT_BDS_MEMO"))`), rebuild ONCE, measure TT_BDS_MEMO unset (baseline) vs set.
   Avoids 2 separate builds. Remove the gate before finalizing.
3. Remeasure transpose (expect ~35 µs / 1.34× per doc) to confirm builders still hold on fresh build.
4. If memo buys little or the gap is elsewhere: profile i2s dispatch (py-spy / tracy) on THIS build
   to get the current bottleneck breakdown. Doc's prior i2s profile: output alloc+free ~45% (shared
   floor), CoreRangeSet::merge ~3 µs (memo target), spec hash ~3 µs, run-args ~2.4 µs.
5. Decide: keep memo (footgun, global static cache) vs move to the spec-hash-keyed layout sidecar
   (the footgun-free target in SPEC_AS_KEY_RUNARGS_PERF.md).

## IN PROGRESS (2026-06-26): CoreRangeSet solid-rectangle fast path
- Constraint from Diego: NEVER touch the spec rebuild (op-writers want it); optimize EVERYWHERE ELSE
  incl. shared-with-legacy paths. See memory feedback_cannot_touch_spec_optimize_elsewhere.
- Edit (UNCOMMITTED in wt-spec): `tt_metal/common/core_coord.cpp` `CoreRangeSet(Span<CoreCoord>)` ctor
  — O(N) solid-rectangle detection (bbox + flat bitset dup-check) emits one CoreRange, skips
  merge_ranges (2D grid alloc + std::set churn). Behavior-preserving; helps i2s + all sharded ops +
  legacy. Targets the profiled CoreRangeSet cost (~5% self / 10.8% incl).
- RESULT (measured, fresh libtt_metal 11:06, PCC 1.0): **i2s 38.4 → 33.1 µs = −5.3 µs / −14%**
  (ratio 2.50× → 2.11×). transpose unchanged (34.1 µs; not built from a coord list). Tight dists
  (i2s min 32.05 vs prior 37.15 — outside noise). Win confirmed.
- STATUS: edit still UNCOMMITTED in wt-spec. Recommend packaging as its OWN PR off main (pure metal
  internal, independent of spec-as-key) so legacy + all sharded ops benefit. Awaiting Diego's go on PR.
- VALIDATION (2026-06-26):
  - Unit tests: added 4 cases to tests/.../api/core_coord/test_CoreRangeSet_construct.cpp
    (SolidRectangle, Single, LShapeNotCollapsed, DuplicateThrows). ALL 13 CoreCoordFixture CoreRangeSet
    tests PASS (build_Release/test/tt_metal/unit_tests_api --gtest_filter='CoreCoordFixture.*CoreRangeSet*').
  - FINDING: original ctor THROWS on duplicate coords (validate_no_overlap). Fast path's flat-bitset
    dup-check preserves that (forces fallback→throw); without it a naive count==area would wrongly
    accept dup-with-hole. See AGENT_LESSONS #11.
  - Hardware PCC 1.0 (i2s, transpose).
  - SIM BLOCKER: no simulator on this box (TT_METAL_SIMULATOR unset, no sim executable). Diego asked
    for a sim run; cannot satisfy unilaterally — surfaced for his guidance.

## EXPERIMENT (2026-06-26): by-slot apply ceiling — SetProgramRunArgs (Audrey's code)
- Threw away `slot_of.find(name)` per-arg hash lookup, used positional apply (builder emits in slot
  order). Rebuilt, measured, REVERTED (her file now clean — see reference_setprogramrunargs_audrey_owned).
- RESULT: i2s 33.1 → **30.6 µs = −2.5 µs / −7.5%**, PCC 1.0. Bigger than the ~1.5 µs guess. Cumulative
  with CoreRangeSet fast-path: 38.4 → 30.6 = **−20%**.
- CAVEAT: SetProgramRunArgs is Audrey's; she prefers strings/maps and may reject a by-slot rep change.
  The win is real (~2.5 µs) → worth a conversation. Audrey-friendly options: (a) additive opt-in
  positional fast-path keyed off builder-emitted order, keeping her name→slot map as fallback;
  (b) small-N linear scan instead of hash (i2s has 7 args; memory says linear beats hash <~8 args,
  but regresses high-N — would need hybrid). Decision pending Diego (whether/how to approach Audrey).

## Files
- Harnesses: `/home/diego/i2s_perf.py`, `/home/diego/tp_perf_wt.py`.
- Build log: `/home/diego/tmp-metal/wt-spec/build_remeasure.log`.
- Methodology/results writeup: `tech_reports/Metal2OpMigration/SPEC_AS_KEY_RUNARGS_PERF.md`.
- Lessons: `tech_reports/Metal2OpMigration/AGENT_LESSONS_spec_as_key_perf.md`.
