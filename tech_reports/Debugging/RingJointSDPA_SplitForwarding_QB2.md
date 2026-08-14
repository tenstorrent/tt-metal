# Ring-joint SDPA: even-ring split forwarding — QB2 debug handoff

Even-ring split forwarding was reverted out of ring-joint SDPA to unbreak the QB2 nightly. **The root
cause was never found.** This document is everything needed to debug it directly on `bh_quietbox_2`
and to decide whether the feature is worth re-landing.

| | |
|---|---|
| Reverted by | PR #53076, commit `3940590cfdc` (6 files, `+35/-110`) |
| Broken by | PR #52730, commit `c080ab0cfc94` |
| Prior fix attempt | PR #53053 (Pavle Josipović) — did not clear the failures |
| **Root cause** | **Found 2026-08-14 on QB2 — single-packet inputs cannot be halved (§0)** |
| As of | 2026-08-14 |

Each claim below is tagged **[VERIFIED]** (measured or observed directly), **[HYPOTHESIS]**
(reasoning, untested), or **[UNRESOLVED]** (unknown, must be checked).

---

## 0. UPDATE 2026-08-14 — root cause found and fixed on QB2 [VERIFIED]

Debugged directly on `bh_quietbox_2` (4x Blackhole p150b, ring 4). **The root cause is not a ring-4
geometry problem.** It is a shape-driven defect in the packet-halving arithmetic:

```cpp
// ring_attention_all_gather_{reader,writer}.cpp
const uint32_t first_half_pages = (num_packets / 2) * packet_size_in_pages;
```

When an input's per-batch-head range is a **single packet**, `first_half_pages == 0` and the two
split predicates degenerate to *"direction 0 relays nothing, direction 1 relays everything"* — while
slice counting, semaphore signaling, and the consumer wait all continue as if a real two-way split
had happened. The fix is one symmetric guard per kernel:

```cpp
const bool split_this_input = is_split_forwarded_slice && (first_half_pages > 0);
```

### Evidence (all measured on QB2, `test_..._indexed_kv_cache_accuracy`)

| Configuration | PCC chunk 0 (fp32) | Result |
|---|---|---|
| Feature restored as merged (`c080ab0cfc94`) | 0.97944098 | FAIL — reproduces §3 exactly |
| Halving disabled; counting + consumer wait still on | 0.99984010 | PASS |
| Halving on, single-packet inputs exempt (**the fix**) | 0.99984010 | PASS |
| Same fix, `kimi_k3-all-qk` + `kimi50k-all-qk` | — | PASS, no hang |

The third row is the load-bearing one: this workload has two inputs, one `npkt=1` and one `npkt=2`.
The fix leaves the `npkt=2` input genuinely split across both links, so **split forwarding stays
active at ring 4**. This is a real fix, not the ring-size gate in disguise.

### What this corrects in the rest of this document

- **§7's ring-4 degenerate-case hypothesis is DISPROVED.** Ring 4 is not special. The `+1` relay
  accounting, the twice-arriving diametric shard, and `split_shard_id` were each confirmed correct
  at ring 4 by on-device `DPRINT`. Note also that §7 conflated the variable with the relay count:
  the Ring relay predicate is `slices_received < (writes_expected + 1)`, so direction 1 at ring 4
  performs **one** relay pre-split and two post-split, not zero and one.
- **§8 is CLOSED, not a defect.** `ntf=2, ntb=1, split_en=1` uniform across all four chips in both
  reader and writer. Both factories apply the identical double-swap, so the signaler and the AG
  kernels agree *by construction*. Nothing here needed fixing.
- **The bug is shape-driven, not topology-driven.** Ring 4 only made it *visible*: with
  `writes_expected == 1`, every relay is the split relay, so one unsplittable input corrupts every
  transfer. At ring 8 most relays are ordinary and the defect is diluted — **it is therefore latent
  on the galaxy today**, on the exact path carrying the LTX win. See §14.
- **A `ring_size > 4` gate is the WRONG fix.** It was validated here (both bugs clear) before the
  real cause was found, but it masks a live ring-8 defect and forfeits the optimization at ring 4
  for no reason. Do not ship it.

Two intermediate hypotheses were also disproved by evidence and are recorded so nobody re-runs them:
the consumer *does* apply the split second-half wait on the correct semaphore
(`fused_op_receiver.hpp:53-56`, arithmetic checks out at ring 4 and ring 8), and the two directions
signal **distinct** semaphores (`ccl_op_fusion.cpp:73`, indexed by direction).

---

## 1. Read this first: the performance case is not supported by measurement [VERIFIED]

A controlled A/B on a BH 4x8 galaxy — same machine, same commit apart from the six files, same env
file, same seed, gen #2 traced pure replay — measured LTX distilled at:

| Stage | Split forwarding ON | Reverted | Δ |
|---|---|---|---|
| Encoder | 0.18 s | 0.18 s | 0 |
| Stage 1 denoise | 2.32 s | 2.35 s | +0.03 |
| Latent upsample | 0.13 s | 0.14 s | +0.01 |
| Stage 2 denoise | 2.72 s | 2.73 s | +0.01 |
| VAE decode | 0.59 s | 0.59 s | 0 |
| Audio decode | 0.42 s | 0.42 s | 0 |
| **Total** | **6.36 s** | **6.41 s** | **+0.05 s (0.8%)** |

Per-stage deltas are at or below single-sample resolution, and the split-forwarding arm was also
faster on both capture generations (45.6 → 9.0 vs 46.9 → 9.7) — the wrong direction for a real
effect. Treat this as **indistinguishable from zero**, not as a measured 0.8% regression.

The measurement is specific to LTX distilled at ring 8. The feature may still pay off at other
shapes, on longer rings, or on the strided-AGMM path (which was *not* reverted). But the workload
that motivated it does not show a benefit. **Establish where the win actually is before investing in
a fix.**

Note also that the 6.1 s figure quoted in #52730's description is not reproducible on that box in
either arm; both land at ~6.4 s. Do not use 6.1 s as a baseline.

---

## 2. Current state

`main` is green. The revert restored six files to a byte-identical copy of their pre-#52730 state and
removed the feature entirely — it did not fix anything.

| Run | Scope | Result |
|---|---|---|
| [31733956385](https://github.com/tenstorrent/tt-metal/actions/runs/31733956385/job/94563823330) | Targeted — `sdpa nightly tests (QB2 only)` | success |
| [31738463549](https://github.com/tenstorrent/tt-metal/actions/runs/31738463549) | Full default sanity, all 5 suites, both SKUs | 43/43, 0 failures |
| [31719235944](https://github.com/tenstorrent/tt-metal/actions/runs/31719235944/job/94514373094) | PR #53053, prior fix attempt | 10 failures |
| [31550656743](https://github.com/tenstorrent/tt-metal/actions/runs/31550656743) | Last known-good nightly before #52730 | success |

---

## 3. What was failing [VERIFIED]

Ten test failures on `sdpa nightly tests (QB2 only) [bh_quietbox_2]`, but only **two real bugs**. Six
are cascade failures from the dead device and will clear once the hang is fixed. Do not chase them.

| Group | Tests | Signature |
|---|---|---|
| **Hang** | `test_ring_mla_chunked_accuracy[kimi_k3-all-qk-chunk2560]`, `[kimi50k-all-qk-chunk2560]` | `TIMEOUT: device timeout, potential hang detected, the device is unrecoverable` (`system_memory_manager.cpp:779`) |
| **PCC** | `test_ring_joint_attention_chunked_nd_sharded_indexed_kv_cache_accuracy[fp32_acc]`, `[bf16_acc]` | PCC 0.9794 vs 0.997 (fp32); 0.9796 vs 0.994 (bf16); RMSE ≈ 0.0664 vs 0.05 |
| Cascade (not independent) | `perf_check` ×2, `determinism` ×3, `minimax3_gqa` determinism ×1 | `Device 0 init: failed to initialize FW!` |

Two observations that matter:

- The PCC pair sits **earlier in the test file** than the hang (line 3464 vs 5390), so it ran before
  the device died and is independent of it.
- fp32 and bf16 produce **near-identical PCC** (0.97944 / 0.97957). That reads as systematic data
  corruption, not a race.

**Attack the PCC bug first.** It is deterministic, fast, and does not wedge the device between
iterations.

---

## 4. Why it only shows on QB2 [VERIFIED]

The suite runs on exactly one SKU. `tests/pipeline_reorg/blackhole_multi_card_sanity_tests.yaml:51`
defines the entry `sdpa nightly tests (QB2 only)` with `cmd: pytest tests/nightly/blackhole/sdpa/ -svv`,
and its `skus:` block lists only `bh_quietbox_2`. **No galaxy SKU runs that directory.** That is how
the regression reached `main`: LTX was developed and validated on a 4x8 galaxy, and the SDPA gate
lives exclusively on a 4-device QuietBox.

**Do not try to reproduce on a galaxy.** `MeshConfig.detect()` (`tests/nightly/sdpa_perf_utils.py`)
derives ring size from counting `/dev/tenstorrent/*` with no env override, and sequence length scales
as `128 × sp_size`. The failing parametrization does not exist there.

| | QB2 (fails) | 4x8 galaxy |
|---|---|---|
| devices | 4 | 32 |
| `is_galaxy` | false | true |
| **ring — `sp_size`** | **4** | **8** |
| `tp_size` | 1 | 4 |
| core grid | 11×10 | 12×10 |
| SDPA cores (CCL column reserved) | 100 | 110 |
| total seq (`128 × sp`) | 512 | 1024 |
| **chunk-0 `logical_n`** | **256** | **512** |
| MLA chunked test id | `all-qk-chunk2560` | `q32-k512-chunk5120` |

For reference: on a galaxy the same PCC test *passes* at 0.9975 / 0.9994. 41 ring-joint SDPA accuracy
and determinism tests were run there post-revert with zero genuine failures — which only establishes
absence of ring-8 regression, and says nothing about ring 4.

---

## 5. Reproducing on QB2

Restore the feature on top of `main`, build, and run the two real failures directly.

```bash
# 1. put split forwarding back (6 files, +110/-35)
git checkout c080ab0cfc94 -- \
  ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp \
  ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp \
  ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp \
  ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_fusion.cpp \
  ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_fusion.hpp \
  ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp

# 2. build (only 3 of the 6 are host code; the two kernels and the receiver header are JIT)
./build_metal.sh

# 3a. the PCC bug — deterministic, fast, does not wedge the device. Start here.
pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "test_ring_joint_attention_chunked_nd_sharded_indexed_kv_cache_accuracy" -v -s

# 3b. the hang — expect to reset the device between attempts
pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py \
  -k "test_ring_mla_chunked_accuracy" -v -s

# the full nightly, as CI runs it
pytest tests/nightly/blackhole/sdpa/ -svv
```

Arm the device-side watchdog so a genuine hang writes triage artifacts instead of just stalling:

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
export TT_TRIAGE_OUTPUT_PATH=$HOME/triage/triage_output.txt
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="$TT_METAL_HOME/tools/tt-triage.py \
  --disable-progress --triage-summary-path=$HOME/triage/triage_summary.txt 2>&1 | tee $TT_TRIAGE_OUTPUT_PATH"
```

If the watchdog never fires and the triage directory stays empty, no device op exceeded 300 s and you
are looking at host compute, not a wedge.

---

## 6. What the feature does

On an even ring, the shard from the diametrically opposite device is relayed **split across both
links** — each direction carries half — instead of one direction carrying the whole thing. The intent
is to balance the terminal relay hop. The gate is:

```cpp
(topology == Topology::Ring) && (ring_size % 2 == 0) && (ring_size > 2)
```

so it is active at both ring 4 and ring 8.

```
        ring 8                              ring 4
          self                                self
      7 .   0   . 1                            0
    6 .           . 2                     3        1
      5 .   4   . 3                            2
            ^                                  ^
      diametric = +4                    diametric = +2
   (4 hops either way)          (2 hops either way — and the forward
                                 direction already reaches distance 2)
```

---

## 7. ~~Leading hypothesis: ring 4 is a degenerate case~~ [DISPROVED — see §0]

> **Superseded 2026-08-14.** Every claim in this section was checked on device and the mechanism it
> proposes does not occur. Kept for the record; do not act on it. The arithmetic below is still a
> correct description of the ring counts — it is the *conclusion* drawn from them that is wrong.


Ring counts come from `get_forward_backward_configuration`
(`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp`):

```cpp
num_targets_forward  = tt::div_up(ring_size - 1, 2);
num_targets_backward = ring_size - 1 - num_targets_forward;
if (ring_index % 2 == 0) std::swap(num_targets_forward, num_targets_backward);  // static_alternate
```

Both the AG factory
(`ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp:412`) and the SDPA
factory (`ring_joint_sdpa_program_factory.cpp:389`) then swap **again** on even index, cancelling the
internal swap. Net result — uniform across devices:

| `ring_size` | `num_targets_forward` | `num_targets_backward` | direction-1 `writes_expected`, pre-split |
|---|---|---|---|
| 8 | 4 | 3 | 2 |
| **4** | **2** | **1** | **0** |

The reader sets `writes_expected = num_targets_backward_direction - 1` for direction 1, then split
forwarding adds `+1` to both that and `slices_expected`. At ring 8 that turns 2 relays into 3. **At
ring 4 it turns *zero* relays into one** — the feature introduces a relay on a direction that
otherwise performs none.

Combined with the geometry: at ring 4 the diametric device is 2 hops away in both directions, while
`num_targets_forward = 2` means the forward direction already reaches it. The suspicion is that the
diametric shard is **delivered twice** — once whole by the forward direction, once as halves via the
split path. That is a clean mechanism for both symptoms: a partial overwrite gives a deterministic
PCC error, and the surplus semaphore increment strands a later `wait_min` as a hang.

### First things to instrument

1. At `ring_size = 4`, count actual direction-0 and direction-1 signal increments per shard against
   `split_second_half_wait = backward_writes_expected + 1`.
2. Check whether the diametric shard's destination pages are written twice — instrument the writer's
   `relay_this_packet` path.
3. Confirm which shard `split_shard_id = (ring_index + backward_writes_expected + 1) % ring_size`
   resolves to at ring 4, and whether it matches the shard the writer is splitting.

---

## 8. ~~Open question — resolve before trusting any of the above~~ [RESOLVED — not a defect, see §0]

> **Closed 2026-08-14.** Confirmed by on-device `DPRINT`: `ntf=2, ntb=1, split_en=1` uniform on all
> four chips, reader and writer agreeing. The mapping is correct by construction. The direction-
> inversion warning below is accurate and still worth reading, but nothing here is broken.


`split_second_half_wait` and `split_shard_id` are both derived from the signaler's
`backward_writes_expected`, set via `RingSDPAFusedOpSignaler::init_all_gather()` at
`ring_joint_sdpa_program_factory.cpp:943`. **The values the locals passed there actually hold were
never confirmed**, nor whether they match the `num_targets_*` the AG kernels receive as compile args.

Everything in §7 depends on that mapping. Pin it down first.

Beware: the direction conventions are **inverted between the two ops on purpose**. The receiver
comments state that the all-gather's BWD semaphore belongs to direction 1 and its FWD semaphore to
direction 0. A naive reading of the names will mislead you. Note also that
`build_ring_write_plan()` (`ring_joint_sdpa_program_factory.cpp:367`) assigns
`plan.forward/backward_writes_expected` differently for Linear vs Ring topology.

---

## 9. Two latent defects [VERIFIED]

Both are in the code the revert removed, both were harmless only by accident, and both will bite
whoever re-lands the feature.

| Where | Defect | Why it is currently masked |
|---|---|---|
| `fused_op_indexer.hpp:36` | Advances `rt_args_idx += 2` while `push_ring_sdpa_fused_op_rt_args` pushes 5 — a permanent 3-arg desync for anything reading after it. | That constructor has no callers; `ring_joint_sdpa.cpp:145` uses the explicit 4-arg one. |
| `fused_op_receiver.hpp:60` | `get_next_ring_id_and_consume_one_signal()` never received the split-shard second-half wait its sibling `get_next_ring_id_and_sync()` got, and uses `down(1)` rather than `wait_min`. | Only the sliding-window path calls it (`ring_joint_reader.cpp:668`), which #53053 gated off. |

Same failure shape twice: the feature added protocol state that not every consumer of the runtime-arg
block learned about. Three separate classes parse that block — `RingSDPAOpReceiver`,
`RingSDPAOpIndexer`, and the `exp_` variant — and #52730 updated one.

---

## 10. The implementation that gets this right [VERIFIED]

PR #52513 added split forwarding to strided AGMM **independently**, and it is correct — it was not
reverted and is unaffected. Use it as the template for any re-land.

It splits at *chunk* granularity with byte-identical producer and consumer predicates:

```cpp
// minimal_default_writer.cpp:330          // minimal_default_mm_signal_aggregator.cpp:81
bool relay_this_chunk =                    bool receive_this_chunk =
  !is_split_forwarded_slice ||               !is_split_received_slice ||
  (direction == 0                            (direction == 0
     ? (chunk_idx <  first_half_chunks)         ? (chunk_idx <  first_half_chunks)
     : (chunk_idx >= first_half_chunks));       : (chunk_idx >= first_half_chunks));
```

…and it waits on a **monotonic event counter** — `event_target++` then
`noc_semaphore_wait_min(per_worker_sem_ptrs[w], event_target)`, per worker, per chunk, per sub-band.
Every increment is individually accounted, so there is no threshold to get wrong and the `+1` vs `+2`
question cannot arise.

The structural difference matters more than the code. In AGMM the producer and consumer live in the
same op family and derive the predicate from the same compile-time constants, so they agree by
construction. In SDPA the decision must be **mirrored across an op boundary** — #52730's own comment
says *"Must match the all-gather kernels' split-forwarding gate exactly."* #53053 improved that by
passing the flag instead of re-deriving it, but left the coarse single-threshold wait underneath.

**If you re-land, close that gap: make the consumer count events, not wait on one derived threshold.**

History worth knowing: this optimization originated as Ligang Long's AG-minimal balanced traffic
pattern and was **reverted twice** (#36607 → #37832, #37878 → #38202) before landing on the third
attempt in #38256. It has a track record of being hard to get right.

---

## 11. File map

**The feature — the six files the revert touched**

- `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp` — relay + slice accounting
- `.../ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp` — packet-half split, relay loop
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp` — consumer wait
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_fusion.cpp` / `.hpp` — rt-arg push, thresholds
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` — gate, write plan, signaler init

**Ring geometry and direction convention — start here**

- `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp` — `get_forward_backward_configuration`, `choose_worker_cores`
- `.../ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp` / `.hpp` — second swap at `:412`, compile args
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_utils.hpp` — `RingIdSequencer`

**Other consumers of the same protocol**

- `.../sdpa/device/kernels/dataflow/fused_op_indexer.hpp`, `exp_fused_op_indexer.hpp` — latent desync
- `.../sdpa/device/kernels/dataflow/ring_joint_reader.cpp`, `ring_joint_writer.cpp` — sync call sites
- `.../sdpa/device/kernels/compute/ring_joint_sdpa.cpp`, `exp_ring_joint_sdpa.cpp` — compute-side sequencer
- `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/ring_indexer_score_dsa_program_factory.cpp` — third fused consumer

**Signaler infrastructure**

- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` — `synchronize_workers_and_signal_op`, the actual increment
- `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp` / `.hpp` — signaler construction

**Reference implementation — correct, not reverted**

- `.../strided_all_gather_async/device/kernels/minimal_default_writer.cpp`, `minimal_default_reader.cpp` — #52513, chunk-granular
- `.../strided_all_gather_async/device/kernels/minimal_default_mm_signal_aggregator.cpp` — monotonic event counter
- `.../all_gather_async/device/kernels/minimal_default_reader.cpp`, `minimal_default_writer.cpp` — Ligang's original

**Config surface that makes the two machines differ**

- `tests/nightly/sdpa_perf_utils.py` — `MeshConfig`, hardcoded 12×10 / 11×10 grids
- `tests/pipeline_reorg/blackhole_multi_card_sanity_tests.yaml:51` — the QB2-only SKU list
- `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` — the failing tests

---

## 12. Things that will waste your time

- **Core placement is probably not the cause.** `choose_worker_cores` is fully derived — it queries
  `device->worker_cores()`, intersects an optional sub-grid, and warns rather than fails when short.
  No split-forwarding code touches core coordinates. And fp32/bf16 producing identical PCC does not
  fit a placement or contention artifact.
- **A red nightly is not automatically a code failure.** In the window that pinned this regression,
  two consecutive QB2 nightly failures were infrastructure — a runner losing communication, and
  `Value cannot be null (ContainerId)`. Check the annotation before concluding anything.
- **Silence is not a hang.** The chunked tests compute a torch CPU reference for minutes with the
  device idle — measured at ~16 cores saturated and 33.6 GB RSS, while the device-side watchdog never
  fired and wrote zero triage artifacts. Job schedulers that infer "hung" from stdout silence will
  reap these runs falsely.
- **Do not pipe pytest through `tail`, `sed`, or `cut`** without `stdbuf -oL` — output buffers until
  exit and the run looks dead.
- **Beware the test-id trap.** `-k "4x8sp1tp0nl2_ring_is_fsdp0"` also matches the i2v variant, whose
  id is a superstring and which collects first. Verify with `--collect-only` and count every match.
- **`CI=true` changes which tests exist.** `_generate_chunked_test_configs`
  (`test_ring_joint_sdpa.py:4897`) branches on `os.environ.get("CI") == "true"`: under CI each model
  collapses to a single `-all-qk` test sweeping every (q,k) combo in one device session, while
  locally it expands into separate per-(q,k) tests. **The `kimi_k3-all-qk-chunk2560` and
  `kimi50k-all-qk-chunk2560` ids that hang in the nightly do not exist without `CI=true`** — running
  the hang repro locally without it silently tests a different execution pattern and passes.

---

## 13. Remaining work on QB2

The root cause is fixed (§0) and the two real failures from §3 pass. What the fix has *not*
exercised:

1. **The sliding-window path.** `get_next_ring_id_and_consume_one_signal()` still lacks the split
   second-half wait its sibling has (§9, defect 2). The failing tests used
   `get_next_ring_id_and_sync()`, so this path is untested against split forwarding. Find or add a
   test that reaches `ring_joint_reader.cpp:670` with split forwarding enabled.
2. **The `indexer_score` op** — the third consumer of the same runtime-arg block. Run its tests with
   split forwarding on. See also the stale constant in §9.
3. **Full nightly, clean.** `CI=true pytest tests/nightly/blackhole/sdpa/ -svv`. Note `CI=true` is
   mandatory to collect the `-all-qk` ids (see §12).

## 14. Remaining work on a galaxy — this is the blocking handoff

None of this can be obtained on QB2.

1. **Confirm the defect is live at ring 8.** §0 argues the single-packet case is latent on the
   galaxy, diluted rather than absent. Find or construct a ring-8 workload whose relayed input is
   one packet wide. **If it reproduces, this is a correctness fix for the galaxy, not just a QB2
   unbreak** — which changes the priority of landing it.
2. **Regression-check ring 8.** Re-run the ring-joint SDPA accuracy and determinism suites where
   split forwarding already worked, to confirm the guard changes nothing there.
3. **Re-measure LTX distilled end-to-end.** The fix relays single-packet inputs whole, so slightly
   less traffic is balanced than in the merged version. The §1 A/B must be repeated to confirm the
   0.05s survives. Use the §1 methodology (same machine, same seed, gen #2 traced pure replay).
4. **Close the coverage gap.** Add a galaxy SKU to `tests/nightly/blackhole/sdpa/`, or a ring-4 case
   to a galaxy-visible suite. This gap is what let the regression reach `main`.

## 15. Still worth doing before a re-land

1. Fix both latent defects in §9, or delete the dead `RingSDPAOpIndexer(rt_args_idx)` constructor
   outright.
2. `ring_indexer_score_dsa_program_factory.cpp:57` hardcodes `fused_rt_width = 6`, but
   `push_ring_sdpa_fused_op_rt_args` now emits **9**. Currently benign — host and kernel both append
   and consume sequentially, so they shift together — but the `static_assert` at `:71` advertises
   that it catches exactly this drift and structurally cannot, since it only checks the derived
   constants against each other. Either derive the width from the pusher or drop the false comfort.
3. Consider the §10 recommendation (per-event counting, following #52513's AGMM aggregator) on its
   own merits. It was not the cause here — the consumer wait is correct (§0) — but it would have
   made this class of producer/consumer disagreement unrepresentable.

## 16. Environment and iteration notes (QB2)

- **`ttnn` is built but not installed.** `uv pip install -e . --no-deps` into the active venv; also
  needs `graphviz`. Do **not** put the repo root on `PYTHONPATH` — it shadows the installed package
  and surfaces as a misleading `No module named 'ttnn.device'` from `conftest.py`.
- **Builds are fast.** A host-file change (touching `ring_joint_sdpa_program_factory.cpp`) rebuilt
  10 targets in **37 seconds** on this box. An earlier claim of ~3.5 h in this workflow was an
  artifact of inferring elapsed time from file mtimes across a dropped connection — it does not
  reproduce. `build_metal.sh` defaults to `Enable ccache: OFF`; `-c` enables it.
- **Kernel-only changes need no rebuild at all** — the two AG kernels, `fused_op_receiver.hpp`, and
  `fused_op_indexer.hpp` are JIT-compiled per run. A full bisect iteration is ~6 s.
- **Never edit kernel sources while a test run is in flight.** They are JIT-compiled per test, so
  edits are picked up mid-run and silently invalidate everything after that point.
- **`DPRINT` is printf-style here**, not stream-style: `DPRINT("x={}\n", v)`. `DPRINT << ...` fails
  with a `static_assert` ("Old style DPRINT is deprecated"), and the error surfaces as a *kernel
  compile* failure inside the pytest log, not a build failure. Include `api/debug/dprint.h`; enable
  with `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_CHIPS=all TT_METAL_DPRINT_RISCVS=BR,NC`.
