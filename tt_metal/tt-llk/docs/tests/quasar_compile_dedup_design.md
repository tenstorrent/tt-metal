# Design: deduplicating `--compile-producer` test items by `variant_id`

Status: **design / not implemented.** This documents an investigation into the
single largest remaining lever for the `llk-build-quasar` PR-gate job and the
reasons it has not (yet) been shipped. It is intended to be read alongside the
broader write-up in [`quasar_compile_investigation.md`](quasar_compile_investigation.md)
and PR [#48767](https://github.com/tenstorrent/tt-metal/pull/48767).

## Problem

The `llk-build-quasar` job runs `pytest --compile-producer -m quasar` over the
full Quasar test enumeration — **413,234 test items** in the measured run. Each
item pays the full pytest per-item cost (collection slot, fixture setup/teardown,
test-body Python, xdist marshalling, JUnit reporting) even though the *compile*
each item asks for is very often a no-op cache hit against an artifact another
item already produced.

The artifacts themselves are already deduplicated. `build_elfs()`
(`helpers/test_config.py`) keys the ELF output directory on `variant_id` and
guards it with a `.build_complete` done-marker plus a per-`variant_id`
`FileLock`; the second and subsequent items that resolve to the same `variant_id`
hit the fast path and return without recompiling. What is **not** deduplicated is
the per-item *framework* cost that runs before that fast path: the item is still
collected, its fixtures still run, its test body still constructs a `TestConfig`,
generates stimuli, and computes the hash — only then does `build_elfs()` notice
the work is redundant.

## Evidence

Across the 6 largest quasar test files — **94.7%** of the 413,234-item suite —
the items collapse onto roughly **2,500–3,200 distinct `variant_id`s**. That is a
**~120–160:1 redundancy ratio**: only ~2.5–3.2k items are genuine compiles; the
other ~410k are cache-hit no-ops that still each pay the full per-item Python /
pytest overhead (JUnit median ~7–9 ms/item).

(The collapse ratio was measured by projecting the per-item parametrize arguments
onto the fields that actually feed `generate_variant_hash()`; see "Why it's hard"
for why that projection is not authoritative on its own.)

### What `variant_id` is

`variant_id` is a sha256 hash computed in `TestConfig.generate_variant_hash()`
(`helpers/test_config.py`). It hashes `str(value)` of every field in the
`TestConfig` instance `__dict__` **except** a documented `NON_COMPILATION_ARGUMENTS`
exclusion list. That list always excludes bookkeeping fields
(`run_configs`, `variant_id`, `runtime_arguments_struct`, `runtime_format`,
`passed_templates`, `passed_runtimes`, `current_run_type`, `temp_elfs`) and,
when **not** running `--speed-of-light`, additionally excludes
`variant_stimuli`, `pack_size`, `unpack_size_a`, `unpack_size_b`, `runtimes`,
and (unless `compile_time_formats`) `formats_config`. In other words: stimuli
*values* and most runtime-only arguments deliberately do **not** affect the
compile hash — which is exactly why so many items collapse onto one `variant_id`.

## Why it's hard

The blocker is **when** `variant_id` becomes computable.

`generate_variant_hash()` runs inside `.run()` → `.prepare()`, i.e. at the very
end of each test body — **after** the parametrized fixtures have run,
`generate_stimuli()` has executed, and the `TestConfig` has been fully
constructed from that test's bespoke logic. By the time you can compute the key
that tells you an item is a duplicate, you have already paid essentially all of
that item's per-item cost. An in-body "am I a duplicate? then skip" check saves
almost nothing, because the expensive part already ran to build the object the
check reads.

There is **no existing generic mechanism** to derive an equivalent dedup key
from just the pytest parametrize arguments at *collection* time, before fixtures
run. The mapping from "parametrize params" to "what actually affects the compile"
is not centralized: it lives as bespoke logic scattered across each test body
(which templates/runtimes each test constructs, which format fields it passes,
which options it toggles). `generate_variant_hash()` is the single point that
currently reifies that mapping, and it only exists once the body has already run.

### Correctness stakes are asymmetric

Consumer-mode correctness risk is assessed as **near-zero if implemented
correctly**. Artifacts are already deduplicated today via the `variant_id` +
`.build_complete` marker, so consumer mode already loads shared artifacts across
many items with no change. The *only* correctness risk is getting the dedup
**key** exactly right: a collection-time key that does not perfectly mirror
`generate_variant_hash()` could silently drop an item that was in fact a genuinely
distinct variant. In consumer mode that surfaces as a **hard missing-ELF failure**
(the dropped variant was never compiled), not a soft slowdown. A wrong key fails
loud, not silent — which is a safety net, but it also raises the bar: you want
high confidence the key mirrors the hash before you ship, because a mismatch
breaks the gate rather than merely degrading it.

## Options

### (a) Collection-time dedup hook

Add a `pytest_collection_modifyitems` hook in `conftest.py` (there is already one
there for `--test-order-file`; `item.callspec.params` is already read elsewhere in
the conftest, so the collection-time params are accessible) that computes a
*would-be compile key* directly from `item.callspec.params` and deselects items
whose key has already been seen.

- **Pro:** attacks the cost at the only point where dropping an item removes
  *all* of its per-item overhead (collection, fixtures, body, marshalling,
  reporting) — the only kind of drop that shrinks both the call-phase term and
  the framework-overhead term (see Savings).
- **Con:** requires building and maintaining a `params → compile-key` mapping
  **outside** the per-test-body logic that currently owns it. That mapping can
  drift out of sync with `generate_variant_hash()` over time (a test starts
  toggling a new compile-affecting option; the hook doesn't know; it over-dedups;
  consumer mode breaks).
- **Mitigation — CI belt-and-suspenders assertion:** on a full producer run,
  assert `#distinct(collection_key) == #distinct(variant_id)` (the latter is
  observable from the produced artifact dirs / logs). If the counts diverge, the
  mapping has drifted and the run fails loudly *in producer mode* (safe) rather
  than silently dropping a variant that only bites later in consumer mode. This
  turns silent drift into a caught, actionable CI failure.

### (b) Reduced, separately-enumerated producer parameter set

Restructure producer mode so it enumerates a *reduced* parameter set up front
(one item per genuine compile), rather than enumerating the full test matrix and
deduplicating after the fact.

- **Pro:** cleaner separation of concerns — the reduced enumeration *is* the set
  of genuine compiles, so there is no after-the-fact key to keep in sync; no
  ~410k phantom items are ever created.
- **Con:** a bigger architectural change. It needs each test (or a shared layer)
  to expose "the distinct compile configurations I need" independently of the
  full functional matrix, and to keep consumer mode able to map its full-matrix
  items back onto those shared artifacts. More invasive, more up-front design.

### (c) Leave it as-is

Given (1) the effort and (2) the asymmetric failure mode (a wrong dedup key is a
hard consumer-mode failure, not a soft degradation), a legitimate option is to
not ship dedup and instead keep improving the cheaper, lower-risk levers
(coordinator throughput, worker count, per-item Python cost, PCH). The safety net
raises the confidence bar for (a)/(b); until that bar is met, the status quo is
defensible.

## Recommended next steps

1. **Prototype option (a) offline, non-gating.** Build the `params → compile-key`
   mapping for the 6 largest files only (94.7% of the suite), run a full producer
   pass with the belt-and-suspenders assertion enabled, and confirm
   `#distinct(key) == #distinct(variant_id)` before deselecting anything. This
   validates the mapping with zero risk to the gate.
2. **Only then wire it into the gate**, with the assertion left permanently on so
   any future drift fails the producer run loudly.
3. **Keep the assertion cheap** — it is a set-size comparison over data the run
   already produces; it should not itself become a per-item cost.
4. Treat option (b) as the longer-term target if the `params → compile-key`
   mapping in (a) proves hard to keep in sync, since (b) removes the sync problem
   by construction.

## Savings estimate (and its uncertainty)

At the best-known config (`-n 10`), the JUnit artifacts give:

- **Ideal-packed producer wall-clock** (Σ of JUnit call-phase times ÷ worker
  count) ≈ **544 s**.
- **Actual observed producer wall-clock** ≈ **1212 s**.

The ~**668 s gap** is per-item framework overhead (fixture setup/teardown,
reporting, execnet marshalling, collection) sitting *on top of* call-phase time.
Crucially, **only a true collection-time drop** (option (a)/(b)) — not an in-body
skip — shrinks **both** terms for the ~410k redundant items: an in-body skip still
pays collection + fixtures + body + marshalling + reporting, so it barely touches
the 668 s framework term and does nothing for the items that are already fast in
the call phase.

If the ~410k redundant items were never created, producer wall-clock would be
dominated by the ~2.5–3.2k genuine compiles plus a small fixed overhead —
an **order-of-magnitude** speedup in principle. **But this exact number could not
be measured** in the investigation sandbox: there is no Quasar toolchain
(`riscv-tt-elf-g++`) or torch/pytest environment available, so the genuine-compile
wall-clock and the residual fixed overhead were not profiled directly. The
order-of-magnitude framing is an upper bound implied by the redundancy ratio and
the measured Σ-call-phase/overhead split, not a measured result. Real profiling
on a runner with the actual toolchain is required to turn it into a committed
number.

## Open questions

- **Does the `params → compile-key` mapping stay tractable across all 26 files,**
  or only the 6 largest? The 94.7% coverage from the top 6 may make a partial
  dedup (top-6-only) worthwhile on its own, sidestepping the long tail.
- **What is the true genuine-compile wall-clock** for ~2.5–3.2k variants at the
  best config, and what fixed overhead remains after dedup? Needed to commit to a
  savings number.
- **How stable is `generate_variant_hash()`'s field set** across test churn?
  The exclusion list (`NON_COMPILATION_ARGUMENTS`) is documented but has grown;
  the drift-detection assertion is the guard, but its cost and false-positive
  behaviour on legitimately-new variants need validation.
- **Interaction with the pytest-xdist coordinator:** if dedup cuts item count
  ~150×, does the coordinator-throughput cap (the `--maxschedchunk` / `worksteal`
  finding in the main investigation) stop being the binding constraint, and does
  the optimal `-n` change? Dedup and worker tuning are not independent.
