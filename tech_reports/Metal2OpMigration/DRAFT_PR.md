# [Draft / RFC] rand → Metal 2.0: tiered dynamic-arg re-apply + op-authoring perf wins

> Draft for review. Based on `dgomez/descriptor-static-dynamic-partition` (so the diff is just the
> Metal 2.0 work, not the descriptor migration underneath). Related: #45961 (@akertesz, Metal 2.0
> FactoryConcepts) — this is a first real op on that direction plus a concrete proposal.

## Summary

`rand` is the first ttnn op ported to the Metal 2.0 host API, end-to-end on Wormhole. The
headline: **with correct op authoring, the Metal 2.0 `rand` is *faster* than the descriptor path**
(19.4µs vs 22µs warm host dispatch), and the same op-authoring fix gives **−28% on `uniform`**.
The framework was not the bottleneck — per-core over-materialization in the ops was.

## What's in this PR

1. **Framework support for tiered ProgramSpec factories** (`mesh_device_operation_adapter.hpp`,
   `operation_concepts.hpp`, `program.hpp`, `program_run_args.cpp`):
   - `ApplyDynamicArgs` — in-place re-apply of named RTAs/CRTAs on a cache hit (the named-scalar
     analog of `UpdateTensorArgs`; the Metal 2.0 replacement for the descriptor
     `apply_dynamic_runtime_args` shim).
   - `HasDirectProgramSpec` concept + `DirectProgramSpecFactory` route so a bare
     `create_program_spec` op dispatches without a `program_factory_t`.
   - `ProgramSpec` adapter: device-sourcing fallback to `tensor_return_value` (generator ops with
     no input tensor), three-tier miss path (spec + static + per-coordinate dynamic, merged), and
     the dynamic re-apply on hit. Cache key routed through `extract_immutable_info` when present.
2. **`rand` migrated to Metal 2.0** (`operations/rand/`): three-tier factory
   (`create_program_spec` / `create_static_args` / `create_dynamic_args`) + `immutable_info_t`;
   CBs → one DFB carrying the output dtype; kernels rewritten to `dfb::`/`ta::`/`args::`. Seed is a
   single **broadcast** base; the kernel recovers per-core distinctness as `base + start_id`.
3. **`uniform` op-hygiene win** (`operations/uniform/`): same broadcast-base treatment of the
   per-core seed (+ stop duplicating `from`/`to` across cores) on the existing descriptor path.
4. **Migration guide** (`tech_reports/Metal2OpMigration/`).

## Performance (warm, host dispatch, cache-hit; 5000 warmup + 8×2000 trials, WH B0)

| op | before | after | Δ |
|---|---|---|---|
| `rand` (Metal 2.0) | 33.0µs naïve / 22µs descriptor | **19.4µs** | −42% vs naïve; **beats descriptor** |
| `uniform` (descriptor) | 12.71µs | **9.17µs** | −28% |
| `bernoulli` | 68.6µs | (unchanged) | device-bound — see caveats |

Correctness: `rand` 44/44, `uniform` 101/101, `bernoulli` 38/38.

## The design proposal (for @akertesz / #45961)

This op validates the #45961 direction and suggests two refinements:

1. **Adopt `ImmutableInfo` as the cache key** (done here): the key is the *same* projection that
   feeds `create_program_spec`, so a mutable value (seed) cannot leak into the spec or the key —
   the "right program is cached" invariant becomes structural, and it's a *cheap* typed key (the
   safe version of a custom hash). No custom `compute_program_hash`.
2. **Add a `static` run-arg tier** distinct from `dynamic`: the cache-hit path should re-apply
   *only* the dynamic tier. For multi-core dynamic-arg ops, re-applying the static work-split
   scalars every hit is pure waste. With the tiers explicit, the **caching-strategy enum can be
   derived rather than selected** — the framework knows what varies.

And one op-authoring rule that matters more than any framework choice: **don't materialize a
per-core dynamic value when it's `base + f(core_static)`** — broadcast the base and derive
per-core in the kernel. That single change is what makes Metal 2.0 `rand` beat descriptor.

## Caveats / not done

- **`bernoulli` is device-bound** (~68µs, ~52µs on-device), so dispatch-side work doesn't help it;
  its seed broadcast was a no-op and was reverted. Separate kernel-perf investigation.
- **Multi-device per-device-seed path is wired but unverified** (needs a T3K-class box; the
  single-device mesh tests pass, the multi-device ones skip here).
- **Concept API is in flux** — the exact factory method names should converge with #45961. The
  static/dynamic tier + enum-removal is a proposal, not settled.
- `ApplyDynamicArgs` replaces the descriptor `program_descriptor_patching` shim *for this path*;
  the shim isn't deleted yet (needs a second op on the path to prove generality).

## Test plan

- [x] `pytest tests/ttnn/nightly/unit_tests/operations/rand/test_rand.py` (44 pass; incl.
      `test_rand_different_seed_values` — genuine cache hit, seed re-applied, not stale).
- [x] `test_uniform.py` (101), `test_bernoulli.py` (38).
- [ ] Multi-device mesh-shard tests on T3K (per-device seed).
- [ ] Named CI workflows (runtime-sanity, blackhole-post-commit) — to dispatch on the branch.
