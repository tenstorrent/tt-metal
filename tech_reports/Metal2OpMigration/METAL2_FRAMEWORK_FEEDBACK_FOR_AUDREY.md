# Metal 2.0 host-API feedback — from a ~28-op TTNN migration sweep

**Author:** Diego's migration run (2026-06-11). **Audience:** Audrey (Metal 2.0 host API / genfiles / ProgramSpec validation owner).
**Scope:** issues that live in **`tt_metal/`** (your area). ttnn-side issues we found are being fixed on our side and are NOT in this doc.
**Context:** ported ~28 TTNN ops from the ProgramDescriptor framework to Metal 2.0 (`create_program_artifacts`/`ProgramSpec`). 13 landed + validated on WH; ~11 are blocked, and **every blocker that isn't ttnn-local traces to one of the 5 items below**. Full per-op detail + metrics: `tech_reports/Metal2OpMigration/BULK_MIGRATION_REPORT.md`. We changed **nothing** in `tt_metal/` — all of this is reported, not patched.

**The migration thesis still holds:** the structural win (keep a per-call value out of the cache key → no recompile) is real and large — sampling went **~1,000,000 µs → ~92 µs/call, 1050 → 1 cache entries** by moving the RNG seed out of the key via `immutable_info_t`. The problem is that **every *other* op that would show that win at scale (rotary-decode, sdpa_decode, paged_cache) is currently blocked by #1 or #2 below.** Fixing those two unblocks the headline result on real LLM ops.

---

## Priority 1 — Conditional `dfb::`/`ta::` token emission (genfiles). **Blocks the most ops, incl. dynamic wins.**

**Type:** design flaw in kernel codegen.
**Symptom:** genfiles emits a `dfb::<name>` / `ta::<name>` token only for a resource that some kernel *binds*. A kernel that uses a resource **conditionally** —
```cpp
if constexpr (use_batch_offset) { auto a = TensorAccessor(ta::batch_offset); ... }
```
— fails to **compile** when the feature is off, because the discarded `if constexpr` branch still needs name resolution and the token was never emitted: `'batch_offset' is not a member of 'dfb'` (we saw 1021 such errors across one op's suite).
**Ops blocked:** `nlp_create_qkv_heads_decode` (optional `batch_offset`), `experimental/paged_cache` fill (scalar-fallback path doesn't bind the `batch_idx` tensor), `nlp_create_qkv_heads` transpose path. Any op with optional-feature kernels.
**Why it matters:** these include paged_cache — a dynamic-arg op that *should* showcase the cache-key win.
**Partial workaround we use:** always-declare the optional DFB so its token exists (works only when the resource is structural, e.g. transpose's padding CB; doesn't help optional *tensors*).
**Proposed fix:** emit guarded tokens unconditionally (declare the full `dfb::`/`ta::` namespace from the spec, independent of which kernels bind them), or expose `#if HAS_dfb_<name>` macros from genfiles so kernels can guard cleanly.

## Priority 2 — `borrowed_from` and ProgramSpec validation disagree about "bound"; compute-only kernels can't bind. **Blocks the whole sharded/compute-only class + rotary-decode win.**

**Type:** flaw (internal inconsistency) + a capability mismatch.
**Two coupled problems:**
1. **`borrowed_from` doesn't register a tensor-parameter user.** The spec fully processes a DFB's `.borrowed_from = TensorParamName{x}` (borrowing the tensor's L1 buffer), but the referential-integrity check at `program_spec.cpp:462` builds `tensor_parameter_users` *only* from kernel `TensorBinding`s — then rejects the same tensor: `TensorParameter 'input0' is defined but not bound by any kernel`. Two parts of the framework hold different definitions of "bound." A borrowed output produced by compute with no second kernel separately trips `program_spec.cpp:377` "DFB has no consumer."
2. **Tensor accessors only JIT in a data-movement build context.** Adding a `TensorBinding` to a *compute* kernel to satisfy (1) then fails the JIT — the accessor codegen pulls `dataflow_api_common.h` / `NOC_INDEX`, absent in a compute build. So a sharded op whose **only** kernel is compute (it fills/produces borrowed L1 views itself, no reader/writer) has **nowhere legal** to bind its tensor parameters.
**Ops blocked:** `rotary_embedding_llama` (decode/sharded — compute-only over borrowed q/cos/sin/trans_mat/out; **this is the rotary-decode dynamic win**: legacy used 10 cache entries across positions, migrated would be 1), `rotary_embedding_llama_fused_qk` (7 borrowed tensors, no DM kernel), `embedding` (tiled sharded-output), `concat` (S2STiled — sharded I/O purely via borrowed CBs).
**Escape hatch (only when a DM kernel exists):** bind the borrowed tensor with a body-unused `TensorBinding` on the DM kernel (we do this in `nlp_concat_heads_decode`). Compute-only factories can't use it.
**Proposed fix:** let a `borrowed_from` DFB reference satisfy the referential-integrity check directly; and/or allow an address-supplying, **no-accessor-codegen** `TensorBinding` on compute kernels (the kernel only needs the base address for the borrowed view, not a NOC accessor).

## Priority 3 — No runtime / variadic / indexed tensor-accessor form. **Blocks binary_ng on its *plain* path.**

**Type:** missing feature.
**Symptom:** `ta::name` is a per-binding compile-time symbol. There is no array/runtime-indexed accessor, no chained `TensorAccessorArgs::next_compile_time_args_offset()` equivalent, and no per-binding base-address / page-size override. Kernels that (a) iterate a runtime-variable set of input tensors, (b) chain two accessors and read a trailing CTA after the chain, or (c) read from `base + offset` with a runtime page-size, have no expression.
**Ops blocked:** `eltwise/binary_ng` — fatal on the **plain interleaved** path (every reader chains two `TensorAccessorArgs`); `concat` (interleaved/multi — variadic N inputs, runtime-indexed accessor); `data_movement/sharded/reshard` (runtime page-stride maps); `interleaved_to_sharded` (compile-time-vararg `shard_builder` addrgen); `slice` RM (runtime base + page-size override).
**Proposed fix:** an indexed/runtime `ta::` form (accessor array addressed by a runtime index) + per-access base-address/page-size overrides; and a compile-time-vararg CTA form for addrgen blocks.

## Priority 4 — Varargs have no per-vararg `enqueue_invariant`. **Blocks the sdpa-decode/sdpa fast path.**

**Type:** missing feature.
**Symptom:** variable-length / per-core-count runtime args must use positional varargs (`num_runtime_varargs`), but `KernelAdvancedOptions::enqueue_invariant_runtime_args` is a list of **names** only — varargs are always re-applied wholesale. So a tree-reduction / mcast op can put its dynamic value (e.g. `cur_pos`) out of the key via `immutable_info_t`, but **cannot** get the "++" enqueue-invariant fast path for its (large, structural) per-core vararg blocks.
**Ops blocked:** `sdpa_decode` (per-core `children_per_round` tree + per-group core-coord lists, different arg counts on active vs idle cores → forces the deprecated `num_runtime_varargs_per_node` path too), `sdpa`.
**Proposed fix:** a per-vararg (or per-vararg-range) `enqueue_invariant` capability, or a named variable-length runtime-arg form that participates in the invariant set.

## Priority 5 — Two validation ergonomics (correct, but rough)

**Type:** usability flaws.
- **A `TensorBinding` forces a `KernelRunArgs` entry even with zero scalar args** — else `ValidateProgramRunArgs` aborts ("non-empty RTA/CRTA schema but no runtime parameters"). The framework already has the tensor info; it could synthesize the entry. (Bit `untilize` single-core.)
- **"Local DFB producer and consumer must cover the same WorkUnitSpec"** is a hard invariant with an opaque message; it bit `typecast` and `embedding` where the reader/writer are naturally all-cores but compute is split per-core-group. (Workable — a KernelSpec may join multiple WorkUnitSpecs — but the error doesn't point you there.)

---

## What's working well (not a complaint — context)

`ValidateProgramSpec` is meaningfully **stricter** than the legacy descriptor path, and that's a feature: it **caught a latent legacy bug** (`UnpackToDestFp32` applied when the input DFB isn't Float32 — a tolerated no-op in legacy) and the slice/`mesh_partition` API coupling. The strictness is good; it just needs the **#2 inconsistency** ironed out so it only rejects things that are genuinely wrong. `borrowed_from`, semaphores (`SemaphoreSpec`/`sem::`), and runtime-indexed page access (`{.page_id = tile_id}`) all work as documented.

## Suggested triage order (by ops unblocked × value)

1. **#1 (conditional tokens)** and **#2 (borrowed/compute binding)** — together they unblock the sharded/compute-only class *and* the dynamic-arg wins (rotary-decode, paged_cache). Highest leverage by far.
2. **#3 (runtime/variadic accessors)** — unblocks binary_ng (a very common op) + concat + reshard.
3. **#4 (vararg invariance)** — required before attention ops (sdpa/sdpa_decode) get the fast path.
4. **#5 ergonomics** — cheap quality-of-life.

---

## Audrey's triage response (2026-06-11) — and our follow-ups

Almost nothing here is a permanent blocker. Status per item:

- **#1 (conditional tokens)** — KNOWN ("found ages ago"); a **recipe issue**, not a new bug. Proper fix needs
  **first-class kernel args**; there's an existing **workaround**. → FOLLOW-UP: get the workaround from
  Audrey / the recipe and apply to the #1-blocked ops (nlp_create_qkv_heads_decode, paged_cache fill,
  nlp_create_qkv_heads transpose) — un-blockable now, no metal wait.
- **#2 (borrowed/compute binding)** — Audrey is FIXING both halves: half 1 (borrowed_from not registering a
  user) **fixed yesterday**; half 2 (compute-only accessor binding) **found last night, PR up shortly**.
  → FOLLOW-UP: after her PR lands + rebase, RE-ATTEMPT the #2 ops — esp. **rotary_embedding_llama decode**
  (the headline dynamic-arg win), plus rotary_fused_qk, embedding sharded, concat S2S.
- **#3 (runtime/variadic accessors)** — Audrey suspects an **invocation difference**, not a gap; she **ported
  binary_ng yesterday** (with assorted Device-2.0 issues). → FOLLOW-UP: get her binary_ng port / correct
  chained-accessor invocation; our agent likely used the wrong one. Re-attempt binary_ng + concat/reshard.
- **#4 (vararg per-arg invariance)** — acknowledged; **easy fix but needs a hack**. → wait before the
  sdpa/sdpa_decode fast path.
- **#5 (validation ergonomics)** — Audrey will address (error-messaging improvements).

**Net:** the "blocked" list is mostly TEMPORARY. Re-attempt order once #1 workaround is in hand + #2 PR lands:
rotary-decode (#2, the win) → the #1 ops → binary_ng (#3 invocation) → sdpa/sdpa_decode (#4).
