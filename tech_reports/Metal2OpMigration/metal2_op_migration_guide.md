# Porting a ttnn op from ProgramDescriptor to Metal 2.0

A practical recipe for migrating a device op from the `ProgramDescriptor` framework to the
Metal 2.0 host API, distilled from the `rand` migration (the first op ported) plus the
performance and correctness lessons learned doing it.

> **Status (2026-06): the framework is still moving.** The canonical Metal 2.0 op-factory
> concepts live in @akertesz's PR **#45961** ("TTNN FactoryConcepts for Metal 2.0"). This guide
> describes the *working* shape used by the `rand` migration and the **timeless** parts —
> the mental model, the performance discipline, and the correctness contract — which hold
> regardless of the final concept names. Where the concept API is in flux, that's called out.

---

## 0. Before you start: is this op even a dispatch problem?

Metal 2.0 and these migrations optimize **host-side dispatch** (the cost of *launching* a
program), not device kernel execution. **Profile first** — if an op is device-bound, none of
this will help it.

Quick diagnostic (no code change):

- Loop the op `N` times on a warm program cache, sync once at the end → **pipelined throughput**.
- Loop the op with a `synchronize_device` *before each call* (empty queue), time the single
  call → **single-call host latency**.
- If `pipelined_throughput > single_call_latency`, the op is **device-bound** (the device can't
  keep up, so throughput = device rate). Stop — fix the kernel, not dispatch.
- If `pipelined_throughput ≤ single_call_latency`, it's **host-dispatch-bound**. Proceed.

Also scale the shape: a host-bound op's per-call time is nearly flat vs tile count; a
device-bound op's grows with it. (Real example: `bernoulli` looked like a 68µs dispatch
regression but was device-bound — the seed/dispatch work was a rounding error.)

---

## 1. Mental model

A Metal 2.0 program is split into two objects:

- **`ProgramSpec` — immutable.** Kernels, dataflow buffers (DFBs), semaphores, tensor
  *parameters*, work units, and the runtime-arg *schema* (names). Built on a **cache miss**.
- **`ProgramRunArgs` — mutable.** The *values* supplied on each enqueue: named runtime args,
  common (broadcast) args, and tensor arguments (addresses).

Reason about a migration along **two orthogonal axes**:

1. **Composition** — *what is built, and when.* Three lifecycle tiers:
   - **immutable** (the spec) — built once on a miss, hashed for the cache key;
   - **static run-args** — values fixed for a cache entry (e.g. per-core work-split scalars):
     set once on the miss, left baked on hits;
   - **dynamic run-args** — values that change per dispatch (RNG seed, `lr`/`step`, tensor
     addresses): re-applied on **every** dispatch (miss and hit).
2. **Identity** — *the cache key.* Derive it from an **`ImmutableInfo`** projection of the op
   args (everything the program structure depends on, and nothing mutable). Hashing that same
   object that *also* feeds the spec makes "the right program is cached" structural rather than
   a hand-maintained invariant — see §5.

The single most important rule falls out of this split: **only the dynamic tier is touched on a
cache hit.** Everything you can keep out of it is free.

---

## 2. ProgramDescriptor → Metal 2.0 mapping

| ProgramDescriptor | Metal 2.0 |
|---|---|
| `create_descriptor` returning `ProgramDescriptor` | `create_program_spec` (immutable) + run-args builders |
| `CBDescriptor` | `DataflowBufferSpec` (producer/consumer endpoints declared on `KernelSpec`) |
| `KernelDescriptor` (positional CT/RT args) | `KernelSpec` (named `runtime_arg_schema`, `compile_time_args`, `dfb_bindings`, `tensor_bindings`) |
| `TensorAccessorArgs(*buf).append_to(ct_args)` + addr in RT args | `TensorParameter` (declared in spec) + `TensorBinding` (on the kernel); framework fills the address |
| `emplace_runtime_args(core, {Buffer*, ...})` | tensor address handled by the `TensorParameter`/binding + `UpdateTensorArgs` on hit |
| `get_dynamic_runtime_args` + `DynamicRuntimeArg` (cache-hit patch) | dynamic-tier run-args re-applied via `ApplyDynamicArgs` (named-arg in-place patch) on hit |
| kernel `get_arg_val<T>(i)` / `get_common_arg_val<T>(i)` / `get_compile_time_arg_val(i)` | kernel `get_arg(args::name)` / `dfb::name` (→ CB id) / `ta::name` (TensorAccessor) |

---

## 3. The recipe (worked on `rand`)

### Step 0 — inventory the descriptor op
List: kernels and their roles; CBs and their producer/consumer relationships; per-core runtime
args; which args are **dynamic** (excluded from the program hash and re-applied each dispatch —
look at `get_dynamic_runtime_args` / the custom `compute_program_hash`); tensor inputs/outputs.

### Step 1 — `create_program_spec` (immutable blueprint)
- **CBs → DFBs.** A producer/consumer CB pair becomes one `DataflowBufferSpec`; the producing
  kernel declares `ProducerOf(dfb, accessor)`, the consumer `ConsumerOf(...)`. Set
  `data_format_metadata` for compute-bound DFBs. *Tip: have the DFB carry the **output** dtype
  and let `pack_tile` convert on the producer — this removes scratch CBs and per-dtype writer
  branches (rand's bf16 path collapsed to a single dtype-agnostic writer).*
- **Kernels → `KernelSpec`.** `source`, `hw_config` (`ComputeHardwareConfig` /
  `DataMovementHardwareConfig`), `dfb_bindings`, `tensor_bindings`, and a `runtime_arg_schema`
  naming the per-node and common args. No positional `TensorAccessorArgs` — declare a
  `TensorParameter` + bind it.
- **Work units.** One `WorkUnitSpec{name, kernels, target_nodes}` over the core set; per-core
  *value* differences live in run-args, not here.

### Step 2 — `ImmutableInfo` + the cache key
Define a small struct of exactly what the spec depends on (e.g. `{output TensorSpec, grid}`),
derivable from attributes. `create_program_spec` (and the static-arg builder) take **only**
this — so a mutable value like `seed` *cannot* leak into the spec or the key. Hash this object
for the cache key; do **not** write a custom `compute_program_hash` (see §5).

### Step 3 — run-args, split static vs dynamic
- **Static** (work-split scalars: `start_id`, `num_tiles`): pure function of `ImmutableInfo`,
  set once on the miss.
- **Dynamic** (seed/range/`lr`/addresses): rebuilt each dispatch and re-applied on hits.
- Tensor addresses are dynamic but auto-handled: declare the `TensorParameter`, and the
  framework matches it to the reflected output/input tensors and re-patches via
  `UpdateTensorArgs`.

### Step 4 — rewrite the kernels to the accessor model
- Compute producing into a DFB: `constexpr uint32_t cb = dfb::name;` then the *unchanged*
  `cb_reserve_back / pack_tile(0, cb, 0) / cb_push_back` pipeline. (`dfb::name` implicitly
  converts to the CB id — `pack_tile` into a DFB works.)
- DM consumer → tensor: `TensorAccessor out(ta::name); DataflowBuffer buf(dfb::name);` then
  `noc_async_write(buf.get_read_ptr(), out.get_noc_addr(page), buf.get_entry_size())`.
- Args by name: `get_arg(args::seed)` (resolves named RTA **or** CRTA).

### Step 5 — cache-hit correctness
The hit path re-applies the dynamic tier via `ApplyDynamicArgs` (named RTAs/CRTAs, in place) +
`UpdateTensorArgs` (addresses). Pin both halves with a regression test (see §5).

### Step 6 — build, test, measure (§4).

---

## 4. Performance playbook (the hard-won part)

These are the lessons that actually moved the numbers.

### 4.1 Do not over-materialize per-core dynamic args — **this is the big one**
A per-core dynamic value that is really `base + f(core_static)` (e.g. RNG `seed + i`) should be
**one broadcast scalar** + an in-kernel derivation, *not* one runtime-arg entry per core.
Building `N` per-core entries (especially `unordered_map` per core, but even a flat vector) on
*every* dispatch is the dominant cache-hit cost for light ops.

Recipe: move the value to a **common (broadcast)** runtime arg; recover per-core distinctness in
the kernel from a value it already has statically (e.g. `seed = base_seed + start_id`).

Measured impact:

| op | before | after | Δ |
|---|---|---|---|
| `rand` (Metal 2.0) | 33.0µs (naïve) | **19.4µs** | −42% — *now faster than the descriptor path (22µs)* |
| `uniform` (descriptor) | 12.7µs | **9.17µs** | −28% |

Note this is op-authoring hygiene, **framework-independent** — it helps the descriptor path too.

### 4.2 Measure the baseline *before* optimizing
The per-core-args win only matters where per-core args are a meaningful *fraction* of dispatch.
`bernoulli` is 68µs and **device-bound** — broadcasting its seed changed nothing (it even drifted
+2µs in noise). Profile, then optimize the actual bottleneck.

### 4.3 Diagnose host vs device first
Use the §0 test. If the op is device-bound, dispatch optimizations are wasted effort.

### 4.4 Measurement methodology
Match the team standard (PR #43840): **n≈2000 iters, ≥500 warmup, fresh process per op, paired
rounds, drop the warm-up outlier, trimmed mean.** Specifically:
- **Warm the allocator** — ops that alloc/free per dispatch keep warming over *thousands* of
  calls; 10 warmup calls is far too few (you'll measure a cold trial). Use ≥5000 warmup for
  alloc-heavy ops and report steady-state.
- Beware **bimodal** host noise (±17% on a single sample); use medians/trimmed means.
- Always `print(ttnn.__file__)` and confirm you're loading the binary you built.

### 4.5 Verify you're measuring the cache-hit path
`device.num_program_cache_entries() == 1` across calls that vary only dynamic values. If it
grows, the dynamic value leaked into the hash (a recompile-per-value bug, not a perf win).

---

## 5. Correctness playbook

### 5.1 The cache-hit dynamic-arg contract
For any value excluded from the program hash (so it cache-hits), pin **both halves** in a test,
mirroring `test_rand_different_seed_values`:
- a **different** dynamic value must **not** add a cache entry (`== 1`) — guards re-adding it to
  the hash (the recompile-per-value hack);
- a **different** dynamic value must **change** the output — guards the *frozen-arg* bug (the hit
  reused the previous dispatch's baked value);
- the **same** value must reproduce the output — guards spurious re-randomization.

### 5.2 Prefer `ImmutableInfo`; avoid custom hash + relaxations
Two independent silent-wrong-answer generators (per #45961):
- **Over-permissive hash** — a custom `compute_program_hash` that omits an attribute affecting
  the program → two different programs collide on one cache entry → wrong program reused.
- **Stale runtime args** — the hit path misses a per-call value → frozen.

`ImmutableInfo` closes the first *by construction* (the key is the same projection that builds
the spec). **Do not** pair a custom shape-agnostic hash with a TensorParameter *relaxation*
(`dynamic_tensor_shape` / `match_padded_shape_only`): the relaxation removes the validation
safety net, so a too-loose key silently miscomputes. Relaxations are safe with the
derived/`ImmutableInfo` key because the key *is* the program identity.

---

## 6. Pitfalls & gaps

- **Generator ops (no input tensor).** The adapter sources the `MeshDevice` from `tensor_args`;
  for ops with none (rand), it must fall back to `tensor_return_value` / `attrs.device`.
- **Slow-path rebuild trap.** If a descriptor op bakes buffer addresses as *raw* runtime args
  (no `Buffer*` binding), it falls to a full descriptor **rebuild on every cache hit** — often
  the real cost. In Metal 2.0, use `TensorParameter`/bindings so addresses are patched, not
  rebuilt. Verify with a `create_descriptor`-call counter: it should fire **once** (miss), not
  per dispatch.
- **bf16 / dtype conversion.** Pack the output dtype in the compute kernel (DFB carries output
  format) so the writer is a single dtype-agnostic copy — no scratch CB, no `OUTPUT_DTYPE_*`
  branches.
- **Per-device values (sharded mesh).** A per-device seed offset is per-coordinate, not
  immutable — keep it in the dynamic tier and thread the dispatch coordinate into the dynamic
  builder. (Multi-device validation needs a T3K-class box.)
- **The caching-strategy enum.** Don't reach for an explicit "minimize-hit-cost vs
  maximize-reuse" knob to get speed; derive the strategy from the static/dynamic tier
  decomposition. The enum is a symptom of bundling spec+run-args, and it leaves the
  dynamic-scalar op class (rand et al.) on the slow option. (Discussion: §1 + #45961.)

---

## 7. Migration checklist

- [ ] Profiled: confirmed the op is **host-dispatch-bound** (§0), not device-bound.
- [ ] `create_program_spec` built from `ImmutableInfo` only; CBs→DFBs; TensorParameters declared.
- [ ] Run-args split **static** (once) vs **dynamic** (every dispatch); per-core values that are
      `base + static` moved to a **broadcast** base + in-kernel derivation.
- [ ] No custom `compute_program_hash`; cache key derived from `ImmutableInfo`.
- [ ] Kernels rewritten to `dfb::` / `ta::` / `args::`; `pack_tile` into the DFB; dtype-agnostic writer.
- [ ] Cache-hit re-apply wired (`ApplyDynamicArgs` + `UpdateTensorArgs`).
- [ ] Regression test pins the dynamic-arg contract (§5.1); op's existing tests pass.
- [ ] Warm measurement (≥5000 warmup, trimmed) vs the descriptor baseline; `cache_entries == 1`.

---

## 8. Reference: the `rand` migration

See `ttnn/cpp/ttnn/operations/rand/device/` on branch `dgomez/metal2-rand-migration`:
`rand_device_operation.hpp` (the tiered factory surface + `immutable_info_t`),
`rand_program_factory.cpp` (spec / static / dynamic builders), and the two kernels. The
framework support (`ApplyDynamicArgs`, the `ProgramSpec` adapter wiring) is in
`tt_metal/.../metal2_host_api/program_run_args.cpp` and `ttnn/api/ttnn/mesh_device_operation_adapter.hpp`.
Result: correct on WH (44/44 tests, genuine cache hits with seed re-applied), **19.4µs warm —
faster than the descriptor path.**
