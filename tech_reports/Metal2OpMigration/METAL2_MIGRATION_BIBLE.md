# The Metal 2.0 Op-Migration Bible (for agents)

> Authoritative, agent-actionable recipe for porting a TTNN op to the Metal 2.0 host API.
> It encodes both **the sanctioned model** (Audrey Kertesz's `Metal2FactoryConcept`, PR #45961 +
> the `akertesz/metal2-documentation` recipes) and the **hard lessons** learned porting `rand`.
>
> If you read nothing else, read §0 and §1. They are the difference between a correct port and a
> plausible-looking one that silently regresses.

---

## 0. The non-negotiable golden rules

These are ranked. When two rules seem to conflict, the lower number wins.

1. **A migration is a *mechanical, behavior-preserving* port.** You change *how* the op expresses
   its program (the host API) and *how* kernels read their args/buffers (the accessor namespaces).
   You do **not** change *what* the op computes, *what* values it produces, or *which* dtypes it
   supports. If the ported op is not byte-for-byte equivalent to the original, you have a bug, not a
   migration.
2. **Do not rewrite kernels.** Port the kernel's *access mechanism* (positional `get_arg_val(i)` /
   `get_compile_time_arg_val(i)` → `args::`/`dfb::`/`ta::`); keep its *logic* — every `#ifdef`,
   every guard, every numeric path (e.g. a manual `fp32→bf16` truncation) — exactly as it was. See
   §8 for the allowed-changes whitelist.
3. **Do not invent.** No clever in-place tricks, no new buffer schemes, no new framework primitives.
   If the original used a scratch buffer, reproduce a scratch buffer (§8.3) — don't substitute an
   in-place hack. Inventions are where silent corruption lives.
4. **Do not touch `tt_metal/`.** The op port lives entirely in `ttnn/cpp/ttnn/operations/<op>/`.
   Adding host-API primitives (a `SkipValidation` flag, an `ApplyDynamicArgs`, etc.) is framework
   work owned by the Metal 2.0 team, not part of an op port. If you think you need one, you've
   reached an Advanced-concept case (§7) — **stop and report**, don't build it.
5. **Do not hand-roll the adapter or the concept.** Wire the op through the standard
   `program_factory_t` variant (§3). The framework auto-detects your factory by concept. If you find
   yourself editing `mesh_device_operation_adapter.hpp` or `operation_concepts.hpp`, you are off the
   path.
6. **Delete the custom `compute_program_hash`.** Metal 2.0 factories forbid it (the adapter
   `static_assert`s against it). The default key is correct-by-construction (§5). Do not patch it,
   do not keep it "just in case."
7. **Do not optimize during the port.** The port's job is *correct and equivalent*, not *fast*. The
   per-dispatch performance tier (Advanced) is **stubbed/unimplemented** today. Premature
   op-authoring "optimizations" (e.g. broadcasting a per-core arg) have repeatedly failed to
   reproduce under clean measurement — see §10. Get it correct; measure later; never claim a speedup
   you didn't measure with the discipline in §9.4.
8. **Verify against the original, not against your intuition.** Same test results, same output
   (including pre-existing quirks — see §9.3). A test passing is necessary, not sufficient.

---

## 1. Mental model

Metal 2.0 splits the old monolithic program factory into two values:

- **`ProgramSpec` — immutable.** Everything that defines the *structure* of the program: kernels
  (source, compile-time args, runtime-arg *schema* = the names), dataflow buffers (DFBs, the
  successor to CircularBuffers), semaphores, tensor *parameters* (declared layouts), and work units
  (which kernels run on which nodes). It is built **once per cache entry**.
- **`ProgramRunArgs` — mutable.** The per-dispatch *values*: runtime-arg values (per-node and
  common/broadcast), and the actual tensors bound to the tensor parameters. These can change every
  call.

Your factory returns both, bundled, and **the framework does the rest** — it builds the `Program`
from the spec (`MakeProgramFromSpec`) and applies the values (`SetProgramRunArgs`). **Your factory
never constructs the `Program` and never calls `SetProgramRunArgs` itself.** That is the single most
common conceptual error.

Kernels stop using positional slots. They read through **binding namespaces** generated from the
spec:
- `dfb::<name>` — a dataflow buffer (use as a CB id with the existing `cb_*` API).
- `ta::<name>` — a `TensorAccessor` for a bound tensor.
- `args::<name>` / `get_arg(args::x)` — a named runtime arg; `get_common_arg_val<T>(i)` for common
  (broadcast) args.
- `sem::<name>` — a semaphore.

---

## 2. The concept you will use: `ProgramSpecFactoryConcept`

There are four factory concepts. **You will almost always use the first.**

| Concept | Method(s) | Status |
|---|---|---|
| **`ProgramSpecFactoryConcept`** (Basic, single-program) | `create_program_artifacts` | **Implemented — use this** |
| `WorkloadSpecFactoryConcept` (Basic, multi-program) | `create_workload_artifacts` | Implemented (multi-program / global semaphores) |
| `AdvancedProgramSpecFactoryConcept` | `extract_immutable_info` + `create_program_spec` + `create_program_run_args` | **STUBBED — `static_assert("not yet supported")`** |
| `AdvancedWorkloadSpecFactoryConcept` | workload variants | **STUBBED** |

The Basic concept is a pure shape check — a factory struct with one static method:

```cpp
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& attributes,
        const tensor_args_t&          tensor_args,
        tensor_return_value_t&        tensor_return_value);
};
```

Return type (`ttnn/api/ttnn/metal2_artifacts.hpp`):

```cpp
struct ProgramArtifacts {
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;  // empty unless the op allocates scratch
    tt::tt_metal::experimental::ProgramSpec     spec;
    tt::tt_metal::experimental::ProgramRunArgs  run_args;
};
```

> **The Advanced concept is the performance tier** (immutable extraction per-dispatch, spec
> construction miss-only, dynamic-only re-apply). It is the right home for the "cache the static
> args, re-apply only the dynamic subset" optimization. **It is not implemented.** An op that
> *requires* it for correctness is a **RED gate — blocked on framework**, not a porter task (§7).
> An op that would merely *benefit* from it for speed ports on Basic and notes the gap in its report.

---

## 3. Wiring (how the framework finds your factory)

On the device-operation struct:

```cpp
struct MyDeviceOperation {
    using operation_attributes_t = ...;
    using tensor_args_t          = ...;
    using spec_return_value_t    = ...;
    using tensor_return_value_t  = ...;

    using program_factory_t = std::variant<MyProgramFactory>;   // REQUIRED — the variant is how the framework detects metal2
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return MyProgramFactory{};
    }

    static void                 validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t   compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // ❌ DO NOT define compute_program_hash — the adapter static_asserts against it (Rule 6).
};
```

The framework `std::visit`s the variant; the metal2 overload fires for any alternative satisfying
`Metal2FactoryConcept` and routes to the correct adapter automatically. You write **zero** adapter
code. If a build fails with `AllFactoriesValid` `static_assert`, you have a stale `cached_program_t`
/ leftover legacy factory member alongside the new one — delete it.

---

## 4. The migration is mechanical — what that means concretely

The port is a **transliteration** with a fixed vocabulary. Allowed transformations:

| Legacy | Metal 2.0 |
|---|---|
| `CBDescriptor` / `.cbs` / `CBFormatDescriptor` | `DataflowBufferSpec` (`entry_size`, `num_entries`, `data_format_metadata`, `tile_format_metadata`) |
| `KernelDescriptor` + `.runtime_args` (positional) | `KernelSpec` + `runtime_arg_schema` (names) + `ProgramRunArgs` (values) |
| `get_compile_time_arg_val(i)` for a CB id | `dfb::<name>` |
| `TensorAccessorArgs<N>()` + `TensorAccessor(args, addr)` | `ta::<name>` via a `TensorBinding` |
| `get_arg_val<uint32_t>(i)` (positional RTA) | `get_arg(args::<name>)` |
| (broadcast value duplicated per core) | common runtime arg, `get_common_arg_val<T>(i)` |
| `KernelDescriptor::defines` (`#ifdef`) | `KernelSpec::compiler_options.defines` (same `#ifdef`) |
| custom `compute_program_hash` | **delete** (default spec-hash is correct) |
| manual cache-hit arg patching / `apply_dynamic_runtime_args` | **delete** — the framework re-applies on hit (§5) |

Everything else stays. You do not rename variables, retune `TT_FATAL`s, reorder logic, or "clean
up" the factory beyond this table. The kernel's computation is preserved exactly (§8).

---

## 5. Dynamic runtime args (RNG seed, `[from,to)`, optimizer `lr`/`step`) — the proper way

This is the case everyone gets wrong. The sanctioned answer is **simple and requires no custom
hash and no framework changes**:

> **Put the per-dispatch-varying value in `ProgramRunArgs` (as a per-node RTA or a common arg), use
> the default `MaximizeCacheReuse`, and delete any custom hash.**

Why it is correct by construction:

- Under `MaximizeCacheReuse` (the default), **the cache key is a hash of the immutable
  `ProgramSpec`** — *not* the op attributes. A seed lives in `ProgramRunArgs`, not in the spec, so it
  **does not affect the key**. Two calls differing only in seed hit the same cache entry.
- On a cache **hit**, the framework **re-runs `create_program_artifacts`** to get fresh artifacts and
  **re-applies the full `ProgramRunArgs`** via `SetProgramRunArgs`. So the new seed flows through
  every dispatch. Nothing carries stale state. This is the structural replacement for the old
  custom-hash-excludes-seed + manual-patching dance.

**Do not** reach for `MinimizeCacheHitCost` to carry a seed: that strategy *skips the factory on a
hit* (it only refreshes tensor addresses), so a per-dispatch seed set at miss-time goes **stale** —
silent wrong numerics. It also forbids common runtime args outright. `MinimizeCacheHitCost` is for
op-owned-resource ops, not dynamic-scalar ops (§7).

**Soundness invariant (memorize this):** under `MaximizeCacheReuse` everything is rebuilt each hit so
you're always safe. The danger only appears if you later move to a hit-path that *reuses* miss-time
state (the Advanced tier, or `MinimizeCacheHitCost`): there, **any value reused on a hit must be a
pure function of the cache key.** A seed is not — so it must always be in the re-applied set. The
work-split (`start_id`/`num_tiles`) *is* a function of the key (grid + padded volume), so it may be
baked. The clean way to enforce this is to feed the static builders only the immutable projection,
so the compiler refuses to let a mutable value leak into baked state.

---

## 6. Step-by-step recipe

### Step 1 — Audit the legacy op (write it down before touching code)

Inventory, with `file:line`:
- **Kernels**: each source file, its role (reader/compute/writer), its compile-time args, its
  positional runtime args (which are static vs which change per dispatch), its `#ifdef`s/defines.
- **CBs**: index, data format, page size, total size; which are real data buffers vs **scratch**
  (written and drained locally by one kernel).
- **Semaphores**, **tensor accessors** (`TensorAccessorArgs`), **work split** (how cores/tiles are
  divided).
- **Custom `compute_program_hash`** — note it; you will delete it (mine it first for relaxation
  hints, then delete — see Audrey's audit recipe).
- **Op-owned device resources** — does the factory *allocate* tensors/semaphores itself (scratch,
  lookup tables)? (Determines caching strategy.)
- **Dynamic args** — values that change every call and were excluded from the hash (seed, range,
  lr/step).
- **UNSUPPORTED constructs** — per-execution CB size updates (`UpdateCircularBufferTotalSize` etc.),
  raw `Buffer*`-through-RTA address arithmetic (silently wrong on the fast path), multi-program
  shapes that exist only to carry op-owned tensors. Flag these; some are GATEs.

### Step 2 — Pick the concept + caching strategy (decision tree)

1. **Multiple distinct programs per mesh coordinate, or global semaphores?** →
   `WorkloadSpecFactoryConcept` (`create_workload_artifacts`). Otherwise single-program → Basic
   `ProgramSpecFactoryConcept`. (Heads-up: a legacy `MeshWorkload` used *only* to carry op-owned
   tensors collapses to single-program with `op_owned_tensors` — do not transliterate the workload.)
2. **Does the factory allocate op-owned device resources?** → if yes, you **must** declare
   `static constexpr auto caching_strategy = ProgramCachingStrategy::MinimizeCacheHitCost;` (default
   would re-allocate them every hit → freed-then-reused memory → runtime `TT_FATAL`). But
   `MinimizeCacheHitCost` **forbids common runtime args and DFB size overrides**; if you need those,
   the Basic path is closed → Advanced → **RED gate**.
3. **Tensor-arg relaxations** (dynamic shape) → default is strict; opt into
   `dynamic_tensor_shape`/`match_padded_shape_only` only with explicit sign-off.
4. **Heavy immutable-extraction work that re-running every dispatch is unacceptable for** →
   Advanced concept → **STUBBED → RED gate, stop and report.** The bar is "cannot be ported
   correctly without it," **not** "would be faster." Default: assume Basic suffices.

For the overwhelming common case (no op-owned resources): **Basic + default `MaximizeCacheReuse`,
nothing to declare.**

### Step 3 — Build the `ProgramSpec`

- One `DataflowBufferSpec` per legacy CB. Copy `entry_size`/`num_entries`/`data_format_metadata`
  (and `tile_format_metadata` if the legacy CB set it). **`entry_size`/`num_entries` are set once at
  spec construction** — compute from input shapes / attributes. No placement field (derived from
  bindings). **Every DFB must have exactly one producer and one consumer binding** (the framework
  `TT_FATAL`s "DFB has no consumer" otherwise) — see §8.3 for the scratch-buffer consequence.
- One `KernelSpec` per kernel: `source`, `compiler_options` (incl. `defines` for `#ifdef` paths),
  `dfb_bindings` (`ProducerOf`/`ConsumerOf`), `tensor_bindings` (`TensorBinding`), `runtime_arg_schema`
  (the **names**, split into per-node `runtime_arg_names` and broadcast `common_runtime_arg_names`),
  `hw_config` (Compute vs DataMovement). Designated initializers must follow struct field order
  (`compiler_options` comes before `dfb_bindings`).
- `tensor_parameters` (declared `TensorSpec` per bound tensor), `work_units` (kernels × target nodes).

### Step 4 — Build the `ProgramRunArgs`

- Per-kernel `KernelRunArgs`: per-node `runtime_arg_values` (`{node, {{"name", value}, ...}}`) and
  `common_runtime_arg_values`. Put the **dynamic** values (seed/range) here.
- `tensor_args`: `{tensor_parameter_name, std::cref(output.mesh_tensor())}` for each bound tensor.
- Return `ProgramArtifacts{.spec = std::move(spec), .run_args = std::move(run_args)}`.
  **Do not call `MakeProgramFromSpec` or `SetProgramRunArgs`.**

### Step 5 — Port the kernels (mechanical; preserve logic — §8)

### Step 6 — Delete dropped plumbing

Custom `compute_program_hash`, any `apply_dynamic_runtime_args`/manual patching, `TensorAccessorArgs`
compile-time payloads, all `CBDescriptor`/`.cbs` references, the legacy `program_factory_t`
`cached_program_t`/`create_program` members. No CB API or positional-arg bookkeeping survives.

---

## 7. Caching strategy reference

- **`MaximizeCacheReuse` (default, declare nothing).** Key = spec hash. Hit = re-run factory +
  full `SetProgramRunArgs`. Correct for any op including dynamic-scalar ops. `op_owned_tensors` must
  be empty. This is your default.
- **`MinimizeCacheHitCost` (opt-in `static constexpr`).** Required iff the factory allocates
  `op_owned_tensors`. Key = op-args hash. Hit = factory **skipped**, only `UpdateTensorArgs`.
  Forbids common runtime args and DFB size overrides. **Cannot carry a per-dispatch seed.**
- **Advanced (`extract_immutable_info`/`create_program_spec`/`create_program_run_args`).** Stubbed;
  `static_assert("not yet supported")`. The performance tier. Needing it = RED gate.

---

## 8. Kernel-side porting

### 8.1 The whitelist (the *only* changes you may make to a kernel)

1. CB id from `get_compile_time_arg_val(i)` → `dfb::<name>`.
2. Tensor access via `TensorAccessorArgs<N>` + `TensorAccessor(args, addr)` → `TensorAccessor(ta::<name>)`.
3. Positional `get_arg_val<uint32_t>(i)` → `get_arg(args::<name>)`; broadcast values →
   `get_common_arg_val<T>(i)`.
4. Nothing else. **Not** the compute math, **not** the `#ifdef` structure, **not** the loop bounds,
   **not** a `TT_FATAL`, **not** a numeric narrowing path, **not** variable names beyond the above.

Compute kernels typically need only change (1) and (3): `constexpr uint32_t cb = dfb::<name>;` then
the existing `cb_reserve_back`/`pack_tile`/`cb_push_back` pipeline is **unchanged**.

### 8.2 Varargs

Use `get_vararg(i)` only when arguments are *genuinely* dynamic in kernel code (retrieved in a loop
with a runtime index). When each arg is referenced by a constant index, use the named form. Report
any retained varargs.

### 8.3 Scratch buffers — the DFB consumer trap

A legacy "scratch CB" (one kernel writes it, the NoC drains it, no second kernel reads it) **cannot
be a producer-only DFB** — Metal 2.0 `TT_FATAL`s `DFB '<x>' has no consumer`. Reproduce it by
**binding the same kernel as both `ProducerOf` and `ConsumerOf`** the scratch DFB (the validation
only requires both endpoints non-empty); the kernel uses it as a fixed scratch slot exactly as the
legacy CB did. **Do not** substitute an in-place narrowing into another buffer — that path has a
real cache/visibility corruption bug (it produced wrong values on a fraction of tiles). Reproduce
the original's buffer structure; don't invent a new one. (Concretely, for `rand`'s `fp32→bf16`
truncation: keep the fp32 intermediate DFB + the output-dtype scratch DFB and the original
high-16-bits truncation loop, verbatim.)

### 8.4 Preserve dtype guards and behavior

If the legacy factory did `switch(dtype){ case A:...; case B:...; default: TT_THROW; }` to gate
supported dtypes, reproduce it (the `defines` switch + the `TT_THROW`). Removing it silently
"supports" dtypes that then produce garbage the distribution tests won't catch.

---

## 9. Verification

### 9.1 Build gotchas
- `AllFactoriesValid` static_assert → leftover legacy factory member; delete it.
- Designated-initializer order errors → reorder to match struct field declaration order.
- `log_info`/`log_debug` are macros — call `log_info(tt::LogMetal, ...)`, never `tt::log_info(...)`.

### 9.2 Kernel cache is sticky
JIT kernels are cached at `~/.cache/tt-metal-cache` (NOT `$TT_METAL_HOME/built`). When validating a
kernel change, **`rm -rf ~/.cache/tt-metal-cache`** (or set a fresh `TT_METAL_CACHE`) — otherwise you
measure a stale kernel and chase ghosts. A clean host build is `./build_metal.sh --clean` **followed
by** `./build_metal.sh` (`--clean` only removes artifacts; it does not rebuild).

### 9.3 Correctness = equivalence to the original, including quirks
- Run the op's full pytest suite; the pass/xfail/xpass counts must match the pre-port baseline
  exactly.
- Diff actual output against the **original op** for the same inputs (build the original, dump, compare).
  Match it **including pre-existing oddities** — e.g. `rand` bf16 produces a small number of exactly
  `1.0` values (boundary of `[0,1)`); the original does this too, so the faithful port must reproduce
  it, *not* "fix" it. Errors that exist in the original are not yours to fix in a migration
  (Rule 1, Rule 8).
- Explicitly test the **cache-hit dynamic path**: two calls with different seeds must produce
  different outputs (proves the seed is re-applied, not stale).

### 9.4 Performance discipline (if you measure at all)
- Verify the loaded binary (`print(ttnn.__file__)`); wipe the kernel cache; warm up (thousands of
  calls) before timing; take **hundreds** of batch-mean samples and report the median + percentiles
  (watch for bimodality). A single run is not a measurement.
- Separate host-dispatch from device-bound: pipelined throughput vs single-call-after-sync. If
  pipelined > single, the op is device-bound and host-side work is irrelevant.
- **Do not claim a speedup you have not measured this way.** Multiple "optimizations" on this path
  (broadcasting per-core args; the `ApplyDynamicArgs` "42% win") **did not reproduce** under clean
  measurement. Metal 2.0 Basic is at *parity-to-slower* vs the legacy descriptor path on the
  cache-hit path (the per-core arg representation is the cost; the fix is the Advanced tier, which is
  stubbed). The honest expectation for a Basic port is: **correct, possibly slower on warm dispatch;
  not faster.**

---

## 10. Performance reality (so you don't chase it during a port)

Measured on `rand` (WH B0, 512×512 bf16, clean builds, cache wiped, 500 batch-means):
- Legacy descriptor: ~40µs.
- Metal 2.0 Basic, full re-apply (`MaximizeCacheReuse`): ~178µs (4.5×).
- The dominant cost is **rebuilding per-core `unordered_map<string,uint32_t>` every hit**
  (`create_static`-equivalent ~47µs + dynamic ~23µs + merge ~21µs ≈ 91µs). Validation is ~13µs (the
  *smallest* stage — bypassing legality checks is **not** the fix). Descriptor is fast because it uses
  flat `vector<uint32_t>` + positional indices.
- The real fix is the **Advanced tier**: spec built miss-only, static run-args baked miss-only,
  only the dynamic subset re-applied on a hit, via a flat/precomputed-index layout. **It is not
  implemented.** So: port on Basic, accept current warm-dispatch cost, record the perf gap in the
  report, and do **not** smuggle a perf hack into the op or `tt_metal`.

---

## 11. Anti-pattern catalog (every one of these has actually bitten)

- ❌ Rewriting a kernel "more cleanly" → dropped `#ifdef`s and a `fp32→bf16` rounding bug that the
  distribution test didn't catch.
- ❌ Inventing an in-place narrowing because Metal 2.0 lacks a scratch buffer → cache/visibility
  corruption on a fraction of tiles. (Fix: dual-bind the scratch DFB; reproduce the original buffer.)
- ❌ Adding a `tt_metal` primitive (`ApplyDynamicArgs`) to make hits cheap → framework scope, not op
  scope; and it doesn't reach parity anyway.
- ❌ Hand-rolling a `DirectProgramSpecFactory` + a custom concept + a 3-tier split → all of that is
  the (stubbed) Advanced concept's job; on Basic you write one `create_program_artifacts`.
- ❌ Keeping/patching a custom `compute_program_hash` → forbidden; silent `UpdateTensorArgs` legality
  failures on hits.
- ❌ Using `MinimizeCacheHitCost` to carry a seed → stale seed on hits.
- ❌ Claiming a perf win without clean measurement → "42% faster / beats descriptor" evaporated; the
  real number was parity-to-slower.
- ❌ "Fixing" a pre-existing quirk during a port → scope creep + behavior change.
- ❌ Over-reverting: when asked to undo *one* change, undo exactly that one — don't delete the whole
  migration.
- ❌ Leaving a **ProgramDescriptor** op (the pre-migration form, or any op that isn't migrated yet)
  with a raw `tensor.buffer()->address()` baked into its kernel runtime args instead of
  `kd.emplace_runtime_args(core, {buffer, ...})` → no patchable `BufferBinding`, so the adapter
  re-runs `create_descriptor()` on **every program-cache hit**. This is a non-trace-only host
  regression that is **invisible to trace** (captured once) **and to `num_program_cache_entries`**
  (the outer cache still hits) — it was issue **#46506** (ResNet50 ~20× non-trace). Detect it with
  `TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1` (the adapter `TT_FATAL`s and names the op when
  it rebuilds on a hit). Fix = pass the `Buffer*` (per-core inline `emplace_runtime_args`, or
  `kd.buffer_bindings.push_back({core, arg_idx, buf})` / `kd.common_buffer_bindings.push_back({arg_idx, buf})`
  for manually-built arg vectors). Gotcha: a literal `0` in an `emplace_runtime_args` brace-list must
  be `0u` (bare `0` is a null-pointer-constant, ambiguous with `Buffer*`); and only bind a buffer that
  is an input `tensor_args`/output tensor (`resolve_bindings` `TT_FATAL`s on intermediates). Args that
  are *address-derived* (`addr + offset`, `out_addr - in_addr`) or CB-bound sharded ops need
  `get_dynamic_runtime_args` (§5) instead of a static binding.
- ❌ Validating a descriptor-framework change on a branch where the op is **already migrated to a
  `ProgramSpec`** → the op routes through the spec path (which binds tensors natively and fast-paths),
  so the descriptor version that's still on `main` looks fine while it actually slow-path-rebuilds.
  Test descriptor-framework fixes on a **`main`-based branch**, and run the guard env var above. (This
  bit the #46506 fix: `transpose`'s descriptor factory regressed on `main` but passed on the
  spec-migrated branch.)

---

## 12. Pre-PR self-audit checklist

- [ ] Op satisfies exactly one `Metal2FactoryConcept` leaf (Basic unless genuinely multi-program).
- [ ] One `create_program_artifacts`; **no** `MakeProgramFromSpec`/`SetProgramRunArgs` in the factory.
- [ ] `program_factory_t` variant + `select_program_factory` wired; **no** hand-rolled adapter/concept.
- [ ] Custom `compute_program_hash` deleted.
- [ ] No edits under `tt_metal/`. No new framework primitives.
- [ ] Caching strategy: default `MaximizeCacheReuse` unless `op_owned_tensors` (then
      `MinimizeCacheHitCost` + no CRTAs/size-overrides).
- [ ] Dynamic values (seed/range) live in `run_args`; correctness verified across a cache hit.
- [ ] Kernels changed only per the §8.1 whitelist; all `#ifdef`s, guards, numeric paths preserved.
- [ ] Every DFB has a producer **and** a consumer.
- [ ] Output is byte-for-byte equivalent to the original (incl. pre-existing quirks); full test
      suite matches the baseline counts.
- [ ] Kernel cache wiped before the validating run.
- [ ] No unmeasured perf claims; perf gap (if any) recorded for the Advanced-tier roadmap.
- [ ] (ProgramDescriptor ops, not yet spec-migrated) No raw `buffer->address()` in kernel runtime
      args — tensor addresses go through `emplace_runtime_args(Buffer*)` / `buffer_bindings`; verified
      under `TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1` on a `main`-based branch.

---

### References
- Concepts: `ttnn/api/ttnn/operation_concepts.hpp` (`@ akertesz/ttnn-metal2concept-improvements`, PR #45961).
- `ProgramArtifacts`: `ttnn/api/ttnn/metal2_artifacts.hpp`.
- Host API: `tt_metal/api/tt-metalium/experimental/metal2_host_api/{program_spec,program_run_args,kernel_spec,dataflow_buffer_spec,tensor_parameter}.hpp`.
- Audrey's recipes (`@ akertesz/metal2-documentation`):
  `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/{metal2_migration_guide,port_op_to_metal2_audit,port_op_to_metal2_recipe,port_op_to_metal2_ttnn_factory,metal2_port_patterns}.md`.
- E2E host-API usage: `tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec_hw.cpp`.
