# Metal 2.0 Op Port — Patterns & Anti-Patterns Catalog

This catalog accumulates patterns and anti-patterns observed during Metal 2.0 op ports. It is referenced from the [feasibility audit](port_op_to_metal2_audit.md) for yellow-tier override guidance, from the [port recipe's planning step](port_op_to_metal2_recipe.md#plan-the-spec) for structural-decision reference, and from the recipe's verification step for the anti-pattern self-audit checklist.

Each entry is self-contained. New entries land here as they're discovered during ports.

## Conventions

Entry shape:
- **Category**: `Pattern` (an affordance to reach for) | `Anti-pattern` (a shape to avoid) | `Caution` (a judgment call with consequences).
- **Recognition signal**: what to look for in legacy code (or proposed port code).
- **Decision** (Pattern / Caution) or **Why wrong** (Anti-pattern): the prescribed move and its rationale.
- **Correct port**: code or prose showing the right pattern.
- **Prerequisite**: any Metal 2.0 framework commit / PR / version below which the pattern doesn't apply.
- **See also**: cross-references.

## Entries

### Patterns

- [Self-loop DFB binding (producer == consumer)](#pattern-self-loop-dfb-binding)
- [Conditional / optional DFB bindings](#pattern-conditional--optional-dfb-bindings)
- [Pass DFB handles directly to LLKs and kernel-lib helpers](#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
- [Multi-variant factories](#pattern-multi-variant-factories)
- [Unity-build hygiene for anonymous-namespace symbols](#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)

### Anti-patterns

- [Demoting per-group CTA to RTA](#anti-pattern-demoting-per-group-cta-to-rta)
- [`.id` extraction or temp DFB wrappers at LLK call sites](#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites)

### Cautions

- [Avoid varargs unless absolutely necessary](#caution-avoid-varargs-unless-absolutely-necessary)
- [Modifying a shared dataflow kernel](#caution-modifying-a-shared-dataflow-kernel)

---

## Pattern: Self-loop DFB binding

**Category**: Pattern

**Recognition signal**: A compute kernel that both produces and consumes the same buffer. Common in accumulator patterns where the kernel writes a partial result, reads it back, and updates it across iterations.

**Decision**: Declare two `DFBBinding`s on the same `KernelSpec` for the same DFB — one with `endpoint_type = PRODUCER`, one with `endpoint_type = CONSUMER`. The two bindings may share a single `local_accessor_name` (yielding one device-side `dfb::name` handle), or use two distinct names (yielding two device-side handles aliasing the same DFB).

**Correct port**:

```cpp
// Shared-name form (most natural for self-loop):
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::PRODUCER);
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::CONSUMER);

// Kernel side:
experimental::DataflowBuffer cb_acc(dfb::acc);  // one wrapper drives both directions
```

The two-distinct-names form (`acc_w` for PRODUCER, `acc_r` for CONSUMER, yielding `dfb::acc_w` and `dfb::acc_r` aliasing the same DFB) also works.

**Prerequisite** (shared-name form only): per-kernel accessor-name dedup relaxation, commit `332413412af` on `akertesz/misc-op-port-fixes`. The two-distinct-names form has always worked.

**See also**: [Anti-pattern: Demoting per-group CTA to RTA](#anti-pattern-demoting-per-group-cta-to-rta) (separate pattern, but the same "host-side variation in `KernelSpec`" mental model).

---

## Pattern: Conditional / optional DFB bindings

**Category**: Pattern

**Recognition signal**: A DFB is used by a kernel only on some code paths — e.g., a `cb_scaled` buffer used only when `do_scale = true`, or a `cb_fusion` used only when `FUSE_PRE_ADD`. The configuration is known at host time.

**Why this is hard.** Metal 2.0's `dfb::<name>` namespace is generated from the actual host bindings: if the host omits SCALED from `dfb_bindings`, `dfb::cb_scaled` is not declared in `kernel_bindings_generated.h`. C++ `if constexpr` in a non-template function (which `kernel_main()` is) still performs name lookup on the discarded branch — so `if constexpr (false) { ... dfb::cb_scaled ... }` fails to compile at parse time even though the branch is dead at codegen. Two interim patterns work around this; choose by L1 budget and kernel structure. Both are bridges until NTTP-ified CTAs let the entire kernel body be template-instantiated, at which point `if constexpr` discarded branches truly *do* skip non-dependent name lookup.

### Option A — Conditional host binding + `#ifdef` in kernel

**Use when:**
- L1 budget is tight, and paying ~1 tile/core per conditionally-unused DFB (Option B) is unaffordable. Layernorm and similar are this case.
- The kernel references the conditionally-bound DFB name from a **file-scope context** — ternary expressions like `(do_gamma | do_beta) ? cb_fusion : cb_out`, struct-scope `constexpr` expressions, or other non-`if constexpr` name-lookup contexts. Both ternary operands resolve at parse time regardless of which branch the constant condition selects, so the conditionally-bound name must either exist (Option B) or be `#ifdef`-gated away (Option A).

**Mechanism.** Conditionally bind the DFB on the host. Emit a matching preprocessor define when the binding is present. `#ifdef`-gate the kernel-side constexpr alias of the DFB name, and `#ifdef`-gate all file-scope expressions that reference it.

```cpp
// Host:
const bool fuse_pre_add = ...;
KernelSpec compute{
    .compile_time_arg_bindings = { /* ... */ },
    .defines = fuse_pre_add
        ? std::map<std::string, std::string>{{"FUSE_PRE_ADD", "1"}}
        : std::map<std::string, std::string>{},
    .dfb_bindings = fuse_pre_add
        ? std::vector<DFBBinding>{INPUT, OUTPUT, FUSION}
        : std::vector<DFBBinding>{INPUT, OUTPUT},
};

// Kernel:
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_fusion = dfb::cb_fusion;
#endif

// File-scope expression using the conditional name:
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_x = (do_gamma | do_beta) ? cb_fusion : cb_out;
#else
constexpr uint32_t cb_x = cb_out;
#endif
```

### Option B — Unconditional host binding + `if constexpr` in kernel

**Use when:**
- L1 budget can absorb the unused DFB tile (~1 tile/core per conditionally-unused DFB).
- The kernel's references to the DFB are confined to `if constexpr` blocks — no file-scope ternaries, struct-scope constexpr expressions, or other contexts that force name lookup of both branches.

**Mechanism.** Bind the DFB unconditionally on the host so `dfb::cb_scaled` always exists. Declare the `DataflowBuffer` wrapper at top level in the kernel. Gate uses with `if constexpr`.

```cpp
// Host:
KernelSpec compute{
    .compile_time_arg_bindings = {{"do_scale", do_scale ? 1u : 0u}, /* ... */},
    .dfb_bindings = {INPUT, OUTPUT, SCALED},  // SCALED bound unconditionally
};

// Kernel:
constexpr auto do_scale = get_arg(args::do_scale);
DataflowBuffer cb_scaled(dfb::cb_scaled);
if constexpr (do_scale) {
    cb_scaled.wait_front(...);
    // ... all uses of cb_scaled
}
```

### Choosing between A and B

| Constraint | Option A (`#ifdef`) | Option B (unconditional + `if constexpr`) |
|---|---|---|
| L1 cost | None | ~1 tile/core per conditionally-unused DFB |
| Kernel preprocessor scaffolding | Required | None |
| File-scope ternary / struct-scope constexpr referencing the DFB name | Required | Won't compile |
| Composability across multiple conditionals | Awkward (`#if defined(A) && defined(B)`) | Clean (`if constexpr (a && b)`) |

Prefer Option B when its preconditions hold — cleaner kernel code, easier to compose. Fall back to Option A when L1 budget or file-scope name lookup forces it. Both options are interim; NTTP-ified CTAs will eventually let the kernel be templated end-to-end, at which point `if constexpr` discarded branches skip non-dependent name lookup and Option B subsumes Option A without the L1 cost.

---

## Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers

**Category**: Pattern

**Recognition signal**: A kernel uses LLK compute APIs (`reduce_init`, `pack_tile`, `matmul_init`, `cb_wait_front`, etc.) or kernel-library helpers (`compute_kernel_lib::reduce<>`, `dataflow_kernel_lib::prepare_reduce_scaler<>`, etc.) that take `uint32_t` CB ids as parameters.

**Decision**: Pass `dfb::name` directly. The implicit conversion `DFBAccessor::operator uint32_t()` lets named DFB handles flow into any function expecting a `uint32_t` CB id, without manual extraction or wrapping.

**Correct port**:

```cpp
// LLK calls:
reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::input, dfb::scaler, dfb::output);
cb_wait_front(dfb::input, 1);
pack_tile(0, dfb::output);

// Kernel-lib calls:
compute_kernel_lib::reduce<...>(dfb::in0, dfb::scaler, dfb::out, /* block shape */, ...);
dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);
```

No `.id` extraction, no temporary `DataflowBuffer` constructed just to retrieve its underlying id, no wrapper structs.

**Template-parameter position works too.** The example `dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f)` above uses `dfb::scaler` as a non-type template parameter. The `DFBAccessor::operator uint32_t()` conversion is `constexpr`, so `dfb::name` is a valid integral constant expression wherever the legacy signature took a `uint32_t` non-type template parameter — call-argument and template-argument positions are both fine.

**Today vs. tomorrow**: Today's LLKs and kernel-lib helpers on WH/BH accept `uint32_t` — the implicit conversion is the right bridge for porting-scope work. Kernel-lib and LLK Quasar correctness is upstream of any WH/BH port; do not preemptively wrap or refactor.

**Prerequisite**: implicit conversion on `DFBAccessor::operator uint32_t()`, commit `3fbb1016d08` on `akertesz/dfb-accessor-implicit-conv`, PR #44646.

**See also**: [Anti-pattern: `.id` extraction or temp DFB wrappers](#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites).

---

## Pattern: Multi-variant factories

**Category**: Pattern

**Recognition signal**: One program factory builds multiple `ProgramSpec`s depending on a variant attribute — e.g. Welford reduction's `reduce_dim` selecting among W/H/HW variants, each with its own kernels, its own DFB set, and its own RTA schema.

**Decision**: Branch on the variant inside `create_program_spec`. No class hierarchy is needed; the variant is a configuration, not a factory subclass. Per-variant DFB unique ids and KernelSpec sources are local to each branch.

**Correct port**:

```cpp
ttnn::device_operation::ProgramArtifacts MyFactory::create_program_spec(
    const OpParams& attrs, const OpInputs& inputs, Tensor& output) {
    switch (attrs.variant) {
        case Variant::W:   return build_w_spec(attrs, inputs, output);
        case Variant::H:   return build_h_spec(attrs, inputs, output);
        case Variant::HW:  return build_hw_spec(attrs, inputs, output);
    }
}
```

Each variant's helper builds its own `ProgramSpec` and `ProgramRunParams`. Where variants share kernels, the same kernel source is reused (with different `KernelSpec`s, possibly different CTA bindings).

---

## Pattern: Unity-build hygiene for anonymous-namespace symbols

**Category**: Pattern

**Recognition signal**: Multiple program-factory `.cpp` files in the same CMake target each defining anonymous-namespace symbols (`MakeDFB`, `BindDFB`, `READER_KERNEL`, work-distribution structs, etc.) with the same name. Unity-build concatenates the `.cpp`s into one translation unit; C++ rules merge all `namespace { ... }` blocks into one scope, producing duplicate-symbol errors.

**Decision**: Hoist truly-identical declarations (helper functions, shared DFB ids) into a shared header with `inline` (functions) or `inline constexpr` (constants) linkage. For per-factory constants and structs with the same role but different content, prefix with the factory name (`W_READER_KERNEL`, `H_READER_KERNEL`, `WWorkDistribution`, `HWorkDistribution`). Function overloads distinguished by first-parameter type coexist as overloads in the merged anon namespace.

**Correct port**:

```cpp
// In a shared header (e.g., reduce_metal2_factory_helpers.hpp):
namespace {
inline constexpr const char* INPUT_DFB = "input";
inline constexpr const char* OUTPUT_DFB = "output";
inline DataflowBufferSpec MakeDFB(/* ... */) { /* ... */ }
inline void BindDFB(KernelSpec& k, /* ... */) { /* ... */ }
}

// In per-factory .cpp files:
namespace {
constexpr const char* W_READER_KERNEL = "reader_w";
constexpr const char* H_READER_KERNEL = "reader_h";
struct WWorkDistribution { /* ... */ };
struct HWorkDistribution { /* ... */ };
}
```

This isn't a Metal 2.0-specific issue, but it surfaces during op porting because most ops have multiple factories per device-operation, and Metal 2.0's named-binding pattern tempts authors to introduce more named constants.

---

## Anti-pattern: Demoting per-group CTA to RTA

**Category**: Anti-pattern

**Recognition signal**: A legacy factory uses `split_work_to_cores` and creates two `KernelDescriptor`s for the compute kernel (one per core group) with different per-group CTA values (e.g. one with `Ht=X1`, one with `Ht=X2`). The Metal 2.0 port has *one* `KernelSpec` for the compute kernel, and the dimension that varied per group has been moved into `KernelSpec::runtime_arguments_schema.named_runtime_args` instead of `compile_time_arg_bindings`.

**Why wrong**: The premise — "Metal 2.0 supports only one `KernelSpec` per kernel source" — is false. Metal 2.0 supports multiple `KernelSpec`s referencing the same source with different CTA bindings, each placed in its own `WorkUnitSpec`, sharing upstream/downstream DFBs as multi-bindings. The "two `KernelDescriptor`s per work split" idiom translates 1:1 to "two `KernelSpec`s of the same source, in two `WorkUnitSpec`s, both binding the same input/output DFBs."

The demotion sacrifices compile-time loop unrolling on the demoted dimension — a real, measurable kernel-perf regression — and is unnecessary.

**Correct port**:

```cpp
// Two compute KernelSpecs of the same source, differing only on the per-group CTA:
auto make_compute = [&](const char* unique_id, uint32_t Ht) {
    return KernelSpec{
        .unique_id = unique_id,
        .source = KernelSpec::SourceFilePath{"reduce.cpp"},
        .compile_time_arg_bindings = {{"Ht", Ht}, {"Wt", Wt}, {"NC", NC}},
        .dfb_bindings = { /* INPUT consumer, OUTPUT producer */ },
        // ...
    };
};
std::vector<KernelSpec> kernels = {reader, writer,
    make_compute("compute_g1", num_rows_per_core_group_1)};
if (group_2_present) {
    kernels.push_back(make_compute("compute_g2", num_rows_per_core_group_2));
}

// Two WorkUnitSpecs, one per core group:
WorkUnitSpec wu_g1{.unique_id = "wu_g1", .kernels = {READER, WRITER, "compute_g1"},
                   .target_nodes = core_group_1};
WorkUnitSpec wu_g2{.unique_id = "wu_g2", .kernels = {READER, WRITER, "compute_g2"},
                   .target_nodes = core_group_2};
```

The framework validates that the two compute `KernelSpec`s have non-overlapping WU coverage and that their CONSUMER bindings of INPUT (and PRODUCER bindings of OUTPUT) match — the local hardware invariant that exactly one reader, one writer, and one compute run together at each node is preserved.

**Prerequisite**: multi-PRODUCER and multi-CONSUMER DFB binding relaxation, commit `542287f4ccb` on `akertesz/misc-op-port-fixes`. Before that relaxation, the validator rejected "DFB has multiple producers/consumers"; on older snapshots, the demote-to-RTA fallback may be unavoidable.

---

## Anti-pattern: `.id` extraction or temp DFB wrappers at LLK call sites

**Category**: Anti-pattern

**Recognition signal**: At a kernel call site for an LLK (`reduce_init`, `pack_tile`, etc.) or kernel-lib helper, the code does any of:

- `.id`-extraction: `reduce_init(dfb::input.id, ...)`.
- Temporary wrapper: `experimental::DataflowBuffer in_dfb(dfb::input); reduce_init(in_dfb.get_id(), ...)`.
- Typed-shim struct: a wrapper template (`BufferRef<T>` or similar) that holds a CB id and exposes `operator uint32_t`.

**Why wrong**: Each of these reinvents an implicit conversion that `DFBAccessor` already provides. The `.id` form encodes the LLK's CB-id vocabulary into kernel code that should be DFB-centric; temporary wrappers add construction cost for no benefit; typed shims reproduce the implicit conversion locally and clutter the call site.

**Correct port**: See [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).

**Prerequisite**: `DFBAccessor::operator uint32_t()` exists, commit `3fbb1016d08` on `akertesz/dfb-accessor-implicit-conv`, PR #44646. On older Metal 2.0 snapshots without this conversion, the `.id` workaround was the right answer.

---

## Caution: Avoid varargs unless absolutely necessary

**Category**: Caution

**Recognition signal**: A `KernelSpec` declares `runtime_arguments_schema.num_runtime_varargs > 0` (or `num_common_runtime_varargs > 0`), or a kernel uses `get_vararg(i)` / `get_common_vararg(i)`.

**Decision**: Varargs are designed for kernels whose device-side code retrieves arguments via `get_vararg(i)` with `i` a runtime variable — the canonical case is an N-dimensional shape gated on a CTA-known `rank`. When each argument is referenced by a constant index (`get_vararg(0)`, `get_vararg(1)`, ...), the named form is clearer on both sides.

**Why caution rather than anti-pattern**: A port from positional legacy RTAs to varargs compiles and runs. It can be a reasonable interim step on a large kernel. But it preserves the legacy positional vocabulary instead of upgrading to Metal 2.0's named one; the named form is the recommended endpoint for new code.

**The named/vararg test**: look at the device-side retrieval. If every `get_vararg(i)` uses a constant `i`, you wanted named arguments — give them names. If `i` is a runtime variable iterating over a count, varargs are the right fit.

---

## Caution: Modifying a shared dataflow kernel

**Category**: Caution

**Recognition signal**: A port modifies a kernel source that lives outside the op's own directory — typically a reader / writer in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/` or similar. Common shared kernels include `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `reader_unary_reduce_universal_start_id.cpp`, and `writer_unary_sharded.cpp` (under `data_movement/sharded/device/kernels/dataflow/`).

**Decision**: Pick one of two paths, based on whether all consumers of the kernel can co-migrate to Metal 2.0 in the same PR (or a bundled set of PRs):

1. **In-place modification** — when every consumer op of the kernel is being ported to Metal 2.0 in the same PR or bundled set. Modify the kernel source and update each consumer's factory consistently. Before editing, grep for all consumers (`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`) and verify each one is in the bundled set.

2. **Fork with `_metal2` suffix** — the typical answer during the bulk-port window, when consumers cannot all co-migrate. Copy the kernel into a `_metal2`-suffixed file alongside the original (e.g., `writer_unary_interleaved_start_id_metal2.cpp` next to the legacy `writer_unary_interleaved_start_id.cpp`). Modify the copy for Metal 2.0 — named bindings, `dfb::name`, `ta::name`, named RTAs. Reference the new file from the ported factory's `KernelSpec::source`. The legacy copy stays in place for unmigrated consumers.

   **Sunset:** delete the legacy copy when the last unmigrated consumer ports. The `_metal2` suffix is a load-bearing signal during this window — anyone touching the legacy copy should see the suffix and consider whether the change also belongs on the fork.

   **Drift discipline:** until sunset, bug fixes to the legacy copy should be evaluated for the `_metal2` copy. The fork is intentionally short-lived; keeping the two in sync is the cost of unblocking the bulk-port wave without coordinating dozens of consumer PRs.

A kernel-source change that compiles cleanly for the porting op but breaks a sibling op's CTA layout is one of the failure modes that escapes the recipe's anti-pattern self-audit — it's *not* in the ported op's own files. The legacy-inventory step explicitly notes any cross-op kernel sources, and the port report records the touch under "Open items for downstream" with the chosen path (in-place co-migration list, or fork) so the next sibling-op porter sees the coordination signal.

**Excluded from "cross-op"**: kernel-lib (`ttnn/cpp/ttnn/kernel_lib/`) and standard framework APIs (`tt_metal/hw/inc/api/...`) — these are out of porter scope by the [recipe's scope boundary](port_op_to_metal2_recipe.md#read-this-first) and the porter does not modify them. Cross-op kernels are sources living in *another op's* kernel directory.
