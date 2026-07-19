# Metal 2.0 Op Port — Patterns & Anti-Patterns Catalog

This catalog accumulates patterns and anti-patterns observed during Metal 2.0 op ports. It is referenced from the [feasibility audit](port_op_to_metal2_audit.md) for yellow-tier override guidance, from the [port recipe's planning step](port_op_to_metal2_recipe.md#plan-the-spec) for structural-decision reference, and from the recipe's verification step for the anti-pattern self-audit checklist.

Each entry is self-contained. New entries land here as they're discovered during ports.

## Conventions

Entry shape — load-bearing fields, in order:
- **Category**: `Pattern` (an affordance to reach for) | `Anti-pattern` (a shape to avoid) | `Caution` (a judgment call with consequences). Sanctioned-exception entries carry a parenthetical category suffix (e.g. `Pattern (sanctioned kernel-side exception)`) flagging that the entry documents a deliberate carve-out from a stricter rule elsewhere in the recipe.
- **Recognition signal**: what to look for in legacy code (or proposed port code).
- **Decision** (Pattern / Caution) or **Why wrong** (Anti-pattern): the prescribed move and its rationale.

Optional fields, used when the entry's substance requires them:
- **Why this is hard** (before Decision): explanatory rationale when the recognition→decision step has non-obvious reasoning the porter benefits from understanding before reading the prescription. May also appear as **Why this is a sanctioned X exception** for sanctioned-exception entries.
- **Correct port**: code or prose showing the right pattern. Most Pattern entries include this; omit when the Decision is fully prescriptive on its own.
- **Constraint**: scoping fence on where the entry's authority ends. Beyond that boundary, capitulate per the recipe's [§When the discipline doesn't fit](port_op_to_metal2_recipe.md#when-the-discipline-doesnt-fit).
- **Sanctioned exception note**: for sanctioned-exception entries, explicit acknowledgement that the prescription deviates from a stricter rule (typically the [kernel-side whitelist](port_op_to_metal2_recipe.md#kernel-side-whitelist) or [host-side scope discipline](port_op_to_metal2_recipe.md#host-side-stay-in-the-lane)), with the why-this-is-legitimate reasoning. Paired with the Category suffix.
- **Prerequisite**: Metal 2.0 framework commit / PR / version below which the pattern doesn't apply.
- **See also**: cross-references.

## Entries

### Patterns

- [Self-loop DFB binding (producer == consumer)](#pattern-self-loop-dfb-binding)
- [Sync-free and single-ended CBs → self-loop DFB](#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
- [Conditional / optional DFB bindings](#pattern-conditional--optional-dfb-bindings)
- [Aliased DFBs (legacy aliased CBs)](#pattern-aliased-dfbs-legacy-aliased-cbs)
- [Same-FIFO aliasing (one DFB, multiple kernel-side names)](#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
- [Pass DFB handles directly to LLKs and kernel-lib helpers](#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
- [Multi-variant factories](#pattern-multi-variant-factories)
- [Unity-build hygiene for anonymous-namespace symbols](#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
- [Removing pybound legacy factory entry points](#pattern-removing-pybound-legacy-factory-entry-points)

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

**Decision**: Declare two `DFBBinding`s on the same `KernelSpec` for the same DFB — one with `endpoint_type = PRODUCER`, one with `endpoint_type = CONSUMER`. The two bindings may share a single `accessor_name` (yielding one device-side `dfb::name` handle), or use two distinct names (yielding two device-side handles aliasing the same DFB).

**Correct port**:

```cpp
// Shared-name form (most natural for self-loop):
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::PRODUCER);
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::CONSUMER);

// Kernel side:
experimental::DataflowBuffer cb_acc(dfb::acc);  // one wrapper drives both directions
```

The two-distinct-names form (`acc_w` for PRODUCER, `acc_r` for CONSUMER, yielding `dfb::acc_w` and `dfb::acc_r` aliasing the same DFB) also works.

**See also**: [Anti-pattern: Demoting per-group CTA to RTA](#anti-pattern-demoting-per-group-cta-to-rta) (separate pattern, but the same "host-side variation in `KernelSpec`" mental model).

---

## Pattern: Sync-free and single-ended CBs → self-loop DFB

**Category**: Pattern

**A word on legacy CBs — read this first.** Legacy gave kernel authors essentially *one* primitive for "a chunk of L1 the kernel touches": the CircularBuffer. So they used it for everything — genuine producer→consumer FIFOs, but also private scratch memory, base-pointer windows onto resident tensors, and output staging nothing ever drains. When all you have is a CircularBuffer, everything looks like one. On Gen1 you port each of these **as-is**: a DFB lowers to a plain circular buffer, so whatever shape the legacy CB was in, the DFB carries it faithfully with zero functional change. This entry answers the one question those overloaded CBs raise — *how do you bind a CB that can't present a FIFO producer **and** a FIFO consumer on **distinct** kernels?*

**Recognition signal**: A CB that lacks a usable FIFO producer–consumer pair on distinct kernels, so the spec validator (which requires **≥1 PRODUCER and ≥1 CONSUMER** binding) would otherwise reject it. Two shapes land here:
- **Sync-free** — the kernel uses the CB purely as an *address source*: a base-pointer grab via `get_read_ptr` / `get_pointer_to_cb_data` (or `get_write_ptr`), then a direct memory access, with **no FIFO ops at all** — nothing `push_back`s, nothing `wait_front`s. The legacy idiom borrows a CB onto a resident tensor's buffer because, pre–Metal 2.0, that was the most convenient way to hand a kernel a base pointer to resident memory.
- **Single-ended** — the CB *does* use the FIFO machinery, but on **one** side only: a FIFO producer with no consumer (or vice versa). Canonical case: the compute packer produces tiles (`reserve_back` / `push_back`) straight into an output-tensor-backed CB that nothing drains — a **synchronized** CB (real `push_back`), just missing its consumer. **Examples:** conv2d / pool `OUT` (compute packer → output shard); pool `out_idx_cb` (a DM kernel writes the argmax-index output).

**Decision — self-loop it.** Bind the touching kernel as **both** PRODUCER and CONSUMER of the DFB (a self-loop), borrowing the [Self-loop DFB binding](#pattern-self-loop-dfb-binding) mechanism. **This is legal on Gen1 for compute *and* DM kernels alike** — a Gen1 DFB lowers to a plain circular buffer that a single RISC (compute or DM) can both fill and drain, so the kernel code is untouched and runtime behavior is identical to the legacy CB. The self-loop is the sanctioned Gen1 port shape, not a temporary hack. First confirm the CB *genuinely* lacks a distinct producer+consumer pair across **every** config (per *classify per instantiation*, below) — a real cross-kernel FIFO whose partner you simply missed is an ordinary DFB, not a self-loop.

> **A DM self-loop is a Gen1-only shape.** The spec validator rejects a DM self-loop on **Gen2 (Quasar)**, where a DFB's credit machinery requires producer and consumer to be *distinct* kernels. That is **Quasar-uplift's** concern, not a Gen1 blocker — and Metal 2.0's declarative bindings make a DM self-loop trivial to spot post-port (a kernel bound PRODUCER+CONSUMER on one DFB), so the Quasar audit finds and refactors them; you need not track them here.

**Correct port**:

```cpp
// (A) COMPUTE kernel, sync-free (e.g. a reciprocal LUT read by base pointer).
// Bind the one compute kernel as both PRODUCER and CONSUMER (shared accessor name):
KernelSpec compute{
    // ...
    .dfb_bindings = {
        DFBBinding{.dfb_spec_name = RECIP, .accessor_name = "recip", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = RECIP, .accessor_name = "recip", .endpoint_type = DFBEndpointType::CONSUMER},
    },
};

// (B) DM kernel, sync-free scratch. Same shape — a DM self-loop is legal on Gen1
// (it lowers to a plain CB one DM RISC both fills and drains):
KernelSpec reader{   // the kernel that actually needs the scratch
    .dfb_bindings = {
        DFBBinding{.dfb_spec_name = SCRATCH, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SCRATCH, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::CONSUMER},
    },
};

// Kernel side (both) — unchanged from legacy; base-pointer access, no FIFO ops:
experimental::DataflowBuffer dfb_recip(dfb::recip);      // (A) one handle, both directions
experimental::DataflowBuffer dfb_scratch(dfb::scratch);  // (B) same
// ... read / write via base pointer as before ...
```

(Both endpoints share one `accessor_name`, which relies on the per-kernel accessor-name dedup relaxation noted under [Self-loop DFB binding](#pattern-self-loop-dfb-binding); two distinct names also work.)

**Single-ended CB with a real endpoint — bind the real one; self-loop only if it's the sole toucher.** For a single-ended CB that already has one genuine endpoint — e.g. the packer's producer — first ask: **does any kernel other than the packer actually touch the CB?** If a co-resident kernel also accesses it — e.g. a DM kernel that reads the packed result to drain it elsewhere — bind *that* kernel as the consumer; it is a real endpoint, nothing self-looped. If the packer is the **sole** toucher — the usual shape for **sharded output**, where packed tiles land straight in the resident output shard and nothing drains them — self-loop it (compute or DM alike). Decide by reading the kernel *bodies* for a real access, not by kernel names: a `writer_*`-named kernel can be a weights- or activation-mover that never touches the output CB.

**The verdict can flip per config — classify per instantiation, not per CB.** Whether a CB lands here (sync-free / single-ended) or is a genuine producer→consumer FIFO is *not* a fixed property of the CB; it can change across an op's configs. The same `buffer_index` can be a sync-free **scratchpad** (compute tilizes in place → self-loop) under one sharding and a **real FIFO** (a DM reader produces, compute consumes → ordinary DFB) under another. Re-run the litmus per code-path — one verdict applied across all configs mis-classifies the rest. **Canonical confuser:** conv2d `ACT_TILIZED` — height-sharded → sync-free scratchpad (self-loop); block/width-sharded → real FIFO.

**Orthogonal — endpoint multiplicity.** A self-loop resolves too *few* endpoints (single-ended / sync-free). The opposite case — a CB with **2+ FIFO endpoints of one kind on a node** — is *not* a self-loop case; it is a **multi-binding** CB, port work in its own right (set the DFB multi-binding advanced option). See [DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc).

**See also**: [Self-loop DFB binding](#pattern-self-loop-dfb-binding) (the legitimate accumulator case whose mechanism this borrows — there the producer/consumer do genuine work); [DFB endpoint legality](port_op_to_metal2_audit.md#dfb-endpoint-legality-spsc).

---

## Pattern: Conditional / optional DFB bindings

**Category**: Pattern

**Recognition signal**: A resource — a DFB, a tensor (`TensorAccessor`), or a semaphore — is used by a kernel only on some code paths, with the configuration known at host time (e.g. a `cb_scaled` DFB used only when `do_scale = true`, a `cb_fusion` used only when `FUSE_PRE_ADD`, or an optional `batch_offset` tensor read only when a feature flag is set). The compile-time tell, in a build where the feature is *off*, is a name-resolution failure on the generated token: `'cb_scaled' is not a member of 'dfb'`, `'batch_offset' is not a member of 'ta'`, or the `sem::` equivalent.

**Why this is hard.** Metal 2.0's `dfb::<name>` namespace is generated from the actual host bindings: if the host omits SCALED from `dfb_bindings`, `dfb::cb_scaled` is not declared in `kernel_bindings_generated.h`. C++ `if constexpr` in a non-template function (which `kernel_main()` is) still performs name lookup on the discarded branch — so `if constexpr (false) { ... dfb::cb_scaled ... }` fails to compile at parse time even though the branch is dead at codegen. The same constraint hits kernels that reference the conditional DFB name from file-scope contexts like ternaries: both operands resolve at parse time regardless of which branch the constant condition selects. The fix is to gate the kernel-side references at the **preprocessor** level, before C++ parsing sees them.

This applies verbatim to `tensor::<name>` (optional tensors) and `sem::<name>` — all three namespaces are emitted by genfiles **per-binding**, so a resource no kernel binds on the off-path produces no token at all. Optional **tensors** are the case where the `#ifdef` gate is not merely preferred but **mandatory**: an absent tensor often has nothing to bind even in principle (the argument may simply not be passed), so there is no always-bind fallback to reach for.

**Decision.** Conditionally bind the DFB on the host. Emit a matching preprocessor define via `KernelSpec::defines` when the binding is present. On the kernel side, `#ifdef`-gate the constexpr alias of the DFB name and all expressions that reference it.

```cpp
// Host:
const bool fuse_pre_add = ...;
KernelSpec compute{
    .compile_time_args = { /* ... */ },
    .compiler_options = {.defines = fuse_pre_add
        ? Table<std::string, std::string>{{"FUSE_PRE_ADD", "1"}}
        : Table<std::string, std::string>{}},
    .dfb_bindings = fuse_pre_add
        ? Group<DFBBinding>{INPUT, OUTPUT, FUSION}
        : Group<DFBBinding>{INPUT, OUTPUT},
};

// Kernel:
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_fusion = dfb::cb_fusion;
#endif

// File-scope expression referencing the conditional name:
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_x = (do_gamma | do_beta) ? cb_fusion : cb_out;
#else
constexpr uint32_t cb_x = cb_out;
#endif
```

The `#ifdef` runs at the preprocessor stage, before the C++ compiler sees the code — so `dfb::cb_fusion` never enters name lookup in the unfused build, and the conditionally-bound DFB is honored end-to-end. This avoids the L1 cost of binding the DFB unconditionally (which can be significant for L1-tight ops — layernorm's `cb_fusion` is ~16 tiles, for example) and works regardless of whether the kernel references the DFB name from file-scope expressions or only inside function bodies.

**Promote a CTA gate to a define.** A subtle case: the legacy kernel may gate the conditional DFB's *use* through a **CTA** (`if constexpr (do_gamma) { … cb_gamma … }`) rather than an `#ifdef`. The `if constexpr` still name-looks-up `dfb::cb_gamma` in the discarded branch, so the port must **promote** that gate to the preprocessor — convert the `if constexpr` to `#ifdef FUSE_GAMMA`, and emit the matching define from the host. Watch the emission target: the legacy factory often sent the define to only *some* kernels (e.g. the reader), but the promoted `#ifdef` must be fed to **every** kernel that references the conditionally-bound DFB. The define and the binding share one condition.

**Don't bind unconditionally** as an alternative. You might be tempted to unconditionally bind a DFB so its token always exists. This is a bad solution: it wastes L1 SRAM, a scarce resource that is in short supply on these devices. And beyond the L1 cost, it fails to compile anyway when the kernel needs to reference the conditionally-used name from a file-scope ternary or similar parse-time-resolves-both-branches context. For optional **tensors** it is not even an option — there may be nothing to bind (see *Why this is hard* above).

**Future direction — `#ifdef` is a temporary scaffold.** Metal 2.0 work to NTTP-ify all CTAs will let the entire kernel body be template-instantiated on the CTA values, at which point `if constexpr` discarded branches truly skip non-dependent name lookup, and the `#ifdef` scaffolding is retired in favor of `if constexpr (do_scale) { use dfb::name; }`. That work is in flight and will soon supersede this pattern. Until NTTP CTAs land, `#ifdef`-gated conditional bindings are the recommended pattern for Metal 2.0 today.

---

## Pattern: Aliased DFBs (legacy aliased CBs)

**Category**: Pattern

**Recognition signal**: Legacy code that places two or more `buffer_index` values on the *same* `CBDescriptor` — multi-element `format_descriptors`, multi-key `data_format_spec` map in the imperative form, or repeated `set_page_size` calls with different `buffer_index` arguments on the same `CircularBufferConfig`. The legacy intent is two logically distinct buffers sharing one L1 region. The audit's [Aliased Circular Buffers entry](port_op_to_metal2_audit.md#aliased-circular-buffers-cbs-sharing-backing-memory--landed) catches this in the legacy inventory.

**Decision**: Declare one `DataflowBufferSpec` per legacy `buffer_index`. Aliasing is an **advanced/ninja feature** — the `alias_with` field lives on `DFBAdvancedOptions` (see [`advanced_options.hpp`](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp)), reached via `DataflowBufferSpec::advanced_options.alias_with`. Set each spec's `advanced_options.alias_with` to mutually reference the others — the alias group must be a *strict clique* (every member names every other member). All members of the alias group share these legality constraints:
- Same `num_entries * entry_size` (total backing size identical).
- Bound to the same set of kernels.
- Borrowed-memory consistency (either all `borrowed_from` matching `TensorParameter`s, or none borrowed).

The validator enforces these as the three legality rules; missing any of them surfaces with a message in the [migration guide's troubleshooting table](../metal2_migration_guide.md#cryptic-error--likely-cause). The `DFBAdvancedOptions` header comments are the authoritative source for the field's contract — including the explicit "no clobbering guarantees" note: aliased DFBs share backing memory, and correctness of *which* logical buffer's data is live at any moment is the kernel author's responsibility.

**Correct port**:

```cpp
// Two indices that legacy shared via a single CBDescriptor become two
// DFBs with mutual alias_with entries (advanced_options-nested):
DataflowBufferSpec INTERM0{ .unique_id = INTERM0_ID, .num_entries = N, .entry_size = S,
                            .advanced_options = {.alias_with = {OUTPUT_ID}} };
DataflowBufferSpec OUTPUT{  .unique_id = OUTPUT_ID,  .num_entries = N, .entry_size = S,
                            .advanced_options = {.alias_with = {INTERM0_ID}} };
```

For larger alias groups (three or more), every member names every other member in its `advanced_options.alias_with` — partial mentions are rejected by the validator.

**Don't split** the aliased CB into independent, non-aliased DFBs. That changes the L1 footprint and breaks any kernel assumption that the indices shared an address.

**See also**: [migration guide — DataflowBufferSpec: Aliased DFBs](../metal2_migration_guide.md#dataflowbufferspec); [audit — Aliased Circular Buffers entry](port_op_to_metal2_audit.md#aliased-circular-buffers-cbs-sharing-backing-memory--landed); [Same-FIFO aliasing](#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names) (the *other* kind of "aliasing" — don't confuse them).

---

## Pattern: Same-FIFO aliasing (one DFB, multiple kernel-side names)

**Category**: Pattern

**Recognition signal**: A kernel refers to **one** circular buffer through **multiple names**, via legacy `uint32_t` CB-index aliasing — `constexpr uint32_t cb_x = cb_in;`, or a ternary that resolves one name to a CB index (`constexpr uint32_t cb_x = fuse ? cb_fusion : cb_out;`). Both names are then used for the *same* FIFO: a `push`/`reserve_back` via one name is seen by a `wait_front`/`pop_front` via the other, because they are literally the same buffer with the same read/write pointers.

**Two forms — kernel-side and host-side.** The signal above is kernel-side (a `constexpr` aliases one CB index to another). The *same* one-FIFO-two-names situation can also be set up **on the host**: the program factory **mirrors one CB's index onto another** (and the buffer handle, in the allocation path) — often a 0-page CB pointed at a real one — so two logical CB names resolve to the same buffer before any kernel runs (one legacy spelling is a CB-descriptor field such as `overlapped_by_cb` whose handler does `cb.handle = other.handle; cb.index = other.index;`). The consequence is identical — one FIFO, two names — but a kernel-only scan won't catch it; you have to read the factory's CB setup. Treat it exactly as the kernel-side form below: **one** `DataflowBufferSpec`, the second name aliased to it — *not* a second spec, and *not* `alias_with`.

**This is not [Aliased DFBs](#pattern-aliased-dfbs-legacy-aliased-cbs).** The two are easy to conflate and the distinction is correctness-critical:

| | Aliased DFBs (`alias_with`) | Same-FIFO aliasing (this entry) |
|---|---|---|
| Buffers | Two+ **distinct** DFBs | **One** DFB |
| FIFO pointers | **Independent** per DFB | **Shared** — one FIFO |
| What's shared | The physical L1 region | The buffer's identity |
| Legacy form | Two CB *indices* configured at the same L1 address | Same CB index under two names — kernel-side (`uint32_t` alias) or host-side (factory copies handle + index) |

Modeling same-FIFO aliasing with `alias_with` is a **bug**: you would get two independent FIFOs at one address, and the producer/consumer pointer coherence the kernel relies on (produce via one name, consume via the other) is lost silently.

**The converse trap: don't be fooled into the Same-FIFO box by a kernel that manually keeps two views' pointers in lock-step.** If the two views are distinct `buffer_index`es / `CBFormatDescriptor`s — the tell is a different `page_size` or face geometry — they are **Aliased DFBs** (`alias_with`), *even though* the kernel walks them at matching L1 addresses. The buffers are structurally distinct; the pointer-alignment is a kernel convention, not shared FIFO identity. The disambiguator is the **index**, not the runtime pointer values.

**Decision**: Keep **one** `DataflowBufferSpec` and **one** `DFBBinding`. Express the second name as a **kernel-side handle alias** — a `constexpr` alias of the generated token:

```cpp
// Legacy: cb_x and cb_in are the same CB index.
constexpr uint32_t cb_x = cb_in;
// ... uses both cb_x and cb_in for the same FIFO ...

// Metal 2.0: one binding (dfb::cb_in); alias the handle, construct ONE object.
constexpr auto cb_x = dfb::cb_in;     // cb_x and dfb::cb_in are the same handle
DataflowBuffer dfb_in(dfb::cb_in);    // a SINGLE object for the FIFO
```

**Don't**:
- **Don't add a second `DFBBinding`** to the same DFB under a different `accessor_name` to manufacture a `dfb::cb_x` token. A kernel binds a given DFB under exactly one accessor name (the [self-loop pair](#pattern-self-loop-dfb-binding) — one name, PRODUCER+CONSUMER — is the only "twice" form), and the spec validator rejects a second name for the same DFB.
- **Don't construct two `DataflowBuffer` objects from the same `DFBAccessor`** (`DataflowBuffer a(dfb::cb_in); DataflowBuffer b(dfb::cb_in);`). It compiles and runs, but two objects aliasing one FIFO break the object↔DFB identity that device-side debug tooling depends on. Alias the *handle*, keep *one* object.

**Path-dependent variant.** When the aliased name resolves to *different* DFBs on different compile-time paths (`cb_x` is `dfb::cb_fusion` when fused, `dfb::cb_out` otherwise), gate the handle alias under the matching `#ifdef` so each path names only DFBs bound on it — the [Conditional / optional DFB bindings](#pattern-conditional--optional-dfb-bindings) pattern applied to the *name→DFB mapping*:

```cpp
#ifdef FUSE
constexpr auto cb_x = dfb::cb_fusion;
#else
constexpr auto cb_x = dfb::cb_out;
#endif
```

**See also**: [Aliased DFBs](#pattern-aliased-dfbs-legacy-aliased-cbs) (the distinct-buffers / shared-memory kind); [Conditional / optional DFB bindings](#pattern-conditional--optional-dfb-bindings); [Self-loop DFB binding](#pattern-self-loop-dfb-binding).

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

**See also**: [Anti-pattern: `.id` extraction or temp DFB wrappers](#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites).

---

## Pattern: Multi-variant factories

**Category**: Pattern

**Recognition signal**: One program factory builds multiple `ProgramSpec`s depending on a variant attribute — e.g. Welford reduction's `reduce_dim` selecting among W/H/HW variants, each with its own kernels, its own DFB set, and its own RTA schema.

**Decision**: Branch on the variant inside `create_program_artifacts`. No class hierarchy is needed; the variant is a configuration, not a factory subclass. Per-variant DFB unique ids and KernelSpec sources are local to each branch.

**Correct port**:

```cpp
ttnn::device_operation::ProgramArtifacts MyFactory::create_program_artifacts(
    const operation_attributes_t& attrs,
    const tensor_args_t&           inputs,
    tensor_return_value_t&         output) {
    switch (attrs.variant) {
        case Variant::W:   return build_w_artifacts(attrs, inputs, output);
        case Variant::H:   return build_h_artifacts(attrs, inputs, output);
        case Variant::HW:  return build_hw_artifacts(attrs, inputs, output);
    }
}
```

Each variant's helper builds its own `ProgramSpec` and `ProgramRunArgs`. Where variants share kernels, the same kernel source is reused (with different `KernelSpec`s, possibly different CTA bindings).

---

## Pattern: Unity-build hygiene for anonymous-namespace symbols

**Category**: Pattern

**Recognition signal**: Multiple program-factory `.cpp` files in the same CMake target each defining anonymous-namespace symbols (`MakeDFB`, `BindDFB`, `READER_KERNEL`, work-distribution structs, etc.) with the same name. Unity-build concatenates the `.cpp`s into one translation unit; C++ rules merge all `namespace { ... }` blocks into one scope, producing duplicate-symbol errors.

**Decision**: Hoist truly-identical declarations (helper functions, shared DFB ids) into a shared header with `inline` (functions) or `inline constexpr` (constants) linkage. For per-factory constants and structs with the same role but different content, prefix with the factory name (`W_READER_KERNEL`, `H_READER_KERNEL`, `WWorkDistribution`, `HWorkDistribution`). Function overloads distinguished by first-parameter type coexist as overloads in the merged anon namespace.

**Correct port**:

```cpp
// In a shared header (e.g., reduce_metal2_factory_helpers.hpp):
namespace {
inline const DFBSpecName INPUT_DFB{"input"};
inline const DFBSpecName OUTPUT_DFB{"output"};
inline DataflowBufferSpec MakeDFB(/* ... */) { /* ... */ }
inline void BindDFB(KernelSpec& k, /* ... */) { /* ... */ }
}

// In per-factory .cpp files:
namespace {
const KernelSpecName W_READER_KERNEL{"reader_w"};
const KernelSpecName H_READER_KERNEL{"reader_h"};
struct WWorkDistribution { /* ... */ };
struct HWorkDistribution { /* ... */ };
}
```

This isn't a Metal 2.0-specific issue, but it surfaces during op porting because most ops have multiple factories per device-operation, and Metal 2.0's named-binding pattern tempts authors to introduce more named constants.

---

## Anti-pattern: Demoting per-group CTA to RTA

**Category**: Anti-pattern

**Recognition signal**: A legacy factory uses `split_work_to_cores` and creates two `KernelDescriptor`s for the compute kernel (one per core group) with different per-group CTA values (e.g. one with `Ht=X1`, one with `Ht=X2`). The Metal 2.0 port has *one* `KernelSpec` for the compute kernel, and the dimension that varied per group has been moved into `KernelSpec::runtime_arg_schema.runtime_arg_names` instead of `compile_time_args`.

**Why wrong**: The premise — "Metal 2.0 supports only one `KernelSpec` per kernel source" — is false. Metal 2.0 supports multiple `KernelSpec`s referencing the same source with different CTA bindings, each placed in its own `WorkUnitSpec`, sharing upstream/downstream DFBs as multi-bindings. The "two `KernelDescriptor`s per work split" idiom translates 1:1 to "two `KernelSpec`s of the same source, in two `WorkUnitSpec`s, both binding the same input/output DFBs."

The demotion sacrifices compile-time loop unrolling on the demoted dimension — a real, measurable kernel-perf regression — and is unnecessary.

**Correct port**:

```cpp
// Two compute KernelSpecs of the same source, differing only on the per-group CTA:
const KernelSpecName COMPUTE_G1{"compute_g1"};
const KernelSpecName COMPUTE_G2{"compute_g2"};
auto make_compute = [&](KernelSpecName unique_id, uint32_t Ht) {
    return KernelSpec{
        .unique_id = std::move(unique_id),
        .source = "reduce.cpp",
        .compile_time_args = {{"Ht", Ht}, {"Wt", Wt}, {"NC", NC}},
        .dfb_bindings = { /* INPUT consumer, OUTPUT producer */ },
        // ...
    };
};
Group<KernelSpec> kernels = {reader, writer,
    make_compute(COMPUTE_G1, num_rows_per_core_group_1)};
if (group_2_present) {
    kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
}

// Two WorkUnitSpecs, one per core group:
WorkUnitSpec wu_g1{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1},
                   .target_nodes = core_group_1};
WorkUnitSpec wu_g2{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2},
                   .target_nodes = core_group_2};
```

The framework validates that the two compute `KernelSpec`s have non-overlapping WU coverage and that their CONSUMER bindings of INPUT (and PRODUCER bindings of OUTPUT) match — the local hardware invariant that exactly one reader, one writer, and one compute run together at each node is preserved. (See [`dataflow_buffer_spec.hpp`](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp) for the full canonical statement of the DFB endpoint invariant — including the third condition, "same kernel kind," that's implicit in this all-compute example.)

---

## Pattern: Removing pybound legacy factory entry points

**Category**: Pattern (sanctioned host-side exception)

**Recognition signal**: A pybind file inside the op directory (typically `*_pybind.cpp` / `*_pybind.hpp`, or a `bindings/` subdirectory under the op) exposes a legacy host-API function that ceases to exist post-port. The canonical case is `create_program_descriptor` — once the op moves to a `MetalV2FactoryConcept` factory, that function is gone, and the pybind line that exposed it references a non-existent symbol. The rule generalizes: *any* pybind exposure of a legacy factory entry point that the port deletes falls under this entry.

**Why this is a sanctioned host-side exception**: The [host-side scope discipline](port_op_to_metal2_recipe.md#host-side-stay-in-the-lane) keeps op-level host code outside the program factory body off-limits during the port. A pybind file is op-level host code outside the program factory body, so the blanket rule would forbid touching it — but a pybind line that references a vanished symbol is structurally untenable: leaving it as-is means the post-port code doesn't compile (or links against nothing). Deletion is mandatory; the rule's blanket prohibition is lifted *narrowly* for this case.

**Decision**:

1. **Delete** the pybind line(s) that expose the vanished function. Do not attempt to "update" the binding to point at the new factory's entry point (e.g., `create_program_artifacts`) — the two functions have different signatures and use patterns, and the Python-side callers (if any) need to be addressed separately, not silently retargeted. Make the smallest change that restores compilation: just remove the line(s).
2. **Record the deletion prominently in the port report** under [Handoff points](port_op_to_metal2_recipe.md#handoff-points). Include: the pybind file path, the function name(s) deleted, and a one-line description of what the function was for (if you can tell from the surrounding code). This is a *user-visible API surface change* — downstream Python consumers (tests, notebooks, internal tooling) that called into the legacy entry point need to be findable by the people who maintain them. The prominent report entry is how they get found.

**Constraint**: This exception applies *only* to pybind exposures of factory entry points that the Metal 2.0 port causes to disappear. Other op-level pybind lines — the user-facing op binding itself, attribute conversions, return-value handling — remain off-limits per the host-side scope discipline. If you're uncertain whether a given pybind line falls inside or outside the exception, the safe move is to leave it and write a finding in the report.

---

## Anti-pattern: `.id` extraction or temp DFB wrappers at LLK call sites

**Category**: Anti-pattern

**Recognition signal**: At a kernel call site for an LLK (`reduce_init`, `pack_tile`, etc.) or kernel-lib helper, the code does any of:

- `.id`-extraction: `reduce_init(dfb::input.id, ...)`.
- Temporary wrapper: `experimental::DataflowBuffer in_dfb(dfb::input); reduce_init(in_dfb.get_id(), ...)`.
- Typed-shim struct: a wrapper template (`BufferRef<T>` or similar) that holds a CB id and exposes `operator uint32_t`.

**Why wrong**: Each of these reinvents an implicit conversion that `DFBAccessor` already provides. The `.id` form encodes the LLK's CB-id vocabulary into kernel code that should be DFB-centric; temporary wrappers add construction cost for no benefit; typed shims reproduce the implicit conversion locally and clutter the call site.

**Correct port**: See [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).

---

## Caution: Avoid varargs unless absolutely necessary

**Category**: Caution

**Recognition signal**: A `KernelSpec` declares `advanced_options.num_runtime_varargs > 0` (or `advanced_options.num_common_runtime_varargs > 0`), or a kernel uses `get_vararg(i)` / `get_common_vararg(i)`. The vararg fields live on `KernelAdvancedOptions` (see [`advanced_options.hpp`](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp)) — they're an explicit advanced/temporary mechanism slated for deprecation in favor of `std::array` typed arguments.

**Decision**: Varargs are designed for kernels whose device-side code retrieves arguments via `get_vararg(i)` with `i` a runtime variable — the canonical case is an N-dimensional shape gated on a CTA-known `rank`. When each argument is referenced by a constant index (`get_vararg(0)`, `get_vararg(1)`, ...), the named form is clearer on both sides.

**Why caution rather than anti-pattern**: A port from positional legacy RTAs to varargs compiles and runs. It can be a reasonable interim step on a large kernel. But it preserves the legacy positional vocabulary instead of upgrading to Metal 2.0's named one; the named form is the recommended endpoint for new code.

**The named/vararg test**: look at the device-side retrieval. If every `get_vararg(i)` uses a constant `i`, you wanted named arguments — give them names. If `i` is a runtime variable iterating over a count, varargs are the right fit.

---

## Caution: Modifying a shared dataflow kernel

**Category**: Caution

**Recognition signal**: A port modifies a kernel source that lives outside the op's own directory — typically a reader / writer in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/` or similar. Common shared kernels include `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `reader_unary_reduce_universal_start_id.cpp`, and `writer_unary_sharded.cpp` (under `data_movement/sharded/device/kernels/dataflow/`).

**Decision**: Pick one of two paths, based on whether all consumers of the kernel can co-migrate to Metal 2.0 in the same PR (or a bundled set of PRs):

1. **In-place modification** — when every consumer op of the kernel is being ported to Metal 2.0 in the same PR or bundled set. Modify the kernel source and update each consumer's factory consistently. Before editing, grep for all consumers (`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`) and verify each one is in the bundled set.

2. **Fork with `_metal2` suffix** — the typical answer during the bulk-port window, when consumers cannot all co-migrate. Copy the kernel into a `_metal2`-suffixed file alongside the original (e.g., `writer_unary_interleaved_start_id_metal2.cpp` next to the legacy `writer_unary_interleaved_start_id.cpp`). Modify the copy for Metal 2.0 — named bindings, `dfb::name`, `tensor::name`, named RTAs. Reference the new file from the ported factory's `KernelSpec::source`. The legacy copy stays in place for unmigrated consumers.

   **Sunset:** delete the legacy copy when the last unmigrated consumer ports. The `_metal2` suffix is a load-bearing signal during this window — anyone touching the legacy copy should see the suffix and consider whether the change also belongs on the fork.

   **Drift discipline:** until sunset, bug fixes to the legacy copy should be evaluated for the `_metal2` copy. The fork is intentionally short-lived; keeping the two in sync is the cost of unblocking the bulk-port wave without coordinating dozens of consumer PRs.

A kernel-source change that compiles cleanly for the porting op but breaks a sibling op's CTA layout is one of the failure modes that escapes the recipe's anti-pattern self-audit — it's *not* in the ported op's own files. The legacy-inventory step explicitly notes any cross-op kernel sources, and the port report records the touch under "Open items for downstream" with the chosen path (in-place co-migration list, or fork) so the next sibling-op porter sees the coordination signal.

**Excluded from "cross-op"**: kernel-lib (`ttnn/cpp/ttnn/kernel_lib/`) and standard framework APIs (`tt_metal/hw/inc/api/...`) — these are out of porter scope by the [recipe's scope boundary](port_op_to_metal2_recipe.md#read-this-first) and the porter does not modify them. Cross-op kernels are sources living in *another op's* kernel directory.
