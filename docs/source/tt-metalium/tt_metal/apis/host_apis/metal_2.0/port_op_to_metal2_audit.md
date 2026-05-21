# Porting an Op to Metal 2.0 — Feasibility Audit

> This is the first of two documents covering the Metal 2.0 op port workflow. **This document covers the feasibility audit only — the gate that decides whether a given op can be ported today.** The port recipe (inventory, planning, construction, verification) lives in [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md) and is loaded only after the audit clears with explicit user go-ahead.
>
> **Last validated against main**: commit `ddbdd3c7ba5` (2026-05-20). Appendix A entries reflect Metal 2.0 feature support as of that commit. If you observe a feature in the codebase whose Appendix A status seems stale — particularly an `UNSUPPORTED` entry whose API has clearly landed in the framework headers — see [§Maintenance: keeping Appendix A current](#maintenance-keeping-appendix-a-current) for the override rule.

## Read this first

**Why this audit exists.** Metal 2.0 is incomplete. Past attempts to port ops that weren't ready have produced wasted human and agent time, broken code, and PRs that had to be rolled back. This audit is the safeguard: it determines whether a given op *can* be ported today, before any port work begins. When the audit's actual finding is "no, not yet," a clearly grounded refusal — with file references and reasons — is a complete, valid deliverable. Producing such a refusal is not a half-finished port; it *is* the work. Equally, when the actual finding is GREEN, that's the deliverable. Your job is to follow the evidence; no thumb on the scale either way.

**Audience**: AI agents asked to determine whether a TTNN op can be ported from the **`ProgramDescriptor` API** to the Metal 2.0 host API. Humans looking for a conceptual map of API differences should read [`metal2_migration_guide.md`](metal2_migration_guide.md) instead.

**Scope**: This guide is for porting **TTNN ops that target Gen1 architectures** (Wormhole / WH, Blackhole / BH) from the **`ProgramDescriptor` API** to the Metal 2.0 host API, on the same Gen1 target.

The op being ported **must already be on the `ProgramDescriptor` API** — i.e. its program-factory code populates a `ProgramDescriptor` and uses `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`, etc. Ops still on the older imperative-builder style from `host_api.hpp` (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` / etc.) are **out of scope for this guide**. `ProgramDescriptor` migration is a substantial, separate body of work with TTNN-infrastructure implications; it is a **prerequisite** to Metal 2.0 porting and lives in its own PR. If you encounter this case during an audit, report it as a blocker and stop — do not bundle the prereq work into this audit or the eventual Metal 2.0 port.

This guide is **not** for the following adjacent tasks. If your task is one of these, stop and surface the mismatch to the user — do not use this guide:

- **Porting an op that is still on legacy `host_api.hpp`** (i.e. not yet on `ProgramDescriptor`). `ProgramDescriptor` migration is a prerequisite to Metal 2.0 porting and is its own PR. Report and stop.
- **Porting from Gen1 to Quasar** (different target architecture, different threading model). Out of scope entirely.
- **Porting legacy Quasar tests** (those built against the temporary `experimental::quasar::CreateKernel` / `experimental::dfb::CreateDataflowBuffer` APIs) to Metal 2.0. A separate guide will cover that case; it is not this guide.

If you are unsure whether your task fits the in-scope description, ask the user before proceeding.

**Operating principle**: Your job is to identify gaps, not to invent solutions for unimplemented features.

Metal 2.0 is incomplete. Many features the legacy API supports are not yet available in Metal 2.0. When you encounter such a feature, the correct response is to **refuse the port and report the gap to the user.** Refusing is not a failure mode — it is the correct outcome when a gate fails.

If you find yourself constructing a clever workaround for something that "doesn't quite fit" in Metal 2.0 — packing data into varargs to simulate a missing field, threading a buffer address through an RTA because the binding mechanism doesn't support what you need, hand-rolling a synchronization primitive because the spec doesn't expose one — **stop**. AI agents have done all of these in real port attempts; the resulting code looks plausible, compiles cleanly, and fails subtly in production or behaves correctly in tests but represents the wrong design. Whatever you are about to write is almost certainly wrong. Surface the problem; do not paper over it.

When in doubt about feature support, **ask the user.** Do not infer support from API surface — the absence of a compile error does not mean the construct is supported.

## Workflow at a glance

Porting an op is a workflow split across two documents:

1. **Feasibility audit.** *(This document.)* Decide whether this op can be ported at all. Output: write `METAL2_PREPORT_AUDIT.md` to the op directory, then STOP.
2. **Port recipe** — legacy inventory, spec planning, construction, and verification. Lives in [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md). Loaded only after the audit clears with explicit user go-ahead.

This audit document covers the feasibility audit only. **Your job in this document is to decide whether the port is feasible — not to perform it.** Producing the audit report and stopping is the complete deliverable. The recipe document is loaded as a separate step, after the user has reviewed your audit and explicitly asked you to proceed.

You do not skip the audit. You do not pre-load the recipe document. The audit is its own unit of work.

---

## Feasibility audit

For the op in scope, run through two checks. Each check has three possible outcomes:

- **Green** — proceed past this check.
- **Yellow** — requires user judgment (ambiguous signal, or a supported-but-trade-off construct). Ask the user; respect the answer.
- **Red** — STOP. Record the reason in the audit report.

**Scope of the audit.**

- **Follow kernel references, not directory boundaries.** Audit every kernel referenced by any `KernelDescriptor::kernel_source` in the op's program factories — cross-op kernels living in adjacent directories (e.g. `eltwise/`, `data_movement/`, `kernels/dataflow/`) are in scope when the op uses them.
- **Unreferenced kernel files in the op's directory are out of scope.** If the op's directory contains kernel files that no factory references (dead code, tests, work-in-progress), do not audit their contents. If their presence could confuse a reader of the report, mention them in the identifying section as unreferenced; otherwise ignore them.
- **Multiple device-operations in one op directory.** If the directory contains more than one `DeviceOperation` type sharing factories or kernels (e.g. `ReduceDeviceOperation` plus `WelfordReduceDeviceOperation`), audit them together and produce a single combined report. If the device-operations are independent, audit each separately. Ask the user if unsure whether to bundle.
- **`RuntimeArgsDescriptor` and runtime-arg setup are not audit signals.** RTAs translate directly to `KernelSpec::runtime_arguments_schema` and `ProgramRunParams`; treat them as routine port work, not gates. In particular, **buffer-address RTAs in an otherwise-clean `ProgramDescriptor` op are normal here** — the `tensor.buffer()->address()` pattern in an RTA, paired with the corresponding `get_arg_val<uint32_t>(0)` on the kernel side, is the standard `ProgramDescriptor`-era idiom that the Metal 2.0 tensor-binding mechanism subsumes during the port. Do not flag this as an audit issue. (The recipe doc's anti-pattern against buffer-address RTAs applies *after* the binding mechanism is in scope; it is not an audit gate.)

A red on any check fails the audit. Do not attempt to port a red op (modulo the scoped-subset case described in the report-output section).

### Step 0.1 — Porting prerequisites

Metal 2.0 migration sits at the end of a chain of prior modernizations. Three prereqs must be confirmed before porting can begin. They differ in scope:

- **Standalone prereq** (Check 1 below): `ProgramDescriptor` migration is a substantial, separate body of work with TTNN-infrastructure implications. If unmet, it is its own PR — it does not bundle with the other prereq checks or with the Metal 2.0 port. During this audit, report the gap and stop; do not attempt the migration here.
- **Bundled prereqs** (Checks 2 and 3 below): smaller, mechanical kernel-side work. If unmet, address in a **single bundled prereq PR**, separate from the Metal 2.0 port. Bundle the two together if both are unmet — they're conceptually one body of work (kernel-side modernization). Yellow sub-cases (see Check 3) require user judgment.

**Check 1 (standalone prereq): Op is on the `ProgramDescriptor` API.**

Confirm the op's program-factory code populates a `ProgramDescriptor` and uses `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`, etc. — *not* the older imperative-builder style from `host_api.hpp` (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` / etc.).

- **Green**: op uses the `ProgramDescriptor` API.
- **Red**: op uses the imperative `host_api.hpp` builder API (not `ProgramDescriptor`). STOP. Report to the user that `ProgramDescriptor` migration is a **prerequisite to Metal 2.0 porting** — a substantial, standalone body of work with TTNN-infrastructure implications, addressed in its own PR. The audit's deliverable here is the report identifying the prereq; the prereq work itself is a separate session. **Do not attempt it as part of this audit, do not bundle it with anything, do not propose a partial conversion.**

**Check 2 (AI-doable): Device 2.0 Data Movement migration.**

Confirm all kernel-side data-movement code in this op is Device 2.0 compliant. See the [Device 2.0 Data Movement migration guide](../../kernel_apis/data_movement/device_api_migration_guide.md) for what that entails.

- **Green**: kernels are Device 2.0 compliant.
- **Yellow — substantively compliant with isolated legacy holdovers.** Kernel uses `experimental::Noc`, `experimental::CircularBuffer`, etc. for the bulk of operations and has a small number of isolated legacy holdovers from the **CB-index-keyed free-function family**: free functions taking a `uint32_t` CB index where the corresponding Device-2.0 wrapper object is already in scope at the call site. The family includes (non-exhaustively) `get_read_ptr(cb_id)`, `get_write_ptr(cb_id)`, `get_tile_size(cb_id)`, and similar `cb_*` helpers in the same shape. The defining characteristic is the *shape* — single CB-index argument, wrapper already in scope — not the specific function name; if you encounter a free function in that shape, treat it as a holdover.

  Each holdover is a 1-line mechanical replacement (e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`). Report yellow with `file:line` for each holdover and ask the user how to handle them. The yellow tier applies when the holdovers are isolated within a kernel that otherwise consistently uses the wrappers; absolute count is a heuristic, not a rule. **Default recommendation: fold into the Metal 2.0 port as port-time cleanup** rather than spinning a separate prereq PR — the change is trivial and decoupling adds churn.
- **Red**: kernels broadly use legacy Device 1.0 idioms (raw `noc_async_read`, manual CB index management, etc.). The Device 2.0 migration is a separate, prior body of work; do not bundle it with the Metal 2.0 port. May bundle with Check 3 in one prereq PR.

**Check 3 (AI-doable): TensorAccessor in use for every tensor read.**

For each kernel that reads tensor data directly (i.e. via host-managed `Buffer` addresses), identify how it accesses the tensor. **Compute kernels that only consume from / produce to circular buffers are out of scope for Check 3** — they read CB pointers, not tensor memory, and the tensor read happens upstream in a dataflow kernel. Audit only the kernels that actually touch tensor memory.

**Causal-link gate (run this first).** Before classifying any kernel under the cases below, check whether its tensor-access pattern is a **borrowed-memory DFB read**: the kernel reads tensor data through `cb_*.wait_front` / `cb_*.get_read_ptr` from a CB that is itself a borrowed-memory CB (see Step 0.2's [Dynamic CircularBuffer entry](#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed)). In that case, the kernel's lack of `TensorAccessor` is *intended* — the borrowed-memory DFB **is** the tensor access — and the port handles it via `DataflowBufferSpec::borrowed_from`. Do **not** classify the kernel under any of the cases below; the borrowed-memory DFB plan handles it.

Step-ordering tip: if a kernel involves sharded code paths or reads from a CB rather than via `TensorAccessor`, scan Step 0.2's Dynamic CircularBuffer rule for the same code path *before* finalizing your Check 3 classification. The borrowed-memory DFB pattern is the supported replacement for the legacy dynamic CB, and kernels reading from it correctly do not use `TensorAccessor` — mis-classifying such a kernel as RED ("convert to TensorAccessor") would be a regression.

**For kernels that pass the causal-link gate** (i.e. the lack of `TensorAccessor` is not downstream of a dynamic CB), classify by one of these three cases:

- **Green — uses `TensorAccessor`** (with `TensorAccessorArgs<N>()` plumbing on the host side and `TensorAccessor(args, addr)` on the device side). Ready for the port.
- **Red — doesn't use `TensorAccessor`, but the access pattern is page-by-page or otherwise iteratable.** Convert the kernel to use `TensorAccessor` in a separate, prior PR (may bundle with Check 2). Then return to this guide.
- **Yellow — doesn't use `TensorAccessor`, and the access pattern genuinely cannot be expressed via `TensorAccessor`** (exotic NoC walks; sub-page access; address arithmetic the iterators don't support). Porting is possible without `TensorAccessor` for this kernel, but the user must make an explicit call. Report yellow and surface the following rationale to the user, **verbatim**:

  > The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support.

  **Do not self-classify into this bucket.** AI agents tend to misclassify the previous case (kernel laziness or pre-`TensorAccessor` cruft) as this one. Always confirm with the user before treating a kernel as genuinely exotic — assume the previous case until the user confirms otherwise. On user override (proceed without `TensorAccessor`), the kernel will need a buffer-address RTA threaded through during the port; treat this as the documented escape hatch rather than a workaround.

### Step 0.2 — Feature compatibility check

Some legacy-API features are not yet supported in Metal 2.0. If the op uses any such feature, it cannot be ported until support lands.

For each entry in [Appendix A: Metal 2.0 feature compatibility](#appendix-a-metal-20-feature-compatibility), scan the op (host code, kernel code, factory functions, descriptors) using the recognition signals listed for that feature. Each entry declares its tier in the header — `UNSUPPORTED` (red action: refuse and wait for support), `DISCOURAGED` (yellow action: ask the user; respect the override), or `LANDED` (green: feature is supported in Metal 2.0 as of the doc's "Last validated against main" commit; no port gate).

- **Green**: no entry's recognition signals fire.
- **Yellow**: either a `DISCOURAGED` entry's signals match, **or** an `UNSUPPORTED` entry's signals match ambiguously (you cannot be sure whether the feature is in use). Ask the user. On override, proceed per the entry's guidance.
- **Red**: an `UNSUPPORTED` entry's signals match definitively. Report the feature name, the `file:line` where it appears, and the recognition signal that fired. Do not proceed.

If the op uses something *not listed* in Appendix A and you are uncertain of its support status, treat as yellow and ask. Do not assume support from API surface.

### Output: the audit report

The audit produces a written report. Write the report as `METAL2_PREPORT_AUDIT.md` in the op's directory (alongside the program factory `.cpp` files). This file is the audit's deliverable and is committed alongside the port — it sits next to `METAL2_PORT_PLAN.md` and `METAL2_PORT_REPORT.md` (both written by the port recipe), so all generated docs for the port land in one spot.

**Important framing for the human reader.** RED entries gate *this specific port attempt*, but they are **not permanent blockers**. Most RED entries mean "Metal 2.0 hasn't implemented this yet" — the port will become possible once the missing feature lands. A few (today: just `address_offset`) require a runtime-team consultation about a redesigned API. Each Appendix A entry's **Status** field describes the future path. **You must surface that future path explicitly in the report** for every RED row, so the human reader does not misread RED as "this op can never be ported." Reassuring framing matters here — a colleague seeing the report should understand the path forward, not just the gate.

**Code-path scope.** Blockers are often confined to specific code paths within an op (e.g., a single factory's `if (use_width_sharding)` branch). When this is the case, **explicitly identify which code paths are clean vs. blocked** in the report, and offer the user the option of a scoped-subset port — e.g., "interleaved-only paths, omitting the sharded path." A partial port that delivers value now may be preferable to waiting for the full upstream gate to clear. The Overall line should reflect this when applicable: `RED at op level; subset <X> is clear` rather than just `RED`.

**Output format.** The file is the deliverable — in chat, surface only the **Result** line plus the file path so the user can open the file when ready. Do not paste the full report inline; an audit of any non-trivial op runs to dozens or hundreds of lines and chat-scrollback isn't the right home for it.

Markdown formatting is required, not optional — the headers, tables, and inline-code spans are what make a sizeable report skim-friendly for a human reviewer. Use:

- H1 for the audit's title.
- H2 for major sections (Result, Porting prerequisites, Feature compatibility check, Path forward, Questions for the user).
- H3 for sub-sections (per-prereq-check headings; per-feature detail sections).
- **Tables** for the Device 2.0 DM holdover list (when present) and the Feature compatibility check summary. These are non-negotiable — flat bullet lists at this scale lose readability fast.
- Inline `code formatting` for file paths, function names, type names, identifiers throughout.

The conclusion appears at the top so a colleague glancing at the report sees the bottom line first; the detailed sections follow. **Required structure** (extend any cell, row, or paragraph with multi-line context where it improves clarity):

````markdown
# Pre-port audit: `<op path or qualified name>`

<Identifying section: device operations sharing the directory, factory file list, anything else needed to disambiguate. For multi-device-op directories, use a nested bullet list — outer bullets for device-operations, inner bullets for their program factories. Example shape:

- **`ReduceDeviceOperation`**
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`**
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)
>

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN | YELLOW | RED** — <one-line summary>.

- For **RED**: state the **primary blocker(s)** in plain language. If blockers are localized to specific code paths, name a clean subset that could be ported as a scoped subset.
- For **YELLOW**: state the count of open questions and reference the Questions for the user section.
- For **GREEN**: note that handoff to the recipe doc is appropriate after explicit user go-ahead.

### Clean subset (omit unless RED with localized blockers)

<List the parts of the op that are clean and could be ported as a scoped subset; list the parts that would be left on the legacy API.>

### Yellow side-issues (omit if none)

<Brief list with pointers to the Questions for the user section.>

## Porting prerequisites

### ProgramDescriptor API: **GREEN | RED**

<Findings. For RED, name the imperative-API calls that disqualify and state that the `ProgramDescriptor` migration is a prerequisite to Metal 2.0 porting — a substantial, standalone body of work addressed in its own PR.>

### Device 2.0 DM: **GREEN | YELLOW (isolated holdovers) | RED**

<For YELLOW, list the holdovers as a table:>

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `<path>` | `<n>` | `<call>` | `<wrapper>` |
| ... | ... | ... | ... |

<Default recommendation: port-time cleanup unless the user requests a separate prereq PR.>

### TensorAccessor usage: **GREEN | YELLOW | RED**

<Findings. For in-scope-subset cases, say so explicitly. Include the causal-link note when applicable.>

## Feature compatibility check

Every entry from Appendix A appears in this summary table, in the same order as Appendix A:

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | GREEN | |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | (LANDED in Appendix A; if the op uses this, the port uses `borrowed_from`) |
| CBDescriptor `address_offset` (non-zero) | RED | see detail below |
| Aliased Circular Buffers | GREEN | |
| GlobalSemaphore | GREEN | |
| Non-zero semaphore initial value | GREEN | |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | |
| `UpdateCircularBuffer*` | GREEN | |

For each non-GREEN row, follow up with an H3 detail section. Omit detail sections for GREEN rows.

### CBDescriptor `address_offset` (non-zero): **RED**

**Signal:** <what fired the recognition rule>

**Sites:**

- `<file>:<line>` — `<call site context>`
- ...

**Expected resolution:** <one-line summary derived from the Appendix A entry's Status field>

## Path forward

<Omit for GREEN. For RED, name the resolutions for each RED entry — typically "wait for Metal 2.0 feature(s) X, Y to land," but see individual entry Status fields for cases that need redesign discussions instead. If blockers are localized to specific code paths, describe the scoped-port option and any hybrid-file consequences. For YELLOW, restate the open questions and what proceeding under each likely answer would entail.>

## Questions for the user

<Omit if none.>

1. **<short title>:** <question, with the `file:line` context that prompted it>
2. **<short title>:** <next question>
3. ...
````

For UNSUPPORTED feature-detail sections, the **Expected resolution** is usually a short paraphrase of the entry's Status field — e.g., "not yet supported in Metal 2.0; port will be possible once GlobalCircularBuffer support lands on `KernelSpec` / `DataflowBufferSpec`." For the `address_offset` entry specifically, the expected resolution is the runtime-team-consultation message; surface that verbatim per the entry's Action field.

**N/A vs. GREEN.** If an Appendix A entry's *precondition* is absent from the op (e.g., the op uses no semaphores at all, so `Non-zero semaphore initial value` cannot fire), report the row as `N/A` rather than a vacuous `GREEN`. Use `N/A` only when the entire feature category is absent; if the feature is present but clean, that's a GREEN.

**RED short-circuit: Check 1 fires.** When Check 1 (ProgramDescriptor API) fires RED, later checks become moot — there is no point doing a full audit of an op that cannot be ported in this state. **Do not stop with a one-line report.** Instead: fill in Checks 2 & 3 and the Feature compatibility check on a best-effort basis (the user will benefit from knowing what they'll face *after* the ProgramDescriptor migration clears), and mark each subsequent finding with a clear "(observed-but-moot until Check 1 clears)" caveat. Do not block the report on completing them; if any subsequent check is impractical to evaluate without the `ProgramDescriptor` translation, mark it `(deferred — re-evaluate after Check 1 clears)` and move on. The Result and Path forward sections still emphasize Check 1 as the primary blocker; the moot-with-caveat findings are forward-looking context.

Save the report file and surface its path to the user along with the Result line. **Stop here.** The audit file is the complete deliverable of this document.

### After the audit: what happens next

- **On RED**: this op cannot be ported in its current state. Surface the file path and Result; stop. Do not load the recipe document.
- **On YELLOW**: surface the file path, the Result, and the open questions. Wait for the user's decisions. On override, re-run the affected checks, update the report file in place, and confirm GREEN before any handoff.
- **On GREEN + explicit user go-ahead**: load [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md) to perform the port. Pass the audit report file as context to the next session — the port recipe needs to know which features and decisions cleared. Do not load the recipe document on your own initiative; the user must explicitly approve.

---

## Appendix A: Metal 2.0 feature compatibility

This appendix lists legacy-API features that gate the port. Each entry falls into one of three tiers, declared in the entry's header:

- **UNSUPPORTED** — Metal 2.0 does not currently support this feature. Action: refuse the port and report (red). Each entry's **Status** field describes the future path: most entries will be supported as-is when implemented; a few will only be addressable via a redesigned, semantically different construct (and may require a runtime-team consultation before re-attempting). Always check the Status field before telling the user "wait and revisit."
- **DISCOURAGED** — Metal 2.0 supports the feature today, but its use is discouraged in favor of a planned alternative. Action: report yellow and ask the user; if the user overrides, proceed per the entry's guidance.
- **LANDED** — Metal 2.0 supports the feature as of the doc's "Last validated against main" commit. Action: no port gate; the feature is supported. The entry's **Status** field names the Metal 2.0 construct that replaces the legacy form.

### Maintenance: keeping Appendix A current

Appendix A entries reflect Metal 2.0 feature support as of the **Last validated against main** commit declared at the top of this document. When the framework changes Metal 2.0 feature support, the doc maintainer updates the relevant entry — typically by changing the tier (e.g., from `UNSUPPORTED` to `LANDED`), rewriting the Status / Action paragraphs to reference the new construct, and bumping the doc's `Last validated against main` commit hash.

**Staleness override for porting AIs.** If during the audit you observe a feature in the codebase whose Appendix A entry is marked `UNSUPPORTED` but the framework headers clearly show the API has landed (e.g., the spec/field/method the legacy construct would need to translate to is *visibly present* in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`), this likely means the audit doc is stale. Do not refuse the port reflexively. Instead:

1. Report the row as **YELLOW (staleness override)** rather than RED.
2. In the report's Questions for the user section, flag the apparent discrepancy: cite the Appendix A row, name the framework header / commit that contradicts it, and ask the user to confirm whether the feature is now supported.
3. Respect the user's answer; if they confirm support, proceed with the port using the new construct. The doc maintainer will update Appendix A separately.

This override mechanism exists because Metal 2.0 is moving fast and the audit doc lags reality. Do not invent it for entries that are clearly still unsupported (no API surface present); only for cases where the codebase contradicts the doc.

When scanning during Step 0.2 of the feature compatibility check, match each feature's recognition signals against the op's source. If any signal matches, take the action declared in the entry.

> **For maintainers adding new entries — skim if you're applying the recipe, not editing it.** Features whose underlying *functionality* Metal 2.0 will *never* support are handled differently: they are either reclassified as a prereq fix (the legacy use is replaced before porting) or get a dedicated fix-up recipe in the port recipe. They do *not* live here, because the action for them is not "wait" or "ask" — it is "transform." (Features whose *current API form* will not be supported but whose *underlying functionality* will be — via a different construct — do belong here as UNSUPPORTED entries; the entry's Status field calls out the redesign requirement.) If you are about to add an entry and the underlying functionality has no planned support, route it to one of those other locations instead.

Each entry follows this uniform format:
- **Status** — support state and tier framing.
- **Recognition — definitely this feature** — signals that, if matched, mean the feature is in use. Trigger the entry's action.
- **Recognition — false-positive guard** — superficially similar constructs that are *not* this feature. Do not trigger the action on these.
- **Action** — what to do when the rule fires (refuse vs. ask).
- **Examples in the wild** — real op locations using this feature, for ground-truthing your match.

If your op uses something not listed here and you are unsure of its support status, treat as yellow and ask the user. Do not assume support from API surface alone.

### GlobalCircularBuffer — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. No equivalent of `experimental::GlobalCircularBuffer` on `KernelSpec` or `DataflowBufferSpec`.

**Recognition — definitely this feature** (refuse and report):

- Any reference to the type `tt::tt_metal::experimental::GlobalCircularBuffer` (qualified or via a `using` alias).
- Calls to `experimental::CreateGlobalCircularBuffer(...)`.
- `#include <tt-metalium/global_circular_buffer.hpp>` paired with any of the other signals (header presence alone is suggestive but not definitive).
- **Descriptor-API attachment**: a `CBDescriptor` literal or struct with its `.global_circular_buffer` field set to a non-null pointer. The type token does not appear at the assignment site — look for the **field name** `global_circular_buffer` on a `CBDescriptor`. This is the arcane signal; an AI scanning a `CBDescriptor` setup can easily miss it.
- **Imperative-API attachment**: the `UpdateDynamicCircularBufferAddress(program, cb_handle, const GlobalCircularBuffer&)` overload (the three-arg form taking a `GlobalCircularBuffer`). The two-arg form taking a `Buffer&` is unrelated.
- Op factory function signatures with parameter type `std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>&` (commonly named `global_cb`).

**Recognition — false-positive guard**:

Plain `CircularBuffer`, `CBHandle`, `CBDescriptor`, or `CBFormatDescriptor` *without* the GCB attachment field set are the regular path → supported in Metal 2.0 as `DataflowBufferSpec`. Do not refuse these. The disambiguator is either the literal token `Global` in the type name **or** the `global_circular_buffer` field on a `CBDescriptor` being non-null.

**Action**: STOP. Report to the user that this op uses `GlobalCircularBuffer`, which is not yet supported in Metal 2.0. Do not invent a workaround.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/`

### Dynamic CircularBuffer (CB built on borrowed Buffer memory) — LANDED

**Status**: Supported in Metal 2.0. The legacy "dynamic circular buffer" pattern — `CBDescriptor::buffer = <some_buffer>` placing a CB on top of an existing `Buffer`'s memory — translates to Metal 2.0 as a **borrowed-memory DFB** via `DataflowBufferSpec::borrowed_from`. See `tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:95` for the spec field.

**Recognition — definitely this feature** (no port gate; use borrowed-memory DFB):

- **Descriptor-API** (the in-scope path): a `CBDescriptor` literal or struct assignment with its `.buffer` field set to a non-null `Buffer*` (any expression that is not statically `nullptr`). The type token does not appear at the assignment site — look for the **field name** `buffer` on a `CBDescriptor`. The companion field `.address_offset` is meaningful only when `.buffer` is set; it is not an independent signal.
- **Imperative-API** (the op's own program-factory code using these will already be flagged red by Step 0.1 Check 1, but they can also leak in via shared utility code — e.g. `cb_utils.hpp` — that the op calls):
  - `CircularBufferConfig::set_globally_allocated_address(buffer)`
  - `CircularBufferConfig::set_globally_allocated_address_and_total_size(buffer, total_size)`
  - The three-argument constructor `CircularBufferConfig(total_size, data_format_spec, buffer)`

**Recognition — false-positive guard**:

- A `CBDescriptor` with `.buffer = nullptr` (or with `.buffer` simply not set) is a regular CB → standard `DataflowBufferSpec` (no `borrowed_from`). Do not require the borrowed-memory path for these.
- The `.global_circular_buffer` field on `CBDescriptor` is a *different* feature, covered by the GlobalCircularBuffer rule above. Do not conflate the two — `.buffer` and `.global_circular_buffer` are independent fields on `CBDescriptor` with different meanings.
- A `CircularBufferConfig` constructed via the one-argument form `CircularBufferConfig(total_size)` followed by `set_page_size(...)` calls (no `set_globally_allocated_address`, no three-arg constructor) is a regular static CB → standard `DataflowBufferSpec`.

**Action**: Proceed with the port. On the Metal 2.0 side, declare the affected `DataflowBufferSpec` with `borrowed_from = <tensor_parameter_name>` naming the `TensorParameter` whose buffer backs the DFB. The DFB's handle (`dfb::name`) resolves to the borrowed L1 address at runtime; the kernel-side code that previously read from the borrowed-memory CB continues to work via the DFB wrapper.

**Examples in the wild** (op locations whose port exercises this construct):
- Descriptor-API form (`CBDescriptor::buffer` set):
  - `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp`
  - `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_*_program_factory.cpp` (sharded, mcast, no_mcast)
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_*_program_factory.cpp`
- Imperative-API form (`set_globally_allocated_address`, often via shared utilities):
  - `ttnn/cpp/ttnn/operations/cb_utils.hpp` (utility; called from many ops)
  - `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/experimental/plusone/device/plusone_program_factory.cpp`

### CBDescriptor `address_offset` set to non-zero — UNSUPPORTED (current form not planned)

**Status**: Not supported in Metal 2.0, and **not slated for support in its current form**. The `address_offset` field on `CBDescriptor` is a recently introduced interim mechanism for placing a CB at a non-zero offset within a `Buffer` (or at an absolute address, when used without a `buffer`). Metal 2.0 will not carry this construct forward as-is. The functional capability the field provides will be available in a **semantically different way** in Metal 2.0; a direct translation is not possible. The user must consult the runtime team about their actual use case before this op can be ported.

This rule will most often fire together with the Dynamic CircularBuffer rule above (offset is most often used with a Buffer-backed CB), but it is checked separately because:
- It is severable: a CB can use a non-zero `address_offset` without a Buffer-backed CB (placing the CB at an absolute L1 address — the "manually placed CB" mode documented in `ttnn/api/ttnn/tensor/tensor_utils.hpp`).
- It carries a stronger signal than dynamic CB alone — the form is being phased out, not just "not yet implemented."

**Recognition — definitely this feature** (refuse and report; flag prominently):

- A `CBDescriptor` literal or struct assignment with `.address_offset` set to a non-zero literal or any expression that is not statically `0`.
- `CircularBufferConfig::set_address_offset(non_zero)` (imperative API; will already be flagged red by Step 0.1 Check 1, but worth recognizing in shared utility code).
- `UpdateDynamicCircularBufferAddress(program, cb_handle, buffer, offset)` — the four-argument overload — when `offset` is non-zero.
- Calls to helpers like `cb_descriptor_from_sharded_tensor(cb_index, tensor, address_offset, ...)` in `ttnn/api/ttnn/tensor/tensor_utils.hpp` where the third argument (`address_offset`) is passed a non-zero value.

**Recognition — false-positive guard**:

- `.address_offset = 0` or `.address_offset` not set (default zero) is fine for *this* rule → green. (The Dynamic CircularBuffer rule may still fire if `.buffer` is set; check independently.)
- `UpdateDynamicCircularBufferAddress(program, cb_handle, buffer)` — three-argument form with no offset → not this rule.
- Kernel-side `bank_address_offset` parameters on calls like `get_noc_addr_from_bank_id<...>(bank_id, bank_address_offset)` are an unrelated kernel-side feature → not this rule.

**Action**: STOP. **Flag prominently in the audit report** — do not bury this among other RED entries. Use stronger emphasis than for routine UNSUPPORTED items, and surface the following message to the user **verbatim**:

> The `address_offset` field is a recently introduced interim mechanism that will not survive into Metal 2.0 in its current form. The underlying capability you need will be available, but only via a different API that is not a direct translation. Please reach out to the runtime team to discuss your use case before proceeding with this port.

Recommended report shape when this rule fires:

```
*** CRITICAL: CBDescriptor address_offset feature in use ***
File: <file:line>
[verbatim message above]
```

Do not invent a workaround. Do not propose an alternative implementation. Do not attempt a partial port. The correct path is a runtime-team consultation, then revisit.

**Examples in the wild** (for ground-truthing your match):

`address_offset` was introduced recently and has limited adoption in checked-in code today. Likely paths to a non-zero usage:
- Direct `CBDescriptor` literal with `.address_offset = <non-zero>`.
- Helper `cb_descriptor_from_sharded_tensor(cb_index, tensor, address_offset, ...)` in `ttnn/api/ttnn/tensor/tensor_utils.hpp` — inspect callers for non-zero third arguments.
- Python op authors using the nanobind-exposed `address_offset` parameter on `CBDescriptor` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp`).

If you find no concrete non-zero usage in the op being ported, this rule is green — that is the expected outcome for most ops today.

### Aliased Circular Buffers (CBs sharing backing memory) — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0; the equivalent feature is **aliased DFBs**, referenced in `tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp` but not yet implemented. The legacy API permits a single CB to back two or more *logically distinct* buffer indices using shared memory — the term "aliased CB" is descriptive (it does **not** appear in the legacy API surface). The legacy expression of this feature is unobvious: it lives in the array-shape of certain `CircularBufferConfig` fields (sized `NUM_CIRCULAR_BUFFERS`, almost always populated with one entry) and in the `SmallVector<CBFormatDescriptor, 1>` shape of `CBDescriptor::format_descriptors`. The signal is the *cardinality* of those collections, not any named field.

**Recognition — definitely this feature** (refuse and report):

- **Descriptor API** (the in-scope path): a `CBDescriptor` whose `format_descriptors` initializer contains **more than one** `CBFormatDescriptor` element. Concretely:
  - Single-element (normal): `format_descriptors = {{CBFormatDescriptor{...}}}` → green.
  - Multi-element (aliased): `format_descriptors = {{CBFormatDescriptor{...}, CBFormatDescriptor{...}}}` (or three+) → red.
  - The differing element is typically `buffer_index` — two distinct indices sharing the same backing storage.
- **Imperative API** (will already be flagged red by Step 0.1 Check 1 if used in the op's own program-factory; can also leak in via shared utility code):
  - `CircularBufferConfig(total_size, data_format_spec)` where `data_format_spec` is a `std::map<uint8_t, tt::DataFormat>` with **more than one** key (e.g. `{{idx1, fmt1}, {idx2, fmt2}}`).
  - Two or more `.set_page_size(buffer_index, ...)` calls **with different `buffer_index` values** chained on the same `CircularBufferConfig` instance.
  - Companion signal: `.set_tile_dims(buffer_index, ...)` chained with multiple distinct `buffer_index` values on the same config.

**Recognition — false-positive guard**:

- A file that creates *many* CBs, each with a single buffer index, is **not** aliased — aliased means a *single* config has multiple indices. Confirm the multiple `set_page_size` calls are on the *same* `CircularBufferConfig` instance, not on different ones.
- Single-element initializers (`{{CBFormatDescriptor{...}}}` for the descriptor form, single-key `{{idx, fmt}}` map for the imperative form) are the dominant pattern by a wide margin → green for this rule.
- The `CBDescriptor::remote_format_descriptors` field is a *different* concept (relates to remote DFBs, a separate planned feature) and is not covered by this rule. Multi-element values there have a different meaning; do not conflate.

**Action**: STOP. Report to the user that this op uses an aliased CB (multiple logical buffer indices sharing backing memory), which Metal 2.0 does not yet support — the planned "aliased DFBs" feature is referenced in `dataflow_buffer_spec.hpp` but not implemented. Do not invent a workaround; in particular, do **not** "split" the aliased CB into two independent CBs (changes memory footprint and may break the kernel's assumption that the indices share an address).

**Examples in the wild** (for ground-truthing your match):

The descriptor form (in-scope) is currently *not exercised* by any checked-in ttnn op — every `format_descriptors` initializer in current op factories is single-element. The descriptor API does support aliased CBs cleanly: the `CircularBufferConfig(const CBDescriptor&)` constructor at `tt_metal/impl/buffers/circular_buffer_config.cpp` iterates all elements of `format_descriptors` and preserves the multi-element semantics end-to-end. So a descriptor-form match is valid usage — just unusual, since no in-tree op has needed it yet. Treat as a red.

The imperative form (currently out of scope per Step 0.1 Check 1, but will move into scope once those ops are migrated to `ProgramDescriptor`) is used in:
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (around line 840 — output + interim sharing memory; has the comment "share buffer")
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`
- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp` (multiple sites — cos/sin interim/sync pairs)
- `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_multi_core_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp`

When porting one of these post-Pass-1, expect this rule to fire even though the imperative form was caught upstream.

### GlobalSemaphore — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. The Metal 2.0 source confirms this with a TODO at `tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp` (search for `TODO -- GlobalSemaphore bindings`).

**Recognition — definitely this feature** (refuse and report):

- Any reference to the type `tt::tt_metal::GlobalSemaphore` (qualified or via a `using` alias). Note: the type lives in plain `tt::tt_metal::`, not `experimental::`.
- Calls to `experimental::CreateGlobalSemaphore(...)`. Note: the factory *is* in `experimental::` even though the type is not.
- `#include <tt-metalium/global_semaphore.hpp>`.
- Op factory function signatures with parameter type `const GlobalSemaphore&` or `std::optional<GlobalSemaphore>` (commonly named `semaphore`, `global_semaphore`, `multi_device_global_semaphore`).

**Recognition — false-positive guard**:

Plain `Semaphore` / `CreateSemaphore(program, core_spec, initial_value)` is the regular semaphore path → supported in Metal 2.0 as `SemaphoreSpec`. Do not refuse these. The disambiguator is the literal token `Global` in the type name; if it is not there, this rule does not apply.

**Action**: STOP. Report to the user that this op uses `GlobalSemaphore`, which is not yet supported in Metal 2.0. Do not invent a workaround.

**Examples in the wild** (for ground-truthing your match):
- Most CCL ops under `ttnn/cpp/ttnn/operations/experimental/ccl/`, e.g. `llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp`, `all_gather_concat_heads_fused/device/`, `llama_all_gather_matmul_async/device/`.

### Non-zero semaphore initial value — DISCOURAGED

**Status**: Supported by Metal 2.0 on Gen1 today, but discouraged. The legacy path lets you create a semaphore with a non-zero initial value via `CreateSemaphore(program, core_spec, initial_value)` (where `initial_value != 0`) or by setting `SemaphoreDescriptor::initial_value` to a non-zero value. Metal 2.0's `SemaphoreSpec` accepts a non-zero `initial_value` on Gen1, so the translation works; however, the planned **Remote DFB** feature is intended to supplant this use case once it lands, at which point non-zero-init semaphores will likely be deprecated. Use of non-zero initial values today is therefore a yellow flag, not a red one.

**Recognition — definitely this feature** (report yellow; await user decision):

- `CreateSemaphore(program, core_spec, initial_value)` calls where `initial_value` is:
  - A non-zero integer literal (`1`, `2`, etc.).
  - A constant or symbol whose value is **not self-evidently zero** (e.g. `INVALID`, `INVALID_SEM`, project-specific sentinel constants). When in doubt, treat as yellow — do not assume the symbol is zero. If you can resolve the constant's definition and it is in fact zero, downgrade to green.
- `SemaphoreDescriptor` literals or struct assignments where `.initial_value` is set to a non-zero literal or a not-evidently-zero symbol.

**Recognition — false-positive guard**:

- `CreateSemaphore(program, core_spec, 0)` — explicit zero literal → green, no action.
- `SemaphoreDescriptor{ ..., .initial_value = 0 }` — explicit zero literal → green.
- `GlobalSemaphore` is a separate type, covered by its own rule above. Do not match this rule against `GlobalSemaphore` constructions or `experimental::CreateGlobalSemaphore(...)` calls.

**Action**: Report **yellow** in the audit. Include in the report:
- The semaphore creation site (`file:line`).
- The initial-value expression as written.
- A note that the construct is supported today but discouraged, and that the user may override and proceed.

**Do not refuse the port outright on this signal alone.** The user may decide to proceed; this is their call. On override: the translation is direct — set `SemaphoreSpec::initial_value` to the same value used in the legacy code. The port's mechanical work is unaffected.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp` (`INVALID` sentinel)
- `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp` (`INVALID` sentinel)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp` (literal `1`)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/moe_gpt_program_factory.cpp` (`INVALID` / `INVALID_SEM` sentinels)

### Runtime tensor-accessor flavors (RuntimeTensorShape, RuntimeRank, RuntimeNumBanks, RuntimeShardShape, RuntimeBankCoords) — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. The legacy `TensorAccessorArgs(buffer, ArgConfig::Runtime*)` family lets the host defer parts of the tensor accessor's metadata (shape, rank, bank info) to runtime, threading them through positional CTAs. Metal 2.0 has no positional-CTA mechanism, so this plumbing has no equivalent yet.

**Recognition — definitely this feature** (refuse and report):

- Any reference to one of the following enumerators in op host code:
  - `tensor_accessor::ArgConfig::RuntimeTensorShape`
  - `tensor_accessor::ArgConfig::RuntimeRank`
  - `tensor_accessor::ArgConfig::RuntimeNumBanks`
  - `tensor_accessor::ArgConfig::RuntimeShardShape`
  - `tensor_accessor::ArgConfig::RuntimeBankCoords`
- A canonical grep target that catches all five: `ArgConfig::Runtime`. If this token appears in op host code (not in `tt_metal/impl/buffers/tensor_accessor_args.cpp`, which is the implementation file), the rule fires.
- Calls of the form `TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::Runtime*)` — i.e. the two-argument form of `TensorAccessorArgs` whose second argument is one of the `Runtime*` flavors. The most common appearance is `.append_to(...)` or `.get_common_runtime_args()` immediately after.

In practice, `RuntimeTensorShape` is the dominant flavor; the other four have ~zero user sites outside tests. A grep for `ArgConfig::Runtime` will surface all of them in one pass.

**Recognition — false-positive guard**:

The single-argument form `TensorAccessorArgs(buffer)` (with no second arg, or with a non-`Runtime*` `ArgConfig` value) is the standard path → supported in Metal 2.0 via `TensorParameter` + `TensorBinding`. Do not refuse these. The disambiguator is the literal token `Runtime` on an `ArgConfig::` qualifier.

**Action**: STOP. Report to the user that this op uses one or more `ArgConfig::Runtime*` tensor-accessor flavors, which Metal 2.0 does not yet support. Do not invent a workaround. Note that the op may otherwise look ready for Metal 2.0 — the gating signal is just the `ArgConfig::Runtime*` token, possibly buried in an otherwise-standard `TensorAccessorArgs(...)` call chain.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_wh_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

### Per-execution CircularBuffer size updates (UpdateCircularBufferTotalSize, UpdateCircularBufferPageSize) — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. The legacy free functions `UpdateCircularBufferTotalSize(program, cb_handle, total_size)` and `UpdateCircularBufferPageSize(program, cb_handle, buffer_index, page_size)` mutate a CB's total size or per-buffer-index page size between Program executions. Metal 2.0 does not yet expose an equivalent — `ProgramRunParams` reserves placeholder fields for per-execution DFB size / entry size, but they are explicitly "not yet supported."

**Recognition — definitely this feature** (refuse and report):

- Calls to `UpdateCircularBufferTotalSize(program, cb_handle, total_size)`.
- Calls to `UpdateCircularBufferPageSize(program, cb_handle, buffer_index, page_size)`.
- A canonical grep target that catches both: `UpdateCircularBuffer`. (See guard below for one false positive.)
- Typical call site: cached-program override hooks (e.g. `override_runtime_arguments` and similar callbacks), where CB sizing is re-tuned per shape between executions of the same Program.

**Recognition — false-positive guard**:

`UpdateDynamicCircularBufferAddress` is a different function with different semantics — do not refuse based on the `UpdateCircularBuffer` substring matching it. (One of its overloads takes a `GlobalCircularBuffer` and is covered by the GlobalCircularBuffer rule above; the other takes a `Buffer&` and is outside the scope of this rule.)

**Action**: STOP. Report to the user that this op uses `UpdateCircularBufferTotalSize` and/or `UpdateCircularBufferPageSize`, which Metal 2.0 does not yet support. Do not invent a workaround. In particular, do not "fix" this by recreating the Program from scratch on every execution — that defeats the cached-program model the call site is part of and is not what the user wants.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/attn_matmul_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/group_attn_matmul_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp`

---

## After you submit

A grounded RED audit is not a failed port; it is the audit working as designed. The same goes for a YELLOW where you raised the question rather than guess. The deliverable here is clarity — what porting this op would actually require, surfaced clearly enough that a colleague can act on it. Your job in this document was to decide whether the port is feasible, not to perform it; once the report is on its way, that work is complete.
