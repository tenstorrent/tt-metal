# Porting an Op to Metal 2.0 — Feasibility Audit

> This is the first of two documents covering the Metal 2.0 op port workflow. **This document covers the feasibility audit only — the gate that decides whether a given op can be ported today.** The port recipe (inventory, planning, construction, verification) lives in [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md) and is loaded only after the audit clears with explicit user go-ahead.
>
> If you encounter a feature in the framework headers whose Appendix A status looks stale — most commonly, an `UNSUPPORTED` entry whose API has clearly landed — see [§Maintenance: keeping Appendix A current](#maintenance-keeping-appendix-a-current) for the override procedure.

## Read this first

**Why this audit exists.** Past attempts to port ops that weren't ready have produced wasted human and agent time, broken code, and PRs that had to be rolled back. This audit is the safeguard: it determines whether a given op *can* be ported today, before any port work begins. When the audit's actual finding is "no, not yet," a clearly grounded refusal — with file references and reasons — is a complete, valid deliverable. Producing such a refusal is not a half-finished port; it *is* the work. Equally, when the actual finding is GREEN, that's the deliverable. Your job is to follow the evidence; no thumb on the scale either way.

**Audience**: AI agents asked to determine whether a TTNN op can be ported from the **`ProgramDescriptor` API** to the Metal 2.0 host API. Humans looking for a conceptual map of API differences should read [`metal2_migration_guide.md`](../metal2_migration_guide.md) instead.

**If you're new to this stack — quick orientation:**

- **Tenstorrent accelerators** come in two architectural generations. **Gen1** is the shipping silicon today: `WH` = Wormhole, `BH` = Blackhole. **Gen2** is in development: `Quasar` (and siblings). This audit covers Gen1 ops; Metal 2.0 is designed to serve both architectures.
- **TTNN** is the high-level neural-network library for Tenstorrent accelerators. Ops live in `ttnn/cpp/ttnn/operations/<family>/<op>/`. A typical op has a device-operation class on the host side and one or more program factories that build what runs on the accelerator.
- **Metal 2.0** is the new **host API** — what the program factory uses to declare kernels, buffers, semaphores, and bindings. It also introduces **DFB** (Dataflow Buffer) at the spec layer, replacing the legacy **CB** (CircularBuffer); the two are essentially synonyms on Gen1, but DFB's semantics diverge meaningfully on Gen2.
- **Device 2.0** is a *separate, earlier* overhaul of the **kernel-side** data-movement APIs (safer, more object-oriented wrappers — `experimental::Noc`, kernel-side `CircularBuffer` wrappers, etc.). The [Prerequisites step](#prerequisites) gates on Device 2.0 migration — for the op's own kernels *and* any donor kernels it calls — as a hard prerequisite to Metal 2.0, but Device 2.0 is *not* part of Metal 2.0 itself.
- **`ProgramDescriptor` API** is a TTNN-side framework that ops must migrate to before a Metal 2.0 port becomes possible. The [Prerequisites step](#prerequisites) gates on it.
- **Common acronyms you'll see throughout:** `CB` = CircularBuffer; `DFB` = DataflowBuffer (see above); `RTA` = runtime args; `CTA` = compile-time args; `CRTA` = common runtime args (values broadcast to all nodes); `TA` = TensorAccessor; `LLK` = Low-Level Kernel (the framework-provided kernel-side primitives); `NoC` = Network-on-Chip (the on-die fabric); `SPSC` = single-producer / single-consumer (per CB instance, per node).

For the conceptual map of how Metal 2.0 abstractions fit together — `ProgramSpec`, `KernelSpec`, `TensorParameter` / `TensorBinding`, `DataflowBufferSpec`, the spec/run-args split — see [`metal2_migration_guide.md`](../metal2_migration_guide.md).

**Scope**: This guide is for **TTNN ops that target Gen1 architectures** (Wormhole / WH, Blackhole / BH), to assess Metal 2.0 portability and gather findings about what the port will require.

The audit produces useful findings for ops in two states:

- **Already on the `ProgramDescriptor` API.** The audit decides whether the Metal 2.0 port can proceed. GREEN → port can begin (after explicit user go-ahead).
- **Still on legacy `host_api.hpp` (imperative builder).** The audit RED's the ProgramDescriptor prerequisite and continues — the remaining subjects still surface findings that inform what the eventual Metal 2.0 port will need, once the `ProgramDescriptor` migration (a substantial, separate workstream) lands. This is data-gathering for downstream planning, not a port attempt.

In either case, the audit's deliverable is the report. Porting itself happens via the [port recipe](port_op_to_metal2_recipe.md), which is loaded only after the audit clears (GREEN) with explicit user go-ahead.

This guide is **not** for the following adjacent tasks. If your task is one of these, stop and surface the mismatch to the user — do not use this guide:

- **Porting from Gen1 to Quasar** (different target architecture, different threading model). Out of scope entirely.
- **Porting legacy Quasar tests** (those built against the temporary `experimental::quasar::CreateKernel` / `experimental::dfb::CreateDataflowBuffer` APIs) to Metal 2.0. A separate guide will cover that case; it is not this guide.

If you are unsure whether your task fits the in-scope description, ask the user before proceeding.

**Operating principle**: Your job is to identify gaps, not to invent solutions for unimplemented features.

Some features the legacy API supports are not yet available in Metal 2.0. When you encounter such a feature, the correct response is to **refuse the port and report the gap to the user.** Refusing is not a failure mode — it is the correct outcome when a gate fails.
When in doubt about feature support, **ask the user.** Do not infer support from API surface — the absence of a compile error does not mean the construct is supported.

**What happens to your report.** Each audit report becomes a direct input to a downstream effort — not just an entry in a tracking spreadsheet:

- A **RED** report feeds the **prereq-migration efforts**. The `ProgramDescriptor` migration team and the Device 2.0 migration team consume RED audits to scope and sequence the work that unblocks the Metal 2.0 port. A RED isn't a dead end — it's evidence routed to the team that resolves the gap.
- A **GREEN** report feeds the **Metal 2.0 port recipe** at [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md). The porter loads the audit as context when performing the port itself.
- A **YELLOW** report is **genuinely ambiguous** and is evaluated case-by-case with the user to decide which downstream path the op takes.

Your tier assignment, your subset suggestions, and the specificity of your finding details all carry weight in these downstream uses. Too-conservative RED misroutes work to a prereq team that doesn't need it; too-lenient GREEN sends a port attempt into a fail. The single strongest thing you can do is **be specific**: name files and lines; quote the construct you saw; describe what triggered the rule. Vague findings are the hardest to use downstream.

**About this recipe.** This recipe is the product of iteration — earlier auditors' observations have already shaped it, and yours can too. If during your audit a step feels unclear, a rule contradicts itself, the recipe doesn't anticipate a case you're hitting, or guidance conflicts with what you observe in the code, **write it down in the audit report's "Recipe notes" section** rather than silently picking an interpretation. Treat the recipe as your guide, not your shackle. The recipe maintainer reads every report; the friction you log makes the next auditor's job easier.

## Workflow at a glance

Porting an op is a workflow split across two documents:

1. **Feasibility audit.** *(This document.)* Assess this op's Metal 2.0 portability and capture findings — including what work will be required if the port can't proceed yet. Output: write `METAL2_PREPORT_AUDIT.md` to the op directory, then STOP.
2. **Port recipe** — legacy inventory, spec planning, construction, and verification. Lives in [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md). Loaded only after the audit clears with explicit user go-ahead.

This audit document covers the feasibility audit only. **Your job in this document is to decide whether the port is feasible — not to perform it.** Producing the audit report and stopping is the complete deliverable. The recipe document is loaded as a separate step, after the user has reviewed your audit and explicitly asked you to proceed.

You do not skip the audit. You do not pre-load the recipe document. The audit is its own unit of work.

---

## Feasibility audit

For the op in scope, work through the audit in eight subjects, in order: **[Prerequisites](#prerequisites)** (ProgramDescriptor + Device 2.0), **[Feature compatibility](#feature-compatibility)**, **[TensorAccessor handling](#tensoraccessor-handling)**, **[DFB endpoint legality](#dfb-endpoint-legality-spsc)**, **[Out-of-directory coupling](#out-of-directory-coupling)**, **[Custom program hash](#custom-program-hash)**, **[Other signals](#other-signals)**, and **[TTNN factory concept analysis](#ttnn-factory-concept-analysis)** (run last, since it draws on the others' findings). Each subject's checks have three possible outcomes:

- **Green** — proceed past this check.
- **Yellow** — requires user judgment (ambiguous signal, or a supported-but-trade-off construct). Ask the user; respect the answer.
- **Red** — record the reason in the audit report and continue the audit. A RED outcome means the port is blocked on this finding; it does not mean stop auditing. Always complete the remaining checks and steps so the report captures everything the port will eventually need to clear, not just the first blocker.

**Reference data (recommended).** Before working the subjects, fetch Diego's per-factory porting-readiness data (his *"Operations analysis"* sheet) and grep out your op's rows — it pre-classifies several of the signals you're about to check (factory concept, custom hash, RTA-smuggled pointers, pybind-of-internals, custom override-runtime-args). Treat it as an informative **prior, not ground truth**: let it orient your search, but your own `file:line` evidence decides every finding, and you note any place the sheet and your evidence disagree. Fetch procedure + column legend: [`../analyses/ttnn_op_porting_readiness.md`](../analyses/ttnn_op_porting_readiness.md).

**Scope of the audit.**

- **Follow kernel references, not directory boundaries.** Audit every kernel referenced by any `KernelDescriptor::kernel_source` in the op's program factories — cross-op kernels living in adjacent directories (e.g. `eltwise/`, `data_movement/`, `kernels/dataflow/`) are in scope when the op uses them.
- **Unreferenced kernel files in the op's directory are out of scope.** If the op's directory contains kernel files that no factory references (dead code, tests, work-in-progress), do not audit their contents. If their presence could confuse a reader of the report, mention them in the identifying section as unreferenced; otherwise ignore them.
- **Multiple device-operations in one op directory.** If the directory contains more than one `DeviceOperation` type sharing factories or kernels (e.g. `ReduceDeviceOperation` plus `WelfordReduceDeviceOperation`), audit them together and produce a single combined report. If the device-operations are independent, audit each separately. Ask the user if unsure whether to bundle. **When bundling, retain per-DeviceOperation attribution where findings differ** — name which DeviceOperation (or which of its factories) a given finding applies to, so a downstream consumer (per-op spreadsheet, ticket tracker, port planner) can extract per-DeviceOperation status from the bundled report when their accounting needs it. Bundling reflects the porting unit (shared code → shared port); downstream tools may legitimately operate at the DeviceOperation level (e.g. Tracy profiling, per-op leadership reporting).
- **Routine runtime-arg setup is not a general audit signal — [TensorAccessor handling](#tensoraccessor-handling) handles the one specific case.** Most RTAs translate directly to `KernelSpec::runtime_arg_schema` and `ProgramRunArgs`; treat them as routine port work, not gates. The historical `tensor.buffer()->address()`-as-RTA pattern is the one exception: pre–Metal 2.0 it was style-yuck-but-correct, but under TTNN's recent fast-path-cache binding-injection changes it is now a per-binding correctness hazard. The TensorAccessor-handling subject catches and reports buffer-address RTAs specifically (as Case 2 bindings); routine runtime-arg setup outside that pattern remains non-signal.

**Finding roles and routing.** Every audit finding carries one of four roles, and the role decides which output document it lands in (see [Output: the two documents](#output-the-two-documents)). This table is the audit's backbone — the per-check sections below *produce* these findings; they are not a separate set of rules.

| Finding | Role | Routing |
|---|---|---|
| Op on ProgramDescriptor API | **GATE** | brief: cleared/blocked · team: detail → ProgramDescriptor team |
| Device 2.0 compliance (own + donor kernels) | **GATE** | brief: cleared/blocked · team: exact violations → Device 2.0 team |
| UNSUPPORTED feature in use (incl. CTA varargs) | **GATE** | brief: cleared/blocked · team: detail → wait-for-feature |
| TTNN factory analysis — op-owned tensors · genuine MeshWorkload need | **FYI-U** | team only |
| TTNN factory analysis — pybind `create_descriptor` · other risky pybind · custom override-RTA | **FYI-P** | brief (Watch-for) + team |
| Per-binding TensorAccessor handling — Case 1 (`TensorAccessor`) / Case 2 (raw pointer → bridge) | **PORT WORK** | brief (Construct) + team |
| Delete custom `compute_program_hash` (→ default) | **PORT WORK** | brief (Construct) + team |
| Notable constructs — aliased CB / borrowed-mem DFB / dynamic TA (confusing); non-zero sem init (deprecated-but-fine) | **FYI-P** | brief (Watch-for) + team |
| Sync-free CB (address-only; no FIFO producer + consumer) — port applies the interim workaround | **FYI-P** | brief (Watch-for) + team |
| DFB endpoint legality — SPSC violation (hidden 2nd writer / multi-reader tensor-view on a node) | **GATE** (config-scoped) | brief: blocked / clean subset · team: detail + pre-port fix → op owner |
| DFB endpoint legality — dead CB (zero endpoints) | **FYI-P** | brief (Watch-for) + team |
| Cross-op / shared-kernel flags | **FYI-P** | brief (Watch-for) + team |
| RTA varargs | **FYI-P** | brief (Watch-for) + team |
| Out-of-directory coupling & donor shape analysis | **FYI-U** | team only |
| Tensor-parameter relaxation candidates (fallible) | **FYI-U** | team only |
| Incidental code anomalies — dead RTAs, dead-but-hashed attributes, suspicious constants | **FYI-U** | team only |

**The four roles:**
- **GATE** — blocks the port (an unmet prereq, an UNSUPPORTED feature, or an SPSC endpoint-legality violation needing an op-owner pre-port fix). On PASS, the porter brief carries a one-line "cleared"; on FAIL, *no brief is issued* (there is no port) and the detail routes to the owning team. (A *config-scoped* GATE — e.g. GlobalCircularBuffer or an SPSC violation confined to one factory/path — still issues a brief for the clean subset; see [Code-path scope](#output-the-two-documents).) Always complete every check even after a GATE fails — the report captures everything the port will eventually need.
- **PORT WORK** — the porter must *act* on it during the port.
- **FYI-P** — informational, surfaced *to the porter* (and recorded for the team).
- **FYI-U** — informational, *team-only* (feeds other workstreams; never reaches the porter).

Findings flow to the two output documents by role: the **porter brief** carries GATE-cleared lines + all PORT WORK + all FYI-P; the **team findings** doc carries everything. See [Output: the two documents](#output-the-two-documents).

### Prerequisites

Metal 2.0 migration sits at the end of a chain of prior modernizations. This subject confirms the two hard prerequisites — the **`ProgramDescriptor` API** and **Device 2.0 data-movement migration** — and **both GATE the port**:

- **Check 1 — `ProgramDescriptor` API** is the standalone hard prereq. If unmet, it is its own PR — substantial, separate work with TTNN-infrastructure implications that does *not* bundle with the Metal 2.0 port. Record the gap and continue the audit; do not attempt the migration here.
- **Check 2 — Device 2.0 migration** (the op's own kernels *and* any donor kernels it calls) is also a hard prereq: the Metal 2.0 binding tokens attach to Device 2.0 wrapper objects, so a kernel still on Device 1.0 idioms cannot take the whitelisted swaps.

**Complete both checks regardless of individual outcomes, then continue through the remaining subjects.** The audit's job is to gather a complete picture of what porting this op will require, including features the op uses that may be blocked on prereq work, characteristics that shape the port's scope, downstream dependencies on donor migrations, and per-binding correctness hazards. Do not exit early on a RED prereq — surface all findings to the report.

**Check 1 (GATE): Op is on the `ProgramDescriptor` API.**

Confirm the op's program-factory code populates a `ProgramDescriptor` and uses `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`, etc. — *not* the older imperative-builder style from `host_api.hpp` (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` / etc.).

**Populating a `ProgramDescriptor` is necessary but not sufficient — it must be the ProgramDescriptor *factory concept*.** A clean PD-concept op exposes `create_descriptor()` returning a `ProgramDescriptor` and lets the framework handle cache-hit patching. An op that exposes the legacy `create()` + `override_runtime_arguments()` factory concept is **not** PD-concept even if it builds a `ProgramDescriptor` internally as a data structure and hand-rolls its own patching in the override hook. The tell for this hybrid is an `override_runtime_arguments()` that imperatively re-patches device state — especially `UpdateCircularBuffer*` / `UpdateDynamicCircularBufferAddress` calls (see [Per-execution CircularBuffer size updates](#per-execution-circularbuffer-size-updates-updatecircularbuffertotalsize-updatecircularbufferpagesize-updatedynamiccircularbufferaddressandtotalsize--unsupported-via-the-pd-concept-port-legacy-factory-concept)). Treat such a hybrid as RED — it needs migration to the true PD-concept factory first.

- **Green**: op uses the `ProgramDescriptor` **factory concept** — `create_descriptor()` returning a `ProgramDescriptor`, with framework-managed cache-hit patching.
- **Red (GATE)**: op uses the imperative `host_api.hpp` builder API (not `ProgramDescriptor`), **or** the legacy `create()` + `override_runtime_arguments()` factory concept (even if it populates a `ProgramDescriptor` internally as a data structure). Record the prereq gap — `ProgramDescriptor` migration is a **prerequisite to Metal 2.0 porting**, a substantial standalone body of work with TTNN-infrastructure implications, addressed in its own PR. **Do not attempt the migration as part of this audit, do not bundle it with anything, do not propose a partial conversion.** Continue with Check 2 and the remaining subjects — the feature-compatibility, TensorAccessor, call-surface, and other scans still produce useful findings (some features may need attention regardless of which API the op is currently on; others will only become relevant after the prereq lands).

**Check 2 (GATE): Device 2.0 Data Movement migration — every kernel the op uses.**

Confirm **every kernel this op exercises** is Device 2.0 compliant — **regardless of where the kernel file lives**. The op's own kernels, shared kernel-library code, in-family shared kernels, and borrowed/donor kernels from other families all count equally; location does not change the gate. What matters is whether the op's program factory instantiates or calls into the kernel. (An op may own *no* kernels of its own and file-path-instantiate all of them from a shared pool — those instantiated kernels are still fully subject to this gate; treat them as the op's effective kernels here.) See the [Device 2.0 Data Movement migration guide](../../../kernel_apis/data_movement/device_api_migration_guide.md) for what compliance entails. The *coupling* that borrowing induces is inventoried under [Out-of-directory coupling](#out-of-directory-coupling); the *gating* judgment lives here.

- **Green**: every kernel the op uses — wherever it lives — is Device 2.0 compliant.
- **Yellow — substantively compliant with isolated legacy holdovers** (structurally portable; the holdover is Device 2.0 cleanup, *not* port scope). Kernel uses `experimental::Noc`, `experimental::CircularBuffer`, etc. for the bulk of operations and has a small number of isolated legacy holdovers from the **CB-index-keyed free-function family**: free functions taking a `uint32_t` CB index where the corresponding Device-2.0 wrapper object is already in scope at the call site. A free function is a holdover **only if a Device-2.0 wrapper-method replacement exists for it** — e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`, `get_write_ptr(cb_id)` → `cb_obj.get_write_ptr()`. The shape (single CB-index argument, wrapper already in scope) is the cue, but it is **not** sufficient on its own: some CB-index free functions are *sanctioned* in Device 2.0 — its own migrated code uses them — and those are **not** holdovers. **If Device 2.0 allows the free function, so do we.** Currently sanctioned (do **not** flag): `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, both of which the [Device 2.0 migration guide](../../../kernel_apis/data_movement/device_api_migration_guide.md) keeps as free functions in its migrated examples. *Breadcrumb:* the Metal 2.0 `DataflowBuffer` now exposes a full tile/format metadata accessor set (in `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h`) — so a Metal 2.0 *port* moves these lookups onto the object ([kernel-side whitelist rule 7](port_op_to_metal2_recipe.md#kernel-side-whitelist)). That does **not** move the *Device 2.0* boundary here: the `CircularBuffer` wrapper's `get_tile_size()` just forwards to the free function, so `get_tile_size(cb_id)` stays sanctioned at this stage as long as Device 2.0 uses it — check the current Device 2.0 surface rather than assuming the shape alone makes it a holdover.

  Each holdover is a 1-line mechanical replacement (e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`). The op is otherwise structurally Device 2.0 — the wrapper objects are in scope, so the Metal 2.0 binding tokens attach — so the op is *feasible*, and the audit **still issues the porter brief** (no point forcing a re-audit after a trivial cleanup). **But the port cannot start until the holdover is fixed, and the fix is not part of the port:** a Device 2.0 change is out of port scope even when it's one line (the [kernel-side whitelist](port_op_to_metal2_recipe.md#kernel-side-whitelist) lets the port touch no Device 2.0 idioms — the port never scoops up stray holdovers). So: report each holdover with `file:line`, route it to the Device 2.0 effort to be cleaned **first** on that track, and have the brief flag it **prominently as a blocker the porter must clear before porting** (see [Output: the two documents](#output-the-two-documents)). The yellow tier applies when the holdovers are isolated within a kernel that otherwise consistently uses the wrappers; absolute count is a heuristic, not a rule.
- **Red (GATE)**: any kernel the op uses — own, shared-library, in-family shared, or cross-family donor — broadly uses legacy Device 1.0 idioms (raw `noc_async_read`, manual CB index management, `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, raw sem addresses, etc.). **The port is blocked** until that kernel's Device 2.0 migration lands; route the exact violations to the team that owns Device 2.0 migration, naming the kernel file (and, for a borrowed/donor kernel, its owning family) so the dependency is schedulable.

**Why Device 2.0 gates the port.** Device 2.0 cleanup is *not* on the [kernel-side whitelist](port_op_to_metal2_recipe.md#kernel-side-whitelist) of sanctioned port-time changes, and — more fundamentally — the Metal 2.0 binding tokens (`dfb::name`, `sem::name`, `ta::name`) attach to the Device 2.0 wrapper objects. A kernel still on Device 1.0 idioms has nothing for those tokens to bind to, so it cannot take the whitelisted Metal 2.0 swaps. Device 2.0 is therefore a hard structural prerequisite, on par with the `ProgramDescriptor` migration. (The isolated-holdover YELLOW above is the one carve-out, and only because the wrappers are *already in scope* there — the kernel is structurally Device 2.0 and the tokens attach.)

### Feature compatibility

Some legacy-API features are not yet supported in Metal 2.0. If the op uses one, it cannot be ported until support lands — those are **GATE** findings. Most legacy features, though, already have a Metal 2.0 home; a few of those translate via a non-obvious construct or carry a caveat, and those surface to the porter as heads-ups (**FYI-P**).

**Run this scan regardless of the [Prerequisites](#prerequisites) outcome.** Each Appendix A entry's recognition signals work against both ProgramDescriptor-form and imperative-`host_api.hpp`-form code — see the per-entry recognition bullets. Even when the ProgramDescriptor prereq RED's the op, the feature scan still surfaces which features it uses; that's the data point the human reader needs to plan downstream work.

For each entry in [Appendix A: Metal 2.0 feature compatibility](#appendix-a-metal-20-feature-compatibility), scan the op (host code, kernel code, factory functions, descriptors) using the recognition signals listed for that feature. Each entry declares its tier in the header — `UNSUPPORTED` (no Metal 2.0 support yet — or, for the per-execution CB-size entry, a capability the host API has but the PD-concept port path cannot reach, which gates for the same practical effect) or `LANDED` (supported today; the Status field names the replacement construct).

- **Green**: no entry's recognition signals fire, or only `LANDED` entries fire whose translation is routine.
- **Red (GATE)**: an `UNSUPPORTED` entry's signals match definitively. Report the feature name, the `file:line` where it appears, and the recognition signal that fired. The port is blocked on this finding; continue scanning the remaining Appendix A entries and the rest of the audit — complete the full audit even after a RED match.
- **Yellow**: an `UNSUPPORTED` entry's signals match *ambiguously* (you cannot be sure whether the feature is in use). Ask the user; on confirmation it becomes a GATE, otherwise green.

**Notable `LANDED` features (FYI-P heads-up).** A handful of supported features are worth flagging to the porter even though they don't gate — they translate via a non-obvious construct or carry a caveat. When one fires, surface it as a heads-up with its `file:line` and the construct the port will use:

- **Aliased CBs** → `DataflowBufferSpec::advanced_options.alias_with` (a ninja feature; see the entry for the legality constraints).
- **Borrowed-memory DFB** (dynamic CB on borrowed `Buffer` memory) → `DataflowBufferSpec::borrowed_from`.
- **Dynamic `TensorAccessor`** (`ArgConfig::Runtime*`) → the **UNSAFE** `TensorParameterAdvancedOptions` relaxation opt-ins; adopting them has per-dispatch-caching implications, so flag for the user's awareness.
- **Non-zero semaphore initial value** → `SemaphoreSpec::advanced_options.initial_value`, which is `[[deprecated]]` and unsupported on Gen2. Deprecated-but-fine on Gen1; note it so nobody mistakes it for a blocker.

If the op uses something *not listed* in Appendix A and you are uncertain of its support status, treat as yellow and ask. Do not assume support from API surface.

### TensorAccessor handling

Every tensor a kernel reads must reach Metal 2.0 through the typed binding channel (`TensorParameter` / `TensorBinding`). This subject inventories, **per binding**, how the legacy op accesses each tensor and classifies the port work into one of two cases. Both cases are **PORT WORK** — the porter acts on them during the port; neither gates. (This subject merges what earlier versions of the audit split across a "TensorAccessor usage" check and a separate "TensorAccessor bypass" check: they detect the same population — tensors not cleanly on `TensorAccessor` — so they are one subject now.)

**Why this matters.** The legacy hazard is a tensor base address that reaches the kernel through an RTA/CRTA. Under TTNN's fast-path-cache binding-injection model, the framework patches the typed-binding channel on cache hit but leaves RTAs untouched — so a buffer address routed through an RTA stays at whatever value the cache-populating call wrote, and on later cache hits with new tensor storage of the same shape the kernel reads the *original* buffer. No assertion fires; just wrong numerics, only on cache hits with non-identical storage. Pre–Metal 2.0 this was a style concern (*"yuck, raw pointers"*); the fast-path change made it silently wrong. The port replaces that RTA-smuggled address with a typed `TensorParameter` binding (which *does* refresh) — regardless of what the kernel then does with the base pointer (the two cases below), and including kernels that use no `TensorAccessor` at all (older addr-gen idioms, hand-rolled NoC walks).

This resolves **at port time** — it waits for no framework feature, with one exception: a raw-pointer (Case 2) binding inside a **compute** kernel, which is blocked pending the compute-kernel `TensorBinding` fix (see the compute-kernel callout under *The two cases*).

**Scope.** Audit only kernels that actually touch tensor memory. **Compute kernels that only consume from / produce to circular buffers are out of scope** — they read CB pointers, not tensor memory; the tensor read happens upstream in a dataflow kernel.

**Causal-link gate (run this first, per binding).** Before classifying any binding, check whether the kernel's access is a **borrowed-memory DFB read**: it reads tensor data through `cb_*.wait_front` / `cb_*.get_read_ptr` from a CB that is itself a borrowed-memory CB (see the [Dynamic CircularBuffer entry](#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed) under Feature compatibility). There the lack of `TensorAccessor` is *intended* — the borrowed-memory DFB **is** the tensor access — and the port handles it via `DataflowBufferSpec::borrowed_from`. Mark such a binding **clean**; do not force it into Case 1 or Case 2. Mis-classifying it as "convert to TensorAccessor" would be a regression.

**But "clean" requires the CB to be *synchronized*.** A borrowed-memory CB is a genuine DFB only if the kernels actually drive its FIFO machinery — something `push_back`s into it and something `wait_front`s on it (a sharded reader presenting already-resident data via `push_back`, satisfying a waiting compute consumer, is the canonical legit case). **Litmus: does any kernel drive the sync machinery on this CB — a FIFO producer *and* a FIFO consumer — or is it pointer-only?** (The same core may be both endpoints.) If nothing produces into the CB and it is merely read by base pointer — no FIFO anyone waits on — it is a **sync-free CB**, *not* a clean borrowed-memory DFB. A sync-free CB cannot be expressed as a Metal 2.0 DFB (the spec validator requires ≥1 producer and ≥1 consumer), so the port resolves it with the sanctioned **interim workaround** (see [the porting recipe](metal2_port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround)). This does **not** gate — the workaround keeps the port unblocked — but **report it as a heads-up (FYI-P)** at the **(CB, endpoint)** edge (the same CB can be a synchronized LLK operand on one binding and pointer-only on another). A resident input read by raw pointer, with no producer — e.g. a reciprocal lookup table — is the trap this catches. If a kernel involves sharded code paths or reads from a CB rather than via `TensorAccessor`, scan the Dynamic CircularBuffer rule for the same code path before finalizing. **The synchronized/sync-free verdict can itself flip per config:** the same CB can be a sync-free scratchpad under one sharding and a real FIFO under another, so classify it per instantiation, never once for the op (conv2d `ACT_TILIZED` is the canonical confuser — height-sharded → scratchpad, block/width-sharded → FIFO; see [the catalog's per-config note](metal2_port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround)). This is only the *floor* (≥1 of each); a CB that passes it can still fail the *ceiling* — see [DFB endpoint legality](#dfb-endpoint-legality-spsc) for the SPSC (≤1 per node) and dead-CB (zero-endpoint) checks, including the **hidden second writer** — invisible to a FIFO-sync trace, so it slips this floor as "clean," yet a real SPSC violation best caught here pre-port (otherwise a cryptic late port-validation rejection).

**The two cases.** For each `TensorParameter` the op declares (or would declare in the port), classify by **what the kernel does with the tensor's base pointer**. In *both* cases the legacy host smuggles `buffer()->address()` in through an RTA/CRTA — the distinction is purely what the kernel does with that raw pointer on the device side, not whether one exists. A mechanical observation, not a judgment call:

- **clean** — a borrowed-memory DFB read (the causal-link gate above). The DFB *is* the tensor access; the port handles it via `DataflowBufferSpec::borrowed_from`. Neither Case 1 nor Case 2 — no work item here.
- **Case 1 — via `TensorAccessor`** (the common case). The kernel feeds that base address into a `TensorAccessor` constructor (`TensorAccessor(args, addr)`) and does all its memory access *through* the accessor (`accessor.get_noc_addr(page)` and friends). **Port work:** express the binding as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(ta::name)` instead, and the legacy address-via-RTA plus its `TensorAccessorArgs` plumbing both disappear. Mechanical, low-risk.
- **Case 2 — raw pointer.** The kernel uses that base address *directly* — explicit address arithmetic and hand-rolled NoC calls, never constructing a `TensorAccessor` from it. **Port work:** express the binding as a `TensorParameter` / `TensorBinding` too (the address never stays on an RTA), but the kernel pulls the base via the sanctioned `TensorAccessor::get_bank_base_address` bridge and **keeps its existing raw arithmetic unchanged**. We do **not** rewrite raw access into `TensorAccessor` iteration — that conversion is deliberately out of scope (too high-risk for a port). A buffer-address RTA is *not* an acceptable substitute for the bridge.

  > ⚠ **Case 2 in a *compute* kernel is currently blocked.** `get_bank_base_address` lives on `TensorAccessor`, which a compute (TRISC) kernel cannot bind today — so a compute kernel needing a raw base address has no bridge. Such a binding **gates the port**: report it and stop, routing it to the compute-kernel `TensorBinding` fix. Do *not* fall back to smuggling the address through a CRTA/RTA — even from an op-owned tensor, where it would be strictly correct, that path is closed (see [recipe rule 5](port_op_to_metal2_recipe.md#kernel-side-whitelist)).

**Detection — host side.** Any site where `buffer->address()` (or `->address()` / `(*buffer).address()` / `tensor.buffer()->address()`) flows into a runtime-args context. Common shapes:

- **Descriptor form** (the in-scope case for `ProgramDescriptor`-API ops): `KernelDescriptor::runtime_args` or `runtime_common_args` initializers containing the address expression directly. E.g. `kd.runtime_args = {{core_coord, {input_buffer->address(), num_pages}}};`.
- **Imperative form** (only in ProgramDescriptor-prereq RED ops, but record matches since the eventual port still needs them): `SetRuntimeArgs` / `SetCommonRuntimeArgs` argument lists containing the address expression.
- **Helper-function form**: a function takes a `Buffer*` / `Buffer&` and injects its address into an arg vector (`args.push_back(...)`, in-place vector init, named accumulator). Read the helper body — it often hides the bypass.
- **`Buffer*`-binding form** (descriptor API): the factory pushes a `Buffer*` (the pointer object itself, *not* `->address()`) into `KernelDescriptor::RTArgList` / `emplace_runtime_args`. The framework auto-registers these as `BufferBinding`s and **patches them on cache hits**, so this shape is *correct-on-cache-hit today* — it is **not** the silent-wrong hazard. (It's the framework's interim hack for plugging the stale-pointer hole in `ProgramDescriptor` ports; the Metal 2.0 typed binding supersedes it.) **Still enumerate it** — the kernel consumes a raw `uint32_t` base, so it is **Case 2** (raw pointer → bind as `TensorParameter`, bridge via `get_bank_base_address`) — but record it as routine port work, not a correctness hazard (the framework patches it on cache hits today). Enumerating *all* pointer arguments is the point; just don't over-state the urgency of this one.
- **CTA-baked-address form**: the factory bakes `buffer->address()` into a **compile-time** arg (not an RTA) that a kernel consumes as a `TensorAccessor` base address (often via an NTTP). Like the `Buffer*` form, this is **not** the silent-wrong hazard — a compile-time arg forces a recompile per address, so a stale base can't survive a cache hit — but it **is** a pointer argument to enumerate: classify it **Case 1** (via `TensorAccessor` — express it as a `TensorParameter`, and the CTA-baked address + kernel-side NTTP base both disappear). Detect it by a `buffer->address()` / `.address()` expression flowing into the factory's *compile-time*-args list rather than its runtime-args list.

For each `TensorParameter`, cross-check: does the same buffer also appear in an RTA address argument? If yes, that buffer is consumed as a raw pointer → **Case 2** (bind as `TensorParameter`, bridge via `get_bank_base_address`).

**Detection — kernel side.**

- `get_noc_addr_from_bank_id<bank_type>(bank_id, addr)` where `addr` traces back to a `get_arg_val`. The textbook anti-pattern.
- `get_noc_addr(x, y, addr)` where `addr` traces back to RTAs.
- `noc.async_read(addr, ...)` / `noc_async_read(addr, ...)` where `addr` was computed from an RTA-sourced base + offset arithmetic.
- Variable names that telegraph buffer-address purpose — `src_addr`, `dst_addr`, `remote_addr`, `base_addr`, `input_addr` — when their value comes from `get_arg_val`.

**False-positive guards — do NOT flag.**

- `get_arg_val<uint32_t>` for shape / count / index / control values. Those uint32s aren't addresses.
- `accessor.get_noc_addr(page_id)` outputs — that's the `TensorAccessor` path (Case 1), not a raw-pointer Case 2; don't flag it.
- Host-side `set_globally_allocated_address(buffer)` — the borrowed-memory pattern, covered by the Dynamic CircularBuffer feature entry (clean via the causal-link gate).
- Kernel reading from a borrowed-memory CB via `cb.get_read_ptr()` — clean; the causal-link gate applies.

**Granularity — per binding, not per op.** An op may have multiple tensor bindings, some clean and some needing work. Report per binding — a single Case-1 or Case-2 binding fires this subject even when the op's primary I/O is via `TensorAccessor`. **Classification can also vary per factory within one bundled op** — the *same* `TensorParameter` may be clean (borrowed-memory DFB) in one factory and Case 1 in another (e.g. a sharded vs. an interleaved factory). When that happens, record the per-factory split via the report's Per-DeviceOperation attribution rather than forcing one flat verdict for the binding.

**Op-level roll-up:** `✓ clean` (every binding clean) / `⚠ port work` (one or more Case-1 / Case-2 bindings), with the per-binding inventory in the report.

### DFB endpoint legality (SPSC)

**Precondition — Device 2.0 clean.** This subject's recognition signals assume **Device-2.0 kernel idioms** (`get_write_ptr` methods, `get_local_cb_interface`, `Semaphore` objects). Run it only when the [Device 2.0 gate](#prerequisites) is clean (GREEN, or YELLOW on isolated holdovers). On a **Device-2.0-RED** op, *defer* it — mark it `(deferred — re-evaluate after Device 2.0 migration)` — because the kernel rewrite changes the idioms the scan keys on, and a best-effort pass over legacy idioms can **false-negative the hidden writer** (worse than deferring: it would report "clean"). (A ProgramDescriptor-RED op already defers it via the [RED short-circuit](#red-short-circuit-the-programdescriptor-prerequisite-fires).)

**This subject checks the *count* of producers and consumers — the ceiling the [floor check in TensorAccessor handling](#tensoraccessor-handling) never reaches.** That check asks whether a CB has *at least* one producer and one consumer (the floor): too few → it's sync-free or single-ended → the port bridges it (compute self-loop / DM fabricated consumer; one case GATEs — see the single-ended note). This subject asks the complement — *at most* one of each **per node** (the ceiling) — and flags the two ways a CB lands outside the legal `(1 producer, 1 consumer)` window: **zero endpoints** (a dead CB) and **excess endpoints on a node** (an SPSC violation).

**Why the ceiling matters — and why to catch it pre-port.** Metal 2.0 enforces **single-producer / single-consumer per node** (SPSC) as a spec-validator legality rule — fundamental on Gen2, whose per-node dataflow hardware assumes it. The host **cannot waive SPSC selectively**: at spec-construction it cannot tell a sync-free multi-endpoint CB from a synchronized one, so relaxing the check for one would unsafely relax it for all. Two consequences: an SPSC violation is **not port-fixable** — the op owner must resolve it *before* the port, as a functional change (out of port scope). And although the spec validator **will** reject it at port time — once the offending access is bound, the second endpoint surfaces and the SPSC check fires — that is a *cryptic, late* failure: a 2-producer (or 2-consumer) rejection whose root cause isn't obvious, on a violation the porter can't fix anyway. So catch it **pre-port**: name the cause and route the op-owner rewrite before any conversion effort is sunk. One face — the hidden second writer — is moreover invisible to a FIFO-sync trace (its raw co-fill uses no FIFO ops), so you must actively hunt for it; miss it and the auditor hands the porter a clean bill that detonates mid-port. Run this subject for every op, even when the floor check passed.

**The endpoint census.** SPSC binds a **CB *instance*** — the device-side per-node materialization — not the host `CBDescriptor` / `CreateCircularBuffer` (one descriptor over a core range makes **one instance per node**). So count **per CB, per node**: on each node's instance, tally the endpoints. An endpoint is any kernel that touches the CB — FIFO-produces (`reserve_back`/`push_back`), FIFO-consumes (`wait_front`/`pop_front`), **or** accesses the memory by **raw pointer** (`get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(<cb>).fifo_*_ptr`). *Any* access counts: in Metal 2.0 a kernel structurally cannot touch a DFB it hasn't bound — there is no back-door base-pointer grab — so every access is a binding, hence an endpoint (the hidden raw writers below included).

> **Count and sync are orthogonal — don't fuse them.** The census counts endpoints by *access* (FIFO or raw pointer — all count); the [synchronized/sync-free axis](#tensoraccessor-handling) separately judges whether those accesses drive the FIFO machinery (**synchronized**) or just walk the memory (**sync-free**). A lone pointer reader is **one** endpoint *and* sync-free (→ interim bridge); two pointer readers on a node are **two** endpoints (→ SPSC violation) whatever their sync. Count first, judge sync second.

Classify by the pair:

| (producers, consumers) on a node | Verdict | Handled |
|---|---|---|
| (0, 0) | **Dead CB** — allocated, never referenced | here (drop pre-port) |
| (1, 0) / (0, 1) | **Single-ended CB** — one endpoint | [interim bridge](metal2_port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround) (port fabricates the missing side) — **except a DM single-ended *producer*, which GATEs** (note below) |
| (1, 1) | **Legal** — one producer + one consumer | — |
| (≥2, ·) / (·, ≥2) | **SPSC violation** — excess on a node | here (op-owner pre-port fix) |

> **Single-ended forks by kernel + sync.** A single-ended CB is bridged at port time — compute → self-loop (INTRA); DM sync-free → fabricated consumer — **except a single-ended *producer* on a DM kernel** (real `reserve_back` / `push_back`, no consumer; a DM "packer"): a DM self-loop has no backend lowering and a fabricated consumer would risk deadlocking a real producer, so it has **no port-time bridge → GATE.** Route the op-owner rewrite (a DM kernel can write its output tensor directly, so the FIFO is gratuitous). See the [recipe's fork](metal2_port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround).

#### Dead CB (0, 0) — FYI-P, pre-port drop

**Recognition.** A CB whose `buffer_index` is referenced by **no** kernel — no `reserve_back`/`push_back`, no `wait_front`/`pop_front`, no `get_read_ptr`/`get_write_ptr`, no raw access. Grep the bound kernels for the index and any named CTA carrying it; zero hits.

**Why it matters.** Left in place, the port mechanically builds a `DataflowBufferSpec` for it that the validator then rejects — a DFB needs ≥1 producer + ≥1 consumer, and a zero-endpoint DFB has neither. Pre-flagging spares the porter a dead end.

**Action.** **FYI-P.** Flag for the op owner to drop the allocation (and any dead CTA carrying its index) as a **pre-port cleanup** — a functional change, out of port scope. Surface it to the porter as a heads-up so an unreferenced CB index doesn't read as a missed binding. Record `file:line`.

**Examples in the wild:** conv2d `L1_ARRAY` — a 1-page "L1 scratchpad CB" whose index is threaded to the reader as a (dead) CTA, yet no kernel ever accesses it.

#### SPSC violation (≥2 of one kind on a node) — GATE (config-scoped), route to the op owner

**Why it can't be bridged.** The interim bridge fabricates the *missing* endpoint for a CB with a single real toucher (a compute self-loop, or a DM fabricated consumer); it cannot absorb a *second* kernel that also genuinely touches the CB on the same node — that is the very two-endpoint shape SPSC rejects. So an SPSC violation has **no port-time workaround**; it is an op-owner pre-port functional change. Two faces:

**(a) Hidden second writer — the hidden face.** A CB presents as single-producer to a FIFO trace (one kernel does `reserve_back`/`push_back`), but a *second* kernel co-fills it via a **raw write** — `<cb>.get_write_ptr()` / `get_local_cb_interface(<cb>).fifo_wr_ptr` + offset, **with no** `reserve_back`/`push_back` — coordinated by **dedicated semaphores** (e.g. `reserve_done` / `write_done`) rather than CB FIFO sync. It is invisible to the floor check (the CB has a FIFO producer + consumer → "clean") and to an **auditor's FIFO-sync trace** (the raw co-fill uses no FIFO ops). The port-time validator *does* catch it — the `dfb::` handle the co-fill writes through doesn't exist unless the host binds it, and binding it makes the CB a two-PRODUCER node → SPSC fires — but only as a cryptic late rejection, on a violation the porter can't fix. So **hunt for it pre-port:** for each CB, scan *every* kernel that touches it for a `get_write_ptr()` / `fifo_wr_ptr` write by a kernel that is **not** the CB's FIFO producer, gated by a semaphore wait/post pair. It is a genuine two-producers-on-a-node, hidden behind side-channel sync.

> *Resolution (op owner, pre-port):* typically a Gen2-conditional that has **one** DM fill the whole CB instead of the split co-fill — output-preserving, with the split-reader L1 saving demoted to a deferred optimization.
>
> *Example:* conv2d `ACT` under `split_reader_cb_shared` — the writer co-fills `cb_act_second_obj.get_write_ptr()+offset` gated by `reserve_done`/`write_done` (CTAs 32/33) while the reader is the FIFO producer.

**(b) Multiple readers — the visible face.** A borrowed-memory, **sync-free** tensor-view CB (read by base pointer) whose read sites span **2+ co-resident kernels**: a split-reader's two DM readers, or a reader plus a writer acting as a second reader. A single-reader version routes to the interim bridge; with 2+ readers the bridge cannot express it (it would put two consumer endpoints on the node → SPSC rejection).

> *Resolution (op owner, pre-port):* convert the borrowed CB to **per-reader `TensorAccessor`s** — each reader reads the resident tensor through its own accessor (no DFB, no SPSC endpoint). A functional change.
>
> *Recognition:* a borrowed-memory, sync-free tensor-view CB whose base-pointer read sites span 2+ co-resident kernels. **A FIFO producer that also raw-reads its own buffer counts as one of those consumers** — a kernel that `push_back`s then `get_read_ptr`s, alongside a second reader, is a (1 producer, 2 consumers) violation, not a clean (1,1).
>
> *Examples:* pool `raw_in` / `in_reader_indices` / `config_cb` (split-reader, two DM readers); conv2d `ACT_SHARDED` / `READER_INDICES` in split-reader / mcast configs (reader + writer-as-2nd-reader); halo `src_cb` ROW_MAJOR (1P + 2C).

**Config-dependence (both faces).** The *same* CB is typically single-endpoint under one config (→ legal, or a port-handled sync-free/single-ended CB) and multi-endpoint under another (split-reader / mcast → SPSC violation). Classify **per instantiation**, and apply [Code-path scope](#output-the-two-documents): RED the violating config-path and name the clean configs as a portable subset (`RED at op level; subset <X> is clear`). The clean subset still gets a brief. **But when the multi-endpoint shape is *unconditional* — structural, not one branch among portable siblings (e.g. a split reader that is always on) — say so explicitly: there is no portable subset, the op is RED whole, and the op-owner functional fix is the only path.** Don't leave "no subset" to be inferred from silence.

**Finding role.** **GATE** at config-path granularity — the port is blocked on the violating path until the op owner's pre-port fix lands; route the detail (and the recommended fix) to the **op owner**, and offer the clean subset. Unlike the prereq / feature GATEs, the resolution is an op-owner *functional change*, never framework work and never folded into the port.

**Op-level roll-up:** `✓ legal` (every CB in the legal window or a port-handled sync-free/single-ended CB) / `⚠ dead CB(s)` / `⛔ SPSC violation(s) — pre-port op-owner fix`, with the per-`(CB, config)` inventory in the report.

### Out-of-directory coupling

Run this subject regardless of the other subjects' outcomes. **The findings here are informational (FYI).** The one place donor coupling *gates* the port — a donor kernel still on pre-Device-2.0 idioms — is judged in [Prerequisites § Check 2](#prerequisites); this subject inventories the coupling in full so that gate is well-scoped and any future multi-op coordination is visible. This is the most substantial subject in the audit; budget time accordingly. It covers **two distinct escape types**: *function-call escape* (the kernel `#include`s and calls another op's helper function) and *file-path kernel instantiation* (the program factory `CreateKernel`s a kernel `.cpp` owned by another op). The bulk of the machinery addresses function-call escape; file-path escape is surfaced separately at the end as a coupling signal.

**Why this check exists.** Op kernels frequently `#include` headers outside their own directory. When a Metal 2.0 port crosses one of these boundaries, the kernel's named tokens (`dfb::name`, `sem::name`, `ta::name`) need to translate into whatever shape the donor's signature expects. Some shapes cross cleanly; others require donor-side conversion work, most commonly because **the donor itself isn't on Device 2.0 yet**.

The Device 2.0 → Metal 2.0 sequencing rule applies: ops must complete Device 2.0 migration before Metal 2.0 can proceed. A donor consumed by this op that is still on pre-Device-2.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, raw sem addresses, `CircularBuffer&`) blocks this op's Metal 2.0 port until the donor migrates — that **GATE judgment is recorded in [Prerequisites § Check 2](#prerequisites)**; here, inventory the donor shape so the gate is specific and schedulable.

**Inventory phase.** For each kernel file in the op, list every `#include` whose resolved path lies outside the op's own directory. Identify the donor class:

1. **`tt_metal/*`** — LLK / HAL / firmware. No concern.
2. **`ttnn/cpp/ttnn/kernel_lib/`** — official shared kernel library; lib team handles internally.
3. **`ttnn/cpp/ttnn/kernel/`** (singular) — a second shared-kernel pool. Treat as shared-lib class.
4. **`ttnn/cpp/ttnn/operations/kernel_helper_functions/`** — small shared utility pool.
5. **In-family shared** — kernels within the same op family. In-family escapes don't *gate* the Metal 2.0 port; you port the family together. (This concerns the Metal 2.0 *syntax* rewrite, not Device 2.0 — in-family kernels remain fully subject to the [Device 2.0 gate](#prerequisites), which is location-independent.)
6. **Cross-family donor** — kernels in another op family's directory.

**Per-call shape analysis.** For each donor file consumed by the op, identify which public functions the op's kernels actually call, and classify each by the shape of the resource handles in its signature.

| Shape | Status | Notes |
|---|---|---|
| `Semaphore` / `Semaphore&` / `const Semaphore&` | ✓ excellent | Device 2.0 native. |
| `uint32_t sem_id` | ⚠ suboptimal | `sem::name`'s constexpr cast to `uint32_t (sem_id)` handles once landed. Workable today. |
| `uint32_t sem_addr` (L1) or `uint64_t` NOC-encoded sem | ✗ not OK | No clean Metal-2.0 → donor bridge today. A backdoor could be added. |
| `TensorAccessor<DSpec>` / ref (Shape 1) | ✓ excellent | Porter constructs `TensorAccessor(ta::name)` and passes. |
| `TensorAccessorArgs<N>` (Shape 2) | ✗ not OK | Porter can pass `ta::name.args`. Workable, but suboptimal. |
| Tensor CTA offset as NTTP (Shape 3) | ✗ not OK | No workaround today; would require a one-line `ta::name::cta_offset` add to `TensorAccessorBindingToken`. |
| Old-style addr-gen — `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*` (Shape 4) | ⭐ ✗ very not OK | Donor is pre-Device-2.0 — the donor-side Device 2.0 **GATE** (recorded in [Prerequisites § Check 2](#prerequisites); the op's *own* kernels are gated there too). Inventory the donor file + kernel here so the gate is specific. |
| `uint32_t cb_id` | ✓ OK | `dfb::name`'s constexpr cast handles runtime AND template-parameter position. |
| `CircularBuffer` / `CircularBuffer&` / `const CircularBuffer&` | ⭐ ⚠ flag | Op-by-op porting + DFB-replaces-CB on the consumer side leaves no clean per-op story today. Flag for cross-team discussion. |

One donor file can have multiple functions with different shapes — classify per function.

**Report format.** Three-part structure:

1. **Op-level roll-up** — one-line headline status (`✓ clean` / `⚠ workable` / `⭐ blocked`), plus a tight bullet inventory of the kinds of issues found across all donors.
2. **Summary table** — one row per (op kernel, donor file) pair across all buckets.
3. **Per-call detail** — per-function breakdown for donors with ⚠ / ✗ / ⭐ entries. Omitted entirely if all rolls are ✓.

Status roll-up uses ✓ / ⚠ / ✗ / ⭐. The star is reserved for entries that create scheduling blockers — Shape 4 (donor pre-Device-2.0, the donor-side Device 2.0 gate per [Prerequisites § Check 2](#prerequisites)) and `CircularBuffer&` (op-by-op friction). Other ✗/⚠ items are workable today or need donor work, but don't sequence-block.

**Borrowed kernel files (file-path kernel instantiation).** Separate from the function-call escapes inventoried above: list every kernel `.cpp` file the op's program factory instantiates whose source it does **not** own — anything from a shared pool, in-family or cross-family. (Some ops own *none* of their kernels and instantiate every one from a shared pool — list them all.) For each, record:

- The kernel file's path.
- The owning op family (or shared pool).
- Whether the file is also instantiated by other ops (broadly-shared) or is a one-off borrow — and, where cheap to determine, *which* other ops.

This signal **does not gate the port**, but it induces a **port-the-family-together coupling that must be reported**. A shared kernel's Metal 2.0 rewrite (CB→DFB, named-token bindings, etc.) is a *single* rewrite: every op that instantiates that kernel must adopt it in the same change, or the co-borrowers break the instant one op migrates in isolation. So the set of ops sharing a kernel forms a Metal 2.0 **port-together set** — report that set (or as much of it as is cheap to find) so planners can sequence the shared rewrite as one unit. Surface this even when the function-call escape roll-up is `✓ clean` — file-path coupling is independent. (This is distinct from the [Device 2.0 gate](#prerequisites), which applies to every one of these kernels regardless of coupling.)

### Custom program hash

**Recognition.** The op's device-operation type defines a `compute_program_hash` member or override, replacing the default reflection-based hash. Record the location (`file:line`).

**Finding role: PORT WORK — delete it.** If the op has a custom `compute_program_hash`, the port **deletes it and reverts to the default TTNN hash.** This is a *sanctioned exception* to the rule that the device-operation class is off-limits during the port — parallel to the [pybind-deletion exception](port_op_to_metal2_recipe.md#host-side-stay-in-the-lane). The justification is structural:

- **No Metal 2.0 factory concept reads the custom hash.** It has no role in the ported caching path.
- **PD-ported custom hashes are frequently incorrect now.** Many were written against an earlier framework and silently omit `TensorSpec` from the key — which trips `UpdateTensorArgs` legality failures on fast-path cache hits (a cache hit on a `ProgramSpec` built for different inputs).
- **The default is correct-by-construction.** The default hash keys on the op args that determine the program — its type, attributes, and tensor args (including each tensor's `TensorSpec`) — so a cache hit only happens for a genuinely equivalent invocation.

So this is neither "patch the custom hash" nor "wait and see if it bites at verification" — it is a deletion the porter performs as part of the port, recorded prominently in the port report (a device-op-class change). Do **not** try to repair a custom hash to include `TensorSpec`; that path leads to subtle bugs.

**Before deleting — mine it for relaxation candidates (FYI-U, team-only, FALLIBLE).** A custom hash sometimes encodes which tensor properties the op *actually* depends on — a clue to where a `TensorParameter` relaxation (`dynamic_tensor_shape` / `match_padded_shape_only`) might be safe. Before the hash disappears, read it and record any such candidates for the team's relaxation roadmap. **This is fallible**: many custom hashes are themselves wrong, so a candidate mined from one is a *candidate to verify*, not a conclusion. The default remains strict (per the [TTNN integration doc](port_op_to_metal2_ttnn_factory.md)); a relaxation is applied only on explicit user OK. These candidates never reach the porter brief — they feed the team record only.

### Other signals

A residual subject for signals that don't belong to any of the other subjects. Today it holds one check.

**RTA varargs (FYI-P).** Recognition: a kernel reads `num_runtime_varargs > 0` from its `KernelDescriptor`, OR pulls arguments in a counted loop (`for (int i = 0; i < N; ++i) { get_arg_val<uint32_t>(i); ... }`) where `N` is itself runtime-known. Typical in ops with a runtime-variable input count.

Metal 2.0 **supports** RTA varargs via the kernel-side vararg mechanism, so this does not gate — it is a porter heads-up. Report the kernel and the recognition site (`file:line`), and note that the port will choose between named RTAs (the recommended endpoint — one named RTA per legacy positional argument) and Metal 2.0's RTA vararg mechanism (only when the kernel genuinely loop-retrieves with a runtime-varying index, per the recipe's [kernel-side whitelist rule 4](port_op_to_metal2_recipe.md#kernel-side-whitelist)).

**Not to be confused with CTA varargs**, which **do** gate the port (caught by the [CTA varargs Appendix A entry](#variable-count-compile-time-arguments-cta-varargs--unsupported)): kernels that loop over `get_compile_time_arg_val(i)` with a runtime-varying `i`, or ops whose `tensor_args_t` carries a variable-count container like `std::vector<Tensor>`.

**Incidental code anomalies (FYI-U).** While reading the op you will sometimes notice latent issues that are neither audit findings nor the porter's to fix — a dead/unused RTA, an attribute the factory forces or ignores yet still feeds to `compute_program_hash`, a suspicious hardcoded constant. Record these in the report's **Misc anomalies** section: team-only, non-gating, *not* porter-actionable (they route to the op owner, never into the port diff). Don't go hunting for them — just note what you happen to see while auditing.

### TTNN factory concept analysis

**This subject is analysis, not a decision.** Ops port to a single Metal 2.0 factory concept (`MetalV2FactoryConcept`); this subject does **not** re-decide that and does **not** itself gate. Your job is to answer six questions about the op's TTNN-side shape and record each with `file:line` evidence; the downstream port (see [`port_op_to_metal2_ttnn_factory.md`](port_op_to_metal2_ttnn_factory.md)) consumes them to confirm the op fits that concept and to surface any residual blocker (genuine multi-program, op-owned `GlobalSemaphore`s). Run this subject **last**, after the other subjects have produced their findings — op-owned-tensor recognition draws on the [TensorAccessor-handling](#tensoraccessor-handling) per-binding inventory.

**Cross-reference — Diego's readiness sheet.** Several of these questions map straight onto columns in Diego's [readiness sheet](../analyses/ttnn_op_porting_readiness.md): Q3/Q4 ↔ *Pybind descriptor*, Q5 ↔ *Custom hash*, Q6 ↔ *Runtime-args update*, plus *Concept*. If you fetched it at the start, cross-check your answers against it — but still record your own `file:line` evidence for each; the sheet is a prior, not a substitute for the check.

**Scope reminder.** Throughout, *"does this op …"* means *"do **any** of this op's ProgramFactories …"* — inspect every ProgramFactory the op defines; a yes on any one is a yes for the op, and you attribute the finding to the factory (and DeviceOperation, when bundled) where it appears.

Answer each question Yes/No with the evidence that decided it:

1. **Op-owned tensors?** Does any ProgramFactory allocate and manage device tensors of its own — intermediate / scratch tensors beyond the op's declared input and output tensors? *Recognition:* a factory (or the device-operation's tensor plumbing) that creates a device tensor it owns — e.g. `create_device_tensor` / `allocate_tensor_on_device` in the factory or device-op `.cpp`, or an intermediate `Tensor` constructed and threaded into the program that is not in `tensor_args` / `tensor_return_value`. Record each owned-tensor site.

2. **MeshWorkload concept needed?** Does any ProgramFactory *genuinely* require multi-program / MeshWorkload structure (true cross-program or cross-device coordination)? *Recognition:* the factory provides `create_mesh_workload` / `create_workload_descriptor`, or the device-operation carries `cached_mesh_workload_t`. **False-positive trap — read this before answering Yes.** A *legacy* op may sit on the MeshWorkload path only because the old framework couldn't carry op-owned tensors on the single-program path — a plumbing artifact, **not** a genuine MeshWorkload need. `MetalV2FactoryConcept` now carries op-owned tensors natively (`op_owned_tensors`), so such an op is morally single-program and **ports cleanly** — it is not blocked. Distinguish "needs MeshWorkload because the op is genuinely multi-program / cross-device-coordinated" (→ **Yes**) from "appears on the MeshWorkload path only because it has op-owned tensors" (→ **No**, and say so explicitly, naming Q1 as the cause). When you can't tell, mark it a question for the user rather than guessing Yes.

3. **Pybind `create_descriptor`?** Does the op pybind the *innards of a ProgramFactory*? **Carve-out first:** every op pybinds the op *itself* — the user-facing function (`bind_function<"op_name">` / `mod.def("op_name", …)`), its program-config classes, its enums. **That normal surface is expected and is *not* a finding.** What this question targets is the surprise: a binding of the **ProgramFactory class** that reaches into `create_descriptor`. *Recognition:* an `nb::class_<…ProgramFactory>(…).def_static("create_descriptor", …)` (or `.def`) in the op's `*_nanobind.cpp`. The port deletes this (a sanctioned device-op-class edit — see `port_op_to_metal2_ttnn_factory.md`); record the binding site. *(Canonical example: layernorm's `bind_normalization_layernorm_program_factory` binds `create_descriptor` on both factory classes — note the extra `core_range_set` parameter that exists **only** to drive the pybind hook, the tell that this is factory-innards, not the normal op surface.)*

4. **Other migration-risky pybind?** Does the op pybind anything *else* from the **ProgramFactory or DeviceOperation internals** that would interfere with the Metal 2.0 migration? *(Deliberately broad — and the Q3 carve-out applies again: the normal op-function / program-config / enum bindings don't count.)* The discriminator is *what the `nb::class_<>` wraps*: a `DeviceOperation` or factory/param class is the surprise. Look for the device-op's methods exposed to Python (`compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory`), pybound attribute / input structs, a factory parameter that exists only to drive a pybind hook, or any introspection entry point returning `ProgramDescriptor`. When something looks like it could complicate the port, surface it with its site rather than deciding it's harmless. *(Canonical example: layernorm's `bind_normalization_layernorm_device_operation` and `..._params_and_inputs`.)*

5. **Custom hash?** Does the op declare a custom `compute_program_hash`? Yes/No + `file:line`. *This is a presence check only* — the treatment (the port deletes it, reverting to the default) lives in the [Custom program hash](#custom-program-hash) subject; cross-reference it rather than re-deriving.

6. **Custom override-runtime-args?** Does any ProgramFactory define a custom `override_runtime_arguments` (the cached-program override hook)? Yes/No + `file:line`. *Recognition:* a `static void <Factory>::override_runtime_arguments(...)` declaration / definition.

**Finding roles.** None of these gate. Q1 and Q2 are **FYI-U** (team-only — they inform the port's TTNN ProgramFactory wiring). Q3, Q4, and Q6 are also **FYI-P** porter heads-ups, because each presages a device-op-class edit; Q5 (custom hash) is already carried as PORT WORK by the [Custom program hash](#custom-program-hash) subject, so here it's a presence cross-reference. Record all six in the team doc; surface Q3/Q4/Q6 in the porter brief.

**Output.** Record the six answers in `METAL2_PREPORT_AUDIT.md`: the Yes/No summary fills the *TTNN Readiness* rows of the [Status summary](#output-the-two-documents), the full evidence lands under **TTNN factory analysis** in the Team-only section, and the porter-relevant items (Q3, Q4, Q6) are mirrored into **Heads-ups** (and thus the brief).

### Output: the two documents

The audit emits up to **two** files, by audience. Write them to the **op's root directory** — one level above `device/`, alongside the op's host-facing `.cpp` / `.hpp` files (e.g. `ttnn/cpp/ttnn/operations/<family>/<op>/`), **not** inside the `device/` subdir even though the program-factory `.cpp` files live there. They sit next to `METAL2_PORT_PLAN.md` and `METAL2_PORT_REPORT.md` (written later by the port recipe), so all generated docs for the port land in one spot.

- **`METAL2_PREPORT_AUDIT.md`** — **team-facing, always emitted.** The complete record, and the source the cross-team readiness spreadsheet is filled from. Every finding lands here, regardless of role.
- **`METAL2_PORT_BRIEF.md`** — **porter-facing, emitted when the audit clears every gate** — either fully GREEN, **or** YELLOW *only* on isolated Device-2.0 holdovers. (In the holdover case the op is feasible, so the brief issues — no point forcing a re-audit after a trivial cleanup — but it flags the holdovers **prominently as a blocker the porter must clear first**, on the Device 2.0 track, before porting.) The porter's actionable input, ordered by the porter's *workflow*: plan, then construct, then watch-for. On any **RED** there is no brief — there is no port yet.

Findings route to these documents by role (per the [finding-roles routing table](#feasibility-audit) at the top of the audit): the **brief** carries GATE-cleared one-liners + all PORT WORK + all FYI-P; the **team doc** carries everything, including FYI-U.

**Provenance line.** Both files open with a one-line record of the recipe-doc version this audit ran against, so a reviewer can pin the exact guidance. Generate it from the checkout root and paste it verbatim:

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

If the command prints nothing, the docs aren't from a tracked doc-branch checkout — record that instead, since the version can't be pinned.

**Cross-references inside the generated files name the doc plainly** (e.g. `port_op_to_metal2_ttnn_factory.md`) rather than using a markdown link. These files live in the *op directory*, where a doc-relative link (`](port_op_to_metal2_ttnn_factory.md)`) doesn't resolve, and op directories sit at varying depths — so a hardcoded relative path is fragile. The porter has the repo open; a plain doc name is enough.

**In chat, surface only the Result line plus the file path(s)** so the user can open the files when ready. Do not paste a full report inline — an audit of any non-trivial op runs to dozens or hundreds of lines, and chat-scrollback isn't the right home for it. Markdown formatting in the files is required, not optional: the headers, tables, and inline-`code` spans are what make a sizeable report skim-friendly.

**Reassuring framing for the human reader.** A RED gates *this specific port attempt* but is **not** a permanent blocker — most REDs mean "Metal 2.0 hasn't implemented this yet," and the port becomes possible once the feature lands; a few (today: just `address_offset`) need a runtime-team consultation about a redesigned API. Surface the future path explicitly for every RED, so a colleague reads the path forward, not just the gate. In particular, a **ProgramDescriptor-prerequisite RED is the expected outcome** for any op still on the legacy imperative API — not an alarm — and the port unblocks once that op's `ProgramDescriptor` migration lands.

**Code-path scope.** Blockers are often confined to specific code paths (e.g. a single factory's `if (use_width_sharding)` branch). When so, identify clean vs. blocked paths explicitly and offer a scoped-subset port — "interleaved-only paths, omitting the sharded path." A partial port that delivers value now may beat waiting for the full gate to clear; reflect it in the Result (`RED at op level; subset <X> is clear`). **If no clean path exists — the blocking shape is unconditional/structural, not one branch among siblings — say so explicitly (`RED at op level; no portable subset`) rather than leaving it to be inferred from silence.**

#### `METAL2_PREPORT_AUDIT.md` — team-facing (always emitted)

Opens with a **status summary that mirrors the cross-team readiness spreadsheet 1:1** — one field per spreadsheet column, grouped Prereqs / Feature Support / TTNN Readiness — so a parsing reader does straight field→column copies. The detail sections follow. (Extend any cell, row, or paragraph with multi-line context where it improves clarity.)

````markdown
# Metal 2.0 Audit Findings — `<op path>`

<Identifying section: device operations sharing the directory, factory file list, anything needed to disambiguate. For multi-device-op directories, nest — outer bullets for device-operations, inner for their program factories. Example shape:

- **`ReduceDeviceOperation`**
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`**
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)
>

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `<hash> <date> <subject>`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `<path>` |
| **Overall** | GREEN / YELLOW / RED |
| **DOps / Factories** | `<DeviceOperation>` → `<factory list>` |
| *Prereqs* — ProgramDescriptor | Yes / No |
| *Prereqs* — Device 2.0 (every kernel used) | Yes / Yes-with-holdovers (YELLOW — fix on D2.0 track first) / No |
| *Prereqs* — Cross-op escapes | Ok / issue |
| *Feature Support* — overall | GREEN / RED |
| *Feature Support* — Variadic-CTA | Ok / Unsupported |
| *TTNN Readiness* — Op-owned tensors | No / Yes: `<factory + site>` |
| *TTNN Readiness* — MeshWorkload needed | No / No (op-owned tensors — carried natively, single-program) / Yes (genuine): `<reason>` |
| *TTNN Readiness* — Pybind `create_descriptor` | No / Yes: `<nanobind site>` |
| *TTNN Readiness* — Other risky pybind | None / `<description + site>` |
| *TTNN Readiness* — Custom hash | No / Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No / Yes: `<factory + site>` |
| *Ops readiness* — Sync-free CBs (address-only) | None / present: `<(CB, endpoint) sites>` (interim workaround) |

**Sync-free CBs** = CBs used purely as an address source (the kernel grabs the base pointer and walks the memory, no FIFO ops). **Litmus: does any kernel drive the FIFO machinery — a FIFO producer *and* consumer — or is it pointer-only?** (Same core may be both endpoints.) No FIFO producer–consumer pair → sync-free: a Metal 2.0 DFB needs ≥1 of each, so it can't be expressed as a DFB — the port resolves it with the sanctioned interim workaround (see the porting recipe), so it's an **FYI-P heads-up, not a gate**. Granularity is the **(CB, endpoint) edge** — the same CB can be a synchronized LLK operand on one binding and pointer-only on another; record each pointer-only edge.

## Result

**GREEN → brief issued** · **RED → blocked on `<gate>`**, routed to the `<ProgramDescriptor / Device 2.0 / wait-for-feature>` team · **YELLOW → open questions (see below)**. State the primary blocker(s) in plain language; if localized, name a clean subset (`RED at op level; subset <X> is clear`).

## Gate detail

- **ProgramDescriptor:** <GREEN — or — RED with the imperative-API calls that disqualify, plus the "separate ongoing effort; expected outcome for legacy ops; unblocks when the migration lands" framing.>
- **Device 2.0 (every kernel used):** <GREEN — or — YELLOW (isolated CB-index holdovers; routed to the Device 2.0 effort, *not* folded into the port; table below) — or — RED with exact violations @ `file:line`, routed to the Device 2.0 team. Name the kernel file and, for a borrowed/donor kernel, its owning family.>

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `<path>` | `<n>` | `<call>` | `<wrapper>` |

- **Feature compatibility:** every Appendix A entry, in order. Per-row status: `N/A` when the feature category is absent — *including an UNSUPPORTED feature the op doesn't use* (not a vacuous GREEN); `GREEN` only for a LANDED feature actually in use and clean; `RED` for an UNSUPPORTED feature in use. UNSUPPORTED hits (incl. CTA varargs) get an H4 detail block with signal, `file:line` sites, and expected resolution. For `address_offset`, surface the runtime-team-consultation message verbatim per the entry's Action field.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | GREEN / RED / N/A | |
  | Dynamic CircularBuffer (borrowed memory) | GREEN / N/A | port uses `borrowed_from` |
  | CBDescriptor `address_offset` (non-zero) | GREEN / RED / N/A | |
  | Aliased Circular Buffers | GREEN / N/A | port uses `advanced_options.alias_with` |
  | GlobalSemaphore | GREEN / RED / N/A | |
  | Non-zero semaphore initial value | GREEN / N/A | `[[deprecated]]`, Gen2-unsupported — heads-up |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | GREEN / N/A | UNSAFE relaxation opt-in — heads-up |
  | `UpdateCircularBuffer*` | GREEN / RED / N/A | |
  | Variable-count compile-time arguments (CTA varargs) | GREEN / RED / N/A | |

- **DFB endpoint legality (SPSC):** <GREEN — every CB in the legal (1 producer, 1 consumer) window or a port-handled sync-free / single-ended CB — or — RED (config-scoped): each SPSC violation as `(CB, config)` @ `file:line`, its face (hidden 2nd writer / multi-reader), and the op-owner pre-port fix; **and any DM single-ended *producer*** (a single-ended FIFO on a DM kernel — no port-time bridge; op-owner rewrites to a direct tensor write) as `(CB, config)` @ `file:line`; plus the clean subset to port. List any dead CBs for pre-port drop. **Deferred** (re-evaluate post–Device 2.0) if the Device 2.0 gate is RED.>

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `<name>` Case 1 (`TensorAccessor`) / Case 2 (raw pointer → bridge; **blocked in a compute kernel**) / clean (borrowed-DFB).
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception) | none.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:** aliased CB / borrowed-mem DFB / dynamic TA / non-zero sem init — with `file:line` and the construct the port uses.
- **Sync-free CBs (address-only):** each CB used purely as an address source — no FIFO producer + consumer pair — at the `(CB, endpoint)` edge with `file:line`. The port resolves it with the interim workaround (see the porting recipe); it does **not** gate.
- **Dead CBs (zero endpoints):** each allocated-but-unreferenced CB @ `file:line` — the op owner drops the allocation (and any dead CTA carrying its index) pre-port; a functional change, not port work.
- **Cross-op / shared kernels:** borrowed kernel files + shared-kernel coupling.
- **RTA varargs:** kernel + recognition site.
- **TTNN factory analysis (porter-relevant):** pybind `create_descriptor` to delete · other migration-risky pybind · custom `override_runtime_arguments` — each with `file:line`.

## Team-only

- **Out-of-directory coupling & donor shape:** the full by-shape inventory (op-level roll-up, summary table, per-call detail, borrowed kernel files).
- **Relaxation candidates** (mined from the custom hash before deletion): **FALLIBLE — candidates to verify**, default strict.
- **TTNN factory analysis:** the six-question answers — op-owned tensors, MeshWorkload need (genuine vs. op-owned-tensor artifact), pybind `create_descriptor`, other risky pybind, custom hash, custom `override_runtime_arguments` — with `file:line` evidence. Informs the port's TTNN ProgramFactory wiring; does not gate.

## Misc anomalies  *(omit if none; team-only, non-gating)*

<Latent code issues noticed while auditing that are neither audit gates nor porter work — dead/unused RTAs, attributes forced or ignored in the factory yet still fed to `compute_program_hash`, suspicious hardcoded constants, and the like. One bullet each with `file:line`. These route to the op owner; the port does not act on them.>

## Per-DeviceOperation attribution  *(when bundled)*

<One status-summary row per DeviceOperation when the directory bundles more than one and findings differ.>

## Questions for the user  *(omit if none)*

1. **<short title>:** <question, with the `file:line` context that prompted it>

## Recipe notes  *(omit if none)*

<Friction with *this audit recipe itself*, not findings about the op — a step that was unclear or contradictory, a recognition rule that false-fired (name the guard that should cover it), a case the recipe didn't anticipate, a tier boundary that forced an unacknowledged judgment call. Be concrete: cite the section, quote the line. The recipe maintainer reads these.>
````

#### `METAL2_PORT_BRIEF.md` — porter-facing (emitted on all-GREEN, or YELLOW on Device-2.0 holdovers only)

Ordered by the porter's workflow: plan → construct → watch-for. Issued when every gate is cleared — fully GREEN, or YELLOW only on isolated Device 2.0 holdovers (in which case the **Blocked-until** block below is mandatory). Never on RED.

````markdown
# Metal 2.0 Port Brief — `<op path>`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ *(or ▲ if holdovers — see Blocked-until)* · Features ✓

**Recipe docs:** `<hash> <date> <subject>` *(carry this line into the port report's Provenance section)*

<Include this block prominently **only** when the Device 2.0 gate cleared as YELLOW (isolated holdovers); omit it on a fully clean port:>
> ⚠ **BLOCKED until Device 2.0 cleanup.** This port **cannot begin** until these isolated Device 2.0 holdovers are fixed — *separately, on the Device 2.0 track; never in the port diff*: `<kernel:file:line — call → member-form>`, … Once they're clean, proceed with this brief as-is — **no re-audit needed.**

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** <none | `<factory + site>`>
- **MeshWorkload:** <not needed | genuine multi-program need | op-owned tensors only (carried natively — not a real need)>
- **Pybind `create_descriptor`:** <none | delete at `<nanobind site>`>
- **Other risky pybind:** <none | `<description + site>`>
- **Custom `override_runtime_arguments`:** <none | `<factory + site>`>

## Construct — to do

**Tensor bindings** (per binding):

- `<name>` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(ta::name)`.
- `<name>` — **Case 2** (raw pointer) → bind the tensor, pull the base via `get_bank_base_address`, raw walk unchanged. *(Compute kernel → blocked: fail the port pending the compute-kernel `TensorBinding` fix.)*

**Custom hash:** <delete custom `compute_program_hash` → default (sanctioned exception) | none>

## Watch for

- **Notable constructs:** <aliased CB @ loc → [pattern] · borrowed-mem DFB → [recipe] · dynamic TA → [pointer] · non-zero sem init → deprecated, expected | none>
- **Cross-op / shared kernels:** <path → caution per [pattern] | none>
- **RTA varargs:** <kernel → prefer named RTAs | none>
- **Dead CBs:** <each allocated-but-unreferenced CB → op owner drops it pre-port; don't bind it if still present | none>
````

#### N/A vs. GREEN

Per row, the status is one of three:

- **`N/A`** — the feature's precondition is absent from the op (the op uses no semaphores, so `Non-zero semaphore initial value` cannot fire; the op has no variable-count CTAs, so the CTA-varargs entry cannot fire; etc.). **An UNSUPPORTED feature the op simply doesn't use is `N/A`, *not* `GREEN`** — the gate didn't fire because the feature is *absent*, so there is nothing to "pass." This is the common mismark: don't GREEN an absent gate-feature.
- **`GREEN`** — a **LANDED** feature *is* present and clean (e.g. a borrowed-memory DFB in use, translated via `borrowed_from`).
- **`RED`** — an UNSUPPORTED feature is present.

In short: reserve `GREEN` for a feature actually *in use* and supported; an absent feature is `N/A`, whatever its tier. (The *subject's* overall roll-up may still read "GREEN — no gate fired"; that's the subject verdict, distinct from these per-row labels.)

For UNSUPPORTED feature-detail blocks, the **Expected resolution** is usually a short paraphrase of the entry's Status field — e.g. "not yet supported in Metal 2.0; port will be possible once GlobalCircularBuffer support lands on `KernelSpec` / `DataflowBufferSpec`." For the `address_offset` entry specifically, surface the runtime-team-consultation message verbatim per the entry's Action field.

#### RED short-circuit: the ProgramDescriptor prerequisite fires

When the ProgramDescriptor gate fails, the later subjects become moot — there is no point doing a full audit of an op that cannot be ported in this state. **Do not stop with a one-line report.** Instead, fill in the remaining subjects on a best-effort basis (the user benefits from knowing what they'll face *after* the migration clears) and mark each subsequent finding `(observed-but-moot until the ProgramDescriptor prereq clears)`. Don't block the report on completing them; if a subject is impractical to evaluate without the translation, mark it `(deferred — re-evaluate after the prereq clears)` and move on. The Result and Gate detail still emphasize the prerequisite as the primary blocker; the moot-with-caveat findings are forward-looking context.

Save the file(s) and surface the path(s) with the Result line. **Stop here.** The audit file(s) are the complete deliverable of this document.

### After the audit: what happens next

- **On RED**: this op cannot be ported in its current state. Surface the `METAL2_PREPORT_AUDIT.md` path and Result; stop. No brief is written, and the recipe is not loaded.
- **On YELLOW**: surface the path, the Result, and the open questions. Wait for the user's decisions. On resolution, update the team doc in place and confirm GREEN before any handoff.
- **On GREEN + explicit user go-ahead**: both files are written (the team doc and the brief). Load [`port_op_to_metal2_recipe.md`](port_op_to_metal2_recipe.md) to perform the port, passing the audit files as context — the recipe needs the cleared gates and decisions, *including the TTNN factory analysis*. Do not load the recipe on your own initiative; the user must explicitly approve.
- **On GREEN with isolated Device 2.0 holdovers**: both files are written — the brief carries the **Blocked-until** notice. Surface the brief, the team doc, and the holdover list. The path forward: the holdovers are fixed **first**, on the Device 2.0 track (*not* as part of the port), after which the porter proceeds with the already-issued brief — **no re-audit**. The recipe loads once the holdovers are clean and the user gives go-ahead.

---

## Appendix A: Metal 2.0 feature compatibility

This appendix lists legacy-API features relevant to the port. Each entry falls into one of two tiers, declared in the entry's header:

- **UNSUPPORTED** — Metal 2.0 does not currently support this feature. Action: refuse the port and report (a GATE / RED). Each entry's **Status** field describes the future path: most entries will be supported as-is when implemented; a few will only be addressable via a redesigned, semantically different construct (and may require a runtime-team consultation before re-attempting). Always check the Status field before telling the user "wait and revisit."
- **LANDED** — Metal 2.0 supports the feature today; no port gate. The entry's **Status** field names the Metal 2.0 construct that replaces the legacy form. A handful of LANDED features translate via a non-obvious construct or carry a caveat (aliased CBs, borrowed-memory DFB, dynamic `TensorAccessor`, non-zero semaphore initial value) — surface those as porter heads-ups (**FYI-P**) per the [Feature compatibility](#feature-compatibility) subject; they still do not gate.

### Maintenance: keeping Appendix A current

Appendix A is actively maintained as Metal 2.0's feature surface evolves. When framework changes touch a feature listed here, the doc maintainer updates the relevant entry — typically changing the tier (e.g., from `UNSUPPORTED` to `LANDED`) and rewriting the Status / Action paragraphs to reference the new construct.

**Staleness override for porting AIs.** If during the audit you observe a feature in the codebase whose Appendix A entry is marked `UNSUPPORTED` but the framework headers clearly show the API has landed (e.g., the spec/field/method the legacy construct would need to translate to is *visibly present* in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`), this likely means the audit doc is stale. Do not refuse the port reflexively. Instead:

1. Report the row as **YELLOW (staleness override)** rather than RED.
2. In the report's Questions for the user section, flag the apparent discrepancy: cite the Appendix A row, name the framework header / commit that contradicts it, and ask the user to confirm whether the feature is now supported.
3. Respect the user's answer; if they confirm support, proceed with the port using the new construct. The doc maintainer will update Appendix A separately.

This override mechanism is a safety net for the brief windows between a framework merge and the audit-doc update. Do not invent it for entries that are clearly still unsupported (no API surface present); only for cases where the codebase contradicts the doc.

When scanning during the [Feature compatibility](#feature-compatibility) subject, match each feature's recognition signals against the op's source. If any signal matches, take the action declared in the entry.

> **For maintainers adding new entries — skim if you're applying the recipe, not editing it.** Features whose underlying *functionality* Metal 2.0 will *never* support are handled differently: they are either reclassified as a prereq fix (the legacy use is replaced before porting) or get a dedicated fix-up recipe in the port recipe. They do *not* live here, because the action for them is not "wait" or "ask" — it is "transform." (Features whose *current API form* will not be supported but whose *underlying functionality* will be — via a different construct — do belong here as UNSUPPORTED entries; the entry's Status field calls out the redesign requirement.) If you are about to add an entry and the underlying functionality has no planned support, route it to the prereq-fix path or to a port-recipe fix-up entry instead — not here.

Each entry follows this uniform format:
- **Status** — support state and tier framing.
- **Recognition — definitely this feature** — signals that, if matched, mean the feature is in use. Trigger the entry's action.
- **Recognition — false-positive guard** — superficially similar constructs that are *not* this feature. Do not trigger the action on these.
- **Action** — what to do when the rule fires (refuse, or — for LANDED features — note the construct / heads-up).
- **Examples in the wild** — real op locations using this feature, for ground-truthing your match.

If your op uses something not listed here and you are unsure of its support status, treat as yellow and ask the user. Do not assume support from API surface alone.

### GlobalCircularBuffer — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. No equivalent of `experimental::GlobalCircularBuffer` on `KernelSpec` or `DataflowBufferSpec`. GlobalCircularBuffer is a *user-managed* buffer; its eventual analog is the (unimplemented) user-managed `GlobalDataflowBuffer` — the mapping is by *lifetime*. It has **no DataflowBuffer destination of any kind today**: not the local `DataflowBuffer`, and **not** the cross-node DFB stub in `dataflow_buffer_spec.hpp` — despite the legacy *"remote CB"* nickname for GlobalCircularBuffer, that cross-node DFB is a separate *ephemeral* construct with no legacy analog. Do not map a GlobalCircularBuffer onto any DFB variant.

**Recognition — definitely this feature** (refuse and report):

- Any reference to the type `tt::tt_metal::experimental::GlobalCircularBuffer` (qualified or via a `using` alias).
- Calls to `experimental::CreateGlobalCircularBuffer(...)`.
- `#include <tt-metalium/global_circular_buffer.hpp>` paired with any of the other signals (header presence alone is suggestive but not definitive).
- **Descriptor-API attachment**: a `CBDescriptor` literal or struct with its `.global_circular_buffer` field set to a non-null pointer. The type token does not appear at the assignment site — look for the **field name** `global_circular_buffer` on a `CBDescriptor`. This is the arcane signal; an AI scanning a `CBDescriptor` setup can easily miss it.
- **Imperative-API attachment**: the `UpdateDynamicCircularBufferAddress(program, cb_handle, const GlobalCircularBuffer&)` overload (the three-arg form taking a `GlobalCircularBuffer`). The two-arg form taking a `Buffer&` is unrelated.
- Op factory function signatures with parameter type `std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>&` (commonly named `global_cb`).
- **Construction-by-consumption** (the common in-op shape): ops rarely call `CreateGlobalCircularBuffer` themselves — they receive a *pre-built* `GlobalCircularBuffer` and wrap it into a CB via the experimental overload `experimental::CreateCircularBuffer(program, core_spec, cb_config, global_cb)` (the 4-arg form whose last argument is a `GlobalCircularBuffer`), with a `CircularBufferConfig` (often named `remote_cb_config`) configured via `.remote_index(...)`. The "remote CB" idiom — `remote_cb_*` identifiers, `CircularBufferConfig::remote_index()`, the `remote_circular_buffer.h` device header — is itself a strong cue: **in this codebase, "remote CB" means GlobalCircularBuffer.**
- **Both include spellings** resolve to the same type — match either: `#include <tt-metalium/global_circular_buffer.hpp>` and `#include <tt-metalium/experimental/global_circular_buffer.hpp>`.

**Recognition — false-positive guard**:

Plain `CircularBuffer`, `CBHandle`, `CBDescriptor`, or `CBFormatDescriptor` *without* the GCB attachment field set are the regular path → supported in Metal 2.0 as `DataflowBufferSpec`. Do not refuse these. The disambiguator is either the literal token `Global` in the type name **or** the `global_circular_buffer` field on a `CBDescriptor` being non-null. Also do **not** flag the config scalar `num_global_cb_receivers` (or its pybind arg): it is an `int` count, not a buffer — key on the *type* `GlobalCircularBuffer` or the `CreateCircularBuffer(..., global_cb)` / `.remote_index(` construction, not on the `global_cb` substring alone.

**Action**: STOP — at **ProgramFactory** granularity, not necessarily the whole op. Report *which factory* uses `GlobalCircularBuffer` (not yet supported in Metal 2.0); do not invent a workaround. GlobalCircularBuffer is usually confined to a single factory (matmul is the canonical case: only `matmul_multicore_reuse_mcast_1d` uses it, while the op's other factories are clean), so apply **Code-path scope** — RED the offending factory and name the clean factories as a subset port (`RED at op level; subset <X> is clear`) rather than over-blocking the whole op.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/`

### Dynamic CircularBuffer (CB built on borrowed Buffer memory) — LANDED

**Status**: Supported in Metal 2.0. The legacy "dynamic circular buffer" pattern — `CBDescriptor::buffer = <some_buffer>` placing a CB on top of an existing `Buffer`'s memory — translates to Metal 2.0 as a **borrowed-memory DFB** via `DataflowBufferSpec::borrowed_from`. See `tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:95` for the spec field.

**Recognition — definitely this feature** (no port gate; use borrowed-memory DFB):

- **Descriptor-API** (the in-scope path): a `CBDescriptor` literal or struct assignment with its `.buffer` field set to a non-null `Buffer*` (any expression that is not statically `nullptr`). The type token does not appear at the assignment site — look for the **field name** `buffer` on a `CBDescriptor`. The companion field `.address_offset` is meaningful only when `.buffer` is set; it is not an independent signal.
- **Imperative-API** (an op on the imperative `host_api.hpp` builder API also trips the ProgramDescriptor prerequisite RED; these signals additionally catch usage that leaks in via shared utility code — e.g. `cb_utils.hpp` — that the op calls):
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
- `CircularBufferConfig::set_address_offset(non_zero)` (imperative API; an op using it directly also trips the ProgramDescriptor prerequisite, but record matches here too — including any leaking in via shared utility code).
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

### Aliased Circular Buffers (CBs sharing backing memory) — LANDED

**Status**: Supported in Metal 2.0. The legacy aliased-CB pattern — a single `CBDescriptor` whose `format_descriptors` contains multiple `CBFormatDescriptor` elements (each at a distinct `buffer_index`, sharing backing memory) — translates to Metal 2.0 as **aliased DFBs** via `DataflowBufferSpec::advanced_options.alias_with`. Note: aliasing is an **advanced/ninja feature** — the field lives on `DFBAdvancedOptions` (see [`advanced_options.hpp`](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp)), with a CAUTION header documenting that aliased DFBs offer no clobbering guarantees. See the migration guide's "Aliased DFBs" note in the [DataflowBufferSpec section](../metal2_migration_guide.md#dataflowbufferspec) for the porting shape.

The term "aliased CB" is descriptive — it does not appear in the legacy API surface. The legacy expression of this feature lives in the array-shape of certain `CircularBufferConfig` fields (sized `NUM_CIRCULAR_BUFFERS`, almost always populated with one entry) and in the `SmallVector<CBFormatDescriptor, 1>` shape of `CBDescriptor::format_descriptors`. The signal is the cardinality of those collections, not any named field.

**Recognition — definitely this feature** (no port gate; use aliased DFBs):

- **Descriptor API** (the in-scope path): a `CBDescriptor` whose `format_descriptors` initializer contains **more than one** `CBFormatDescriptor` element. Concretely:
  - Single-element (normal): `format_descriptors = {{CBFormatDescriptor{...}}}` → standard `DataflowBufferSpec`.
  - Multi-element (aliased): `format_descriptors = {{CBFormatDescriptor{...}, CBFormatDescriptor{...}}}` (or three+) → one `DataflowBufferSpec` per buffer index, mutually declared via `advanced_options.alias_with`.
  - The differing element is typically `buffer_index` — two distinct indices sharing the same backing storage.
- **Imperative API** (an op on the imperative `host_api.hpp` builder API also trips the ProgramDescriptor prerequisite RED; these signals additionally catch usage that leaks in via shared utility code):
  - `CircularBufferConfig(total_size, data_format_spec)` where `data_format_spec` is a `std::map<uint8_t, tt::DataFormat>` with **more than one** key (e.g. `{{idx1, fmt1}, {idx2, fmt2}}`).
  - Two or more `.set_page_size(buffer_index, ...)` calls **with different `buffer_index` values** chained on the same `CircularBufferConfig` instance.
  - Companion signal: `.set_tile_dims(buffer_index, ...)` chained with multiple distinct `buffer_index` values on the same config.

**Recognition — false-positive guard**:

- A file that creates *many* CBs, each with a single buffer index, is **not** aliased — aliased means a *single* config has multiple indices. Confirm the multiple `set_page_size` calls are on the *same* `CircularBufferConfig` instance, not on different ones.
- Single-element initializers (`{{CBFormatDescriptor{...}}}` for the descriptor form, single-key `{{idx, fmt}}` map for the imperative form) are the dominant pattern by a wide margin → standard DFB for this rule.
- The `CBDescriptor::remote_format_descriptors` field is a *different* concept (relates to remote DFBs, a separate planned feature) and is not covered by this rule. Multi-element values there have a different meaning; do not conflate.

**Action**: Proceed with the port. Declare one `DataflowBufferSpec` per buffer index, each with `advanced_options.alias_with` mutually naming the other(s). All aliased DFBs must have the same `num_entries * entry_size` and must be bound to the same kernels. The `DFBAdvancedOptions` header comments capture the legality constraints; the migration guide section linked above shows the porting shape. **Do not** "split" the aliased CB into independent DFBs — that changes the L1 footprint and breaks the kernel's assumption that the indices share an address.

**Examples in the wild** (op locations whose port exercises this construct):
- Descriptor-API form: currently not exercised by any checked-in ttnn op (every `format_descriptors` initializer in current op factories is single-element).
- Imperative-API form (these ops trip the ProgramDescriptor prerequisite RED in addition to this entry — record both):
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` (around line 840 — output + interim sharing memory; has the comment "share buffer")
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`
  - `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp` (multiple sites — cos/sin interim/sync pairs)
  - `ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_multi_core_program_factory.cpp`
  - `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp`

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

### Non-zero semaphore initial value — LANDED (deprecated; FYI-P heads-up)

**Status**: Supported by Metal 2.0 on Gen1 today, but **explicitly deprecated**. The legacy path lets you create a semaphore with a non-zero initial value via `CreateSemaphore(program, core_spec, initial_value)` (where `initial_value != 0`) or by setting `SemaphoreDescriptor::initial_value` to a non-zero value. The Metal 2.0 destination is `SemaphoreSpec::advanced_options.initial_value` (on `SemaphoreAdvancedOptions`) — which carries a `[[deprecated]]` attribute. The translation works on Gen1, but the field is **unsupported on Gen2** and will be removed when the planned **Remote DFB** feature lands and supplants the use case. Use of non-zero initial values today is therefore a porter heads-up (**FYI-P**), not a gate: it is supported on Gen1 so the port proceeds; the porter just needs to know the field is deprecated and Gen2-unsupported.

**Recognition — definitely this feature** (LANDED — surface as a heads-up):

- `CreateSemaphore(program, core_spec, initial_value)` calls where `initial_value` is:
  - A non-zero integer literal (`1`, `2`, etc.).
  - A constant or symbol whose value is **not self-evidently zero** (e.g. `INVALID`, `INVALID_SEM`, project-specific sentinel constants). When in doubt, treat it as in use and flag the heads-up — do not assume the symbol is zero. If you can resolve the constant's definition and it is in fact zero, no flag.
- `SemaphoreDescriptor` literals or struct assignments where `.initial_value` is set to a non-zero literal or a not-evidently-zero symbol.

**Recognition — false-positive guard**:

- `CreateSemaphore(program, core_spec, 0)` — explicit zero literal → green, no action.
- `SemaphoreDescriptor{ ..., .initial_value = 0 }` — explicit zero literal → green.
- `GlobalSemaphore` is a separate type, covered by its own rule above. Do not match this rule against `GlobalSemaphore` constructions or `experimental::CreateGlobalSemaphore(...)` calls.

**Action**: Surface as a **heads-up (FYI-P)** in the audit. Include in the report:
- The semaphore creation site (`file:line`).
- The initial-value expression as written.
- A note that the construct is supported on Gen1 today but deprecated and Gen2-unsupported.

**This does not gate the port.** The translation is direct — set `SemaphoreSpec::advanced_options.initial_value` to the same value used in the legacy code (acknowledging the `[[deprecated]]` warning and the Gen2 unsupported status). The port's mechanical work is unaffected.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp` (`INVALID` sentinel)
- `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp` (`INVALID` sentinel)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp` (literal `1`)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/moe_gpt_program_factory.cpp` (`INVALID` / `INVALID_SEM` sentinels)

### Dynamic TensorAccessor (`ArgConfig::Runtime*` flavors: RuntimeTensorShape, RuntimeRank, RuntimeNumBanks, RuntimeShardShape, RuntimeBankCoords) — LANDED (UNSAFE opt-in; FYI-P heads-up)

**Status**: Supported by Metal 2.0 today via two opt-ins on `TensorParameter::advanced_options` — but their use carries caveats that make this a porter heads-up (**FYI-P**) rather than a silent green.

- `TensorParameterAdvancedOptions::dynamic_tensor_shape = true` — full relaxation. The bound `MeshTensor`'s `logical_shape` *and* `padded_shape` may differ from the `TensorParameter`'s declared spec. For interleaved tensors the `TensorAccessor` configuration is unchanged; for sharded tensors the accessor reflects the argument's actual shape (shape becomes an implicit runtime argument). This is the translation path for the legacy `ArgConfig::RuntimeTensorShape` (and the closely related `RuntimeShardShape` / `RuntimeBankCoords` flavors).
- `TensorParameterAdvancedOptions::match_padded_shape_only = true` — weaker relaxation. The bound tensor's `logical_shape` may vary, but its `padded_shape` must match the declared spec exactly. `TensorAccessor` configuration is completely unchanged.

Both options are documented **UNSAFE** in the framework header: most kernels will not function correctly if the tensor argument's spec deviates from the declared spec. Adopting them also has structural implications for how the op's factory interacts with the framework's per-dispatch caching path — implications that warrant discussion before a port commits to either opt-in.

The remaining `ArgConfig::Runtime*` flavors — `RuntimeRank`, `RuntimeNumBanks` — do not have a clean translation via these options. Both have ~zero user sites outside tests, so in practice this rarely matters; flag them in the audit for the user's awareness if they appear.

**Recognition — definitely this feature** (LANDED — surface as a heads-up):

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

**Action**: Surface as a **heads-up (FYI-P)** in the audit. Include in the report:
- Each site of `ArgConfig::Runtime*` use (`file:line`) and the flavor in use.
- A note that Metal 2.0 supports the runtime-shape capability via `TensorParameterAdvancedOptions::dynamic_tensor_shape` (or the weaker `match_padded_shape_only`), but the options are marked **UNSAFE** in the framework header and adopting them has structural implications for the factory's interaction with per-dispatch caching. The default remains **strict**; applying a relaxation is an explicit user-OK decision (see the [TTNN integration doc](port_op_to_metal2_ttnn_factory.md)), not an automatic port step.
- If any of the niche flavors (`RuntimeRank`, `RuntimeNumBanks`) appear, surface them separately: those do not have a clean advanced-options translation today.

**This does not gate the port.** When a relaxation is applied, the translation is: set `TensorParameter::advanced_options.dynamic_tensor_shape = true` (full relaxation) or `match_padded_shape_only = true` (padded-shape-only relaxation) on the affected `TensorParameter`(s). The recognition-signal sites — the `ArgConfig::Runtime*` tokens in the legacy code — are the porter's anchor for which `TensorParameter`s need the opt-in.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_hc_tiled_interleaved_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_wh_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

### Per-execution CircularBuffer size updates (UpdateCircularBufferTotalSize, UpdateCircularBufferPageSize, UpdateDynamicCircularBufferAddressAndTotalSize) — UNSUPPORTED via the PD-concept port (legacy factory concept)

**Status**: The legacy per-execution CB-size mutation maps 1:1 to the `ProgramRunArgs` `dfb_run_overrides` fields (`entry_size` / `num_entries`), which Metal 2.0 *does* support at the host-API level. **The catch is the port path:** per-execution CB sizing is **not reachable through the ProgramDescriptor-concept port** this recipe set targets. A PD-concept op (`create_descriptor()` + framework cache-hit patching) cannot mutate CB sizes per execution — the framework patcher re-patches only buffer addresses and hash-excluded scalar RTAs, never `total_size` / `page_size`, which are baked into the descriptor at construction (cache-miss) time. Per-execution CB sizing exists **only through the legacy `create()` + `override_runtime_arguments()` factory concept**, where a hand-rolled override hook calls `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` directly. An op that does this is therefore a **legacy-factory-concept op** — even if it populates a `ProgramDescriptor` internally as a data structure — and so is **not a clean PD-concept port**. This is a Check-1-class prerequisite gap (see [Check 1](#prerequisites)); it GATEs.

`UpdateDynamicCircularBufferAddressAndTotalSize(program, cb_handle, buffer, total_size)` folds the per-execution total-size mutation onto a dynamic-CB address rebind — the address-rebind half maps to borrowed-memory DFBs (supported), but its presence in an override hook is the same legacy-concept signal.

**Recognition — definitely this feature** (refuse and report):

- Calls to `UpdateCircularBufferTotalSize(program, cb_handle, total_size)`.
- Calls to `UpdateCircularBufferPageSize(program, cb_handle, buffer_index, page_size)`.
- Calls to `UpdateDynamicCircularBufferAddressAndTotalSize(program, cb_handle, buffer, total_size)` — the 4-arg "Address And Total Size" combo form. The address-rebind half maps to borrowed-memory DFB; the total-size half is the per-execution CB-sizing this rule catches — its presence in an override hook is the legacy-concept signal. Distinguished from `UpdateDynamicCircularBufferAddress` (3-arg, address only) by the `AndTotalSize` suffix in the function name.
- A canonical grep target that catches all three: `UpdateCircularBuffer`. (See guard below for one false positive.)
- Typical call site: cached-program override hooks (e.g. `override_runtime_arguments` and similar callbacks), where CB sizing is re-tuned per shape between executions of the same Program.

**Recognition — false-positive guard**:

`UpdateDynamicCircularBufferAddress` (the **3-arg** form — `(program, cb_handle, buffer)` or `(program, cb_handle, global_cb)`) is a different function with different semantics — do not refuse based on the `UpdateCircularBuffer` substring matching it. The 4-arg form **with an `address_offset` argument** (`UpdateDynamicCircularBufferAddress(program, cb_handle, buffer, offset)`) is covered by the `address_offset` rule above. The 4-arg form with `AndTotalSize` in the name is **caught by this rule**, not exempted.

**Action**: STOP — GATE. Report (with `file:line` for each `UpdateCircularBuffer*` call) that this op does per-execution CB sizing through a legacy `override_runtime_arguments()` hook, which means it is on the **legacy factory concept**, not the ProgramDescriptor-concept the port requires — a Check-1 prerequisite gap. Do not attempt to port it and do not invent a workaround; route it for a proper ProgramDescriptor-concept migration first. Note for that migration: because the PD concept cannot express per-execution CB sizing, the size variation has to be restructured to be cache-key-driven (a distinct shape → a distinct cached descriptor, sizes computed at construction) — as the already-migrated ops did — *not* preserved via `dfb_run_overrides` (that host-API field exists but is unreachable from the PD-concept port path). In particular, do not "fix" this by recreating the Program from scratch on every execution.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/deepseek_moe_gate_program_factory.cpp` (per-execution `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` in the override hook)
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_program_factory.cpp`

(Earlier examples — `slice_program_factory_rm`, `transpose_wh_sharded_program_factory`, `attn_matmul` / `group_attn_matmul`, `generic_op_program_factory` — have since been migrated to the PD-concept factory and no longer carry these calls. That is the expected outcome of the prerequisite migration, and is why a live match is now rare.)

---

### Variable-count compile-time arguments (CTA varargs) — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. Ops whose structure requires a *variable number of compile-time arguments* — for example, ops that accept a list of input tensors of runtime-varying count, or kernels that iterate over a runtime-varying number of CTAs — cannot be ported today. Metal 2.0's `compile_time_args` schema requires fixed-shape declaration at factory-construction time; there is no kernel-side equivalent of the legacy positional-CTA loop yet. A CTA-vararg feature is on the host API roadmap.

**Recognition — definitely this feature** (refuse and report):

- **Op-level signal.** The op accepts a *variable number of input tensors* — e.g., the device-operation class's `tensor_args_t` carries a `std::vector<Tensor>` (or equivalent variable-count container) rather than a fixed-count tuple of named tensors. The variadic input signature is the strongest cue: if the op author needed a variable-count input list, the legacy kernel almost certainly threads per-input metadata through CTA varargs.
- **Kernel-level signal.** The kernel reads compile-time args using a *runtime-varying index* — e.g., `get_compile_time_arg_val(i)` inside a loop where `i` depends on a count value, or a kernel template instantiated over a variable count derived from a CTA.

Either signal fires the rule.

**Recognition — false-positive guard**:

- *RTA varargs* (`get_vararg(i)` for runtime args) ARE supported in Metal 2.0 via the kernel-side vararg mechanism — see the porter recipe's [kernel-side whitelist rule 4](port_op_to_metal2_recipe.md#kernel-side-whitelist) and the [patterns catalog's Caution on varargs](metal2_port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary). The rule fires only on *compile-time* varargs.
- A fixed-count list of input tensors known at port time (e.g., always exactly 4 inputs) is not variadic — that's a multi-input op with a known shape. Port it as multiple named `TensorParameter`s and `TensorBinding`s.

**Action**: STOP. Report to the user that this op's structure requires a CTA-vararg feature Metal 2.0 does not yet support. Do not attempt to capitulate by demoting CTAs to RTAs (that's the [Demoting per-group CTA to RTA anti-pattern](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)) or by hand-unrolling the variable-count loop in the kernel.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/data_movement/concat/` — accepts a runtime-varying list of input tensors.

---

## After you submit

A grounded RED audit is not a failed port; it is the audit working as designed. The same goes for a YELLOW where you raised the question rather than guess. The deliverable here is clarity — what porting this op would actually require, surfaced clearly enough that a colleague can act on it. Your job in this document was to decide whether the port is feasible, not to perform it; once the report is on its way, that work is complete.
