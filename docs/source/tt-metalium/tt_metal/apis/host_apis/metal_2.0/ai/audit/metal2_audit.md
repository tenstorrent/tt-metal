# Porting an Op to Metal 2.0 — Feasibility Audit

> This is the first of two documents covering the Metal 2.0 op port workflow. **This document covers the feasibility audit only — the gate that decides whether a given op can be ported today.** The port recipe (inventory, planning, construction, verification) lives in [`port/metal2_port.md`](../port/metal2_port.md) and is loaded only after the audit clears with explicit user go-ahead.

## Read this first

**Why this audit exists.** Past attempts to port ops that weren't ready have produced wasted human and agent time, broken code, and PRs that had to be rolled back. This audit is the safeguard: it determines whether a given op *can* be ported today, before any port work begins. When the audit's actual finding is "no, not yet," a clearly grounded refusal — with file references and reasons — is a complete, valid deliverable. Producing such a refusal is not a half-finished port; it *is* the work. Equally, when the actual finding is GREEN, that's the deliverable. Your job is to follow the evidence; no thumb on the scale either way.

**Audience**: AI agents asked to determine whether a TTNN op can be ported from the **`ProgramDescriptor` API** to the Metal 2.0 host API. Humans looking for a conceptual map of API differences should read [`migration_guide.md`](../shared/migration_guide.md) instead.

**If you're new to this stack — quick orientation:**

- **Tenstorrent accelerators** come in two architectural generations. **Gen1** is the shipping silicon today: `WH` = Wormhole, `BH` = Blackhole. **Gen2** is in development: `Quasar` (and siblings). This audit covers Gen1 ops; Metal 2.0 is designed to serve both architectures.
- **TTNN** is the high-level neural-network library for Tenstorrent accelerators. Ops live in `ttnn/cpp/ttnn/operations/<family>/<op>/`. A typical op has a device-operation class on the host side and one or more program factories that build what runs on the accelerator.
  - **Resolving the op you were given.** You may be handed a full path, a `<family>/<op>` (e.g. `data_movement/concat`), or just a bare op name (e.g. `concat`). A `<family>/<op>` appends directly under `ttnn/cpp/ttnn/operations/`; a bare name you locate with `find ttnn/cpp/ttnn/operations -type d -name <op>` — **one** hit is the op directory, **several** means list them and ask the user which, **zero** means say so and stop. Before proceeding, confirm the resolved directory really is an op: it has a `device/` subdirectory containing a `*_device_operation.*` and one or more program factories. (This guards against a bare name resolving to a non-op directory.)
- **Metal 2.0** is the new **host API** — what the program factory uses to declare kernels, buffers, semaphores, and bindings. It also introduces **DFB** (Dataflow Buffer) at the spec layer, replacing the legacy **CB** (CircularBuffer); the two are essentially synonyms on Gen1, but DFB's semantics diverge meaningfully on Gen2.
- **Device 2.0** is a *separate, earlier* overhaul of the **kernel-side** data-movement APIs (safer, more object-oriented wrappers — `Noc`, kernel-side `CircularBuffer` wrappers, etc.). The [Device 2.0 prerequisite](#device-20-prerequisite) gates on Device 2.0 migration — for the op's own kernels *and* any donor kernels it calls (kernels it borrows from other ops or shared pools) — as a hard prerequisite to Metal 2.0, but Device 2.0 is *not* part of Metal 2.0 itself.
- **`ProgramDescriptor` API** is a TTNN-side framework that ops must migrate to before a Metal 2.0 port becomes possible. The [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite) gates on it (an op still on the legacy imperative API shows up on the per-factory readiness sheet — fetched and explained in that subject — as a non-`descriptor` concept, which fails the gate).
- **Common acronyms you'll see throughout:** `CB` = CircularBuffer; `DFB` = DataflowBuffer (see above); `RTA` = runtime args; `CTA` = compile-time args; `CRTA` = common runtime args (values broadcast to all nodes); `TA` = TensorAccessor; `LLK` = Low-Level Kernel (the framework-provided kernel-side primitives); `NoC` = Network-on-Chip (the on-die fabric); `SPSC` = single-producer / single-consumer (per CB instance, per node — a **node** here is one core in the op's core range); `SPMD` = single-program, multiple-data.

For the conceptual map of how Metal 2.0 abstractions fit together — `ProgramSpec`, `KernelSpec`, `TensorParameter` / `TensorBinding`, `DataflowBufferSpec`, the spec/run-args split — see [`migration_guide.md`](../shared/migration_guide.md).

**Scope**: This guide is for **TTNN ops that target Gen1 architectures** (Wormhole / WH, Blackhole / BH), to assess Metal 2.0 portability and gather findings about what the port will require.

The audit produces useful findings for ops in two states:

- **Already on the `ProgramDescriptor` API.** The audit decides whether the Metal 2.0 port can proceed. GREEN → port can begin (after explicit user go-ahead).
- **Still on legacy `host_api.hpp` (imperative builder).** The audit RED's the [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite) — an imperative-builder op isn't on the `ProgramDescriptor` API, so its concept isn't `descriptor` and the gate fails — and continues through the remaining **gate-bearing** subjects, which surface the *other* blockers the eventual Metal 2.0 port must clear alongside the `ProgramDescriptor` migration (a substantial, separate workstream). The porter-only subjects are deferred to the re-audit (see the **Red** outcome scoping rule) — this is blocker-gathering for downstream planning, not a port attempt.

In either case, the audit's deliverable is the report.

This guide is **not** for the following adjacent tasks. If your task is one of these, stop and surface the mismatch to the user — do not use this guide:

- **Porting from Gen1 to Quasar** (different target architecture, different threading model). Out of scope entirely.
- **Porting legacy Quasar tests** (those built against the temporary `experimental::quasar::CreateKernel` / `experimental::dfb::CreateDataflowBuffer` APIs) to Metal 2.0. A separate guide will cover that case; it is not this guide.

If you are unsure whether your task fits the in-scope description, ask the user before proceeding.

**Operating principle**: Your job is to identify gaps, not to invent solutions for unimplemented features.

Some features the legacy API supports are not yet available in Metal 2.0. When you encounter such a feature, the correct response is to **refuse the port and report the gap to the user.**
Which features are unsupported is settled by Appendix A — treat that list as **authoritative**. A construct not listed there is supported (the recipe translates it mechanically); don't gate on it, and don't reverse-engineer support from the API surface.

**Why the auditor carries the judgment.** You run *once*, unhurried, with the whole op in view; the porter runs later, under tighter constraints, one file at a time. So this recipe deliberately pushes judgment onto you to lighten the porter: where a call *can* be made now — classifying a `TensorAccessor` 3rd-argument site, hunting the hidden second CB writer, deciding whether an RTA-vararg count is runtime-varying or fixed — you make it and hand the porter the conclusion, not the raw signal to re-derive. That is the principle behind what this audit keeps versus cuts: a subject earns its place when the auditor can resolve something the porter would otherwise have to.

**What happens to your report.** Each audit report becomes a direct input to a downstream effort — not just an entry in a tracking spreadsheet:

- A **RED** report feeds the **prereq-migration efforts**. The `ProgramDescriptor` migration team and the Device 2.0 migration team consume RED audits to scope and sequence the work that unblocks the Metal 2.0 port. A RED isn't a dead end — it's evidence routed to the team that resolves the gap.
- A **GREEN** report feeds the **Metal 2.0 port recipe** at [`port/metal2_port.md`](../port/metal2_port.md). The porter loads the audit as context when performing the port itself.

Your RED/GREEN verdict, your subset suggestions, and the specificity of your finding details all carry weight in these downstream uses. Too-conservative RED misroutes work to a prereq team that doesn't need it; too-lenient GREEN sends a port attempt into a fail. The single strongest thing you can do is **be specific**: name files and lines; quote the construct you saw; describe what triggered the rule. Vague findings are the hardest to use downstream.

**About this recipe.** This recipe is the product of iteration — earlier auditors' observations have already shaped it, and yours can too. If during your audit a step feels unclear, a rule contradicts itself, the recipe doesn't anticipate a case you're hitting, or guidance conflicts with what you observe in the code, **write it down in the audit report's "Recipe notes" section** rather than silently picking an interpretation. Treat the recipe as your guide, not your shackle. The recipe maintainer reads every report; the friction you log makes the next auditor's job easier.

## Workflow at a glance

Porting an op is a workflow split across two documents:

1. **Feasibility audit.** *(This document.)* Assess this op's Metal 2.0 portability and capture findings — including what work will be required if the port can't proceed yet. Output: write `METAL2_PREPORT_AUDIT.md` (team-facing, always) — and, when the audit clears every gate, `METAL2_PORT_BRIEF.md` (porter-facing) — to the op directory, then STOP.
2. **Port recipe** — legacy inventory, spec planning, construction, and verification. Lives in [`port/metal2_port.md`](../port/metal2_port.md). Loaded only after the audit clears with explicit user go-ahead.

You do not skip the audit. You do not pre-load the recipe document. The audit is its own unit of work.

---

## Feasibility audit

For the op in scope, work through the audit in twelve subjects. The three **prerequisite gates** run first — a failed gate means no port: **[Device 2.0 prerequisite](#device-20-prerequisite)**, **[Feature compatibility](#feature-compatibility)**, and **[TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite)**. The remaining subjects gather the porter's working detail (two of them — **Offset base pointers** and **TensorAccessor 3rd argument** — can also GATE): **[TTNN porting shape](#ttnn-porting-shape)**, **[TensorParameter relaxations](#tensorparameter-relaxations)**, **[Offset base pointers](#offset-base-pointers)**, **[TensorParameter analysis](#tensorparameter-analysis)**, **[TensorAccessor 3rd argument](#tensoraccessor-3rd-argument)**, **[CB endpoints](#cb-endpoints)**, **[Out-of-directory coupling](#out-of-directory-coupling)**, **[RTA varargs](#rta-varargs)**, and **[Incidental anomalies](#incidental-anomalies)**. Each subject's checks have two possible outcomes:

- **Green** — proceed past this check.
- **Red** — record the reason in the audit report and continue. A RED outcome blocks the port on this finding; it does not mean stop auditing. How much of the *rest* of the audit to run depends on whether a port can still happen — **run every gate, skip the porter-only detail once the port is doomed:**
  - **Gate-bearing subjects always run in full, even after a RED.** These are the five that can produce a GATE: **Device 2.0**, **Feature compatibility**, **TTNN factory concept**, **Offset base pointers**, and **TensorAccessor 3rd argument**. Surfacing *every* simultaneous blocker lets the owning teams fix them in parallel — otherwise the user clears one, re-audits, and only then discovers the next, serializing work that could have run at once.
  - **Purely-informational subjects run only when a port is still ahead to consume them.** These seven never gate — they are porter-/team-facing detail: **TTNN porting shape**, **TensorParameter relaxations**, **TensorParameter analysis**, **CB endpoints**, **RTA varargs**, **Out-of-directory coupling**, **Incidental anomalies**. On a **whole-op RED with no portable subset**, **skip them**: no brief will be issued, and the op is re-audited against possibly-changed code once the blockers clear, so producing the detail now is unread and likely stale effort. When a **clean factory subset survives** (a config-scoped gate — see [Code-path scope](#output-the-two-documents)), run them **for that subset** — its brief needs them. **Record each skipped subject** with a one-line note (e.g. `skipped — whole-op RED, no portable subset; re-audit on unblock`) so the omission is never mistaken for a clean result.

  (A grounded RED-no-subset verdict, with every gate surfaced, *is* the complete deliverable — not a truncated audit. The informational census is deferred, not owed.)

**Reference data (recommended).** Before working the subjects, fetch the per-factory porting-readiness data (the readiness *"Operations analysis"* sheet, maintained by the TTNN team) and grep out your op's rows — it pre-classifies several of the signals you're about to check (factory concept, custom hash, RTA-smuggled pointers, pybind-of-internals, custom override-runtime-args). Treat it as an informative **prior, not ground truth**: let it orient your search, but your own `file:line` evidence decides every finding, and you note any place the sheet and your evidence disagree. Fetch procedure + column legend: [`ttnn_op_porting_readiness.md`](../../analyses/ttnn_op_porting_readiness.md). (Path note: the `../../analyses/*` links here resolve to `metal_2.0/analyses/` — a **sibling of `ai/`**, *not* `ai/analyses/`.)

**Scope of the audit.**

- **Follow kernel references, not directory boundaries.** Audit every kernel referenced by any `KernelDescriptor::kernel_source` in the op's program factories — cross-op kernels living in adjacent directories (e.g. `eltwise/`, `data_movement/`, `kernels/dataflow/`) are in scope when the op uses them.
- **Unreferenced kernel files in the op's directory are out of scope.** If the op's directory contains kernel files that no factory references (dead code, tests, work-in-progress), do not audit their contents. If their presence could confuse a reader of the report, mention them in the identifying section as unreferenced; otherwise ignore them.
- **Multiple device-operations in one op directory.** If the directory contains more than one `DeviceOperation` type sharing factories or kernels (e.g. `ReduceDeviceOperation` plus `WelfordReduceDeviceOperation`), audit them together and produce a single combined report. If the device-operations are independent, audit each separately. When it's a borderline call, make your own best-judgement decision on the shared-code test above rather than deferring — the user launching you typically isn't the op's subject-matter owner. **When bundling, retain per-DeviceOperation attribution where findings differ** — name which DeviceOperation (or which of its factories) a given finding applies to, so a downstream consumer (per-op spreadsheet, ticket tracker, port planner) can extract per-DeviceOperation status from the bundled report when their accounting needs it. Bundling reflects the porting unit (shared code → shared port); downstream tools may legitimately operate at the DeviceOperation level (e.g. Tracy profiling, per-op leadership reporting).
- **Routine runtime-arg setup is not a general audit signal — [TensorParameter analysis](#tensorparameter-analysis) handles the one specific case.** Most RTAs translate directly to `KernelSpec::runtime_arg_schema` and `ProgramRunArgs`; treat them as routine port work, not gates. The historical `tensor.buffer()->address()`-as-RTA pattern is the one exception: pre–Metal 2.0 it was style-yuck-but-correct, but under TTNN's recent fast-path-cache binding-injection changes it is now a per-binding correctness hazard. The [TensorParameter analysis](#tensorparameter-analysis) subject explains that mechanism and catches and reports buffer-address RTAs specifically (as Case 1 / Case 2 bindings); routine runtime-arg setup outside that pattern remains non-signal.

**Finding roles and routing.** Every audit finding carries one of four roles, and the role decides which output document it lands in (see [Output: the two documents](#output-the-two-documents)). This table is the audit's backbone — the per-check sections below *produce* these findings; they are not a separate set of rules.

| Finding | Role | Routing |
|---|---|---|
| Device 2.0 compliance (own + donor kernels) | **GATE** | brief: cleared/blocked · team: exact violations → Device 2.0 team |
| UNSUPPORTED feature in use | **GATE** | brief: cleared/blocked · team: detail → wait-for-feature |
| TTNN factory concept — op not on the supported whitelist | **GATE** | brief: cleared/blocked · team: detail → TTNN / PD-migration team |
| TTNN porting shape (target factory concept) | **FYI-P** | brief (TTNN factory analysis section) + team |
| TensorParameter relaxations (from the readiness sheet; auditor confirms) | **PORT WORK** | brief (Construct) + team |
| Offset base pointer (host-folded offset — raw arg or accessor-fed) | **GATE** | brief: cleared/blocked · team: detail → the ops team (Type 1 routine · Type 2 design-discussion, flag early) |
| TensorParameter analysis — Case 1 (`TensorAccessor`) / Case 2 (raw pointer → bridge) | **PORT WORK** | brief (Construct) + team |
| TensorAccessor 3rd argument — page-size arg (Class 1/2 drop) | **PORT WORK** | brief (Construct) + team |
| TensorAccessor 3rd argument — wrong/inexpressible page size (Class 3/4/Special) | **GATE** | brief: cleared/blocked · team: detail → the ops team |
| CB endpoints — endpoint legality (per CB, per node) | **PORT WORK** + **FYI-P** | brief (Construct + Watch-for) + team |
| Cross-op / shared-kernel flags | **FYI-P** | brief (Watch-for) + team |
| RTA varargs | **FYI-P** | brief (Watch-for) + team |
| Out-of-directory coupling & donor shape analysis | **FYI-U** | team only |
| Relaxation candidates mined from a custom hash (fallible; gated ops) | **FYI-U** | team only |
| Incidental code anomalies — dead RTAs, dead-but-hashed attributes, suspicious constants | **FYI-U** | team only |

**The four roles:**
- **GATE** — blocks the port (an unmet prereq or an UNSUPPORTED feature). On PASS, the porter brief carries a one-line "cleared"; on FAIL, *no brief is issued* (there is no port) and the detail routes to the owning team. (A *config-scoped* GATE — e.g. GlobalCircularBuffer confined to one factory — still issues a brief for the clean subset; see [Code-path scope](#output-the-two-documents).) Always complete every *gate-bearing* check even after a GATE fails (so the user sees all blockers at once); the seven purely-informational subjects are skipped when the op is whole-op RED with no portable subset — see the **Red** outcome scoping rule at the top of the [Feasibility audit](#feasibility-audit).
- **PORT WORK** — the porter must *act* on it during the port.
- **FYI-P** — informational, surfaced *to the porter* (and recorded for the team).
- **FYI-U** — informational, *team-only* (feeds other workstreams; never reaches the porter).

Findings flow to the two output documents by role: the **porter brief** carries GATE-cleared lines + all PORT WORK + all FYI-P; the **team findings** doc carries everything. See [Output: the two documents](#output-the-two-documents).

### Device 2.0 prerequisite

Metal 2.0 migration sits at the end of a chain of prior modernizations. The first hard prerequisite is **Device 2.0 data-movement migration**, and it **GATEs the port**. The op's TTNN-side readiness — `ProgramDescriptor` factory concept, custom hash, and the rest — is the *second* prerequisite gate, handled separately in [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite) against the readiness sheet.

**Complete this check regardless of its outcome, then continue through the remaining gate-bearing subjects.** The audit's job is to surface *every* blocker the port must clear — all the prereq gates, not just the first to fail — so the owning teams can work them in parallel. Do not exit early on a RED prereq. (The porter-only subjects follow the **Red** outcome scoping rule at the top of the [Feasibility audit](#feasibility-audit) — run when a port or clean subset is ahead, skip on a whole-op RED with no portable subset.)

**GATE: Device 2.0 Data Movement migration — every kernel the op uses.**

Confirm **every kernel this op exercises** is Device 2.0 compliant — **regardless of where the kernel file lives**. The op's own kernels, shared kernel-library code, in-family shared kernels, and borrowed/donor kernels from other families all count equally; location does not change the gate. What matters is whether the op's program factory instantiates or calls into the kernel. (An op may own *no* kernels of its own and file-path-instantiate all of them from a shared pool — those instantiated kernels are still fully subject to this gate; treat them as the op's effective kernels here.) See the [Device 2.0 Data Movement migration guide](../../../../kernel_apis/data_movement/device_api_migration_guide.md) for what compliance entails. The *coupling* that borrowing induces is inventoried under [Out-of-directory coupling](#out-of-directory-coupling); the *gating* judgment lives here.

- **Green**: every kernel the op uses — wherever it lives — is Device 2.0 compliant. A **sanctioned** CB-index free function is *not* a violation and does not knock the op out of Green: some free functions taking a `uint32_t` CB index are kept by Device 2.0 itself — its own migrated code uses them — so **if Device 2.0 allows the free function, so do we.** Currently sanctioned (do **not** flag): `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, both of which the [Device 2.0 migration guide](../../../../kernel_apis/data_movement/device_api_migration_guide.md) keeps as free functions in its migrated examples. *Breadcrumb:* the Metal 2.0 `DataflowBuffer` now exposes a full tile/format metadata accessor set (in `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h`) — so a Metal 2.0 *port* moves these lookups onto the object ([kernel-side whitelist rule 7](../port/metal2_port.md#kernel-side-whitelist)). That does **not** move the *Device 2.0* boundary here: the `CircularBuffer` wrapper's `get_tile_size()` just forwards to the free function, so `get_tile_size(cb_id)` stays sanctioned at this stage as long as Device 2.0 uses it — check the current Device 2.0 surface rather than assuming the shape alone makes it a holdover.
- **Red (GATE)**: any kernel the op uses — own, shared-library, in-family shared, or cross-family donor — is not fully Device 2.0 compliant. **The port is blocked** until that kernel's Device 2.0 migration lands; route the exact violations to the team that owns Device 2.0 migration, naming the kernel file (and, for a borrowed/donor kernel, its owning family) so the dependency is schedulable. The op is re-audited once the cleanup lands — there is no partial pass. **Note the scope of the incompleteness so the Device 2.0 team can size the work**, since it spans a wide range:
  - *Isolated holdovers* — cheap to clear, but still a gate. The kernel uses `Noc`, `CircularBuffer`, etc. for the bulk of operations and has only a small number of **CB-index-keyed free-function holdovers**: free functions taking a `uint32_t` CB index where the corresponding Device-2.0 wrapper object is already in scope at the call site *and* a wrapper-method replacement exists — e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`, `get_write_ptr(cb_id)` → `cb_obj.get_write_ptr()`. Each is a 1-line mechanical replacement, and the op is otherwise structurally Device 2.0. It is still a gate: a Device 2.0 change is out of port scope even when it's one line (the [kernel-side whitelist](../port/metal2_port.md#kernel-side-whitelist) lets the port touch no Device 2.0 idioms — the port never scoops up stray holdovers), so the fix happens on the Device 2.0 track and the op comes back for a (cheap) re-audit. Report each with `file:line`. (The shape — single CB-index argument, wrapper in scope — is the cue but not sufficient on its own: a *sanctioned* free function per the Green bullet is **not** a holdover.)
  - *Broad Device 1.0* — a full migration, not a cleanup. The kernel broadly uses legacy Device 1.0 idioms (raw `noc_async_read`, manual CB index management, `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, raw sem addresses, etc.).

**Why Device 2.0 gates the port.** Device 2.0 cleanup is *not* on the [kernel-side whitelist](../port/metal2_port.md#kernel-side-whitelist) of sanctioned port-time changes, and — more fundamentally — the Metal 2.0 binding tokens (`dfb::name`, `sem::name`, `tensor::name`) attach to the Device 2.0 wrapper objects. A kernel still on Device 1.0 idioms has nothing for those tokens to bind to, so it cannot take the whitelisted Metal 2.0 swaps. Device 2.0 is therefore a hard structural prerequisite, on par with the TTNN-side readiness gate.

### Feature compatibility

Some legacy-API features are not yet supported in Metal 2.0. If the op uses one, it cannot be ported until support lands — those are **GATE** findings. (Legacy features that *do* have a Metal 2.0 home need no entry here: the port translates them mechanically, so they live in the porting recipe, not the audit.)

**Run this scan regardless of the prerequisite gates' outcome.** Each Appendix A entry's recognition signals work against both ProgramDescriptor-form and imperative-`host_api.hpp`-form code — see the per-entry recognition bullets. Even when a prerequisite gate REDs the op, the feature scan still surfaces which features it uses; that's the data point the human reader needs to plan downstream work.

For each entry in [Appendix A: Metal 2.0 feature compatibility](#appendix-a-metal-20-feature-compatibility), scan the op (host code, kernel code, factory functions, descriptors) using the recognition signals listed for that feature. Every entry is `UNSUPPORTED` — a feature with no Metal 2.0 support yet; using one blocks the port.

- **Green**: no entry's recognition signals fire.
- **Red (GATE)**: an entry's signals match definitively. Report the feature name, the `file:line` where it appears, and the recognition signal that fired. The port is blocked on this finding; continue scanning the remaining Appendix A entries and the other gate-bearing subjects even after a RED match — this feature scan is itself a gate, so it always runs. (The purely-informational subjects follow the **Red** outcome scoping rule at the top of the [Feasibility audit](#feasibility-audit).)
- **Ambiguous match**: if you cannot tell from the code whether the feature is in use, investigate harder — the auditor reads the code, so ambiguity usually means look further. If it stays genuinely undeterminable, default **conservative** (treat it as in use → RED the gate) and record the uncertainty in the report's *Questions* section for the owner. Do not hand the call to the launcher.

A feature **not listed** in Appendix A is supported — port it; don't gate on it or reverse-engineer support from the API surface. (In the rare case something genuinely unsupported slips through, it surfaces during the port and gets added to Appendix A then.)

### TTNN factory concept prerequisite

The second hard prerequisite is the op's **TTNN-side shape**: is it on a factory concept the Metal 2.0 port can handle today, and is its prior `ProgramDescriptor` migration free of the correctness bugs that migration sometimes introduced? Both **GATE the port**. Unlike Device 2.0, you do **not** re-derive this from the code — the hard classification already lives in the **per-factory readiness sheet** (the TTNN team's *"Operations analysis"* sheet). This subject is a *lookup with a lightweight cross-check*, not an analysis.

**Fetch and locate.** Pull a fresh copy of the sheet every run and find your op's row(s) per [`ttnn_op_porting_readiness.md`](../../analyses/ttnn_op_porting_readiness.md) — that doc owns the fetch procedure, the column meanings, and the standing rule to **reference every column by header name, never by position** (the sheet owner adds and reorders columns; existing names are stable and no column is deleted). The sheet has **one row per (op, DeviceOperation, ProgramFactory variant)** — an op with several factories has several rows. Read them all; the gate is per-factory, so carry each factory's verdict through to the report.

**The gate verdict: the `Is able to port?` column.** For each factory row, that cell *is* the gate — `yes` clears it, `no` blocks. It already composes everything below, but you must *understand* its derivation, both to route a `no` correctly and to know what to cross-check:

```
Is able to port?  =
      Is safe to port?          == "yes"
  AND Custom hash               == "no"
  AND Runtime-args update       == "no"
  AND Pybind descriptor         == "no"
  AND ( Concept == "descriptor"
        OR (Concept == "WorkloadDescriptor" AND Secretly SPMD Workload? == "yes") )
```

Two axes are folded into that one verdict, and which one fails decides where a `no` routes:

- **`Is safe to port?` — the correctness axis (the readiness-sheet owner's).** Whether the op's prior `ProgramDescriptor` migration introduced a bug — most often a *smuggled pointer* (in `ProgramDescriptor`, every pointer must ride a CRTA/RTA, and the factory concept only works if each is explicitly marked; a missed one silently mis-patches on cache hits) or a pybound `create_descriptor` whose dependent experimental infra a port would break (surfaced as `warning`). We gate on it only because we won't port a known-buggy op. **This is the readiness-sheet owner's expert judgment — trust it. Do not try to re-derive "did the migration introduce a subtle bug."**
- **The shape criteria — the portability axis (ours).** `Concept`, `Custom hash`, `Runtime-args update`, `Pybind descriptor`: whether the factory shape is one the Metal 2.0 recipe + TTNN infra support today. `descriptor` is the vanilla single-program PD concept; a `WorkloadDescriptor` op clears only if it is *secretly SPMD* — morally single-program (one program across the mesh, on the multi-program path only as an artifact). The rest must be absent. **These gates are current-state, not structural** — each lifts as support lands (the custom-hash gate goes once the recipe interprets custom-hash + relaxation; `Runtime-args update` and pybound-`create_descriptor` go once their TTNN infra ships).

**Lightweight cross-check (trust, but verify).** The sheet is a shortcut to work you'd otherwise do by hand, so confirm it cheaply before relying on it. Verify the **cheaply-checkable factual columns** against the op's code (the factory-definition and device-op files are named in the sheet's last two columns):

- `Concept` — legible from the factory's methods: `create_descriptor()` returning a `ProgramDescriptor` (descriptor), a mesh-workload return (`WorkloadDescriptor`), `create()` + `override_runtime_arguments()` (legacy), or an already-`MetalV2` factory.
- `Custom hash` — grep the device-op for a `compute_program_hash` override. (One case muddies this: a **Pybound-`create_descriptor`** op may *rename* the hook so the framework uses the default hash and reach the custom hash only through the pybind name — see the `Pybind descriptor` note below. Those ops are hand-ported, so don't chase a grep-vs-sheet mismatch on them.)
- `Runtime-args update` — grep the factory for `get_dynamic_runtime_args` / `override_runtime_arguments`.
- `Pybind descriptor` — grep the op's `*_nanobind.cpp` for a `create_descriptor` binding. **These ops are hand-ported (out of this recipe's automated audit/port scope)** and carry bespoke complexity the audit does not untangle — notably a renamed custom hash: `MatmulDeviceOperation`, for one, renames `compute_program_hash` → `compute_descriptor_program_hash` and exposes it only via the pybind name, so the framework silently falls back to the default hash (which is why a naive `Custom hash` grep can disagree with the sheet's `yes` here). Flag the gate and route it; do not reverse-engineer the descriptor internals.
- `Secretly SPMD Workload?` (only when `Concept == WorkloadDescriptor`) — `create_descriptor` returns a `WorkloadDescriptor`; a **single entry** in its `programs` vector (each a `PerCoordProgram` = `MeshCoordinateRange` + `ProgramDescriptor`) ⇒ SPMD.

Also check **cross-column invariants** — a violated one means the sheet is internally inconsistent: e.g. `Runtime-args update == "yes"` is only possible on `descriptor` / `WorkloadDescriptor` concepts, never `legacy device-op`; and `Op-owned tensors? == "yes"` is only possible on `WorkloadDescriptor` (the `descriptor` concept can't carry op-owned tensors, so a `descriptor` row with op-owned tensors is a broken sheet). **Do not verify** `Is safe to port?` (or the smuggled-pointer signal it subsumes) — that is the expert-judgment axis. **Do not fetch or cross-check in a subagent** — the Drive connector authorizes only in the main session, and the cross-check needs the op's code in hand.

**Routing.**

- **`Is able to port?` == `yes`** → TTNN gate cleared; carry the factory forward (its target concept is recorded in [TTNN porting shape](#ttnn-porting-shape)).
- **`no`** → GATE. Read the derivation to attribute the cause and route it: a **shape** failure (`Concept` / `Custom hash` / `Runtime-args update` / `Pybind`) → the **TTNN / ProgramDescriptor-migration team**, with the note that the gate lifts when the relevant support lands; a **`safe`** failure → the **readiness-sheet owner** (the correctness call is theirs; the buggy PD migration must be reconciled before the port). Name which conjunct failed.
- **Cross-check conflicts with the sheet, or the op has no row** → **"spreadsheet is broken"** → GATE, routed to the **readiness-sheet owner** to reconcile. The sheet *is* the analysis we're relying on; if it is wrong or silent for this op, stop rather than proceed on data we can't trust. (Ignore the sheet's trailing summary block — those non-op rows are category totals, not a missing-op signal.)
- **`Concept == MetalV2`** → the factory is *already* ported; report it as done, not blocked.

**Finding role: GATE** (per-factory). On PASS, the porter brief carries a cleared line; on a whole-op FAIL, no brief. Apply [Code-path scope](#output-the-two-documents): if some factories clear and others don't, RED the op and offer the clean factory subset.

### TTNN porting shape

Once the [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite) clears, record the **target** Metal 2.0 factory concept the porter builds toward — a porter heads-up (**FYI-P**), not a gate. The sheet's `Concept` column is the op's *current* concept; the target is derived from it plus `Op-owned tensors?`:

- **`descriptor`** → **`MetalV2FactoryConcept`**. A `descriptor` op has **no** op-owned tensors — the `ProgramDescriptorFactoryConcept` doesn't support them (see below).
- **`WorkloadDescriptor` + SPMD** → **`MetalV2FactoryConcept`**. The SPMD MeshWorkload collapses to the single-program concept; if the op has **op-owned tensors** (`Op-owned tensors? == "yes"`), `MetalV2FactoryConcept` carries them natively.

**Op-owned tensors force `WorkloadDescriptor`.** The `descriptor` concept (`ProgramDescriptorFactoryConcept`) can't carry op-owned tensors, so an op that needs them must move up to the `WorkloadDescriptorFactoryConcept` — where it is expressed as an SPMD workload *purely* to unlock the op-owned-tensor feature. So **`Op-owned tensors? == "yes"` always co-occurs with `Concept == WorkloadDescriptor` (and SPMD)**; a `descriptor` op with op-owned tensors is not a real state. (A quirky design: every op that needs op-owned tensors is morally single-program, yet has to wear the SPMD-workload shape to get them.)

Code basis for `Op-owned tensors?`: a non-empty `buffers` vector on the returned `WorkloadDescriptor` (the field is named `buffers`, not `tensors` — a historical quirk).

The supported target set is narrow today and will grow; the plain `MetalV2FactoryConcept` (no op-owned tensors) is the common target. If a cleared op maps to none of the above, treat it as a spreadsheet/recipe gap and flag it (per the prerequisite subject's spreadsheet-broken routing) rather than guessing.

**Finding role: FYI-P** — surface the target concept in the porter brief; it feeds the port's TTNN ProgramFactory wiring (see [`ttnn_factory.md`](../shared/ttnn_factory.md)).

### TensorParameter relaxations

The sheet's `TensorParameter relaxation` column proposes, per factory, a relaxation the port should apply so the ported op accepts the same range of tensor shapes the legacy op did — `dynamic_tensor_shape`, `match_padded_shape_only`, `none`, or a descriptive `OTHER(...)`. This is **PORT WORK** (the porter applies the relaxation on the affected `TensorParameter`), not a gate.

**Not yet active — but coming very soon.** A relaxation-bearing op has a **custom hash** (the relaxation *is* the hash excluding the relaxed property from the cache key), and the [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite) currently gates custom-hash ops — so today a real relaxation value co-occurs with a gate and this subject rarely fires. It activates when that gate lifts (the recipe learning to interpret custom-hash + relaxation).

**The check when it fires (involved, not a passthrough).** Applying a relaxation must not change behavior — the port promises no semantic difference. So confirm the **existing custom hash's logic matches the listed relaxation**: the properties the hash excludes from the key must be exactly the ones the relaxation says may vary. A mismatch would let the cache reuse a program for an input the legacy op would have rebuilt for — a silent semantic change. On mismatch, **do not apply the relaxation**; flag it (`file:line` for the hash, the sheet's proposed relaxation, and the discrepancy) for the ops team.

**Relaxation candidates (FYI-U).** Even while custom-hash ops are gated, a custom hash sometimes reveals which tensor properties the op *actually* depends on — a candidate for a future relaxation. Record any such candidate for the team's relaxation roadmap; it is **fallible** (many custom hashes are themselves wrong) and never reaches the porter brief.

**Finding role: PORT WORK** (with a mismatch flagged to the ops team); the relaxation-candidate note is **FYI-U**.

### Offset base pointers

Metal 2.0 hands a kernel only the **base address** of a bound memory object — via a `TensorBinding` (the base rides a CRTA; the kernel rebuilds addresses through a `TensorAccessor` whose args the framework auto-builds from that base) or a DFB `borrowed_from` a `TensorParameter` (base only). There is **no mechanism to deliver a non-base pointer** — an interior address, a `base + offset`, or a base the framework can't regenerate. So a legacy op that folds an offset into a device pointer *before* it enters a kernel arg hits a **porting wall**: binding the tensor recovers the base but loses the offset. Splitting that offset back out — passing base + offset as separate args and adding them in the kernel — is a **semantic change** (new arg, kernel-side arithmetic, a possible minor perf shift), off the porter's [kernel-side whitelist](../port/metal2_port.md#kernel-side-whitelist) (a pure syntax swap). So where a fold is present it **GATEs the port**, and the **ops team** clears it first, on their own track — never in the port diff. The affected ops are few.

**Run this before [TensorParameter analysis](#tensorparameter-analysis): same scan, one question that comes first.** Both subjects inventory the op's `buffer()->address()` RTAs. TensorParameter analysis treats each as a clean base and classifies it Case 1 (fed to a `TensorAccessor`) or Case 2 (raw pointer) — both port work. This subject asks the question that has to precede that: is the address a clean base, or `base + offset`? An offset-bearing address is **this gate**, not a portable Case 1/2 — fork it out here, so TensorParameter analysis only ever sees clean bases. (Classify an offset base as Case 1/2 and the offset silently vanishes — the ported kernel mis-addresses, with nothing to flag it.)

**The checked-in triage is a prior, not an authority — your own scan is the source of truth.** The triage doc [`2026-07-19_offset_base_pointers.md`](../../analyses/2026-07-19_offset_base_pointers.md) carries op→type tables with each offending arg named by **kernel + role** (`reader RTA — src_bank_addr`), not line number — a shortcut layered on top of the recognition scan below, not a substitute for it. This is a **different contract from the [readiness sheet](#ttnn-factory-concept-prerequisite)**: the sheet is maintained and authoritative, so a disagreement means *the sheet is broken* (stop and escalate); this doc is **dated and not kept current**, so a disagreement means *the doc is stale* — trust your scan. Two things drift after a dated analysis, in opposite directions: the ops team may have **fixed** a catalogued op (→ now GREEN), and an op author may have **introduced a new fold the doc never saw** (→ yours to catch). So run the recognition scan on **every** address RTA, and use the doc to accelerate and cross-check it — a strong prior on the listed ops, a backstop that flags a fold you missed. **Never let "not in the tables" stand in for "scanned and clean."** The scan is nearly free here: you are already resolving every address RTA for [TensorParameter analysis](#tensorparameter-analysis); the only added question is whether an offset is folded in.

**The four types.** Every non-base pointer falls into one of four types — two gate, one is covered elsewhere, one is a non-issue:

| Type | Source | Disposition | Role |
|---|---|---|---|
| **1 — raw offset arg** | host folds `base + offset` into an RTA; kernel uses it **raw** as a NoC address | ops team splits it — pass base + offset as *separate* args, add in the kernel (mechanical, routine) | **GATE** → ops team |
| **2 — accessor-fed offset arg** | same host fold, but the offset address is fed to a **`TensorAccessor`** as its base | ops team resolves it — **not mechanical**; needs design discussion (RM-path rejigger, or a tensor-view feature) | **GATE** → ops team **+ framework/Audrey, flag early** |
| **3 — borrowed-CB `address_offset`** | a CB borrowed at a non-base offset (`address_offset`) | **already gated** by the [`address_offset` entry in Appendix A](#appendix-a-metal-20-feature-compatibility); DeepSeek-Python-only, out of C++ scope | (covered there — no separate gate) |
| **4 — `narrow` MeshTensor trick** | `ttnn::narrow`'s interior-base sub-tensor view (`MeshBuffer::create(…, parent_base + offset)`) | **ports as-is** — interior base delivered verbatim, re-emitted each enqueue (cache-safe), size accounting narrowed; correct provided the narrowed `TensorSpec` matches, which `ValidateTensorArgs` enforces | **not a gate — do not flag** |

**Recognition.** You are already scanning address RTAs for [TensorParameter analysis](#tensorparameter-analysis); the extra question here is whether an offset is folded into the base. Resolve each address RTA to its host computation (trace the RTA/CRTA back to the factory, following any local variable a few lines away):

- **Type 1** — `…->address()` with host arithmetic folded in (`buffer()->address() + <offset expr>`), consumed on-device **directly** as a NoC address (the `.addr` of `noc_async_read`/`noc_async_write`), never through a `TensorAccessor`.
- **Type 2** — the same `…->address() + <offset>` fold, but the offset address is fed to a `TensorAccessor` as its **base** (`TensorAccessor(args, that_addr, …)`). The offset *is* the accessor's base — **not** a relocatable trailing `+` — so the Type-1 fix doesn't apply, and Metal 2.0's implicitly-constructed accessor leaves no seam to hand it an offset base (why it gates — see Routing).
- **Type 3** — `.address_offset` non-zero, `set_address_offset`, the 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor(…, address_offset, …)`. Recognized and gated by [Appendix A](#appendix-a-metal-20-feature-compatibility); no C++ op in scope passes a non-zero offset.
- **Type 4** — `ttnn::narrow`, or a `MeshBuffer::create(…, parent_base + offset)` interior-base view. **Not a gate** — see the table.

In every catalogued Type-1/2 op the offset is deterministic from cache-miss inputs (shard / bank / head geometry), so the fold is a real addressing pattern, not a one-off constant.

**Reconcile each RTA against the doc — four outcomes.** For every address RTA you resolved, cross the *fold?* answer with whether the op is in the tables:

- **Fold present, op in the tables** → confirmed; gate per the table (Type 1/2).
- **Fold present, op _not_ in the tables** → a fold introduced since the analysis; classify it by the recognition model above and gate it (Type 1 raw / Type 2 accessor-fed). **Do not wave it through for being unlisted** — this is the case a doc-anchored lookup would silently miss.
- **No fold, op in the tables** → the doc is stale: the ops team split the offset out (a bare `->address()` base + a *separate* scalar offset arg, added in the kernel). **GREEN** — the op drops to ordinary [TensorParameter analysis](#tensorparameter-analysis) port work (Case 1/2 on the now-clean base). *(Not hypothetical: `roll`'s own DRAM_RM mode already passes base + offset separately, in the same file as its Type-1 DRAM_TILE mode.)*
- **No fold, op not in the tables** → clean; hand the RTA to TensorParameter analysis.

If you can't resolve whether an offset is folded in, **don't guess** — flag it (`file:line`) for the triage-doc owner and gate the site conservatively.

**Routing.** A Type-1/2 gate routes to the **ops team** to refactor the op *before* the port — the fix is theirs, not the porter's:

- **Type 1** → ops team, **mechanical** arg-split (routine). Record the op, the arg by kernel + role, and the offset expression.
- **Type 2** → ops team **and** framework/Audrey, **flag early**. The difficulty is structural: in legacy the op assembles the `TensorAccessorArgs` and passes the base pointer explicitly, so it can hand over `base + offset` as the base; Metal 2.0 builds the accessor straight from the `tensor::name` binding, which rolls the (auto-built) args and the base-only address together implicitly — no seam for an offset base. The resolution needs design discussion (the affected variants are all row-major): the likely-but-unsettled shape is a normal `TensorAccessor` on the clean base plus kernel-side pointer manipulation (a base-binding + kernel-side offset, at a possible perf cost), weighed against a first-class tensor-view binding. Surface it prominently, not as a mechanical porter task.

**Code-path scope — the gate is factory-scoped.** The wall is a **row-major-layout** phenomenon: the *tiled* variants of the same ops (`slice` / `padded_slice` / `slice_write`) pass a clean tile-index scalar and are unaffected. So a slice-family RED applies [Code-path scope](#output-the-two-documents) — RED the RM factory and name the tiled factories as a clean subset (`RED at op level; subset <tiled factories> is clear`).

**Finding role: GATE** (Type 1, Type 2). On a clean scan — no offset fold, or every fold already split out — the porter brief carries a one-line "cleared." On a Type-1/2 hit there is no brief for the affected factory: record the op, the arg (kernel + role), the offset expression, and the route (ops team; Type 2 also framework/Audrey). Type 3 belongs to Appendix A; Type 4 ports as-is.

### TensorParameter analysis

Every tensor a kernel reads must reach Metal 2.0 through the typed binding channel (`TensorParameter` / `TensorBinding`). This subject inventories, **per binding**, how the legacy op accesses each tensor and classifies the port work into one of two cases. Both cases are **PORT WORK** — the porter acts on them during the port; neither gates.

**Clean bases only — offset-bearing addresses are gated upstream.** An address RTA with a host-folded offset (`buffer()->address() + <offset>`) is **not** a portable Case 1/2 — it is the [Offset base pointers](#offset-base-pointers) gate, resolved *before* this subject. Classify only **clean-base** address RTAs here; a Case-1/Case-2 verdict on an offset-bearing base would silently drop the offset.

**Why this matters.** The legacy hazard is a tensor base address that reaches the kernel through an RTA/CRTA. Under TTNN's fast-path-cache binding-injection model, the framework patches the typed-binding channel on cache hit but leaves RTAs untouched — so a buffer address routed through an RTA stays at whatever value the cache-populating call wrote, and on later cache hits with new tensor storage of the same shape the kernel reads the *original* buffer. No assertion fires; just wrong numerics, only on cache hits with non-identical storage. Pre–Metal 2.0 this was a style concern (*"yuck, raw pointers"*); the fast-path change made it silently wrong. The port replaces that RTA-smuggled address with a typed `TensorParameter` binding (which *does* refresh) — regardless of what the kernel then does with the base pointer (the two cases below), and including kernels that use no `TensorAccessor` at all (older addr-gen idioms, hand-rolled NoC walks).

This resolves **at port time** — it waits for no framework feature.

**Scope.** Audit only kernels that actually touch tensor memory. **Compute kernels that only consume from / produce to circular buffers are out of scope** — they read CB pointers, not tensor memory; the tensor read happens upstream in a dataflow kernel.

**Causal-link gate (run this first, per binding).** Before classifying any binding, check whether the kernel's access is a **borrowed-memory DFB read**: it reads tensor data through `cb_*.wait_front` / `cb_*.get_read_ptr` from a CB that is itself a borrowed-memory CB (a CB backed by a device buffer — host-side `set_globally_allocated_address(buffer)`; the borrowed-memory translation is a mechanical porting-recipe step via `DataflowBufferSpec::borrowed_from`, not an Appendix A gate). There the lack of `TensorAccessor` is *intended* — the borrowed-memory DFB **is** the tensor access — and the port handles it via `DataflowBufferSpec::borrowed_from`. Mark such a binding **clean**; do not force it into Case 1 or Case 2.

Marking a binding **clean** here defers the *legality* of that borrowed-memory CB — is it a valid 1:1 DFB, a sync-free CB (resolved by a self-loop or, with two touchers, a 1P+1C assignment), or a genuine multi-endpoint CB? — to [CB endpoints](#cb-endpoints), which assesses every CB per node. Do not resolve it here: mark the borrowed-memory DFB read clean and move on.

**The two cases.** For each `TensorParameter` the op declares (or would declare in the port), classify by **what the kernel does with the tensor's base pointer**. In *both* cases the legacy host smuggles `buffer()->address()` in through an RTA/CRTA — the distinction is purely what the kernel does with that raw pointer on the device side, not whether one exists. A mechanical observation, not a judgment call:

- **clean** — a borrowed-memory DFB read (the causal-link gate above). The DFB *is* the tensor access; the port handles it via `DataflowBufferSpec::borrowed_from`. Neither Case 1 nor Case 2 — no work item here.
- **Case 1 — via `TensorAccessor`** (the common case). The kernel feeds that base address into a `TensorAccessor` constructor (`TensorAccessor(args, addr)`) and does all its memory access *through* the accessor (`accessor.get_noc_addr(page)` and friends). **Port work:** express the binding as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` instead, and the legacy address-via-RTA plus its `TensorAccessorArgs` plumbing both disappear. Mechanical, low-risk.
- **Case 2 — raw pointer.** The kernel uses that base address *directly* — explicit address arithmetic and hand-rolled NoC calls, never constructing a `TensorAccessor` from it. **Port work:** express the binding as a `TensorParameter` / `TensorBinding` too (the address never stays on an RTA), but the kernel pulls the base via the sanctioned `TensorAccessor::get_bank_base_address` bridge and **keeps its existing raw arithmetic unchanged**. We do **not** rewrite raw access into `TensorAccessor` iteration — that conversion is deliberately out of scope (too high-risk for a port). A buffer-address RTA is *not* an acceptable substitute for the bridge.

**Detection — host side.** Any site where `buffer->address()` (or `->address()` / `(*buffer).address()` / `tensor.buffer()->address()`) flows into a runtime-args context. Common shapes:

- **Descriptor form** (the in-scope case for `ProgramDescriptor`-API ops): `KernelDescriptor::runtime_args` or `runtime_common_args` initializers containing the address expression directly. E.g. `kd.runtime_args = {{core_coord, {input_buffer->address(), num_pages}}};`.
- **Imperative form** (only in ProgramDescriptor-prereq RED ops, but record matches since the eventual port still needs them): `SetRuntimeArgs` / `SetCommonRuntimeArgs` argument lists containing the address expression.
- **Helper-function form**: a function takes a `Buffer*` / `Buffer&` and injects its address into an arg vector (`args.push_back(...)`, in-place vector init, named accumulator). Read the helper body — it often hides the bypass.
- **`Buffer*`-binding form** (descriptor API): the factory pushes a `Buffer*` (the pointer object itself, *not* `->address()`) into `KernelDescriptor::RTArgList` / `emplace_runtime_args`. The framework auto-registers these as `BufferBinding`s and **patches them on cache hits**, so this shape is *correct-on-cache-hit today* — it is **not** the silent-wrong hazard. (It's the framework's interim hack for plugging the stale-pointer hole in `ProgramDescriptor` ports; the Metal 2.0 typed binding supersedes it.) **Still enumerate it** — the kernel receives a raw `uint32_t` base, so **classify it by what the kernel does with that base** (per the two cases above, exactly as for an `->address()` RTA): fed into a `TensorAccessor` → **Case 1**; used raw in hand-rolled NoC arithmetic → **Case 2** (bridge via `get_bank_base_address`). The `Buffer*` is only the *delivery* mechanism — it does not by itself imply raw consumption. Either way, record it as routine port work, not a correctness hazard (the framework patches it on cache hits today). Enumerating *all* pointer arguments is the point; just don't over-state the urgency of this one.
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
- Host-side `set_globally_allocated_address(buffer)` — the borrowed-memory pattern (a mechanical porting-recipe translation via `borrowed_from`); clean via the causal-link gate.
- Kernel reading from a borrowed-memory CB via `cb.get_read_ptr()` — clean; the causal-link gate applies.

**Granularity — per binding, not per op.** An op may have multiple tensor bindings, some clean and some needing work. Report per binding — a single Case-1 or Case-2 binding fires this subject even when the op's primary I/O is via `TensorAccessor`. **Classification can also vary per factory within one bundled op** — the *same* `TensorParameter` may be clean (borrowed-memory DFB) in one factory and Case 1 in another (e.g. a sharded vs. an interleaved factory). When that happens, record the per-factory split via the report's Per-DeviceOperation attribution rather than forcing one flat verdict for the binding.

**Op-level roll-up:** `✓ clean` (every binding clean) / `⚠ port work` (one or more Case-1 / Case-2 bindings), with the per-binding inventory in the report.

### TensorAccessor 3rd argument

`TensorAccessor`'s constructor takes an **optional** third argument — an explicit page size (`TensorAccessor(args, base_addr, page_size)`). Most accessors omit it and only a handful of ops set it, so this subject fires rarely. When it *is* present, deciding what the port does with it is **not** the mechanical drop it looks like — a subtlety (below) tripped up the first pass through these ops.

**The checked-in triage is a prior, not an authority — your own read is the source of truth.** The triage doc [`2026-07-06_tensor_accessor_3rd_arg_triage.md`](../../analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md) carries an op→class lookup table — a shortcut layered on the classification model below, not a substitute for it. This is a **different contract from the [readiness sheet](#ttnn-factory-concept-prerequisite)**: the sheet is maintained and authoritative, so a disagreement means *the sheet is broken* (stop and escalate); this doc is **dated and not kept current**, so a disagreement means *the doc is stale* — trust your read. Two things drift after a dated analysis: the ops team may have **fixed** a catalogued site (value corrected or accessor refactored → the class changes), and an op author may have **added a new 3rd-arg site** the doc never saw. A new site can't hide from you — a 3rd argument is a syntactic signal you scan for regardless — but you must then **classify it yourself** (the two questions below are complete for that), not assume an unlisted site is benign. So scan every accessor that passes a 3rd arg, and use the doc to accelerate and cross-check: a strong prior on the listed ops, a backstop for a site you under-classified. **Never let "not in the table" stand in for "classified Class 2."**

**The load-bearing subtlety: the 3rd arg means different things for sharded vs. interleaved.** The two `TensorAccessor` specializations interpret the page-size argument differently, so you **cannot judge the value until you know which specialization the accessor is**:

- **Interleaved** strides by `align_power_of_2(page_size, alignment)` — it *silently realigns* the passed value up to the allocator alignment. So any value of the **right order of magnitude** is inert; only a wrong-*magnitude* value (off by more than alignment rounding) actually mis-addresses. This realignment is what dissolved most of the "bug" flags an earlier sweep raised — they were correct-magnitude-but-unaligned interleaved values, i.e. harmless.
- **Sharded** uses the passed value **verbatim** as the stride. There is no realignment safety net — *any* wrong value mis-addresses.

Metal 2.0 supplies the `aligned_page_size` implicitly and provides **no explicit page-size override API** (there is no valid use case for one). So the port's default job is to confirm the manual override is redundant and drop it — *unless* the override is load-bearing or wrong, which is where the gate lives.

**The taxonomy.** Every accessor that passes a 3rd argument falls into one of five classes. The triage doc enumerates which op is which; the classes are captured here so a blocking verdict routes correctly:

| Class | What the 3rd arg is | Port action | Role |
|---|---|---|---|
| **1 — Dynamic page size** | genuinely *varies* with row width across cache-reused shapes (interleaved row-major) | set `dynamic_tensor_shape` and drop the manual override — the page size is then auto-supplied | **PORT WORK** (a relaxation — see [TensorParameter relaxations](#tensorparameter-relaxations)) |
| **2 — Redundant / inert** | `== aligned_page_size`, **or** a correct-magnitude value on an *interleaved* accessor (realigned), **or** never used (`page_id ≡ 0`) | drop the arg (pure no-op) | **PORT WORK** (mechanical) |
| **3 — Latent bug** | a **wrong-magnitude** value the realignment can't repair, but masked today by a dead path / test config | — (blocked) | **GATE** — the ops team fixes it first (the recipe makes no functional changes) |
| **4 — Live bug** | a **wrong-magnitude** (or sharded-verbatim-wrong) value that mis-addresses **in the default config** | — (blocked) | **GATE** — the ops team analyzes and fixes it first |
| **S — Special** | a manual override the binding model **cannot express** — a sharded raw-pack page, or a sub-page base offset | stays on manual addressing | **GATE** — the ops team ports it by hand |

**The two questions that classify any site** (in the table or not): resolve these two and they place the accessor in the taxonomy above. For a listed op they confirm its row — or override it, when the op was fixed or refactored after the analysis; for an unlisted one they *are* the classification, no table needed.

1. **Sharded or interleaved?** (`src_args.is_sharded`, or trace the tensor's memory config.) This decides whether realignment is in play at all.
2. **Correct or wrong magnitude?** Resolve the expression to a host value (trace the CTA/RTA back to the factory) and compare to the true logical page (`buffer->page_size()`): `tt::tile_size` / `get_tile_size(cb)`, `buffer->page_size()`, and `aligned_page_size()` are correct magnitude; `element_size()*1024` (drops the block-float exponent — bf8 gives 1024, not 1088), a stale hardcoded constant, or a sub-page fragment are wrong magnitude. Evaluate alignment on **Blackhole/Quasar DRAM (64)** — the strictest, and this hardware's target.

If your op isn't in the table, or the code no longer matches its row (the accessor changed sharding, the op was fixed on `main`, the value moved), **classify it yourself from the two questions above** — that's what they're for; the table's silence or staleness doesn't block you. Escalate only on genuine ambiguity: if you can't resolve sharding or magnitude, **don't guess** — flag it (`file:line`) for the triage-doc owner and gate the site conservatively (a possibly-wrong-magnitude value gates).

Divergence from a clean drop is rare — across all current ops the triage found just one latent-bug op and a couple of Special cases — so the overwhelmingly common outcome is Class 2, a clean mechanical drop. But **do not port a divergent site silently**: Classes 3/4/S each block the port until the owning team acts.

**Finding role: PORT WORK (Classes 1, 2) + GATE (Classes 3, 4, Special).** On a Class-1/2 site the porter drops the arg (Class 1 also sets the relaxation, cross-referenced to [TensorParameter relaxations](#tensorparameter-relaxations)); on a Class-3/4/Special site the port is blocked — record the op, the accessor `file:line`, the class, and the route (Class 3/4 → the ops team to fix the value; Special → the ops team to port by hand).

### CB endpoints

This subject assesses **CB endpoint legality**: for each CB, on each node, is the number of endpoints legal? It unifies the *floor* (too few endpoints) and the *ceiling* (too many) into one classification. **It is now GATE-free** — every out-of-window case has a port-time resolution (a self-loop, a **1P+1C role assignment** across two distinct touchers, the multi-binding advanced option, or a documented drop), so nothing here blocks a Gen1 port. The auditor's job is to classify each CB and record the disposition the porter will apply. (The refactors that some of these resolutions defer belong to the later **Quasar-uplift** stage — a separate, out-of-scope effort with its own audit — never port work here.)

**Precondition — Device 2.0 idioms intact.** The recognition signals assume Device-2.0 kernel idioms (`get_write_ptr` methods, `get_local_cb_interface`, `Semaphore` objects). Run this only when the op's kernels are structurally Device 2.0 — the [Device 2.0 gate](#device-20-prerequisite) is GREEN, or RED **only** on isolated CB-index holdovers (the idioms the scan keys on are intact). On a **broadly** Device-1.0 RED op, **defer** it — mark `(deferred — re-evaluate after Device 2.0 migration)` — because the kernel rewrite changes the idioms the scan keys on, and a best-effort pass over legacy idioms can false-negative the hidden second writer (worse than deferring: it would report "clean").

**The endpoint census.** Count **per CB, per node** — a CB is a device-side per-node instance (one `CBDescriptor` over a core range makes one instance per node), so tally endpoints on each node's instance. An endpoint is any kernel that touches the CB: FIFO-produces (`reserve_back`/`push_back`), FIFO-consumes (`wait_front`/`pop_front`), **or** accesses the memory by **raw pointer** (`get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(<cb>).fifo_*_ptr`). Any access counts — in Metal 2.0 a kernel cannot touch a DFB it hasn't bound, so every access is a binding, hence an endpoint (the hidden raw writers below included).

> **Count *every* access — but the disposition turns on *how* each kernel touches the CB, not just how many.** The easy miss is a non-FIFO access: a raw-pointer read or write is a binding in Metal 2.0, hence a toucher, exactly like a FIFO produce/consume. A CB with no FIFO ops still has touchers (its pointer readers/writers) — a lone pointer reader is one, two on a node are two — so never skip a CB just because nothing `push_back`s into it. But once you have the full census, the resolution is **not** a pure function of a raw endpoint count: a *sync-free* touch does not lock the DFB into a producer or consumer role, so two sync-free touchers resolve to a plain **1P+1C**, not a multi-binding. Count every access; then read the roles off the census.

**Classify by the census of distinct touchers on a node** — how many distinct kernel instances touch the CB, and whether each is *locked* to a FIFO role or is *role-free*:
- A kernel that **FIFO-produces** (`reserve_back`/`push_back`) is a **locked producer**; one that **FIFO-consumes** (`wait_front`/`pop_front`) is a **locked consumer** — its binding role must match its FIFO ops.
- A kernel that only **raw-*peeks*** (`get_write_ptr`/`get_read_ptr` + a direct read/write, no FIFO ops) is **role-free**. On Gen1 the DFB lowers to a plain circular buffer, and the producer/consumer label drives FIFO machinery a role-free kernel never invokes — so the label is *cosmetic*, assigned purely to satisfy the validator. (`get_*_ptr` is a public peek on either role, so the assignment is never blocked by access control.) **But only true peeks are role-free:** a raw call that *mutates* the FIFO cursor (`evil_set_write_ptr`/`evil_set_read_ptr`) drives the shared cursor rather than reading it, so it **locks** the corresponding role exactly like a FIFO op — tag a write-cursor driver as a locked producer, a read-cursor driver as a locked consumer.

| Census on a node | Verdict | Port-time resolution |
|---|---|---|
| 0 touchers | **Dead CB** — allocated, never referenced | porter drops it + documents (below) |
| 1 toucher | **single-ended / sync-free** | **self-loop** — bind the one kernel PRODUCER **and** CONSUMER (legal on Gen1 for compute *and* DM) |
| 2 touchers, ≤1 locked producer **and** ≤1 locked consumer | **plain 1:1** | bind **one PRODUCER + one CONSUMER**; a role-free toucher takes whichever side is open — **no flag** |
| ≥3 touchers, **or** ≥2 locked producers, **or** ≥2 locked consumers | **multi-binding** — genuine excess on a node | **set the DFB multi-binding advanced option** + record Quasar debt (below) |

The old "legal `(1, 1)`" case is the 2-toucher sub-row where the pair is one locked producer + one locked consumer — a genuine FIFO whose roles the ops already fix; it needs no special action. The **new** thing the 2-toucher row makes explicit: when the two touchers are sync-free (or one FIFO + one role-free), you *assign* 1P+1C rather than reaching for the flag. **Multi-binding is the last resort** — reach for it only when the census genuinely cannot fit 1P+1C (a third distinct toucher, or two kernels locked to the same FIFO role).

**Classify per instantiation, not once for the op.** A CB's endpoint count flips with config, so its disposition flips too: the same CB can be a single-ended scratchpad under one sharding and a legal `(1, 1)` FIFO under another (conv2d `ACT_TILIZED`: height-sharded → self-loop, block/width-sharded → legal), or single-endpoint under one config and multi-binding under another (split-reader / mcast). Classify each `(CB, config)` separately.

**Single-ended / sync-free → self-loop (no gate).** A CB touched by a **single** kernel — one real endpoint, or pointer-only access by that one kernel — is bridged by binding it as **both** producer and consumer (a self-loop). On Gen1 a DFB lowers to a plain circular buffer that a single RISC — compute *or* DM — can both fill and drain, so the self-loop is legal for both kernel types (the spec validator rejects a DM self-loop only on Gen2). The kernel code is untouched and runtime behavior is identical to the legacy CB. *(A DM self-loop is rejected only at the later Quasar-uplift stage — recorded for the Quasar audit, not a Gen1 blocker.)* **The self-loop is a one-toucher resolution.** If **two distinct** kernels touch the CB, it is *not* a self-loop and *not* (by default) a multi-binding: assign 1P+1C — bind one kernel PRODUCER, the other CONSUMER (see the [dual-instance work-split face](#multi-binding-2-of-one-kind-on-a-node) below, the dominant two-toucher shape).

#### Dead CB (0, 0)

**Recognition — and distrust a `(0, 0)` result.** A CB whose `buffer_index` is referenced by **no** kernel: no `reserve_back`/`push_back`, no `wait_front`/`pop_front`, no raw access. Grep the bound kernels for the index and any named CTA carrying it; zero hits. **But a dead CB should be *exceedingly rare*** — ops are well-optimized, and burning L1 on a CB no kernel touches is a flagrant waste you would not expect to find. So treat a `(0, 0)` result as **more likely a gap in your own analysis than a real dead CB**: before believing it, rule out every *indirect* path the index could take to a kernel — a CTA carrying it handed to a helper function, an index computed/offset/aliased from another value, a reference that appears only under a config you didn't inspect. Confirm across **all instantiations**, not just the default.

**Resolution — a dead CB *must* be dropped; the danger is dropping a *live* one.** A zero-endpoint CB **cannot be carried into Metal 2.0 at all**: a DFB with no producer and no consumer binding is rejected by the spec validator — it is structurally impossible to express a dead DFB. So a genuinely dead CB *must* be dropped, and doing so honors the port's **zero-functional-change** contract (a dead CB has no behavior, so removing its allocation changes L1 footprint and nothing else). The hazard is the mirror error — marking a CB dead when it is **actually live** (a reference hid behind indirection, or an unchecked config) and dropping it — which can **silently** mis-address or lose a real access → wrong numerics or a hang, the worst outcome this port can produce. And the safety net runs **one way only**: the validator structurally catches a *dead* DFB — a truly-dead CB you *fail* to flag simply resurfaces at port time as a bindingless DFB the porter can't build (loud, resolved then) — but nothing catches a wrongly-dropped *live* one. So bias hard toward caution: **report a CB dead only once you have positively confirmed its index is unreferenced by every kernel in every config; on any residual doubt, do not mark it dead — raise it as a question for the ops team.** (A real dead CB resurfaces at the validator regardless; a live one you drop does not.)

For a *confirmed* dead CB, frame it to the porter as *"a dead CB has no behavior, so removing it changes none"* — **not** as a sanctioned exception to "don't modify behavior" (that framing invites a porter to generalize the permission). The condition itself is deterministic (*no kernel references the `buffer_index`*); the care goes into *establishing* it (above). So a confirmed drop is a rule the porter executes, not a judgment call: **the porter drops the allocation and any dead CTA carrying its index during the port, and records each drop with `file:line` prominently in the report.** PORT WORK when confirmed; a question to the ops team on any doubt.

*Example:* conv2d `L1_ARRAY` — a 1-page "L1 scratchpad CB" whose index is threaded to the reader as a (dead) CTA, yet no kernel ever accesses it.

#### Multi-binding (≥2 of one kind on a node)

Metal 2.0 enforces single-producer / single-consumer per node (SPSC) as a spec-validator rule. When a node genuinely has **≥2 of one role that no relabelling can remove** — two kernels locked to the same FIFO role, or three-plus distinct touchers — Gen1 carries a **multi-binding advanced option** that permits the extra endpoint faithfully (the legacy co-fill or multi-read still happens, byte-for-byte). Setting it is **port work, not a gate**, and the flag *self-documents the Quasar debt* — a multi-binding DFB is exactly what Quasar uplift must refactor, so no separate tracking is needed.

**The flag is the *last* resort — most CBs with two touchers are 1P+1C, not multi-binding.** This is the trap the faces below set for a hurried reader: they surface CBs that more than one kernel touches, and it is tempting to translate "two touchers of a kind" straight into the flag. Don't. A *sync-free* (raw-pointer) touch is **role-free** — labellable PRODUCER or CONSUMER at no runtime cost on Gen1 — so **two** touchers, at least one relabellable, resolve to a plain **1P+1C** (bind one PRODUCER, one CONSUMER). The flag is forced only when the census genuinely cannot fit 1P+1C: **≥3 distinct touchers**, or **≥2 kernels locked to the same FIFO role** (two real `push_back`ers, or two real `pop_front`ers). Run the faces to build an accurate census; then apply the [classification table](#cb-endpoints) — **the disposition is a function of the census, not of which face matched.**

**The faces are a hunt for *touchers*, and the danger is missing one — a false negative.** Unlike a dead CB (where over-calling is the hazard), here the hazard is under-counting: the hidden second writer (face (a)) is invisible to a FIFO-sync trace, so miss it and the port silently drops a real co-fill. And even where the extra toucher resolves to a 1P+1C assignment rather than the flag, you must *find* it to bind it — in Metal 2.0 an unbound toucher is a kernel that cannot legally access the DFB. So a genuine multi-binding is uncommon, but multi-*toucher* shapes cluster: an op built on the dual-instance work-split (face (c)) co-touches *several* of its borrowed CBs — most resolve to 1P+1C, and only those pushed to ≥3 touchers or FIFO-doubling need the flag. Treat that clustering as a reason to run the hunt *carefully*, never to skip it. **Hunt for all three faces** — the porter's job is harder than the auditor's, and one face hides from a casual read:

**(a) Hidden second writer — hunt for it.** A CB presents as single-producer to a FIFO trace (one kernel `reserve_back`/`push_back`s), but a *second* kernel co-fills it via a **raw write** — `<cb>.get_write_ptr()` / `get_local_cb_interface(<cb>).fifo_wr_ptr` + offset, with **no** `reserve_back`/`push_back` — coordinated by dedicated semaphores (e.g. `reserve_done`/`write_done`) rather than CB FIFO sync. It is invisible to a FIFO-sync trace (the raw co-fill uses no FIFO ops), so **actively scan every kernel that touches each CB** for a `get_write_ptr()` / `fifo_wr_ptr` write by a kernel that is not the CB's FIFO producer, gated by a semaphore wait/post pair. (The coordinating semaphore is not itself a CB toucher — it does not enter the census; it ports as an ordinary `SemaphoreSpec`, and porting a *pre-existing* one is **not** the hand-rolled sync primitive the scope discipline forbids.) The raw co-filler is a genuine second toucher on the fill side; **whether it forces the flag depends on the full census.** The canonical case is genuinely multi-binding because a *third* kernel also touches the CB: *Example:* conv2d `ACT` under `split_reader_cb_shared` — the reader FIFO-produces, the writer co-fills `cb_act_second_obj.get_write_ptr()+offset` (CTAs 32/33), **and** compute consumes (`wait_front`/`pop_front`) — three distinct touchers → multi-binding. Were the FIFO producer and the raw co-filler the *only* two touchers (nothing downstream consumes), it would instead be 1P+1C — bind the role-free co-filler CONSUMER.

**(b) Multiple readers — the visible face.** A borrowed-memory, sync-free tensor-view CB read by base pointer whose read sites span **2+ co-resident kernels** (a split-reader's two DM readers, or a reader plus a writer acting as a second reader). *Recognition:* base-pointer read sites spanning 2+ co-resident kernels. **Count distinct kernels, not distinct accesses:** a kernel's raw-read is a peek on whatever binding it already holds, not a separate endpoint — a FIFO producer that *also* raw-reads its own buffer is **one** toucher (a PRODUCER binding covers the peek), so `push_back` + `get_read_ptr` alongside a second reader is **two** distinct touchers, not "1 producer + 2 consumers." *Disposition per the census:* **two** reader-kernels (both role-free) → **1P+1C**; **three or more** distinct touchers → multi-binding. *Shapes to census:* pool `raw_in` / `in_reader_indices` / `config_cb`; conv2d `ACT_SHARDED` / `READER_INDICES` in split-reader / mcast; halo `src_cb` ROW_MAJOR.

**(c) Dual-instance work-split — usually 1P+1C, *not* multi-binding.** One kernel source instantiated **twice** over the same core range (a Reader-config + Writer-config pair on BRISC/NCRISC, to double NoC bandwidth), the two instances splitting the work by disjoint offset/row ranges. Each borrowed CB **both** instances touch has two co-resident touchers — the co-*written* output, the co-*read* inputs — and these co-fills/co-reads are **sync-free** (raw `get_write_ptr`/`get_read_ptr` + offset; two kernels can't share one FIFO write-pointer to write disjoint regions, so this shape is raw by construction). Two role-free touchers → **1P+1C**: bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1, no runtime effect). **Reach for the flag only when a *third* distinct kernel also touches the CB** — e.g. a compute kernel consuming a co-produced intermediate in a fused variant — pushing the census to ≥3. So a pure-data-movement factory built this way has **several** co-touched CBs but typically **zero** that need the flag; a fused variant can have one where compute is the third toucher. Unlike (a) it is fully **visible** — no semaphore, no hidden co-fill; unlike (b) the excess falls on the *producer* side. *Recognition:* the factory pushes the **same `kernel_source`** into two `KernelDescriptor`s that differ only by `ReaderConfigDescriptor` / `WriterConfigDescriptor` and the per-instance work-split args, both over one `core_ranges`. **This is the dominant two-toucher shape — do not confuse it with the [demoting-per-group-CTA anti-pattern](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta), where two same-source KernelSpecs cover *disjoint* node sets** (each node sees one instance → ordinary 1:1, no assignment question). Here both instances hit *every* node, so each node genuinely has two touchers to assign. *Verified example:* **reshard** generic factory — one output CB, both same-source instances raw-write disjoint page ranges via `dfb.get_write_ptr() + offset`, output resident (nothing drains) → **1P+1C** (this is the case a porter mis-slotted as multi-binding). *Shapes to census:* concat `S2SRM` / `S2SMulti` / `BlockSharded` (pure DM → expect 1P+1C); concat `S2S`-tiled (has compute → a co-touched intermediate may hit ≥3 touchers).

**Config-dependence.** Record **which disposition applies per config** — the census flips with config, so a CB can be single-ended (self-loop) under one and a two-toucher (1P+1C) or genuine ≥3-toucher (flag) under another (per *Classify per instantiation*, above). Note per `(CB, config)`.

**Finding role: PORT WORK** — self-loop the one-toucher (single-ended/sync-free) CBs, assign **1P+1C** the two-toucher CBs (the dual-instance work-split above is the common case), set the multi-binding advanced option only where the census can't fit 1P+1C (≥3 distinct touchers, or ≥2 kernels locked to the same FIFO role), drop the *confirmed*-dead CBs (with `file:line`; on any doubt, raise a question rather than dropping). Surfaced to the porter (an **FYI-P** heads-up for the multi-toucher shapes to watch). Nothing here gates. **Because nothing here gates, the whole subject is porter-only — so on a whole-op RED with no portable subset, skip it** (per the **Red** outcome scoping rule in [Feasibility audit](#feasibility-audit)): a full per-`(CB, config)`-per-node census is unread detail for a port that cannot start, and re-audit would redo it against changed code. This is the acute case on mega-ops — dozens of CBs across many factories — where the census is most expensive and least likely to survive to port time.

**Op-level roll-up:** `✓ legal`, else the per-CB dispositions — `self-loop`, `1P+1C` (assign roles across two touchers), `multi-binding flag` (with configs), `dead-CB drop` — with the per-`(CB, config)` inventory in the report.

### Out-of-directory coupling

**The findings here are informational (FYI).** The one place donor coupling *gates* the port — a donor kernel still on pre-Device-2.0 idioms — is judged and named in [the Device 2.0 prerequisite](#device-20-prerequisite) (which always runs); this subject inventories the coupling in full so that gate is well-scoped and any future multi-op coordination is visible. Because that full inventory is porter/planning detail, it follows the **Red** outcome scoping rule: run it when a port or a clean subset is ahead; **skip it on a whole-op RED with no portable subset** — the Device 2.0 gate has already named any donor blocker, so nothing gate-relevant is lost by deferring the inventory to re-audit. This is a substantial subject; budget time accordingly. It covers **two distinct escape types**: *function-call escape* (the kernel `#include`s and calls another op's helper function) and *file-path kernel instantiation* (the program factory `CreateKernel`s a kernel `.cpp` owned by another op). The bulk of the machinery addresses function-call escape; file-path escape is surfaced separately at the end as a coupling signal.

**Why this check exists.** Op kernels frequently `#include` headers outside their own directory. When a Metal 2.0 port crosses one of these boundaries, the kernel's named tokens (`dfb::name`, `sem::name`, `tensor::name`) need to translate into whatever shape the donor's signature expects. Some shapes cross cleanly; others require donor-side conversion work, most commonly because **the donor itself isn't on Device 2.0 yet**.

The Device 2.0 → Metal 2.0 sequencing rule applies: ops must complete Device 2.0 migration before Metal 2.0 can proceed. A donor consumed by this op that is still on pre-Device-2.0 idioms (`InterleavedAddrGen`, `ShardedAddrGen`, raw sem addresses, `CircularBuffer&`) blocks this op's Metal 2.0 port until the donor migrates — that **GATE judgment is recorded in [the Device 2.0 prerequisite](#device-20-prerequisite)**; here, inventory the donor shape so the gate is specific and schedulable.

**Inventory phase.** For each kernel file in the op, list every `#include` whose resolved path lies outside the op's own directory. Identify the donor class:

1. **`tt_metal/*`** — LLK / HAL / firmware. No concern.
2. **`ttnn/cpp/ttnn/kernel_lib/`** — official shared kernel library; lib team handles internally.
3. **`ttnn/cpp/ttnn/kernel/`** (singular) — a second shared-kernel pool. Treat as shared-lib class.
4. **`ttnn/cpp/ttnn/operations/kernel_helper_functions/`** — small shared utility pool.
5. **In-family shared** — kernels within the same op family. In-family escapes don't *gate* the Metal 2.0 port; you port the family together. (This concerns the Metal 2.0 *syntax* rewrite, not Device 2.0 — in-family kernels remain fully subject to the [Device 2.0 gate](#device-20-prerequisite), which is location-independent.)
6. **Cross-family donor** — kernels in another op family's directory.

**Per-call shape analysis.** For each donor file consumed by the op, identify which public functions the op's kernels actually call, and classify each by the shape of the resource handles in its signature.

| Shape | Status | Notes |
|---|---|---|
| `Semaphore` / `Semaphore&` / `const Semaphore&` | ✓ excellent | Device 2.0 native. |
| `uint32_t sem_id` | ⚠ not workable today | `sem::name` has **no** implicit `uint32_t` conversion (only `dfb::name` does), so there is no bridge today; a constexpr `sem_id` cast could add one later. A call site outside the op that needs it is an [assumption-violation stop](../port/metal2_port.md#kernel-side-whitelist) for the porter, not a fix. |
| `uint32_t sem_addr` (L1) or `uint64_t` NOC-encoded sem | ✗ not OK | No clean Metal-2.0 → donor bridge today. A backdoor could be added. |
| `TensorAccessor<DSpec>` / ref (Shape 1) | ✓ excellent | Porter constructs `TensorAccessor(tensor::name)` and passes. |
| `TensorAccessorArgs<N>` (Shape 2) | ✗ not OK | Porter can pass `tensor::name.args`. Workable, but suboptimal. |
| Tensor CTA offset as NTTP (Shape 3) | ✗ not OK | No workaround today; would require a one-line `tensor::name::cta_offset` add to `TensorAccessorBindingToken`. |
| Old-style addr-gen — `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*` (Shape 4) | ⭐ ✗ very not OK | Donor is pre-Device-2.0 — the donor-side Device 2.0 **GATE** (recorded in [the Device 2.0 prerequisite](#device-20-prerequisite); the op's *own* kernels are gated there too). Inventory the donor file + kernel here so the gate is specific. |
| `uint32_t cb_id` | ✓ OK | `dfb::name`'s constexpr cast handles runtime AND template-parameter position. |
| `CircularBuffer` / `CircularBuffer&` / `const CircularBuffer&` | ⭐ ⚠ flag | Op-by-op porting + DFB-replaces-CB on the consumer side leaves no clean per-op story today. Flag for cross-team discussion. |

One donor file can have multiple functions with different shapes — classify per function.

**Report format.** Three-part structure:

1. **Op-level roll-up** — one-line headline status (`✓ clean` / `⚠ workable` / `⭐ blocked`), plus a tight bullet inventory of the kinds of issues found across all donors.
2. **Summary table** — one row per (op kernel, donor file) pair across all buckets.
3. **Per-call detail** — per-function breakdown for donors with ⚠ / ✗ / ⭐ entries. Omitted entirely if all rolls are ✓.

Status roll-up uses ✓ / ⚠ / ✗ / ⭐. The star is reserved for entries that create scheduling blockers — Shape 4 (donor pre-Device-2.0, the donor-side Device 2.0 gate per [the Device 2.0 prerequisite](#device-20-prerequisite)) and `CircularBuffer&` (op-by-op friction). Other ✗/⚠ items are workable today or need donor work, but don't sequence-block.

**Borrowed kernel files (file-path kernel instantiation).** Separate from the function-call escapes inventoried above: list every kernel `.cpp` file the op's program factory instantiates whose source it does **not** own — anything from a shared pool, in-family or cross-family. (Some ops own *none* of their kernels and instantiate every one from a shared pool — list them all.) For each, record:

- The kernel file's path.
- The owning op family (or shared pool).
- Whether the file is also instantiated by other ops (broadly-shared) or is a one-off borrow — and, where cheap to determine, *which* other ops.

This signal **does not gate the port**, but it induces a **port-the-family-together coupling that must be reported**. A shared kernel's Metal 2.0 rewrite (CB→DFB, named-token bindings, etc.) is a *single* rewrite: every op that instantiates that kernel must adopt it in the same change, or the co-borrowers break the instant one op migrates in isolation. So the set of ops sharing a kernel forms a Metal 2.0 **port-together set** — report that set (or as much of it as is cheap to find) so planners can sequence the shared rewrite as one unit. Surface this even when the function-call escape roll-up is `✓ clean` — file-path coupling is independent. (This is distinct from the [Device 2.0 gate](#device-20-prerequisite), which applies to every one of these kernels regardless of coupling.)

### RTA varargs

**FYI-P.** Legacy runtime args (RTAs) and common runtime args (CRTAs) are **positional** — read by index. The Metal 2.0 port converts them to **named** arguments, inferring each name from the kernel code (the variable a `get_arg_val` unpacks into: `uint32_t num_tiles = get_arg_val<uint32_t>(17)` → the binding `num_tiles`). Named args are **strongly preferred**; because legacy args are all positional, a lazy port could smuggle *everything* in as varargs, which we don't want. This subject flags the specific args that genuinely **can't** be named and so must be ported as varargs — sparing the porter that call.

**Recognition:** a kernel pulls RTAs or CRTAs in a **counted loop, indexed by the loop variable** — `for (int i = 0; i < N; ++i) { get_arg_val<uint32_t>(i); ... }`, or the common-arg form `get_common_arg_val<uint32_t>(i)`. Read positionally through a loop index (rather than as distinct `get_arg_val<T>(<constant>)` reads), there are no per-argument names to infer, so the block must become varargs. **This holds even when `N` is fixed at compile time** — the usual case: the kernel still consumes a variable-length indexed block that can't be expressed as named args.

**Non-signal — the common, preferred case:** a kernel that reads each arg with a *distinct constant* index (`get_arg_val<uint32_t>(17)`, `...(18)`, …) → the porter names each. Ordinary port work, not this subject; don't flag it.

Metal 2.0 **supports** both RTA and CRTA varargs via the kernel-side vararg mechanism, so this does **not** gate — it's a porter heads-up. The value is the auditor doing, unhurried, a classification the porter would otherwise have to: report the kernel and the recognition site (`file:line`) and note it's a genuine loop-indexed varargs case (RTA or CRTA), so the porter reaches for the vararg mechanism rather than trying to name each — per the recipe's [kernel-side whitelist rule 4](../port/metal2_port.md#kernel-side-whitelist).

**Not to be confused with CTA varargs**, which **do** gate the port (caught by the [CTA varargs Appendix A entry](#variable-count-compile-time-arguments-cta-varargs--unsupported)): kernels that loop over `get_compile_time_arg_val(i)` with a **runtime-varying count**, or ops whose `tensor_args_t` carries a variable-count container like `std::vector<Tensor>`. The distinction that matters: a *runtime-varying count* is what makes **CTA** varargs unsupportable, whereas an **RTA/CRTA** loop needs varargs regardless of whether its count is fixed.

### Incidental anomalies

**FYI-U — a standing, opportunistic instruction, not a scan.** While working the other subjects you will sometimes notice latent issues that are neither audit findings nor the porter's to fix — a dead/unused RTA, an attribute the factory forces or ignores yet still feeds to `compute_program_hash`, a suspicious hardcoded constant. Record these in the report's **Misc anomalies** section: team-only, non-gating, *not* porter-actionable (they route to the ops team, never into the port diff). **Don't go hunting** — just note what you happen to see. The auditor is the one agent that reads every line of the op's code, so this cheap capture surfaces latent bugs nobody else would catch.

### Output: the two documents

The audit emits up to **two** files, by audience. Write them to the **op's root directory** — one level above `device/`, alongside the op's host-facing `.cpp` / `.hpp` files (e.g. `ttnn/cpp/ttnn/operations/<family>/<op>/`), **not** inside the `device/` subdir even though the program-factory `.cpp` files live there. They sit next to `METAL2_PORT_PLAN.md` and `METAL2_PORT_REPORT.md` (written later by the port recipe), so all generated docs for the port land in one spot.

- **`METAL2_PREPORT_AUDIT.md`** — **team-facing, always emitted.** The complete record — consumed downstream to track this op's porting readiness. Every finding lands here, regardless of role.
- **`METAL2_PORT_BRIEF.md`** — **porter-facing, emitted only on a fully GREEN audit** (every gate cleared). The porter's actionable input, ordered by the porter's *workflow*: plan, then construct, then watch-for. On any **RED** there is no brief — there is no port yet.

Findings route to these documents by role (per the [finding-roles routing table](#feasibility-audit) at the top of the audit): the **brief** carries GATE-cleared one-liners + all PORT WORK + all FYI-P; the **team doc** carries everything, including FYI-U.

**Provenance line.** Both files open with a one-line record of the recipe-doc version this audit ran against, so a reviewer can pin the exact guidance. Generate it from the checkout root and paste it verbatim:

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

If the command prints nothing, the docs aren't from a tracked doc-branch checkout — record that instead, since the version can't be pinned.

**Cross-references inside the generated files name the doc plainly** (e.g. `ttnn_factory.md`) rather than using a markdown link. These files live in the *op directory*, where a doc-relative link (`](../shared/ttnn_factory.md)`) doesn't resolve, and op directories sit at varying depths — so a hardcoded relative path is fragile. The porter has the repo open; a plain doc name is enough.

**In chat, surface only the Result line plus the file path(s)** so the user can open the files when ready. Do not paste a full report inline — an audit of any non-trivial op runs to dozens or hundreds of lines, and chat-scrollback isn't the right home for it. Markdown formatting in the files is required, not optional: the headers, tables, and inline-`code` spans are what make a sizeable report skim-friendly.

**Reassuring framing for the human reader.** A RED gates *this specific port attempt* but is rarely a permanent blocker — every gate has a path forward, and surfacing it is part of the report. A RED traces to one of a few sources, each routed to whoever clears it: an **op-readiness prerequisite** (Device 2.0 migration, an offset-base-pointer refactor, a 3rd-arg page-size fix — the ops / Device-2.0 teams); **missing TTNN infrastructure** (the factory-concept gate — lifts when the op's `ProgramDescriptor` migration lands, or when the TTNN infra for custom-hash / runtime-args-update ships); a **porting-recipe TODO** (a case the recipe doesn't yet handle mechanically); or, least often, a **missing Metal 2.0 feature** — the small, enumerated set in Appendix A (**Metal 2.0 is nearly feature-complete; most ops touch none of these**). Most such entries land when implemented; `address_offset` alone needs a redesign + runtime-team consult. Surface the future path explicitly for every RED, so a colleague reads the path forward, not just the gate. In particular, a **TTNN-factory-concept RED is the expected outcome** for any op still on the legacy imperative API (its concept isn't `descriptor`) — not an alarm — and the port unblocks once that op's `ProgramDescriptor` migration lands.

**Code-path scope.** Blockers are often confined to specific code paths (e.g. a single factory's `if (use_width_sharding)` branch). When so, identify clean vs. blocked paths explicitly and offer a scoped-subset port — "interleaved-only paths, omitting the sharded path." A partial port that delivers value now may beat waiting for the full gate to clear; reflect it in the Result (`RED at op level; subset <X> is clear`). **If no clean path exists — the blocking shape is unconditional/structural, not one branch among siblings — say so explicitly (`RED at op level; no portable subset`) rather than leaving it to be inferred from silence.**

#### `METAL2_PREPORT_AUDIT.md` — team-facing (always emitted)

Opens with a **status summary** grouped Prereqs / Feature Support / TTNN Readiness / Port work — a compact, skim-first snapshot a downstream reader consumes to track this op's porting readiness. The detail sections follow. (Extend any cell, row, or paragraph with multi-line context where it improves clarity.)

````markdown
# Metal 2.0 Audit Findings — `<op path>`

<Identifying section: device operations sharing the directory, factory file list, anything needed to disambiguate. For multi-device-op directories, nest — outer bullets for device-operations, inner for their program factories. Example shape:

- **`ReduceDeviceOperation`**
  - `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)
  - `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)
- **`WelfordReduceDeviceOperation`**
  - `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)
>

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `<hash> <date> <subject>`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `<path>` |
| **Overall** | GREEN / RED |
| **DOps / Factories** | `<DeviceOperation>` → `<factory list>` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes / No (RED — routed to the Device 2.0 track; note isolated-holdover vs broad) |
| *Prereqs* — Cross-op escapes | Ok / issue |
| *Feature Support* — overall | GREEN / RED |
| *Feature Support* — Variadic-CTA | Ok / Unsupported |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes / No: `<failing conjunct>` |
| *TTNN Readiness* — Concept (current) | `descriptor` / `WorkloadDescriptor` / `legacy device-op` / `MetalV2` (already ported) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A / Yes / No (genuine multi-program → gate) |
| *TTNN Readiness* — Is safe to port? | Yes / warning: `<site>` / No (→ readiness-sheet owner) |
| *TTNN Readiness* — Custom hash | No / Yes (gate) |
| *TTNN Readiness* — Runtime-args update | No / Yes (gate): `<factory + site>` |
| *TTNN Readiness* — Pybind `create_descriptor` | No / Yes (gate): `<nanobind site>` |
| *TTNN Readiness* — Op-owned tensors | No / Yes: `<factory + site>` |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (+ op-owned tensors, if any) |
| *Port work* — Offset base pointer | none / **GATE** → ops team (Type 1 raw · Type 2 accessor-fed, flag early) |
| *Port work* — Tensor bindings (per binding) | clean / Case 1 / Case 2 |
| *Port work* — TensorParameter relaxation | none / `<relaxation>` |
| *Port work* — TensorAccessor 3rd arg | drop (Class 1/2) / **flag → GATE** (Class 3/4/Special) |
| *Port work* — CB endpoints | legal / self-loop / 1P+1C / multi-binding flag / dead-CB drop |

**CB endpoints** are dispositions, not gates (see [CB endpoints](#cb-endpoints)): every out-of-window CB has a port-time resolution — a **self-loop** (one toucher: single-ended / sync-free), a **1P+1C assignment** (two touchers, e.g. a dual-instance work-split), the **multi-binding advanced-option flag** (genuine ≥2 of a role on a node the census can't relabel — ≥3 touchers or FIFO-doubling), or a **dead-CB drop** (zero endpoints). Record the disposition per `(CB, config)`, and classify per instantiation — the same CB's disposition often flips with config.

## Result

**GREEN → brief issued** · **RED → blocked on `<gate>`**, routed to the owning team (`Device 2.0` / `TTNN-or-PD-migration` / feature / `ops` / `readiness-sheet owner`). State the primary blocker(s) in plain language; if localized, name a clean subset (`RED at op level; subset <X> is clear`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** <GREEN — the sheet's `Is able to port?` == `yes`, cross-check clean — or — RED: name the failing conjunct and route it. A **shape** failure (`Concept` non-`descriptor` / `Custom hash` / `Runtime-args update` / `Pybind`) → the TTNN / ProgramDescriptor-migration team (the gate lifts when support lands; for a `legacy device-op` concept, add the "separate ongoing effort; expected outcome for legacy ops; unblocks when the `ProgramDescriptor` migration lands" framing). A **`safe`** failure → the readiness-sheet owner. A **spreadsheet-broken / missing-op** conflict → the readiness-sheet owner to reconcile.>
- **Device 2.0 (every kernel used):** <GREEN — or — RED with exact violations @ `file:line`, routed to the Device 2.0 team (note whether the incompleteness is isolated CB-index holdovers — 1-line mechanical, idioms intact — or broad Device 1.0 requiring a full migration, so the team can size it; table below). Name the kernel file and, for a borrowed/donor kernel, its owning family.>

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `<path>` | `<n>` | `<call>` | `<wrapper>` |

- **Feature compatibility:** every Appendix A entry, in order. Every entry is UNSUPPORTED, so per-row status is `N/A` when the feature is absent (not a vacuous GREEN) or `RED` when it is in use — there is no `GREEN` row. UNSUPPORTED hits get an H4 detail block with signal, `file:line` sites, and expected resolution. For `address_offset`, surface the runtime-team-consultation message verbatim per the entry's Action field.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | RED / N/A | config-scoped — RED the offending factory, name the clean subset |
  | CBDescriptor `address_offset` (non-zero) | RED / N/A | runtime-team consultation (verbatim message) |
  | GlobalSemaphore | RED / N/A | |
  | Variable-count compile-time arguments (CTA varargs) | RED / N/A | |

- **CB endpoints (GATE-free):** <every CB is either legal (1 producer, 1 consumer) or carries a port-time disposition — **self-loop** (one toucher: single-ended / sync-free, legal on Gen1 for compute *and* DM), **1P+1C assignment** (two touchers, e.g. a dual-instance work-split — bind one PRODUCER, one CONSUMER, no flag) with its `(CB, config)` list, **multi-binding advanced-option flag** (genuine ≥2 of a role the census can't relabel — ≥3 touchers or FIFO-doubling; hidden-2nd-writer or multi-reader face) with its `(CB, config)` list, or **dead-CB drop** (zero endpoints) @ `file:line`. Nothing here blocks a Gen1 port. **Deferred** (re-evaluate post–Device 2.0) if the Device 2.0 gate is RED.>
- **Offset base pointers:** <GREEN — no address RTA folds a host-side offset into its base, or every fold has already been split out by the ops team (→ clean base, handled as ordinary tensor-binding port work) — or — RED: Type 1 (raw offset arg) / Type 2 (accessor-fed offset arg → flag early, ops team + framework) @ `file:line` by kernel + role. Factory-scoped: an RM slice-family RED names the tiled factories as a clean subset. Type 3 (`address_offset`) is the Appendix A entry; Type 4 (`narrow`) ports as-is. Cross-references the offset-base-pointer triage analysis (a dated prior).>
- **TensorAccessor 3rd argument:** <GREEN — every site Class 1/2 (redundant → drop; Class 1 also sets `dynamic_tensor_shape`) — or — RED: Class 3/4 (wrong-magnitude page size @ `file:line` → the ops team fixes it first) / Special (an override the binding model can't express @ `file:line` → the ops team ports by hand). Cross-references the 3rd-arg triage analysis (a dated prior).>

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `<name>` Case 1 (`TensorAccessor`) / Case 2 (raw pointer → `get_bank_base_address` bridge) / clean (borrowed-DFB).
- **TensorParameter relaxation:** `<relaxation>` on `<binding>` (confirm the custom hash matches) | none.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg @ `<sites>` (Class 1 also sets `dynamic_tensor_shape`) | none.
- **CB endpoints:** self-loop `<CB, config>` / 1P+1C assign `<CB, config>` / multi-binding advanced-option flag `<CB, config>` / dead-CB drop `<CB @ file:line>` | all legal.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** each multi-binding CB the porter flags — hidden-2nd-writer (a raw co-fill, semaphore-gated) or multi-reader — with `(CB, config)` @ `file:line`. (Self-loop and dead-CB dispositions live in Port-work.)
- **Cross-op / shared kernels:** borrowed kernel files + shared-kernel coupling.
- **RTA varargs:** kernel + recognition site.

## Team-only

- **Out-of-directory coupling & donor shape:** the full by-shape inventory (op-level roll-up, summary table, per-call detail, borrowed kernel files).
- **Relaxation candidates** (mined from a custom hash on a gated op): **FALLIBLE — candidates to verify**, default strict.
- **TTNN factory analysis:** the sheet-derived facts, with `file:line` evidence — op-owned tensors, MeshWorkload need (genuine vs. op-owned-tensor artifact), pybind `create_descriptor`, other risky pybind, custom hash, custom `override_runtime_arguments`. Several are **gate conjuncts** (custom hash, pybind `create_descriptor`, custom `override_runtime_arguments`, genuine multi-program) recorded in the [TTNN factory concept prerequisite](#ttnn-factory-concept-prerequisite); op-owned tensors and the target concept are the non-gating facts that inform the port's TTNN ProgramFactory wiring.

## Misc anomalies  *(omit if none; team-only, non-gating)*

<Latent code issues noticed while auditing that are neither audit gates nor porter work — dead/unused RTAs, attributes forced or ignored in the factory yet still fed to `compute_program_hash`, suspicious hardcoded constants, and the like. One bullet each with `file:line`. These route to the ops team; the port does not act on them.>

## Per-DeviceOperation attribution  *(when bundled)*

<One status-summary row per DeviceOperation when the directory bundles more than one and findings differ.>

## Questions for the user  *(omit if none)*

1. **<short title>:** <question, with the `file:line` context that prompted it>

## Recipe notes  *(omit if none)*

<Friction with *this audit recipe itself*, not findings about the op — a step that was unclear or contradictory, a recognition rule that false-fired (name the guard that should cover it), a case the recipe didn't anticipate, a RED/GREEN boundary that forced an unacknowledged judgment call. Be concrete: cite the section, quote the line. The recipe maintainer reads these.>
````

#### `METAL2_PORT_BRIEF.md` — porter-facing (emitted only on an all-GREEN audit)

Ordered by the porter's workflow: plan → construct → watch-for. Issued only on a fully GREEN audit — every gate cleared. Never on RED.

````markdown
# Metal 2.0 Port Brief — `<op path>`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `<hash> <date> <subject>` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** <`descriptor` | `WorkloadDescriptor` (secretly SPMD — collapses to single-program)>
- **Op-owned tensors:** <none | `<factory + site>` — carried natively by the target concept>
- **Target concept:** `MetalV2FactoryConcept`<, with op-owned tensors, if any>
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all gate conjuncts — plus **other migration-risky pybind**, which surfaces as a `safe` warning that also fails the gate. All `no` on a cleared op.

## Construct — to do

**Tensor bindings** (per binding):

- `<name>` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::name)`.
- `<name>` — **Case 2** (raw pointer) → bind the tensor, pull the base via `get_bank_base_address`, raw walk unchanged.

**TensorParameter relaxation:** <`<relaxation>` on `<binding>` — confirm the custom hash excludes exactly the relaxed property | none>

**TensorAccessor 3rd arg:** <drop the redundant page-size arg @ `<sites>`; Class 1 sites also set `dynamic_tensor_shape` | none>

**CB endpoints:** <self-loop `<CB, config>` (one toucher: single-ended / sync-free) · assign 1P+1C on `<CB, config>` (two touchers, e.g. dual-instance work-split) · set the multi-binding advanced option on `<CB, config>` (genuine ≥3-toucher / FIFO-doubling only) · drop dead CB `<CB @ file:line>` (and any dead CTA carrying its index) | all legal>

## Watch for

- **CB endpoints (multi-binding):** <each multi-binding CB → hunt the hidden 2nd writer (a raw co-fill, semaphore-gated) before setting the flag; `(CB, config)` | none>
- **Cross-op / shared kernels:** <path → caution per [pattern]; port the shared kernel as one unit | none>
- **RTA varargs:** <kernel → prefer named RTAs | none>
````

#### Per-row feature status: N/A vs. RED

Every Appendix A entry is UNSUPPORTED, so a per-row status is one of two — there is no `GREEN` row:

- **`N/A`** — the feature is absent from the op (the op uses no `GlobalSemaphore`, so that entry cannot fire; it has no variable-count CTAs, so the CTA-varargs entry cannot fire; etc.). Since every entry is a gate-feature, an absent one is `N/A` — the gate didn't fire because the feature is *absent*, so there is nothing to "pass."
- **`RED`** — an UNSUPPORTED feature is present → the port is gated.

A clean feature scan is therefore all-`N/A`. (The *subject's* overall roll-up may still read "GREEN — no gate fired"; that's the subject verdict, distinct from these per-row labels.)

For UNSUPPORTED feature-detail blocks, the **Expected resolution** is usually a short paraphrase of the entry's Status field — e.g. "not yet supported in Metal 2.0; port will be possible once GlobalCircularBuffer support lands on `KernelSpec` / `DataflowBufferSpec`." For the `address_offset` entry specifically, surface the runtime-team-consultation message verbatim per the entry's Action field.

#### RED short-circuit: the op is still on the legacy imperative API

When the [TTNN factory concept gate](#ttnn-factory-concept-prerequisite) fails because the op isn't on the `ProgramDescriptor` API yet (a `legacy device-op` concept), it is a **whole-op RED with no portable subset**, so the **Red** outcome scoping rule (top of [Feasibility audit](#feasibility-audit)) governs — apply it, don't improvise. **Run every gate-bearing subject in full** (Device 2.0, Feature compatibility, Offset base pointers, TensorAccessor 3rd argument): the user needs to see *all* the blockers they will have to clear alongside the ProgramDescriptor migration, not just the concept gate, so they can be worked in parallel. **Skip the seven purely-informational subjects** — the re-audit produces them fresh (and correctly) once the migration lands and the code has settled; producing them now against soon-to-change legacy code is unread, stale effort. Record each skip with the one-line disclosure so the omission never reads as clean. The Result and Gate detail emphasize the concept gate as the primary blocker. (This is the legacy-API case specifically. A concept-gate failure on a *`descriptor`* op — e.g. a custom hash — often leaves a portable subset or is nearly portable, so it is *not* automatically no-subset: apply the scoping rule to that op's actual subset scope, which usually means auditing it in full.)

Save the file(s) and surface the path(s) with the Result line. **Stop here.** The audit file(s) are the complete deliverable of this document.

### After the audit: what happens next

- **On RED**: this op cannot be ported in its current state. Surface the `METAL2_PREPORT_AUDIT.md` path and Result; stop. No brief is written, and the recipe is not loaded.
- **On GREEN**: both files are written (the team doc and the brief) and you STOP — that is the audit deliverable. **Then, only on explicit user go-ahead**, load [`port/metal2_port.md`](../port/metal2_port.md) to perform the port, passing the audit files as context — the recipe needs the cleared gates and decisions, *including the TTNN factory analysis*. Do not load the recipe on your own initiative; the user must explicitly approve.

---

## Appendix A: Metal 2.0 feature compatibility

This appendix lists legacy-API features that **Metal 2.0 does not yet support** — every entry is **UNSUPPORTED**. If the op uses one, refuse the port and report (a GATE / RED). Each entry's **Status** field describes the future path: most features will be supported as-is when implemented; a few are addressable only via a redesigned, semantically different construct (and may need a runtime-team consultation before re-attempting) — so always check the Status field before telling the user "wait and revisit." (Legacy features Metal 2.0 *does* support are translated mechanically by the porting recipe and are not listed here.)

### Maintenance: keeping Appendix A current

Appendix A is actively maintained as Metal 2.0's feature surface evolves. When a feature listed here gains Metal 2.0 support, the doc maintainer **removes its entry** — it becomes a mechanical porting-recipe translation, not an audit gate — rather than reclassifying it in place.

**For porting AIs: treat the current list as definitive.** If an entry looks stale to you — the API seems to have landed — do **not** override it or reverse-engineer support from the framework headers; report per the list as written. Metal 2.0 is mature and its supported/unsupported surface is settled; keeping Appendix A current is the maintainer's job, and any genuine lag is fixed there (a re-run then clears it), never worked around in your audit.

When scanning during the [Feature compatibility](#feature-compatibility) subject, match each feature's recognition signals against the op's source. If any signal matches, take the action declared in the entry.

> **For maintainers adding new entries — skim if you're applying the recipe, not editing it.** Features whose underlying *functionality* Metal 2.0 will *never* support are handled differently: they are either reclassified as a prereq fix (the legacy use is replaced before porting) or get a dedicated fix-up recipe in the port recipe. They do *not* live here, because the action for them is not "wait" or "ask" — it is "transform." (Features whose *current API form* will not be supported but whose *underlying functionality* will be — via a different construct — do belong here as UNSUPPORTED entries; the entry's Status field calls out the redesign requirement.) If you are about to add an entry and the underlying functionality has no planned support, route it to the prereq-fix path or to a port-recipe fix-up entry instead — not here.

Each entry follows this uniform format:
- **Status** — the (UNSUPPORTED) support state and the future path.
- **Recognition — definitely this feature** — signals that, if matched, mean the feature is in use. Trigger the entry's action.
- **Recognition — false-positive guard** — superficially similar constructs that are *not* this feature. Do not trigger the action on these.
- **Action** — what to do when the rule fires (refuse the port and report).
- **Examples in the wild** — real op locations using this feature, for ground-truthing your match.

This list is **authoritative**: if a construct isn't here, it's supported — don't reverse-engineer its status from the API surface.

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

### CBDescriptor `address_offset` set to non-zero — UNSUPPORTED (current form not planned)

**Status**: Not supported in Metal 2.0, and **not slated for support in its current form**. The `address_offset` field on `CBDescriptor` is an interim mechanism for placing a CB at a non-zero offset within a `Buffer` (or at an absolute address, when used without a `buffer`). Metal 2.0 will not carry this construct forward as-is. The functional capability the field provides will be available in a **semantically different way** in Metal 2.0; a direct translation is not possible. The user must consult the runtime team about their actual use case before this op can be ported.

This offset is most often used with a Buffer-backed CB (the legacy borrowed-memory pattern — now a mechanical porting-recipe translation, no longer an Appendix A entry), but `address_offset` gets its own entry because:
- It is severable: a CB can use a non-zero `address_offset` without a Buffer-backed CB (placing the CB at an absolute L1 address — the "manually placed CB" mode documented in `ttnn/api/ttnn/tensor/tensor_utils.hpp`).
- It carries a stronger signal than dynamic CB alone — the form is being phased out, not just "not yet implemented."

**Recognition — definitely this feature** (refuse and report; flag prominently):

- A `CBDescriptor` literal or struct assignment with `.address_offset` set to a non-zero literal or any expression that is not statically `0`.
- `CircularBufferConfig::set_address_offset(non_zero)` (imperative API; an op using it directly is still on the legacy imperative API, so it also fails the TTNN factory concept gate — but record matches here too, including any leaking in via shared utility code).
- `UpdateDynamicCircularBufferAddress(program, cb_handle, buffer, offset)` — the four-argument overload — when `offset` is non-zero.
- Calls to helpers like `cb_descriptor_from_sharded_tensor(cb_index, tensor, address_offset, ...)` in `ttnn/api/ttnn/tensor/tensor_utils.hpp` where the third argument (`address_offset`) is passed a non-zero value.

**Recognition — false-positive guard**:

- `.address_offset = 0` or `.address_offset` not set (default zero) is fine for *this* rule → green.
- `UpdateDynamicCircularBufferAddress(program, cb_handle, buffer)` — three-argument form with no offset → not this rule.
- Kernel-side `bank_address_offset` parameters on calls like `get_noc_addr_from_bank_id<...>(bank_id, bank_address_offset)` are an unrelated kernel-side feature → not this rule.

**Action**: STOP. **Flag prominently in the audit report** — do not bury this among other RED entries. Use stronger emphasis than for routine UNSUPPORTED items, and surface the following message to the user **verbatim**:

> The `address_offset` field is an interim mechanism that will not survive into Metal 2.0 in its current form. The underlying capability you need will be available, but only via a different API that is not a direct translation. Please reach out to the runtime team to discuss your use case before proceeding with this port.

Recommended report shape when this rule fires:

```
*** CRITICAL: CBDescriptor address_offset feature in use ***
File: <file:line>
[verbatim message above]
```

Do not invent a workaround. Do not propose an alternative implementation. Do not attempt a partial port. The correct path is a runtime-team consultation, then revisit.

**Examples in the wild** (for ground-truthing your match):

`address_offset` has limited adoption in checked-in code today. Likely paths to a non-zero usage:
- Direct `CBDescriptor` literal with `.address_offset = <non-zero>`.
- Helper `cb_descriptor_from_sharded_tensor(cb_index, tensor, address_offset, ...)` in `ttnn/api/ttnn/tensor/tensor_utils.hpp` — inspect callers for non-zero third arguments.
- Python op authors using the nanobind-exposed `address_offset` parameter on `CBDescriptor` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp`).

If you find no concrete non-zero usage in the op being ported, this rule is green — that is the expected outcome for most ops today.

### GlobalSemaphore — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. The `KernelSpec` source confirms it: `GlobalSemaphore bindings` is listed under *"Additional program parameter binding types (coming soon)"* in `tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp`, alongside the not-yet-implemented `GlobalDataflowBuffer` and `MeshBuffer` bindings.

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

### Variable-count compile-time arguments (CTA varargs) — UNSUPPORTED

**Status**: Not yet supported in Metal 2.0. Ops whose structure requires a *variable number of compile-time arguments* — for example, ops that accept a list of input tensors of runtime-varying count, or kernels that iterate over a runtime-varying number of CTAs — cannot be ported today. Metal 2.0's `compile_time_args` schema requires fixed-shape declaration at factory-construction time; there is no kernel-side equivalent of the legacy positional-CTA loop yet. A CTA-vararg feature is on the host API roadmap.

**Recognition — definitely this feature** (refuse and report):

- **Op-level signal.** The op accepts a *variable number of input tensors* — e.g., the device-operation class's `tensor_args_t` carries a `std::vector<Tensor>` (or equivalent variable-count container) rather than a fixed-count tuple of named tensors. Treat this as a **prompt to inspect the kernel, not a verdict on its own:** a variable-count input list is necessary but *not* sufficient. The author may thread per-input metadata through CTA varargs — but may equally use RTAs or runtime CB-streaming, both of which Metal 2.0 supports. Do not fire the rule off this signal alone.
- **Kernel-level signal (the decider).** The kernel reads *compile-time* args using a *runtime-varying index* — e.g., `get_compile_time_arg_val(i)` inside a loop where `i` depends on a count value, or a kernel template instantiated over a variable count derived from a CTA. (Args read at **constexpr** offsets — even computed ones — are fixed-count, not this.)

The **kernel-level signal fires the rule**; the op-level signal only tells you to go read the kernel. If you genuinely cannot resolve how the kernel consumes the count, default conservative — treat it as CTA varargs (RED) and record the uncertainty.

**Recognition — false-positive guard**:

- *RTA varargs* (`get_vararg(i)` for runtime args) ARE supported in Metal 2.0 via the kernel-side vararg mechanism — see the porter recipe's [kernel-side whitelist rule 4](../port/metal2_port.md#kernel-side-whitelist) and the [patterns catalog's Caution on varargs](../shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary). The rule fires only on *compile-time* varargs.
- A fixed-count list of input tensors known at port time (e.g., always exactly 4 inputs) is not variadic — that's a multi-input op with a known shape. Port it as multiple named `TensorParameter`s and `TensorBinding`s.
- A **variable-count input list whose per-input data rides RTAs or runtime CB-streaming** (an RTA-driven count loop, *not* a compile-time-arg loop) is **not** CTA varargs — the compile-time arg *count* is fixed. Classify N/A. *Example:* `matmul` carries a `std::vector<Tensor>` (a, b, weights) and a runtime-varying multi-weight path, yet its kernels read compile-time args only at **constexpr** offsets — no runtime-varying CTA index — so it is N/A here. This is the case the `std::vector<Tensor>` op-level cue over-fires on (it is a *prompt to read the kernel*, per the op-level signal above, not a verdict).

**Action**: STOP. Report to the user that this op's structure requires a CTA-vararg feature Metal 2.0 does not yet support. Do not attempt to capitulate by demoting CTAs to RTAs (that's the [Demoting per-group CTA to RTA anti-pattern](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)) or by hand-unrolling the variable-count loop in the kernel.

**Examples in the wild** (for ground-truthing your match):
- `ttnn/cpp/ttnn/operations/data_movement/concat/` — accepts a runtime-varying list of input tensors.

---

## After you submit

A grounded RED audit is not a failed port; it is the audit working as designed. The deliverable here is clarity — what porting this op would actually require, surfaced clearly enough that a colleague can act on it. Your job in this document was to decide whether the port is feasible, not to perform it; once the report is on its way, that work is complete.
