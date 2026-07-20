# Porting an Op to Metal 2.0 — Port Recipe

> This is the second of two documents covering the Metal 2.0 op port workflow. **This document covers the port itself — legacy inventory, spec planning, construction, and verification.** The feasibility audit (the gate that precedes the port) lives in [`audit/metal2_audit.md`](../audit/metal2_audit.md) and is a hard prerequisite to anything in this document.

## Read this first

**Audience**: AI agents asked to perform the actual Metal 2.0 port of a TTNN op, *after* the feasibility audit has cleared with GREEN status and the user has explicitly approved proceeding.

**If you're new to this stack — quick orientation:**

- **Tenstorrent accelerators** come in two architectural generations. **Gen1** is the shipping silicon today: `WH` = Wormhole, `BH` = Blackhole. **Gen2** is in development: `Quasar` (and siblings). The ops you'll port target Gen1, but Metal 2.0 is designed to serve both — the same API targets both architectures.
- **TTNN** is the high-level neural-network library for Tenstorrent accelerators. Ops live in `ttnn/cpp/ttnn/operations/<family>/<op>/`. A typical op has a device-operation class on the host side and one or more program factories that build what runs on the accelerator.
  - **Resolving the op you were given.** You may be handed a full path, a `<family>/<op>` (e.g. `data_movement/concat`), or just a bare op name (e.g. `concat`). A `<family>/<op>` appends directly under `ttnn/cpp/ttnn/operations/`; a bare name you locate with `find ttnn/cpp/ttnn/operations -type d -name <op>` — **one** hit is the op directory, **several** means list them and ask the user which, **zero** means say so and stop. Before proceeding, confirm the resolved directory really is an op: it has a `device/` subdirectory containing a `*_device_operation.*` and one or more program factories. (This guards against a bare name resolving to a non-op directory.)
- **Metal 2.0** is the new **host API** — what the program factory uses to declare kernels, buffers, semaphores, and bindings. It also introduces **DFB** (Dataflow Buffer) at the spec layer, replacing the legacy **CB** (CircularBuffer); the two are essentially synonyms on Gen1, but DFB's semantics diverge meaningfully on Gen2. **This 1:1 CB→DFB mapping covers the _plain_ CircularBuffer only.** A legacy **GlobalCircularBuffer** — a distinct, *user-managed* buffer the codebase often calls a *"remote CB"* — is **not** a DataflowBuffer and is not portable yet; it is a separate, blocked case (see [kernel-side whitelist rule 1](#kernel-side-whitelist)).
- **Device 2.0** is a *separate, earlier* overhaul of the **kernel-side** data-movement APIs (safer, more object-oriented wrappers — `experimental::Noc`, kernel-side `CircularBuffer` wrappers, etc.). It's a **bundled prereq** to Metal 2.0 — the audit checks that ops are already on it — not part of Metal 2.0 itself.
- **Common acronyms you'll see throughout:** `CB` = CircularBuffer; `GlobalCB` = GlobalCircularBuffer (a *separate*, user-managed buffer — **not** a plain CB, and not portable yet; see [rule 1](#kernel-side-whitelist)); `DFB` = DataflowBuffer (see above); `RTA` = runtime args; `CTA` = compile-time args; `CRTA` = common runtime args (values broadcast to all nodes); `TA` = TensorAccessor; `LLK` = Low-Level Kernel (the framework-provided kernel-side primitives); `NoC` = Network-on-Chip (the on-die fabric).

For the conceptual map of how Metal 2.0 abstractions fit together — `ProgramSpec`, `KernelSpec`, `TensorParameter` / `TensorBinding`, `DataflowBufferSpec`, the spec/run-args split — see [`migration_guide.md`](../shared/migration_guide.md). The recipe below assumes you've at least skimmed it.

**Precondition — non-negotiable**: You may only invoke this document if:

1. The audit in [`audit/metal2_audit.md`](../audit/metal2_audit.md) was performed for this op and produced an **overall GREEN** result.
2. The user has **explicitly asked you to proceed** with the port. A green audit alone is not sufficient — the user must have read the audit and given an unambiguous go-ahead.

If either condition is unmet, stop. Return to the audit document. Do not improvise.

**What a cleared audit guarantees — don't redo (or absorb) the op owner's pre-port work.** One class of audit finding is an **op-owner pre-port functional change**, fixed on a separate branch/PR *before* the port — so by the time you run, it is already done:

- **Offset base pointers are split to clean bases.** An op that folded a host-computed offset into a base pointer — passing `buffer()->address() + offset` through an address RTA — is refactored by the op owner to pass a clean base (which becomes a `TensorBinding`) plus any offset as a separate scalar, *before* the port. So every tensor address you receive is a **clean base you can bind directly**; you never move offset arithmetic host→device yourself — that would breach the syntax-swap invariant (see [kernel-side whitelist rule 5](#kernel-side-whitelist)). This is the audit's [Offset base pointers](../audit/metal2_audit.md#offset-base-pointers) gate.

If you nonetheless hit a host-computed `base + offset` folded into an address arg mid-port, **stop and surface it; do not fix it in the port.** It means a pre-port gate wasn't cleared, and folding a functional fix into the Metal 2.0 port entangles two changes that must stay separate — which makes any resulting bug nearly impossible to localize. Route it back per [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit).

**Dead CBs and multi-binding CBs are *your* work, not the op owner's — you handle them during the port.** On Gen1 a DFB has full CircularBuffer parity, so these need no pre-port fix: the audit clears them as PORT WORK (the [CB endpoints](../audit/metal2_audit.md#cb-endpoints) subject is GATE-free) and the brief lists each with its disposition. You apply the disposition during construction — **drop** a dead CB, **set the multi-binding flag** on a multi-bound one — never a reason to stop. Mechanics in [§Construct paired spec + run-args](#construct-paired-spec--run-args).

**Why this migration matters.** Metal 2.0 is a substantial overhaul of the host API for programming Tenstorrent hardware — hold this motivation in mind, because it shapes the judgment calls the recipe itself can't fully specify, and it's why the work is worth doing. The legacy host API hurt the project for years in concrete, structural ways. Positional `uint32_t` kernel arguments hide the meaning of every slot; the compiler can't catch a mismatch. Magic CB indices propagate through CTA lists with no compiler-checked connection back to the CB they reference. Raw pointers flow through the host interface straight into device kernels with no safeties, surfacing as hard-to-debug device hangs when they go wrong. And because the legacy API doesn't tell the framework which kernel uses which CB or in what role, the framework can't catch your mistakes at spec-construction time — they surface at runtime as hangs or wrong numerics. Every legacy idiom you'll replace in this port — magic CB indices, positional CTAs, buffer-address RTAs, host-side `TensorAccessorArgs` plumbing — is an instance of one of those problems. **Metal 2.0's typed binding model is the fix:** bindings the compiler checks, a host that knows the device topology (so the spec validator catches mistakes before launch), and a binding mechanism that carries the meaning the legacy `uint32_t` channel couldn't.

**Your port is one link in a chain.** Metal 2.0 is pulling adjacent migrations through with it under a Gen2-customer-driven schedule: Device 2.0 and TensorAccessor adoption are bundled prereqs the audit gates on, and TTNN's ProgramDescriptor migration — a parallel TTNN-side effort, coupled to Metal 2.0 by design — is racing to finish under the same pressure. The API you're learning here is the same API that will target Gen2 (Quasar) hardware; there aren't two APIs to learn. Landing your port unblocks the next op, which unblocks its consumers, which unblock framework features the runtime team has planned. The cumulative effect, once the push lands, is a meaningfully cleaner way to specify what runs on a Tenstorrent accelerator.

**Operating posture.** Two principles govern every judgment call in the port. Both cut against the natural pull to be maximally helpful by solving everything in front of you — so they're stated plainly here, with the full reasoning in the sections named:

- **When Metal 2.0 doesn't fit, stop and report — this is a successful outcome, not a failed one.** The audit cleared the features and prereqs known at audit time, but during the port you may discover something it missed: a feature gate that didn't fire, a kernel pattern that doesn't translate cleanly, an interaction nobody anticipated (including an API that behaves as though it has a bug — report it, don't paper over it). The correct response is to surface it, not improvise around it. In particular, if you find yourself constructing a clever workaround — packing data into varargs to simulate a missing field, threading a buffer address through an RTA because the binding mechanism doesn't fit, hand-rolling a synchronization primitive — **stop.** Whatever you're about to write is almost certainly wrong. A grounded stop is a complete, valued deliverable, on the same success tier as a finished port — see [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit).
- **When you notice an unrelated improvement, leave it and route it to the report.** Refactors, bug fixes, stale comments, a too-tight validation you can see how to loosen — real as they may be, they don't belong in this diff. They aren't suppressed; they're delivered to the people with the context to act on them, via `METAL2_PORT_REPORT.md`. The full rationale — and why bundling them is actively harmful, not merely unnecessary — is in [§Scope discipline](#scope-discipline); read it before you touch a file.

**The "why" as a judgment heuristic.** When the recipe doesn't quite fit and you're tempted to improvise, the test is whether what you're about to do preserves typed bindings, the host-aware binding model, and the spec/run-args separation that the verification step audits. If yes, you're probably on a supported path. If no — varargs-packing, address-through-RTA, hand-rolled sync — you're recreating the legacy problems Metal 2.0 was built to fix. Stop and ask.

**Working practices for the bulk-port effort.** This is one of many ports happening in parallel against actively-evolving docs. Read the following before you start; they shape how you should approach the work.

- **Your friction signals are a real deliverable.** The team needs them from each port to improve the recipe / patterns catalog / migration guide; a port that captures a real blocker and stops is more valuable than one that powers through with workarounds. The `METAL2_PORT_REPORT.md` you produce is part of why you are doing this work, not an afterthought.
- **Capture friction as you go, not at the end.** Open `METAL2_PORT_REPORT.md` at the start of the port. When something surprises you — a missing doc, an unexpected error, a near-miss — write a one-line note immediately, even rough. A friction note added in the moment is worth more than a polished paragraph reconstructed two hours later. Polish at the end; capture in the moment.
- **Time budget on stuck points.** If you spend more than ~30 minutes on a single stuck point with no traction, **stop**. Capture what you tried, what failed, and your current hypothesis in `METAL2_PORT_REPORT.md` under "Friction" or "Open items," then move on or surface the question to the invoker. Burning the day powering through a stuck point is not the assignment.
- **Do not 'fix' the legacy kernel.** If the legacy kernel does something you don't understand or that seems wrong, the legacy kernel is the source of truth for what the op does. Surface kernel-level questions as questions in the report; do not modify the kernel's logic to match what you think Metal 2.0 wants. (Mechanical edits to make the kernel build against Metal 2.0 APIs — named handles, generated headers, etc. — are a different category and are expected.)
- **Don't lean on the already-ported ops as templates — the recipe outranks the wider codebase.** The ops already on Metal 2.0 under `ttnn/cpp/ttnn/operations/experimental/quasar/` were converted in an earlier, pre-completion push, before the API and its conventions (hardware configs especially) had settled — so they **do not represent current best practice**, however tempting a ready worked example is. Treat them, and anything else in the wider codebase, as skeptically-held reference at most; where it disagrees with this recipe, the [patterns catalog](../shared/port_patterns.md), or the [migration guide](../shared/migration_guide.md), those three are authoritative. A ported op that contradicts the recipe is friction to note, not a pattern to copy.

**Scope boundary — read carefully.** The porter's writeable surface is the **op's own directory** (the device-op factory, its kernels, its tests, the four `METAL2_*.md` artifacts). Files outside that directory — shared kernel-lib headers under `ttnn/cpp/ttnn/kernel_lib/`, LLKs under `tt_metal/`, framework primitives — are out of scope. The port respects this boundary; it does not propose changes to those files.

**The atomic unit of a port is one ProgramFactory — not the whole device-operation.** A device-op may hold several factories in its `program_factory_t` variant (e.g. an interleaved factory and a sharded factory). You port **one factory together with the kernel entry points it binds**, as a unit; you do *not* have to convert the whole op at once. A `program_factory_t` variant is valid with its factories on *different* concepts — one on Metal 2.0's `MetalV2FactoryConcept`, the others still on the legacy `ProgramDescriptorFactoryConcept` — and the framework dispatches per-factory at runtime, so a half-ported op builds and runs correctly. For a large multi-factory op this is the lever that keeps the port tractable: port one factory now, the rest later. **When handed a multi-factory op, default to porting the factories one at a time, autonomously — don't raise a "which factory?" question.** Each factory is a complete sub-port: a finished factory with its other factories cleanly reported as remaining work is a valid, shippable deliverable, not a partial failure. Port one factory fully (its own `METAL2_PORT_PLAN.md` / `METAL2_PORT_REPORT.md` entries), then continue to the next only if it's clearly tractable in the same pass; stopping after one with the rest enumerated for the next pass is the expected shape for a large op. The variant's other factories stay on their legacy concept meanwhile, and the op keeps building and running.

*Within* that unit the conversion **is** atomic: a factory that speaks Metal 2.0 bindings can only launch kernels whose entry points read those bindings, so the factory and every kernel-source entry point it can select (including runtime-selected sources) flip together — there is no half-Metal-2.0 factory. **Shared kernels between the op's own factories** are therefore the thing to watch, and not all "sharing" is alike:
- **Shared top-level entry point** (two of the op's factories bind the *same* kernel `.cpp`'s `kernel_main`): that source gets Metal-2.0-ified for whichever factory you port, which breaks the other factory's use of it. Treat it like a shared kernel — port both factories together, or fork the source (see [Caution: Modifying a shared dataflow kernel](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel)).
- **Shared routines / cross-calls** (kernels calling common helpers or kernel-lib functions): bridged by the boundary features (`dfb::name`→`uint32_t`, etc. — see *Crossing the boundary* below). Not a coupling; leave the helpers alone.

**Crossing the boundary in kernel code.** Some kernel call sites in the ported kernels invoke functions whose source lives outside the op directory — kernel-lib helpers (`dataflow_kernel_lib::*`, `compute_kernel_lib::*`), LLKs (`reduce_init`, `pack_tile`, `cb_wait_front`, etc.). These callees take `uint32_t` CB ids today.

- **`dfb::name` crosses freely.** Pass `dfb::name` directly at the call site. The `DFBAccessor::operator uint32_t()` implicit conversion bridges the named handle to the legacy `uint32_t` signature without `.id` extraction, temporary wrappers, or typed shims. See [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](../shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).
- **`sem::name` and `tensor::name` do NOT cross — assumption.** Unlike `dfb::name`, the semaphore and tensor-accessor handles have no implicit conversion to `uint32_t` today. **The recipe assumes that no out-of-op call site requires passing one** — semaphores and tensor accessors are consumed inside the op's own kernels. If you encounter a call site whose callee lives outside the op directory and that requires a operand that is structurally unavailable to the Metal 2.0-converted code, this is an **assumption violation**. These are supposed to have been already prepared for Metal 2.0 migration. Do not write the call. Do not preemptively wrap, refactor, or extract — the fix is upstream of the porter's scope. Stop and record the site in [`METAL2_PORT_REPORT.md` — Handoff points](#capture-the-port-report) so the kernel-lib / API owners can address it.

**Two exceptions to the boundary rule** — these are not "out-of-op" call graphs:

- **Cross-op kernel files** — some ops share dataflow kernels that live in another op's directory (e.g., `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` reused by many ops). The legacy inventory step flags these; modifying them is porter-touchable with caution per [Caution: Modifying a shared dataflow kernel](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel). These are *peer ops*, not framework callees. Document the changes you found necessary in the `METAL2_PORT_REPORT.md`.
- **Framework primitives the porter uses directly** — `noc.async_read(...)`, `dfb.wait_front(...)` on a `DataflowBuffer` the porter constructs locally from `dfb::name`, the `TensorAccessor(tensor::name)` constructor, etc. These are *consumed by* the porter's kernel code (named handles flow in via the documented constructors); they are not handoffs to out-of-op code.

**Generated docs in the op directory.** Four `METAL2_*.md` files live in the op's directory alongside the program factory `.cpp` files — two written by the audit (your inputs), two you write during the port:

- `METAL2_PORT_BRIEF.md` — **the porter-facing audit brief: your actionable input** (the inherited factory concept, tensor-binding cases, and watch-fors). Written by the [audit doc](../audit/metal2_audit.md), not this one; present as input.
- `METAL2_PREPORT_AUDIT.md` — the team-facing full audit record, present alongside the brief. The brief mirrors its porter-relevant items; consult the team doc only for detail the brief doesn't carry (e.g. its identifying section's full factory / kernel file listing — the starting point for your own inventory — plus the complete gate detail).
- `METAL2_PORT_PLAN.md` — the port plan (this recipe's load-bearing artifact). Externalizes structural decisions before mechanical translation begins. Read by you during construction and verification, by human reviewers during PR review, and by future debuggers. See [Appendix A](#appendix-a--metal2_port_planmd-template) for the template.
- `METAL2_PORT_REPORT.md` — the post-port report. Records handoff points, successes, friction, and open items observed during the port. Written at the end of the port; feeds doc evolution and informs the kernel-lib / API teams. See [Capture the port report](#capture-the-port-report) for the structure.

All four are committed alongside the port.

**Workflow at a glance**:

1. [**Legacy inventory**](#legacy-inventory) — Consume the audit; record the legacy structure to `METAL2_PORT_PLAN.md`.
2. [**Plan the spec**](#plan-the-spec) — Apply host-side specialization principles; identify legacy plumbing that should evaporate. Externalize all structural decisions to the plan.
3. [**Construct paired spec + run-args**](#construct-paired-spec--run-args) — Construct the spec and run-args, paired by resource. Mechanical translation per the plan.
4. [**Verification**](#verification) — Build, run tests, run anti-pattern self-audit against the [patterns catalog](../shared/port_patterns.md).
5. [**Capture the port report**](#capture-the-port-report) — Write `METAL2_PORT_REPORT.md` recording handoff points, successes, friction, and open items for downstream.

**Reference material** the recipe relies on, loaded on demand:

- [Migration guide — Design Principles](../shared/migration_guide.md#design-principles): why certain legacy plumbing should evaporate during port.
- [TTNN integration doc](../shared/ttnn_factory.md): the factory concept, the `create_program_artifacts` entry point, the cache lifecycle, and the device-op-class edits the port forces.
- [Patterns catalog](../shared/port_patterns.md): recognition signals + decisions for structural patterns and anti-patterns.
- [CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md): the authoritative CB-method → DFB-method mapping for the kernel-side swap — canonical FIFO, size / layout and metadata getters, cursor peeks, and cursor surgery (`evil_set_*`).

---

## Before you begin

### Inputs the invoker should have supplied

Before starting, confirm you have the following. If any is missing or unclear, ask the invoker before proceeding — the invoker can answer in seconds; you will burn substantial time guessing.

- **Legacy op source path** — the directory containing the op's legacy program-factory `.cpp`/`.hpp` files and kernel sources.
- **Tests directory** — the path under `tests/ttnn/unit_tests/operations/<op-family-slug>/` where the op's pytests live. Note that the directory uses the **op-family slug**, not always the literal op name (e.g., reduction's tests live at `tests/ttnn/unit_tests/operations/reduce/`, not `reduction/`). If the invoker didn't supply this, discover it with `find tests/ttnn -path '*operations*' -name 'test_*.py' | grep -i <op>`.
- **`METAL2_PORT_BRIEF.md`** (with the team-facing **`METAL2_PREPORT_AUDIT.md`** alongside it) — present in the op directory; the brief is issued only when the audit cleared (precondition above). The brief is your actionable input: its **TTNN factory analysis** section records the Metal 2.0 factory concept the port targets, inherited not re-derived. See the [TTNN integration doc](../shared/ttnn_factory.md) for the concept and the device-op-class edits the port forces.
- **(Optional) Reference port** — a recently-completed similar op the invoker recommends studying for shape. The invoker may not always have one; absence is not a blocker. The first worked end-to-end `create_program_artifacts` port is **accumulation** (cumsum/cumprod) on branch `akertesz/porting-experiment-accumulation-jun10` — a small single-program op exercising tensor bindings, a self-loop DFB, work-split multiplicity, and the custom-hash deletion. A good shape reference when no closer op exists. It lives on that sibling branch, **not** your port branch — read it with `git show akertesz/porting-experiment-accumulation-jun10:<path>` rather than expecting it in your working tree.

### Workspace bootstrap

If you have no existing tt-metal checkout, follow [`workspace_setup.md`](../shared/workspace_setup.md) for the clone / Python-env / build sequence. Key facts captured there that often trip up porters:

- The right build flag for op work is `./build_metal.sh --build-tests`. `--build-metal-tests` alone is **insufficient** — it omits TTNN gtests entirely.
- `PYTHONPATH` must be `$(pwd)` from inside your clone, not a path copied from someone else's instructions.
- Iterative rebuilds during the port: `cmake --build build_Release --target ttnncpp unit_tests_ttnn -j 8`.

### Use a subagent for builds and tests

Builds and test runs produce large output that will pollute your working context — `./build_metal.sh` writes thousands of lines, gtest output on a full op test directory is even worse. **Delegate every build and every test run to a subagent.** The subagent runs the command, captures all output to a log file, and returns only a tight summary (exit code + extracted errors + log path). You do the reasoning; the subagent absorbs the noise.

**Subagent prompt template:**

````
You are a build/test helper for a Metal 2.0 op porter. Run the command below, capture all output to <log_path>, and return ONLY a tight summary.

Command: <command>
Working directory: <cwd>
Log path: <log_path>

Return format:
- Exit code: <code>
- Status: SUCCESS / FAILURE / TIMEOUT
- Key errors (if FAILURE): up to 10 lines of compiler or test errors, no boilerplate stack frames unless they show the cause.
- Log path for deep dive: <log_path>

Do NOT try to fix anything. Do NOT run other commands. Do NOT read unrelated files. Do NOT make recommendations about next steps.

If the build or test hangs (>15 min with no log progress), kill it, return TIMEOUT, and report the last 20 lines logged.
````

**When to use:** every build, every gtest run, every pytest run. If you are iterating on a single failing test and have already seen the error once, reading the log file directly is fine — but the first invocation of any build or test command should always go through the helper to get a clean error extract.

**Don't use the helper to make decisions.** The helper runs and reports. You read the report and decide what to do next. If you need to investigate a specific error in detail, read the log file the helper points you at.

---

## Legacy inventory

*This is an observation step. No decisions yet.*

**Inputs**:
- The audit artifacts in the op directory: the **brief** (`METAL2_PORT_BRIEF.md`, your actionable input) and the team-facing **`METAL2_PREPORT_AUDIT.md`**, whose identifying section and gate detail name the factories and kernel files — the starting point for the inventory.
- The op's program-factory `.cpp` / `.hpp` files.
- The kernel sources referenced by the factories.

**Output**: write the **Legacy Inventory** section of `METAL2_PORT_PLAN.md` to the op's directory. Record:

- **Legacy factory shape**: which `ttnn::device_operation` concept the legacy factory currently satisfies (`ProgramFactoryConcept` / `ProgramDescriptorFactoryConcept` / `MeshWorkloadFactoryConcept`). For each variant (if the device-operation is multi-variant), record separately. *(The Metal 2.0 factory concept the port lands on was chosen during the audit — see the brief's TTNN factory analysis section. The recipe inherits that decision; carry it forward to the [TTNN ProgramFactory port-plan section](#ttnn-programfactory) below.)*
- **Custom `compute_program_hash`**: does the device-operation define a custom `compute_program_hash` (overriding the default reflection-based hash)? If yes, record file:line — **the port deletes it**, reverting to the default hash. This is one of the two sanctioned device-op-class edits the port forces; the rationale and procedure live in the [TTNN integration doc — Delete a custom `compute_program_hash`](../shared/ttnn_factory.md#1-delete-a-custom-compute_program_hash). Record the deletion in the port report.
- **Kernels**: every `KernelDescriptor` (one row per descriptor):
  - `kernel_source` (file path; flag any path outside the op's own directory — cross-op kernels are a Caution case).
  - `core_ranges` (verbatim).
  - `compile_time_args` (positional values) — read these from the **host factory's emission order**, which is authoritative for slot positions; a kernel-side reconstruction (counting `get_compile_time_arg_val(N)` reads) is easy to misalign, especially in a delegated inventory.
  - `named_compile_time_args` (name → value pairs).
  - `runtime_args` and `common_runtime_args` (names if known, dimensions, and shapes).
  - `defines` (key → value pairs).
  - `config` (the descriptor type: `ReaderConfigDescriptor` / `WriterConfigDescriptor` / `ComputeConfigDescriptor` and its content).
- **CBs**: every `CBDescriptor` (one row per descriptor):
  - `total_size`, `core_ranges`, format descriptors (`buffer_index`, `data_format`, `page_size`, `tile` if set).
  - **Flag any GlobalCircularBuffer separately — it is not a plain CB.** Signals: a `CBDescriptor` with `.global_circular_buffer` set; an `experimental::GlobalCircularBuffer` / `global_cb` parameter; or the `remote_cb_config` + `CreateCircularBuffer(program, cores, cfg, *global_cb)` construction idiom (legacy code calls these *"remote CBs"*). A GlobalCircularBuffer maps to the *user-managed* GlobalDataflowBuffer (split by lifetime — **not** to a `DataflowBuffer`), which is unimplemented, so the factory using it is **blocked**. The audit should have caught it ([audit — GlobalCircularBuffer UNSUPPORTED](../audit/metal2_audit.md#globalcircularbuffer--unsupported)); if it slipped through, record it here and the port capitulates on that factory ([§When the discipline doesn't fit](#when-the-discipline-doesnt-fit)) — the op's other factories may still be portable.
- **Semaphores**: every `SemaphoreDescriptor`:
  - `id`, `core_type`, `core_ranges`, `initial_value`.
- **Tensor accessors**: every `TensorAccessor` use site (host and device):
  - Originating `Tensor` (input / output / which input/output).
  - Where its address surfaces in the host RTA list.
- **Work split**: which `split_work_to_cores` call (or similar) drives the per-core counts. Record `(num_cores, all_cores, core_group_1, core_group_2, count_per_group_1, count_per_group_2)`.
- **Cross-op kernels**: explicitly list any kernel source path outside the op's directory; flag for [Caution: Modifying a shared dataflow kernel](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel).
- **Factory variants**: if the device-operation has multiple variants (e.g., reduce W vs H vs HW; Welford W/H/HW), record each variant's complete inventory.
- **Runtime kernel-source selection**: if the factory chooses its kernel *source file* at runtime (a switch over kernel paths, rather than one fixed source per `KernelDescriptor`), record every source it can select. Because the spec's bindings must match the selected source, the factory and **all** of its selectable sources convert together — there is no "port the common path only" sub-target that builds. This (not the whole device-op — see [the atomic-unit note](#read-this-first)) is the true size of the port: one factory + every kernel entry point it can bind. Selection often runs on **several independent axes at once** — broadcast type × compute flavor (`is_sfpu` / `is_where`) × layout (tile / row-major) — and they *multiply*: a single "no-broadcast" path can still fan out to multiple compute kernels (e.g. fp32 `add` routes to the *SFPU* kernel, not the FPU one). Enumerate entry points across **all** axes, not just the obvious one. Size the effort against that — against the budget of the fresh **primary** session on a 1M-context model that the porting workflow runs you in, not a subagent's narrower one. A single factory is the atomic unit and does not build until all its selected sources convert together, so a genuinely too-big factory is a *vehicle* limit, not a fit gap — see [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit) for why size is not grounds for capitulation, and for the whole-factory handoff. Map each DFB's producer/consumer roles **per selected source path**, not once for the op: a DFB's producer can *move between kernels* across paths — e.g. an input DFB filled by the *reader* on a tile path but by *compute* (via `tilize`) on a row-major path — so one fixed role assignment will mis-bind the other path.

**Stop signals**:
- An unreferenced kernel file in the op's directory: not a stop, but note in the inventory's "Flags" subsection (so the report makes clear what was *not* audited).
- A descriptor type not in the audit's scan: stop and report. The audit's Appendix A is the authoritative scope — anything the legacy factory uses that doesn't map onto an entry there is a signal the audit was incomplete.

---

## Plan the spec

*This is the load-bearing planning step. Externalize all structural decisions before writing code.*

**Inputs**:
- The Legacy Inventory written in the previous step.
- **The brief's TTNN factory analysis section** — the Metal 2.0 factory concept your port targets was confirmed during the audit. Plan against it; do not re-derive it.
- The [migration guide — Design Principles](../shared/migration_guide.md#design-principles), especially Principle 2 (named bindings) which drives the "Dropped Plumbing" section below.
- The [TTNN integration doc](../shared/ttnn_factory.md#the-metal-20-factory-concept) for the factory concept the ported op will satisfy and the entry point that returns the spec.
- The [patterns catalog](../shared/port_patterns.md).
- The brief's **Watch for** notable constructs.

**Output**: extend `METAL2_PORT_PLAN.md` with the following sections (see [Appendix A](#appendix-a--metal2_port_planmd-template) for the template). Each section may say "none" with a one-line justification when no items apply.

### TTNN ProgramFactory

Carry forward the audit's factory decision. This section is brief — the audit confirmed the concept and the recipe inherits the result; the recipe is *not* the place to re-derive the choice.

- **Concept (inherited from audit)**: copy the concept name from the brief's TTNN factory analysis section (`MetalV2FactoryConcept` for every portable op today).
- **Custom `compute_program_hash`**: note whether the audit flagged one for deletion (carried from the Legacy Inventory).
- **Implementation notes** (optional): anything specific to how this op will realize the concept that's worth surfacing before construction. Most ports don't need this.

If you find yourself disagreeing with the audit's choice, **stop and surface the disagreement to the invoker** — do not unilaterally override. The audit is the source of truth for the chosen concept; an in-port revision is a signal the audit was incomplete and the invoker needs to know. See the [TTNN integration doc](../shared/ttnn_factory.md) for the concept and the device-op-class edits the port forces.

### Planned Spec Shape

The Metal 2.0 spec shape. Default: 1:1 with legacy.

- **KernelSpecs**: one per legacy `KernelDescriptor` (default). When the legacy factory has multiple `KernelDescriptor`s of the same source for work-split, **preserve the multiplicity**: one `KernelSpec` per legacy `KernelDescriptor`, with the per-group CTAs reproduced. See [Anti-pattern: Demoting per-group CTA to RTA](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) for why this is non-negotiable.
- **DataflowBufferSpecs**: one per legacy `CBDescriptor`, with one extra spec per additional `buffer_index` for aliased CBs (legacy multi-element `format_descriptors`). For aliased cases, each member of the alias group declares the other(s) via `advanced_options.alias_with` — see [Pattern: Aliased DFBs](../shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs) for the legality constraints. For borrowed-memory cases (legacy `CBDescriptor::buffer` set), plan the DFB with `borrowed_from = <tensor_parameter_name>` naming the `TensorParameter` whose buffer backs the DFB.
- **SemaphoreSpecs**: one per legacy `SemaphoreDescriptor`.
- **TensorParameters**: one per distinct legacy `TensorAccessor` originating tensor. Note that multiple kernel-side accesses to the same tensor collapse to one `TensorParameter` with multiple `TensorBinding`s.
- **WorkUnitSpecs**: one per distinct (set of kernels, target nodes) pairing. Most ops have one or two.
- **Op-owned tensors** (rare — most ops have none): one per legacy device tensor the factory allocated beyond the op's declared io — the tensors the legacy `WorkloadDescriptor` carried as `WorkloadBuffer`s (sliding-window config / reader-index LUTs and the like; the audit's [TTNN porting shape](../audit/metal2_audit.md#ttnn-porting-shape) names them). They are carried in the artifact's `op_owned_tensors`, built as today and released into the artifact, then bound like io tensors — see the op-owned step under [§Construct paired spec + run-args](#construct-paired-spec--run-args).

### Preserved Multiplicity

For each work-split case, explicitly list the multi-`KernelSpec` mapping:

```
Legacy KernelDescriptors [list] of source <path>
  → KernelSpecs [list] of same source
  → in WorkUnitSpecs [list]
  → sharing upstream/downstream DFBs as multi-bindings: [list]
```

If the legacy code has no multi-`KernelDescriptor` work split, this section reads "none — no work-split multiplicity in legacy."

### Dropped Plumbing

For each legacy RTA or CTA that should *not* survive the port, list it with the replacement primitive:

- **Buffer-address RTAs** (`tensor.buffer()` / `tensor.buffer()->address()` in legacy RTA values; `get_arg_val<uint32_t>(N)` in kernel passed to `TensorAccessor`): replaced by `TensorBinding`. List each kernel's affected RTA slot.
  - *Case 2 (raw pointer) bindings.* If the audit classified a binding as **Case 2** (the kernel uses a raw base pointer), it still flows through the typed channel as a `TensorBinding`; the base pointer is pulled kernel-side via the sanctioned `get_bank_base_address` bridge (see [kernel-side whitelist rule 5](#kernel-side-whitelist) for the mechanics). List each affected binding here. **A Case 2 binding in a *compute* kernel is blocked** — no bridge exists there yet (rule 5) — so the port fails pending the compute-kernel `TensorBinding` fix. (Case 1 bindings — via `TensorAccessor`, the common case — need no special note.)
- **Magic CB indices in CTAs**: replaced by `DFBBinding`. List each kernel's affected CTA slot.
- **`TensorAccessorArgs` plumbing**: replaced by the binding mechanism end-to-end. List each `TensorAccessorArgs(buffer).append_to(cta)` site and its kernel-side `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` chain.
- **Page-size 3rd-argument CTAs/RTAs**: a `page_size` value emitted solely to feed a `TensorAccessor`'s third constructor argument. Dropped — the binding token supplies the aligned page size automatically (see [kernel-side whitelist rule 3](#kernel-side-whitelist)). List each affected CTA/RTA slot. (The brief lists the accessor sites; a site could also carry a `TensorParameter` relaxation, which the brief flags separately.)
- **Semaphore-ID RTAs**: replaced by `SemaphoreBinding`. List each kernel's affected RTA slot.
- **Positional CTAs**: replaced by named CTAs. List each kernel's positional CTA list with the names you'll assign.

This section's enumeration is the gate against builder-pattern carry-over. If a legacy RTA / CTA is not listed here, it will be translated by reflex during construction — which is exactly the failure mode this gate exists to prevent. See [migration guide — Principle 2](../shared/migration_guide.md#principle-2-first-class-named-resource-bindings) for the rationale.

### Applied Patterns

For each non-trivial pattern from the [catalog](../shared/port_patterns.md) invoked by this port, name the pattern and the context:

- "[Self-loop DFB binding](../shared/port_patterns.md#pattern-self-loop-dfb-binding): ACC_DFB on compute KernelSpec (both PRODUCER and CONSUMER)."
- "[Conditional optional binding](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings): SCALED_DFB on compute KernelSpec, gated by `do_scale` CTA."
- "[Multi-variant factory](../shared/port_patterns.md#pattern-multi-variant-factories): `reduce_dim` variant selection inside `create_program_artifacts`."

### Deferred / Flagged

Any new findings the planning step uncovered:

- New findings: anything the audit missed that surfaced during structural planning.

**Stop signal**: if planning uncovers a structural issue the audit didn't catch — e.g., a kernel that genuinely cannot be expressed without one of the legacy workarounds, or a feature gate that the audit's Appendix A doesn't cover — **stop and report**. Don't paper it over by demoting CTAs, packing varargs, or hand-rolling primitives. The audit's gate set improves with what later steps discover.

---

## Scope discipline

*Read this before you touch a file. This section governs every edit in the port — host-side and device-side.*

### The principle

**The premise.** A Metal 2.0 port is a *targeted, scope-tight* transformation: the program factory is restructured to the new host API, and that restructuring projects mechanically into the kernel. Nothing else changes. The expected post-port behavior of the op — its numerics, its performance, every observable side effect — is *identical* to the pre-port behavior. If your diff looks like anything other than the Metal 2.0 transformation, something has gone off-script.

**Two deliverables.** You're producing two things, not one: the diff and the port report. Both are load-bearing. The diff is what ships; the report is what feeds back into the framework's understanding of where its assumptions break, which ops were tricky and why, which patterns surfaced that we hadn't anticipated. A clean diff with a thorough report has the same success-tier as a clean diff with a sparse report — possibly higher, depending on what the report contains. Don't treat the report as a write-up of what you already did; treat it as a deliverable in its own right.

**Improvements are not suppressed — they are routed.** This recipe will at points ask you to leave things alone that you can see how to improve. Refactors, bug fixes, perf optimizations, stale comments, suboptimal type checks, missing validations, hardcoded constants that should be parameters — leave them all. *Write each one up in the report.* Those observations don't disappear into the void; the report is a real channel, read by people with the context to act on findings. The improvement isn't being deleted, it's being delivered to the right hands at the right time.

**Why this is actively harmful, not just unnecessary.** The instinct here — *I see a problem, I can fix it cheaply, it's the right thing to do* — is a good instinct in most software work. In this work it's wrong, and not by a small margin. Four reasons, compounding:

1. **The diff loses attribution.** A port PR that bundles improvements becomes ambiguous in future bisection. When a regression appears six weeks from now and `git bisect` lands on this commit, the question "did the port cause it, or did the bundled change?" is expensive to answer — and the trust damage compounds. A scope-tight diff says "if a regression appears here, it's a Metal 2.0 issue"; a bundled diff says "good luck."

2. **The improvement is denied proper review.** Reviewers looking at a Metal 2.0 port PR have their attention budgeted on the *port*: spec construction, binding shape, kernel projection. They are not budgeted to evaluate a bug-fix patch sitting alongside. The improvement gets less scrutiny than it would get in its own PR — exactly the opposite of what an improvement deserves.

3. **You are not in a position to evaluate the improvement.** The porter sees a local pattern and reasons from local evidence. But ops carry context the porter doesn't have: silicon-generation constraints, calling-convention assumptions, historical bug-fix scars, downstream caller expectations. A change that looks self-evidently correct from inside the file may be wrong from outside it. Example: a porter tightening a `TT_FATAL` on a tensor dtype check to "also accept INT32" may not realize that INT32 support is Quasar-only and this op doesn't target Quasar yet — a "correct-looking" change that is, in context, a bug. *The porter's confidence here is structurally lower than it feels.*

4. **Codebase policy.** PRs that do multiple things at once are explicitly contrary to the codebase's review norms, regardless of how good the bundled changes are individually.

If at any point during the port you find yourself reaching past the rules below, treat that as a signal — *the port has met its limit, not the porter has met theirs.* The [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit) off-ramp at the bottom of this section is the proper resolution; it is on the same success tier as a completed port.

### Host-side: stay in the lane

The host-side discipline divides cleanly along a line between *the program factory body* and *everything around it*:

- **The program factory body is the port.** The `create_program_artifacts` function (and any helpers it calls) is where Metal 2.0's host API is constructed — this code will be heavily rewritten by the port, and that's the work. Stay inside the Metal 2.0 transformation patterns documented in [§Construct paired spec + run-args](#construct-paired-spec--run-args) and [migration guide](../shared/migration_guide.md). Don't refactor adjacent code in the factory while you're there. Don't tighten or loosen a `TT_FATAL` that lives inside the factory. Don't reorder, rename, or "clean up" variables beyond the API renames documented (e.g., `cb_*` → `dfb_*`). The legacy CB API (`CBDescriptor`, `.cbs`, etc.) is replaced by `DataflowBufferSpec` and its companions — no CB references should survive in the post-port factory code (see also Kernel-side whitelist rule 1 for the symmetric kernel-side statement).

- **The op-level host code outside the factory is off-limits.** The device-operation class itself (`validate`, `invoke`, `compute_output_specs`, attribute parsing, the `OpInputs` / `OpParams` definitions, runtime-arg validation, tensor-dtype checks, etc.) is *not* part of the Metal 2.0 port. Do not edit it. Even if the change seems trivial, even if it seems correct, even if you can see exactly what it should say. If you encounter something here that wants changing — a too-tight validation, a stale comment, a `TT_FATAL` whose message is wrong, a missing check — write it up in the port report under findings. Do not change the file.

There are three *documented* exceptions to the off-limits rule — each a device-op-class edit the port forces, each recorded prominently in the port report. The full rationale and procedure for each live in the [TTNN integration doc — Device-operation-class edits the port forces](../shared/ttnn_factory.md#device-operation-class-edits-the-port-forces); in brief:

1. **A custom `compute_program_hash`** — delete it, reverting to the default TTNN hash (when the audit flagged one). Don't patch it to add `TensorSpec`; delete it.
2. **Pybind lines for a legacy factory entry point the port makes vanish** (`create_program_descriptor` is the canonical case) — deletion is mandatory to keep the build green, and it's a user-visible surface change worth a Handoff-points entry.
3. **A factory parameter that exists only for a pybind hook** (e.g. layernorm's `create_descriptor` taking an extra `core_range_set` used only by its pybind hook) — drop the parameter, inline its production default in the factory body, and delete the pybind hook that passed it (mechanically exception 2 with an extra parameter to unwind).

None of these extend to any other device-op-class edit. Everything else here stays exactly as it is, routed to the report.

Concrete example of what *not* to do, drawn from a prior porting attempt:

```cpp
// In the op's device-operation class (NOT the program factory):

// Before the port:
TT_FATAL(k.dtype() == DataType::UINT32, "Only UINT32 dtypes are supported for k!");

// Tempting "improvement" during the port:
TT_FATAL(k.dtype() == DataType::UINT32 || k.dtype() == DataType::INT32,
         "Only UINT32 & INT32 dtypes are supported for k!");
```

This change is wrong in context: INT32 support is Quasar-only, and this op doesn't target Quasar yet. But more importantly, *whether or not it's correct in context isn't the porter's question to answer.* The change is unrelated to the Metal 2.0 port. It belongs in a separate PR with its own review. Bundled into the port, it muddies attribution and crowds out the actual deliverable. The right action: leave the `TT_FATAL` exactly as it is, and write a finding in the port report ("`k.dtype()` is restricted to UINT32; INT32 also seems plausible — flagging for owner review").

### Kernel-side whitelist

The following are the *only* changes you should be making to kernel code during a Metal 2.0 port. The single `#include` a porter adds to a kernel is `experimental/kernel_args.h` (which pulls in `get_arg`, `args::`, `dfb::`, `sem::`, `tensor::`); the generated headers are auto-included by the build system — do not `#include` them yourself.

**1. CircularBuffer → DataflowBuffer.** The object-type swap. The canonical FIFO methods (`reserve_back` / `push_back` / `wait_front` / `pop_front`) map **1:1**, names unchanged. Variable-name updates are limited to following the API rename (`cb_*` → `dfb_*`), to keep the kernel readable — don't rename for any other reason.

The full CB-method → DFB-method mapping is the **[CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md)** — canonical FIFO, size / layout queries, tile / format metadata getters, public cursor peeks, scalar tile reads, and cursor surgery. It is **authoritative**: consult it for anything past the canonical FIFO calls, and **do not improvise a DFB spelling**. Two notes on how *this* port uses it:
- **Transfers stay as-is** — minimal-diff wins over the whitelist's preference. The whitelist prefers rewriting DM transfers to the Device 2.0 `Noc` APIs (`noc.h`) and marks the leftover-peek idiom "❌ avoid." That preference is a **Device 2.0 cleanup, out of scope here.** Keep the existing transfer idiom: a public `get_*_ptr` peek feeding the kernel's existing `noc_async_*` call is fine — the whitelist itself calls it "cleanup debt, not evil." Rewriting transfers enlarges the diff and the outcome variability this recipe exists to bound.
- **Cursor surgery is whitelisted.** A kernel that mutates FIFO pointers directly — `get_local_cb_interface(cb).fifo_wr_ptr = …`, rewind / jump / hold-wr — maps to the DFB `evil_set_*` setters (whitelist §D). Snapshot first with a public `get_*_ptr` peek if the pattern needs it. This is a sanctioned WH/BH kernel change; **never** leave a raw `LocalCBInterface` field write in place.

> ⚠ **This rule covers the _plain_ CircularBuffer only — a `GlobalCircularBuffer` does NOT become a `DataflowBuffer`.** A GlobalCircularBuffer is a *user-managed* buffer (its lifetime outlives the Program). The legacy codebase pervasively calls it a *"remote CB"* — `global_cb` parameters, a `CircularBufferConfig` named `remote_cb_config`, the `CircularBufferConfig::remote_index()` method, the `remote_circular_buffer.h` device header. Its Metal 2.0 analog is the user-managed **`GlobalDataflowBuffer`**: the split is by *lifetime* (GlobalCB → GlobalDFB), **not** by the word "remote." `GlobalDataflowBuffer` is **not implemented today**, so a factory that uses a GlobalCircularBuffer is **blocked**. Two traps to avoid:
> - **Don't** map it onto `DataflowBuffer` on the strength of "CircularBuffer → DataflowBuffer" above.
> - **Don't** reach for the **`CrossNodeDataflowBuffer`** stub because the legacy name says *"remote."* `CrossNodeDataflowBuffer` (the cross-node DFB sketched in `dataflow_buffer_spec.hpp`) is an *ephemeral* buffer split from `DataflowBuffer` on the *locality* axis (same-node vs. cross-node). It has **no legacy analog** — legacy cross-node dataflow was hand-rolled NoC + semaphores — and is **never a port target**.
>
> A GlobalCircularBuffer should have been blocked upstream by the [audit's GlobalCircularBuffer entry](../audit/metal2_audit.md#globalcircularbuffer--unsupported). If one reaches you anyway, capitulate on that factory ([§When the discipline doesn't fit](#when-the-discipline-doesnt-fit)) — the op's other factories may still be portable.

> **A few CB APIs have no `DataflowBuffer` twin at all** — notably `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` (whitelist §D). A kernel that calls one **cannot be ported today.** Do not work around it: stop, note it prominently in the port report, and capitulate on that factory ([§When the discipline doesn't fit](#when-the-discipline-doesnt-fit)) — the op's other factories may still be portable.

> A few `CircularBuffer` methods have no `DataflowBuffer` twin — the pointer-selection wrappers (`AddrSelector`, `CircularBufferView`, `use<AddrSelector::READ_PTR>(cb)`). These don't port across as methods; instead the wrapper *drops*, because a bare `DataflowBuffer` used as a NoC source/destination is already pointer-sourced (e.g. `noc.async_write(use<…READ_PTR>(cb_out), …)` becomes `noc.async_write(dfb_out, …)`). When a CB-only method has no DFB equivalent, expect the DFB to make the wrapper unnecessary rather than porting it.

This transition is **total**. Post-port, *no* `CircularBuffer` references survive — neither on the kernel side (the rule above) nor on the host side (where `CBDescriptor`, `.cbs`, and any other legacy CB-API references are replaced by `DataflowBufferSpec` and friends per [§Construct paired spec + run-args](#construct-paired-spec--run-args)). Sweep both sides: stale comments referencing CB, unused `#include`s, helper functions named `MakeCB*`, dead-code paths still building `CBDescriptor` — all gone. A grep for `CircularBuffer` and `CBDescriptor` across the op directory at the end of the port should return zero hits in code (only legacy-comparison artifacts in the port report, if any).

```cpp
// Legacy:
CircularBuffer cb_in(cb_in_idx);
cb_in.wait_front(1);
cb_in.pop_front(1);

// Metal 2.0:
DataflowBuffer dfb_in(dfb::in);
dfb_in.wait_front(1);
dfb_in.pop_front(1);
```

**2. CB id → DFBAccessor at LLK / kernel-lib call sites.** Magic-number CB indices (some older kernels hardcoded these) and `uint32_t cb_*_idx` CTA reads are replaced by the `dfb::<name>` handle. The accessor's implicit conversion to `uint32_t` lets it flow into LLK primitives (`reduce_init`, `pack_tile`, `matmul_init`, etc.) and kernel-library helpers expecting a CB id, without extraction or wrapping.

> A CB index carried by a *named* CTA (`get_named_compile_time_arg_val("cb_in")`) converts the same way — to `dfb::cb_in`, **not** to `get_arg(args::cb_in)`. Whether the legacy CB index arrived positionally or by name, it becomes a DFB binding, never a named argument (rule 4). Named args are for non-CB scalars.

```cpp
// Legacy:
constexpr uint32_t cb_in_idx  = get_compile_time_arg_val(0);
constexpr uint32_t cb_out_idx = get_compile_time_arg_val(1);
reduce_init<...>(cb_in_idx, cb_scale_idx, cb_out_idx);
pack_tile(0, cb_out_idx);

// Metal 2.0:
reduce_init<...>(dfb::in, dfb::scale, dfb::out);
pack_tile(0, dfb::out);
```

> **The `dfb::name → uint32_t` conversion is a decoupling shim, not the default.** It exists to bridge a named handle into call sites that aren't on Metal 2.0 — LLK primitives and shared / other-op helpers (*escapes*) that can't all port at once. Where Metal 2.0 offers a native mechanism, prefer it: e.g. tile/format metadata now comes off the `DataflowBuffer` object directly (rule 7), so query the object rather than passing `dfb::name` into a legacy `get_*(cb_id)` helper.

**3. Resource construction from named tokens.** Every kernel-side resource is constructed from its named binding token:

```cpp
DataflowBuffer  dfb_in(dfb::in);
TensorAccessor  input(tensor::input);
Semaphore       done(sem::done);
```

For `TensorAccessor` specifically, this *replaces* the legacy multi-step construction dance. The host-side binding mechanism now packs the layout metadata into the kernel's compile-time args at program creation and auto-injects the per-enqueue base address — work the kernel used to do explicitly via `TensorAccessorArgs<N>()` (with manual offset chaining) and a buffer-address RTA read. The kernel-side rewrite collapses to the one-line construction:

```cpp
// Legacy:
constexpr auto input_args = TensorAccessorArgs<N>();    // N = preceding CTA count
constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
uint32_t input_addr = get_arg_val<uint32_t>(0);
uint32_t index_addr = get_arg_val<uint32_t>(1);
auto input = TensorAccessor(input_args, input_addr);
auto index = TensorAccessor(index_args, index_addr);

// Metal 2.0:
auto input = TensorAccessor(tensor::input);
auto index = TensorAccessor(tensor::index);
```

If positional RTAs survive after the buffer-address RTA goes away (uncommon — most ports also convert all RTAs to named per rule 4), re-index them.

> *Third (page-size) argument drops.* Some legacy accessors pass an explicit page size as a **third** constructor argument — `TensorAccessor(args, addr, page_size)`, where `page_size` is a CTA or RTA value. The Metal 2.0 form takes no such argument: the binding token supplies the correct aligned page size automatically, so the third argument simply falls away in the collapse to `TensorAccessor(tensor::name)` (its host-side CTA/RTA emission drops with it — see [Dropped Plumbing](#dropped-plumbing)). **Don't preserve or re-thread it** — carrying the page size forward as a named arg recreates exactly the plumbing the binding model removes. The audit has already confirmed that every third-argument site reaching you is safe to drop, and lists the sites in the brief; a site whose page size could *not* be dropped never arrives (the audit gates it). A listed site may *additionally* carry a `TensorParameter` relaxation — the brief flags that separately; it is distinct port work, not part of the third-argument drop.

> *Boundary note:* `dfb::name` implicitly converts to `uint32_t`; `sem::name` and `tensor::name` do not. The recipe assumes no out-of-op call site requires passing a `sem::` or `tensor::` handle — see [§Read this first](#read-this-first). If you encounter one, that's an assumption violation; stop and document per the off-ramp below.

**4. Named arguments throughout.** Compile-time, runtime, and common runtime arguments all become named via `get_arg(args::<name>)`. The CTA / RTA / CRTA distinction is a host-only concern; the kernel uses one uniform retrieval syntax. Pick names that match the variables they were going to be assigned to.

```cpp
// Legacy:
constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr uint32_t num_pages = get_compile_time_arg_val(1);
uint32_t start_page          = get_arg_val<uint32_t>(0);
uint32_t bank_id             = get_common_arg_val<uint32_t>(0);

// Metal 2.0:
constexpr auto page_size = get_arg(args::page_size);  // CTA
constexpr auto num_pages = get_arg(args::num_pages);  // CTA
auto start_page          = get_arg(args::start_page); // RTA
const auto bank_id       = get_arg(args::bank_id);    // CRTA
```

Varargs (`get_vararg(i)`) only when a subset of arguments is *genuinely dynamic in kernel code* — i.e. retrieved in a loop where `i` is a runtime variable (the canonical case: an N-dimensional shape gated on a CTA-bound `rank`). When each argument is referenced by a constant index, the named form is clearer on both sides. **Report any retained varargs use in the port report.**

**5. No raw pointers in arguments.** Raw pointers — typically buffer base addresses — must not be passed as compile-time, runtime, common runtime, or vararg arguments. The binding mechanism handles base addresses for tensors automatically; if a kernel *genuinely* needs a raw pointer (a **Case 2** binding — the legacy kernel used a raw base address with explicit arithmetic), get it from a `TensorAccessor` (zero overhead):

```cpp
// Legacy:
uint32_t input_addr = get_arg_val<uint32_t>(0);
// ... explicit address arithmetic on input_addr ...

// Metal 2.0:
auto input = TensorAccessor(tensor::input);
uint32_t input_addr = input.get_bank_base_address(bank_id);  // when truly needed
```

Most ports don't reach for the raw pointer at all — `TensorAccessor`'s normal page-access methods are the standard path (**Case 1** bindings, the common case; they never touch the bridge). `get_bank_base_address` is the escape hatch for **Case 2** bindings — a kernel that used a raw base address with explicit arithmetic. The binding still flows through the typed channel; only the base pointer is extracted here, and the raw arithmetic is left unchanged (no conversion to `TensorAccessor` iteration).

> *Compute kernels — Case 2 is blocked, not an escape valve.* The `get_bank_base_address` bridge lives on `TensorAccessor`, and **a compute kernel cannot bind a `TensorAccessor` today** — the accessor's codegen pulls in the NOC dataflow API, absent from a compute (TRISC) build. So a compute kernel that genuinely needs a raw L1 base address has no bridge, and **the port is blocked**: stop and report it, routing the binding to the compute-kernel `TensorBinding` fix. Do **not** smuggle the address through a CRTA/RTA — even from an **op-owned** tensor, where the address is stable and it would be *strictly* correct, this is a path we deliberately do not take (it normalizes the very pattern Metal 2.0 exists to remove). The one legitimate alternative: if the kernel's access fits a **borrowed-memory DFB** — it reads the data through a `dfb::name` (`get_read_ptr()`), not raw arithmetic — use that. A borrowed DFB is a real framework-managed binding (its backing address refreshes on every cache hit), not a smuggle, and it sidesteps the block. If neither an accessor nor a borrowed DFB fits, the port stops here.

> *Host-computed base-pointer offset — abort the port.* When the legacy host pre-composed a base pointer with a host-side-computed offset (`tensor.buffer()->address() + compute_offset(attrs)`) and passed the result through an RTA, there is **no** Metal 2.0 path: a `TensorBinding` auto-captures the base address but provides no offset hook, and adding the `+ offset` arithmetic kernel-side is off-whitelist (it relocates host computation into the kernel — exactly what the whitelist forbids). **Prominently document the site in the port report and abort the port** — capitulate per [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit). This is an op-owner pre-port fix, now gated by the audit's [Offset base pointers](../audit/metal2_audit.md#offset-base-pointers) gate; a cleared audit means none remain.

**6. Conditionally bound resources (DFB, tensor, semaphore).** A binding declared conditionally on the host (omitted from the kernel's bindings when a path isn't taken) requires kernel-side coordination:

- The condition that selects the binding moves from a CTA to a kernel-side `#define` (emitted by the host via `KernelSpec::compiler_options.defines`).
- `#ifdef`-gate the `constexpr` alias of the binding token at file scope.
- `#ifdef`-gate any expression — including file-scope ternaries — that references the alias.

```cpp
// Legacy:
constexpr uint32_t cb_fusion_idx = get_compile_time_arg_val(2);
constexpr bool fuse_pre_add      = get_compile_time_arg_val(3);
constexpr uint32_t cb_x = fuse_pre_add ? cb_fusion_idx : cb_out_idx;

// Metal 2.0:
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_fusion = dfb::fusion;
#endif
#ifdef FUSE_PRE_ADD
constexpr uint32_t cb_x = cb_fusion;
#else
constexpr uint32_t cb_x = cb_out;
#endif
```

The `#ifdef` runs at the preprocessor stage, before the C++ compiler sees the code — so `dfb::fusion` never enters name lookup in the unfused build, and the conditionally-bound resource is honored end-to-end. See [patterns catalog — Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings) for the deeper rationale.

**7. DFB metadata via the object, not cb-id free functions or JIT arrays.** Compile-time tile / format metadata the kernel used to read by cb id — free functions (`get_tile_size(cb_id)`, `get_dataformat(cb_id)`) or `chlkc_descriptors.h` array lookups (`pack_tile_size[cb_id]`, `unpack_src_format[cb_id]`, …) — is now a **member getter** on the `DataflowBuffer` object: `get_tile_size(cb_id)` → `dfb.get_tile_size()`, and so on. The cb id is gone in Metal 2.0, so these lines must be rewritten — query the object; don't extract `.id` to keep the legacy form. The complete legacy-array / free-helper → DFB-getter table (metadata **and** size / layout queries) is the [CB→DFB API whitelist §A / §B](../shared/cb_dfb_api_whitelist.md) — use it as the lookup rather than guessing a getter name.

**8. Comments — preserve.** The kernel's self-documentation is load-bearing. Slight tweaks to align an existing comment with the line you're forced to change are fine; **deletion is not.** When in doubt, err on the side of preserving information. **A previous porter deleted huge blocks of highly relevant comments while adding `#ifdef`s** — do not repeat that.

The temptation is strongest when adding `#ifdef` blocks — the new structure may feel like it "obviates" an old explanatory comment that sat above the conditional. It doesn't. The explanation of *why* the conditional exists is still valuable; the change in HOW it's gated doesn't retire the WHY. Keep the comments.

**9. No edits to kernel code outside the op's directory.** Shared kernels (under `ttnn/cpp/ttnn/kernel_lib/`, framework primitive locations, or anywhere outside this op's own top-level directory) are vetted for Metal 2.0 compatibility *before* op porting begins. If you find you need to edit one to complete the port, that's a signal the vetting missed something — and the most valuable thing you can do is surface that fact, not bundle the fix. Document in the port report:

- The shared kernel file (path).
- The function or construct that needed change.
- What Metal 2.0–side change would have been needed (named-arg conversion, `dfb::` handle plumbed through a uint32_t parameter, etc.).

The fix belongs in a separate PR, owned by the team responsible for the shared kernel. **Do not bundle external-kernel changes into the port PR.** If the implicit conversion `DFBAccessor → uint32_t` is enough to make a call site work without modifying the callee, that's fine — pass the handle and move on. But if you'd need to edit the callee, stop.

For shared kernels that live *inside* `ttnn/cpp/ttnn/operations` but are used by multiple ops (a kernel that several siblings rely on), see [patterns catalog — Modifying a shared dataflow kernel](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel) for the fork-vs-in-place decision.

### When the discipline doesn't fit

If you reach a point where the changes the port would require fall outside the host-side scope or the kernel-side whitelist above, **stop.** Don't improvise. Don't introduce a "clever" workaround. Don't expand the rules by one "just this once." Capitulate gracefully.

**This is not a failed port — it is a *successful capitulation*.** Same success tier as a clean port. The framework's calibration depends on knowing where its assumptions break, and a grounded capitulation gives the maintainers a clear picture of what to refine before the next porting wave. A creative workaround buried in a diff gives them noise; a clean writeup gives them signal.

**Size is not a fit gap — do not capitulate on it.** This off-ramp is for when Metal 2.0 genuinely *cannot express* something, or the change reaches outside the op's directory — a real capability or scope limit. It is **not** for a port that is merely large. The workflow runs you as a fresh primary session on a 1M-context model; a long stretch with no green build until the last source flips is the *expected* shape of an atomic multi-source factory (see [Construct](#construct-paired-spec--run-args)), not a stop signal. A factory that genuinely overruns even a 1M primary budget is a *vehicle* limit, not a capability gap: hand the **whole** factory to another fresh primary instance to continue from `METAL2_PORT_PLAN.md` — never leave a half-converted factory, which does not build and is not a deliverable. And before recording any capitulation, re-read the [patterns catalog](../shared/port_patterns.md): the construct you think is unsupported is often covered there.

Record in the port report's *Successful failure* section:

- The op (path, factory).
- The file and the specific lines / constructs that needed to change.
- A short explanation of *why* mechanical conversion failed — what the code was doing that didn't fit a binding-token replacement, a comment-preserving `#ifdef`, the host-side-scope boundary, or whichever rule was the sticking point.
- (If you can sketch it) what the off-rules change would have been, accurate enough that a maintainer can evaluate the gap.

**The stop-signal we see most often:** *if you find yourself reaching past the op's own directory to make kernel changes, that's the signal.* The op shouldn't have made it to porting if its kernel needs out-of-dir changes — flag this prominently in the report. Other signals will surface as porting experience accumulates; this list will grow.

---

## Construct paired spec + run-args

*Mechanical translation from the plan. Build each resource's spec entry and its run-args entry together. Every edit in this step — host-side and kernel-side — is governed by the [§Scope discipline](#scope-discipline) section above.*

**Expect a long stretch with no green build — that's normal.** When the factory runtime-selects multiple kernel sources, the conversion is *atomic*: the factory and every source it can bind flip together, and nothing compiles until they all do (see [the atomic-unit note](#read-this-first)). For a large multi-source factory that can mean converting thousands of lines across a dozen files before the first successful build. That is the expected shape of the work, not a sign the approach is wrong — don't read the missing green build as a stop signal, and don't reach for a workaround to force an early partial compile. Convert the whole unit per the plan, then build.

**Operating principle**: prefer designated initializers. Metal 2.0 was designed to support them and the spec reads as data, not as procedure. This holds for resource bindings specifically: write `DFBBinding`s in the full designated-initializer form —

```cpp
DFBBinding{
    .dfb_spec_name = INPUT,
    .accessor_name = "in0",
    .endpoint_type = DFBEndpointType::CONSUMER,
}
```

— **not** the `ProducerOf(...)` / `ConsumerOf(...)` / `StridedConsumerOf(...)` / `AllConsumerOf(...)` convenience factories. Those factories exist to trim boilerplate in unit tests, but inside a block of designated-initializer specs a call expression reads as an outlier, and it hides the `endpoint_type` and `access_pattern` that the full form states inline. Keep the spec uniform and the roles explicit.

**`Table`s are maps, not vectors.** Several spec fields that look list-like — `compile_time_args`, `runtime_arg_values` / `common_runtime_arg_values`, `unpack_modes`, `defines`, `tensor_args` — are `Table` (a hash-friendly map type), *not* `std::vector`. `Table` has **no `push_back` and no iterator-pair constructor**; building one the way you'd build a vector won't compile. Use brace-init `{{key, value}, …}`, `insert` / `emplace`, `operator[]`, or the single-argument range constructor `Table(existing_map)` (e.g. to convert a legacy `std::map` of defines). When a `Table` must be built conditionally, declare it and `insert`/`emplace` into it — don't reach for `push_back`.

For each resource type, construct the spec entry and its run-args entry as a pair. The order emerges naturally from the op's existing structure (reader / writer / compute order, tensor → DFB → semaphore precedence); the recipe does not prescribe a fixed sequence.

- **`KernelSpec` ↔ `KernelRunArgs`.** For each planned `KernelSpec`, build the schema (`compile_time_args`, `runtime_arg_schema`, `dfb_bindings`, `tensor_bindings`, `semaphore_bindings`, `hw_config` — the last has its own subsection, [Hardware configuration](#hardware-configuration), because it is the easiest field to break silently); alongside, build the corresponding `KernelRunArgs` entry (`runtime_arg_values` and `common_runtime_arg_values`). If the kernel has no RTAs, the run-args entry may be omitted entirely.

  `runtime_arg_values` is keyed **name-first, then node** (`name → node → value`). Legacy factories almost always set RTAs **node-first** (a loop over cores, values assigned per core). To bridge that without hand-inverting the loop, use `AddRuntimeArgsForNode` (from `program_run_args.hpp`) — keep the legacy per-node loop as-is and let the helper transpose into the name-first table:

  ```cpp
  KernelRunArgs kra{.kernel = READER};
  for (const auto& core : cores) {
      AddRuntimeArgsForNode(kra.runtime_arg_values, core,
          {{"start_page", start_page_of(core)}, {"num_pages", num_pages_of(core)}});
  }
  ```

  For a single-node kernel, `MakeRuntimeArgsForSingleNode(node, {{"num_pages", n}})` builds the table inline. **Use whichever form makes the port easiest** — the helper for a node-first loop (the common case), or the native name-first table written directly where that is simpler. What you should *not* do is **re-architect the legacy loop into name-first form as part of this port**: inverting the loop nesting buys nothing for the port and adds transposition-error risk on top of an already-large host rewrite. A name-first restructure is a worthwhile *separate* cleanup, not port work — as is the tidy-up it exposes: an RTA set to the **same value on every node** is really a CRTA (`common_runtime_arg_values`). Note such a case for that later pass; do **not** convert it here (RTA→CRTA changes dispatch semantics). See the [migration guide](../shared/migration_guide.md) for the `runtime_arg_values` shape.
- **`DataflowBufferSpec`.** Build with `entry_size`, `num_entries`, `data_format_metadata`, and `tile_format_metadata` **copied from the legacy CB's `format_descriptors[i].tile`** when that field was set (see [migration guide for the rationale](../shared/migration_guide.md#dataflowbufferspec)). No placement field — placement is derived from the kernel bindings. **`entry_size` and `num_entries` are set once at spec construction** — compute them from anything the spec construction has access to (input tensor shapes / shard specs, fields on `operation_attributes`). A **GlobalCircularBuffer** is likewise *not* a `DataflowBufferSpec` target — it maps to the unimplemented user-managed `GlobalDataflowBuffer`, so the [audit's GlobalCircularBuffer entry](../audit/metal2_audit.md#globalcircularbuffer--unsupported) blocks it; if one reaches construction, don't build a `DataflowBufferSpec` for it and don't substitute the `CrossNodeDataflowBuffer` stub — capitulate on the factory per [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit). For borrowed-memory DFBs, set `borrowed_from = <tensor_parameter_name>` naming the `TensorParameter` whose buffer backs the DFB; the backing L1 address resolves at runtime from the corresponding `TensorArgument`, so no `dfb_run_overrides` entry is needed for the backing memory. For aliased DFBs (legacy aliased CBs), set each spec's `advanced_options.alias_with` to mutually name the other members of the alias group — see [Pattern: Aliased DFBs](../shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs). For a **sync-free or single-ended CB** — one that can't present a FIFO producer *and* a FIFO consumer on distinct kernels (a sync-free address-source CB with no FIFO ops, or a single-ended one like the packer writing into resident output) — build a normal `DataflowBufferSpec` and **self-loop it**: bind the touching kernel as both PRODUCER and CONSUMER. This is legal on Gen1 for **compute and DM kernels alike** — the DFB lowers to a plain circular buffer that one RISC both fills and drains, so runtime behavior is identical to the legacy CB. A DM self-loop is legal on Gen1 — it is Quasar-uplift's concern (rejected on Gen2), not a Gen1 blocker. See [Pattern: Sync-free and single-ended CBs → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb). For a **dead CB** the brief flagged (zero endpoints), build **no** spec — drop the allocation and any dead CTA carrying its index, recording each with `file:line` in the report (a dead CB has no behavior, so the drop is zero-functional-change; a bindingless DFB is rejected by the validator regardless). For a **multi-binding CB** the brief flagged (≥2 producers and/or ≥2 consumers on a node — a hidden co-filler or a multi-reader), build the normal `DataflowBufferSpec` and set `advanced_options.allow_instance_multi_binding = true`: the legacy co-fill / multi-read still happens byte-for-byte, and the flag self-documents the Quasar debt (no separate tracking — it is a hard error on Gen2, exactly where the uplift must refactor).
- **`SemaphoreSpec`.** Build with `target_nodes`. (Semaphores have no per-execution counterpart on `ProgramRunArgs`.)
- **`TensorParameter` ↔ `TensorArgument`.** Declare each tensor as a `TensorParameter` (using `<tensor>.tensor_spec()`); alongside, add the corresponding `TensorArgument` to `ProgramRunArgs::tensor_args`. Here `<tensor>` is the device-resident `ttnn::Tensor` arriving via `tensor_args` / `tensor_return_value`; the `TensorArgument` must reference that same tensor (the framework matches it back by `MeshTensor` identity, so a copy fails) — see the [TTNN integration doc — Extracting the tensor](../shared/ttnn_factory.md#extracting-the-tensor). Build the `TensorArgument` by passing the `MeshTensor` **directly** (`{INPUT, input}`); do **not** wrap it in `std::cref` / `std::reference_wrapper<const MeshTensor>`. `TensorArgument` is a `reference_wrapper`-backed variant and the implicit conversion binds the reference for you — the explicit `std::cref` adds noise without changing behavior.
- **Op-owned tensors — keep the legacy construction, release the `MeshTensor`.** Only when the legacy factory allocated its own device tensors beyond the op's io (audit Q1) — the tensors the legacy `WorkloadDescriptor` carried as `WorkloadBuffer`s (sliding-window config / reader-index LUTs and the like). Most ports have none. In the legacy world the *whole* `ttnn::Tensor` had to be kept alive (a `std::shared_ptr<Tensor>` owner) because `~Tensor` force-deallocates the device memory regardless of who else holds the buffer; Metal 2.0 replaces that with an owning `MeshTensor` parked in `op_owned_tensors`.

  **Keep the legacy tensor construction verbatim** — including any host-data population (`construct_on_host_config_tensor` → `move_config_tensor_to_device`, `create_device_tensor`, `to_device`). Only the tail changes: move the owning `MeshTensor` *out* of the built tensor with `release_mesh_tensor()` and into the artifact, then bind it like an io tensor.

  ```cpp
  // Legacy (WorkloadDescriptor):
  //   auto owner = std::make_shared<Tensor>(std::move(config_tensor));
  //   workload_descriptor.buffers.push_back({owner, owner->buffer()});

  // Metal 2.0 — the construction of `config_tensor` above this line is UNCHANGED
  // (build host data → move_config_tensor_to_device / to_device):
  std::vector<tt::tt_metal::MeshTensor> op_owned;
  op_owned.reserve(n);    // n = number of op-owned tensors (halo: 4 — pad0/pad1/gather0/gather1).
                          // MANDATORY before any bind: pins element addresses so the refs below stay valid.

  // Repeat this block for EACH op-owned tensor. Construction of `config_tensor` is the
  // UNCHANGED legacy build (host data -> move_config_tensor_to_device / to_device).
  op_owned.push_back(config_tensor.device_storage().release_mesh_tensor());  // owning move-out
  const auto& owned = op_owned.back();                  // THIS tensor's element — bind against it, never reuse
  spec.tensor_parameters.push_back(
      TensorParameter{.unique_id = CONFIG, .spec = owned.tensor_spec()});
  run_args.tensor_args.emplace(CONFIG, owned);          // direct MeshTensor ref; no std::cref

  // ... at assembly: ProgramArtifacts{ .spec = ..., .run_params = ..., .op_owned_tensors = std::move(op_owned) }
  ```

  `release_mesh_tensor()` (on `DeviceStorage`) transfers the `MeshTensor` out and leaves the source `Tensor` deallocated-but-harmless — the later `~Tensor` finds nothing to free. The released tensor's `TensorSpec` and mesh-coordinate coverage ride along inside the `MeshTensor`, already correct: no hand-built spec, no topology assumption, and exactly the coverage the legacy `WorkloadBuffer` already bound across every per-coord program. Two consequences: the source `Tensor` is a **build vehicle only** — discard it after release; and **never release a tensor that is also an io argument or shares storage with one** (a view, a copy) — release moves out the sole-owned allocation. To tell which you have: only release a tensor a construction call produced (`create_device_tensor` / `to_device` / `move_config_tensor_to_device`) — that allocation is sole-owned and safe. If the legacy code instead *derived* the tensor from an io tensor (a reshape / slice / view / clone), it **shares** that allocation — do not release it; route it to the port report as a structural mismatch.

  **Identity footgun.** The framework resolves each `TensorArgument` to its tensor **by pointer identity** against the `op_owned_tensors` elements. Two distinct ways to get this wrong:
    - *Binding a pre-release local* (or any tensor not in the vector) — a runtime `TT_FATAL` ("unowned MeshTensor"). Always bind against the vector element (`op_owned.back()` right after its `push_back`, as above, or `op_owned[i]`), never the local you released from.
    - *Binding several parameters against the **same** element* — e.g. reusing `op_owned.back()` after you have pushed all the tensors, so every binding names the last one. This does **not** trip the fatal (the last element *is* genuinely owned); it ships as silently-wrong numerics. Bind each op-owned tensor against its own element.

    `reserve(n)` up front is **mandatory** whenever there is more than one: without it a later `push_back` reallocates and invalidates the element references you already handed to earlier `TensorArgument`s. With the reservation, element addresses are stable, so the final `std::move` into the artifact and returning by value are both safe.

  **First exercise of this path** — no production op has run op-owned tensors yet (compile-coverage only). conv2d / halo / pool are its shakedown; surface any friction (release ergonomics, binding identity, cache-hit re-patching) rather than trusting it blindly.
- **`WorkUnitSpec`.** Build with `kernels` (by `unique_id`) and `target_nodes`. No per-execution counterpart.

After all resources are built, assemble the `ProgramSpec` (collecting `kernels`, `dataflow_buffers`, `semaphores`, `tensor_parameters`, `work_units`) and the `ProgramRunArgs` (collecting `kernel_run_args`, `tensor_args`). Return them wrapped as `ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)}` from the factory's `create_program_artifacts` method — see the [TTNN integration doc](../shared/ttnn_factory.md#the-metal-20-factory-concept) for the method signature and the cache lifecycle the framework wraps around it. When the op has op-owned tensors, also `std::move` them into the artifact's `op_owned_tensors` field (built and bound during construction, per the op-owned step above); omit the field otherwise — it defaults empty.

**Stop signals**: any urge to —

- Demote a CTA to a runtime arg to make a single `KernelSpec` work where the legacy had multiple. ([Anti-pattern](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).)
- Bind a conditionally-used DFB *unconditionally* on the host to dodge the `if constexpr` name-lookup problem. That wastes L1 and breaks the moment the kernel references the name from a file-scope ternary or similar context. The right shape is conditional host binding + matching `KernelSpec::defines` + kernel-side `#ifdef`-gated alias and uses — see [Pattern: Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings).
- Extract `.id` from a `dfb::name`, or construct a temporary `DataflowBuffer` to retrieve its underlying id. ([Anti-pattern](../shared/port_patterns.md#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites).)
- Pack data into varargs that should be named arguments. ([Caution](../shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary).)
- Thread a buffer address through an RTA because the binding mechanism doesn't fit. *(A **compute** kernel needing a raw L1 base address cannot bind a `TensorAccessor`, so it has no bridge — that does **not** make an RTA acceptable; the port is **blocked** there pending the compute-kernel `TensorBinding` fix, per [kernel-side whitelist rule 5](#kernel-side-whitelist).)*
- Hand-roll a synchronization primitive.
- Modify a kernel's `wait_front` / `pop_front` calls to "balance" a DFB's producer / consumer topology. The host-side spec validator's "every DFB needs ≥1 PRODUCER and ≥1 CONSUMER" check is satisfied by adjusting the *binding* declaration on the host (declare the conditional-side endpoint unconditionally), not by adding consumes/produces inside the kernel. Per-execution DFB state is reinitialized, so a tile produced and never consumed is harmless across enqueues — and, symmetrically, a tile consumed via `wait_front` but deliberately never `pop_front`'d (fill-once, read-many reuse — e.g. a bias buffer re-read every iteration) is intentional reuse, not an unbalanced FIFO to "fix" by adding a `pop`.

If any of these appear in your draft, **stop and report**. The likely cause is a structural decision during planning that should be revisited.

**Exception — Case 2 (raw pointer) bindings.** If the audit classified a `TensorParameter` as **Case 2** (the kernel uses a raw base pointer), the binding still flows through the typed channel; only the *base pointer* is extracted kernel-side via the sanctioned `get_bank_base_address` bridge (see [kernel-side whitelist rule 5](#kernel-side-whitelist)), never threaded through an RTA. This is the **one** carve-out from the "buffer address through an RTA" stop signal above, and it is **data-movement only**: a **compute** kernel cannot bind a `TensorAccessor`, so a Case 2 binding there has no bridge and **blocks the port** (rule 5) rather than falling back to an RTA.

### Hardware configuration

`KernelSpec::hw_config` is a `std::variant<DataMovementHardwareConfig, ComputeHardwareConfig>`, and each of those is itself a variant over a Gen1 (WH/BH) and a Gen2 (Quasar) config. **Everything here is a pure performance / precision setting.** A wrong value almost never fails the build, and usually slips past the op's tests too; it silently shifts the op's perf/precision tradeoff and surfaces much later as a model-level regression that is miserable to trace back to this line. This is the one field where the "syntax-only swap, no semantic change" promise is easy to break by accident, with no safety net but your own attention. So the discipline throughout is: **read the legacy op's *resolved* settings, port them to their exact equivalents, and diff before against after.** The field *names* were deliberately renamed for clarity — carry over the *values*, not the names.

The port targets Gen1. **Build only the Gen1 config, and add no `if (arch == QUASAR)` branch of your own** — the reasons are in [Gen2 is out of scope](#gen2-is-out-of-scope) below.

#### Data movement kernels

Resolve the legacy DM kernel's effective `(processor, noc, noc_mode)` — the *values*, not the constructor spelling. Legacy code reaches the same config three ways: `ReaderDataMovementConfig{}` / `WriterDataMovementConfig{}` (which default the triple), a raw `DataMovementConfig{.processor = …, .noc = …}` that happens to equal a default, or a genuinely custom triple. Decide on the resolved values:

- **Matches the reader or writer default** → use the arch-agnostic TTNN helper (from `ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`). It selects the generation internally and supplies the Gen2 branch for free, so default-case host code needs no arch branch:

  ```cpp
  .hw_config = create_reader_datamovement_config(device->arch()),   // or create_writer_datamovement_config
  ```

  Match on the *values*, not the role name: a kernel whose resolved triple is the reader default takes the *reader* helper even if the op calls it a writer — on Gen1 the helper reproduces that triple byte-for-byte. The two defaults are value-distinct:

  | | processor | noc | noc_mode |
  |---|---|---|---|
  | reader | `RISCV_1` | `NOC_0` | `DM_DEDICATED_NOC` |
  | writer | `RISCV_0` | `NOC_1` | `DM_DEDICATED_NOC` |

  (The Metal 2.0 [migration guide](../shared/migration_guide.md) uses the Metal-layer `CreateReader1xxDataMovementConfig()` / `CreateWriter1xxDataMovementConfig()` for these same defaults — identical on Gen1. The TTNN helper here wraps those and adds generation selection; prefer it for a TTNN port, since it also supplies the Gen2 branch the Gen1-only Metal helpers lack.)

- **Custom** (any field differs from both defaults) → replicate it *exactly* with a Gen1 config, every field copied verbatim from the legacy config:

  ```cpp
  .hw_config = DataMovementGen1Config{.processor = …, .noc = …, .noc_mode = …},
  ```

  Do **not** reach for a helper that is "close." A single flipped NOC is precisely the silent regression this section exists to prevent.

**`noc_mode` is a paired, per-node setting.** A custom `DM_DYNAMIC_NOC` (typically chosen to free a NOC for fabric/CCL traffic) configures shared per-node hardware and must be set *identically on both DM kernels on the node* — it is not a per-kernel property. Port it on both kernels together.

*Where the framework catches you — and where it doesn't.* The spec validator enforces the Gen1 node invariants: the two DM kernels on a node must use **distinct RISC cores**, must **agree on `noc_mode`**, and under `DM_DEDICATED_NOC` must use **distinct NOCs** (two dedicated kernels sharing a NOC hang the device). So a core or NOC *collision* is a loud `TT_FATAL`, not a silent hang. But the validator does not compare your values against the legacy op's — swapping a reader's and writer's NOCs (still distinct) passes validation and regresses silently. That comparison is yours to make.

*One rare trap.* The legacy dedicated-NOC distinctness check had a bug (it ran before the second DM kernel registered), so it did not reliably fire for the common reader+writer pair. If you faithfully replicate a legacy op that pinned **two dedicated-NOC DM kernels to the same NOC**, the Metal 2.0 validator will correctly reject what legacy silently tolerated. That op was genuinely misconfigured — **stop and report** per [§When the discipline doesn't fit](#when-the-discipline-doesnt-fit); do not invent a NOC assignment to get past it (that would change its behavior).

#### Compute kernels

The compute config has **two sources, and you must consult both** — the second is the one the "did the helper cover everything?" instinct misses.

1. **The TTNN `ComputeKernelConfig`** — the op-level knobs. Most ops carry one; translate it with `to_compute_hardware_config(device->arch(), config)` (from `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp`). Like the DM helper, it selects the generation and returns the matching variant. It covers four knobs, two of them with a representation change baked in:

   | legacy `ComputeConfig` | Metal 2.0 (`ComputeGen1Config`) | transform |
   |---|---|---|
   | `math_fidelity` | `fpu_math_fidelity` | 1:1 |
   | `math_approx_mode` (bool) | `sfpu_precision_mode` (`Precision`) | `true` → `Approximate`, `false` → `Precise` |
   | `fp32_dest_acc_en` | `enable_32_bit_dest` | 1:1 |
   | `dst_full_sync_en` | `double_buffer_dest` | **inverted**: `double_buffer_dest = !dst_full_sync_en` |

2. **The legacy metal `ComputeConfig` at the compute kernel's creation site** — two fields the TTNN config never carried, so the helper *structurally cannot* set them. Sweep the legacy `ComputeConfig{}` and set each by hand on the returned Gen1 config (via `std::get<ComputeGen1Config>(compute_hw).<field> = …`):

   - **`bfp_pack_precision_mode`** (legacy `bfp8_pack_precise`). Rare; a clean bool→enum. Gen1-only — Gen2 replaces BFP with MXFP and has no such field:

     | legacy `bfp8_pack_precise` | Metal 2.0 `bfp_pack_precision_mode` |
     |---|---|
     | `false` (default) | `Precision::Approximate` (default) |
     | `true` | `Precision::Precise` |

     If the legacy op left it default, do nothing — the defaults coincide.

   - **`unpack_modes`** (legacy `unpack_to_dest_mode`) — **the dangerous one.** Three things change at once:

     1. **Reindexing.** Legacy is a `vector<UnpackToDestMode>` *indexed by CB id*; Metal 2.0 is a `Table<DFBSpecName, UnpackMode>` *keyed by DFB name*. Map each legacy entry's CB id to its DFB name. (The legacy vector is usually a computed local, not a literal — trace what it resolves to per CB.)
     2. **Value translation — and it flips silently.** `UnpackToDestFp32` → `UnpackMode::UnpackToDest`; `Default` → `UnpackMode::UnpackToSrc`, which you normally express by *omitting* the entry. Reverse the mapping and you have flipped the precision/perf tradeoff with no compile or test signal.
     3. **A newly-required explicit entry.** The Metal 2.0 validator is stricter than legacy: when a compute kernel **consumes a Float32 DFB with `enable_32_bit_dest = true`**, an explicit `unpack_modes` entry is *required* where legacy silently defaulted. Add one — and derive its value from the legacy vector (`Default` → `UnpackToSrc`, `UnpackToDestFp32` → `UnpackToDest`); do not guess. This is **Float32-only for now**; Int32/UInt32 are deliberately not required yet (issue #49936), so don't preemptively add entries there.

     ```cpp
     std::get<ComputeGen1Config>(compute_hw).unpack_modes = {
         {INPUT_A, UnpackMode::UnpackToDest},   // legacy CB entry was UnpackToDestFp32
     };
     ```

   *Where the framework catches you.* Unlike the DM side, several `unpack_modes` mistakes are loud: the validator rejects an entry naming a DFB the kernel doesn't bind, rejects a `UnpackToDest` that can't fit its DFB's format (a 32-bit format into a 16-bit Dest — rejected on every generation; a ≤16-bit format with `UnpackToDest` — rejected on Gen1 as a pure perf loss), and enforces the required-entry rule above. But a *wrong-but-legal* entry, or a wrongly-omitted one that no rule forces, is silent. The loud checks are a partial backstop, not a substitute for the before/after diff.

   (A conditionally-bound DFB's `unpack_modes` entry must be gated on the *same condition as its binding* — the validator rejects a key naming a DFB the kernel doesn't bind. See [Pattern: Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings).)

#### Gen2 is out of scope

Build only the Gen1 config; do not populate a `Gen2Config` or add your own `if (arch == QUASAR)` branch. Two reasons:

- **A custom DM config has no mechanical Gen1→Gen2 mapping.** Gen2 has no processor/NOC concept, so its config isn't derivable from the Gen1 one — authoring it means Quasar-specific judgment (e.g. whether to set `disable_dfb_implicit_sync_for_all`) that you cannot validate on a Gen1 bench.
- **The Gen2-only compute fields are temporary.** `enable_2x_src_register` and `unpack_to_dest_en` are explicitly interim LLK hacks (issue #49445).

In the *default* DM and compute cases the arch-agnostic helpers already emit a correct Gen2 branch for free; anything beyond that is a separate, later Quasar-uplift pass carried out with the right expertise.

---

## Verification

*Build, test, anti-pattern self-audit.*

### Build

Build the ported op's TTNN target and its test binary via the [build/test helper subagent](#use-a-subagent-for-builds-and-tests):

```bash
cmake --build build_Release --target ttnncpp unit_tests_ttnn -j 8
```

(If the op's tests live in a sibling binary — `unit_tests_ttnn_udm`, `unit_tests_ttnn_tensor`, `unit_tests_ttnn_ccl`, etc. — substitute or add it to the target list.)

The helper returns SUCCESS / FAILURE + key errors. On FAILURE, common causes:

- `AllFactoriesValid` `static_assert` fires → a factory satisfies two concepts (likely a stale `cached_program_t` declaration alongside the new `create_program_artifacts`). Audit for missed deletions in the header.
- Unresolved symbol for `override_runtime_arguments` → some code path still calls it. Should only happen for the framework adapter, which doesn't for `MetalV2FactoryConcept` factories. Re-audit.
- Error referencing `metal_v2_artifacts.hpp` (or other framework header) not found → the framework dependency is not on this branch. Stop and report; the framework PR was a precondition for the audit (which should have failed pre-port).
- `kernel_args_generated.h` mentions a name that doesn't exist → host added a named CTA / RTA without the kernel referencing it (or vice versa). Reconcile.
- Linker error `undefined reference to 'dfb::...'` inside a kernel TU → the kernel `#include`s the wrong generated header. `dfb::*` lives in `kernel_bindings_generated.h`, `args::*` in `kernel_args_generated.h`. The only `#include` a ported kernel should add is `experimental/kernel_args.h`; the framework injects both generated headers automatically.

### Run tests

Run the op's correctness tests via the helper. Use the **tests directory the invoker supplied** in [Before you begin](#before-you-begin). Two layers:

```bash
# C++ gtests — fast, fails fast on a broken port
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*<Op>*'

# Python pytests — broader coverage (dtype/layout sweeps, program-cache behavior)
pytest <invoker-supplied-tests-dir> -x -v
```

Recommended order: gtests first (faster, exercises the C++ op directly), then pytests once gtests are green. If the op has a UDM/specialized variant, its gtests live in a sibling binary (`unit_tests_ttnn_udm`, etc.) — the recipe's worked example in [`workspace_setup.md`](../shared/workspace_setup.md) shows the pattern.

> **A selected-but-unconverted kernel path can crash the whole pytest *session*, not just fail it.** While converting kernels path-by-path, a test that selects a not-yet-converted kernel hits that kernel's positional `get_compile_time_arg_val(0)` (the host now emits only named args), which `static_assert`s at JIT; the resulting `TT_FATAL` can segfault the Python process during traceback rendering (exit 139) and take down the entire pytest run rather than reporting a clean failure. Exclude not-yet-converted paths with `-k` until you convert them. This is a known tooling issue — not a port error, and not a card hang (no `0xdeadc0de`).

All tests passing pre-conversion should continue to pass post-conversion. If a previously-passing test now fails, **stop and report** — likely cause is a structural error in the spec that compiled but failed at `MakeProgramFromSpec` validation, or an incorrect tensor-arg / runtime-arg layout.

If compilation passes but the test fails with a `TT_FATAL` from `program_spec.cpp` or `program_run_args.cpp`, the [patterns catalog](../shared/port_patterns.md) has entries for the most common failure modes (DFB binding multiplicity mismatches, missing kernel run-args entries, etc.). Cross-reference the error message against the catalog.

For symptom-organized lookup (errors whose fix isn't obvious from the message text), see the [migration guide's troubleshooting table](../shared/migration_guide.md#cryptic-error--likely-cause).

**Custom `compute_program_hash` failure mode.** The port should already have deleted any custom `compute_program_hash` (reverting to the default) per the [TTNN integration doc](../shared/ttnn_factory.md#1-delete-a-custom-compute_program_hash) — it's proactive port work, not a wait-and-see. The signature of one that survived: `UpdateTensorArgs` `TensorSpec` legality failures on the *second and later* test invocations (program cache hot), not the first. If you see that, find the surviving custom hash and delete it; do not patch it to include `TensorSpec`. Confirm the deletion is recorded in `METAL2_PORT_REPORT.md`.

### Anti-pattern self-audit

Scan the ported code against this checklist. Each item is a Metal 2.0 design-intent failure to look for; the [patterns catalog](../shared/port_patterns.md) has the full discussion of each.

- [ ] **No `tensor.buffer()->address()` survived.** Search the factory `.cpp` for this string; if present, the corresponding tensor needs a `TensorBinding` instead.
- [ ] **No magic-number CB indices in CTAs.** Search `compile_time_args` for values that are CB indices (typically small integers or `CBIndex::c_*`); if found, the value should come from a `DFBBinding` instead.
- [ ] **No `TensorAccessorArgs<N>()` survived in any ported kernel.** Search for this; if present, the kernel needs `TensorAccessor(tensor::name)` instead.
- [ ] **Conditional DFB bindings follow [Pattern: Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings).** For each conditionally-used DFB: the host conditionally binds it; `KernelSpec::defines` carries the matching preprocessor flag; the kernel `#ifdef`-gates both the constexpr alias of the DFB name and every expression referencing it. No unconditional bindings introduced as a workaround.
- [ ] **No `.id` extraction at LLK call sites.** Search for `.id` on `dfb::` handles; if present, pass `dfb::name` directly.
- [ ] **No CTA→RTA demotion in compute kernels.** If a per-group dimension was moved from CTA to RTA in the port, the structural decision is wrong; revisit planning.
- [ ] **All CTAs are named.** Search the factory for positional `compile_time_args = {...}`; should be `compile_time_args = {{name, value}, ...}` only.
- [ ] **No new varargs unless the kernel reads them in a loop.** Check `num_runtime_varargs` use; if the kernel reads `get_vararg(0)`, `get_vararg(1)`, ..., the named form is the right answer.
- [ ] **Every `hw_config` reproduces the legacy op's resolved values.** For each kernel, diff the ported config against the legacy one: DM `(processor, noc, noc_mode)`, and the compute knobs — *including* the two fields the helper does not cover, `bfp_pack_precision_mode` and `unpack_modes`, swept from the legacy `ComputeConfig`. These are silent perf/precision settings with no test net; see [Hardware configuration](#hardware-configuration).

If any checklist item fails, return to planning / construction to fix. Do not paper over with kernel-side modifications.

---

## Capture the port report

**Open `METAL2_PORT_REPORT.md` at the start of the port and update as you go** — do not write it retrospectively. Add a one-line note the moment something surprises you, even rough. Polish entries at the end; capture in the moment. A friction note written when it happened is worth more than a paragraph reconstructed two hours later.

The final report is committed when the port reaches its stopping point — whether that's "all factories ported and tests pass" or "stuck on issue X and cannot proceed" — alongside `METAL2_PORT_BRIEF.md`, `METAL2_PREPORT_AUDIT.md`, and `METAL2_PORT_PLAN.md` in the op directory. The report captures what happened during the port: things that need handoff to other teams, things the docs got right, things the docs missed, and things the next porter or doc maintainer should know.

The report is read by the kernel-lib / API owners (for handoff points), by the doc maintainers (for friction-driven evolution of the audit / recipe / catalog / migration guide), and by future porters of related ops.

Structure the report with the following sections. Each section may be empty (write "none" with a one-line note); do not omit sections.

### Provenance

Record which version of the recipe docs this port ran against, so a reviewer can reconstruct the exact guidance you followed. Run this from your checkout root and paste the output verbatim — do not hand-edit it:

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

- **Recipe docs (this port):** `<hash> <date> <subject>`
- **Audit docs (inherited):** `<the provenance line copied from METAL2_PORT_BRIEF.md>`

If the command prints nothing, the recipe docs are not from a tracked doc-branch checkout, so the version can't be pinned — record that fact instead.

### TTNN ProgramFactory

Confirms the realized factory shape against the audit's decision, and records the device-op-class edits the port forced. Filled out per the [TTNN integration doc — Port report deliverable](../shared/ttnn_factory.md#port-report-deliverable-porter-facing): concept realized, custom-hash deletion (file:line or none), pybind entry points removed (or none), and open items (relaxation candidates, capabilities not yet on main the op would benefit from, friction with the concept fit).

If the port stayed on the default concept with no device-op edits, this section is short — that's the success case.

### Handoff points

Escalations to teams outside the porter's scope. Each entry is something the porter cannot fix from within the op directory and that should not be papered over.

Includes (not exhaustive):

- **Boundary-rule assumption violations.** A call site outside the op directory that required `sem::name` or `tensor::name` (per the [scope boundary](#read-this-first)). Cite the file:line, the callee, and the named handle that the call site demands. Tagged "API: requires implicit conversion / refactor."
- **Kernel-lib gaps.** Cases where a shared kernel-lib helper or LLK is incompatible with Metal 2.0 binding semantics in a way the porter cannot work around. Cite the helper, the call site, the specific incompatibility.
- **Framework gaps.** Audit-time entries that were flagged (e.g., an UNSUPPORTED feature) and that bit during the port. Cite the audit entry, what the port needed, and the workaround (if any) you adopted.
- **Removed pybind surface.** Any pybind line(s) deleted because the port made a legacy factory entry point (e.g., `create_program_descriptor`) vanish. Cite the pybind file path, the function name(s) removed, and a one-line description of what the function was for. Tagged "API surface: removed entry point." This is a *user-visible* surface change — downstream Python consumers (tests, notebooks, internal tooling) need to find this entry to update their callers. See [Pattern: Removing pybound legacy factory entry points](../shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points).

Each handoff entry should be writable as a standalone ticket. The porter is the original reporter; the listed team is the owner.

### Successes

Places where the docs steered the port right. Especially valuable when the porter almost did something the catalog warned against and the warning fired correctly — these are the entries that justify keeping a doc section in its current form.

Cite the doc section by name and link; cite the file:line in the ported code where the warning applied. Brief — one short paragraph per entry.

### Friction

What bit during the port and how the docs helped or didn't. Two subcategories:

- **Gaps** — where the docs didn't have an answer, or had a stale answer. Most actionable kind of entry — directly translates to a doc improvement.
- **Confusion** — where the docs were ambiguous, hard to follow, or led to a near-miss before the right path became clear.

Cite the doc section, the file:line in the port, and what the right answer turned out to be.

### Open items for downstream

Anything the porter discovered that is in scope for *some* future work but not the current port. The next porter or doc maintainer reads this section to know what to pick up. Includes:

- **Cross-op kernel touches.** Any kernel source the port modified or forked that lives outside the op directory (per the [scope boundary](#read-this-first) and the [shared-dataflow-kernel Caution](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel)). Excludes `ttnn/cpp/ttnn/kernel_lib/` and standard framework APIs, which are scope boundaries the porter does not cross. For each cross-op kernel, record: (a) the kernel path; (b) the path taken — **in-place modification** (with the bundled-set consumer list) or **fork** (with the `_metal2`-suffixed new file's path); (c) the remaining unmigrated consumer op directories. This list is the coordination signal for the next sibling-op port and, for forks, the sunset checklist for when the legacy copy can be deleted.
- Per-op carry-over (sibling ops the porter noticed would benefit from the same pattern).
- Doc-evolution suggestions that don't fit cleanly into a Gap entry (broader restructure, new pattern entry candidate).
- Test coverage notes the verification step surfaced but didn't act on.

---

**Substance over comprehensiveness** — 5–15 well-targeted entries across the four sections beats 30 shallow ones. Be specific: cite file paths, line numbers, doc sections.

Commit `METAL2_PORT_REPORT.md` alongside the port code, audit brief, audit report, and port plan. All five artifacts (port code + the four `METAL2_*.md` files) form the port's PR.

---

## Appendix A — `METAL2_PORT_PLAN.md` template

Copy this template to the op's directory at the start of the port:

````markdown
# Port Plan — <op name>

Port plan for `<op>`, ported from `<legacy api>` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape
- Concept: <ProgramFactoryConcept | ProgramDescriptorFactoryConcept | MeshWorkloadFactoryConcept>
- Variants: <list, or "single">
- Custom `compute_program_hash`: <deleted → default (sanctioned exception), was at file:line | none — already default reflection-based hash>

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

> **Multi-variant ops** (e.g., Welford W/H/HW; Reduce W/H/HW): repeat the Kernels / CBs / Semaphores / Tensor accessors / Work split sub-sections **per variant**. Nest the per-variant blocks under a `### Variant: <name>` heading and downshift the per-resource headings to `####`. Cross-op kernels and Flags stay top-level — they typically apply across variants. The single-variant skeleton below is the inner shape of each variant block.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | ... | ... | ... | ... | ... | ... | ... | ... |
| writer | ... | ... | ... | ... | ... | ... | ... | ... |
| compute | ... | ... | ... | ... | ... | ... | ... | ... |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | ... | ... | ... | ... | ... |

### Semaphores
| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 | ... | ... | ... |

(or "none")

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| ... |

### Work split
- Driver: `split_work_to_cores(<args>)`
- num_cores: ...
- core_group_1: ..., count_per_core: ...
- core_group_2: ..., count_per_core: ...

(or "n/a — single core")

### Cross-op kernels
List any kernel `source` path outside the op's directory. Each one is a Caution case (see [catalog](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel)) and is also reported in `METAL2_PORT_REPORT.md` under "Open items for downstream."

(or "none")

### Flags
Anything the inventory step noticed but didn't classify — unreferenced kernel files, unusual descriptors, etc.

(or "none")

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: <MetalV2FactoryConcept>
- **Custom `compute_program_hash`**: <delete (was at file:line) | none>
- **Implementation notes** (optional): <anything specific to how this op will realize the concept that's worth surfacing before construction>

(If you find yourself disagreeing with the audit's choice, stop and surface to the invoker — do not override.)

## Planned Spec Shape

*Filled in during the planning step.*

> **Multi-variant ops**: repeat this section per variant, nested under `### Variant: <name>` headings.

- KernelSpecs: ...
- DataflowBufferSpecs: ...
- SemaphoreSpecs: ...
- TensorParameters: ...
- WorkUnitSpecs: ...

## Preserved Multiplicity

> **Multi-variant ops**: repeat per variant if work-split multiplicity differs across variants; a single shared row suffices if the pattern is identical across variants.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| ... |

(or "none — no work-split multiplicity in legacy")

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader.cpp RTA slot 0 | `tensor.buffer()->address()` | `TensorBinding(input)` |
| reader.cpp CTA slot 0 | `cb_idx = 0` | `DFBBinding(input, "in0", CONSUMER)` |
| ... |

## Applied Patterns

List any patterns from the catalog invoked by this port:

- ... ([catalog entry link])
- ...

(or "none")

## Deferred / Flagged

- New findings during planning: ...

(or "none")
````

---

## Appendix B — Cross-references

- [Feasibility audit](../audit/metal2_audit.md)
- [Migration guide (concept map, design principles, TTNN integration)](../shared/migration_guide.md)
- [Patterns catalog](../shared/port_patterns.md)
