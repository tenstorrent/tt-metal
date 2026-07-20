# CB → DFB Kernel Audit Report

> **Status:** Living document (2026-07-14). **Standalone device-side audit** — classifies legacy **kernel** CB usage and CB→DFB port readiness per op or ProgramFactory. Does **not** audit host `ProgramSpec`, binding multiplicity, or factory refactors ([`audit/metal2.md`](metal2.md) covers host/spec). **Scope is machine-discovered** from program factories and PR diffs — see [Automated scope discovery](#automated-scope-discovery).
>
> **Companion docs:** [CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md) (CB API → DFB / evil_get|set mapping), [CB→DFB flowchart](../../human/CB-to-DFB-flowchart.svg), [Metal 2.0 port patterns](../shared/port_patterns.md) (host binding — cross-ref only), repo-root inventory `2026-07-07 Quasar CB port-readiness audit — illegal CB usage by op.md`.

---

## How to use this doc

**What this is.** A **kernel-only** auditor: given an op or factory slice, discover in-scope device kernels, scan for illegal / weird CB patterns, classify **every CB**, and produce a **CB portability** report with **1xx** and **2xx** status columns.

**What this is not.** Host-side Metal 2.0 feasibility (SPSC, endpoint counts, `DataflowBufferSpec` legality, tensor binding cases). Do not block or roll up on host issues here — cross-reference [`audit/metal2.md`](metal2.md) separately when doing a full op port.

**When to run.**

| Trigger | Example |
|---------|---------|
| Pre-kernel-port readiness | "Audit layernorm sharded Welford kernels for CB→DFB" |
| PR touching `*/kernels/*` | Diff-driven scope from changed kernel or factory file |
| Port planning | Before rewriting `CircularBuffer` → `DataflowBuffer` in device code |

**Procedure (5 steps).**

1. **[Scope discovery](#automated-scope-discovery)** (Steps 0–3) — resolve op/factory → `KERNEL_FILES` → `SCAN_FILES` (kernels + `#include` closure). Uses factory files only to **find** kernel paths; does not edit or audit host factory code.
2. **[Classification scans](#step-4--classification-scans-on-scan_files-only)** — run `rg` on `SCAN_FILES` only.
3. **[Classify](#step-5--classify-cbs)** — every CB gets a class (1–6), verdict, and device port strategy (DFB / LTA / scratchpad / workaround).
4. **[Report template](#report-template)** — one row per CB; **1xx** and **2xx** portable status + Notes.
5. **Deliver** — write `CB_DFB_KERNEL_AUDIT.md` in the op directory, paste into a PR comment, or append as the kernel section of `METAL2_PREPORT_AUDIT.md` if a full op port is in flight.

**Rollup (device kernel port).**

| Verdict | Meaning |
|---------|---------|
| **GREEN** | All CBs **Portable** or **Portable (workaround)** (flag ptr hacks in Notes) |
| **YELLOW** | Any **Portable (prereq: LTA)** and/or **Blocked (runtime)** on 2xx with 1xx clear |
| **RED** | Any **Blocked**, unresolved **GATE**, or NEEDS-DESIGN without v1 strategy |

**Quick links:** [Verdict legend](#verdict-legend) · [Issue taxonomy](#issue-taxonomy-kernel-side) · [LTA vs scratchpad rollup](#port-recipe-rollup-lta-vs-scratchpad) · [CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md) · [Example report (layernorm)](#example-audit-report-layernorm-sharded-welford-path)

---

## Read this first

**Why this exists.** On Gen1 (WH/BH), kernels could treat `LocalCBInterface` as a mutable struct: read `fifo_page_size`, rewrite `fifo_rd_ptr`, skip FIFO credits, use a CB as scratch. This interface is meant to be private (as it is in the internal directory). Code that still touches `get_local_cb_interface(...).<field>` may compile on Gen1 but is **un-portable** or **silent-wrong** on Quasar since there are totally different fields. Queries should be going through the DataflowBuffer object.

**Audience:** Humans and AI agents auditing **device kernels** (`*/kernels/*`, `kernel_util/`) for Metal 2.0 / Quasar CB→DFB port readiness. Run **standalone** — no host audit prerequisite.

**Operating principle:** Classify each CB/DFB usage along **two orthogonal axes** (same as the flowchart):

1. **Synchronized vs sync-free** — does the kernel use real FIFO ops (`reserve_back` / `push_back` / `wait_front` / `pop_front`) for a cross-kernel handoff?
2. **Memory model** — is this a **linear FIFO**, **windowed/scatter L1**, **pointer-only bookkeeping**, or **in-place accumulator**?

Do not assume “it's a CB, so use DFB + push_back.” Many legacy CBs were **pinned L1** or **credit/decoupled** patterns.

**Three Metal 2.0 memory/sync primitives** (classify which one applies):

- `**DataflowBuffer`** — canonical cross-kernel FIFO (Class 1).
- `**LocalTensorAccessor`** — sync-free borrowed tensor view (Class 6 borrowed; some Class 2/5 pointer-only).
- `**ScratchpadSpec` + `SemaphoreSpec**` — private L1 with **explicit** sync when FIFO credits would lie (Classes 2–4 window/ring/staging; Class 6 scratch; Quasar DM self-loop replacement). See [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync).

**Audit default (dual-track for Classes 2–5):**

```mermaid
flowchart LR
  class25[Class 2-5 CB] --> v1["WH/BH v1: DFB + evil_get/set no semantic change"]
  class25 --> qsr["Quasar end-state: redesign scratchpad/LTA/self-loop"]
  v1 --> flag["Portable workaround + Quasar redesign required"]
  qsr --> design["NEEDS-DESIGN-DECISION until Quasar port"]
```

| Track | Strategy | Report |
|-------|----------|--------|
| **1xx / WH-BH initial port** | Mechanical `CircularBuffer` → `DataflowBuffer`. Map `get_local_cb_interface(...).fifo_*_ptr` mutates to **`DataflowBuffer` `evil_set_*` / `evil_get_*`** ([whitelist §D](../shared/cb_dfb_api_whitelist.md) — status **in flight**). No dedicated hold API: `cb_push_back_hold_wr_ptr` → `push_back` + `evil_set_write_ptr`. Keep kernel semantics. | **Portable (workaround)** — Notes: **undesirable but OK hack** + **Quasar redesign required** |
| **2xx / Quasar end-state** | Prefer **scratchpad + semaphores**, **LocalTensorAccessor**, compute self-loop, or strided DFB. **Do not** treat `evil_*` as the lasting Quasar fix. | **NEEDS-DESIGN-DECISION** until redesign lands |

Record status and workaround in the **CB portability** table for every buffer (see [Report template](#report-template)). API names: [`cb_dfb_api_whitelist.md`](../shared/cb_dfb_api_whitelist.md).

### Port-recipe rollup (LTA vs scratchpad)

Aligned with [`port/metal2.md`](../port/metal2.md):

| End-state | Port recipe treatment | Report status | Op rollup |
|-----------|----------------------|---------------|-----------|
| **`ScratchpadSpec` + semaphores** | **Autoportable** Quasar / redesign end-state | **Portable** | **GREEN** |
| **`LocalTensorAccessor`** | **Port prerequisite** — must land in the port (host `TensorBinding` + kernel ctor swap) | **Portable (prereq: LTA)** | **YELLOW** |
| DFB + `evil_get_*` / `evil_set_*` (1xx v1) | Documented WH/BH mechanical hack — **undesirable but OK**; opens Quasar design debt | **Portable (workaround)** | **GREEN** (flag hack + Quasar debt in Notes) |

**What is already portable (do not churn):**

- Class 1 DM transfers via **`Noc` from `noc.h`** (`async_read` / `async_write` + `offset_bytes`) — kernel must **not** call `get_*_ptr` or use `dataflow_api.h` `noc_async_*` with a grabbed L1 addr ([whitelist access control](../shared/cb_dfb_api_whitelist.md#access-control-get_ptr-vs-evil)).
- Canonical producer/consumer FIFO (Pattern A/B in the audit).
- Classes 2–5 pointer surgery → map to **evil_get/set** ([whitelist](../shared/cb_dfb_api_whitelist.md)), not leave `LocalCBInterface` field writes — and **not** blessed `get_*_ptr`.

**What is rewrite debt (do not treat as OK on Device 2.0 ports):**

- Kernel `dfb.get_*_ptr()` + `noc_async_*` from `dataflow_api.h` (~many Class 1 sites) → rewrite to `noc.h`.
- Public `get_*_ptr` remaining on DFB until that migration + privatize/`friend` lands.

**What is the litmus for illegal/un-portable** (run on [auto-discovered scope](#step-4--classification-scans-on-scan_files-only), not the whole repo, when auditing one op):

```bash
# GATE — any field read/write via get_local_cb_interface blocks Metal 2.0 port
rg 'get_local_cb_interface' $SCAN_GLOB

# QUASAR-BLOCKED / WH read APIs — see Runtime fixes (#49617 in flight)
rg 'read_tile_value|get_tile_address' $SCAN_GLOB

# NEEDS-FIX — migrate sync-free borrowed reads to LocalTensorAccessor (Portable prereq: LTA → YELLOW)
rg 'get_pointer_to_cb_data' $SCAN_GLOB

# Wrong on Quasar
rg 'get_cb_tiles_acked_ptr|get_cb_tiles_received_ptr' $SCAN_GLOB
```

---

## Audit scope (device kernels only)

Kernel discovery follows the same **follow factory references, not directory boundaries** rule as the host audit, but **this audit scans and gates on device code only**:

- **Follow kernel references, not directory boundaries.** Every path assigned to `KernelDescriptor::kernel_source` (or equivalent string literal / helper that builds a kernel path) in the op's program factories is in scope — including cross-op donor kernels. Factory `.cpp` files are read **only to extract kernel paths**, not audited for host/spec legality.
- **Unreferenced files under `*/kernels/*` in the op tree are out of scope.** List them in the report as *unreferenced* only if their presence could confuse a reader; do not scan or RED-gate on them.
- **Multiple `DeviceOperation` types** in one directory: one combined report when they share factories/kernels; separate reports when independent (ask user only if bundling is ambiguous).
- **Atomic unit:** one **ProgramFactory** (or factory helper bundle) at a time when the op has several factories — same as `[port/metal2.md](../port/metal2.md)`.

**Path exclusions (mark OUT-OF-SCOPE, never RED-roll up):**


| Pattern                      | Reason                               |
| ---------------------------- | ------------------------------------ |
| `**/deepseek_moe_gate/`**    | Firmware-style CB reconfig           |
| `**/generalized_moe_gate/`** | Same                                 |
| `**/deepseek_prefill/`**     | Combine/dispatch/post_combine_reduce |


These need separate device/framework design (firmware-style reinit, expert routing, etc.) and should not block or gate other op ports tracked by this audit.

---

## Automated scope discovery

### Step 0 — Resolve target and discover factories

Pick **one** trigger; derive `OP_ROOT`(s), `FACTORY_FILES`, and optional `FACTORY_FILTER` without asking the user to paste paths.


| Trigger                                           | How to resolve                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **User names an op** (e.g. "layernorm", "conv2d") | `find ttnn/cpp/ttnn/operations -type d -name '<slug>'` or `rg -l 'DeviceOperation' ttnn/cpp/ttnn/operations`                                                                                                                                                                                                                                |
| **PR / branch diff**                              | `git diff --name-only "${BASE_REF:-origin/main}...HEAD"` → keep paths under `ttnn/cpp/ttnn/operations/` → collapse to op roots: `ttnn/cpp/ttnn/operations/<family>/<op>/` (first six path segments after `operations/`). Audit **each** changed op root; if the diff only touches one factory file, set `FACTORY_FILTER` to that basename. |
| **Prior audit / plan doc**                        | Read `CB_DFB_KERNEL_AUDIT.md`, `METAL2_PORT_PLAN.md`, or a PR description for op root, factory names, and donor kernels — re-run kernel closure below to verify.                                                                                                                                                                           |
| **Kernel port PR** (only `.cpp` under `kernels/`) | Walk up to op root from the changed kernel path; find **all factories that reference that kernel** (factory enumeration below) so sibling factory paths are not missed.                                                                                                                                                                    |


```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
OP_ROOT=ttnn/cpp/ttnn/operations/normalization/layernorm

# Example: op roots from a PR diff
git diff --name-only origin/main...HEAD \
  | rg '^ttnn/cpp/ttnn/operations/[^/]+/[^/]+/' \
  | sed -E 's|(ttnn/cpp/ttnn/operations/[^/]+/[^/]+/).*|\1|' \
  | sort -u

# Factory / program-build hosts (used only to locate kernel paths)
FACTORY_FILES="$(find "$OP_ROOT" \( -name '*.cpp' -o -name '*.hpp' \) \
  | rg -l 'ProgramFactory|program_factory|KernelDescriptor|CreateKernel|create_program|ProgramSpec' \
  | rg 'program_factory|factory_helpers|_op_.*\.cpp|ProgramFactory')"
```

If `FACTORY_FILTER` is set, restrict `FACTORY_FILES` to paths matching that substring.

### Step 1 — Extract in-scope kernel paths from factories

Run on `FACTORY_FILES` from Step 0. Collect **repo-relative** paths (prefix `ttnn/cpp/ttnn/operations/...` when factories use `base_path` + suffix).

```bash
# FACTORY_FILES set in Step 0

# String literals pointing at kernel sources
rg -oN '"ttnn/cpp/ttnn/operations/[^"]+\.(cpp|hpp|h)"' $FACTORY_FILES | sort -u

# Relative kernel paths (resolve against OP_ROOT/device/)
rg -oN '"[^"]*device/kernels/[^"]+\.(cpp|hpp|h)"' $FACTORY_FILES | sort -u

# kernel_source assignments (trace variables — read the helper that sets compute_path / reader_path)
rg -n 'kernel_source\s*=' $FACTORY_FILES
```

**Variable `kernel_source`:** When the RHS is not a string literal (e.g. `kernel_config.compute_path`, ternary on `use_welford`), read the helper struct / function that populates it and expand **all reachable paths** for the factory variant under audit. Do not audit only the default branch.

**Cross-op donors:** Any extracted path outside `OP_ROOT` stays in scope; note *donor op* in the report.

Write the result to `KERNEL_FILES` (absolute or repo-relative). This is the **only** kernel set Step 2 scans.

### Step 2 — Transitive `#include` closure

For each file in `KERNEL_FILES`, pull shared headers (often `kernel_util/`, sibling op headers):

```bash
for k in $KERNEL_FILES; do
  rg -oN '#include "([^"]+)"' "$k"
done
```

Resolve includes relative to the including file and to known kernel include roots (`ttnn/`, `tt_metal/`). Add headers to `SCAN_FILES = KERNEL_FILES + included headers`. Deduplicate.

### Step 3 — Unreferenced kernel inventory (informational)

```bash
find "$OP_ROOT" -path '*/kernels/*' \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) | sort \
  > /tmp/all_kernels.txt
# KERNEL_FILES minus all_kernels → unreferenced (ignore in scans)
comm -23 /tmp/all_kernels.txt <(printf '%s\n' $KERNEL_FILES | sort)
```

### Step 4 — Classification scans (on `SCAN_FILES` only)

Do **not** scan the whole op tree unless re-discovering factories. Apply OUT-OF-SCOPE path filter before counting hits.

```bash
SCAN_GLOB=$(printf '%s ' $SCAN_FILES)   # or xargs file list

# GATE — hard blocker: LocalCBInterface field access
rg -n 'get_local_cb_interface\s*\([^)]*\)\s*\.|cb_interface\.' $SCAN_GLOB

# Silent-wrong on Quasar
rg -n 'get_cb_tiles_acked_ptr|get_cb_tiles_received_ptr' $SCAN_GLOB

# QUASAR-BLOCKED / WH read APIs — see Runtime fixes (#49617 in flight)
rg -n 'read_tile_value|get_tile_address' $SCAN_GLOB

# NEEDS-FIX — migrate sync-free borrowed reads to LocalTensorAccessor (Portable prereq: LTA → YELLOW)
rg -n 'get_pointer_to_cb_data' $SCAN_GLOB

# Portable pointer use (WEIRD-OK candidates — classify, do not auto-fail)
rg -n 'get_read_ptr\s*\(|get_write_ptr\s*\(' $SCAN_GLOB

# Pointer surgery / credit hacks (Classes 2–5)
rg -n 'fifo_wr_ptr|fifo_rd_ptr|push_back_hold|llk_push_pages' $SCAN_GLOB

# Mechanical field reads (NEEDS-FIX — use DFB getters)
rg -n 'fifo_page_size|fifo_num_pages' $SCAN_GLOB

# Buffer inventory signals (derive per-kernel rows — do not hand-enumerate every cb in the repo)
rg -n 'CircularBuffer\s+\w+|DataflowBuffer\s+\w+|cb_[a-zA-Z0-9_]+\s*\(|get_compile_time_arg.*cb|#define\s+cb_' $SCAN_GLOB
```

### Step 5 — Classify CBs

Use scan hits + short context (±5 lines) to assign verdicts per the [Issue taxonomy](#issue-taxonomy-kernel-side). **List every in-scope CB** from Step 4 inventory signals across `SCAN_FILES` — one row per CB in the report.


| Scan match                                                        | Default class | Default verdict                                                               | Override when                                                                                                                               |
| ----------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_local_cb_interface(...).`                                    | —             | **GATE**                                                                      | Never WEIRD-OK                                                                                                                              |
| `get_cb_tiles_*_ptr`                                              | —             | **NEEDS-FIX** (silent-wrong)                                                  | —                                                                                                                                           |
| `get_pointer_to_cb_data` | 6 | **NEEDS-FIX** | Migrate to **LocalTensorAccessor** — report **Portable (prereq: LTA)** → op **YELLOW** |
| `read_tile_value` / `get_tile_address` on DFB | 6 | **QUASAR-BLOCKED** on Quasar until DFB read API lands; WH/BH **in flight** (#49617) | Use **LocalTensorAccessor** when access is sync-free borrowed L1; DFB read API when LLK/DFB id required |
| `get_read_ptr` / `get_write_ptr` used for NOC addr + `dataflow_api.h` `noc_async_*` | 1 | **NEEDS-FIX** | Rewrite to `Noc` from **`noc.h`** — not OK to keep |
| `get_read_ptr` / `get_write_ptr` used for save/jump/rewind | 2–5 | **WEIRD-OK** (1xx) after → `evil_get_*` | Do not leave blessed getters for surgery |
| `fifo_wr_ptr` / `fifo_rd_ptr` jumps / `get_local_cb_interface` ptr assign | 2–5           | **WEIRD-OK** (1xx) / **NEEDS-DESIGN-DECISION** (2xx end-state)                | **1xx:** map to [`evil_get/set`](../shared/cb_dfb_api_whitelist.md) (**in flight**). **2xx:** redesign scratchpad/LTA/self-loop/strided — do not treat evil_* as Quasar end-state |
| `push_back_hold` / `cb_push_back_hold_wr_ptr` / partial save-restore on **compute** | 4–5           | **WEIRD-OK** (1xx) / **NEEDS-DESIGN-DECISION** (2xx)                          | **No** hold DFB API — `push_back` + `evil_set_write_ptr`. Quasar + canonical PACK→UNPACK → also flag **SELF-LOOP-CANDIDATE**                                            |
| Compute · sync-free or single-ended · **Quasar target**           | 5–6           | **SELF-LOOP-CANDIDATE** (audit annotation)                                    | WH/BH: evil_get/set workaround, LTA, or scratchpad — not self-loop                                                                                                        |
| `fifo_page_size` / `fifo_num_pages`                               | 1             | **NEEDS-FIX**                                                                 | Use DFB getters                                                                                                                             |
| FIFO sync calls only, no hits above                               | 1             | **PORTABLE**                                                                  | —                                                                                                                                           |
| Path matches OUT-OF-SCOPE table                                   | —             | **OUT-OF-SCOPE**                                                              | Exclude from rollup                                                                                                                         |


**Overall rollup:** Same as [Verdict legend](#verdict-legend) — compute from classified hits across `SCAN_FILES`.

---

## Workflow at a glance

For each factory slice under audit:

1. **Run [scope discovery](#automated-scope-discovery)** (Steps 0–3) — derive `KERNEL_FILES` and `SCAN_FILES`.
2. **Run classification scans** on `SCAN_FILES` only (Step 4).
3. **Classify** each CB in the [Issue taxonomy](#issue-taxonomy-kernel-side) (classes 1–6).
4. **Assign verdict** and device port strategy — dual-track for Classes 2–5: [CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md) (`evil_get/set` on 1xx) vs [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync) / [LTA](#localtensoraccessor-lta) / [compute self-loop](#quasar-only-compute-self-loop-self-loop-candidate) / [strided DFB](#architecture-fork-strided-scatter-class-3) (Quasar end-state).
5. **Fill the [report template](#report-template)** — one row per CB with **1xx** / **2xx** portable status → `CB_DFB_KERNEL_AUDIT.md` or PR comment. Notes: `1xx_port: evil_DFB` / `2xx_endstate: scratchpad|LTA|self-loop|strided` when Class 2–5.

```mermaid
flowchart TD
    A[Legacy CB usage in kernel] --> B{FIFO sync used?}
    B -->|Yes, canonical| C[Class 1: Linear FIFO]
    B -->|No or partial| D{What memory model?}
    D --> E[Class 2: Window scratch]
    D --> F[Class 3: Scatter write]
    D --> G[Class 4: Credit/address split]
    D --> H[Class 5: In-place accumulator]
    D --> I[Class 6: Structural non-FIFO]
    C --> J[DataflowBuffer + sync APIs]
    E --> K["1xx: evil_get/set · 2xx: scratchpad/sems"]
    F --> L["1xx: evil_set scatter · 2xx: strided DFB"]
    G --> M["1xx: push_back + evil_set · 2xx: redesign"]
    H --> N["1xx: evil save/restore · 2xx: self-loop/LTA"]
    I --> O[Scratchpad / LTA / borrowed_from]
```



---

## GATE: `get_local_cb_interface` field access

**Policy (settled):** Any **read or write** of `get_local_cb_interface(cb).<field>` in an op's in-scope kernels **blocks the kernel CB→DFB port**. This is a hard **GATE** on the device side. The kernel port does not proceed until every hit is resolved.

**What counts as a violation:**

- Any `get_local_cb_interface(...).fifo_`* field read or write
- Any `get_local_cb_interface(...).tiles_acked` / `tiles_received` access
- Indirect access that still reaches `LocalCBInterface` fields (e.g. `cb_interface.fifo_page_size` after grabbing the struct)

**What does *not* count (not a GATE):**

- Bare `get_read_ptr()` / `get_write_ptr()` on `CircularBuffer` or `DataflowBuffer` — **not** a GATE by itself, but Class 1 Device 2.0 ports **must rewrite** to `noc.h` (see whitelist); Classes 2–5 surgery → `evil_*`
- Official DFB getters (`get_entry_size()`, `get_total_num_entries()`, etc.)
- `get_pointer_to_cb_data` — **NEEDS-FIX** (migrate to **LocalTensorAccessor**)
- `read_tile_value` / `get_tile_address` via sanctioned DFB APIs (QUASAR-BLOCKED until API lands, not GATE)

**Resolution paths** — for each GATE hit, pick exactly one:


| Situation                                             | Action                                                                                               |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Known field → existing DFB getter                     | **NEEDS-FIX** — rewrite before port merges (e.g. `fifo_page_size` → `get_entry_size()`)              |
| Known field → getter in flight (Runtime fixes table) | **BLOCKED** — port waits for API; track in audit report                                              |
| Field has no getter yet                               | **File issue to Almeet** — see below; port BLOCKED until getter exists or pattern is redesigned away |
| `fifo_wr_ptr` / `fifo_rd_ptr` *written* (Classes 2–5) | **WEIRD-OK** on 1xx — uplift to `dfb.evil_set_*` / `evil_get_*` ([whitelist §D](../shared/cb_dfb_api_whitelist.md), **in flight**); report **Portable (workaround)** + **Quasar redesign required**. Leaving raw `LocalCBInterface` field writes = unresolved **GATE**. |


### Filing a missing-getter issue (→ Almeet)

When a field read cannot be replaced by an existing or in-flight getter, **do not invent a workaround in the port PR**. File an issue to **Almeet** with:

1. **Op + kernel file** (path, line numbers)
2. **Field accessed** (e.g. `fifo_size`, `fifo_limit`)
3. **What the kernel is doing** (one paragraph — ring wrap, zero-fill span, debug ASSERT, etc.)
4. **Proposed getter name + semantics**
5. **Class** (from taxonomy below) and whether a **non-getter redesign** is also viable (Scratchpad, disable opt, etc.)

Almeet owns the DFB getter API surface. The port stays RED until the issue is triaged: getter added, or the op owner picks an approved redesign that eliminates the field read.

**Audit report routing:** Record each GATE hit in **GATE hits** (file:line, field, fix). Every in-scope CB gets a row in **CB portability** with **1xx** and **2xx** status. A non-empty unresolved GATE list → audit **RED** for Metal 2.0 port.

---

## Verdict legend


| Verdict                   | Meaning                                                                                                                                                | Blocks Metal 2.0 port?               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| **PORTABLE**              | Class 1 linear FIFO; mechanical `CircularBuffer` → `DataflowBuffer`                                                                                    | No                                   |
| **WEIRD-OK**              | Non-canonical pattern (ptr offset, manual credits, etc.) but **known workaround** exists; port can proceed with documented hack                        | No (if strategy recorded)            |
| **NEEDS-FIX**             | Mechanical fix only (getter swap, delete duplicate mailbox block, include `memory.h` once)                                                             | Yes until fixed                      |
| **QUASAR-BLOCKED**        | Needs runtime API (`read_tile_value`, `get_tile_address` on DFB, typed read)                                                                          | Yes on Quasar until API lands        |
| **NEEDS-DESIGN-DECISION** | Classes 2–5 structural pattern; pick scratchpad + semaphores / LTA / compute self-loop (Quasar) / strided DFB (2xx) / disable opt                      | Yes until strategy chosen            |
| **SELF-LOOP-CANDIDATE**   | **Quasar only** — compute kernel may bind same DFB as PRODUCER+CONSUMER (canonical PACK→UNPACK on one kernel). Audit annotation, not a WH/BH strategy. | No (informational; record in report) |
| **GATE**                  | `get_local_cb_interface(...).<field>` read or write                                                                                                    | **Yes — hard stop**                  |
| **STRUCTURAL**            | Class 6 — never a real FIFO; needs scratchpad / LTA / borrowed DFB                                                                                     | Yes until kernel strategy chosen       |
| **OUT-OF-SCOPE**          | MOE gate / DeepSeek — track elsewhere                                                                                                                  | No (for this op audit)               |


**Overall op rollup:**


| Rollup     | Condition                                                                                                               |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| **GREEN**  | No unresolved GATE; no **Blocked** rows; no **Portable (prereq: LTA)** — **Portable** and **Portable (workaround)** both count as GREEN (workaround CBs must be flagged in Notes as **undesirable but OK hack**) |
| **YELLOW** | Any **Portable (prereq: LTA)**; and/or mechanical **NEEDS-FIX** (non-LTA); and/or **Blocked (runtime)** on 2xx with 1xx path clear |
| **RED**    | Any unresolved GATE, SILENT-WRONG, **Blocked**, or NEEDS-DESIGN-DECISION without v1 strategy                                         |


**WEIRD-OK examples (highlight in report, do not treat as GATE):**

- `get_write_ptr() + byte_offset` + `dataflow_api.h` NOC — **NEEDS-FIX** → `noc.h` (not WEIRD-OK)
- Kernel `get_read_ptr()` / `get_write_ptr()` for Class 1 transfers — **NEEDS-FIX** → `Noc::async_*` (~many op files = rewrite debt)
- `get_pointer_to_cb_data` — **NEEDS-FIX** → **Portable (prereq: LTA)**; op rollup **YELLOW**
- sanctioned `read_tile_value` / `get_tile_address` on DFB (QUASAR-BLOCKED until API, not GATE)
- Manual `fifo_wr_ptr` assignment → uplift to `evil_set_write_ptr` — **Portable (workaround)**; Notes: **undesirable but OK hack:** … + **Quasar redesign required** (rollup stays **GREEN**)

**GATE (never WEIRD-OK):** Any `get_local_cb_interface(cb).<field>` read or write.

**Mapping classification → report status:**

| Classification verdict | Report status (per arch) | Op rollup |
| ---------------------- | ------------------------ | --------- |
| **PORTABLE** (Class 1, scratchpad end-state) | **Portable** | GREEN |
| **NEEDS-FIX** → **LocalTensorAccessor** (`get_pointer_to_cb_data`, sync-free borrowed) | **Portable (prereq: LTA)** | YELLOW |
| **NEEDS-FIX** (other mechanical: DFB getters, etc.) | **Portable** — note fix in Notes | YELLOW |
| **WEIRD-OK**, **SELF-LOOP-CANDIDATE** | **Portable (workaround)** — Notes: **undesirable but OK hack:** + `evil_get/set` uplift + **Quasar redesign required** for Classes 2–5 | GREEN |
| **NEEDS-DESIGN-DECISION**, **STRUCTURAL** (no strategy) | **Blocked** | RED |
| **QUASAR-BLOCKED** | **Blocked (runtime)** on **2xx**; **Portable (workaround)** on **1xx** if ptr hack documented | YELLOW (2xx blocked) or GREEN (1xx only) |
| **GATE** | **Blocked** on both arches | RED |

---

## Runtime fixes in flight

These unblock large swaths of the audit without per-kernel hacks. **Reclassify affected ops when they land.**


| Fix                                                                 | Status                                                                   | Unblocks                                                                                                                                                 |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DataflowBuffer::get_total_num_entries()`                           | Merged — [PR #49197](https://github.com/tenstorrent/tt-metal/pull/49197) | `fifo_num_pages` field reads (`pool_kernels_common`, `clear_out_tiles`, `zero_out_tiles`, debug ASSERTs)                                                 |
| Extent getters (`get_total_size_bytes`, local/span size & entries)  | **In flight** — [#49652](https://github.com/tenstorrent/tt-metal/pull/49652) | `fifo_page_size` / capacity / ring-span field reads; TRISC sizes in **bytes**                                                                            |
| Templated `read_tile_value<T>` / `get_tile_address` on DFB (WH/BH) | **In flight** — [#49617](https://github.com/tenstorrent/tt-metal/pull/49617) | WH/BH scalar tile reads; Quasar follow-up after #49617 / related blockers                                                                                |
| `read_tile_value` / `get_tile_address` on DFB (Quasar)              | **Blocked / follow-up** after #49617                                     | In-scope `QUASAR-BLOCKED` compute kernels; `memory.h` Welford family; sdpa sparse ctrl (DeepSeek / moe gate out of scope)                                |
| `evil_get_write_ptr` / `evil_get_read_ptr` / `evil_set_*` on DFB    | **In flight** (not in tree yet) — [whitelist §D](../shared/cb_dfb_api_whitelist.md) | Classes 2–5 `get_local_cb_interface(...).fifo_*_ptr` mutates; SDPA hold-wr → `push_back` + `evil_set_write_ptr` (~66 in-scope mutate sites)               |


**Already use today on DFB:**


| Legacy field                                 | DFB getter / note                    |
| -------------------------------------------- | ------------------------------------ |
| `fifo_page_size` (entry bytes)               | `get_entry_size()` (+ #49652 aliases when merged) |
| Stride between entries                       | `get_stride_size()`                  |
| `fifo_rd_ptr` / `fifo_wr_ptr` as Class 1 NOC L1 cursor | **Do not** expose to kernels — `Noc` from `noc.h` (private/`friend` `get_*_ptr`) |
| Classes 2–5 pointer jumps / save-restore     | **`evil_get_*` / `evil_set_*`** (**in flight**) — never leave `LocalCBInterface` field writes |
| Tile format bytes                            | `get_tile_size()` (from descriptors) |


---

## Quasar-only: compute self-loop (self-loop candidate)

> **Scope:** **Quasar (Gen2) only.** WH/BH ports must **not** plan on compute self-loop — use **`evil_get/set`** ([whitelist](../shared/cb_dfb_api_whitelist.md)), scratchpad + semaphores, LTA, or canonical cross-kernel DFBs instead.
>
> **Audit terminology:** Mark matching buffers `**SELF-LOOP-CANDIDATE`** in the report. This is **not** a separate port-time option — it means “this buffer on **compute** could use a **self-loop DFB binding** (PRODUCER + CONSUMER on the same compute kernel) when the tile stream is canonical PACK→UNPACK on that kernel.” On Quasar the framework lowers that binding automatically; porters only choose the host self-loop bind pattern.

When a **compute kernel** binds the same DFB as both `PRODUCER` and `CONSUMER` (**compute self-loop**), Metal 2.0 uses Quasar tile-counter hardware for PACK→UNPACK credits within the same Neo. Credits flow through canonical `push_back` / `wait_front` at the DFB API — not manual `LocalCBInterface` pointer surgery.

Host binding (from `[port_patterns.md](../shared/port_patterns.md)`):

```cpp
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::PRODUCER);
BindDFB(compute, ACC_DFB, "acc", DFBEndpointType::CONSUMER);
// Kernel: DataflowBuffer cb_acc(dfb::acc);  // one handle, both directions
```

**Where to flag `SELF-LOOP-CANDIDATE` on Quasar** (prefer compute self-loop over ptr hacks when the pattern fits):


| Audit class / pattern                                                                            | Self-loop candidate on Quasar? | Notes                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Class 5** — compute accumulator with real pack→unpack loop                                     | **Yes**                        | Compute self-loop replaces “fake FIFO + save/restore rd/wr” when PACK produces tiles UNPACK reads back in canonical order. LLK partials paths that bypass tile-counter posting → **DFB ptr/credit surgery** (**WEIRD-OK**). |
| **Class 6** — sync-free compute scratch / tensor view                                            | **No**                         | Prefer **LocalTensorAccessor** (borrowed) or **scratchpad** (private L1) — see [LocalTensorAccessor](#localtensoraccessor-lta).                                                                                             |
| **Single-ended compute packer** (`OUT` → resident output shard)                                  | **Yes**                        | Synchronized producer with no drain kernel → self-loop on compute.                                                                                                                                                          |
| **Class 1** — BRISC reader → TRISC compute                                                       | **No** (use ordinary DFB)      | Cross-**kernel** PRODUCER on reader, CONSUMER on compute — not compute self-loop.                                                                                                                                           |
| **Class 4** — credit decoupled from address (`cb_push_back_hold_wr_ptr`, bilinear pack-untilize) | **No**                         | Self-loop assumes aligned pack/unpack credits. Decoupled ptr/credit hacks remain for v1.                                                                                                                                    |
| **UNPACK ↔ MATH ↔ PACK** engine handoff inside compute                                           | **No**                         | Self-loop is PACK↔UNPACK only. Use **program semaphores**, **mailbox** scalars, or **STALLWAIT** — not DFB.                                                                                                                 |
| **DM self-loop** (same BRISC kernel PRODUCER+CONSUMER)                                           | **No**                         | Quasar runtime rejects DM self-loop DFB. Use scratchpad + semaphores, or redesign.                                                                                                                                          |


**Audit heuristic:** On **Quasar**, before defaulting to Class 4/5 ptr workarounds inside a **compute** kernel, ask: “Is this a PACK-produced / UNPACK-consumed tile stream on one kernel?” If yes → flag `**SELF-LOOP-CANDIDATE`** and plan **compute self-loop** binding. If credits are decoupled from addresses or MATH sits in the handoff → not a candidate; use semaphores or ptr hacks per Class 4/5.

**See also:** [Self-loop DFB binding](../shared/port_patterns.md#pattern-self-loop-dfb-binding); [Sync-free / single-ended → self-loop interim](temp_cb_dfb_reference_info.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround).

---

## LocalTensorAccessor (LTA)

> **Scope:** `LocalTensorAccessor` is the **local-L1, typed** counterpart to `TensorAccessor`. It works on **both DM and compute** kernels (TRISC-safe — no NoC machinery). It replaces the legacy **“pinned CB as L1 pointer”** idiom: `get_pointer_to_cb_data`, sync-free `get_read_ptr()` / `get_write_ptr()` on a **borrowed** tensor CB, or Case-2 `TensorAccessor::get_bank_base_address()` when access stays on the node-local shard.
>
> **Port recipe:** LTA is a **port prerequisite**, not autoportable — audit as **Portable (prereq: LTA)** → op rollup **YELLOW** until the port lands host `TensorBinding` + kernel `LocalTensorAccessor` ctor (see [Port-recipe rollup](#port-recipe-rollup-lta-vs-scratchpad)).

**Litmus:** Does the kernel touch resident tensor L1 **without** FIFO sync that another kernel waits on (`push_back` / `wait_front` never form a real hand-off)? If yes **and** backing is **borrowed from a tensor** → target **LTA**, not a DFB. If backing is **private** (non-borrowed) L1 → use `**ScratchpadSpec`**, not LTA.

Host binding: `TensorBinding` on the touching kernel; kernel constructs `LocalTensorAccessor<T>(tensor::name)` (see `[local_tensor_accessor.h](../../../../../../../../../tt_metal/hw/inc/api/tensor/local_tensor_accessor.h)`). For Case-2a ports, prefer `**get_bank_base_address()` / `get_unsafe_ptr()`** and keep legacy byte arithmetic — do not rewrite into `operator[]` iteration during the port.

**Where LTA applies** (by audit class):


| Audit class / pattern                                                       | LTA?             | Notes                                                                                                                                                                                                             |
| --------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Class 6** — sync-free · **borrowed** · read-only                          | **Yes (target)** | Tensor view by base pointer only. Example: Welford reciprocal LUT (`get_pointer_to_cb_data` on `cb_reciprocals` in six normalization kernels).                                                                    |
| **Class 6** — sync-free · **borrowed** · read-write                         | **Yes (target)** | In-place RMW on a resident shard. Example: conv2d `MATMUL_PARTIALS` when `partials_cb_uses_output` aliases the output buffer.                                                                                     |
| **Class 6** — sync-free · **regular-backed** (private scratch)              | **No**           | `**ScratchpadSpec`** — not tensor-backed.                                                                                                                                                                         |
| **Class 2** — sync-free borrowed lookup / mailbox read                      | **Yes (target)** | Light pointer-only cases (e.g. fused_swiglu scratch CBs used as address sources) once `read_tile_value` is not required.                                                                                          |
| **Class 2** — window reuse with pipelined FIFO credits                      | **No (LTA)**     | **Prefer scratchpad + semaphores** for window layout — not DFB ptr jumps. LTA only for sync-free borrowed pointer-only reads.                                                                                     |
| **Class 4** — LLK tile-stream / credit decoupling                           | **No**           | LLK still targets **CB/DFB ids** (`pack_tile`, `cb_push_back_hold_wr_ptr`). LTA does not substitute until LLK accepts non-FIFO L1 views.                                                                          |
| **Class 4** — tensor-backed output, **raw L1 + offset only** (no LLK cb id) | **Maybe**        | Prefer LTA over `borrowed_from` DFB when there are zero FIFO ops. If `reserve_back`/`push_back` still run → `borrowed_from` DFB or compute self-loop (Quasar — flag **SELF-LOOP-CANDIDATE**)              |
| **Class 5** — partials RMW via base ptr only                                | **Maybe**        | When partials are pure pointer RMW on an output shard, not LLK tile streams — LTA or read-write LTA. LLK-driven partials stay on ptr workarounds; on Quasar flag **SELF-LOOP-CANDIDATE** if PACK→UNPACK loop fits |
| **Single-ended synchronized packer** (`OUT`)                                | **No**           | Genuine `push_back` into resident output → `borrowed_from` DFB; on Quasar use compute self-loop (**SELF-LOOP-CANDIDATE**)                                                                                      |
| **Class 1** — cross-kernel producer→consumer FIFO                           | **No**           | Ordinary **DFB** (+ implicit sync on Quasar when canonical).                                                                                                                                                      |
| **Class 3** — scatter write into reserved region                            | **No**           | Strided/multi-DFB or disable split reader.                                                                                                                                                                        |
| Remote / NoC tensor access                                                  | **No**           | Use `**TensorAccessor`**, not LTA.                                                                                                                                                                                |


**LTA vs `borrowed_from` DFB vs compute self-loop:**


| Pattern                                            | End state                                                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| Sync-free borrowed · pointer-only                  | `**LocalTensorAccessor`**                                                      |
| Sync-free · private L1                             | `**ScratchpadSpec`**                                                           |
| Synchronized single-ended pack into resident shard | `borrowed_from` DFB (or compute self-loop on Quasar — **SELF-LOOP-CANDIDATE**) |
| Cross-kernel FIFO hand-off                         | **Ordinary DFB**                                                               |


**Porting heuristic:** When classifying a sync-free CB, ask **backing first**: borrowed tensor → **LTA**; private L1 → **scratchpad**. Do not keep a fabricated self-loop or DFB binding once LTA (or scratchpad) can replace it — record interim hacks in the port report for rollback ([port_patterns.md § Sync-free](temp_cb_dfb_reference_info.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround)).

**See also:** [Sync-free classify-it table](temp_cb_dfb_reference_info.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround); [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync) when sync is real but not FIFO-shaped.

---

## Scratchpad + semaphores (explicit sync)

> **Scope:** `ScratchpadSpec` is **private L1** that is **not** a DFB — no FIFO credits, no canonical producer/consumer endpoint model. Pair it with `**SemaphoreSpec`** bindings (`sem::name` in kernel) when kernels must coordinate who may read or write a region **without** pretending the handoff is `push_back` / `wait_front` on a linear FIFO.
>
> **Port recipe:** Scratchpad + semaphores is **autoportable** — audit as **Portable** → op rollup stays **GREEN** (standard port move per [`port/metal2.md`](../port/metal2.md)).

**Litmus:** Would expressing this buffer as a DFB force pointer surgery, fake FIFO credits, window jumps, or a fabricated self-loop endpoint? If yes → classify the **memory** as scratchpad and the **handshake** as semaphores (or program semaphores + stallwait inside compute), not as a hacked DFB.

**Why this matters for the audit:** Scratchpad + semaphores (or LTA where backing is borrowed) is the **preferred Quasar end-state** for Classes 2 and 4 ring/window/staging — **not** lasting DFB `evil_*` ptr surgery. On **WH/BH v1**, mechanical `evil_get/set` is allowed as **Portable (workaround)** with **Quasar redesign required**. The report must state which track was chosen.

**Where scratchpad + semaphores helps** (by class):


| Class | Pattern                               | Scratchpad role                                                                                                                | Semaphore role                                                                                                            |
| ----- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **2** | Window / stripe reuse                 | Reader (or compute) writes activation stripes at fixed layout offsets in scratchpad — **not** a sliding `fifo_wr_ptr` on a DFB | Reader signals “stripe N ready”; compute waits before consuming; reverse for pipelined reuse                              |
| **3** | Scatter staging before linear handoff | Multi-writer scatter lands in scratchpad first                                                                                 | Per-phase or per-writer semaphores before compute reads into a **separate** linear DFB                                    |
| **4** | Ring / packet staging                 | Staging buffer is scratchpad (e.g. ring_attention packet slots)                                                                | Per-packet or per-slot sems replace manual CB ring credits + `push_back` bookkeeping                                      |
| **5** | Non-LLK partials / spill buffer       | Named partials region without FIFO semantics                                                                                   | Phase sem between pack/spill/accumulate steps; on Quasar flag **SELF-LOOP-CANDIDATE** if canonical PACK→UNPACK on compute |
| **6** | Private scratch, alignment, temp      | Direct replacement for “CB that never FIFO’d” (`interleaved_to_sharded`, alignment padding)                                    | Usually none, or single barrier sem between setup and use                                                                 |
| **6** | **DM self-loop on Quasar**            | Runtime rejects DM PRODUCER+CONSUMER on same DFB — use scratchpad instead of self-loop DFB                                     | Cross-kernel or reader↔writer sems for the real producer/consumer                                                         |


**Scratchpad vs LTA vs DFB** (pick one end state):


| Backing                                | Synchronization                                    | End state                                             |
| -------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| Private (non-borrowed) L1              | Explicit cross-kernel or pipelined                 | `**ScratchpadSpec` + `SemaphoreSpec`**                |
| Borrowed tensor shard                  | Sync-free pointer access                           | `**LocalTensorAccessor`**                             |
| Borrowed or linear L1                  | Canonical tile FIFO between kernels                | **Ordinary DFB**                                      |
| Compute-only PACK→UNPACK loop (Quasar) | Tile-counter credits via compute self-loop binding | **SELF-LOOP-CANDIDATE** → compute self-loop on Quasar |


**What scratchpad does *not* replace:**

- **LLK tile streams** that still call `pack_tile(cb_id)` / expect a CB/DFB id — when LLK coupling cannot be avoided, audit recommends **DFB ptr/credit surgery** (**WEIRD-OK**) instead of scratchpad. On Quasar, flag **SELF-LOOP-CANDIDATE** only when PACK→UNPACK on compute fits without decoupled credits.
- **Cross-kernel canonical FIFO** — if the pattern is genuinely linear producer→consumer, use DFB (Class 1), not scratchpad.

**Host / kernel sketch:**

- Host: declare `ScratchpadSpec` on `ProgramSpec`; bind with `ScratchpadBinding` on each touching kernel; declare `SemaphoreSpec`(s) for each handshake edge.
- Kernel: scratchpad base via scratchpad accessor API; `sem::foo.wait()` / `sem::foo.post()` (or legacy equivalent during migration) at phase boundaries — **no** `get_local_cb_interface` field access on a fake CB.

**Audit report:** Class 2–5 CBs with **1xx `evil_get/set`** → **Portable (workaround)** (GREEN) — Notes: **undesirable but OK hack:** + **Quasar redesign required** + cite [whitelist](../shared/cb_dfb_api_whitelist.md). **Scratchpad + sems** Quasar end-state → **Portable** (GREEN). **LTA** end-state → **Portable (prereq: LTA)** (YELLOW).

**See also:** [port_patterns.md § Sync-free](temp_cb_dfb_reference_info.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb-interim-workaround); [migration_guide.md](../shared/migration_guide.md) (`ScratchpadSpec`, `SemaphoreSpec`).

---

## Architecture fork: strided scatter (Class 3)

Legacy **split-reader / multi-producer scatter** (e.g. conv2d tilize) often simulates a **2xx strided producer** by writing at computed offsets inside one CB backing store (`fifo_wr_ptr` jumps, scatter after `reserve_back`).

**The audit must branch on target architecture:**


| Target            | Port strategy                                                                                                                                                                                                                                                                                                                                                                          | Verdict                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **WH / BH (1xx)** | **Keep pointer scatter via DFB `evil_set_write_ptr` / `evil_get_*`** ([whitelist](../shared/cb_dfb_api_whitelist.md) — **in flight**) — same layout conventions as Gen1. **1xx has no native strided DFB producer**. Linear FIFO regions use mechanical CB→DFB; scatter paths stay pointer-driven through evil setters. | **WEIRD-OK** / **Portable (workaround)** — Quasar redesign required |
| **Quasar (2xx)**  | **Prefer 2xx strided multi-producer DFB** — host `stride_in_entries` (and related 2xx `DataflowBufferSpec` fields) models BRISC+NCRISC (or multi-endpoint) scatter into one logical buffer with canonical producer bindings. Replaces ptr surgery with first-class strided producer semantics when split-reader is retained.                                                           | **NEEDS-DESIGN-DECISION** until factory chooses strided DFB vs multi-DFB combine vs disable split reader |


**Alternatives on either arch:** separate DFBs per producer + combine in compute; disable split reader (simplest); scratchpad staging + semaphores then linear DFB push (see [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync)).

**Flagship op:** **conv2d** `tilize_in_reuse_split_reader`, `tilize_single_block_with_out_cb_update`.

**Audit report (Class 3 — record, do not leave open):**


| Target arch       | What to record in report                                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **WH / BH (1xx)** | Split-reader scatter → **`evil_set_write_ptr`** (**Portable (workaround)**). Note: no strided DFB on 1xx; Quasar redesign required. |
| **Quasar (2xx)**  | If split-reader kept: note **strided multi-producer DFB** as preferred end state, or document alternative (multi-DFB combine / disable split reader). |


---

## Issue taxonomy (kernel-side)

Six categories cover essentially all illegal audit findings. Map flowchart red boxes and audit verdicts here.

### Class 1 — Linear FIFO (canonical) ✅

**Recognition:** `reserve_back` → **`Noc::async_write`** (etc. from `noc.h`) → `push_back` on producer; `wait_front` → **`Noc::async_read`** → `pop_front` on consumer. No kernel `get_*_ptr`; no `get_local_cb_interface` field access.

**Verdict:** `PORTABLE` once ids become `DataflowBuffer` / `dfb::name`.

**Port strategy:**

- Host: normal `DFBBinding` PRODUCER/CONSUMER per kernel.
- Kernel: `DataflowBuffer dfb(dfb::foo)`; replace `cb_foo_id` CTAs with binding token where needed for LLK.
- Mechanical field reads only: `fifo_page_size` → `get_entry_size()`; `fifo_num_pages` → `get_total_num_entries()`.

**Examples:** Most matmul readers/writers, sdpa dataflow, index_fill reader path, ring_attention **writer** (consumer only).

---

### Class 2 — Window scratch (sliding logical view) 🔴

**Recognition:** Fixed-size CB reused as a **stripe/window**; reader and/or compute **jump** to offsets inside backing store (`fifo_wr_ptr` / `fifo_rd_ptr` assignment, `window_reuse_offset`). FIFO credits may still run, but **addresses are not FIFO-linear**.

**Verdict:** `NEEDS-DESIGN-DECISION`

**Why CB is wrong:** Producer/consumer credits imply contiguous front/back semantics; this pattern uses CB L1 as a **ring/window** with a shared layout convention.

**Port strategy (pick one — dual-track):**


| Option                               | When                                                     | Notes                                                                                                                                                                |
| ------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DFB + `evil_get/set` (1xx v1)**    | WH/BH mechanical port — no semantic change               | **Initial WH/BH default.** Map `fifo_*_ptr` jumps to `dfb.evil_set_*` / `evil_get_*` ([whitelist](../shared/cb_dfb_api_whitelist.md) — **in flight**). **Portable (workaround)**; Notes: **Quasar redesign required**. |
| **Scratchpad + semaphores**          | Quasar / redesign end-state; window with explicit sync   | Preferred **2xx** end-state. See [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync).                                                                    |
| **LocalTensorAccessor + semaphores** | Sync-free **borrowed** stripe/LUT reads only             | Pointer-only views on tensor-backed L1 — not live pipelined window reuse with credits.                                                                               |
| **Disable optimization**             | Simplest v1 bring-up                                     | Larger linear DFB, no reuse — back to Class 1                                                                                                                        |


**Flagship ops:** **conv2d** activation reuse (`conv_reader_common.hpp`, `conv_bmm_tilize.cpp` reader paths), **fused_swiglu** (scratch CBs — use `read_tile_value` once #49617 lands, not field reads).

---

### Class 3 — Scatter write into reserved region 🔴

**Recognition:** `reserve_back(N)` then writes at **computed offsets** within the region (not contiguous from `get_write_ptr()`). Often paired with Class 2.

**Verdict:** `NEEDS-DESIGN-DECISION` on Quasar when choosing strided DFB vs alternatives; **WEIRD-OK** / **Portable (workaround)** on WH/BH with `evil_set_write_ptr` scatter (see [Architecture fork](#architecture-fork-strided-scatter-class-3)).

**Port strategy (dual-track):**


| Option                                   | When                    | Notes                                                                                                                        |
| ---------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Raw pointer scatter via `evil_set_write_ptr` (WH / BH / 1xx)** | **1xx** split-reader    | **Initial WH/BH default** — keep Gen1 ptr pattern on DFB; **Portable (workaround)** + Quasar redesign required               |
| **Strided DFB (Quasar / 2xx only)**      | Split reader on **2xx** | **Preferred Quasar** when split-reader is retained — see [Architecture fork](#architecture-fork-strided-scatter-class-3). |
| **Scratchpad + semaphores → linear DFB** | Scatter is staging      | Writers scatter into scratchpad; sem handshake; push linearly into a separate DFB                                            |
| **Separate DFBs per producer**           | Two readers             | Document in report if chosen over strided DFB                                                                                |
| **Disable split reader**                 | Simplest v1             | Document in report if chosen                                                                                                 |


**Flagship op:** **conv2d** `tilize_in_reuse_split_reader`, `tilize_single_block_with_out_cb_update` (scatter `fifo_wr_ptr` on `tilized_in0_cb`).

---

### Class 4 — Credit / address decoupling 🔴

**Recognition:** FIFO **credits** posted (`push_back`) but **write pointer** does not match where data was written — or pointer advanced without credits (pack-untilize bookkeeping).

**Sub-patterns:**


| Pattern                                                     | Example                                  |
| ----------------------------------------------------------- | ---------------------------------------- |
| Push then rewind `wr_ptr`                                   | SDPA `cb_push_back_hold_wr_ptr`          |
| Pack writes L1, manual `wr_ptr +=` without `tiles_received` | Bilinear `llk_push_pages_bilinear`       |
| Manual ring wrap + per-packet `push_back`                   | `ring_attention_all_gather_async` reader |


**Verdict:** **WEIRD-OK** / **Portable (workaround)** on 1xx with DFB evil setters; `NEEDS-DESIGN-DECISION` for Quasar end-state.

**Important:** `push_back` is not a drop-in for bilinear's `llk_push_pages_bilinear`. `llk_push_tiles` also bumps `tiles_received` via `llk_push_to_brisc`; the bilinear helper only advances `fifo_wr_ptr` because nothing waits on those credits.

**No `*_hold_wr_ptr` DFB API.** SDPA uplift:

```cpp
dfb.push_back(n);
dfb.evil_set_write_ptr(/* rewound base */);  // wrap in kernel
```

**Audit scope for Class 4:** The audit **does not** track long-term LLK interface changes. For WH/BH v1, map to **`evil_set_*` / `evil_get_*`** ([whitelist](../shared/cb_dfb_api_whitelist.md) — **in flight**). When LLK coupling forces ptr surgery, verdict `**WEIRD-OK`**. Quasar: redesign — we are **not** pursuing `advance_write_ptr` / `push_back_credits_only` as near-term DFB APIs beyond evil setters.

> **Quasar: disable implicit sync on DFB ptr surgery.** When the audit records **DFB evil ptr manipulation** (1xx fallback still used on Quasar bring-up), **do not use Gen2 implicit sync** on that DFB. Use **explicit** FIFO ops or semaphores and opt out via Gen2 `disable_dfb_implicit_sync_for`. Prefer redesign so this callout is not needed.


| Option                                                  | Fits                                                                                                                                                |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DFB + `evil_get/set` (1xx v1)**                       | **Initial WH/BH default** — SDPA hold-wr, bilinear wr advance, ring staging — **Portable (workaround)**; Quasar redesign required                   |
| **Scratchpad + semaphores**                             | Preferred **2xx** for ring/window/staging (ring_attention, non-LLK pipes)                                                                           |
| **LocalTensorAccessor + semaphores**                    | Borrowed tensor, pointer-only, no LLK `pack_tile(cb_id)`                                                                                            |
| **Compute self-loop (Quasar only)**                     | Canonical PACK→UNPACK on one compute kernel — **not** decoupled ptr/credit. Flag **SELF-LOOP-CANDIDATE**.                                           |
| `**borrowed_from` DFB**                                 | Synchronized pack into resident shard still using LLK `pack_tile(cb_id)`                                                                            |


**Flagship ops:** **sdpa** (`compute_streaming.hpp`), **pool/upsample bilinear** (`bilinear.cpp`), **ring_attention_all_gather_async** (reader).

---

### Class 5 — In-place accumulator (L1-backed RMW) 🔴

**Recognition:** Same CB tiles read/modified/written across K or subblocks; save/restore `fifo_rd_ptr`/`fifo_wr_ptr` around `packer_l1_acc` / spill / bias paths.

**Verdict:** **WEIRD-OK** on 1xx with evil get/set; `NEEDS-DESIGN-DECISION` retained for Quasar classification.

**Not fixable by semaphores alone when LLK drives tile streams** — accumulator RMW across K may still require **`evil_get/set`** on WH/BH (**Portable (workaround)**) or Quasar redesign.

**Port gating:** **Not gated** on any LLK redesign. For v1 the audit records: **DFB evil save/restore** (1xx), **SELF-LOOP-CANDIDATE** (Quasar, canonical PACK→UNPACK), **LTA** (output-alias pointer-only), **scratchpad + semaphores** (non-LLK spill), or **disable `packer_l1_acc` / spill**.

**Port strategy (dual-track):**


| Option                                                 | Notes                                                                                        |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **DFB + `evil_get/set` save/restore (1xx v1)**         | **Initial WH/BH default** — replace `get_local_cb_interface(...).fifo_*_ptr =` with `dfb.evil_set_*` (**in flight**). Quasar redesign required |
| **Compute self-loop (Quasar only)**                    | True PACK→UNPACK tile loop on one compute kernel — flag **SELF-LOOP-CANDIDATE**              |
| **LocalTensorAccessor (read-write)**                   | Partials alias output shard; **pointer-only** RMW (e.g. `partials_cb_uses_output`)           |
| **Scratchpad + semaphores**                            | Non-LLK phased partials / spill tiles                                                        |
| **Dest-only accumulation**                             | When K fits in dest registers                                                                |
| **Disable `packer_l1_acc` / spill / fused bias paths** | v1 fallback if workarounds are too fragile                                                   |


**Flagship op:** **conv2d** `conv_bmm_tilize.cpp` (`matmul_partials_cb` save/restore throughout inner dim).

---

### Class 6 — Structural non-FIFO 🔴

**Recognition:** Buffer allocated as CB/DFB but **never** forms a real FIFO: `reserve_back` without `push_back`, no consumer, or DM self-loop on Gen2.

**Verdict:** `STRUCTURAL`

**Port strategy:**


| Pattern                                          | Fix                                                                                                                                                                                                                                |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Alignment / temp scratch                         | `**ScratchpadSpec`** — cannot overlay DFB/CB memory                                                                                                                                                                                |
| Tensor-resident shard · sync-free · pointer-only | `**LocalTensorAccessor`** — see [LTA](#localtensoraccessor-lta)                                                                                                                                                                    |
| Tensor-resident shard · synchronized pack/drain  | `borrowed_from` DFB (or compute self-loop on Quasar — **SELF-LOOP-CANDIDATE**)                                                                                                                                                     |
| Sync-free on **compute** · borrowed              | `**LocalTensorAccessor`** (read-only or read-write)                                                                                                                                                                                |
| Sync-free on **compute** · private L1            | `**ScratchpadSpec` + semaphores**                                                                                                                                                                                                  |
| DM self-loop on Quasar                           | `**ScratchpadSpec` + semaphores** — runtime rejects DM self-loop DFB; use scratchpad for the buffer and semaphores for the real producer/consumer handshake (see [Scratchpad + semaphores](#scratchpad--semaphores-explicit-sync)) |
| Pipelined scratch with cross-kernel handoff      | **Scratchpad + semaphores** — when both sides touch private L1 but sync is not FIFO-shaped                                                                                                                                         |


**Flagship ops:** **interleaved_to_sharded** scratch CB; **index_fill** in-place edit is **WEIRD-OK** (Class 1-ish with `wait_front` + `get_read_ptr` mutate) — optional hardening via `CoreLocalMem` + `scoped_lock`.

---

### Special cases

#### `SILENT-WRONG` — stream register pointer writes (out of scope)

**Op:** `deepseek_moe_gate` / `generalized_moe_gate` — `reconfig_cbs_for_mask` rewrites entire `LocalCBInterface` + `get_cb_tiles_*_ptr` writes.

**Port strategy:** **Out of scope** for this audit and for CB→DFB port-readiness gating. Needs a firmware-style reinit story on Quasar; do not block other op ports on moe gate resolution.

#### `QUASAR-BLOCKED` → `DOABLE` (post API)

**Ops (in scope):** Normalization Welford (`memory.h` + 6 kernels), sdpa/sdpa_decode, manual_seed, embedding_backward, moe_compute/gpt.

**Ops (out of scope — see above):** `deepseek_moe_gate` / `generalized_moe_gate`; **DeepSeek** `deepseek_prefill` combine/dispatch/post_combine_reduce and related kernels — track separately, not under this audit's GATE.

**Port strategy (in-scope ops):** Replace `CircularBuffer::read_tile_value` / `get_tile_address` with `DataflowBuffer` methods — WH/BH via [#49617](https://github.com/tenstorrent/tt-metal/pull/49617) (**in flight**); Quasar follow-up after that. Fix `memory.h` once, unblock six kernels. Separate GATE from field reads — these APIs are not `get_local_cb_interface` access.

#### `NEEDS-FIX` (mechanical, post-getters)


| Op                                        | Change                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------- |
| **fused_swiglu**                          | `read_tile_value(cb, 0, idx)` ×2 — delete manual `fifo_rd_ptr << 4` mailbox block |
| **pool_kernels_common**                   | `get_entry_size()`, `get_total_num_entries()`, `get_backing_size_bytes()`         |
| **quasar/** `*_interleaved_start_id*.cpp` | `dfb.get_entry_size()` instead of `fifo_page_size`                                |


---

## Op severity callouts

Use this table to prioritize kernel work across the repo (optional — not part of the per-op procedure in [How to use this doc](#how-to-use-this-doc)).


| Priority | Op / area                                                                         | Classes                          | Notes                                                                          |
| -------- | --------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| ⬜ N/A    | **deepseek_moe_gate / generalized_moe_gate**                                      | Firmware reconfig + SILENT-WRONG | **Out of scope** for this audit — not a port gate for other ops                |
| ⬜ N/A    | **deepseek_prefill** (combine, dispatch, post_combine_reduce)                     | QUASAR-BLOCKED + uint16          | **Out of scope** for this audit — track separately                             |
| 🔴 P0    | **conv2d** (+ `experimental/quasar/conv2d`)                                       | 2, 3, 4, 5                       | **Worst in-scope kernel-side offender** — Class 5/3 not port-gated on new APIs |
| 🔴 P1    | **sdpa** + sparse SDPA                                                            | 4 + QUASAR-BLOCKED               | Class 4: ptr/credit workarounds; `read_tile_value` on ctrl CB                  |
| 🔴 P1    | **pool/upsample bilinear**                                                        | 4                                | `llk_push_pages_bilinear` — ptr/credit workarounds (LLK/CB coupling)           |
| 🟠 P2    | **unified_routed_expert_ffn (fused_swiglu)**                                      | 2 (light)                        | → `read_tile_value` (NEEDS-FIX)                                                |
| 🟠 P2    | **normalization Welford family**                                                  | QUASAR-BLOCKED                   | Single `memory.h` fix                                                          |
| 🟡 P3    | **ring_attention_all_gather_async**                                               | 4 + field reads                  | Staging pipe; writer is clean                                                  |
| 🟡 P3    | **interleaved_to_sharded**                                                        | 6                                | Scratchpad migration                                                           |
| 🟢 P4    | **eltwise, bernoulli, uniform, reduction/generic, padded_slice, sharded readers** | 1 + GATE (field read)            | Mechanical getter swap — **must clear GATE before port merges**                |


### conv2d — why it is the reference bad example

One fused compute kernel (`conv_bmm_tilize.cpp`) combines:


| Case                                 | Mechanism                                         | Class |
| ------------------------------------ | ------------------------------------------------- | ----- |
| Activation reuse window              | `pass_to_the_next_image_width` sets `fifo_wr_ptr` | 2     |
| Split-reader scatter tilize          | `tilize_single_block_with_out_cb_update`          | 3     |
| Held-base restore before `push_back` | End of `tilize_in_reuse_split_reader`             | 4     |
| Partials L1 acc                      | Save/restore `matmul_partials_cb` rd/wr           | 5     |


**WH/BH v1 vs Quasar end-state:** WH/BH Class 2–5 → **`evil_get/set`** ([whitelist](../shared/cb_dfb_api_whitelist.md) — **in flight**), **Portable (workaround)** + Quasar redesign required. Quasar: prefer **scratchpad + semaphores** for Class 2 window reuse; Class 3 — **strided DFB**; Class 4/5 — redesign (self-loop / LTA / scratchpad) — do **not** treat evil_* as Quasar end-state.

---

## Report template

The agent fills this block after Steps 0–5. **List every in-scope CB** (from Step 4 inventory + Step 5 classification). One row per CB — do not split by kernel subsection. Include **Kernel(s)** so readers can find the usage; omit scan dumps, hit-line inventories, and other noise.

**Portable status** (use exactly one per arch column):

| Status | Meaning | Op rollup |
|--------|---------|-----------|
| **Portable** | Good to port — Class 1 linear FIFO, or **ScratchpadSpec + semaphores** Quasar end-state (**autoportable**) | GREEN |
| **Portable (prereq: LTA)** | Sync-free borrowed tensor view — **LocalTensorAccessor** must land in the port (host binding + kernel ctor) | YELLOW |
| **Portable (workaround)** | Non-canonical class — WH/BH proceeds with **`evil_get/set`** (or documented ptr hack); Notes **must** lead with **undesirable but OK hack:** + **Quasar redesign required** for Classes 2–5 | GREEN |
| **Blocked** | Cannot port — GATE, unresolved design, or STRUCTURAL without end-state | RED |
| **Blocked (runtime)** | Waiting on a [Runtime fix](#runtime-fixes-in-flight) (usually **2xx**; also WH until #49617 merges for scalar reads) | YELLOW (1xx clear) or RED |

**Overall rollup:** **GREEN** when no **Blocked**, no **Portable (prereq: LTA)**, and no unresolved GATE — **Portable (workaround)** rows are GREEN but must flag ptr hacks in Notes. **YELLOW** for LTA prereqs or 2xx runtime blocks with 1xx clear. **RED** for **Blocked** or GATE.

For every **Portable (workaround)** Class 2–5 row: Notes must include **undesirable but OK hack**, name the uplift (`evil_set_write_ptr`, `push_back` + `evil_set_write_ptr` for former `cb_push_back_hold_wr_ptr`, etc.), cite [cb_dfb_api_whitelist.md](../shared/cb_dfb_api_whitelist.md), and state **Quasar redesign required** with intended `2xx_endstate: scratchpad|LTA|self-loop|strided`.

```markdown
# CB→DFB Kernel Audit: `<op_name>` [factory: `<FactoryClass>` if not whole op]

**Date:** YYYY-MM-DD
**Op root:** `ttnn/cpp/ttnn/operations/<family>/<op>/`

**Scope:** `<factory helper or ProgramFactory>` → kernels: `path/to/kernel_a.cpp`, `path/to/kernel_b.cpp`, …

## Overall verdict: GREEN | YELLOW | RED

**Summary (1–2 sentences):** …

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` | 1 | `reader.cpp`, `compute.cpp` | Portable | linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_lut` | 6 | `compute.cpp` | Portable (prereq: LTA) | sync-free borrowed read → **LocalTensorAccessor** | Portable (prereq: LTA) | same |
| `cb_scratch` | 2 | `reader.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 2 → `evil_set_write_ptr` (whitelist §D, in flight); **Quasar redesign required**; `2xx_endstate: scratchpad` | NEEDS-DESIGN until redesign | scratchpad+sems |
| `cb_x` | 6 | `compute.cpp` | Portable (prereq: LTA) | — | Blocked (runtime) | needs `read_tile_value` on DFB (#49617 Quasar follow-up) |
| `cb_y` | 2 | `reader.cpp` | Portable (workaround) | **undesirable but OK hack:** Class 2 `fifo_wr_ptr` jump → `evil_set`; Quasar redesign required | Portable (after redesign) | strided DFB or scratchpad |
| `cb_hold` | 4 | `compute_streaming.hpp` | Portable (workaround) | **undesirable but OK hack:** was `cb_push_back_hold_wr_ptr` → `push_back` + `evil_set_write_ptr`; Quasar redesign required | NEEDS-DESIGN | — |
| `cb_z` | 4 | `compute.cpp` | Blocked | GATE: still `get_local_cb_interface(cb_z).fifo_wr_ptr` write — uplift to `evil_set_write_ptr` before merge | Blocked | same |

## GATE hits (must be empty to merge)

- (none | `file:line` — `get_local_cb_interface(...).<field>` — fix)

## Blocked on runtime (2xx rollup)

- (none | API + CBs affected, e.g. `read_tile_value` → `cb_x`)
```

---

## Example audit report: `layernorm` (sharded Welford path)

Filled example for the sharded Welford factory slice (`sharded_layernorm_factory_helpers.cpp`, `use_welford=True`).

```markdown
# CB→DFB Kernel Audit: `layernorm` [factory: sharded / Welford]

**Date:** 2026-07-09
**Op root:** `ttnn/cpp/ttnn/operations/normalization/layernorm/`

**Scope:** `sharded_layernorm_factory_helpers.cpp` (Welford branch) → kernels: `layernorm_sharded_welford.cpp`, `reader_mcast_sender_unary_sharded_ln.cpp`, `writer_unary_sharded_ln_rm_gb.cpp`, `writer_unary_sharded_ln.cpp`, `combine_welford.h`

## Overall verdict: YELLOW

Class 1 CBs with canonical FIFO are **Portable** once transfers use **`noc.h`**. Sites still on `get_*_ptr` + `dataflow_api.h` NOC are **NEEDS-FIX**. `cb_reciprocals` is **Portable (prereq: LTA)**.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0`, `cb_in1` | 1 | `layernorm_sharded_welford.cpp` | Portable | pre-add inputs, linear FIFO | Portable | — |
| `cb_in`, `cb_x_welford` | 1 | `layernorm_sharded_welford.cpp` | Portable | fp32 alias; two bindings or alias CTA | Portable | — |
| `cb_ex_partial` | 1 | `layernorm_sharded_welford.cpp` | Portable | partial mean/var | Portable | — |
| `cb_ex_external` | 1 | `layernorm_sharded_welford.cpp` | Portable | multicast partials | Portable | — |
| `cb_ex`, `cb_ex_global` | 1 | `layernorm_sharded_welford.cpp`, `combine_welford.h` | Portable | `combine_welford_partials` helper | Portable | — |
| `cb_transpose` | 1 | `layernorm_sharded_welford.cpp` | Portable | column layout workaround | Portable | — |
| `cb_xmm`, `cb_im`, `cb_out` | 1 | `layernorm_sharded_welford.cpp` | Portable | normalize pipeline | Portable | — |
| `cb_gamma`, `cb_beta` | 1 | `layernorm_sharded_welford.cpp`, `writer_unary_sharded_ln_rm_gb.cpp` | NEEDS-FIX | Class 1: rewrite `get_write_ptr() + offset` + legacy NOC → `Noc` from `noc.h` | NEEDS-FIX | same |
| `cb_reciprocals` | 6 | `layernorm_sharded_welford.cpp` | Portable (prereq: LTA) | sync-free LUT → **LocalTensorAccessor** (replaces `get_pointer_to_cb_data`) | Portable (prereq: LTA) | same |
| `cb_out` | 1 | `writer_unary_sharded_ln_rm_gb.cpp` | Portable | pack → output | Portable | — |
| `cb_partial` (+ reduce stage) | 1 | `reader_mcast_sender_unary_sharded_ln.cpp` | NEEDS-FIX | rewrite `get_read_ptr()` as NOC source → `noc.h` | NEEDS-FIX | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
```

---

## Recommended triage order (kernel fixes)

1. **GATE scan first** — zero unresolved `get_local_cb_interface` field access in op scope (hard port blocker)
2. **Out of scope (do not gate other ports):** moe gate, DeepSeek prefill kernels — see [Audit scope](#audit-scope-device-kernels-only)
3. **Mechanical GATE clears:** `fifo_page_size` / `fifo_num_pages` → existing DFB getters (~25 files)
4. **File to Almeet:** any remaining field read with no getter (`fifo_size`, `fifo_limit`, etc.) before port proceeds
5. **Runtime APIs:** land `read_tile_value` / `get_tile_address` on Quasar DFB → unblock in-scope compute kernels + Welford
6. **Quick wins:** fused_swiglu → `read_tile_value`
7. **Class 2–5 (1xx):** map ptr/credit reach-arounds to **`evil_get/set`** ([whitelist](../shared/cb_dfb_api_whitelist.md)); Notes: **undesirable but OK hack** + **Quasar redesign required** (`2xx_endstate: scratchpad|LTA|self-loop|strided`)
8. **Class 3:** record arch fork — **1xx:** `evil_set_write_ptr` scatter; **Quasar:** strided DFB if split-reader kept
9. **Quasar end-state (Classes 2–5):** redesign to scratchpad + sems / LTA / self-loop / strided — do **not** treat `evil_*` as lasting fix

---

## Scan commands (repo-wide metrics)

Use these for **inventory / prioritization** across the whole tree. Per-op audits use [Step 4](#step-4--classification-scans-on-scan_files-only) on `SCAN_FILES` only.

```bash
# Portable ptr accessors — do NOT mass-refactor
rg -l 'get_read_ptr\(|get_write_ptr\(|\.get_read_ptr\(\)|\.get_write_ptr\(\)' \
  ttnn/cpp/ttnn/operations --glob '*.{cpp,h,hpp}' | wc -l

# Illegal field access
rg -l 'get_local_cb_interface' ttnn/cpp/ttnn/operations --glob '*.{cpp,h,hpp}' | wc -l

# Tile scalar reads (blocked until DFB API on Quasar)
rg -l 'read_tile_value|get_tile_address|get_pointer_to_cb_data' ttnn/cpp/ttnn/operations --glob '*.{cpp,h,hpp}' | wc -l

# Silent-wrong
rg -l 'get_cb_tiles_acked_ptr|get_cb_tiles_received_ptr' \
  ttnn/cpp/ttnn/operations --glob '*.{cpp,h,hpp}'
```

---

## Relationship to other docs


| Concern                            | Host/spec ([`audit/metal2.md`](metal2.md)) | This audit (device kernels)                                                         |
| ---------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| SPSC / endpoint legality           | **In scope**                                                            | Cross-ref only — not gated here                                                     |
| Self-loop / sync-free CB           | `port_patterns.md`                               | Class 6 + DM vs compute fork                                                        |
| `borrowed_from` / aliased DFB      | `port_patterns.md`                               | Class 4/6 — synchronized pack into resident shard                                   |
| Scratchpad + semaphores            | `migration_guide.md`                             | Classes 2–6 — explicit sync without FIFO lies; DM self-loop replacement on Quasar   |
| LocalTensorAccessor                | `port_patterns.md`                               | Class 2/6 sync-free borrowed views; Class 5 output-alias partials when pointer-only |
| `get_local_cb_interface` in kernel | —                                                       | **GATE**                                                                            |
| `fifo_`* field R/W in kernel       | —                                                       | **GATE** — blocks port                                                              |
| Ptr offsets / scatter / partials   | —                                                       | Classes 2–5 → `evil_get/set` (1xx) + Quasar redesign flag                            |
| Self-loop / LTA / borrowed         | `port_patterns.md`                               | Class 4/6 strategies                                                                |


**Rule of thumb:** Host binding legality → [`audit/metal2.md`](metal2.md). Kernel `LocalCBInterface` / CB memory model → **this doc**.

---

## References

- [How to use this doc](#how-to-use-this-doc) — start here
- [Metal 2.0 op-porting recipe README](../../README.md) — full op port workflow (host audit + recipe)
- [audit/metal2.md](metal2.md) — host/spec feasibility (separate from this kernel audit)
- [port/metal2.md](../port/metal2.md) — port execution after audit
- [port_patterns.md](../shared/port_patterns.md) — binding patterns (self-loop, sync-free, borrowed)
- [migration_guide.md](../shared/migration_guide.md) — CB→DFB concepts
- [CB-to-DFB-flowchart.svg](../../human/CB-to-DFB-flowchart.svg) — decision flowchart
- DFB getters: [PR #49197](https://github.com/tenstorrent/tt-metal/pull/49197) (`get_total_num_entries`)
