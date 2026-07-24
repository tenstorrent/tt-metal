# metal Runtime (runtime) — Baseline Decision Record

Open decisions, ownership questions, and gaps that must be resolved (or explicitly
accepted) in the §15 baselining session before the metal Runtime item definition and
first-cut requirements can move from DRAFT to baselined.

Status legend: **OPEN** (needs decision) · **PROPOSED** (recommendation pending
sign-off) · **RESOLVED** (decided; record outcome).

No `<NEIGHBORS>`, `<OPEN_DECISIONS>`, `<PRODUCT_REQS>`, `<ENVELOPE>`, or
`<SAFETY_README_URL>` inputs were supplied for this pass, so proposed owners below are
derived from code location only and must be confirmed — ownership is **not guessed to
close a gap**.

---

## 1. Ownership decisions

| ID | Question | Evidence | Proposed owner | Status |
|----|----------|----------|----------------|--------|
| OWN-01 | Is the **Fabric control plane** (`tt_metal/fabric/`) inside metal Runtime or a separate Fabric safety domain? | Code lives under `tt_metal/fabric/control_plane.cpp`, `fabric.cpp`; template lists Fabric as its own domain | Fabric domain owner (confirm) | OPEN |
| OWN-02 | Where is the exact **metal Runtime ↔ UMD** contract boundary (which guarantees UMD provides vs. what metal must re-check)? | `CODE:tt_metal/llrt/tt_cluster.cpp` wraps `tt::umd::Cluster` | UMD domain owner + Runtime | OPEN |
| OWN-03 | Who owns **fault reaction / safe-state / reset** on unrecoverable fault? | No auto reset in metal; `system_memory_manager.cpp` labels device "unrecoverable" | Safety Manager / system-level (confirm) | OPEN |
| OWN-04 | Who owns **hang/timeout detection policy** (enable + FTTI budget)? | Timeout opt-in in `system_memory_manager.cpp` | Runtime + system safety concept | OPEN |
| OWN-05 | Is the **JIT build / code generation** a qualified tool (ISO 26262-8 TCL)? | `CODE:tt_metal/jit_build/build.cpp` generates device code via SFPI | Tool-qualification owner | OPEN |
| OWN-06 | Ownership of **device-side dispatch kernels** (prefetch/dispatch) as part of the item | `CODE:tt_metal/impl/dispatch/kernels/` | Runtime (confirm) | PROPOSED |
| OWN-07 | Ownership of **multi-host MPI** coordination for safety configs | `CODE:tt_metal/distributed/multihost/` | Distributed/Runtime (confirm) | OPEN |

---

## 2. Gaps requiring implementation / requirement decisions

These correspond to `PROPOSED` requirements and `**GAP**` malfunctions.

| ID | Gap | Related req / FM | Recommendation | Status |
|----|-----|------------------|----------------|--------|
| GAP-01 | No default data-integrity (ECC/CRC) on buffer transfers | HLR-02c / FM-T2-02 | Decide allocation: runtime E2E check vs. rely on HW/UMD ECC; document in safety concept | OPEN |
| GAP-02 | Program-binary integrity validation is opt-in only | HLR-02d / FM-T2-03 | Mandate `TT_METAL_VALIDATE_PROGRAM_BINARIES` (or add binary hash check) in safety build | PROPOSED |
| GAP-03 | Device fault detection (Watcher) is default-off | HLR-04c / FM-T6-01/02 | Require Watcher enabled + configured in safety configuration (AoU-11) or make default in safety build | PROPOSED |
| GAP-04 | Dispatch operation timeout is default-off (infinite wait) | HLR-04b / FM-T5-01/02 | Require `TT_METAL_OPERATION_TIMEOUT_SECONDS` set to a bounded FTTI-derived value | PROPOSED |
| GAP-05 | No automatic safe state / reset on unrecoverable fault | HLR-05a / FM-T9-03 | Allocate safe-state to external mechanism (OWN-03) and/or add runtime reset hook | OPEN |
| GAP-06 | Fault containment across devices/CQs not formally established | HLR-04d, HLR-07a / FM-T9-04, FM-T7-01 | Perform DFA on shared UMD/context/CQ state | OPEN |
| GAP-07 | `TT_ASSERT` checks stripped in release builds | FM-T6-04 | Decide which asserts must be `TT_FATAL` for safety build | OPEN |
| GAP-08 | Unsafe env-var combinations can silently weaken safety | HLR-06c / FM-T8-04 | Define + validate the allowed `TT_METAL_*` safety envelope; enforce at init | PROPOSED |
| GAP-09 | No product SLA / FTTI budgets supplied | PERF-01/02/03, HLR-04b | Obtain product performance + FTTI requirements | OPEN |

---

## 3. Scope / envelope decisions

| ID | Question | Evidence | Status |
|----|----------|----------|--------|
| SCOPE-01 | Which architecture(s) are in the safety baseline (WH B0 / BH / Quasar)? | `runtime-ENV-04/05` | OPEN |
| SCOPE-02 | Which cluster/mesh topologies are in the safety baseline? | `runtime-ENV-06` | OPEN |
| SCOPE-03 | Fast dispatch, slow dispatch, or both in the safety baseline? | `runtime-ENV-07` | OPEN |
| SCOPE-04 | Which backend (Silicon only, or include Sim/Mock/Emule)? | `runtime-ENV-09` | PROPOSED: Silicon only |
| SCOPE-05 | Is multi-host (MPI) in the safety baseline? | `runtime-FR-14` | OPEN |
| SCOPE-06 | Which `TT_METAL_*` values constitute the supported safety envelope vs debug-only? | `CODE:tt_metal/llrt/rtoptions.cpp` | OPEN |

---

## 4. Definition-quality decisions

| ID | Question | Evidence | Status |
|----|----------|----------|--------|
| DEF-01 | Is the current API decomposition stable enough to baseline? Runtime "in flux". | `CODE:tt_metal/fusa-template.md` (lineage note) | OPEN |
| DEF-02 | Confirm `CANDIDATE` requirements reflect **intended** (not incidental) behavior | all `runtime-HLR-*` CANDIDATE | OPEN |
| DEF-03 | Which assumptions stay generic vs move to customer-specific tailoring | `runtime-assumptions-pre-post-deps.md` [VARIANT] tags | OPEN |
| DEF-04 | Provide specs/product requirements to upgrade `CANDIDATE`→`FIRM` | none supplied this pass | OPEN |

---

## 5. Decisions log (to be completed in §15 session)

| Date | Decision ID | Outcome | Decided by |
|------|-------------|---------|-----------|
| _tbd_ | | | |

---

## 6. Next actions

1. Assign real owners to OWN-01..07 (needs the `<NEIGHBORS>` / domain-ownership map).
2. Hold §15 baselining session to resolve GAP-01..09 and SCOPE-01..06.
3. Supply product requirements / specs / FTTI to upgrade `CANDIDATE`→`FIRM` and
   quantify PERF requirements.
4. Commission DFA for GAP-06 and the common-cause candidates in
   `runtime-boundary-and-interfaces.md` §5.
5. Decide tool-qualification path for the JIT build subsystem (OWN-05).
