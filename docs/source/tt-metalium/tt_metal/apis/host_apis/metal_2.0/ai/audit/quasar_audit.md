# Uplifting an Op to Quasar — Feasibility Audit

> **Status:** Living document (2026-07-19). **Scaffold** — the Quasar-uplift audit is
> young. Today it covers two checks (device-side CB→DFB redesign, non-zero-initialized
> semaphores); more Gen1-legal-but-Quasar-unsupported constructs will be added as they surface.

## Read this first

Metal 2.0 is a **single host API for both Gen1 (WH/BH) and Gen2 (Quasar)**, but a *working*
Metal 2.0 port targets **Gen1** — and a Gen1 port may lean on constructs that are legal on
WH/BH yet unsupported on Quasar. This audit finds them, ahead of the **Quasar uplift** step.

**When to run:** *after* an op is ported to Metal 2.0 (WH/BH), as the gate for uplifting it to
Quasar. This is distinct from the pre-port feasibility audit ([`metal2_audit.md`](metal2_audit.md)), which
decides whether the WH/BH Metal 2.0 port can proceed at all.

**Why the split exists.** Under the CB→DFB Gen1-parity plan, a `DataflowBuffer` has full
`CircularBuffer` parity on WH/BH, so the Metal 2.0 port swaps CB→DFB **mechanically** and defers
every "misused CB" cleanup to Quasar. The redesign those mechanical swaps defer — plus any other
Gen1-only construct — is what this audit surfaces.

**Output.** Per op (or ProgramFactory), a Quasar-uplift verdict: the constructs that must change
before the op runs on Quasar, each with the redesign or op-owner fix it needs. (Format will
formalize as the audit matures; for now, a checklist against the sections below.)

## Checks

### 1. Device-side CB → DFB redesign

Delegated to the standalone kernel audit:
**[`cb_dfb_quasar_audit_helper.md`](cb_dfb_quasar_audit_helper.md)**. It classifies every kernel
CB and flags those whose WH/BH `evil_*` mechanical port is a **workaround carrying Quasar
redesign debt** — the ones needing a scratchpad + semaphores, `LocalTensorAccessor`, compute
self-loop, or strided DFB on Quasar. It also covers **self-loop DFBs**, including the **DM
self-loop that Quasar rejects**.

Run that audit and roll up its **2xx (Quasar end-state)** column here: any CB left at
`NEEDS-DESIGN-DECISION` on the 2xx track is a Quasar-uplift blocker until its redesign is chosen.

### 2. Non-zero-initialized semaphores

**Rule.** Semaphores are default-initialized to zero. **A semaphore created with a non-zero
initial value does not port to Quasar** — it is temporarily available on WH/BH, but support is
slated for deprecation once remote-DFB support lands (see
[migration guide → SemaphoreSpec](../shared/migration_guide.md#semaphorespec)).

**Detect.** Any semaphore created with a non-zero initial value:
- Legacy: `CreateSemaphore(program, core_spec, initial_value, ...)` with `initial_value != 0`.
- Metal 2.0: a `SemaphoreSpec` carrying a non-zero initial value.

(A zero-init semaphore — the common case — ports fine; do not flag it.)

**Verdict.** Flag as a Quasar-uplift blocker. The op-owner must remove the non-zero-init
dependency before the uplift (or wait for remote-DFB support). This is an op-owner change, not a
mechanical port step.

## More checks land here

The Quasar-uplift surface is still being mapped. Add Gen1-legal-but-Quasar-unsupported constructs
here as uplift dogfooding surfaces them, and keep this list in sync with the Quasar-support notes
in the [migration guide](../shared/migration_guide.md).
