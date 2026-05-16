# A.2.7 Evaluation — racecondition-hunt branch

## Recommendation recap

A.2.7 says: Document in `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` three things:
1. Fabric init lifetime (what states does fabric go through, what order)
2. Idempotency guarantees (can you call init twice? what happens?)
3. Cost model (how expensive is init/teardown?)

Plus: no tech report currently prescribes the soft-reset-between-tests model.

---

## Q1: Does the problem exist in racecondition-hunt?

**Yes — partially addressed but still largely present.**

The racecondition-hunt branch added a new subsection to the tech report that batch-t3k-ttnn-unit does not have:

**File**: `tech_reports/TT-Fabric/TT-Fabric-Architecture.md:927-932`
```
### Status Mailbox and Teardown Ordering

The status mailbox is also used by the host control plane during fabric
quiesce/restart cycles. When a workload completes and fabric firmware must be
re-initialized (e.g., between consecutive test iterations), the host must poll
the status mailbox of **every active ERISC channel** and confirm
`EDMStatus::TERMINATED` before overwriting that channel's L1 with new firmware.
...

For Tensix-side MUX cores, the same requirement applies: the host must poll for
`EDMStatus::TERMINATED` before overwriting MUX L1. Note that on Wormhole, the
MUX kernel may write `TERMINATED` to its mailbox before completing its
`close_finish()` drain routine — so the host should additionally halt the BRISC
via `assert_risc_reset_at_core()` after observing `TERMINATED`...
```

This is a **teardown ordering contract** — one narrow slice of what A.2.7 asks for. The three major gaps remain:

### Gap 1: Fabric init lifetime — NOT documented

The tech report has no section describing the fabric initialization state machine. The actual lifecycle is complex and undocumented:

- `SetFabricConfig()` sets global config (`fabric.cpp:449`)
- `MeshDevice::create()` triggers `configure_fabric()` on each device
- `configure_fabric_cores()` does per-channel soft-reset, firmware load, health checks (`fabric_init.cpp:91`)
- `quiesce_and_restart_fabric_workers()` does the between-program restart (`device.cpp:727`)
- Various guards: `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART` env var, `FabricTensixConfig::DISABLED` early-returns

None of these states or transitions appear in the tech report. Section 1.1.2 ("Control Plane", line 135) says the control plane "sets up and launches the Data Plane" but gives no lifecycle details.

### Gap 2: Idempotency guarantees — NOT documented

Zero mentions of "idempotent" or "idempotency" in the tech report (grep confirmed: 0 hits). The code has implicit idempotency assumptions:

- `configure_fabric_cores()` can be called on channels that are already running (it does soft-reset first) — `fabric_init.cpp:91-300`
- `SetFabricConfig()` is a global setter with no guard against double-call — `fabric.cpp:449`
- `quiesce_and_restart_fabric_workers()` has multiple early-return guards (`device.cpp:733-776`) but no doc on when re-calling is safe

### Gap 3: Cost model — NOT documented

The one hit for "overhead" in the tech report is about network management overhead (line 986), not init/teardown cost. There is no documentation of:

- How long `configure_fabric_cores()` takes (the AI-JOURNAL.md at line 294 mentions "healthy T3K: ~5-10s")
- The cost of `quiesce_and_restart_fabric_workers()` per program
- Why the branch added a `SIGALRM` timeout for init (undocumented)

### Gap 4: Soft-reset-between-tests model — NOT prescribed

The teardown ordering section (line 929) mentions "between consecutive test iterations" parenthetically but does not prescribe or explain the model. There is no section saying "here is how tests should manage fabric lifecycle" or "here is the expected init/teardown sequence for a test harness."

---

## Q2: Better, worse, or same as batch-t3k-ttnn-unit?

**Better — but only incrementally.**

The diff between the two branches' tech reports is exactly 6 lines (the "Status Mailbox and Teardown Ordering" subsection at lines 927-932). This subsection exists only in racecondition-hunt.

```
batch-t3k-ttnn-unit:  1026 lines, no teardown ordering section
racecondition-hunt:   1032 lines, has teardown ordering section
```

So racecondition-hunt has made a small dent in the documentation gap. The teardown ordering contract is genuinely useful — it documents a non-obvious invariant (poll ALL channels, not just master; halt BRISC after TERMINATED on Wormhole MUX). But it covers ~5% of what A.2.7 asks for. The three major gaps (lifetime, idempotency, cost) remain completely unaddressed in both branches.

---

## Q3: What would implementing A.2.7 look like concretely?

Add a new top-level section (e.g., "8 Fabric Lifecycle" or insert as "1.3 Fabric Initialization Lifecycle") to `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` with three subsections:

### 8.1 Initialization Lifetime

Document the state machine:
1. `SetFabricConfig()` — sets global topology + reliability mode (must precede `MeshDevice::create()`)
2. `MeshDevice::create()` → per-device `configure_fabric()` → `configure_fabric_cores()` — soft-resets each ERISC channel, loads firmware, runs health checks
3. **READY_FOR_TRAFFIC** — fabric is live, programs can run
4. Between programs: `quiesce_and_restart_fabric_workers()` — terminate → reconfigure → relaunch (the existing teardown ordering section would move here)
5. `MeshDevice::close()` → final teardown

Document the env-var escape hatches (`TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART`, etc.) and the degraded-mode flags (`fabric_stale_base_umd_channels_`, `is_fabric_degraded()`).

### 8.2 Idempotency Guarantees

Document:
- `SetFabricConfig()`: can be called multiple times but only before `create()`; calling after create is undefined
- `configure_fabric_cores()`: is re-entrant per device (does soft-reset first), but concurrent calls on the same device are not safe
- `quiesce_and_restart_fabric_workers()`: safe to call between programs; not safe to call concurrently with program execution

### 8.3 Cost Model

Document:
- Init time: ~5-10s on healthy T3K, potentially 30s+ with degraded channels (SIGALRM watchdog fires at 120s)
- Quiesce+restart time: estimate per-device cost
- Why init is expensive: ROM-postcode polling, soft-reset bounce, ring-sync warmup
- Guidance for test authors: init once per fixture, not per test case

---

## Q4: Interactions with race-condition fixes

The racecondition-hunt branch has extensive fabric init/teardown fixes that **increase the urgency** of A.2.7:

1. **The teardown ordering section (lines 927-932) was added by this branch.** It documents a contract that was previously only implicit in code. This shows the branch author already recognized the documentation gap — they just addressed only the most critical part.

2. **FIX QW2** (`257c42a96d5`): Guards `SetFabricConfig(DISABLED)` in TearDown when `FABRIC_1D` init threw. This is an idempotency edge case — calling teardown after a failed init. The tech report says nothing about this.

3. **FIX GAP-78** (`9a91fb6bf81`): Teardown ordering fix with regression tests. The code implements ordering invariants that the tech report's new teardown section only partially describes.

4. **`quiesce_and_restart_fabric_workers()` has grown enormously** in this branch (the function in `device.cpp` starting at line 727 has extensive diagnostic logging, multiple phases, env-var toggles). This complexity is invisible to anyone reading only the tech report.

5. **The `configure_fabric_cores()` return type changed** to `FabricCoresHealth` (`fabric_init.hpp:18-22`) — a structured result with `newly_dead_channels` and `skip_soft_reset_channels_used`. This is a public API contract that has no documentation.

**Net assessment**: The race-condition fixes make A.2.7 MORE important, not less. The branch has added significant complexity to the init/teardown path. Without corresponding documentation, the next developer working on fabric lifecycle will have to reverse-engineer all of this from code + AI-JOURNAL.md. The small teardown ordering section is a good start but covers only a fraction of the contract surface area.

---

## Key file references

- Tech report: `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` (1032 lines; teardown section at 927-932)
- Fabric init: `tt_metal/fabric/fabric_init.cpp:91` (`configure_fabric_cores`)
- Fabric init header: `tt_metal/fabric/fabric_init.hpp:18-57` (`FabricCoresHealth`, inject seam)
- SetFabricConfig: `tt_metal/fabric/fabric.cpp:449`
- Quiesce+restart: `tt_metal/impl/device/device.cpp:727` (`quiesce_and_restart_fabric_workers`)
- Device header lifecycle: `tt_metal/impl/device/device_impl.hpp:174-185`
- AI-JOURNAL.md init timing note: `AI-JOURNAL.md:294`
