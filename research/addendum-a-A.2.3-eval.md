# A.2.3 Evaluation: Add tt_fabric::Quiesce() API — racecondition-hunt branch

## 1. Does the problem described in A.2.3 exist in racecondition-hunt?

**Yes, and it is significantly more severe than in batch-t3k-ttnn-unit.**

The racecondition-hunt branch has an enormous quiesce implementation spread across
multiple files with no clean API boundary:

- **`Device::quiesce_and_restart_fabric_workers()`** in `tt_metal/impl/device/device.cpp:727`
  — 800+ lines (lines 727–1560+) implementing Phases 1, 2, 2.5, 3, 4, 5 of per-device
  fabric teardown and restart. Contains 40+ individual race-condition fixes (FIX RX, FIX AE,
  FIX AF, FIX AI, FIX PG, FIX AN, FIX Q, FIX N, FIX AD, FIX AR, etc.).

- **`MeshDeviceImpl::quiesce_internal()`** in `tt_metal/distributed/mesh_device.cpp:1571–1688`
  — 117 lines orchestrating multi-device quiesce with a three-sub-pass ETH launch strategy
  (Pass 1a/1b/1c + Pass 2) that sequences MMIO devices before non-MMIO to prevent
  simultaneous ETH handshake deadlocks.

- **`MeshDeviceFixtureBase::TearDown()`** in `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:260–321`
  — 60 lines of fixture teardown logic that decides whether to call `quiesce_devices()` or
  skip it based on `fabric_broken` state (FIX RX).

The PAUSE→DRAIN→RUN cycle from A.2.3 specifically does **not** exist in racecondition-hunt
as a standalone function. Instead, the branch replaced the concept with a much heavier
approach: full ERISC termination (Phase 2.5), reconfigure (Phase 3), relaunch (Phases 4/5).
There is no lightweight "flush in-flight messages" path.

The `FabricCommandInterface` class exists in the test tree at:
- `tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp:30` (class declaration)
- `tests/tt_metal/tt_fabric/common/fabric_command_interface.cpp:20` (implementation)

This class provides `pause_routers()`, `resume_routers()`, `wait_for_pause()`, etc., but:
1. It lives under `tests/` — not in the production `tt_metal/fabric/` directory
2. It has no `drain()` or `quiesce()` method
3. It is not used by `quiesce_internal()` or `quiesce_and_restart_fabric_workers()`


## 2. Better, worse, or the same as batch-t3k-ttnn-unit?

**Worse — but for structurally different reasons.**

**batch-t3k-ttnn-unit** has the PAUSE→DRAIN→RUN cycle explicitly implemented as a
standalone `drain_fabric_routers()` function at `multi_device_fixture.hpp:538–668`.
This is the exact code A.2.3 says should be moved to tt_fabric. That branch's
`quiesce_internal()` at `mesh_device.cpp:1422` is only 17 lines — it delegates to
submesh `quiesce_devices()` and drains CQs.

**racecondition-hunt** does not have `drain_fabric_routers()` at all. Instead, the
quiesce logic is far more complex:
- `device.cpp` grew from 1003 lines (batch) to 3917 lines (racecondition) — almost
  entirely due to `quiesce_and_restart_fabric_workers()` and its 40+ race-condition fixes.
- `quiesce_internal()` grew from 17 lines to 117 lines with the 3-sub-pass ETH launch.
- The fixture teardown grew by ~60 lines of fabric-broken guards.

**Critical firmware issue (shared by both branches):** The DRAIN command handler in
`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:2069` contains
`ASSERT(false); // not implemented`. This means `RouterCommand::DRAIN` will cause a
firmware hang (with WATCHER enabled) or silently skip (without asserts). The batch
branch's `drain_fabric_routers()` sends `RouterCommand::DRAIN` at line 637 — this
would assert/hang on debug builds. Both branches share this firmware gap.

Summary comparison:
```
Metric                           batch-t3k    racecondition-hunt
─────────────────────────────    ─────────    ──────────────────
device.cpp total lines           1003         3917
quiesce_and_restart_fabric_*     absent       800+ lines
quiesce_internal() lines         17           117
fixture TearDown complexity      moderate     high (broken-fabric guards)
standalone drain function        YES (150L)   NO
FabricCommandInterface           NO           YES (in tests/ only)
DRAIN firmware implemented       NO           NO
```


## 3. What would implementing A.2.3 look like concretely?

The recommendation is to create a `tt_fabric::Quiesce()` (or `tt_fabric::DrainAndRestart()`)
API that owns the entire PAUSE→terminate→reconfigure→relaunch→handshake cycle. Concretely:

**Step A: Move FabricCommandInterface into production code.**
Relocate `tests/tt_metal/tt_fabric/common/fabric_command_interface.{hpp,cpp}` to
`tt_metal/fabric/` (or `tt_metal/api/tt-metalium/experimental/fabric/`). This class
already has `pause_routers()`, `resume_routers()`, and `wait_for_state()`.

**Step B: Add a `Quiesce()` method to the ControlPlane or a new FabricLifecycle class.**
This method would encapsulate the current contents of `MeshDeviceImpl::quiesce_internal()`
and `Device::quiesce_and_restart_fabric_workers()`:
1. CQ drain (current Phase 1 of quiesce_internal)
2. Per-device ERISC termination (current Phase 2.5)
3. Fabric reconfigure + relaunch with the 3-sub-pass strategy (Passes 1a/1b/1c)
4. Handshake wait (Pass 2)
5. All race-condition guards (FIX RX, FIX AE, FIX AF, etc.)

**Step C: Implement `RouterCommand::DRAIN` in firmware.**
Fix `fabric_erisc_router.cpp:2069` — remove `ASSERT(false)` and implement actual packet
draining. Until this is done, the lighter-weight PAUSE→DRAIN→RUN cycle cannot work, and
the only option is the full teardown-and-restart path.

**Step D: Thin out consumers.**
- `MeshDeviceImpl::quiesce_internal()` becomes a thin wrapper: drain CQs, then call
  `tt_fabric::Quiesce(devices)`.
- `MeshDeviceFixtureBase::TearDown()` calls `mesh_device_->quiesce_devices()` without
  needing to know about fabric-broken guards (those move into the Quiesce API).
- `Device::quiesce_and_restart_fabric_workers()` moves entirely into the fabric layer.

**Step E: Expose a lightweight drain path once firmware DRAIN works.**
Add `tt_fabric::DrainInFlight()` that does only PAUSE→DRAIN→RUN (no ERISC restart),
usable for inter-test cleanup when the fabric is healthy.


## 4. Interactions with race-condition fixes already in this branch

The race-condition fixes create both **strong motivation** and **significant complexity**
for implementing A.2.3:

**Motivation (why A.2.3 is more important here than in batch):**
- The 40+ FIX tags in `quiesce_and_restart_fabric_workers()` represent hard-won knowledge
  about device ordering, ETH handshake deadlocks, relay-path brokenness, and UMD channel
  state. This knowledge is currently embedded in `Device` (impl layer) and `MeshDeviceImpl`
  (distributed layer). Moving it to a `tt_fabric::Quiesce()` API would:
  - Make it testable independently of GTest fixtures
  - Prevent fixture authors from needing to understand fabric internals
  - Centralize the ordering constraints (MMIO-before-non-MMIO, per-device STARTED barriers)

**Complexity (what makes it harder):**
- **FIX RX** (fixture `multi_device_fixture.hpp:280–312`): The "fabric broken, skip quiesce"
  guard currently lives in the test fixture. Moving it into the Quiesce API requires the API
  to accept/return a "fabric is too broken to quiesce cleanly" status rather than throwing.
- **FIX AE/AF** (mesh_device.cpp:1600–1672): The 3-sub-pass ETH launch strategy depends on
  mesh-level knowledge (which devices are MMIO, which are non-MMIO, relay latency
  characteristics). A `tt_fabric::Quiesce()` API would need access to this topology info,
  likely via ControlPlane.
- **FIX AN/PG** (device.cpp:1088–1133): Per-device relay-path-broken detection with retry
  logic creates per-device mutable state (`fabric_relay_path_broken_`, `fabric_channels_not_ready_`,
  `fabric_stale_base_umd_channels_`). The Quiesce API would need to read and write these
  flags, or the Device would need to expose them through a fabric-state interface.
- **FIX TI/RZ** (device.cpp:429–435, 693–704): Stale-base-UMD and ring-sync-timeout flags
  are cleared during `configure_fabric()` (called from Phase 3 of quiesce). The Quiesce API
  must ensure these resets happen at the right point in the sequence.

**No blocking conflicts.** None of the race-condition fixes fundamentally prevent extracting
the quiesce logic into a `tt_fabric::Quiesce()` API. The fixes add ordering constraints and
error-recovery paths, but these are all expressible within a fabric-owned API. The main risk
is that the extraction is a large refactor touching ~1000 lines of subtle, interleaved code.

**Recommendation:** A.2.3 applies and is arguably more urgent on this branch than on batch,
because the complexity is higher and the blast radius of getting quiesce wrong is larger.
However, the implementation should be done as a follow-up PR (not within the race-condition
hunt itself) to avoid destabilizing the existing fixes. The firmware DRAIN gap
(`fabric_erisc_router.cpp:2069`) should be addressed first or in parallel, as it blocks the
lightweight drain path that A.2.3 envisions.
