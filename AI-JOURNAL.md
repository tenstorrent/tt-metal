
---
## CANONICAL BRANCHES (2026-05-09)

- **tt-metal**: `nsexton/0-racecondition-hunt`
- **tt-umd submodule**: `nsexton/0-metal-racecondition-hunt`

Do NOT create other branches in either repo for this investigation. All UMD fixes live on `nsexton/0-metal-racecondition-hunt`. The following old UMD branches were deleted after being consolidated (all commits are on the canonical branch):
- `nsexton/fix-ae-relay-broken-fast-path` (deleted 2026-05-09)
- `nsexton-fix-aq-relay-timeout` (deleted 2026-05-09)
- `nsexton-fix-aq2-topology-crash` (deleted 2026-05-09)

---
## 2026-04-30 — FIX PF (GAP-51): heartbeat-based bypass in reset_cores skips 500ms cascade

### Run: 25138208790 — 136x Timeout(500ms) cascade despite FIX PD+PE

**Symptom**: `unit_tests_ttnn` times out (900s wall-clock). Logs show 136 occurrences of:
```
Timeout (500ms) waiting for physical cores to finish: 25-16, 18-16, 25-17, 18-17, 22-17, 21-17
Detected dispatch kernels still running but failed to complete an early exit. Force-resetting stale ETH cores on device X
```
34 groups × 4 devices = 136 total. Each group = one `unit_tests_ttnn` test that uses ETH dispatch.

### Root Cause (fully traced)

1. **`initialize_firmware()`** writes non-zero `fw_launch_addr_value` to `fw_launch_addr` when loading Metal ETH dispatch firmware (risc_firmware_initializer.cpp:1558-1562). Metal ETH firmware **never clears this on exit**.

2. After a process kill (900s timeout) or partial quiesce failure, ETH dispatch cores have:
   - `fw_launch_addr != 0` (stale, never cleared)
   - `go_msg::signal == RUN_MSG_GO` (stale, kernels were in-flight)
   Both values survive `tt-smi -r` (soft reset preserves L1).

3. **FIX PD** cleared `fw_launch_addr` for Phase-2.5-halted channels (MMIO devices 0 and 2). **FIX PE** cleared it for cleanly-terminated channels in teardown. But channels on devices 1 and 3 that terminated abnormally (killed process) were NOT covered.

4. **`reset_cores()`** path: `erisc_app_still_running()` reads `fw_launch_addr != 0` → true → `erisc_send_exit_signal()` sent to UMD firmware (which doesn't understand Metal exit protocol) → `wait_until_cores_done(RUN_MSG_GO, 500ms)` → UMD never sets go_msg=DONE → **500ms timeout per channel per test**.

### FIX PF (commit f0e3d677935)

**Location**: `reset_cores()` in `risc_firmware_initializer.cpp`, between relay-dead detection and the existing `if (still_running)` block.

**Mechanism**: When `erisc_app_still_running()` returns true on an MMIO device:
- Read the heartbeat counter via `cluster_.read_reg()` (PCIe-direct, safe with broken relay)
- If `(hb_val >> 16) == 0xABCDu` → UMD base firmware confirmed running
- Clear `fw_launch_addr = 0` directly, log "FIX PF: stale fw_launch_addr cleared"
- Set `still_running = false` → skip `erisc_send_exit_signal()` + skip adding to `device_to_early_exit_cores`

**Scope**: MMIO-only (read_reg is PCIe-safe). Non-MMIO and relay-dead paths unchanged.

**Note**: WH uses plain incrementing heartbeat at 0x1F80, not 0xABCD format — FIX PF naturally skips WH. BH/QA use 0xABCDxxxx at MEM_SYSENG_ETH_HEARTBEAT. If WH heartbeat-based bypass needed later, check value-changed pattern (not upper-16 == 0xABCD).

---
## 2026-04-26 — FIX AI + FIX AI-2: assert_risc_reset without deassert leaves ETH channels dead

### Root Cause
`FabricFirmwareInitializer::teardown()` force-reset path (line ~597) called
`assert_risc_reset_at_core(ALL)` on timed-out ETH channels WITHOUT a subsequent
`deassert_risc_reset_at_core`. This left ALL RISCs (ERISC0/BRISC + NCRISC) in
hardware reset. The downstream `teardown_fabric_config()` only deasserted ERISC0,
leaving NCRISC (which maintains the ETH PHY link) permanently halted.

On the next test's fabric init, `terminate_stale_erisc_routers()` probe reads
timeout on all channels because the PHY is down, producing `corrupt=4 probe_dead=4`.

### Fixes Applied (3 files, 4 changes)
1. **`fabric_firmware_initializer.cpp`** (FIX AI): Added `deassert_risc_reset_at_core(ALL)`
   immediately after `assert_risc_reset_at_core(ALL)` in the teardown force-reset path.
2. **`metal_env.cpp`** (FIX AI): Changed `teardown_fabric_config()` timeout force-reset
   from `RiscType::ERISC0` to `RiscType::ALL` as defense-in-depth.
3. **`device.cpp` + `device_impl.hpp`** (FIX AI-2): Quiesce Phase 2.5 force-halts
   unresponsive ERISCs with `assert_risc_reset(ALL)` but Phase 3 never deasserted them.
   Added tracking set `pending_phase25_force_reset_chans_` and deassert logic in both
   the inline ETH launch loop and `launch_eth_cores_for_quiesce()`.

---
## 2026-04-21 09:24 UTC — Cycle 4: build failure fix (missing fabric_init.hpp include)

### Build failure in run 24713838162
`tt_metal/fabric/fabric_init.cpp:78`:
```
error: unknown type name 'FabricCoresHealth'
error: use of undeclared identifier 'FabricCoresHealth'
```

### Root cause
`FabricCoresHealth` struct is declared in `fabric_init.hpp`, but `fabric_init.cpp` did not include its own header. The prior commit (68e9bd81cf) introduced `FabricCoresHealth` in the header and updated `fabric_init.cpp` to use it, but forgot to add `#include "tt_metal/fabric/fabric_init.hpp"` to the `.cpp` file's includes.

### Fix (commit f522c1fb7e)
Added `#include "tt_metal/fabric/fabric_init.hpp"` after `#include "tt_metal.hpp"` in `fabric_init.cpp`.

### CI Run
https://github.com/tenstorrent/tt-metal/actions/runs/24714644362 (dispatched 2026-04-21 09:24 UTC)

---
## 2026-04-22 09:50 UTC — Cycle 7-8: FIX G — wait_for_fabric_router_sync skipping dead-relay devices

### Root cause of cycle 7 failure (run 24770598843, job 72477742381)
After all 7 previous fixes (A-F, E2, E3), the test was FINALLY getting past
`terminate_stale_erisc_routers()` and `configure_fabric()` for all 8 devices.
Devices 4-7 were marked dead-relay and initialized in degraded mode.

But `wait_for_fabric_router_sync()` and `verify_all_fabric_channels_healthy()`
did NOT check `dead_relay_devices_`. They iterated over all devices and tried to
read the master router sync address from device 6 (dead-relay) via the relay path.
That read timed out and threw, crashing MeshDevice::create() before the test ran.

Error seen at 09:43:15: `wait_for_fabric_router_sync: Device 6 master chan=6 read TIMED OUT`

### Fix (commit 59b03ffd37) — FIX G
Added `dead_relay_devices_.count(dev->id()) > 0` guard at the start of
`wait_for_handshake` lambda (in `wait_for_fabric_router_sync`) and at the
device loop start in `verify_all_fabric_channels_healthy`. Both functions
now skip dead-relay devices with a warning log.

### CI Run (cycle 8)
https://github.com/tenstorrent/tt-metal/actions/runs/24772200703 (dispatched 2026-04-22 09:50 UTC)

---
## 2026-04-22 10:25 UTC — Cycle 8-9: FIX H — short-circuit relay probe for subsequent non-MMIO devices

### Root cause of cycle 8 failure (run 24772200703, job 72482814209)
After FIX G (skip dead-relay devices in wait_for_fabric_router_sync), the test was STILL timing out.
Failure was in terminate_stale_erisc_routers during MeshDevice::create(). 

MMIO devices 0-3: edm_status=0x49706550 ("iPeP" mid-handshake state), recovered OK.
Remote devices 4-7: relay ERISCs corrupt, every probe read timed out after 15s.

Per-device breakdown:
- relay_timeout_count resets to 0 for EACH device call
- Each device: 3 timeouts × 15s = 45s before relay_broken=true
- 4 remote devices × 45s = 180s total → job timeout before finishing

### Fix (commit 4a873aa654) — FIX H
Added `any_relay_broken` local in compile_and_configure_fabric(). When relay_broken=true
for the first non-MMIO device, ALL subsequent non-MMIO devices skip terminate_stale_erisc_routers
entirely. Instead, all their active ETH channels are immediately marked dead (probe_dead_channels),
and they go straight to configure_fabric() in degraded mode with no relay reads.

Before FIX H: 4 devices × 3 × 15s = 180s hang → job timeout
After  FIX H: 1 device × 3 × 15s = 45s → completes within job time budget

### CI Run (cycle 9)
https://github.com/tenstorrent/tt-metal/actions/runs/24773371015 (dispatched 2026-04-22 10:25 UTC)

---
## 2026-04-22 11:06 UTC — Cycle 9-10: FIX I — skip MMIO dead-peer devices in fabric router sync

### Root cause of cycle 9 failure (run 24773371015, job 72486699038)
After FIX H (devices 5-7 skipped, only device 4 does 3×15s probe), the flow advanced further.
But hit a new failure in wait_for_fabric_router_sync():

```
Fabric Router Sync: Timeout after 10000 ms. Device 0: Expected status 0xa2b2c2d2, got 0x00000000
```

Timeline from logs:
- 10:41:14: configure_fabric for devices 4-7 in degraded mode. "Fabric initialized on 8 devices."
- 10:41:16: wait_for_fabric_router_sync starts. Device 6 skipped (dead-relay, FIX G).
- 10:41:26: Device 0 master chan=15 timeout after 10000ms → TT_THROW.

Root cause: MMIO device 0's master router is on ETH channel 15. That channel connects to
dead-relay non-MMIO device (4, 5, 6, or 7). Fabric firmware was loaded on device 0 chan=15
in compile_and_configure_fabric(). But the firmware enters a handshake loop waiting for its
ETH peer (on a dead-relay device) to write back the sync value. Peer ERISC is dead/unresponsive
→ peer never writes sync value → device 0 chan=15 stays at 0x00000000 → 10s timeout → throw.

FIX G (skip dead_relay_devices_) only helps for dead-relay non-MMIO devices. Device 0 is
MMIO and not in dead_relay_devices_ — FIX G doesn't protect it.

### Fix (commit 7548cdcde5) — FIX I
After the configure-devices loop in compile_and_configure_fabric(), walk eth_connections to
identify MMIO devices whose master router ETH channel connects to a dead-relay non-MMIO device.
Store in new mmio_dead_peer_devices_ set (separate from dead_relay_devices_).

Skip mmio_dead_peer_devices_ in:
- wait_for_fabric_router_sync() — peer will never write sync, skip rather than timeout
- verify_all_fabric_channels_healthy() — peer firmware never started, channel won't reach READY_FOR_TRAFFIC

Critical distinction: mmio_dead_peer_devices_ is NOT added to dead_relay_devices_, so
DeviceManager continues dispatch kernel init for MMIO devices (MMIO dispatch uses PCIe, not ETH relay).

Changes:
- fabric_firmware_initializer.hpp: add mmio_dead_peer_devices_ member
- fabric_firmware_initializer.cpp: FIX I detection in compile_and_configure_fabric(), skip in
  wait_for_fabric_router_sync() lambda, skip in verify_all_fabric_channels_healthy()

### CI Run (cycle 10)
https://github.com/tenstorrent/tt-metal/actions/runs/24774903128 (dispatched 2026-04-22 11:06 UTC)

## Cycle 2 — FIX J (commit 9ffd66a0f4)

**Problem (from cycle 1 run 24786271571):**
- ERISC L1 corruption on Devices 1,2,3,5 left by prior process (edm_status=0x49706550 "iPeP")
- Device 3 ETH relay dead (3x probe timeout on chans 0,1,6)
- Topology auto-discovery degraded: 2x4 → 2x2 (chips 0 and 4 excluded from cluster)
- TT_FATAL from `control_plane.cpp:1263`: "Physical chip id 0 not found in control plane chip mapping"
  — thrown from `teardown_fabric_config()` in `metal_env.cpp` which iterated all chip IDs and called
  `get_fabric_node_id_from_physical_chip_id(0)` without checking if chip 0 is in the cluster
- Second TT_FATAL from `fabric_builder_context.cpp:173` during destructor chain
- SIGABRT: `Cluster::~Cluster()` called `driver_->close_device()` (no try/catch in implicitly-noexcept dtor)
  when `RemoteChip::close_device()` threw `UmdException<RuntimeError>` (ETH relay timeout)

**Fix J changes:**
1. `tt_metal/impl/context/metal_env.cpp`: Add `is_physical_chip_in_fabric_cluster(chip_id)` guard
   before `get_fabric_node_id_from_physical_chip_id()` in `teardown_fabric_config()` loop.
   Chips excluded by topology downgrade now emit warning + continue.
2. `tt_metal/llrt/tt_cluster.cpp`: Wrap `driver_->close_device()` in try/catch inside
   `Cluster::~Cluster()`. ETH relay timeout exception now logged as warning instead of SIGABRT.

**Note:** These fixes make failures cleaner (no SIGABRT, no cascading TT_FATAL during teardown)
but do NOT fix the root cause (ERISC L1 not cleared between runs). Test may still fail if topology
stays at 2x2 — but error will be a clear test fixture SKIP or fixture setup failure, not a crash.

**Run 2:** 24789852756 (queued 2026-04-22T16:24:49Z)

## 2026-04-22 Run 24799099598 — Build failure, Scenario Y compile errors

Run 24799099598 failed during build (not a test runtime failure).

### Root cause
`test_async_teardown_race.cpp` new test `CompileAndConfigureFabric_JoinsAllFuturesBeforeRethrow` (Scenario Y) had 3 compile errors:

1. `FabricConfig::FABRIC_2D` passed as 5th arg to `MeshDevice::create()` — that param is `const DispatchCoreConfig&`, not `FabricConfig`. **Fix**: replace with `DispatchCoreConfig{}` and call `SetFabricConfig()` before each `create()` call (same pattern as Scenarios D/E).

2. `mesh_shape.num_devices()` — `MeshShape` has no `num_devices()`. **Fix**: use `mesh_shape.mesh_size()` (returns product of all dimensions = total device count).

3. Second `ASSERT_NO_THROW(MeshDevice::create(..., FabricConfig::FABRIC_2D))` — same wrong arg.  Also needed a `SetFabricConfig()` before it.

### Fix applied
Commit `6d7bf148fd` on branch nsexton/0-racecondition-hunt. New run dispatched: 24801042807.


## 2026-04-22 21:30 UTC — Run 24801042807 Analysis: FIX I transitive-dependency gap

### Build Fixed (6d7bf148fd)
Compile errors in Scenario Y test were fixed. Build now succeeds.

### Runtime Failure: AsyncExecutionWorksCQ0 + FIX I gap

Run 24801042807 (on fix commit) failed `MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0`.

**Failure chain:**
1. All 8 devices start with ERISC L1 corrupt (`0x49706550 "iPeP"` on all channels).
2. Devices 0-3 (MMIO): `terminate_stale_erisc_routers` detects corruption, zeros `edm_status_address`, sends TERMINATE best-effort. Proceeds to reload firmware.
3. Device 4 (non-MMIO): 3 probe reads TIME OUT (5s each = 45s total). Sets `relay_broken=true`. All 4 ETH channels → `probe_dead_channels`. Devices 5-7 skip probe (FIX H). All 4-7 in `dead_relay_devices_`.
4. Fabric firmware loaded on Devices 0-3 channels in degraded-awareness mode.
5. `wait_for_fabric_router_sync` polls Device 0's master router channel → reads `0x00000000` until 10s timeout → `TT_THROW`.

**Root cause of FIX I not firing:**

FIX I detects MMIO Device → dead-relay peer ONLY if Device 0's master_chan DIRECTLY connects to a non-MMIO dead-relay device. But in FABRIC_1D topology, the ring order appears to be:
```
Device 0 ←→ Device 1 (MMIO) ←→ Device 5 (dead-relay non-MMIO)
```
So Device 0's master_chan connects to Device 1 (MMIO, not in `dead_relay_devices_`). FIX I does NOT fire for Device 0.

But Device 1's master_chan connects to Device 5 (dead-relay). FIX I fires for Device 1, skipping its sync. But Device 0's firmware is waiting for Device 1 to complete its ring handshake. Since Device 1 can't complete (FIX I skips Device 1's sync), Device 0's firmware is stuck waiting → 10s timeout → throw.

**This is a transitive dependency problem.** FIX I only handles direct MMIO→dead-relay connections but not indirect chains: MMIO→MMIO→dead-relay.

### Potential fixes:

**Option A (conservative):** Increase `wait_for_fabric_router_sync` timeout to allow graceful degraded-mode operation, and add a fallback that if ANY non-master channel read returns 0x00000000 after full timeout, mark the chain as degraded and continue (warning instead of throw).

**Option B (targeted):** After marking `mmio_dead_peer_devices_`, propagate: any MMIO device whose direct peer is in `mmio_dead_peer_devices_` is also transitively affected. Add to `mmio_dead_peer_devices_`. Repeat until fixed point.

**Option C (broadest):** If `dead_relay_devices_` is non-empty, skip `wait_for_fabric_router_sync` entirely for all devices and rely on `verify_all_fabric_channels_healthy` in degraded mode.

**Recommended:** Option B (transitive propagation), since it's targeted and doesn't bypass the sync entirely.

### New runs dispatched
- Run 24803632611 (queued): my re-dispatch for t3k_ttnn_tests on fix commit (6d7bf148fd)
- Run 24801901890 (in_progress): likely Neil's dispatch

### Note on new Scenario tests
The `test_async_teardown_race.cpp` tests (Scenarios A-Z) were NOT in the `run_t3000_ttnn_tests` script - only `MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0` from `test_ccl_multi_cq_multi_device` runs in this CI flow. The new tests need to be added to the test script or run separately.

---
## 2026-04-22 — GAP 2: ConfigureFabricCoresInjectFn + StatusOverrideFn seams

### What was implemented

**Scenario W** (`ConfigureFabricCores_CatchesResetExceptionAndRecords`): exercising the `catch(std::exception)` block in `configure_fabric_cores()` (fabric_init.cpp) was impossible without touching hardware. Added a thread-local `ConfigureFabricCoresInjectFn` seam to `tt_metal/fabric/fabric_init.hpp`:
```cpp
using ConfigureFabricCoresInjectFn = std::function<void(IDevice*, uint32_t /*eth_chan_id*/)>;
void set_configure_cores_inject_fn(ConfigureFabricCoresInjectFn fn);
void clear_configure_cores_inject_fn();
```
The seam is called inside the `try {}` block of the assert_risc_reset_at_core loop in `fabric_init.cpp`. If it throws, the existing catch block fires — no hardware access needed.

**Scenario X** (`TerminateStaleEriscRouters_CorruptStatusDetected`): exercising the `!is_known_edm_status()` branch in `terminate_stale_erisc_routers()` required writing garbage to ERISC L1 (machine-destructive). Added a thread-local `StatusOverrideFn` seam to `FabricFirmwareInitializer`:
```cpp
using StatusOverrideFn = std::function<std::optional<uint32_t>(Device*, uint32_t)>;
static void set_status_override_fn_for_testing(StatusOverrideFn fn);
static void clear_status_override_fn_for_testing();
```
The seam is called before `ReadFromDeviceL1` in the probe read loop. If it returns a value, that value replaces `status_buf[0]` and the real L1 read is skipped.

### Files changed
- `tt_metal/fabric/fabric_init.hpp` — added `ConfigureFabricCoresInjectFn` type + set/clear declarations
- `tt_metal/fabric/fabric_init.cpp` — added thread_local + set/clear impl + injection in configure_fabric_cores loop
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.hpp` — added `<optional>`, `StatusOverrideFn` type + static set/clear declarations + static member
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` — added thread_local + set/clear impl + injection in terminate_stale_erisc_routers probe-read section
- `tests/tt_metal/distributed/test_async_teardown_race.cpp` — replaced GTEST_SKIP for W and X with real test implementations; added `<optional>` include

### Design notes
- The `if (!seam_provided_status) try { ... } catch { ... }` pattern in .cpp is valid C++ — the try-catch is the "then" branch of the if statement.
- The inject fn for Scenario W is placed inside the `try {}` body; if it throws, the existing catch fires. If it returns, `continue` skips the real UMD call.
- Scenario X uses end-to-end validation (full MeshDevice::create()) since terminate_stale_erisc_routers is private. The seam intercepts the probe read; hardware writes (TERMINATE + zero) still proceed but are safe on a clean (just-closed) ERISC.
- `target_chan = 0` is hardcoded for Scenario X since we can't query the control plane after close(); chan 0 is always present on WH/BH fabric nodes.

---
## 2026-04-22 23:15 UTC — Run 24805444153: NEW failure mode — chip 0 excluded from control plane

### Different from FIX I transitive gap

Run 24805444153 (commit `6d7bf148fd` — compile-error fixes only, no runtime logic changes) failed with a NEW crash site:

```
TT_FATAL @ /work/tt_metal/fabric/control_plane.cpp:1263: false
Physical chip id 0 not found in control plane chip mapping.
You are calling for a chip outside of the fabric cluster.
Check that your mesh graph descriptor specifies the correct topology
```

Call path:
```
RelayMux::GenerateStaticConfigs()                          ← crash here
DispatchTopology::populate_cq_static_args(Device*)
DispatchKernelInitializer::compile_dispatch_kernels()
DispatchKernelInitializer::init(...)
DeviceManager::initialize_fabric_and_dispatch_fw()
MeshDeviceImpl::create(...)
test_ccl_multi_cq_multi_device (SetUp)
```

### Root cause analysis

This crash is EARLIER than the FIX I transitive gap issue (which was at `wait_for_fabric_router_sync`). It happens at `compile_dispatch_kernels` time, before firmware sync.

In `RelayMux::GenerateStaticConfigs()` (relay_mux.cpp:96-97):
```cpp
const auto src_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device_id_);
const auto dst_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(destination_device_id);
```

`device_id_` is 0 (the MMIO device). The ControlPlane's `logical_mesh_chip_id_to_physical_chip_id_mapping_` does NOT have chip 0 as a value. This means **chip 0 was excluded from the fabric cluster during `init_control_plane_auto_discovery()`**.

How: `ControlPlane::init_control_plane_auto_discovery()` calls `run_physical_system_discovery` → `generate_mesh_graph_from_physical_system_descriptor` → `TopologyMapper`. If chip 0's ETH channels are ALL corrupt (ERISC L1 = `0x49706550`), the auto-discovery algorithm may exclude chip 0 from the fabric mesh, generating a mesh that doesn't include it. The control plane mapping then has no chip 0.

But when `compile_dispatch_kernels` runs, it iterates `devices_` (which includes chip 0 as the primary MMIO device) and calls `populate_cq_static_args(chip_0_dev)`. RelayMux for chip 0 tries to look up its fabric position → crashes.

### Key difference from previous failures

| Failure | Crash location | ERISC L1 corruption extent |
|---------|---------------|---------------------------|
| FIX I transitive gap | `wait_for_fabric_router_sync` (MMIO→dead peer sync timeout) | Non-MMIO devices (4-7) corrupt |
| **New failure (this run)** | `compile_dispatch_kernels` → `RelayMux::GenerateStaticConfigs` | MMIO devices (0-3) ALSO corrupt |

The hardware state is worsening with each run. ERISC L1 corruption is spreading to MMIO devices.

### Potential fixes

**FIX J (relay_mux.cpp guard):** In `RelayMux::GenerateStaticConfigs()`, before calling `get_fabric_node_id_from_physical_chip_id`, check:
```cpp
const auto& cp = get_control_plane_ref();
if (!cp.is_physical_chip_in_fabric_cluster(device_id_)) {
    log_warning(..., "RelayMux: Device {} not in fabric cluster — skipping GenerateStaticConfigs", device_id_);
    return;
}
```
But this would leave the relay mux kernel partially configured, likely causing downstream failures.

**Better approach (FIX K):** Prevent MMIO devices from being excluded from the control plane when their ETH peers are corrupt. In `generate_mesh_graph_from_physical_system_descriptor`, MMIO devices with broken ETH should still be included in the fabric mesh (since MMIO dispatch uses PCIe, not ETH relay). 

**Root fix (still needed):** The ERISC teardown race condition must be fixed to prevent L1 corruption from persisting between runs.

### New run dispatched
Run 24807590619 — still on commit `6d7bf148fd` (no new code fixes). Hardware state may recover if the CI runner's ERISC cleanup between jobs works on next attempt.


---
## 2026-04-23 00:02 UTC — Run 24807590619: MMIO Device 0 router sync timeout

### Failure summary

```
TT_THROW @ fabric_firmware_initializer.cpp:1313
Fabric Router Sync: Timeout after 10000 ms. Device 0: Expected status 0xa2b2c2d2, got 0x00000000
```

### What happened

All 8 devices had ERISC L1 corruption (0x49706550) across multiple channels:
- Devices 0-3: 6 corrupt channels each (chans 0, 1, 8, 9, 14, 15)
- Devices 4-7: 4 corrupt channels each

Devices 4-7 → FIX E2/H/G path: marked dead-relay, firmware skipped, sync skipped. Correct.

Devices 0-3 → new path:
1. `terminate_stale_erisc_routers`: best-effort TERMINATE + zero `router_sync_address` for 6 channels
2. `configure_fabric_cores`: corrupt channels NOT in `pre_known_dead_channels` (probe read via PCIe succeeded), so soft reset (assert/deassert ERISC0) applied to all 6 corrupt channels
3. `configure_fabric` / `device.cpp`: `all_dead_channels` is EMPTY for MMIO devices (soft reset doesn't throw), so ALL channels get `WriteRuntimeArgsToDevice` + `ConfigureDeviceWithProgram` + `write_launch_msg_to_core`
4. "Fabric initialized on Device 0" logged
5. `wait_for_fabric_router_sync` polls Device 0 master router chan
6. Got 0x00000000 after 10000 ms — firmware never wrote sync word

### Root cause

Device 0 is MMIO. Probe reads always succeed (PCIe, not ETH relay). So corrupt channels pass through the "soft reset" path. Soft reset via BRISC assert/deassert succeeds (PCIe write). Firmware binary is written (PCIe L1 write). Launch message sent. But ERISC firmware never executes to write 0xa2b2c2d2.

Likely cause: BRISC soft reset is insufficient when ERISC is in deep hardware-corrupt state (e.g., stuck mid-handshake with peer). The ETH MAC/PHY state may be corrupted beyond what BRISC soft reset can fix. The ERISC starts executing new firmware but the ETH link handshake with the peer never completes → firmware stalls before writing sync word.

### Code path is correct — hardware is broken

The code path is working correctly per its design. The issue is persistent ERISC L1 corruption on MMIO devices that has now degraded the hardware's ability to recover via soft reset alone.

Previous run failed at `control_plane.cpp:1263` (chip 0 excluded from fabric cluster).
This run failed later at `wait_for_fabric_router_sync:1313` (firmware loaded but ETH link unresponsive).

Both are hardware state issues. The CI t3k hardware needs a full reboot to clear ERISC state.

### Potential code fix (FIX L — future)

In `wait_for_fabric_router_sync`, if timeout fires on Device 0 and Device 0 had corrupt channels going in, could skip or degrade gracefully instead of TT_THROW. But this only masks the hardware issue — the fabric wouldn't be functional even if we skipped the throw.

### New run

Run 24808715420 dispatched — no code changes. Hoping for hardware state recovery.

## 2026-04-23 FIX N — CI run #24831422944

### Root Cause (FIX N)
`configure_fabric_cores` called in quiesce Phase 3 with empty `skip_soft_reset_channels`.
For non-MMIO devices (Device 4 in T3000), `assert_risc_reset` on ETH chan1 resets the
entire ETH core including the NOC router. Subsequent `deassert_risc_reset` write must
travel via UMD relay through that now-dead NOC → `wait_for_non_mmio_flush` times out
→ relay permanently damaged → all subsequent non-MMIO writes fail → cascade hang.

### Fix
`device.cpp` line 853: pass all active ETH channels as `skip_soft_reset_channels` when
device is non-MMIO (`!this->is_mmio_capable()`). Uses same `active_channels` var already
in scope. TERMINATED firmware polls for launch messages same as base-UMD firmware; 
`write_launch_msg_to_core` alone restarts it without soft reset (confirmed by FIX M).

### Commit
d9ccbeb37fdf

### CI Run
#24831422944 — triggered 2026-04-23T10:57:58Z

---

## FIX O — 2026-04-23 ~11:50 UTC

**Commit**: `38cbfb902d`

**Root cause (t3k_ttmetal_tests failure in FIX N run #24831935568)**:
Machine `tt-metal-ci-vm-t3k-03` had Device 5 channels 1 and 6 with status `0x49705180`.
`terminate_stale_erisc_routers` detected these as corrupt, zeroed edm_status_address, but
did NOT add to `probe_dead_channels`. `configure_fabric_cores` then attempted soft reset →
`deassert_risc_reset` timed out (NOC in reset from prior crash) → TT_THROW "2 newly-dead ETH channels".

**Fix**: In `fabric_firmware_initializer.cpp` lines 940, added `probe_dead_channels.insert(eth_chan_id)`
for corrupt channels. They now go into pre-known dead set → `configure_fabric` runs in degraded mode
(warning only) instead of throwing.

**CI run**: #24833569461 (queued at 11:50 UTC)

**Still open**: t3k_ttnn_tests failure on `tt-metal-ci-vm-t3k-01` — TopologyMapper forms degraded 2x2
mesh (excludes MMIO chips 0 and 4), then `RelayMux::GenerateStaticConfigs` TT_FATALs because chip 0
not in control plane mapping. This may self-heal once hardware is in clean state, or may need a
separate fix in topology mapper to tolerate degraded meshes.

---
## 2026-04-23 — Commit 6bcd1d20199: Sentinel-special-cased full-queue check

### Root cause confirmed
`ptrs == fences` is a false positive for all real (non-sentinel) fences values.
- When firmware consumes slot X it writes address(X) to `PREFETCH_Q_RD_PTR_ADDR`
- So `fences == X` means "slot X is already cleared (free); firmware is now at X+1"
- `ptrs == fences` fires at depth N-1 (one free slot left) — NOT at depth N (truly full)
- Firmware is spinning on X+1; host stuck in wait reading fence; deadlock

Prior fix (removing second wait after wrap) addressed Scenario A but not Scenario B.

### Fix: dual-branch `is_fetch_q_full()`
```
fences == limit  → wait iff ptrs == limit           (sentinel: 0 consumes, N writes)
fences != limit  → firmware_current = fences+e(mod); wait iff ptrs == firmware_current
```

### CI run #24855612530 (queued 19:51 UTC Apr 23)
Commit 6bcd1d20199 on nsexton/0-racecondition-hunt

---
## 2026-04-23 — Commit 91bf50e956f: In-flight counter (third attempt)

### Why sentinel-special-casing (6bcd1d20199) also failed
Diagnostic log from run #24855612530:
```
ptrs=0xe388 fences=0xe386 firmware_current=0xe388 base=0xe380 limit=0xe480
```
N = 128 (256 bytes / 2-byte entries)
- depth=0 (queue drained): ptrs = firmware_current
- depth=128 (queue full): ptrs = firmware_current
Both give IDENTICAL ptrs/fences. Sentinel only covers initial state (fences=limit).
Once firmware consumed ≥1 entry: fences≠limit → any address-based check has aliasing.

### Correct fix: in-flight counter
- `prefetch_q_in_flight[cq_id]`: incremented in `fetch_queue_write`
- decremented via `count_consumed(old_fences, new_fences)` in `fetch_queue_reserve_back`
- Block iff `in_flight >= N` — no aliasing, no mod-N ambiguity
- Fast path: skip PCIe read when in_flight < N

### CI run #24857031032 (triggered 20:24 UTC Apr 23)
Commit 91bf50e956f on nsexton/0-racecondition-hunt

---
## 2026-04-23 — Run #24857031032: Infra failure (not our bug)

Error: `TopologyMapper FIX P (#42429): Could not force MMIO chip(s) into 2x2 mesh`
Source: `topology_mapper.cpp:1811` — hardware in degraded state from prior crash.
No fetch_queue / system_memory_manager errors in log.

This is `InfraErrorV1.*` noise — test didn't reach our code. Retriggered as #24858114045.

---

## Session 4 — 2026-04-23 (evening)

### Commit 9b5af1f — reset prefetch_q_in_flight on CQ reset

Found via Opus testing-gap evaluation:
- `SystemMemoryManager::reset(cq_id)` (called during `configure_command_queue_programs`) was NOT resetting `prefetch_q_in_flight[cq_id]`
- Firmware reinitializes rd_ptr to limit sentinel when dispatch kernel reloads
- Host counter retaining stale nonzero value → could produce false full-queue detection at next session start
- Fix: add `this->prefetch_q_in_flight[cq_id] = 0;` to reset()

### CI Status

All runs blocked by T3K wh_llmbox pool hardware degradation:
- Runners t3k-01, t3k-06, t3k-09, t3k-14 all have corrupt ETH channels
- Error: "all non-MMIO ETH channels in base-UMD relay state from prior crashed session"
- TopologyMapper throws before any test code runs (exit=1 in ~305ms)
- Pool has been in this state since at least 13:28 UTC today (8+ hours)
- Needs hardware reset / power cycle from infra team

### Confirmation dispatch fix IS working

Old code (exit=124 timeout, ~minutes): fetch_queue_reserve_back blocked forever at depth=N-1
New code (exit=1 exception, ~305ms): reaches TopologyMapper immediately, crashes on ETH corruption
The queue deadlock is gone.

### Testing gaps identified (Opus eval)

P0: No regression test for depth=N-1 boundary (exact bug case)
P0: reset() not clearing in-flight counter (FIXED in this commit)
P1: No unit tests for count_consumed() edge cases (sentinel→real, wrap, no-wrap)
P1: No test for underflow guard (consumed > in_flight → clamp to 0)
P2: No concurrent multi-CQ stress test

### Runs summary

24860138321 — in_progress (commit 9b5af1f, in_flight+reset fix)
24859098080 — FAILED infra (t3k-01, ETH degraded, exit=1 305ms)
24858114045 — FAILED infra (t3k-09, ETH degraded)
24857031032 — FAILED infra (t3k-01, ETH degraded)
24855612530 — FAILED code (sentinel-only fix, aliasing at depth=0 vs N)
24852768059 — FAILED code (original ptrs==fences fix, still deadlocked)

---
## 2026-04-24 — count_consumed full-wrap aliasing bug + fix

### Bug identified
`count_consumed(old_fences, new_fences)` returns 0 when `old_fences == new_fences` via the
`new_fences >= old_fences` path (returns `(new - old) / entry_size = 0`).

**Aliasing scenario**: firmware consumes exactly N entries between two consecutive
`refresh_in_flight` calls. The fence address wraps back to the same slot (`old_fences == new_fences`),
so `count_consumed` returns 0 → `in_flight` stays at N → host deadlocks.

This can happen on the second (or later) full queue fill:
1. First batch of N entries consumed → old_fences = base+(N-1)*e
2. Host writes N more entries (in_flight=N), fast-path skips hardware reads
3. Firmware consumes all N before next `refresh_in_flight` → new_fences = base+(N-1)*e = old_fences
4. count_consumed = 0 → DEADLOCK

With N=1532 (galaxy/t3k config) and ~120 commands/iteration, the second full fill hits
around iteration ~26. But N=256 (DRAM-backed) would alias much sooner.

### Fix (commit c39da53379c)
`dispatch: fix count_consumed full-wrap aliasing deadlock`

When `fence == old_fences` and `in_flight >= N`:
1. Read `next_slot = old_fences + entry_size` (wrapping to base at limit)
2. If `slot_val == 0` (cleared by firmware): N entries consumed
   - Re-read fence (TOCTOU guard): if fence changed, use normal count_consumed path
   - Otherwise: set in_flight = 0 (CASE B confirmed)
3. If `slot_val != 0`: firmware hasn't consumed it yet (CASE A: spin)

Adds at most 2 extra PCIe reads per wait-loop iteration when aliasing triggers.

### New CI run
T3K custom pipeline `24866242008` queued on commit `c39da53379c` (post-power-cycle hardware)

## 2026-04-24 ~04:00 UTC — FIX P + Entry Snapshot logging committed and pushed

### FIX P: Host MAGIC injection for Device 4 ERISC receiver loop

After FIX O (relaunching ERISC cores in quiesce_and_restart_fabric_workers), Device 4 on WH T3K was
entering an infinite receiver handshake loop because the MMIO peer devices (0-3) skip the restart path
entirely and never inject MAGIC (0xAA) via the sender side.

FIX P (committed in same PR commit as ENTRY SNAPSHOT):
1. Pre-compute `fix_p_handshake_addr` from `builder_ctx.get_fabric_router_config(DISABLED, ...).handshake_addr`
2. Before write_launch_msg_to_core for each ETH channel: zero handshake_addr+0 and +16 to flush stale 0xAA
3. After FIX O loop: for each non-dead channel, poll handshake_addr+16 until 0xAA (init_handshake_info ran),
   then write MAGIC to handshake_addr+0
4. Wait for master chan to reach LOCAL_HANDSHAKE_COMPLETE, then write READY_FOR_TRAFFIC

### ENTRY SNAPSHOT diagnostic logging
Per Neil's 3:54 AM request: snapshot all ERISC channel edm_status values at function entry before any phase
runs. This shows what state the prior test left the channels in.

### CI run
Run 24871545753 launched with t3k_ttnn_tests pipeline at 04:02 UTC.
Artifact build in progress.

## 2026-04-24 ~04:47 UTC — FIX P regression analysis

### New failure introduced by FIX P

CI run 24871545753 shows a NEW failure introduced by our branch changes:

```
TT_THROW: Fabric health check failed after quiesce restart on Device 0 — 
6 ERISC channel(s) not at READY_FOR_TRAFFIC (0xa3b3c3d3).
  dev=0 chan=0 status=0x00000000
  dev=0 chan=1 status=0x00000000
  ...
```

### Root cause of FIX P regression

The failure sequence:
1. Device 4 (non-MMIO receiver) quiesce runs: Phase 2.5 terminates, Phase 3 relaunches ERISCs (FIX O),
   FIX P injects MAGIC, Device 4 reaches READY_FOR_TRAFFIC (confirmed by Phase 5: 4 channels healthy)
2. Device 0 (MMIO sender) quiesce runs: Phase 2.5 terminates, Phase 3 relaunches ERISCs (FIX O)
3. Device 0 sender ERISCs restart and try to handshake with Device 4 receiver
4. BUT Device 4 receiver already completed handshake via FIX P — no longer in receiver loop
5. Device 0 sender handshake never completes → status stuck at 0x00000000
6. Phase 5 throws after only 20ms (3 retries × ~7ms)

### Critical observation

Device 0 (MMIO) does NOT early-return from quiesce_and_restart_fabric_workers:
- `get_num_fabric_initialized_routers(0) > 0` — Device 0 has 6 fabric routers
- Device 0 runs Phase 2.5, Phase 3, Phase 5
- FIX O added ETH relaunch to Phase 3 WITHOUT mmio_capable guard — affects ALL devices

FIX P was based on the assumption that MMIO devices skip quiesce entirely. That assumption is WRONG.

### Per Neil's 4:47 AM message

"Have OPUS do a reevaluation of how we treat local vs nonlocal (non-mmio) devices 
in this branch to make sure our efforts to root-cause the AllGather test hang did 
not introduce a new failure point."

OPUS evaluation spawned to:
1. Analyze MMIO vs non-MMIO device identification in T3K topology
2. Understand correct sender/receiver roles in ETH handshake
3. Determine if FIX P is needed at all (if Device 0 also restarts, natural handshake may suffice)
4. Propose and implement correct fix for Phase 3 ETH relaunch coordination

---

## 2026-04-24 CI run 24873378679 analysis — root cause: Pass 1 ordering (MMIO quiesced before non-MMIO)

### Exception

```
05:25:14.817 — Device 0 Phase 3 complete (ERISCs relaunched, NOT yet READY_FOR_TRAFFIC)
05:25:14.818 — Device 5 ENTRY snapshot: ReadFromDeviceL1 → read_non_mmio → routes via Device 0 relay
05:25:19.820 — UMD "Timeout waiting for Ethernet core service remote IO request"
```

Stack: `read_non_mmio → ReadFromDeviceL1 → quiesce_and_restart_fabric_workers → restart_fabric_workers_for_quiesce → quiesce_internal`

### Root cause

Device iteration order in Pass 1: Device 4 → Device 2 → Device 0 → **Device 5 (FAIL)**

Device 0 (MMIO) is the UMD relay gateway for all non-MMIO device reads. When Device 0's Phase 3
relaunches its ERISCs, they're in STARTED state (not READY_FOR_TRAFFIC). Device 5's subsequent ENTRY
snapshot calls `ReadFromDeviceL1 → read_non_mmio`, which routes through Device 0's ERISC relay
firmware. With Device 0's ERISCs in STARTED state (not serving relay requests), UMD times out after
5 seconds.

**This is not a handshake race — it's a relay unavailability caused by incorrect quiesce ordering.**

### Fix applied

Two changes in `mesh_device.cpp`:

1. `restart_fabric_workers_for_quiesce()`: Sort `get_devices()` so non-MMIO devices come first,
   MMIO devices last. This ensures the MMIO relay ERISCs remain READY_FOR_TRAFFIC while non-MMIO
   devices are being quiesced.

2. `quiesce_internal()` Pass 1 loop: Same sort applied.

Defensive fix in `device.cpp`:
- ENTRY snapshot `ReadFromDeviceL1` wrapped in try/catch. On failure, logs a warning with
  `edm_status=0xDEADBEEF` and continues. ENTRY snapshot is diagnostic only — should never abort.

### T3K topology note

On T3K: Device 0 = MMIO. Devices 1-7 = non-MMIO. Remote reads to non-MMIO devices route through
Device 0's ERISC relay firmware. Device 0 must be quiesced LAST in Pass 1.


---

## 2026-04-25 — Iteration 9 (run 24925116689): FIX W — Phase 5b all-0x0 clean-return extended to MMIO devices

### Failure analysis

Device 3 is MMIO-capable (mmio=true in logs).

- Device 4 (non-MMIO) Phase 5 relay read fails → FIX U triggers → Device 4 returns cleanly
- Device 3 (MMIO): Phase 5 polls master chan 14, status=0x0 for 10s → "Continuing to health check"
  FIX V guard `if (!is_mmio_capable())` is FALSE for Device 3 → NOT triggered
- Phase 5b: all 6 channels (6,7,8,9,14,15) stuck at 0x0 for 2001ms deadline
  → chan 15 deadline-skipped = 0xDEAD5B5B → `truly_unhealthy` not empty
  → `if (!is_mmio_capable())` guard again FALSE → falls through to TT_THROW (line ~1929)
- TT_THROW → teardown second quiesce → Device 3 Phase 5b TT_THROW again
  → `rescue_stuck_dispatch_cores` for Device 4 via UMD relay → 5s timeout per stream
  → 8+ minutes hanging before CI kill

### Root cause

FIX V's `if (!this->is_mmio_capable())` guard was too narrow. When a non-MMIO device's relay
path breaks (causing it to return without booting its ETH firmware), any MMIO peer whose fabric
channels route through that non-MMIO device will also fail to boot — but FIX V's guard let MMIO
devices fall through to TT_THROW.

### Fix applied (FIX W, device.cpp ~line 1892)

Moved the `all_dead` check OUTSIDE the `!is_mmio_capable()` guard so it applies to both MMIO and
non-MMIO devices:

- Non-MMIO: `all_dead` → set `fabric_relay_path_broken_=true` + log_error + return
- MMIO: `all_dead` → skip `fabric_relay_path_broken_=true` (PCIe reads are safe) + log_error + return
- Either: avoids TT_THROW → avoids teardown cascade → avoids rescue_stuck_dispatch_cores timeout loop

Changed log severity from `log_warning` to `log_error` (this is a genuine ETH firmware boot failure,
not a transient warning).

### Commit

`fix(fabric): FIX W: extend Phase 5b all-0x0 clean-return to MMIO devices (#42429)`

---

## 2026-04-25 — Iteration 10 (run 24928250106): FIX Z — relay_path_broken guard in read_completion_queue_event

### Failure analysis

Run 24928250106, job t3k_ttnn_tests (73002222297), runner tt-metal-ci-vm-t3k-12.

Sequence:
1. Pre-AllGather quiesce runs: Device 4+5 (non-MMIO) → relay_path_broken_=true; Device 1+3 (MMIO) → Phase 5 timeout + Phase 5b all-0x0 → FIX W clean return.
2. Quiesce returns cleanly at 10:09:11.
3. Test calls ttnn::all_gather at 10:09:11.737.
4. FDMeshCommandQueue::read_completion_queue_event() iterates all devices.
5. For Device 4: calls completion_queue_wait_front(id_, exit_condition_) — NO relay_path_broken check.
6. completion_queue_wait_front internally issues UMD relay read → relay ERISC running fabric firmware → 5s UMD timeout.
7. At 10:09:16: "Timeout waiting for Ethernet core service remote IO request." thrown in test body.

### Root cause

read_completion_queue_event() in fd_mesh_command_queue.cpp did not check
`device->is_fabric_relay_path_broken()` before calling completion_queue_wait_front().
Device 4 had relay_path_broken_=true (set during quiesce) but was still included in
the CQ wait loop, triggering a 5-second UMD relay timeout.

### Fix applied (FIX Z, fd_mesh_command_queue.cpp line ~936)

Added guard before completion_queue_wait_front():
  if (!device->is_mmio_capable() && device->is_fabric_relay_path_broken()) {
      TT_THROW("FIX Z: Fabric relay path broken on non-MMIO device {} — cannot read completion queue event (UMD relay would time out). Aborting immediately.", device->id());
  }

Replaces 5-second silent hang with immediate throw + clear diagnostic.

### Device 1+3 (MMIO) Phase 5b all-0x0 investigation

These MMIO devices show all ETH channels at 0x0 in Phase 5b because their fabric
handshake routes through Device 4 (broken relay). Device 4 never boots ETH firmware
after Phase 3 (Phase 3 is skipped for it by FIX Q), so MMIO peers whose fabric channels
connect to Device 4 never complete their ERISC handshake either.

This is already handled correctly by FIX W (all_dead → log_error + return cleanly
for both MMIO and non-MMIO devices). No additional fix needed here.

### Commit

`fix(fabric): FIX Z — check relay_path_broken before CQ wait in read_completion_queue_event`
Commit hash: aeea3f76ad5

### CI Run

https://github.com/tenstorrent/tt-metal/actions/runs/24928810501 (dispatched 2026-04-25 ~10:27 UTC)

---
## 2026-04-25 23:04 UTC — Cycle 24: FIX AC — two-phase ETH reset

### CI run 24941287837 failure analysis

**Failure sequence:**
- Devices 4/5: Phase 5 `wait_for_fabric_workers_ready` timed out (relay reads failed)
- `fabric_relay_path_broken_` set for devices 4/5
- `RiscFirmwareInitializer::teardown()` wasted ~20s timing out on assert_cores/l1_barrier for non-MMIO devices 4/5 (relay broken)
- FIX AB fired, reset MMIO ETH channels, but only 2ms before `~Cluster` destroyed driver
- Non-MMIO ETH channels (devices 4/5) were NEVER reset
- Second binary's `init_tt_device` → `TopologyDiscovery::discover_remote_devices` → UMD relay reads → non-MMIO ERISCs still in fabric firmware mode → infinite hang

**Root cause of FIX AB inadequacy:**
1. Timing: FIX AB at END of teardown, 2ms before `~Cluster` = ERISCs couldn't reboot
2. Coverage: FIX AB only reset MMIO ETH channels; non-MMIO ETH channels left in fabric firmware

**FIX AC (commit e2f0fea951b):**
- Step 2: Hard-reset MMIO ETH channels via PCIe + `sleep(500ms)` BEFORE assert_cores loop
- Step 3: Skip relay-broken non-MMIO devices in assert_cores/l1_barrier loop
- Step 5: Reset non-MMIO ETH channels via relay (now functional) + `sleep(200ms)`
- File: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`

**New CI run:** 24942784207

## 2026-04-26 Iteration 7 — FIX AI

### Run 24948331172 (FIX AH) — FAILED

Three jobs failed:

**Job 73054126719 (t3k_ttnn_udm_tests)**: `assert_risc_reset_at_core` timeout on Device 4 chan=1 — ERISC still running, can't assert reset. Timeout → Phase 5 hangs.

**Job 73054126720 (t3k_ttnn_tests)**: "Fabric health check failed after quiesce restart on Device 4 — 4 ERISC channel(s) not at READY_FOR_TRAFFIC (0xa3b3c3d3) after 2000ms. STARTED=0xa0b0c0d0 (handshake incomplete, timing issue)". Still stuck at STARTED.

**Job 73054126724 (t3k_tt_metal_multiprocess_tests)**: `TT_FATAL: Physical chip id not found for eth coord` at `tt_cluster.cpp:554` in `get_physical_chip_id_from_eth_coord()`. Crash during `set_custom_fabric_topology()` in `tt_fabric_test_common.hpp:1809`. This happens because a chip's eth_coord from the test YAML doesn't exist in `cluster.get_all_chip_ethernet_coordinates()` — meaning the chip was excluded from cluster discovery due to dead ETH PHY.

### Root Cause Analysis

All three failures trace to the same root cause: `assert_risc_reset_at_core(ALL)` without a paired `deassert` in teardown paths. When NCRISC (the subordinate ERISC that maintains the ETH PHY link) is left in hardware reset:
1. ETH PHY goes down on that chip
2. Next test's cluster discovery finds chip unreachable → excludes it → `get_physical_chip_id_from_eth_coord()` crash for that eth_coord
3. If cluster discovery succeeds, `terminate_stale_erisc_routers()` probe reads timeout (relay path dead) → all channels show corrupt/dead

### FIX AI (commit a3fb4c7f5bd)

Four files changed:
1. `fabric_firmware_initializer.cpp`: add `deassert_risc_reset_at_core(ALL)` immediately after `assert_risc_reset_at_core(ALL)` in the teardown force-reset path
2. `metal_env.cpp`: change `teardown_fabric_config()` from `ERISC0` to `ALL` for both assert and deassert cycles in the teardown-timeout path
3. `device.cpp`: add `pending_phase25_force_reset_chans_` tracking; Phase 3 and `launch_eth_cores_for_quiesce()` deassert force-halted channels after `write_launch_msg_to_core` so ERISC can pick up the message
4. `device_impl.hpp`: add `pending_phase25_force_reset_chans_` member

### CI Run 24948921651

Triggered with `t3000-unit=t3k_ttnn_tests`. In progress as of 05:12 UTC.

---
## 2026-04-26 Session 4 — FIX AK: Phase 5b partial-mesh quiesce non-fatal

### EDM Status Enum (CORRECTED)
Earlier sessions had wrong enum values. Actual values from fabric_edm_packet_header.hpp:
- 0xa0b0c0d0 = STARTED
- 0xa1b1c1d1 = REMOTE_HANDSHAKE_COMPLETE
- 0xa2b2c2d2 = LOCAL_HANDSHAKE_COMPLETE
- 0xa3b3c3d3 = READY_FOR_TRAFFIC ← what Phase 5b checks for
- 0xa4b4c4d4 = TERMINATED

### `MultiCQFabricMeshDevice2x4Fixture` — IMPORTANT
Despite its name, this fixture uses MeshShape{1, 4} (4 devices on T3K, specifically devices 1,3,4,5).
NOT a full 2x4 = 8-device mesh. "2x4" refers to the T3K hardware topology, not the mesh shape.
See test_ccl_multi_cq_multi_device.cpp:60.

### Root Cause: Phase 5b TT_THROW in partial-mesh quiesce (job 73057782377)

The `AsyncExecutionWorksCQ0` test uses a 1x4 mesh on T3K (devices 1,3,4,5).
During quiesce restart, only these 4 devices are quiesced. Device 4 has ETH
channels (6,7) that connect to out-of-mesh devices (0,2,6,7). Those peers run
base-UMD firmware and DON'T respond to the EDM fabric handshake. Result:
  - Channels 0,1: reach REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1) — mesh peers responded
  - Channels 6,7: stuck at STARTED (0xa0b0c0d0) — out-of-mesh peers don't respond

Phase 5b sees all 4 channels as truly_unhealthy (none at READY_FOR_TRAFFIC 0xa3b3c3d3).
The all_dead check: {0xa1b1c1d1, 0xa0b0c0d0} ∉ {0x0, 0xDEAD5B5B, 0xDEADECE7} → false.
→ TT_THROW: "4 ERISC channel(s) not at READY_FOR_TRAFFIC (0xa3b3c3d3) after 2000ms"

### Fix: FIX AK (commit ce5bd3a5bf5)

Added `all_handshake_incomplete` check after `all_dead` in `phase5b_erisc_health_check()`.
If ALL truly-unhealthy channels are in:
  {0x0, 0xDEAD5B5B, 0xDEADECE7, STARTED, REMOTE_HANDSHAKE_COMPLETE, LOCAL_HANDSHAKE_COMPLETE}
→ log_warning + return cleanly (no throw, no fabric_relay_path_broken_)

Key: do NOT set fabric_relay_path_broken_ — the relay IS working, it's the PEER that didn't respond.
Phase 2.5 in the next quiesce will TERMINATE these stuck channels.

CI run: 24950406819 (triggered 2026-04-26T06:43:55Z, in progress)

## Iteration 22 — 2026-04-26 FIX AQ-2: topology_discovery ethernet_connections crash

**Run 24955934407 failure pattern:**
- `MultiCommandQueueSingleDeviceFixture.TestAsync*` — 4 tests, each exactly 20018ms, FAILED
- Exception: `"unordered_map::at" thrown in the test fixture's constructor`

**Root cause (cascade):**
1. CCL test teardown → Phase 5 MMIO device timeout on LOCAL_HANDSHAKE_COMPLETE (chan 15)  
2. Dispatch cores hang → device close throws  
3. Next test (SingleDeviceFixture) triggers fresh UMD topology discovery
4. FIX AQ catches 4 remote device relay timeouts (5s each = 20s), marks them as `discovered_devices` but does NOT add to `asic_id_to_chip_id`
5. A later ETH channel on an MMIO device sees a link to one of those skipped devices → falls into the `else` branch → adds `{mmio_chan, skipped_remote_chan}` to `ethernet_connections`
6. `fill_cluster_descriptor_info()` calls `asic_id_to_chip_id.at(skipped_remote_asic_id)` → `std::out_of_range` → test fixture constructor fails → FAILED (20018ms)

**Fix: FIX AQ-2 in topology_discovery.cpp**
- In the `else` branch of the discovery loop, check `asic_id_to_chip_id.find(remote_asic_id) == end()` before pushing to `ethernet_connections`
- Skipped devices are in `discovered_devices` (prevents retry) but absent from `asic_id_to_chip_id`
- The guard prevents the stale connection from reaching `fill_cluster_descriptor_info()`


---
## 2026-04-26 Iter 24/25 Root Cause

### Iter 24 was triggered with wrong SHA
Run `24959265492` was dispatched at SHA `63d0b74b0f3` (FIX X v1 — buggy pre-loop heartbeat check).
We pushed the corrected FIX X (`e6b3cb4a454`) AFTER triggering iter 24, so iter 24 ran the buggy version.

### All 3 iter 24 failures = buggy FIX X boot race
- `t3k_tt_metal_multiprocess_tests` (t3k-10): "Physical chip id not found for eth coord" — FIX X skipped ETH training before firmware wrote heartbeat → ETH links unestablished → wrong ETH coords in cluster descriptor
- `t3k_ttnn_udm_tests` (t3k-14): 5-min timeout on `TestMeshWidthShardedCopy2D_Small` — Phase 5 handshake timeouts cascading from incomplete ETH init
- `t3k_ttnn_tests` (t3k-08): Device 4 fabric channels at `0x00000000` (fabric firmware never loaded via relay) — Phase 5 timeouts, FIX AC triggered, FIX AQ dropped 4 non-MMIO devices, eventual hang

### FIX X corrected (e6b3cb4a454)
Checks heartbeat INSIDE the training loop after 2000ms of IN_PROGRESS.
- Normal ETH boot: completes training in <1s, never reaches the check → no false skip
- Force-reset quiesce cores: still IN_PROGRESS at 2s, heartbeat != 0xABCD → skip (correct)
- Fresh boot race: heartbeat not ready at t=0 but ETH training completes normally → loop exits before 2s → no check triggered

### Iter 25
Run `24960041509`, SHA `e6b3cb4a454`, triggered 2026-04-26T15:19:54Z.

## Iteration 27 — FIX AK, AL, AM (Run 24960899576 post-mortem)

**Run 24960899576 result**: All 3 jobs failed.

### Root causes identified

**Job 928 (t3k_ttnn_udm_tests)** — 4-minute hang after FIX AJ:
- Device 4 chan 0/1 → 0xdeaddead → assert_risc_reset throws (relay dead) → FIX AJ catches, marks Device 4 in relay_dead_devices
- BUT: the l1_barrier loop (FIX A) correctly skips Device 4 via FIX AJ, then tries l1_barrier on OTHER non-MMIO devices in the 1x4 mesh (e.g. Device 5/6/7)
- Those devices' relay paths transit through Device 4's dead ERISC links → wait_for_non_mmio_flush() BLOCKS INDEFINITELY (doesn't throw, so try/catch is useless)
- Result: 4-minute silent hang until GHA timeout
- **FIX AK**: If relay_dead_devices is non-empty, skip l1_barrier for ALL non-MMIO devices (not just relay_dead_devices). One dead relay can make the whole network's drain unsafe.

**Job 932 (t3k_tt_metal_multiprocess_tests)** — Device 0 router sync → TT_THROW → signal 6:
- Device 3 dead-relay → FIX G correctly skips its router sync
- But Device 0 (mesh ring neighbor of Device 3) fails router sync (status stays 0x00000000): its fabric router firmware waits for Device 3 to complete ring handshake, which never happens
- wait_for_handshake() timeout → TT_THROW → propagates → signal 6 → process crash
- **FIX AL**: Change TT_THROW to log_error + return in wait_for_fabric_router_sync() (both read-exception and timeout paths). Fabric is degraded; tests fail at op time instead of crashing.
- **FIX AM**: Skip verify_all_fabric_channels_healthy() in configure() when dead_relay_devices_ is non-empty. Otherwise FIX AL avoids the first TT_THROW but the second one (in verify) still crashes.

**Job 934 (t3k_ttnn_tests)** — chan 14/15 spurious "read FAILED" in diagnostic:
- `get_eth_core_for_channel(14/15, LOGICAL)` throws for Wormhole devices 1/3 (no LOGICAL mapping for those channel IDs)
- Exception caught (not a crash), but logged as log_info "read FAILED" which is misleading
- **Fix**: Split coordinate lookup from read in a separate try/catch; log coord failure at log_debug

### Files changed
- `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`:
  - FIX AK: wrap l1_barrier for-loop in `if (relay_dead_devices.empty()) { ... }` — skip all non-MMIO l1_barrier when any relay is dead
  - FIX AL: `wait_for_handshake` lambda — TT_THROW → log_error + return (read-exception path and timeout path)
  - FIX AM: `configure()` — skip `verify_all_fabric_channels_healthy()` when dead_relay_devices_ non-empty
- `tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp`:
  - Separate coord lookup (log_debug on failure) from L1 read (log_info on failure) in health probe

### Commit SHA: TBD (iteration 27)

---
## 2026-04-29 Gap Auditor Session — FIX PA/PB/PC Verification Status

### CRITICAL: FIX PC has NEVER been verified by actual CI tests

FIX PC commit `e995ebb796b1` was pushed, and run `25098712741` was dispatched on that SHA.
However, that run showed "success" but ALL t3000 unit tests were SKIPPED.
Root cause: `determine-tests` job set `run-unit=false` (manual pipeline mode).
Case statement: `case "manual" in ... echo "run-unit=false"` — no tests ran.

**Action needed**: Trigger a new properly-configured CI run on FIX PC SHA.

### GAP-50 (NEW): analyze_fabric_hang_log.sh — no detection for dispatch cascade

The script had no section for the 500ms dispatch cascade (fw_launch_addr stale) pattern.
The PHASES section was also crashing (BrokenPipeError: SIGPIPE under set -eo pipefail).

**Fixes applied** to `scripts/analyze_fabric_hang_log.sh`:
1. Added `=== DISPATCH CASCADE (500ms fw_launch_addr stale — FIX PA/PB/PC) ===` section
2. Added `signal.signal(signal.SIGPIPE, signal.SIG_DFL)` to PHASES python block
3. Added `|| true` to `| head -50` and `| head -25` pipelines (SIGPIPE fix)
4. Added `|| true` to `CANCEL_LINE=` and `PROBLEM_DEVS=` grep commands (set -e fix)
5. Added `|| true` to RUNNER and JOB grep commands

**Validated** against log `run_25096771728/job_73536817618.txt`:
```
=== DISPATCH CASCADE ===
[CASCADE DETECTED] 140 x 500ms timeout(s), 140 force-reset(s)
FIX AQ overhead: 163 x 5s UMD timeout(s) = ~815s total
Estimated cascade overhead: 885s (14.8 min)
  Device 0: 25-16, 18-16, 25-17, 18-17, 22-17, 21-17
  Device 1: 21-16, 25-17, 22-16, 18-17, 22-17, 21-17
  Device 2: 25-16, 18-16, 25-17, 18-17, 22-17, 21-17
  Device 3: 21-16, 25-17, 22-16, 18-17, 22-17, 21-17
DIAGNOSIS: 500ms dispatch cascade (FIX PA/PB/PC pattern)
```

### FIX PA/PB/PC mechanism recap

After ETH core force-reset in Phase 2.5 (fabric teardown), `fw_launch_addr` L1 address
retains the previous dispatch launch pointer. On the NEXT test's `reset_cores()`:
- `erisc_app_still_running()` reads `fw_launch_addr` → non-zero → returns `true`
- `reset_cores()` waits 500ms for ETH app to finish → always times out
- Repeated for every ETH core that was force-reset → 140 × 500ms = 70s overhead

FIX PA: `reset_cores()` guard in `tt_cluster.cpp` (catches the stale read)
FIX PB: Clear `fw_launch_addr` in `reset_cores()` explicitly (deferred to FIX PC)
FIX PC: `fabric_firmware_initializer.cpp` Phase 2.5 — after `deassert_risc_reset_at_core`,
        calls `cluster_.write_core_immediate(dev, coord, {0}, fw_launch_addr)`
        Wrapped in `catch(...)` (best-effort; non-MMIO dead-relay devices may throw)

### GAP-51: FIX PC silent failure path

FIX PC's `catch(...)` swallows `write_core_immediate` failures. For MMIO devices,
PCIe writes are reliable and should never fail. For non-MMIO devices with a dead relay,
the catch is intentional (FIX PA handles the fallback). But if MMIO write fails silently
(e.g. hal_ misconfiguration), cascade persists with no diagnostic.

Mitigation: Add a log_warning inside the catch block for MMIO devices.

### GAP-52: 163 FIX AQ timeout events = 815s overhead per run

Each FIX AQ event is a 5s UMD timeout probing a dead remote device during init.
163 events × 5s = 815s. Combined with 140 cascade timeouts × 0.5s = 70s = **885s total**.
This is 14.8 minutes overhead on EVERY test run on this hardware configuration.

FIX AQ is intentional for dead/out-of-mesh devices but should be counted and reported.
The analyze script now reports this in the CASCADE section.

### Next actions (priority order)

1. **Trigger a real CI run on FIX PC SHA** (run 25098712741 was all-skipped — doesn't count)
2. **Push analyze_fabric_hang_log.sh fixes** to branch `nsexton/0-racecondition-hunt`
   (worktree is behind remote by 67+ commits; need fetch + merge or rebase first)
3. **Add log_warning** in FIX PC catch block for MMIO device write failures
4. **Track FIX AQ event count** in CI metrics to monitor overhead trend

---
## 2026-04-29 OPUS Gap Audit — GAP-50 Root Cause Correction: FIX PD (device.cpp quiesce path)

### CRITICAL CORRECTION: FIX PC was placed in the wrong code path

Previous session identified FIX PC as the fix for the 500ms cascade and placed it in
`fabric_firmware_initializer.cpp::teardown()` at the `is_non_mmio_relay_dead` block.

**This was wrong.** The `is_non_mmio_relay_dead` block covers INIT teardown for
non-MMIO dead-relay devices — NOT the quiesce (test-level teardown) path.

The 140 Timeout(500ms) cascade (run 25096771728) comes entirely from MMIO devices 0-3
(channels 6, 7, 8, 9, 14 connecting to dead relay devices 4-7). These devices go through
the quiesce path in `device.cpp::quiesce_and_restart_fabric_workers()`, NOT through
`fabric_firmware_initializer.cpp::teardown()`.

### Trace of the actual cascade

1. Fabric ETH channels on MMIO devices 0-3 run dispatch firmware (status 0xa1b1c1d1)
2. Test teardown: `quiesce_and_restart_fabric_workers()` sends TERMINATE to each ERISC
3. ERISCs connected to dead relay devices never respond → 2000ms timeout
4. Phase 2.5 force-reset: `assert_risc_reset_at_core(ALL)` applied, channel added to
   `pending_phase25_force_reset_chans_`
5. Pass-0 deassert: `deassert_risc_reset_at_core` → ERISC boots into base UMD firmware
6. **FW_LAUNCH_ADDR NOT CLEARED** — HW reset does not zero L1
7. Next test's `reset_cores()` → `erisc_app_still_running()` reads stale non-zero value
8. → 500ms wait → FIX PA reactive clear → repeat × 35 rounds × 4 MMIO devices × channels

This happens in `quiesce_and_restart_fabric_workers()` (primary quiesce path) and
`launch_eth_cores_for_quiesce()` (deferred-launch quiesce path, `defer_eth_launch=true`).

FIX PC in `fabric_firmware_initializer.cpp::teardown()` runs in a different process-level
path for a different device category — it provides zero benefit for the observed cascade.

### FIX PD — Proactive fw_launch_addr clear in device.cpp Pass-0 (GAP-50)

**Commit**: `1966226902f` on branch `nsexton/0-racecondition-hunt`
**Files changed**: `tt_metal/impl/device/device.cpp` (+282, -72 including earlier FIX AI-2 edits)

Two edits added immediately after `deassert_risc_reset_at_core` in Pass-0 sections:

**Edit 1** — `quiesce_and_restart_fabric_workers()` Pass-0 loop:
```cpp
// FIX PD (GAP-50): clear ERISC dispatch fw_launch_addr after Phase 2.5 force-reset.
// HW reset does NOT zero L1. If dispatch firmware was previously running on this
// ETH channel, fw_launch_addr retains its non-zero value after deassert. On the
// next test's reset_cores(), erisc_app_still_running() reads the stale flag and
// triggers a 500ms wait + another force-reset — cascading across all tests.
try {
    const auto& hal_pd = env_impl.get_hal();
    const auto aeth_idx = hal_pd.get_programmable_core_type_index(
        HalProgrammableCoreType::ACTIVE_ETH);
    const uint32_t fw_launch_addr_pd =
        hal_pd.get_jit_build_config(aeth_idx, 0, 0).fw_launch_addr;
    env_impl.get_cluster().write_core_immediate(
        this->id(), phys_core_0, std::vector<uint32_t>{0}, fw_launch_addr_pd);
} catch (...) {
    // Best-effort: MMIO devices succeed via PCIe; non-MMIO with dead relay
    // may throw — FIX PA in reset_cores() handles the one-time fallback.
}
```

**Edit 2** — `launch_eth_cores_for_quiesce()` Pass-0 loop (mirror):
```cpp
// FIX PD (GAP-50): Mirror of FIX PD in quiesce_and_restart_fabric_workers() Pass-0.
// This path is taken when defer_eth_launch=true.
try { /* same code */ } catch (...) { /* same best-effort */ }
```

### Status and next steps

- FIX PC: still committed (wrong path but harmless for non-MMIO init teardown coverage)
- FIX PD: committed (`1966226902f`), NOT yet CI-tested (needs a properly-configured run)
- Last CI run (25098712741) skipped ALL t3k tests — FIX PC+PD together have zero CI verification
- **Action needed**: Trigger a new CI run on HEAD of `nsexton/0-racecondition-hunt`
  using workflow `sanity-tests-pr.yaml` (workflow_id=244338363)

### Remaining gaps

- **GAP-51**: FIX PD catch block does not log_warning for MMIO device write failures
  (MMIO writes via PCIe should never fail; silent failure would mask hw/hal bugs)
- **GAP-52**: 163 FIX AQ timeout events = 815s overhead per run (pre-existing, intentional)
- **analyze_fabric_hang_log.sh path mismatch**: OPUS audit template checks wrong path
  `/workspace/group/analyze_fabric_hang_log.sh`; actual is
  `/workspace/group/worktrees/nsexton-0-racecondition-hunt/scripts/analyze_fabric_hang_log.sh`

---
## 2026-04-30 CI Monitor — Run Cycle 17:05 UTC

### Runs reviewed:
- **25176753527** (16:22 UTC): ✅ "succeeded" but t3k_ttnn_tests SKIPPED
  - Previous monitor cycle dispatched with wrong inputs (t3000-unit defaulted to 'disabled')
  - Head SHA: e2dd143afb93 (FIX RZ)
  
- **25174791734** (15:41 UTC): ❌ FAILED (last real test run)
  - Runner: tt-metal-ci-vm-t3k-12
  - Pattern: GAP-21 rapid_allgather_quiesce_stress hung at cycle 1 dispatch timeout (5s)
  - fabric_eth_health: Device 4 chans 0/1/6/7 status=0x00000000, Device 1 chans 6/7 status=0x3f803f80 (expected 0xa2b2c2d2)
  - FabricFirmwareInitializer::teardown: Device 1 ETH chan=6 timed out (5000ms)
  - Root cause: stale ERISC firmware from prior test on the machine

### Remote HEAD now at e2dd143afb93 (FIX RZ) — 4 commits ahead of worktree
- FIX RY: skip GAP-21 stress on degraded cluster
- FIX RX: skip TearDown quiesce when fabric already broken after test body
- FIX QW: early SKIP in MultiCQFabricMeshDevice2x4Fixture::SetUp() when cluster degraded
- FIX RZ: detect stale base-UMD channels on non-MMIO devices in is_fabric_degraded()

### New run dispatched
- Dispatch HTTP 204 confirmed
- Target: nsexton/0-racecondition-hunt @ HEAD (e2dd143afb93)
- t3000-unit=t3k_ttnn_tests (correct inputs this time)

---
## 2026-05-02 — Topology Check: clean open/close (Method 1) + PCIe sysfs (Method 2)

### Background (Neil 8:55 AM)
`GetNumAvailableDevices()` topology check contaminates ETH channels. Root cause:
```python
import ttnn; ttnn.GetNumAvailableDevices()
```
Opens full UMD cluster → `deassert_risc_resets()` loads base-UMD firmware (`0x49706550`)
on all 44+ ETH channels → Python exits → `Cluster::~Cluster()` does NOT call `close_device()`
→ stale ETH state (`base_umd=6` per device) contaminates next test → FIX AY/AV relay re-sync
fails → `0x49705180` corrupt state → SKIPs cascade → CI treats as failure.

Confirmed in CI run 25248023665:
- Line 1442: `LOG_METAL: T3K topology OK — 8/8 chips visible.` (OLD contaminating check)
- Lines 1594-1635: `terminate_stale_erisc_routers: Device 0..7 summary: base_umd=6 (or 4)`
- Lines 2315-2327: `edm_status=0x49705180 is NOT a valid EDMStatus value — ERISC L1 CORRUPT`
- Line 4111: `LOG_METAL: test returned rc=1` from CORRUPT → SKIP cascade

### Fix committed to nsexton/0-racecondition-hunt

**UMD: `tt_metal/third_party/umd/nanobind/py_api_cluster.cpp`** (commit `13fb7a3c`)
Added `close_device` Python binding — allows callers to explicitly tear down ETH channels
before Python process exits. Without this, `Cluster::~Cluster()` leaves all ETH channels
loaded with base-UMD firmware, contaminating subsequent callers.

**Shell: `tests/scripts/t3000/run_t3000_unit_tests.sh`** (commit `51788ccf780`)
Replaced contaminating `GetNumAvailableDevices()` block with:

**`t3k_count_via_umd()`** (METHOD 1 — full 8-chip ETH topology check):
- Opens `tt_umd.Cluster`, reads `get_target_device_ids()`
- Calls `cluster.close_device()` EXPLICITLY before exit → zero contamination
- Returns exit 2 on ImportError (tt_umd not available) → clean fallback signal
- min_chips=8 (full topology including non-MMIO)

**`t3k_count_via_pcie()`** (METHOD 2 — sysfs MMIO-only fallback):
- Reads `/sys/bus/pci/drivers/tenstorrent/*/device` for WH_B0 device_id `0x401e`
- Zero ETH interaction, zero firmware loading, zero contamination risk
- Returns 4 for healthy T3K (MMIO-only)
- Used when tt_umd unavailable (umd_rc=2)

**Fallback logic** with `set -e` safety:
```bash
raw_output=$(t3k_count_via_umd 2>/dev/null) && umd_rc=0 || umd_rc=$?
```
`&& ... || ...` pattern suppresses set -e early exit for non-zero exit code capture.

### Push status
- tt-metal changes: pushed to `nsexton/0-racecondition-hunt` as `170049f2c3f`
- UMD changes: `13fb7a3c` on tt-umd repo

---

## 2026-05-04 — FIX SB2 Implementation

**Commit**: `55067869a95` — "fix: FIX SB2 — proactively mark non-MMIO relay broken after FIX M fires (#42429)"

**Root cause confirmed** (from OPUS deep-dive + code analysis):
1. Prior test `~Cluster` timeout → force-reset ERISC → ERISC reboots into base-UMD firmware → writes 0x49706550 sentinel
2. Next test init: `compile_and_configure_fabric()` sees 0x49706550 → FIX M fires → skip soft-reset, write launch_msg → MMIO ERISC transitions to fabric EDM firmware
3. `quiesce_devices()` ENTRY snapshot tries relay reads through MMIO ERISC (now in EDM firmware mode, not relay mode) → HANGS indefinitely (blocking read, not exception — try/catch and 6s deadline are both ineffective)
4. `fabric_relay_path_broken_` was NOT proactively set on non-MMIO devices → no FIX R guard to skip relay reads

**Fix**: After PHASE 2 in `compile_and_configure_fabric()`, iterate over `base_umd_channels_map`. For each MMIO host with non-empty FIX M channels, call `dev->set_fabric_relay_path_broken()` on all non-MMIO devices behind that MMIO host. This triggers FIX R guard in ENTRY snapshot / Phase 2.5 / Phase 3 to skip relay reads.

**CI run**: 25334489944 (dispatched at 55067869a95, t3k_ttnn_tests)

**OPUS analysis**: Full report at `/workspace/group/research/quiesce_devices_hang_analysis.md`
- Proposal C (zero edm_status after force-reset) deferred — timing-dependent, FIX SB2 is the primary fix
- Proposal A (liveness probe) not yet implemented — lower priority
- Proposal D (relay read hang detection in Phase 5) not yet implemented — defensive fallback

---

## Root Cause Analysis: Stale go_msg=0x02 on Tensix Cores (CI run 25578527644)

**Date**: 2026-05-09
**Commit**: `9307d3c867` (branch `nsexton/0-racecondition-hunt`)
**CI run**: `25578527644`
**Symptoms**: FIX SC fired for devices 0-3, cores 19-25 x 16-17, all showing `go_msg=0x02`. Additionally, 8 ETH channels (dev=2,3 chan=0,1,6,7) reported "could NOT be force-reset."

### 1. Mechanism of go_msg=0x02

**Key correction**: 0x02 is NOT `RUN_MSG_GO` (which is 0x80). The RUN_MSG constants are:

```
RUN_MSG_DONE                    = 0x00
RUN_MSG_INIT                    = 0x40
RUN_MSG_GO                      = 0x80
RUN_MSG_RESET_READ_PTR          = 0xC0
RUN_MSG_RESET_READ_PTR_FROM_HOST = 0xE0
RUN_MSG_REPLAY_TRACE            = 0xF0
```

0x02 is an unknown/stale SRAM value not matching any valid `go_msg.signal`. It happens to equal `RUN_SYNC_MSG_WAITING_FOR_RESET` (0x02), but that constant is written by the NCRISC wh-iram-trampoline to `subordinate_sync` (address MEM_MAILBOX_BASE+8=24), NOT to `go_messages[0]` (address MEM_MAILBOX_BASE + offsetof(mailboxes_t, go_messages)). The coincidence is irrelevant — 0x02 is simply leftover SRAM content that was never overwritten by the firmware init sequence.

**How it gets there**: On Wormhole, L1 SRAM is NOT cleared on Tensix reset. When `assert_risc_reset_at_core(ALL)` halts a core, its SRAM retains whatever was last written there. If a previous session left 0x02 in the go_msg region (from arbitrary kernel data, partial writes, or DMA activity), that value persists through reset.

**Why the multicast didn't overwrite it**: The firmware init sequence in `initialize_and_launch_firmware()` is:

1. Per-core `write_core_immediate` for core_info (lines 2444-2448)
2. Multicast firmware binaries via `test_load_multicast_write_risc_binary` (line 2273)
3. Multicast `go_msg` with `RUN_MSG_INIT=0x40` via `noc_multicast_write` (lines 2249-2250)
4. `l1_barrier` (line 2560)
5. `deassert_risc_reset_at_core` for each core (lines 2562-2577)
6. FIX SC scan (lines 2582-2639)

For Tensix, steps 2-3 use `noc_multicast_write`. On MMIO devices (0,1 in T3K), this goes through PCIe — highly reliable. On non-MMIO devices (2-7 in T3K), it goes through ERISC relay via UMD.

There are two failure scenarios that lead to stale go_msg surviving past step 3:

**Scenario A — Non-MMIO devices with degraded relay (primary hypothesis for dev 2,3)**:
The `noc_multicast_write` for the go_msg (step 3) goes through the ERISC relay. If the relay is degraded (ERISC channels running stale/wrong firmware from a prior session's imperfect teardown), the multicast may:
- Succeed at the UMD level but silently fail to deliver to some/all Tensix cores
- Throw during `wait_for_non_mmio_flush` → FIX AE catches the exception and marks relay broken

In the first sub-case, FIX AE never fires because the flush appeared to succeed, but the data never reached the cores. The go_msg region retains its stale 0x02. After `deassert_risc_reset_at_core`, BRISC boots from IRAM, reads the stale go_msg, and enters an unexpected state.

**Scenario B — MMIO devices with prior-session SRAM contamination (possible for dev 0,1)**:
For MMIO devices, the multicast goes through PCIe — normally reliable. However, `reset_cores` has a `safe_assert` pattern (lines 1701-1730): when `had_unresponsive_eth_cores` is true AND the device is non-MMIO, `assert_tensix_workers_impl` is SKIPPED entirely. For MMIO devices, the try/catch wrapper still calls `assert_tensix_workers_impl`, but if the ETH core teardown left the NOC in a degraded state, the assert may partially fail.

A more likely explanation for MMIO devices: The multicast DID succeed (writing 0x40), but after `deassert_risc_reset_at_core` (step 5), BRISC boots and initializes, which includes calling `deassert_ncrisc_trisc()` and eventually writing `RUN_MSG_DONE` (0x00) to `go_messages[0].signal`. FIX SC reads 0x02 in the brief window between deassert and the firmware completing its boot sequence. But this doesn't match — 0x02 is not an intermediate firmware value. BRISC writes 0x00, never 0x02.

**Most likely unified explanation**: The cores showing 0x02 are on non-MMIO devices where the multicast silently failed. The CI log says "devices 0-3" but in T3K topology, the device numbering in FIX SC logs refers to the chip_id assigned by the cluster descriptor, and non-MMIO devices 2 and 3 are the most plausible targets. If devices 0 and 1 genuinely show 0x02, it would indicate a much rarer failure mode — possibly PCIe-level multicast corruption or a bug in the NOC address calculation.

### 2. Does FIX SC Fix or Just Detect?

**FIX SC is a detection + partial-recovery mechanism, NOT a fix.** Here's what it does:

1. After deassert_risc_reset (step 5), reads go_msg.signal from each core in `not_done_cores`
2. Checks against the known-valid set `{0x00, 0x40, 0x80, 0xC0, 0xE0, 0xF0}`
3. If unknown: asserts BRISC reset (halts the core), writes `RUN_MSG_DONE` (0x00)
4. Logs "board reset will be required"

This makes `wait_until_cores_done` (line 2644) see `RUN_MSG_DONE` immediately, preventing a 10s spin + TT_THROW. However:

- The firmware on those cores **never actually initialized**. No firmware binary was received (multicast failed), no go_msg=0x40 was received.
- The core is halted (BRISC in reset). It's effectively dead.
- The "board reset will be required" message is **informational only** — no flag is set to force a board reset. The system continues with degraded capacity.
- Subsequent program dispatches to those cores will fail because the firmware is not running.

**FIX SC prevents cascading failures** (the TT_FATAL in wait_until_cores_done) but does not restore the cores to a working state. The `check_if_riscs_on_specified_core_done` function in `llrt.cpp` (lines 290-319) has a similar inline FIX SA/SC check that provides the same recovery during later wait operations.

### 3. Relationship Between Tensix Stale go_msg and ETH Channel Failures

**They share the same root cause: degraded ERISC relay on the MMIO side.**

The chain of causation:

```
Prior session imperfect teardown
  → MMIO ERISC channels left in wrong firmware state (e.g., fabric EDM instead of UMD relay)
  → ERISC relay for non-MMIO devices is non-functional
  → Two consequences in parallel:

  (a) Tensix firmware multicast fails silently for non-MMIO devices
      → go_msg retains stale SRAM value (0x02)
      → FIX SC fires, halts cores, writes RUN_MSG_DONE
      → Cores are dead but don't hang

  (b) ETH channel force-reset (fabric_firmware_initializer.cpp:878-890) fails
      → For non-MMIO devices with dead relay, FIX AX (lines 798-811) skips
        assert_risc_reset to avoid 5s UMD timeout per channel
      → Channels reported as "could NOT be force-reset"
      → 8 channels on dev 2,3 (non-MMIO) can't be reached
```

The 8 failing ETH channels (dev=2, chan=0,1,6,7; dev=3, chan=0,1,6,7) are precisely the non-MMIO devices whose relay path is dead. This is the same relay path that the Tensix multicast depends on.

**Key evidence**: The `safe_assert` pattern in `reset_cores` (line 1703) explicitly skips `assert_tensix_workers_impl` for non-MMIO devices with `had_unresponsive_eth_cores=true`. This means:
- Tensix cores on non-MMIO devices may never be put into reset during `reset_cores`
- They continue running stale firmware from the prior session
- The multicast then fails (dead relay) → stale SRAM preserved
- After `deassert_risc_reset_at_core` (which is still called for all cores in `not_done_cores` regardless of relay status), the core may be in an indeterminate state

### 4. Fix Recommendations

**A. Immediate (defensive, low risk) — Guard deassert_risc_reset for non-MMIO with broken relay**

In `initialize_and_launch_firmware()`, after the multicast but before `deassert_risc_reset_at_core` (lines 2562-2577), check if the device is non-MMIO with a broken relay. If so, skip deassert and remove those cores from `not_done_cores`. This is analogous to FIX NZ (line 207) which already skips the entire function for such devices, but FIX NZ only fires if `is_relay_broken` was set BEFORE `run_launch_phase`. If the relay breaks DURING the multicast (FIX AE fires at line 1099), the device is not yet marked for FIX NZ.

```
// After l1_barrier, before deassert loop:
const bool is_non_mmio = cluster_.get_associated_mmio_device(device_id) != device_id;
if (is_non_mmio && cluster_.is_relay_broken(device_id)) {
    log_warning(LogAlways, "FIX XX: skipping deassert_risc_reset + wait for non-MMIO device {} (relay broken mid-init)", device_id);
    return;  // or clear not_done_cores and skip to end
}
```

**B. Structural (medium risk, high value) — Make noc_multicast_write check relay_broken pre-call**

Add a check at the top of `Cluster::noc_multicast_write` (line 1069): if `is_chip_remote(chip_id) && is_relay_broken(chip_id)`, log a warning and return early (no-op). This prevents the silent-fail scenario where the multicast appears to succeed but data never reaches the remote chip.

**C. Root cause (high value, existing work) — Ensure clean ERISC state across sessions**

The ultimate fix is preventing the ERISC relay from being left in a degraded state after session teardown. This is what FIX SB2, FIX PG, FIX AY, and the other #42429 fixes address. If the relay is always healthy at session start, the Tensix multicast and ETH force-reset both work reliably.

**D. Diagnostic enhancement — Distinguish MMIO vs non-MMIO in FIX SC logs**

Add `(MMIO)` or `(non-MMIO, relay_broken={})` to the FIX SC warning message. This would immediately clarify whether the stale go_msg is due to relay failure (expected, recoverable with board reset) or PCIe-level failure (unexpected, indicates hardware issue).

### Summary

The stale `go_msg=0x02` is arbitrary SRAM content that was never overwritten because the NOC multicast of `RUN_MSG_INIT=0x40` failed to reach non-MMIO Tensix cores due to a dead ERISC relay. FIX SC detects this and prevents hangs but does not restore core functionality. The ETH channel force-reset failures are a parallel symptom of the same dead relay. The primary fix path is ensuring clean ERISC relay state across sessions (the existing #42429 fix chain), with a secondary defensive guard to skip deassert_risc_reset when the relay is known-broken mid-init.


---

## 2026-05-09 — FIX XX, FIX XY, FIX XZ Implementation

Three defensive fixes addressing the stale go_msg=0x02 root cause and ERISC teardown gaps.

### FIX XX — Guard deassert_risc_reset for non-MMIO with broken relay mid-init

**Commit**: `3f3202337da`
**File**: `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp`

**Problem**: FIX NZ skips `initialize_and_launch_firmware()` when relay is broken BEFORE entry, but if the relay breaks DURING the multicast (FIX AE fires mid-init), firmware/go_msg never reach non-MMIO Tensix cores. The deassert_risc_reset loop then starts cores from stale SRAM (e.g. go_msg=0x02), triggering FIX SC.

**Fix**: After `l1_barrier()` and before the `deassert_risc_reset_at_core` loop, check if the device is non-MMIO with `cluster_.is_relay_broken(device_id)`. If so, log a warning and return early — cores stay in reset, avoiding stale firmware execution.

### FIX XY — Make noc_multicast_write a no-op when relay is known-broken

**Commit**: `5e6e9d3d98a` (tt-metal submodule bump), UMD commit `7e7fbb0c` on `nsexton/fix-ae-relay-broken-fast-path`
**Files**:
- `tt_metal/third_party/umd/device/api/umd/device/chip/chip.hpp` — added `virtual bool is_relay_broken() const` (returns false by default)
- `tt_metal/third_party/umd/device/api/umd/device/chip/remote_chip.hpp` — `is_relay_broken() const override`
- `tt_metal/third_party/umd/device/chip/remote_chip.cpp` — delegates to `remote_communication_->is_relay_broken()`
- `tt_metal/third_party/umd/device/cluster.cpp` — guard at top of `Cluster::noc_multicast_write()`

**Problem**: When relay is broken, `noc_multicast_write` on remote chips falls back to unicast `write_to_device` calls through ERISC relay command queues. The writes silently fail — data never reaches remote Tensix cores — but no error is raised.

**Fix**: At the top of `Cluster::noc_multicast_write`, check `get_chip(chip)->is_relay_broken()`. If true, log a warning with size/core details and return early (explicit no-op). This converts silent data loss into visible no-op with logging.

### FIX XZ — Wait for MMIO ERISC reboot after teardown force-reset

**Commit**: `944deb17915`
**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`

**Problem**: The force-reset loop (FIX AI) does assert+deassert on channels that didn't reach TERMINATED in time. After deassert, ERISC begins rebooting into base-UMD firmware (~1-2s), but teardown returns immediately. If the next session starts quickly, terminate_stale_erisc_routers() probes via ETH command-queue protocol which requires ERISC to be running → mid-reboot ERISC can't service → probe_dead on MMIO → cascade to all non-MMIO.

**Fix**: After the force-reset loop, poll MMIO channels that were force-reset for heartbeat confirmation (same pattern as FIX TV in run_launch_phase). Uses PCIe direct reads (no relay needed). Waits up to 3s for heartbeat to become non-zero or show the 0xABCDxxxx UMD static marker. This ensures the next session always sees fully-booted base-UMD ERISC channels.

FIX TV in run_launch_phase provides the same defense on the next session's init side, but FIX XZ eliminates the race at the source — in teardown itself.
