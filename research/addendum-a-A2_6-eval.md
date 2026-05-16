<!--
SUMMARY: Evaluation of Addendum A.2.6 — tt_fabric::Reconfigure(new_config) for minimal-diff reroute — against racecondition-hunt branch
KEYWORDS: addendum-a, A.2.6, reconfigure, fabric, config-transition, reinit, SetFabricConfig, teardown
SOURCE: Code analysis of nsexton/0-racecondition-hunt branch at /workspace/group/worktrees/racecondition-main/
SCOPE: Whether Reconfigure() applies to this branch; full reinit path analysis; dependency assessment
USE WHEN: Evaluating whether to implement partial fabric reconfiguration for parameter-only changes
-->

# Addendum A.2.6 Evaluation: tt_fabric::Reconfigure(new_config)

**Recommendation**: Add `tt_fabric::Reconfigure(new_config)` that performs a minimal-diff reroute when only `reliability_mode`, `num_planes`, or `udm_mode` change — avoiding full firmware reinit when only a parameter changes between adjacent tests.

**Verdict: NOT APPLICABLE to racecondition-hunt. Also architecturally blocked by current design.**

---

## Q1: Are there adjacent test suites that switch between different active configs?

**No. Every config transition goes through DISABLED.**

The codebase enforces this with a hard `TT_FATAL` at `tt_metal/impl/context/metal_env.cpp:265-269`:

```cpp
} else {
    TT_FATAL(
        this->fabric_config_ == fabric_config,
        "Tried to override previous value of fabric config: {}, with: {}",
        enchantum::to_string(this->fabric_config_),
        enchantum::to_string(fabric_config));
}
```

This guard at `metal_env.cpp:260-269` only allows two transitions:
1. **DISABLED → X** (any non-DISABLED config)
2. **X → DISABLED** (teardown)

A direct `FABRIC_1D → FABRIC_2D` transition would hit the `TT_FATAL` and abort. The only legal sequence is `FABRIC_1D → DISABLED → FABRIC_2D`.

**Test fixture evidence**: `multi_device_fixture.hpp:167-169` (SetUp) calls `SetFabricConfig(config_.fabric_config)` and `multi_device_fixture.hpp:317-319` (TearDown) calls `SetFabricConfig(DISABLED)`. Every GTest fixture follows this pattern: DISABLED→X in SetUp, X→DISABLED in TearDown.

The fixture classes that define non-DISABLED configs are:
- `MeshDevice2x4Fabric1DFixture` — `FABRIC_1D` (`multi_device_fixture.hpp:483`)
- `GenericMeshDeviceFabric2DFixture` — `FABRIC_2D` (`multi_device_fixture.hpp:489`)
- `MeshDevice2x4Fabric2DFixture` — `FABRIC_2D` (`multi_device_fixture.hpp:496`)
- `MeshDevice1x4Fabric2DUDMFixture` — `FABRIC_2D` + `FabricUDMMode::ENABLED` (`multi_device_fixture.hpp:505`)
- `MeshDevice2x4Fabric2DUDMFixture` — `FABRIC_2D` + `FabricUDMMode::ENABLED` (`multi_device_fixture.hpp:536`)

When different test cases within `distributed_unit_tests` use different fixtures (e.g. one uses `FABRIC_1D`, the next uses `FABRIC_2D`), GTest runs TearDown (→DISABLED) then SetUp (→new config). There is never a direct config-to-config hop.

For the GAP regression tests (test_gap1 through test_gap77 in `tests/tt_metal/distributed/`), most fork subprocesses that do their own `SetFabricConfig` calls inside child processes — these are isolated by process boundaries, not by parameter-diff reconfigure.

**Specific sequences in this branch**: All 70+ GAP tests exclusively use `FABRIC_2D` with `STRICT_SYSTEM_HEALTH_SETUP_MODE`. No test switches between `reliability_mode` values or between `FABRIC_1D`/`FABRIC_2D` within the same process — they all follow the fork-child-SIGKILL-parent-reinit pattern where config changes happen across process boundaries.

---

## Q2: What does the current full reinit path look like?

The full config transition is:

### Teardown (X → DISABLED) — `metal_env.cpp:340-498`

1. **Export channel trimming capture** (`metal_env.cpp:254-256`) — diagnostic data preservation
2. **Wait for all ETH router TERMINATED** (`metal_env.cpp:345-465`) — poll every active ETH channel on every chip for `EDMStatus::TERMINATED`, with 5s timeout per channel. Force-resets (assert+deassert RISC) on timeout.
3. **Restore ERISC0** (`metal_env.cpp:467-478`) — for cleanly terminated channels, assert+deassert ERISC0 reset to restart base UMD firmware
4. **Reset fabric state** (`metal_env.cpp:490-493`):
   - `fabric_config_ = DISABLED`
   - `cluster.configure_ethernet_cores_for_fabric_routers(DISABLED)` — releases ETH core reservations
   - `num_fabric_active_routing_planes_ = 0`
5. **Clear control plane fabric context** (`metal_env.cpp:500-502`)
6. **MetalContext force_reinit** — `metal_env.cpp:248` sets `force_reinit_ = true`, which causes `MetalContext::initialize()` (`metal_context.cpp:186-190`) to call `teardown()` and fully reconstruct on the next `SetFabricConfig`

### Re-init (DISABLED → Y) — `metal_env.cpp:241-332` + `MetalContext::initialize()`

1. **Store new parameters** — `fabric_config_`, `reliability_mode_`, `udm_mode_`, `num_routing_planes_`, etc.
2. **Reconstruct control plane** (`metal_env.cpp:327-330`) — full topology discovery + routing table computation
3. **MetalContext::initialize()** (`metal_context.cpp:186-190`) — `teardown()` + full reinit including:
   - `DeviceManager::initialize()` — opens all devices
   - `FabricFirmwareInitializer::init()` → `compile_and_configure_fabric()` (2680-line function) — compile fabric kernels, load firmware to all ETH cores, run ring-sync handshake
   - `FabricFirmwareInitializer::configure()` — `wait_for_fabric_router_sync()` + `verify_all_fabric_channels_healthy()`

### What's actually necessary for a parameter-only change?

For a change from `(FABRIC_2D, STRICT, UDM_DISABLED)` to `(FABRIC_2D, RELAXED, UDM_DISABLED)`:
- **Could skip**: firmware compilation, firmware loading, ERISC reset/restart, ETH core reservation (same topology = same core allocation)
- **Still need**: control plane routing table update (reliability mode affects routing decisions), ring-sync re-validation
- **Unknown**: whether reliability_mode is baked into the firmware binary at compile time or is a runtime parameter on the ERISC

---

## Q3: Do any race-condition fixes depend on the full teardown during config changes?

**Yes — multiple fixes critically depend on the full teardown/reinit cycle.**

Key dependencies:

1. **FIX RZ2** (`fabric_firmware_initializer.cpp:282-297`): Clears `fabric_stale_base_umd_channels_` flag ONLY after successful ring-sync + health check in `configure()`. A partial reconfigure that skips ring-sync would leave stale flags set, causing cascading skips.

2. **FIX QU** (`fabric_firmware_initializer.cpp:230-261`): Re-asserts `fabric_relay_path_broken_` and `fabric_channels_not_ready_for_traffic_` flags after `Device::configure_fabric()` resets them. This fix depends on the full `configure()` → `configure_fabric()` → `FabricFirmwareInitializer::configure()` call chain running.

3. **FIX M** (`device.cpp:664` area): Detects channels with base-UMD relay firmware and transitions them via launch_msg during `configure_fabric()`. A partial reconfigure that skips device configure would leave base-UMD channels running stale firmware.

4. **FIX EXT/EXT2** (`device.cpp:461-467`): External out-of-mesh ETH channel handling during `configure_fabric()` — depends on full firmware loading path.

5. **FIX TG2** (`test_gap77`): Partial L1 clear for base-UMD channels — happens during firmware init, not separable from it.

6. **teardown_fabric_config()** (`metal_env.cpp:340-498`): The entire 150-line TERMINATED-poll + force-reset sequence runs during every X→DISABLED transition. This is where stale ERISC firmware from prior sessions is cleaned up. Skipping this for a "parameter-only" change would leave potentially corrupt ERISC firmware running with the wrong configuration.

**A minimal-diff Reconfigure() that skips teardown would break multiple FIX patches** — specifically FIX RZ2, FIX QU, FIX M, and the teardown ERISC cleanup that all GAP tests depend on for inter-test isolation.

---

## Q4: Is there any existing "reconfigure" or "partial reinit" mechanism?

**No.** The codebase has:

- **`DYNAMIC_RECONFIGURATION_SETUP_MODE`** (`fabric_types.hpp:87`) — defined as enum value `2` in `FabricReliabilityMode` but annotated as `// Unsupported - fabric can be setup at runtime. Placeholder`. Only referenced in `rtoptions.cpp:638` for env var parsing. Not implemented anywhere.

- **`soft_reset`** — `skip_soft_reset_channels` in `device.cpp:406-467` is about which ETH channels to preserve during `configure_fabric()`, not a partial reinit mechanism.

- **`quiesce_and_restart_fabric_workers()`** (`device.cpp:727`) — a per-device quiesce/restart that terminates + relaunches fabric workers. This is closer to a partial reinit but operates at the device level (within a single config), not at the config-transition level. It's used during `MeshDevice::close()` → `quiesce_devices()`, not for config changes.

- **`force_reinit_`** (`metal_env.cpp:248`, `metal_context.cpp:160-190`) — a blunt boolean flag that forces full MetalContext teardown+reinit. No diffing or parameter comparison.

There is no mechanism to diff old-vs-new config parameters and skip unchanged subsystems.

---

## Q5: What would Reconfigure() need to do differently?

For a hypothetical same-firmware parameter change (e.g. `STRICT → RELAXED` reliability mode):

### Steps that COULD be skipped
1. Firmware compilation (`compile_and_configure_fabric`) — ~2680 lines of FFI, biggest time sink
2. ERISC reset/restart — firmware binary is the same
3. ETH core reservation/release (`configure_ethernet_cores_for_fabric_routers`) — topology unchanged
4. Base-UMD channel transition (FIX M path) — firmware already loaded

### Steps that MUST still run
1. Control plane reconstruction (reliability mode affects routing tables)
2. Ring-sync re-validation (must verify new config parameters propagated)
3. Health check (verify_all_fabric_channels_healthy)
4. All FIX flag management (FIX RZ2, FIX QU, FIX M stale detection)

### Why this is impractical today
1. **The TT_FATAL guard** (`metal_env.cpp:265`) makes config-to-config transitions illegal. Removing it requires careful auditing of every code path that assumes the only legal transitions are DISABLED↔X.
2. **Reliability mode is passed to `ControlPlane` constructor** (`metal_env.cpp:600-615`), which is constructed once in `initialize_control_plane_impl()`. Changing it requires reconstructing the control plane, which is nearly as expensive as full reinit.
3. **No test benefits from this** — all tests in racecondition-hunt use `FABRIC_2D` + `STRICT_SYSTEM_HEALTH_SETUP_MODE`. There are no parameter-only transitions to optimize.

---

## Summary

| Question | Answer |
|----------|--------|
| Adjacent tests with different active configs? | **No** — TT_FATAL prevents it; all transitions go through DISABLED |
| Full reinit path cost? | ~150 lines of teardown (TERMINATED poll + ERISC reset) + full MetalContext reinit including firmware compile+load |
| Race-condition fixes depend on full teardown? | **Yes** — FIX RZ2, FIX QU, FIX M, FIX EXT, and ERISC cleanup all require full cycle |
| Existing partial reinit mechanism? | **No** — `DYNAMIC_RECONFIGURATION_SETUP_MODE` is a placeholder enum, not implemented |
| Concrete Reconfigure() design? | Would need to skip firmware compile/load, keep ETH reservations, but still rebuild control plane — net savings unclear |

**Recommendation: SKIP A.2.6.** The current architecture prevents config-to-config transitions (TT_FATAL guard), no test in this branch would benefit from it, and multiple race-condition fixes critically depend on the full teardown/reinit cycle. The `DYNAMIC_RECONFIGURATION_SETUP_MODE` placeholder suggests this is a known future feature, but it's orthogonal to the racecondition-hunt fixes.
