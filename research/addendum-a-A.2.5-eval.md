# A.2.5 Evaluation: MeshDeviceConfig::auto_enable_fabric_for_eth_dispatch

**Branch**: `nsexton/0-racecondition-hunt`
**Worktree**: `/workspace/group/worktrees/racecondition-main/`

---

## 1. Does the problem described in A.2.5 exist in racecondition-hunt?

**Yes, identically to main.** The implicit FABRIC_1D auto-enable is present and unchanged.

**Primary site** — `tt_metal/impl/device/device_manager.cpp:272-288`:
```cpp
if (any_remote_devices && !is_mock) {
    auto fabric_config = ctx_.get_fabric_config();
    if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
        fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D;
        // TODO: This is using an internal API.
        // Externally, we should decide how/where to have SetFabricConfig on the correct MetalEnv
        ctx_.set_fabric_config(
            fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
```

The `any_remote_devices` flag is computed at `device_manager.cpp:219-233` by scanning for non-MMIO chips or galaxy clusters. When it's true and the user hasn't explicitly called `SetFabricConfig()`, the DeviceManager silently upgrades from `DISABLED` to `FABRIC_1D`.

**`MeshDeviceConfig` has no fabric control** — `tt_metal/api/tt-metalium/mesh_config.hpp:16-38`:
The class only carries `mesh_shape_`, `offset_`, and `physical_device_ids_`. There is no `auto_enable_fabric_for_eth_dispatch` flag or any fabric-related field. Users creating a `MeshDevice` have zero visibility into whether FABRIC_1D will be auto-enabled behind their back.

The existing `TODO` comment at `device_manager.cpp:276-277` literally says:
> "This is using an internal API. Externally, we should decide how/where to have SetFabricConfig on the correct MetalEnv"

This confirms the authors already know this is architecturally wrong.

---

## 2. Comparison with nsexton/0-batch-t3k-ttnn-unit

**The situation is identical.** Both branches have:

- The same `device_manager.cpp` auto-enable block (lines 272-296 in racecondition-hunt, same range in batch-t3k)
- The same `MeshDeviceConfig` with no fabric fields (`mesh_config.hpp`)
- The same `TODO` comment acknowledging the internal API misuse

The diff between the two branches in `device_manager.cpp` is trivial — batch-t3k uses `get_target_device_type() == tt::TargetDevice::Mock` for the mock check (line ~271), while racecondition-hunt uses the newer `is_mock_or_emulated()` API. This is an unrelated refactor; the fabric auto-enable logic is byte-for-byte identical.

**Neither branch** has made any progress on surfacing the flag in `MeshDeviceConfig`.

---

## 3. What would implementing this recommendation look like?

The change involves three pieces:

**a) Add a field to `MeshDeviceConfig`** (`tt_metal/api/tt-metalium/mesh_config.hpp`):
Add a boolean `auto_enable_fabric_for_eth_dispatch_` field (default `true` for backward compatibility). Extend the constructor to accept it. This makes the implicit behavior explicit at the API surface.

**b) Thread the flag through `MeshDevice::create()` → `DeviceManager::open_devices()`**:
Currently `create_mesh_device()` in `metal_env.cpp:781-803` passes the `MeshDeviceConfig` only to `MeshDeviceImpl::create()`, which doesn't feed it into `DeviceManager`. The flag would need to reach `DeviceManager::open_devices()` so the auto-enable block at line 274 can check it before silently upgrading to `FABRIC_1D`.

**c) Modify the auto-enable guard** in `device_manager.cpp:274`:
Change from:
```cpp
if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
```
To something like:
```cpp
if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED && config.auto_enable_fabric_for_eth_dispatch()) {
```

When the flag is `false`, the DeviceManager would either skip the auto-enable entirely (leaving dispatch without fabric — only valid for MMIO-only topologies) or log a fatal/warning if remote devices are present but fabric is disabled and auto-enable is off.

---

## 4. Interactions with race-condition fixes on this branch

The racecondition-hunt branch has **significant interactions** that make A.2.5 more, not less, important:

### 4a. GAP-78 teardown ordering (metal_env.cpp:136-161)
The branch upgrades `check_use_count_zero()` from a `log_error` to a `TT_THROW` to catch teardown ordering violations where dispatch is still active during fabric teardown. When FABRIC_1D is silently auto-enabled, callers don't know fabric is active, so they have no reason to expect teardown ordering requirements. Making the auto-enable explicit would help callers understand they have fabric-teardown obligations.

### 4b. FIX QW2 — SetFabricConfig(DISABLED) guard (multi_device_fixture.hpp:199-200)
The branch adds guards in test fixtures that call `SetFabricConfig(DISABLED)` only when `fabric_config != DISABLED`. This was needed because auto-enabled FABRIC_1D could fail during init, leaving the state inconsistent. If the auto-enable were explicit, fixtures could skip the fabric setup path entirely rather than needing after-the-fact cleanup guards.

### 4c. teardown_fabric_config() TERMINATED wait (metal_env.cpp:319-500+)
The branch adds a ~170-line `teardown_fabric_config()` implementation that polls every ETH channel for `EDMStatus::TERMINATED` before releasing resources. This entire teardown path only exists because fabric was enabled. When it was enabled implicitly, the complexity of the teardown (and the race conditions it prevents) is invisible to the caller. An explicit flag would at minimum make it clear that "I opted into fabric, so I need to worry about fabric teardown."

### 4d. FIX NS — Cluster double-discovery elimination (metal_env.cpp:175-190)
The branch eliminates a redundant topology discovery that could hang when stale fabric firmware from a prior session fills relay command queues. This is directly related: the prior session auto-enabled FABRIC_1D, left stale firmware on ETH cores, and the next session's topology discovery hung. An explicit opt-in flag wouldn't prevent stale firmware per se, but it would reduce the number of sessions that accidentally enable fabric (and leave stale firmware) when they didn't intend to use it.

### Summary of interaction

The race-condition fixes on this branch are **downstream consequences** of the implicit auto-enable behavior. The fixes add significant complexity (teardown polling, ordering enforcement, skip guards) that callers don't know they need because they didn't know fabric was enabled. Making the auto-enable explicit (A.2.5) would:
1. Let callers opt out when they don't need remote dispatch, reducing the surface area for race conditions
2. Make teardown obligations visible at the call site
3. Reduce the number of sessions that leave stale ETH firmware behind

A.2.5 is **complementary** to the race-condition fixes, not conflicting. However, it is a **lower priority** than the fixes already on this branch — the fixes address active crashes and hangs, while A.2.5 is an API hygiene improvement that would prevent future callers from stumbling into those crashes.
