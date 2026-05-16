# A.2.1 Evaluation: Make SetFabricConfig Idempotent/Lazy

Branch: `nsexton/0-racecondition-hunt`
Worktree: `/workspace/group/worktrees/racecondition-main/`

---

## 1. Does the problem exist in racecondition-hunt?

**Yes.** `SetFabricConfig` is not idempotent. Every call unconditionally sets `force_reinit_ = true` on line 248 of `metal_env.cpp`, regardless of whether the incoming parameters match the already-stored config:

```
tt_metal/impl/context/metal_env.cpp:248:    force_reinit_ = true;
```

The stored config fields in `MetalEnvImpl` (`metal_env_impl.hpp:121-128`) are:
- `fabric_config_` (line 121)
- `fabric_reliability_mode_` (line 122-123)
- `fabric_tensix_config_` (line 124)
- `fabric_udm_mode_` (line 125)
- `fabric_manager_` (line 126)
- `fabric_router_config_` (line 127)
- `num_fabric_active_routing_planes_` (line 128)

All the necessary state to compare is already stored, but no comparison is performed before `force_reinit_ = true`.

The `force_reinit_` flag is consumed in `MetalContext::initialize()` at `metal_context.cpp:163-164`:
```cpp
if (MetalEnvAccessor(*env_).impl().consume_force_reinit()) {
    force_reinit_ = true;
}
```
Which then triggers a full `teardown()` + reinit at `metal_context.cpp:186-191`, even when the config tuple is identical.

**Where this gets called repeatedly with same params:**

- **Test fixtures** (`multi_device_fixture.hpp:168-173` and `:368-373`): every test's `SetUp()` calls `SetFabricConfig(config_.fabric_config, STRICT_SYSTEM_HEALTH_SETUP_MODE, nullopt, config_.fabric_tensix_config, config_.fabric_udm_mode)`. When tests in a parametric suite share the same config, this forces a full teardown+reinit between each test even though nothing changed.

- **DeviceManager** (`device_manager.cpp:278` and `:291`): the legacy path calls `ctx_.set_fabric_config(fabric_config, STRICT_SYSTEM_HEALTH_SETUP_MODE, 1)` even when fabric is already enabled with the same config. Line 291 explicitly does this for the "Use the same mode" branch — a no-op that still triggers reinit.

- **FabricSwitchManager** (`fabric_switch_manager.cpp:37`): `setup()` calls `SetFabricConfig` each invocation, even if the FabricSwitchManager is being re-setup with the same config.

## 2. Better, worse, or same as nsexton/0-batch-t3k-ttnn-unit?

**Same fundamental problem, but the consequences are slightly worse in racecondition-hunt.**

The core issue — `force_reinit_ = true` on every `set_fabric_config()` call regardless of parameter equality — exists identically in both branches. Neither branch has any idempotency check.

However, racecondition-hunt makes the cost of an unnecessary reinit higher:

- **`teardown_fabric_config()` is now significantly heavier** (`metal_env.cpp:321-474`): the branch adds per-chip, per-channel polling for `EDMStatus::TERMINATED` with a configurable timeout (default appears to be several seconds), force-reset logic for timed-out channels, ERISC0 restore-reset, and a `teardown_timed_out_chips_` set that feeds into `FabricFirmwareInitializer::post_teardown()`. An unnecessary teardown+reinit cycle now involves iterating over all chips and all their ETH channels twice — once for teardown polling, once for reinit.

- **GAP-78 enforcement** (`metal_env.cpp:136-157`): the branch converts the `check_use_count_zero()` check from `log_error` to `TT_THROW`. If any dispatch thread is still draining during an unnecessary reinit triggered by a same-config `SetFabricConfig`, this now throws instead of logging. A lazy/idempotent check would avoid entering teardown entirely, sidestepping this failure mode.

- **FIX NS double-discovery elimination** (`metal_env.cpp:175-195`): the branch specifically eliminates redundant topology discovery. Having `SetFabricConfig` also skip redundant work would be philosophically aligned with this fix.

## 3. What would implementing this look like concretely?

Add an early-return guard at the top of `MetalEnvImpl::set_fabric_config()` (around line 248 of `metal_env.cpp`), **before** the `force_reinit_ = true` assignment:

1. Compare the incoming tuple `(fabric_config, reliability_mode, num_routing_planes, fabric_tensix_config, fabric_udm_mode, fabric_manager, router_config)` against the stored members `(fabric_config_, fabric_reliability_mode_, num_fabric_active_routing_planes_, fabric_tensix_config_, fabric_udm_mode_, fabric_manager_, fabric_router_config_)`.

2. For `num_routing_planes`: the incoming value is `std::optional<uint8_t>` while stored is `uint8_t`. The comparison would need to handle the `nullopt` case (which currently means "use max"). A same-config match should treat `nullopt` as equal to the stored value if the stored value hasn't changed, or compare `num_routing_planes.value_or(current_stored_value) == current_stored_value`.

3. For `FabricRouterConfig`: need to verify it has `operator==` or implement one. It's stored at `metal_env_impl.hpp:127` as a struct — likely needs an equality operator added.

4. If all fields match and `control_plane_ != nullptr` (i.e., already initialized), return `false` without setting `force_reinit_` and without tearing down or reinitializing the control plane.

5. Log at `log_debug` level: "SetFabricConfig called with identical parameters, skipping reinit."

**The change would be ~15 lines in `MetalEnvImpl::set_fabric_config()` plus potentially an `operator==` on `FabricRouterConfig`.**

Separately, the `DeviceManager::initialize()` "Use the same mode" path at `device_manager.cpp:290-292` could simply skip the `set_fabric_config` call entirely when the config already matches, but the idempotency guard makes this unnecessary.

## 4. Interactions with race-condition fixes in this branch

Several race-condition fixes interact with this recommendation:

**GAP-78 teardown ordering (metal_env.cpp:136-157):**
The branch converts `check_use_count_zero()` from `log_error` to `TT_THROW` to catch dispatch-still-active during teardown. An idempotent `SetFabricConfig` would **reduce the surface area for GAP-78 violations** — if the config hasn't changed, we never enter teardown, so we never hit the use-count check. This is purely beneficial.

**Teardown TERMINATED polling (metal_env.cpp:321-474):**
The expensive new per-chip/per-channel TERMINATED polling and force-reset logic in `teardown_fabric_config()` is only meaningful when actually tearing down fabric. Skipping it on same-config calls avoids wasting seconds per unnecessary teardown cycle. **Strongly beneficial** — directly reduces the time window where race conditions could occur.

**FIX AB / teardown_timed_out_chips_ (metal_env_impl.hpp:73-78, 134-138):**
The `teardown_timed_out_chips_` set is populated during `teardown_fabric_config()` and consumed by `FabricFirmwareInitializer::post_teardown()`. An idempotent skip means this set is never populated for no-op calls, which is correct — there's nothing to report if we didn't tear down.

**FIX NS double-discovery (metal_env.cpp:175-195):**
Eliminating redundant topology discovery in `initialize_base_objects()` is the same philosophy as A.2.1. No conflict.

**FIX QW2 (commit 257c42a96d5):**
Guards `SetFabricConfig(DISABLED)` in TearDown when FABRIC_1D init threw. This already-existing guard prevents calling SetFabricConfig when it shouldn't be called. An idempotency check would be a complementary layer — if DISABLED is already the current state, skip the teardown. Currently, calling `SetFabricConfig(DISABLED)` when `fabric_config_` is already `DISABLED` still sets `force_reinit_ = true` (line 248) and enters the teardown path (line 279). The guard at line 260-261 does allow `DISABLED -> DISABLED` transitions (both sides of the `||` are true), and line 272 then calls `teardown_fabric_config()` unnecessarily. **An idempotency check would eliminate this wasted teardown.**

**No conflicts found.** The recommendation is purely additive and would reduce teardown frequency, making every race-condition fix in this branch more effective by reducing the number of times the risky teardown+reinit path executes.
