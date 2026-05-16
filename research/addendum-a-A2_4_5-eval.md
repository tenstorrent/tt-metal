<!--
SUMMARY: Evaluation of Addendum A recommendations A.2.4 (batched fabric heartbeat reads) and A.2.5 (explicit ETH-dispatch fabric auto-enable) against nsexton/0-racecondition-hunt branch
KEYWORDS: addendum-a, A.2.4, A.2.5, heartbeat, batched-read, auto_enable_fabric, FABRIC_1D, eth-dispatch, racecondition-hunt, batch-t3k
SOURCE: Code analysis of worktrees racecondition-main and nsexton/0-batch-t3k-ttnn-unit
SCOPE: Applicability of two specific optimization recommendations to the race-condition fix branch
USE WHEN: Deciding whether A.2.4/A.2.5 need to be addressed before merging racecondition-hunt
-->

# A.2.4 and A.2.5 Evaluation — nsexton/0-racecondition-hunt

## A.2.4 — Batched fabric heartbeat reads

### 1. Does racecondition-hunt have capture_fabric_heartbeats or equivalent?

**No.** The function `capture_fabric_heartbeats` does not exist in this branch. It is a batch-t3k-only addition.

```
grep -rn "capture_fabric_heartbeats" /workspace/group/worktrees/racecondition-main  → (no results)
grep -rn "capture_fabric_heartbeats" /workspace/group/worktrees/nsexton/0-batch-t3k-ttnn-unit  → 3 hits
```

In batch-t3k, `capture_fabric_heartbeats` lives at:
- `tests/tt_metal/tt_metal/common/multi_device_fixture.hpp:510-539`

It iterates over `mesh->get_device_ids()`, calls `tt::tt_fabric::read_fabric_telemetry(fabric_node_id)` per device, which internally iterates per-channel (`fabric_telemetry_reader.cpp:104`), issuing one `cluster.read_from_device()` MMIO read per channel (`fabric_telemetry_reader.cpp:80`). On T3K with 8 devices x ~4 active ETH channels each, this is ~32 individual reads per snapshot.

### 2. Has this branch changed the heartbeat capture mechanism at all?

**No.** This branch does not touch the `fabric_telemetry_reader.cpp` API. The telemetry read path (`read_fabric_telemetry`) is identical between both branches.

What this branch *does* have is two per-router heartbeat poll loops in `risc_firmware_initializer.cpp` used during firmware init/teardown:

- **FIX TV** (`risc_firmware_initializer.cpp:237-333`): Polls MMIO ETH heartbeat after `reset_cores()` during `run_launch_phase()`. One `cluster_.read_reg()` per poll_state per iteration, with 10ms intervals, up to 3000ms.

- **FIX AR/AC** (`risc_firmware_initializer.cpp:534-589`): Bulk parallel heartbeat poll during teardown. Same pattern — one `cluster_.read_reg()` per ETH core per poll iteration, shared 5000ms deadline.

Both use raw `cluster_.read_reg()` (single 4-byte PCIe read per core) rather than the telemetry API.

### 3. Are heartbeat checks used more or less frequently?

**More frequently in racecondition-hunt**, but at a different level. The FIX TV and FIX AR polls happen during firmware init and teardown (once per mesh open/close cycle). They are NOT per-test telemetry checks.

In batch-t3k, `capture_fabric_heartbeats` runs twice per test: once in `SetUp()` (line 936) and once in `TearDown()` via `check_fabric_heartbeats_advanced()` (line 552). This is the per-test cost that A.2.4 targets.

Racecondition-hunt's `MeshDeviceFixtureBase` has no per-test heartbeat capture at all — it opens/closes mesh per test, so the FIX TV/AR polls fire on every SetUp/TearDown but as part of the init sequence, not as separate telemetry reads.

### 4. Do race-condition fixes depend on per-router granularity?

**Yes, critically.** The FIX TV and FIX AR heartbeat polls check individual ETH core heartbeats to determine which specific cores have rebooted after a PCIe reset:

- `risc_firmware_initializer.cpp:267-291`: Each `poll_state` tracks a specific `tt_cxy_pair` (chip + core). The fix needs to know per-core whether it shows a static `0xABCDxxxx` UMD marker vs an incrementing counter.

- `risc_firmware_initializer.cpp:556-589` (FIX AR): Same per-core poll pattern. The `ac_heartbeat_any_ready` flag (line 556) gates FIX AY downstream — if NO individual core confirmed heartbeat, the relay is not restored.

A batched/aggregate read would destroy this per-core granularity, making it impossible to distinguish "core 3 on chip 0 is still in ROM boot" from "all cores healthy."

### 5. Is there an AERIS/telemetry API that supports batched reads?

The existing `read_fabric_telemetry(FabricNodeId)` (`fabric_telemetry_reader.cpp:86-113`) already does a per-device batch (it iterates channels internally and issues `l1_barrier` once), but it still does one `cluster.read_from_device()` per channel. There is no single-read API that returns all channels' heartbeats in one MMIO transaction.

The telemetry infrastructure (`fabric_telemetry_reader.hpp:45-62`) supports per-node reads. A true batched read would require firmware changes to aggregate heartbeat counters into a single L1 region readable in one transaction. No such mechanism exists in either branch.

### A.2.4 Verdict

**Does not apply to racecondition-hunt.** This branch:
- Does not have `capture_fabric_heartbeats` or per-test telemetry checks
- Uses per-core heartbeat reads for firmware init/teardown only (not per-test)
- Fundamentally depends on per-core granularity for its race-condition fixes
- A batched read optimization is relevant only to batch-t3k's shared-fixture per-test health checks

---

## A.2.5 — Explicit ETH-dispatch fabric auto-enable

### 1. Does the T3K auto-enable FABRIC_1D behavior exist in this branch?

**Yes.** The auto-enable logic is in `device_manager.cpp:272-275`:

```cpp
// device_manager.cpp:272-275
if (any_remote_devices && !is_mock) {
    auto fabric_config = ctx_.get_fabric_config();
    if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
        fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D;
```

This is the same code path as main. When `DeviceManager::create_all_devices()` sees remote (non-MMIO) devices and no explicit fabric config, it auto-enables `FABRIC_1D` for ETH dispatch.

### 2. Does racecondition-hunt add fixtures that use only MMIO chips but pay fabric init cost?

**No.** All new test fixtures in this branch (`MeshDeviceFixtureBase` subclasses in `test_gap*.cpp`) explicitly set `fabric_config = tt_fabric::FabricConfig::FABRIC_2D` in their constructor. They intentionally enable fabric because they are testing fabric race conditions.

There are no new MMIO-only fixtures that unintentionally pay fabric init cost. The nightly Python tests in `tests/nightly/t3000/ccl/` use `ttnn.open_mesh_device()` which goes through the auto-enable path, but they need fabric for their CCL workloads.

### 3. Do any race-condition fixes touch or workaround the auto-enable behavior?

**Yes — FIX QW2.** Commit `257c42a96d5` adds a guard for `SetFabricConfig(DISABLED)` in `TearDown` when `FABRIC_1D` init threw:

```
fix: FIX QW2 — guard SetFabricConfig(DISABLED) in TearDown when FABRIC_1D init threw (#42429)
```

This directly handles the consequence of auto-enable: when the auto-enabled FABRIC_1D fails during init and throws, the TearDown must not call `SetFabricConfig(DISABLED)` because the fabric was never successfully initialized.

The FIX RX guard in `multi_device_fixture.hpp:280-307` also handles broken fabric in TearDown, which can occur after auto-enabled FABRIC_1D degrades during test execution.

### 4. What would the flag look like?

`MeshDeviceConfig` (at `tt_metal/api/tt-metalium/mesh_config.hpp:16-38`) currently has no fabric-related fields — it only carries `mesh_shape`, `offset`, and `physical_device_ids`.

`MeshDeviceFixtureConfig` (at `multi_device_fixture.hpp:38-50` in batch-t3k) already has `fabric_config` as a field, and batch-t3k added `auto_enable_fabric()` as a Traits static method:
- `multi_device_fixture.hpp:393-401` (batch-t3k): type trait detection
- `multi_device_fixture.hpp:773-782` (batch-t3k): compile-time resolution

In racecondition-hunt, `MeshDeviceFixtureBase` has a simpler model: `fabric_config` is set in the `Config` struct passed to the constructor. There is no auto-enable detection because every gap test explicitly sets `FABRIC_2D`.

To add the opt-out flag:
- **Runtime level**: Add `bool enable_fabric_for_eth_dispatch = true` to `MeshDeviceConfig` (`mesh_config.hpp`)
- **Test fixture level**: Add `auto_enable_fabric` trait (as batch-t3k already does) or a `skip_fabric_auto_enable` field to `MeshDeviceFixtureBase::Config`

### 5. Which test suites in this branch would opt-out?

**None.** Every new C++ gap test in this branch explicitly enables `FABRIC_2D`:
- 50+ `test_gap*.cpp` files, all with `.fabric_config = tt_fabric::FabricConfig::FABRIC_2D`
- These tests *need* fabric to exercise the race conditions they're testing

The only tests that might benefit from opt-out are the ones that don't exist in this branch but do in batch-t3k (e.g., `MeshDevice1x2SharedTraits` which has `fabric_config = DISABLED` and no `auto_enable_fabric`).

### A.2.5 Verdict

**Does not apply to racecondition-hunt.** This branch:
- Has the auto-enable code path (unchanged from main) but does not add any MMIO-only fixtures
- All new fixtures explicitly request `FABRIC_2D` — they need fabric
- FIX QW2 workarounds the auto-enable failure case, but that's a fix for a crash, not a performance optimization
- The `auto_enable_fabric()` trait pattern from batch-t3k is the right solution, but it's only needed when shared fixtures with `fabric_config = DISABLED` need to detect runtime auto-enable — which only batch-t3k does

---

## Summary

```
Recommendation    Applies?  Reason
─────────────────────────────────────────────────────────────────────────
A.2.4 Batched     NO       No per-test heartbeat capture exists.
  heartbeat                 FIX TV/AR polls are per-core by design
  reads                     (needed for race-condition diagnosis).

A.2.5 Explicit    NO       All new fixtures explicitly enable FABRIC_2D.
  auto-enable               No MMIO-only fixtures paying fabric init
  opt-out                   cost unnecessarily. FIX QW2 handles the
                            auto-enable failure case but is a crash fix.
```

Both recommendations are relevant only to batch-t3k's shared-fixture model where per-test telemetry and MMIO-only suites create unnecessary overhead. Racecondition-hunt's per-test open/close model and explicit FABRIC_2D configs make both recommendations moot for this branch.
