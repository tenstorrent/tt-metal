# ARC NOC Bug RCA - Shared Context for Researchers

## The Bug

After `ttnn::all_gather` + `quiesce_devices()`, calling "Enqueue dummy ops" 
(`ttnn::write_buffer` + `dispatch_ops_to_device`) to device index 3 
(physical chip 4, the far N300, accessed via non-MMIO ETH fabric) causes a hang.

The hang manifests as:
```
TT_THROW: TIMEOUT: device timeout, the device is unrecoverable
```
fired by `TT_METAL_OPERATION_TIMEOUT_SECONDS=5` after 5 seconds of no acks.

## Symptoms from CI job 72045468285 (run 24641059431) on tt-metal-ci-vm-t3k-14

Device map (T3K, 1x4 mesh):
  - mesh[0,0] = chip 0 (N300, MMIO, left)
  - mesh[0,1] = chip 1 (N300, MMIO, right)  
  - mesh[0,2] = chip 4 (N300, non-MMIO, far left) ← HANGS HERE
  - mesh[0,3] = chip 5 (N300, non-MMIO, far right)

After `post-AllGather quiesce_devices()`, ETH fabric health probe shows:
  Device 0: chans 0-5 → "OTHER" states (0x3f803f80, 0x40004000, 0x40404040) — initialized
  Device 1: chans 0-5 → similar non-zero "OTHER" states — initialized
  Device 4: ALL channels → 0x00000000 — ZERO STATE (never initialized or fully cleared)
  Device 5: similar non-zero states — initialized

The "zero state" on device 4 is anomalous. Other devices show active ERISC router states.

## The 0x880030060 Address

From `device.cpp` comments:
> The CCL MUX kernel writes TERMINATED *before* close_finish() completes. If we proceed 
> directly to Phase 3's ConfigureDeviceWithProgram, the still-running BRISC executes 
> whatever instructions now reside in its overwritten L1, generating invalid NOC traffic — 
> including writes to ARC_RESET_SCRATCH_ADDR (0x880030060) — that corrupt ERISC or ARC state.

The NOC address 0x880030060 decodes to:
- High bits 0x8 = NOC destination in the Wormhole address space
- The ARC reset scratch register is at address 0x30060 in ARC space
- NOC address 0x880030060 is how Tensix cores address ARC's register space via NOC

## The quiesce_and_restart_fabric_workers Flow

```cpp
// Called for EACH device in quiesce_internal() after command queues drain

void Device::quiesce_and_restart_fabric_workers() {
    // Guard: FabricTensixConfig::DISABLED means no Tensix MUX — skip Phases 1/2/4
    // But Phase 2.5 (ERISC termination) still runs!
    
    // Phase 1: Send IMMEDIATELY_TERMINATE to Tensix MUX workers (skipped for DISABLED)
    // Phase 2: Poll MUX for TERMINATED, then assert_risc_reset (skipped for DISABLED)
    // Phase 2.5: Send TERMINATE signal to each active ERISC channel, wait for response
    //   - Reads ERISC router sync address
    //   - If status == 0 or TERMINATED → skip (already clean)
    //   - Otherwise → send TERMINATE signal, poll for TERMINATED
    // Phase 3: configure_fabric_cores() + re-launch ETH cores
    // Phase 4: Poll MUX for READY_FOR_TRAFFIC (skipped for DISABLED)
}
```

## The Fixture Configuration

```cpp
class MultiCQFabricMeshDevice2x4Fixture {
    MultiCQFabricMeshDevice2x4Fixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 4}, .num_cqs = 2}) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }
    void SetUp() override {
        setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", 0);  // ← THIS IS SET
        setenv("TT_METAL_FABRIC_HEALTH_PROBE", "1", 0);
        MeshDeviceFixtureBase::SetUp();
    }
};
```

**CRITICAL**: `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` is set in SetUp().  
This means `quiesce_and_restart_fabric_workers()` returns IMMEDIATELY (early-exit at top).
So ALL of Phases 1-4 are skipped for BOTH the pre-AllGather and post-AllGather quiesces.
The ERISC routers are NEVER terminated/restarted between operations.

## The FabricTensixConfig::DISABLED path

Even without `TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1`, the fixture uses:
```cpp
// .fabric_config is left at default (DISABLED) — so SetUp does NOT call 5-arg SetFabricConfig
// Only 1-arg SetFabricConfig(FABRIC_1D) is called → FabricTensixConfig stays DISABLED
```
This means: no Tensix MUX workers exist. Phases 1/2/4 are always skipped.
Only Phase 2.5 (ERISC termination) + Phase 3 (re-launch) would run — but are bypassed by env var.

## Source Files of Interest

1. `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tt_metal/impl/device/device.cpp`
   - Lines 420-800: `quiesce_and_restart_fabric_workers()` full implementation
   
2. `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tt_metal/distributed/mesh_device.cpp`
   - Lines 1433-1462: `quiesce_internal()` and `quiesce_devices()`

3. `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tt_metal/impl/event/dispatch.cpp`
   - Full `issue_record_event_commands()` — dispatches event to device's completion queue
   - For non-MMIO devices: goes via ETH fabric relay

4. `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tt_metal/distributed/fd_mesh_command_queue.cpp`
   - Lines 239, 740-757: `enqueue_record_event_helper` → `dispatch_thread_pool_->wait()`

5. `/workspace/group/worktrees/nsexton-0-racecondition-hunt/tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp`
   - Lines 83-95: The bug description comment
   - Lines ~280-310: "Enqueue dummy ops" section

## Key Question Neil Wants Answered

Neil asked for "in-depth and comprehensive investigation into access patterns and 
dispatch around the ARC firmware NOC access bug" — specifically:

1. Why does device 4 show all-zero ETH fabric router states post-quiesce?
2. Why do Tensix workers (on device 4) access 0x880030060 (ARC scratch) during "Enqueue dummy ops"?
3. What is the exact code path that causes the unsafe NOC write?
4. What fix would prevent this: quiesce → re-init before dummy ops, or something else?

Options Neil selected:
- Option 2: Investigate why device 4 ETH routers show zero post-quiesce
- Option 3: Check if quiesce → re-init → dummy ops avoids the hang

