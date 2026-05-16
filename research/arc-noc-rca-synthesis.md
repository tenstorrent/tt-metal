<!--
SUMMARY: Complete root cause analysis of T3K AllGather→dummy-ops hang: TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART leaves ERISC routers dead on device 4, causing dispatch commands to stall on dead fabric endpoints
KEYWORDS: ARC, 0x880030060, NOC, device 4, non-MMIO, ERISC, quiesce, fabric restart, TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART, AllGather, T3K hang
SOURCE: Opus swarm synthesis (researchers A, B, C)
SCOPE: Full RCA + Options 2 & 3 analysis + specific fix recommendations
USE WHEN: Investigating T3K AllGather/dummy-ops hang on nsexton/0-racecondition-hunt branch
-->

# ARC NOC Bug - Synthesis & Root Cause Analysis

## TL;DR

`TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1` (set in the test fixture's SetUp) causes
quiesce_and_restart_fabric_workers() to be a no-op. After AllGather's teardown terminates
ERISC fabric routers on device 4, they are never restarted. "Enqueue dummy ops" then
tries to dispatch through dead fabric endpoints → unbounded spin → 5s timeout.

**The fix: remove TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1 and let Phase 2.5+3 run.**
This is now safe because Phase 2 was fixed on this branch to halt BRISC before L1 overwrite.

---

## The Full Causal Chain

```
1. Fixture SetUp() sets TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1

2. Initial fabric init: All devices (0,1,4,5) get ERISC routers → READY_FOR_TRAFFIC

3. pre-AllGather quiesce_devices() → quiesce_and_restart_fabric_workers() → EARLY RETURN
   (env var active — Phases 1/2/2.5/3/4 all skipped)

4. ttnn::all_gather() runs — uses ERISC fabric routers on all devices
   AllGather teardown sends TERMINATE to ERISC channels
   Device 4's ERISC routers terminate, L1 sync address → 0x00000000 (zeroed by firmware)
   Devices 0/1/5: L1 overwritten with CCL data remnants (non-zero "OTHER" values)

5. post-AllGather quiesce_devices() → EARLY RETURN again (same env var)
   No restart → device 4 ERISC routers remain dead

6. Health probe (post-allgather-post-quiesce):
   Device 4: ALL channels → 0x00000000  ← DEAD ERISC ROUTERS
   Others: non-zero "OTHER" states (also corrupt, but different remnants)

7. "Enqueue dummy ops" → ttnn::write_buffer + dispatch_ops_to_device for each device

8. For device 4 (non-MMIO, DISPATCH_D variant with FABRIC_RELAY=1):
   - Commands enqueued via dispatch relay chain
   - Relay chain calls wait_for_fabric_endpoint_ready(mux_x, mux_y, ...) — unbounded spin
   - Dead ERISC routers → fabric endpoint never becomes ready
   - dispatch_thread_pool_->wait() blocks

9. TT_METAL_OPERATION_TIMEOUT_SECONDS=5 fires → TT_THROW: TIMEOUT: device timeout
```

---

## What Is 0x880030060?

NOC address decoding (Wormhole):
```
0x880030060:
  Y coordinate: 0
  X coordinate: 0
  Local addr:   0x880030060 (bit 35 SET → NOC-to-AXI bridge activated)
  → Targets ARC firmware at AXI offset 0x80030060 (ARC_RESET_SCRATCH0)
```

NOC node (0,0) = ARC/router subsystem (NOT a Tensix core).
Writing garbage here corrupts ARC firmware postcodes/state → device becomes unresponsive.

The write is NOT intentional. It comes from corrupted firmware executing garbled instructions
from an overwritten L1 (the close_finish/TERMINATED race for Tensix MUX, OR
stale ERISC firmware running on zeroed L1 from prior iteration's uncleaned teardown).

**For THIS specific test failure, the 0x880030060 write comes from dispatching to dead
ERISC fabric endpoints → something on device 4 executing corrupted state, not from
the quiesce BRISC race (which requires Tensix MUX / FabricTensixConfig != DISABLED).**

---

## Why Device 4 Specifically Shows All Zeros

- EDMStatus zero (0x00000000) is NOT a valid state (all 15 enum values are magic numbers like 0xA0B0C0D0)
- After AllGather teardown, device 4's ERISC router firmware completes its cleanup and zeroes its sync address
- Devices 0/1/5 get non-zero "OTHER" values because CCL data payloads happen to overwrite their L1 with non-zero remnants
- Both are invalid — but the code treats zero as "already clean" (Phase 2.5 line 637, terminate_stale_erisc_routers line 449)

**Key: two independent communication layers to non-MMIO chips:**
1. UMD dispatch tunnel (dedicated ERISC cores for PCIe→remote forwarding) — ALWAYS operational
2. tt-metal fabric ERISC routers (software-defined mesh fabric for CCL ops + fabric-relay dispatch) — DEAD

Dispatch commands for dummy ops use FABRIC_RELAY path (layer 2) for non-MMIO DISPATCH_D →
they go through the dead layer-2 ERISC routers → hang.

---

## Option 2: Why Device 4 Shows Zeros

**Not a code bug — it's AllGather CCL teardown behavior.**

AllGather's teardown sends TERMINATE to ERISC fabric routers. When ERISC firmware
writes TERMINATED, it zeroes the router_sync_address as part of clean shutdown.
Device 4 gets zeros because its ERISCs complete teardown fully. Other devices get
non-zero "OTHER" values because CCL data payloads partially overwrote their L1
before teardown completed.

**No separate fix needed.** The zero state is expected post-teardown. The problem
is that we never restart the routers afterward (Option 3 addresses this).

---

## Option 3: Remove TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART=1

**Is it now safe?** YES, with the branch's Phase 2 fix:

Phase 2 now calls `assert_risc_reset_at_core()` on the Tensix MUX BRISC AFTER seeing
TERMINATED and BEFORE Phase 3's L1 overwrite. This prevents the close_finish() race.

**What Phase 2.5+3 would do (FabricTensixConfig::DISABLED, no Tensix MUX):**

Phase 2.5 for each device:
- Device 4: all channels read 0 → skipped ("already clean" at device.cpp:637)
- Devices 0/1/5: "OTHER" non-zero values → send TERMINATE, poll 500ms each
  - These ERISCs are likely dead (just have corrupt L1), so polls time out
  - Warning logged, continue (acceptable — configure_fabric_cores still runs)
  - Cost: ~500ms × N_active_channels × 3 devices ≈ 3-9s per quiesce call

Phase 3 for each device:
- configure_fabric_cores(): zero all router L1 addresses
- WriteRuntimeArgsToDevice() + ConfigureDeviceWithProgram(): fresh firmware
- l1_barrier() + write_launch_msg_to_core(): start ERISC routers

After Phase 3:
- ALL devices (including device 4) have fresh, running ERISC routers
- "Enqueue dummy ops" can dispatch through healthy fabric endpoints → no hang

**Risks:**
1. +3-9s per quiesce call (Phase 2.5 timeouts) — functionally fine, just slower
2. Phase 2.5 may not terminate devices 0/1/5 ERISCs before Phase 3 overwrites their L1
   - They're likely already dead; worst case: brief garbled instruction execution
   - On WH can't hard-reset ERISCs (tears down ETH PHY) — accepted risk
3. No Phase 4 wait (FabricTensixConfig::DISABLED) → next op might start before ERISCs ready
   - write_buffer uses UMD dispatch tunnel, not fabric routing → probably safe

**Verdict: Option 3 (remove env var) should fix the hang.**

---

## Recommended Code Change

In `tests/ttnn/unit_tests/gtests/multi_thread/test_ccl_multi_cq_multi_device.cpp`:

```cpp
// SetUp() currently:
void SetUp() override {
    ...
    setenv("TT_METAL_DISABLE_QUIESCE_FABRIC_RESTART", "1", /*overwrite=*/0);  // REMOVE THIS LINE
    setenv("TT_METAL_FABRIC_HEALTH_PROBE", "1", /*overwrite=*/0);
    MeshDeviceFixtureBase::SetUp();
}
```

Also update the comment at lines 83-95 that says the fix is pending.

**Additionally**, to handle Phase 2.5 timeout delays, we could reduce the ERISC timeout
from 500ms to 100ms for the quiesce path (device.cpp:628, constexpr erisc_timeout_ms).
This would reduce the total delay from 3-9s to under 2s.

---

## Related Issues After This Fix

Once Option 3 is applied, watch for:
1. Phase 2.5 timeouts showing up in logs — expected, not errors
2. If Phase 3's L1 overwrite causes a brief ERISC garbled-instruction window, we might
   still see sporadic 0x880030060 writes, but much less frequently
3. The legacy MUX close_finish/TERMINATED race (Path A in tt_fabric_mux.cpp:295-303)
   still exists but only affects FabricTensixConfig != DISABLED configurations

