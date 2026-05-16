<!-- SUMMARY: Investigation of FIX Z/GAP-A TT_THROW in read_completion_queue_event on N300 — Device 1 ETH channels 0+1 stuck at ROM postcode 0x49705530 with no recovery path for non-MMIO dead channels
KEYWORDS: dispatch-hang, FIX-Z, GAP-A, read_completion_queue_event, N300, ROM-postcode, 0x49705530, non-MMIO, dead-relay, probe_dead_channels, FIX-RR, FIX-EV, racecondition-hunt, fabric-degraded
SOURCE: log job_75290512431 + source analysis of nsexton-0-racecondition-hunt branch @ e04daeea
SCOPE: Root cause analysis of explicit throw (vs previous hang), recovery gap for non-MMIO dead channels, and comparison with previous prefetch timeout hang (job 75282755440)
USE WHEN: Investigating FIX Z/GAP-A throws, non-MMIO dead channel recovery, or comparing throw-vs-hang failure modes on N300 -->

# Dispatch Failure: FIX Z/GAP-A Throw on N300 (Dead Non-MMIO Channels)

**Job**: 75290512431
**Runner**: `tt-ubuntu-2204-n300-viommu-stable-dsrc9-runner-4btsl` (aus2 group)
**Branch**: `nsexton/0-racecondition-hunt` @ `e04daeea`
**Test**: `test_deepseek[True-Topology.Linear-1-3-Layout.TILE-...]` (deepseek_v3 all_gather)
**System**: N300 (2-chip Wormhole: chip 0 MMIO + chip 1 non-MMIO)

---

## 1. Timeline Reconstruction

```
T+0.0s   05:12:45.564  UMD: Opening user mode device driver
T+0.2s   05:12:45.626  Chip 0 harvesting: Tensix 0x240, DRAM 0x0, ETH 0x0
T+1.8s   05:12:46.375  Chip 1 harvesting: Tensix 0x201, DRAM 0x0, ETH 0x0
T+1.8s   05:12:46.375  Opening local chip 0 + remote chip 1, IOMMU enabled, KMD 2.7.0
T+2.2s   05:12:47.767  Sysmem mapped to IOVA 0x7f80000000, NOC 0x800000000
T+2.5s   05:12:48.041  DeepSeek detected N300, mesh shape [1,2]
T+2.5s   05:12:48.100  TopologyMapper: 2 chips, shape 2x1, FABRIC_1D with 1 CQ
T+2.6s   05:12:48.174  FIX TV: all 3 MMIO ETH channels confirmed heartbeat in 0ms
T+2.6s   05:12:48.180  [init] Starting fabric+dispatch FW for 2 devices (new session)

         ---- CORRUPT CHANNELS DETECTED AT STARTUP ----

T+2.6s   05:12:48.180  WARNING: Device 0 has 2 corrupt ETH channels BEFORE routing table write
T+2.6s   05:12:48.180  WARNING: Device 1 has 2 corrupt ETH channels BEFORE routing table write
T+2.6s   05:12:48.191  FabricBuilder: Dev 0 master_router_chan=8, Dev 1 master_router_chan=0

         ---- TERMINATE STALE ERISC ROUTERS ----

T+4.7s   05:12:50.345  FIX BT: Dev 0 chan=8 ROM postcode 0x49705530 -> probe_dead_channels
T+4.7s   05:12:50.345  FIX BT: Dev 0 chan=9 ROM postcode 0x49705530 -> probe_dead_channels
T+4.7s   05:12:50.345  Dev 0 summary: corrupt=2, probe_dead=2
T+4.7s   05:12:50.345  FIX BT: Dev 1 chan=0 ROM postcode 0x49705530 -> probe_dead_channels
T+4.7s   05:12:50.345  FIX BT: Dev 1 chan=1 ROM postcode 0x49705530 -> probe_dead_channels
T+4.7s   05:12:50.345  Dev 1 summary: corrupt=2, probe_dead=2

         ---- FIX E2: Device 1 marked dead-relay ----

T+4.7s   05:12:50.345  FIX E2: Dev 1 ETH relay compromised (probe_dead=2) -> dead-relay device
T+4.7s   05:12:50.345  configure_fabric: Dev 1 relay-broken flag reset by configure_fabric
T+4.7s   05:12:50.345  Dev 1: skipping assert_risc_reset for 2 pre-confirmed problematic channels
T+4.7s   05:12:50.345  Dev 1: degraded mode — 2 dead channels skipped, 0 recovered by FIX RR
T+4.7s   05:12:50.345  Dev 1: skipping WriteRuntimeArgsToDevice and ConfigureDeviceWithProgram
T+4.7s   05:12:50.345  Dev 1: skipping l1_barrier (dead ETH channels present)
T+4.7s   05:12:50.345  Fabric initialized on Device 1 (degraded)

         ---- FIX RR: Device 0 channels RECOVERED ----

T+4.7s   05:12:50.346  FIX RR: Dev 0 (MMIO) — PCIe-direct soft reset for 2 channels
T+4.7s   05:12:50.346  Dev 0 chan=8: FIX BH — booted to 0x49705530 after 0ms
T+4.7s   05:12:50.346  Dev 0 chan=8: FIX RR+BH — RECOVERED, L1 init will proceed
T+4.7s   05:12:50.346  Dev 0 chan=9: FIX BH — booted to 0x49705530 after 0ms
T+4.7s   05:12:50.346  Dev 0 chan=9: FIX RR+BH — RECOVERED, L1 init will proceed
T+4.7s   05:12:50.346  FIX RS: Dev 0 — 2 channels recovered, effective_pre_dead=0
T+4.7s   05:12:50.346  Fabric initialized on Device 0 (healthy)

         ---- FIX I: master router syn skip ----

T+4.7s   05:12:50.346  FIX I: Dev 0 MMIO master chan=8 connects to dead-relay Dev 1
T+4.7s   05:12:50.346  FIX ST: Dev 0 chan=8 recovered by FIX RR — sync will proceed
T+4.7s   05:12:50.346  Fabric config: FABRIC_1D
T+4.7s   05:12:50.345  Dispatch init SKIPPED for Dev 1 (dead ETH relay)

         ---- ROUTER SYNC SKIPS ----

T+7.5s   05:12:53.101  FIX G: Dev 1 dead-relay — skipping router sync
T+7.5s   05:12:53.101  FIX I: Dev 0 MMIO master connects to dead-relay peer — skip sync
T+7.5s   05:12:53.101  FIX AM: 1 dead-relay device — skip verify_all_fabric_channels_healthy()
T+7.5s   05:12:53.101  FIX QU: re-asserting fabric_relay_path_broken_ for Dev 1

         ---- TEST BEGINS: first dispatch succeeds (Device 0 only) ----

T+7.5s   05:12:53.105  enqueue_write_shards_nolock dispatch_thread_pool_->wait() OK
T+8.2s   05:12:53.766  enqueue_write_shard_to_sub_grid wait() OK
T+8.2s   05:12:53.766  enqueue_mesh_workload cq=0 lock acquired
T+8.2s   05:12:53.766  enqueue_mesh_workload cq=0 done
T+8.2s   05:12:53.766  enqueue_record_event_helper (notify_host=true)

         ---- THE THROW ----

T+8.2s   05:12:53.767  read_completion_queue_event cq=0 waiting for event=1
T+8.2s   05:12:53.767  cq=0 completion_queue_wait_front device=0 -> OK
T+8.2s   05:12:53.767  cq=0 calling completion_queue_wait_front device=1
T+8.2s   05:12:53.767  TT_THROW: FIX Z/GAP-A: Fabric degraded on non-MMIO device 1
                        relay_path_broken=true channels_not_ready=false stale_base_umd=false
T+8.2s   05:12:53.767  FIX EV: EventSynchronize skips spin-wait for device 1 (relay broken)

         ---- POST-THROW: test framework continues ----

T+8.8s   05:12:54.403  Second enqueue_write_shard_to_sub_grid + enqueue_mesh_workload
T+8.8s   05:12:54.403  Second enqueue_record_event_helper
T+163s   05:15:37.453  ##[error] The operation was canceled (CI timeout)
```

## 2. Root Cause Analysis

### 2.1 Why Are Device 1's ETH Channels Stuck at 0x49705530?

The postcode `0x49705530` is a ROM boot postcode — the ERISC core is stuck in the ROM boot phase and never progressed to base UMD firmware. Both Device 0 (channels 8+9) and Device 1 (channels 0+1) show the same postcode at startup.

**This is a "stale hardware state" condition from the previous test session.** The prior session either:
1. Terminated abruptly (CI timeout, crash, SIGKILL) before the ERISC firmware quiesce sequence ran
2. The quiesce sequence ran but UMD relay writes to non-MMIO channels were silently dropped or timed out
3. A hardware ETH link glitch during the prior session's cleanup left the ERISCs in a mid-boot state

The key evidence: the corrupt state is present BEFORE any routing table write in the current session. The current session inherits this hardware state from whatever ran before on the same runner.

### 2.2 Why Device 0 Recovers But Device 1 Does Not

The recovery path diverges based on the PCIe topology:

**Device 0 (MMIO):**
- FIX RR has PCIe-direct access: `assert_risc_reset_at_core()` + `deassert_risc_reset_at_core()` via PCIe bus
- This halts the ERISC, then restarts it — ERISC boots from ROM to base firmware
- FIX BH confirms boot completed (postcode transitions away from ROM phase)
- Channel is removed from dead_channels; firmware loads normally

**Device 1 (non-MMIO):**
- NO PCIe-direct path exists — all access goes through ETH relay via Device 0
- The ETH relay itself depends on the channels that are stuck (chicken-and-egg)
- `configure_fabric_cores()` SKIPS `assert_risc_reset_at_core()` entirely for non-MMIO dead channels
- Channels stay dead; device enters degraded mode with no firmware loaded

The core architectural gap: **FIX RR only works for MMIO devices.** Non-MMIO devices with dead ETH channels have no recovery path at the Metal software level.

### 2.3 The Cascade to FIX Z/GAP-A

1. FIX E2 marks Device 1 as dead-relay (`dead_relay_devices_`)
2. `set_fabric_relay_path_broken()` is called on Device 1
3. FIX QU re-asserts `fabric_relay_path_broken_=true` after `configure_fabric()` resets it
4. The test dispatches a 2-device mesh workload (DeepSeek all_gather on [1,2] mesh)
5. `read_completion_queue_event` iterates over both devices in the mesh
6. Device 0 completes fine (MMIO, PCIe-direct access to completion queue)
7. Device 1 hits the FIX Z/GAP-A guard: `is_fabric_relay_path_broken()` returns true
8. `TT_THROW` fires immediately, preventing a 5-second UMD relay timeout

## 3. Analysis: Is FIX Z/GAP-A's Throw Correct?

### 3.1 The throw IS correct behavior (fail-fast)

FIX Z/GAP-A is performing exactly its designed function: preventing a 5-second UMD relay timeout hang by detecting the known-broken state upfront. Without this guard, `completion_queue_wait_front()` would call a UMD relay read that traverses the dead ETH channel, waits 5 seconds for a response that never comes, and then throws an opaque exception.

The alternatives and why throw wins:

```
Option A: TT_THROW (current FIX Z/GAP-A) — CORRECT
  + Clear diagnostic: relay_path_broken=true, device=1
  + Fails in <1ms instead of 5s
  + Exception propagates to test framework
  - Test fails instead of being skipped

Option B: Skip device (like FIX EV does) — WRONG for CQ reads
  - completion_queue_wait_front reads an event from the HW CQ
  - Skipping means the event is never consumed — CQ read pointer is misaligned
  - All subsequent reads from this CQ would return wrong data
  - Silent data corruption is worse than a throw

Option C: Return sentinel/error event — POSSIBLE but fragile
  - Every caller of read_completion_queue_event would need to handle the sentinel
  - Risk of silent mishandling
```

**FIX EV's skip in EventSynchronize is different** — it skips a spin-wait, not a data read. EventSynchronize checks if `last_completed_event >= target_id`, which is a polling loop. Skipping it means "don't wait for completion on this device." That's safe because the exception from FIX Z/GAP-A already stored in `thread_exception_state_` will surface through the normal error path.

### 3.2 The problem is NOT the throw — it's the test framework's response

The log shows that after FIX Z/GAP-A throws:
- FIX EV correctly skips EventSynchronize for Device 1
- But then **the test continues dispatching more work** (second enqueue_write_shard_to_sub_grid at T+8.8s)
- The second enqueue_record_event_helper fires
- Then silence for 2m43s until CI cancels the job

The gap: after the first FIX Z/GAP-A throw, the test framework does not abort the entire test. It catches the exception, skips the EventSynchronize, and continues to the next iteration. The second iteration likely throws again or hangs.

## 4. The Non-MMIO Recovery Gap (Core Architectural Issue)

### 4.1 Current recovery paths

```
                          MMIO Device              Non-MMIO Device
                          ===========              ===============
PCIe soft reset           YES (FIX RR)             NO (no PCIe path)
                          assert/deassert via PCIe  assert/deassert needs ETH relay
                                                    which is the thing that's broken

MMIO relay-mediated       N/A (is MMIO)            NO (not implemented)
soft reset                                          Would need: MMIO Device 0's
                                                    healthy ETH to relay the
                                                    assert/deassert command to
                                                    Device 1's broken ERISC
                                                    But: broken ERISC IS the relay

UMD-level board reset     YES (via PCI reset)       POSSIBLE (via MMIO host)
                          Full chip reset            tt::umd::Cluster::pcie_hot_reset()
                                                    or kmd ioctl for non-MMIO
```

### 4.2 Could the MMIO relay drive a reset for non-MMIO dead channels?

The dead channels on Device 1 (chan 0+1) are the ETH links connecting to Device 0 (chan 8+9). These pairs form the relay path:

```
Device 0 chan 8  <---ETH link--->  Device 1 chan 0
Device 0 chan 9  <---ETH link--->  Device 1 chan 1
```

Device 0's channels 8+9 were RECOVERED by FIX RR (PCIe-direct reset). After recovery, they have healthy firmware. The ETH PHY link between the chips should still be functional — only the ERISC firmware on Device 1 is stuck.

**In theory**, Device 0 could relay an `assert_risc_reset_at_core` command to Device 1 through the now-healthy Device 0 channels 8+9. But:

1. **UMD relay reads/writes go through the ERISC firmware endpoint** on the destination chip. If Device 1's relay ERISC is the one stuck in ROM, the relay command cannot be serviced.
2. **`assert_risc_reset_at_core` is a register write to the ERISC's reset register** on the target chip. On MMIO devices this is a PCIe MMIO write. On non-MMIO devices, UMD wraps it as a relay command that goes: Host -> PCIe -> Device 0 ERISC -> ETH -> Device 1 ERISC -> register write. But Device 1's ERISC is the endpoint that would execute the register write, and it's stuck.
3. **A separate ERISC** (not the stuck one) on Device 1 could theoretically service the relay command. On N300 there are 16 ETH channels per device, and only channels 0+1 are stuck. But UMD routes relay commands through the master relay channel, which IS one of the stuck channels.

### 4.3 Viable recovery options

**Option 1: Cross-channel relay reset (not yet implemented)**
If Device 1 has other healthy ETH channels (not 0+1), UMD could route the reset command through one of those channels instead. This requires:
- UMD to support "alternative relay channel" selection when the primary relay is broken
- The fabric firmware on Device 0 to relay commands to Device 1 via a non-primary channel
- Device 1 to have at least one ERISC in base-UMD firmware (able to service relay reads)

On the N300, Device 1 only has 2 active channels (chan 0 and chan 1) connecting to Device 0. Both are stuck. There is no alternative path.

**Option 2: KMD-level non-MMIO reset via ERISC mailbox**
The Tenstorrent KMD could expose an ioctl to reset a non-MMIO chip's ERISC via the ethernet PHY layer (below the firmware). This would bypass the stuck firmware entirely. Not currently available.

**Option 3: Full board/link reset via KMD**
A heavier option: reset the entire non-MMIO chip via the PCIe bus of the MMIO chip (leveraging the PCI topology). Wormhole N300 has both chips on the same PCIe card; a PCI function-level reset of the non-MMIO chip's endpoint would restart all ERISCs. This is destructive (resets entire chip state) but guarantees recovery.

**Option 4: Accept degraded mode and skip at test level**
Instead of recovering the hardware, detect the degraded fabric early and skip tests that require the non-MMIO device. This is a software-only fix that doesn't solve the hardware problem but prevents CI noise.

## 5. Comparison with Previous Hang (Job 75282755440)

```
Aspect                  Job 75282755440 (previous)     Job 75290512431 (this)
======================  ============================   =========================
Failure mode            fetch_queue_reserve_back        FIX Z/GAP-A TT_THROW in
                        timeout (5s hang)               read_completion_queue_event
ERISCs stuck            Dev 0 cores 1-6, 9-6           Dev 0 chan 8+9, Dev 1 chan 0+1
                        (base firmware mid-execution)   (ROM postcode at startup)
Timing                  ~5s into workload execution    Before first CQ read
Devices opened          1 device (chip 0 only)          2 devices (mesh [1,2])
FIX TV at startup       Passed 0ms                     Passed 0ms
FIX BT/RR applied       No (pre-FIX BT code?)          Yes — Dev 0 recovered, Dev 1 not
Relay broken at boot    No (broke during execution)    Yes (Dev 1, at startup)
Dispatch topology       PREFETCH_HD -> DISPATCH_HD     2-device mesh, FD dispatch on Dev 0
                        -> DISPATCH_S (all on Dev 0)
```

### 5.1 Common root

Both failures share the same fundamental cause: **ETH cores stuck in a non-functional firmware state that prevents relay/fabric operations**. The difference is timing:

- **Previous (75282755440)**: ERISCs lost firmware MID-EXECUTION. The session started clean, but something (hypothesis: firmware crash, ETH link glitch, or context-switch failure) caused cores 1-6 and 9-6 to revert to base firmware. The prefetcher then hung trying to relay commands to unresponsive cores.

- **This (75290512431)**: ERISCs were in ROM boot state AT STARTUP — inherited from a prior session. FIX BT correctly detected them, FIX RR recovered Device 0 (MMIO), but Device 1 (non-MMIO) had no recovery path.

### 5.2 Key insight

The previous investigation's hypothesis (A) — "firmware never loaded on cross-chip cores" — is consistent with what we see here. In that case (1 device opened), the ETH cores bridging to chip 1 were possibly never initialized because only chip 0 was opened. In THIS case (2 devices opened), the ETH cores on Device 1 were detected as dead AT STARTUP and explicitly skipped.

The common thread: **non-MMIO ETH channels that get into a bad state have no recovery mechanism.** The only difference is whether the bad state was inherited (this job) or developed mid-run (previous job).

## 6. Proposed Remediation

### 6.1 IMMEDIATE: Test framework should detect degraded fabric and skip

**Files**:
- `tt_metal/distributed/fd_mesh_command_queue.cpp:942-1000` (read_completion_queue_event)
- Test conftest / fixture that opens the mesh

**Action**: Before dispatching 2-device mesh workloads, check `is_fabric_degraded()` or `is_fabric_relay_path_broken()` on all devices in the mesh. If any non-MMIO device has a broken relay, skip the test with `pytest.skip("Fabric degraded — non-MMIO device unreachable")` rather than letting it proceed to the FIX Z/GAP-A throw. This eliminates CI noise from inherited hardware state.

Alternatively, FIX Z/GAP-A could log a warning and return a "degraded" status instead of throwing — but this risks silent data corruption in the CQ read pipeline (see Section 3.1).

### 6.2 MEDIUM-TERM: Relay-mediated reset via alternative channels

**Files**:
- `tt_metal/fabric/fabric_init.cpp:152-293` (configure_fabric_cores FIX RR block)
- UMD: `tt_metal/third_party/umd/device/cluster.cpp` (relay routing)

**Action**: For non-MMIO devices where SOME channels are dead but the MMIO host's peer channels were recovered by FIX RR, attempt to relay `assert_risc_reset_at_core` through the healthy channels. On N300, this is only viable if there are more than 2 ETH links (N300 only has 2, so this doesn't help here, but helps on T3K with more links).

For N300 specifically: after Device 0 channels 8+9 are recovered by FIX RR, they are running healthy firmware and the ETH PHY link to Device 1 is intact. If UMD could send a raw register write (not relay-routed) through Device 0's ERISC to Device 1's stuck ERISC, it could assert the reset. This requires a UMD-level "direct ETH register write" primitive that does not depend on Device 1's ERISC firmware.

### 6.3 LONG-TERM: KMD-level non-MMIO chip reset

**Files**: KMD (tenstorrent kernel module), UMD (tt-umd)

**Action**: Expose a KMD ioctl to reset a non-MMIO chip via the MMIO chip's PCI path. On N300, both chips share a PCIe card — the MMIO chip's PCI function can trigger a function-level reset on the non-MMIO chip's function. This is the nuclear option but guarantees recovery from any firmware state.

### 6.4 DIAGNOSTIC: Enhance FIX Z/GAP-A throw to suggest skipping

**File**: `tt_metal/distributed/fd_mesh_command_queue.cpp:980-988`

**Action**: Change the TT_THROW message to include guidance: "Test should check is_fabric_degraded() before dispatching multi-device work." This helps developers hitting the throw in CI understand it's a hardware-state issue, not a code bug.

---

## Summary

The failure is caused by **Device 1 (non-MMIO) ETH channels 0+1 stuck at ROM postcode 0x49705530 at startup**, inherited from a prior session. Device 0's equivalent channels recovered via FIX RR (PCIe-direct reset), but Device 1 has no recovery path — all access requires the ETH relay that runs on the same channels that are stuck.

FIX Z/GAP-A's throw is **correct behavior**: it prevents a 5-second UMD relay timeout by failing fast with a clear diagnostic. The alternative (skip like FIX EV) is unsafe for CQ reads because it would misalign the completion queue read pointer.

The real gap is the lack of a non-MMIO dead channel recovery mechanism. On N300, Device 1 is permanently degraded once its 2 ETH channels are stuck — there is no software-level recovery path until either a UMD "direct ETH register write" or KMD chip-level reset is implemented. The practical fix is test-level detection of degraded fabric with pytest.skip.

---

## Deferred: Remediation 3 — KMD PCIe FLR for non-MMIO chip

**Status**: Deferred by Neil (2026-05-11). Implement after relay-assisted reset (Remediation 2) is validated.

**Description**: N300 has both chips on one PCIe card. Expose a KMD ioctl for PCIe function-level reset (FLR) of the non-MMIO chip via the MMIO chip's PCIe path. This is the nuclear option — guarantees recovery from any firmware state, including cases where relay-assisted reset fails (e.g., PHY link is completely dead).

**Why deferred**: Requires KMD changes + ioctl interface work. Remediation 2 (relay-assisted ERISC reset) is lower-risk and covers the common case.
