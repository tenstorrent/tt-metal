<!-- SUMMARY: Investigation of fetch_queue_reserve_back timeout hang on N300 during SDXL op perf test — ERISC cores stuck in base firmware while prefetcher blocks on full command queue
KEYWORDS: dispatch-hang, fetch_queue_reserve_back, N300, ERISC, base-firmware, process_relay_ringbuffer_cmd, FIX-AE, relay-broken, cq_prefetch, cq_dispatch_subordinate, racecondition-hunt
SOURCE: log job_75282755440 + source analysis of nsexton-0-racecondition-hunt branch
SCOPE: Root cause analysis of dispatch hang on N300 with chip 0 (MMIO) + chip 1 (non-MMIO), covering timeline, failure mode, code paths, and remediation
USE WHEN: Investigating fetch_queue_reserve_back timeouts, ERISC base firmware during execution, or relay broken conditions on N300 -->

# Dispatch Hang: fetch_queue_reserve_back Timeout on N300

**Job**: 75282755440
**Runner**: `tt-ubuntu-2204-n300-viommu-stable-dsrc9-runner-x574r` (aus2 group)
**Branch**: `nsexton/0-racecondition-hunt` @ `e04daeea`
**Test**: `test_sdxl_op_unit_test_perf.py::test_dram_group_norm_welford_reciprocal_vae`
**System**: N300 (2-chip Wormhole: chip 0 MMIO + chip 1 non-MMIO)

---

## 1. Timeline Reconstruction

```
T+0.0s   03:34:52.452  Fabric topology mapping: 1 logical mesh -> 1 physical mesh
T+1.2s   03:34:53.687  FIX TV: all 3 MMIO ETH channels confirmed base firmware heartbeat in 0ms
T+1.2s   03:34:53.689  Submesh 1 instantiated with 1 device
T+1.2s   03:34:53.702  [initialize_fabric_and_dispatch_fw] Starting fabric + dispatch FW init for 1 device
T+3.6s   03:34:56.076  First enqueue_write_shards_nolock dispatches begin (writes succeed)
T+3.6s   03:34:56.926  enqueue_mesh_workload cq=0 completes, record_event_helper queued
T+9.5s   03:35:02.026  TIMEOUT DETECTED (metal_context.cpp:736)
T+9.5s   03:35:02.027  Executing hang_report.py + tt-triage.py
T+9.5s   03:35:02.041  fetch_queue_reserve_back timeout: cq_id=0 in_flight=128
                        ptrs=0x0000e430 fences=0x0000e3b4 base=0x0000e380 limit=0x0000e480 N=128
T+18.5s  03:35:11.052  tt-triage callstack dump shows:
                          - cq_prefetch (7-0): stuck in process_relay_ringbuffer_cmd -> careful_copy_from_l1_to_local_cache
                          - cq_dispatch (3-0): stuck in early_exit() (cq_helpers.hpp:14)
                          - cq_dispatch_subordinate (8-6): stuck in cb_acquire_pages_dispatch_s -> update_worker_completion_count
                          - ERISC 1-6, 9-6: PC in base ERISC firmware (not in any loaded ELF)
T+19.6s  03:35:12.085  TT_THROW: device timeout, device unrecoverable
T+19.6s  03:35:12.085  TT_THROW: TIMEOUT in fetch queue wait
         03:35:17.263  2nd fetch_queue_reserve_back timeout (during synchronize_device in close)
         03:35:22.376  3rd timeout (during close_impl -> terminate_command_queues)
         03:35:27.476  4th timeout (during ScopedDevices destructor)
T+37.4s  03:35:29.871  FIX AE: Marking relay broken for chip 1 (UMD cluster.cpp:650)
T+37.4s  03:35:29.871  Closing devices in cluster
```

## 2. Root Cause Analysis

### 2.1 The Hang Mechanism

The prefetch queue has **128 entries (N=128)** and is completely full (`in_flight=128`). The host-side `fetch_queue_reserve_back()` cannot push new commands because the firmware prefetcher has not consumed any entries. The pointer and fence values are frozen:

```
ptrs=0x0000e430  fences=0x0000e3b4  base=0x0000e380  limit=0x0000e480
```

The fence (`0xe3b4`) is behind the write pointer (`0xe430`), confirming firmware stopped advancing. The host spins for 5 seconds (`TT_METAL_OPERATION_TIMEOUT_SECONDS=5`), then fires the timeout.

### 2.2 Why the Prefetcher Is Stuck

The tt-triage callstack shows `cq_prefetch` (device 0, core 7-0, ERISC) frozen at:

```
#0 careful_copy_from_l1_to_local_cache<6, 113> () at cq_common.hpp:625
#1 process_relay_ringbuffer_cmd<false> () at cq_prefetch.cpp:1836
#2 process_cmd<false, false> () at cq_prefetch.cpp:2128
#3 kernel_main_hd () at cq_prefetch.cpp:2711
```

`process_relay_ringbuffer_cmd` reads sub-command data from L1 to a local cache buffer, then dispatches relay ring-buffer sub-commands to downstream cores. The function at line 1836 is performing a `careful_copy_from_l1_to_local_cache` — this is a NOC read that copies data from L1 memory. If the source L1 region is never populated (because upstream data never arrived via fabric relay), this read would stall indefinitely or return stale data causing the prefetcher to loop.

### 2.3 Why ERISCs 1-6 and 9-6 Are in Base Firmware

The critical finding: **two active ETH cores (1-6/e0,9 and 9-6/e0,8) on device 0 have PCs that do not match any loaded ELF**. The triage tool reports:

> "PC was not in range of any provided ELF files. Probably context switch occurred and PC is contained in base ERISC firmware."

These are **active ethernet cores** (shown in the watcher ring buffer as `active_eth`). On the N300, these cores bridge chip 0 (MMIO) to chip 1 (non-MMIO) via the ethernet link. They should be running either:
- Fabric router firmware (if fabric is active), or
- Dispatch relay firmware (if used for command relay)

Instead, they are in **base ERISC firmware** — the idle firmware loaded by ROM that simply heartbeats and waits for dispatch or fabric firmware to context-switch in. This means:

1. Fabric firmware was never properly loaded onto these cores, OR
2. Fabric firmware was loaded but crashed/hung and fell back to base firmware

Since FIX TV passed in 0ms at startup (all 3 MMIO ETH channels confirmed heartbeat), the initial boot was clean. The fabric firmware initialization for 1 device succeeded. But the active ETH cores for the N300 cross-chip link (1-6 and 9-6) appear to have either:
- Never received their fabric FW context switch, or
- Had their FW die/reset during the first few seconds of workload execution

### 2.4 Why FIX AE Fires

`FIX AE: Marking relay broken for chip 1` fires at process shutdown (cluster.cpp:650). This is triggered in `Cluster::mark_relay_broken()` when the UMD detects that the relay path to chip 1 is non-functional. Since the ERISCs bridging to chip 1 are stuck in base firmware (not relay firmware), any attempt to communicate with chip 1 via the relay would fail or timeout, triggering the relay-broken marking.

The `mark_relay_broken` call happens during `close_devices` / UMD teardown — it is a **consequence** of the hang, not a cause.

### 2.5 The Dispatch Topology

```
Dev 0, 8-6 (e0,10): cq_dispatch_subordinate (DISPATCH_S), cq_id=0, servicing device 0
Dev 0, 3-0 (e0,5):  cq_dispatch (DISPATCH_HD), cq_id=0, servicing device 0
Dev 0, 7-0 (e0,4):  cq_prefetch (PREFETCH_HD), cq_id=0, servicing device 0
```

All dispatch kernels run on device 0 ERISCs, servicing device 0. Only 1 device was opened in this session. The dispatch pipeline is:

```
Host -> fetch_queue -> cq_prefetch (7-0) -> cq_dispatch (3-0) -> cq_dispatch_subordinate (8-6) -> workers
```

The `cq_dispatch_subordinate` is blocked in `cb_acquire_pages_dispatch_s` waiting for pages from `cq_dispatch`. `cq_dispatch` is in `early_exit()` — it has detected a termination condition or is waiting. `cq_prefetch` is stuck trying to relay ring-buffer data.

## 3. Failure Mode Classification

### Not the known N300 simultaneous-boot race

The known N300 simultaneous-boot issue involves both chips trying to boot fabric firmware concurrently, causing a deadlock in ETH link training. Key differences:

| Aspect | Known simultaneous-boot race | This failure |
|--------|------|------|
| FIX TV at startup | Would fail or show delays | **Passed in 0ms** |
| Number of devices opened | Both chips | **Only chip 0 (1 device)** |
| Timing | During device open | **~5s into workload execution** |
| Relay state at boot | Broken from start | **Breaks during execution** |

### This is a **mid-execution fabric firmware crash/reset** on the active ETH cores

The failure signature is:
1. Clean startup (FIX TV passes, fabric init succeeds for 1 device)
2. Initial dispatch commands succeed (writes go through)
3. ~5 seconds into execution, the prefetcher gets stuck processing a relay ring-buffer command
4. Active ETH cores 1-6 and 9-6 are found in base firmware, not fabric firmware
5. The prefetch queue fills up because firmware stops consuming entries
6. Host times out waiting for queue space

**Hypothesis**: The fabric firmware on cores 1-6 and 9-6 either:
- (A) Was never loaded because only 1 device was opened and these cores were not part of the dispatch topology, but the prefetcher issued relay commands targeting them anyway, OR
- (B) Crashed or was reset by a hardware event (ETH link glitch, watchdog, thermal) causing a fallback to base firmware

Hypothesis (A) is the more likely root cause — a **command routing mismatch** where the prefetcher has relay ring-buffer commands targeting fabric cores that were never initialized with dispatch/fabric firmware for this session (since only chip 0 was opened).

## 4. Code Path Analysis

### 4.1 fetch_queue_reserve_back (Host Side)

File: `tt_metal/impl/dispatch/system_memory_manager.cpp:675-839`

The host writes dispatch commands to a circular prefetch queue. When all 128 slots are full (`in_flight >= N`), it reads the hardware fence pointer to check if firmware consumed entries. The firmware writes a fence address after consuming each slot and clears the slot to 0.

The timeout loop in `loop_and_wait_with_timeout` checks for dispatch progress (XOR'd with fabric ERISC progress via `FIX LT9-PROGRESS-B2`). After 5 seconds with no progress, it fires the timeout callback.

### 4.2 cq_prefetch (Firmware Side)

File: `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:1807-1841`

`process_relay_ringbuffer_cmd` processes `CQPrefetchCmd::relay_ringbuffer` commands:
1. Reads `count` and `stride` from the command header
2. Copies sub-command data from L1 to a local cache via `careful_copy_from_l1_to_local_cache`
3. Calls `process_relay_ringbuffer_sub_cmds` to execute the relay sub-commands

The prefetcher is stuck at step 2 (line 1836), performing the L1-to-cache copy. This could stall if the L1 data at `data_ptr` is not valid/available, or if a NOC operation is blocked waiting for a response from a core that is unresponsive (the base-firmware ERISCs).

### 4.3 cq_dispatch_subordinate (Firmware Side)

File: `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp:180-195`

`cb_acquire_pages_dispatch_s` spins waiting for pages from cq_dispatch:
```cpp
while (wrap_gt(num_pages_acquired + n, *sem_addr)) {
    invalidate_l1_cache();
    update_worker_completion_count_on_dispatch_d();
    IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
}
```

The subordinate is starved because `cq_dispatch` is stuck in `early_exit()`, which itself is waiting because the prefetcher cannot feed it new commands.

### 4.4 FIX AE Trigger Path

File: `tt_metal/third_party/umd/device/cluster.cpp:649-651`

```cpp
void Cluster::mark_relay_broken(const ChipId chip_id) {
    log_info(LogUMD, "FIX AE: Marking relay broken for chip {}", chip_id);
    get_chip(chip_id)->mark_relay_broken();
}
```

Called during UMD shutdown when the driver detects non-MMIO chip 1's relay is non-functional.

## 5. Proposed Remediation

### 5.1 Immediate Fix: Validate relay targets before issuing relay commands

**File**: `tt_metal/impl/dispatch/hardware_command_queue.cpp`

Before issuing relay ring-buffer commands that target fabric cores, verify that the target cores have been initialized with appropriate firmware for this session. If only 1 device is opened, relay commands targeting cross-chip ETH cores should be suppressed or routed differently.

### 5.2 Guard: Check active ETH core firmware state before dispatch

**File**: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`

Add a post-init verification that all active ETH cores expected to participate in the dispatch/fabric pipeline are actually running the correct firmware (not stuck in base firmware). If any are in base firmware after initialization, fail fast with a clear error rather than allowing dispatch commands to target them.

### 5.3 Defensive: Timeout on NOC reads in process_relay_ringbuffer_cmd

**File**: `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` (line ~1836)

The `careful_copy_from_l1_to_local_cache` call has no timeout or abort mechanism. If the source L1 data is never available (because the source core is in base firmware), the prefetcher hangs indefinitely. Adding a watchdog or heartbeat check inside the copy loop would allow the firmware to detect and report the stall rather than hanging silently.

### 5.4 Diagnostic: Log active ETH core states at timeout

**File**: `tt_metal/impl/dispatch/system_memory_manager.cpp` (line ~809, the `fetch_on_timeout` lambda)

When `fetch_queue_reserve_back` times out, the timeout handler should dump the firmware state (PC, postcode) of all active ETH cores, not just the dispatch cores. This would immediately identify the "ERISC in base firmware" condition at timeout time rather than requiring a separate tt-triage analysis.

### 5.5 Root Cause Investigation: Why did ETH cores 1-6 and 9-6 lose their firmware?

The deepest question is: **what caused these cores to be in base firmware during execution?** The possibilities are:

1. **They were never loaded**: The session opened only 1 device — if the fabric init for a single-device N300 session does not load firmware onto the cross-chip ETH cores, but the dispatch topology still generates relay commands targeting those cores, there is an initialization-to-dispatch mismatch.

2. **Firmware crashed**: A hardware event (link flap, thermal, watchdog timeout) on the ETH link between chips caused the ERISCs to reset back to base firmware.

3. **Context switch did not complete**: The cooperative ERISC model requires the base firmware to context-switch to the dispatch/fabric kernel. If the context switch handshake fails (e.g., mailbox corruption, interrupted by link event), the core stays in base firmware.

**Recommended next step**: Add logging at fabric firmware load time that records exactly which ETH cores receive firmware and which do not. Then compare against the cores that relay commands target. This will definitively distinguish cause (1) from causes (2) and (3).

---

## Summary

The hang is caused by a **stalled prefetcher** (`cq_prefetch` on core 7-0) that is stuck in `process_relay_ringbuffer_cmd` trying to copy data from L1 while two active ETH cores (1-6 and 9-6) are in base firmware instead of fabric/dispatch firmware. The prefetch queue fills to capacity (128/128), the host cannot push new commands, and the 5-second timeout fires. This is **not** the known N300 simultaneous-boot race — it is a mid-execution failure where fabric ETH cores lose their firmware context. FIX AE (relay broken for chip 1) is a consequence, not a cause.
