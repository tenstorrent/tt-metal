<!--
SUMMARY: T3K ERISC handshake race/deadlock: root cause analysis, attempted fixes, and ETH DMA pre-ping restructure design
KEYWORDS: T3K, ERISC, ETH DMA, handshake, race condition, deadlock, pre-ping, fabric init, MMIO, non-MMIO
SOURCE: Code analysis of tt_metal/fabric/ + erisc_datamover_builder.cpp + fabric_firmware_initializer.cpp
SCOPE: Covers the simultaneous-handshake deadlock mechanism, FIX CV/CY failure analysis, and ETH DMA pre-ping design spec
USE WHEN: Implementing or reviewing changes to T3K fabric ERISC handshake init sequence
-->

# T3K ERISC Handshake Race Condition -- Technical Report

## 1. Executive Summary

T3K systems with 8 chips experience a deadlock during fabric initialization. The root cause is a timing race between paired ERISC cores on MMIO (chips 0-3) and non-MMIO (chips 4-7) devices. MMIO ERISCs, which are always the handshake sender (lower tie-break ID), enter the sender loop and begin transmitting nonce packets via raw ETH DMA before their non-MMIO peers have entered the receiver loop. The nonce packets are silently dropped, and both sides spin forever.

Three fixes have been attempted. FIX CU added a HANDSHAKE_READY status signal (necessary infrastructure, still in place). FIX CV attempted host-side staging with MMIO-first launch ordering and PCIe polling of non-MMIO readiness, but failed because ERISCs complete all init states in under 1ms while a single PCIe ReadFromDeviceL1 round trip takes approximately 5ms. FIX CY (current HEAD, commit 6cda4889fe7) introduced a host-written gate: MMIO ERISCs spin on HOST_GATE_OPEN, which the host writes only after polling confirms all non-MMIO ERISCs have reached HANDSHAKE_READY. This works but places the host on the critical path of every fabric init, adding latency and fragility.

The proposed restructure eliminates host involvement entirely. Non-MMIO ERISCs (receivers) send a "peer_ready" pre-ping over raw ETH DMA to their MMIO peer immediately after reaching HANDSHAKE_READY. MMIO ERISCs (senders) spin waiting for this pre-ping in a dedicated L1 slot instead of waiting for HOST_GATE_OPEN. Once received, the MMIO ERISC enters the sender handshake loop with certainty that its peer is listening. FIX CY and the host gate machinery (~80 LOC across 8 files) can then be removed.


## 2. Background: T3K Topology and ERISC Handshake Protocol

### Topology

T3K has 8 Wormhole chips. Chips 0-3 are MMIO-capable (direct PCIe host access). Chips 4-7 are non-MMIO (host cannot write their L1 once ERISC firmware launches on the MMIO device). Each chip has up to 16 ethernet channels, each running its own ERISC firmware instance. Paired ERISC cores across an ETH link perform a point-to-point handshake.

### Role Assignment

The `is_handshake_sender` compile-time arg (fabric_erisc_router_ct_args.hpp:129) is determined by tie-break in erisc_datamover_builder.cpp:933: the side with the lower ID is always the sender. Since MMIO devices have IDs 0-3 and non-MMIO have IDs 4-7, MMIO is ALWAYS the sender, non-MMIO ALWAYS the receiver.

### handshake_info_t Layout (edm_handshake.hpp:49-56)

The struct is 32 bytes, 16-byte aligned:

    Bytes 0-3:   local_value        (where remote nonce lands via ETH DMA)
    Bytes 4-5:   neighbor_mesh_id
    Byte  6:     neighbor_device_id
    Byte  7:     padding
    Bytes 8-15:  padding[2]
    Bytes 16-31: scratch[4]         (source of outgoing nonce + identity)

### Handshake Protocol

1. Both sides call `init_handshake_info()` which zeroes `local_value` and writes `session_nonce` + identity into `scratch`.

2. Sender loop (fabric_router_eth_handshake.hpp:24-78): repeatedly calls `eth_send_packet(0, scratch_addr, local_val_addr, 1)` -- sends 16B from local scratch to remote local_value. Polls own local_value for the peer's nonce. Exits when nonce matches session_nonce. Sends one final packet post-loop (FIX HS2) to handle the simultaneous-sender edge case.

3. Receiver loop (fabric_router_eth_handshake.hpp:83-131): does NOT send in the loop. Polls local_value with cache invalidation. After the nonce arrives, sends one reply packet back.

### ETH DMA Mechanism

`internal_::eth_send_packet()` (tunneling.h:82) writes directly to ETH MAC TX queue registers. Parameters are in 16-byte words (byte_addr = word_addr * 16). It is a raw L1-to-L1 DMA with no software protocol layer. The function spins on `eth_txq_is_busy()` before writing. The "unsafe" variant skips the busy-wait.

### Pre-Handshake Sequence (fabric_erisc_router.cpp:3647-3685)

1. `wait_for_other_local_erisc()` barrier (line 3658) -- syncs two RISC cores within the same ETH channel using stream scratch registers. LOCAL only, not cross-link.
2. Write HANDSHAKE_READY to `edm_status_ptr` (line 3663) -- FIX CU
3. If `host_gate_enabled` (MMIO cores only): spin on HOST_GATE_OPEN (lines 3673-3684) -- FIX CY


## 3. Root Cause: The Simultaneous-Handshake Deadlock

The deadlock arises from a fundamental timing asymmetry:

1. Host launches ERISC firmware on MMIO devices first (by necessity -- it must program them to relay commands to non-MMIO devices).

2. MMIO ERISC boots, completes channel/stream init, writes HANDSHAKE_READY, and enters the sender loop. The sender loop IMMEDIATELY begins calling `eth_send_packet` on every iteration -- it fires the nonce packet without waiting for any signal from the peer.

3. Non-MMIO ERISC has not yet finished init (or hasn't even been launched yet). Its `local_value` is uninitialized or zeroed. The nonce packet lands in L1 at `local_value`, but the ERISC is not yet in the receiver loop to observe it.

4. Non-MMIO ERISC eventually enters `init_handshake_info()`, which ZEROES `local_value` (edm_handshake.hpp:85). This erases ALL prior nonce writes from the MMIO sender.

5. Non-MMIO ERISC enters the receiver loop. But the MMIO sender has been iterating the entire time. If the sender happened to be in a `run_routing()` context switch or a non-send iteration when the receiver cleared local_value, the receiver sees zero and waits.

6. The sender continues sending, but now there is a subtle race: on WH, the sender sends on every non-context-switch iteration. If both sides' `init_handshake_info()` calls overlap with the other side's sends, both can zero the other's value simultaneously. This is the "simultaneous-sender race" that FIX HS2 addresses with a post-loop final send -- but it only helps when both sides are already IN their respective loops.

The core problem is that ERISCs complete all states (STARTED -> HANDSHAKE_READY -> enter loop) in under 1 millisecond. Host-side PCIe polling has a round-trip time of approximately 5ms. The ERISCs outrun ANY host-side orchestration that depends on reading L1 state.


## 4. Attempted Fixes

### FIX CU -- HANDSHAKE_READY Signal

Location: fabric_erisc_router.cpp:3660-3663

Each ERISC writes `EDMStatus::HANDSHAKE_READY` (0xA0B0C0D1) to its status word after completing channel/stream init and passing the local ERISC barrier. This gives the host (and diagnostic tools) a observable milestone: the ERISC's data structures are initialized and it is about to enter the handshake loop.

This fix is necessary infrastructure -- it provides the signal that FIX CV and FIX CY depend on. It remains in place regardless of the pre-ping restructure. However, it is insufficient alone because writing HANDSHAKE_READY does not prevent the MMIO ERISC from racing ahead into the sender loop before the non-MMIO peer reaches its receiver loop.

### FIX CV -- MMIO-First Host Staging

Location: fabric_firmware_initializer.cpp:3526-3538

Pass A waits for all MMIO ERISCs to reach HANDSHAKE_READY. Pass B waits for all non-MMIO ERISCs to reach HANDSHAKE_READY. The intent: by the time MMIO ERISCs enter the handshake, non-MMIO ERISCs are guaranteed ready.

Why it failed: Pass B uses `wait_for_eth_cores_launched()` which polls via PCIe ReadFromDeviceL1. A single read round trip is approximately 5ms. ERISCs reach HANDSHAKE_READY and enter the handshake loop in under 1ms. By the time the host confirms non-MMIO readiness via PCIe, the non-MMIO ERISC may have already entered the receiver loop AND the MMIO ERISC may have already been sending for milliseconds -- but Pass B cannot RELEASE the MMIO ERISC fast enough because there is no gate. Without a gate, Pass A/B ordering is merely observational, not causal.

### FIX CY -- Host-Written Gate (Current HEAD)

Location: fabric_erisc_router.cpp:3666-3685 (firmware), device.cpp:2878-2925 (host write), fabric_firmware_initializer.cpp:3540-3553 (Phase C orchestration)

The `host_gate_enabled` CT arg (set to 1 for MMIO ERISCs, 0 for non-MMIO) causes MMIO ERISCs to spin on `HOST_GATE_OPEN` (0xA0B0C0D2) in their status word. The host writes this value only after Pass B confirms all non-MMIO ERISCs have reached HANDSHAKE_READY.

Why it partially works: This IS a causal gate -- MMIO ERISCs cannot enter the handshake until the host explicitly releases them. Non-MMIO ERISCs skip the gate (host cannot reach their L1) and proceed directly to the receiver loop after HANDSHAKE_READY.

Why it is suboptimal:
- Host is on the critical path of every fabric init. PCIe latency adds milliseconds.
- The host must iterate all MMIO devices and write HOST_GATE_OPEN to every active ETH core.
- If the host is slow (system load, PCIe contention), fabric init stalls.
- There is a residual window: between non-MMIO writing HANDSHAKE_READY and actually entering the receiver loop, the ERISC executes a few more instructions (nop, potential constexpr branches). If HOST_GATE_OPEN arrives before the non-MMIO ERISC enters the loop, the same race recurs. In practice this window is extremely tight (nanoseconds), but it is not zero.

The FIX CY spin loop includes proper termination checking: it polls `termination_signal_ptr` for IMMEDIATELY_TERMINATE on each iteration, with cache invalidation. On WH, the check uses a raw dereference instead of `got_immediate_termination_signal()` (which is compiled out behind `#ifndef ARCH_WORMHOLE` guards). WAYPOINT codes HGWT/HGOP/HGXT provide diagnostic observability.


## 5. Proposed Restructure: ETH DMA Pre-Ping

### 5.1 Design

The pre-ping replaces FIX CY's host-written gate with a direct ERISC-to-ERISC signal over ETH DMA. The non-MMIO ERISC (receiver) signals its MMIO peer (sender) that it is ready, using the same raw ETH DMA mechanism as the handshake itself.

Step-by-step:

NON-MMIO (receiver) side:
1. [existing] `wait_for_other_local_erisc()` barrier (line 3658)
2. [existing] Write HANDSHAKE_READY to edm_status_ptr (line 3663)
3. [NEW] Flush ETH TXQ if busy (guard against stale state from prior kernel)
4. [NEW] Write pre-ping sentinel to local pre-ping source buffer
5. [NEW] `eth_send_packet(0, preping_src_word_addr, preping_dst_word_addr, 1)` -- sends 16B to MMIO peer's pre-ping slot
6. [existing] Enter `fabric_receiver_side_handshake` (line 3697)

MMIO (sender) side:
1. [existing] `wait_for_other_local_erisc()` barrier (line 3658)
2. [existing] Write HANDSHAKE_READY to edm_status_ptr (line 3663)
3. [NEW] Spin waiting for pre-ping sentinel in local pre-ping slot (with cache invalidation + termination check)
4. [REMOVED] No HOST_GATE_OPEN spin (FIX CY eliminated)
5. [existing] Enter `fabric_sender_side_handshake` (line 3689)

The pre-ping guarantees that when the MMIO ERISC enters the sender loop, the non-MMIO ERISC has AT MINIMUM reached HANDSHAKE_READY and executed the eth_send_packet call. Since the non-MMIO ERISC falls through directly to `fabric_receiver_side_handshake` after sending the pre-ping, by the time the MMIO sender's first nonce packet arrives, the receiver is in its loop.

Gating: The pre-ping send should be gated on `!is_handshake_sender && enable_ethernet_handshake`. The pre-ping wait should be gated on `is_handshake_sender && enable_ethernet_handshake`. This uses existing CT args -- the pre-ping sender is always the NON-MMIO ERISC (the handshake receiver), which is the INVERSE of the handshake sender role. No new `is_preping_sender` CT arg is needed because the roles are strictly `!is_handshake_sender`. The `host_gate_enabled` CT arg is removed entirely.

### 5.2 Buffer Requirements

CRITICAL FINDING: `AERISC_FABRIC_SCRATCH_BASE` is NOT suitable for ETH DMA.

`eth_send_packet` operates on 16-byte word addresses (byte_addr = word_addr * 16). The SCRATCH_BASE address is computed as:

    TELEMETRY_BASE + 160 - 4 = TELEMETRY_BASE + 156

TELEMETRY_BASE is 32-byte aligned (dev_mem_map.h:260: `& ~31`). Offset 156 modulo 16 = 12. The SCRATCH_BASE is 4-byte aligned, NOT 16-byte aligned. Passing a non-16-byte-aligned address to `eth_send_packet` would corrupt the source/destination addressing.

Furthermore, SCRATCH_BASE is only 28 bytes (7 uint32_t slots). While this is enough space, the alignment problem is fatal for ETH DMA use.

RECOMMENDED: Allocate the pre-ping buffer via `erisc_datamover_builder`, the same way `handshake_addr` is allocated. The builder already enforces 16-byte alignment (erisc_datamover_builder.hpp:264: `eth_word_l1_alignment = 16`) and manages L1 layout. A new 16-byte slot after `handshake_addr` at line 316-317:

    this->handshake_addr = next_l1_addr;
    next_l1_addr += eth_channel_sync_size;  // 16 bytes
    this->preping_addr = next_l1_addr;       // NEW
    next_l1_addr += eth_channel_sync_size;   // 16 bytes, NEW

Pass `preping_addr` as a new CT arg `PRE_PING_ADDR` alongside `HANDSHAKE_ADDR`. This is clean, aligned, and follows the existing allocation pattern.

Pre-ping sentinel value: Use the `session_nonce` (same as the handshake). This provides stale-L1 protection -- if the L1 slot retains a value from a prior session, it will not match the current session's nonce. The sentinel must not be 0 (L1 default after reset) and must not be the same as HANDSHAKE_READY or any EDMStatus value.

The MMIO ERISC must zero the pre-ping slot before entering the spin (guard against stale sentinel from a prior session on the same slot). This is analogous to `init_handshake_info()` zeroing `local_value`.

### 5.3 Termination Handling

The pre-ping spin on the MMIO side MUST include termination checking. The codebase has a two-path pattern for WH vs non-WH (fabric_router_eth_handshake.hpp:42-47):

    #ifndef ARCH_WORMHOLE
        && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
    #else
        && *termination_signal_ptr != static_cast<uint32_t>(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE)
    #endif

On WH, `got_immediate_termination_signal()` is compiled out. The FIX CY spin loop (fabric_erisc_router.cpp:3677-3678) uses the raw dereference path. The pre-ping spin MUST replicate this exact pattern.

On early exit (termination received during pre-ping wait): write `*edm_status_ptr = EDMStatus::TERMINATED` and return immediately, matching FIX CY's HGXT behavior.

Each iteration of the spin must call `router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>()` to ensure the ERISC sees fresh L1 data.

WAYPOINT codes: PPWT (Pre-Ping Wait), PPOK (Pre-Ping received OK), PPXT (Pre-Ping eXit on Terminate), PPSD (Pre-Ping SenD). These replace HGWT/HGOP/HGXT.

### 5.4 Watchdog

Consistent with the codebase pattern, use WAYPOINT-only watchdog with kWatchdogIter = 100,000,000 (~2-4 seconds). Do NOT implement a hard break-out.

Rationale for no hard break-out: If the pre-ping never arrives (non-MMIO peer crashed during init, link failure), breaking out of the spin would cause the MMIO ERISC to enter the sender handshake loop with no guarantee the peer is listening -- recreating the exact race condition this fix eliminates. The MMIO ERISC must spin until either (a) the pre-ping arrives, or (b) the host sends IMMEDIATELY_TERMINATE.

The host DOES reliably send IMMEDIATELY_TERMINATE during teardown. The teardown path (fabric_firmware_initializer.cpp:634-822) writes the termination signal to the master router ETH core. The master then propagates to subordinates via the local notification mechanism. This has been verified in the code -- the escape hatch is reliable.

The `wait_for_other_local_erisc()` barrier at line 3658 DOES use a hard break-out (kSyncMaxIter at line 2944), but that is appropriate for same-device peers where a local ERISC crash is immediately diagnosable and where breaking out does not create a cross-link race. The pre-ping crosses an ETH link to a different chip -- different failure domain, different recovery model.

### 5.5 ETH Link Readiness

The ETH physical link is established during ERISC ROM boot, before the application firmware runs. There is no explicit ETH link-up check in the ERISC init path. The existing handshake itself relies on this -- `eth_send_packet` writes to ETH TXQ registers that would fail if the link were not trained.

The pre-ping runs BEFORE `init_handshake_info()` (which is called inside `fabric_sender/receiver_side_handshake`). The `init_handshake_info()` call includes an ETH TXQ flush (edm_handshake.hpp:78-81) to clear stale state from prior kernel runs. Since the pre-ping runs before this flush, it must include its own TXQ flush:

    if (eth_txq_is_busy()) {
        eth_txq_reg_write(0, ETH_TXQ_CMD, ETH_TXQ_CMD_FLUSH);
        eth_txq_reg_read(0, ETH_TXQ_CMD);
        while (eth_txq_is_busy()) {}
    }

This is a direct copy of the pattern in init_handshake_info. It guards against a prior kernel crash leaving the TXQ in a busy state. The flush uses queue 0 (same as the handshake).

ETH link drop between HANDSHAKE_READY and pre-ping send: The pre-ping packet would be silently lost. The MMIO ERISC spins until IMMEDIATELY_TERMINATE. This is NOT a new risk -- the existing handshake has the identical vulnerability. The watchdog WAYPOINT provides diagnostic visibility.

### 5.6 FIX CY Cleanup

Complete list of artifacts to remove:

FIRMWARE SIDE:
- fabric_erisc_router.cpp:3666-3685 -- entire `if constexpr (host_gate_enabled)` block and its FIX CY comment (HGWT/HGOP/HGXT waypoints)
- fabric_erisc_router_ct_args.hpp:268-272 -- `host_gate_enabled` CT arg definition and comment
- fabric_edm_packet_header.hpp:62 -- `HOST_GATE_OPEN = 0xA0B0C0D2` enum value in EDMStatus

HOST SIDE:
- erisc_datamover_builder.cpp:1226 -- `named_args["HOST_GATE_ENABLED"]` assignment
- device.cpp:2878-2925 -- entire `Device::open_erisc_handshake_gate()` method
- device_impl.hpp:195-199 -- declaration of `open_erisc_handshake_gate()`
- fabric_firmware_initializer.cpp:3540-3553 -- Phase C block calling `open_erisc_handshake_gate()` on MMIO devices

UTILITY:
- edm_status_utils.hpp:61,87 -- `HOST_GATE_OPEN` cases in status-to-string mappers

TOTAL: ~80 lines removed across 8 files.

Preserve FIX CY comment references (#42429) where they document the race condition, updated to reference the pre-ping mechanism.

Pass A / Pass B (fabric_firmware_initializer.cpp:3526-3538): Pass A (MMIO HANDSHAKE_READY polling) provides observability that MMIO ERISCs have completed init -- useful for diagnostics and for any future host-side gating. Pass B (non-MMIO HANDSHAKE_READY polling) is no longer needed for CORRECTNESS -- the pre-ping provides the synchronization. However, Pass B still provides DIAGNOSTIC VALUE: if a non-MMIO ERISC never reaches HANDSHAKE_READY, Pass B's timeout surfaces the failure to the host, which can then send IMMEDIATELY_TERMINATE. Without Pass B, the failure would be silent until the overall fabric-init timeout fires. RECOMMENDATION: Keep Pass B as a non-blocking diagnostic. Convert it to fire-and-forget: log the readiness state but do not block init on it. Remove Phase C entirely.


## 6. Open Questions / Risks

1. STALE L1 ON RE-INIT: If a test fails and fabric is torn down without zeroing the pre-ping slot, the next session's pre-ping slot may contain the OLD session's nonce. The MMIO ERISC zeroes the slot before spinning (see 5.2), but verify that the builder does not re-use the same L1 address across sessions without clearing it. The session_nonce is regenerated per session (FIX CT), so even if the old nonce remains, it will not match the new session_nonce. However, if by coincidence the new nonce equals the old stale value, the MMIO ERISC could proceed before the peer is ready. Mitigation: always zero the pre-ping slot before the spin.

2. MULTI-HOP TOPOLOGIES: The analysis assumes direct MMIO-to-non-MMIO ETH links. In multi-hop topologies where an intermediate relay chip exists, the pre-ping must traverse the same direct ETH link as the handshake. Since the handshake is per-ETH-link (not per-hop), and each ETH link has its own ERISC pair, the pre-ping design is correct for multi-hop.

3. BLACKHOLE DIFFERENCES: BH can have NUM_ACTIVE_ERISCS > 1 (two RISC cores per ETH channel). The pre-ping is per-ETH-link, and `wait_for_other_local_erisc()` ensures both local RISCs have completed init before either sends the pre-ping. No additional BH-specific handling needed. However, BH has a different `eth_txq_is_busy()` implementation (tunneling.h:74-78) due to hardware bug BH-55 -- verify the TXQ flush pattern works correctly on BH.

4. CI VALIDATION BEFORE FIX CY REMOVAL: The pre-ping should be deployed ALONGSIDE FIX CY first (both active, pre-ping checked first, host gate as fallback). Run T3K CI (racecondition-hunt workflow) for several days. Only remove FIX CY after zero deadlocks.

5. TERMINATION SIGNAL ON NON-MMIO: The teardown writes IMMEDIATELY_TERMINATE to the master router core (fabric_firmware_initializer.cpp:822). Non-MMIO devices are reachable via the MMIO relay. But if the MMIO device itself is stuck in pre-ping wait, the relay path may be blocked, preventing the terminate signal from reaching non-MMIO devices. This is a pre-existing issue with FIX CY as well (if MMIO is stuck in HOST_GATE_OPEN spin, same situation). The termination write goes via PCIe/NOC to the specific ETH core's L1, NOT through the fabric -- so ERISC firmware state does not block host L1 writes. This is safe.


## 7. Implementation Checklist

Ordered list of changes:

1. ERISC_DATAMOVER_BUILDER: Add `preping_addr` allocation (16B, 16-byte aligned) immediately after `handshake_addr`. Add `named_args["PRE_PING_ADDR"]` CT arg assignment.

2. CT_ARGS: Add `constexpr size_t preping_addr = NAMED_CT_ARG("PRE_PING_ADDR")` to fabric_erisc_router_ct_args.hpp.

3. FIRMWARE (non-MMIO path): After HANDSHAKE_READY write (line 3663), add:
   - TXQ flush guard
   - Write session_nonce to local pre-ping source buffer (at preping_addr)
   - `eth_send_packet(0, preping_src_word_addr, preping_dst_word_addr, 1)`
   - WAYPOINT("PPSD")
   Gate on: `!is_handshake_sender && enable_ethernet_handshake`

4. FIRMWARE (MMIO path): After HANDSHAKE_READY write (line 3663), REPLACING the host_gate_enabled block:
   - Zero the local pre-ping slot (preping_addr)
   - WAYPOINT("PPWT")
   - Spin loop: poll preping_addr for session_nonce, with cache invalidation, termination check (WH two-path pattern), and watchdog (kWatchdogIter = 100M, WAYPOINT "PPST" on timeout)
   - On success: WAYPOINT("PPOK"), fall through to sender handshake
   - On terminate: WAYPOINT("PPXT"), write TERMINATED, return
   Gate on: `is_handshake_sender && enable_ethernet_handshake`

5. CI VALIDATION: Deploy with FIX CY still active (pre-ping checked first, host gate as safety net). Run racecondition-hunt CI for multiple days.

6. FIX CY REMOVAL (after CI validation):
   - Remove firmware gate (fabric_erisc_router.cpp:3666-3685)
   - Remove CT arg (ct_args.hpp:268-272)
   - Remove HOST_GATE_OPEN enum (fabric_edm_packet_header.hpp:62)
   - Remove named_args["HOST_GATE_ENABLED"] (erisc_datamover_builder.cpp:1226)
   - Remove Device::open_erisc_handshake_gate() (device.cpp:2878-2925, device_impl.hpp:195-199)
   - Remove Phase C (fabric_firmware_initializer.cpp:3540-3553)
   - Remove status-to-string cases (edm_status_utils.hpp:61,87)
   - Convert Pass B to non-blocking diagnostic (log-only, no blocking poll)
   - Update FIX CY comments to reference pre-ping mechanism
