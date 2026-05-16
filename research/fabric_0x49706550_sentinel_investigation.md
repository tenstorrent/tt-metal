<!-- SUMMARY: Investigation of why 0x49706550 UMD relay sentinels persist between sessions
KEYWORDS: fabric, teardown, erisc, relay, sentinel, 0x49706550, quiesce
SOURCE: tt-metal source code analysis
SCOPE: Device teardown path, ETH relay initialization and cleanup
USE WHEN: Debugging why base-UMD relay firmware persists across sessions -->

# Investigation: 0x49706550 UMD Relay Sentinel Persistence Between Sessions

## What is 0x49706550

The value 0x49706550 ("IpeP" in ASCII big-endian) is not a member of the EDMStatus enum. All fabric
firmware status values follow a 0xA_B_C_D_ pattern (STARTED=0xA0B0C0D0, TERMINATED=0xA4B4C4D4, etc).

The sentinel appears at `edm_status_address`, which is allocated starting at
`get_erisc_l1_unreserved_base()`. On Wormhole T3K, `is_base_routing_fw_enabled()` returns true, so
`ERISC_L1_UNRESERVED_BASE = ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE = 0x18000`.

The UMD relay firmware's own TILE_HEADER_BUFFER occupies exactly 0x18000–0x1FFFF (32KB). The fabric
firmware's `edm_status_address` lands within that range. When the UMD relay firmware is running, the
word at that L1 address contains tile header data written by the relay — whatever value happens to be
at that offset reads back as 0x49706550 in normal relay operation.

The same sentinel also appears when a process is killed mid-handshake: fabric firmware init writes
partial L1 data starting at 0x18000 (edm_local_sync_address), then writes edm_status_address
(= edm_local_sync_address + 16) as one of its first initialization writes. A SIGKILL between those
writes leaves the channel with 0x49706550 at edm_status_address from the partial UMD relay state or
from the partial fabric firmware write.

In `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`, the cleanup function
`terminate_stale_erisc_routers()` classifies this explicitly:

    static constexpr uint32_t kBaseUmdFirmwareSentinel = 0x49706550u;
    const bool is_base_umd = (status_buf[0] == kBaseUmdFirmwareSentinel);

Channels with this sentinel are added to `base_umd_channels` and `skip_soft_reset_channels`.
`configure_fabric_cores()` then skips the BRISC soft-reset for those channels and uses
`write_launch_msg_to_core` instead (FIX M / PR #42429).

The code cannot distinguish "UMD relay firmware running normally" from "fabric firmware crashed
mid-init leaving 0x49706550". Both look identical to the probe read.


## What Teardown Is Supposed to Do

`teardown_fabric_config()` in `tt_metal/impl/context/metal_env.cpp` is the sole path that
deliberately transitions ERISCs back to UMD relay state:

1. Wait up to 5000ms per ERISC channel for the fabric router kernel to write TERMINATED
   (0xA4B4C4D4) at edm_status_address.
2. On clean termination (`terminated_cleanly=true`): call `assert_risc_reset_at_core` +
   `deassert_risc_reset_at_core` on ERISC0. This resets the core, at which point it boots the
   UMD relay firmware and resumes normal relay operation.
3. Quiesce non-MMIO devices before MMIO devices (relay ordering requirement — if Device 0 MMIO
   relay is quiesced first, non-MMIO devices cannot issue L1 reads and time out).

After a successful teardown, each ETH channel's ERISC is running UMD relay firmware and
edm_status_address contains whatever tile header data the relay wrote, i.e. 0x49706550. That is
the expected clean state for the next session's `terminate_stale_erisc_routers()` to find.


## What Is Failing

Two failure modes leave channels with 0x49706550 in a way that confuses the next session or causes
topology mapper failures:

### Failure Mode 1: Teardown Skipped Entirely (SIGKILL / Hard Kill)

If the process is killed with SIGKILL (e.g. CI job timeout, OOM, or test harness forcibly
terminating a hung test), `teardown_fabric_config()` never runs. The fabric router kernels are left
mid-flight — possibly in STARTED or LOCAL_HANDSHAKE_COMPLETE state. The ERISC cores retain their
L1 state from the fabric firmware init. The next session's `terminate_stale_erisc_routers()` sees
either a valid EDMStatus (and sends TERMINATE, then waits) or 0x49706550 (and treats it as
base-UMD). The problem arises when a multi-device session was killed during the handshake phase:
some channels are in known EDMStatus states, others in 0x49706550, and the topology mapper must
handle the mixed state.

### Failure Mode 2: Teardown Timeout Skips ERISC Reset (F5a Path)

When `teardown_fabric_config()` waits 5000ms and the channel does not write TERMINATED (e.g. the
fabric router kernel hung or never started), the code sets `terminated_cleanly=false` and
deliberately skips the `assert/deassert_risc_reset_at_core` step. This is a conservative choice
(avoid disturbing a potentially-live core) but it leaves the ERISC in an undefined state. Its
edm_status_address is not cleared. The next session reads it, cannot classify it reliably, and may
see 0x49706550 from a partial L1 state.

### Failure Mode 3: Relay Ordering Race (Multi-Device Sessions)

On T3K, Device 0 is the MMIO chip that provides L1 read relay service to non-MMIO devices. If
Device 0's teardown quiesces / resets its ERISC relay before non-MMIO devices finish their own
teardown, non-MMIO devices lose their read path. The L1 read RPCs through Device 0 time out after
5s (UMD default). This causes `teardown_fabric_config()` on non-MMIO devices to time out waiting
for TERMINATED, which then triggers Failure Mode 2 for those channels. Those channels are found
with 0x49706550 by the next session.

The AI-JOURNAL.md session 4 notes document a concrete occurrence: all non-MMIO ETH channels were
in base-UMD relay state from a prior crashed session, causing TopologyMapper to throw before test
code even runs.


## Root Cause Summary

The root cause is that `teardown_fabric_config()` does not guarantee ERISC reset for timed-out
channels. When teardown is skipped or times out (either from SIGKILL or from relay ordering race),
channels are left in whatever L1 state they had, which can be 0x49706550. The next session then
has to recover from this state, and the recovery path (base_umd_channels + skip_soft_reset) works
correctly for channels that are truly running UMD relay, but is ambiguous for channels that are
stuck in partial fabric firmware state.


## What Fix Would Address the Root Cause

Three complementary changes:

1. **Force-reset on teardown timeout**: In `teardown_fabric_config()`, when a channel times out
   (terminated_cleanly=false), unconditionally assert+deassert ERISC0 reset. This guarantees the
   ERISC boots into UMD relay firmware with a clean L1 state. The current conservative skip was
   motivated by fear of disturbing a live core, but a 5s timeout already establishes the core is
   not cooperating.

2. **Enforce relay quiesce ordering**: Non-MMIO devices must fully complete their
   `teardown_fabric_config()` before MMIO Device 0 begins its own ERISC reset. If Device 0 resets
   its relay ERISCs first, non-MMIO teardowns time out and fall into Failure Mode 2.

3. **Distinguish base-UMD from crashed-mid-init**: Write a known canary value (e.g. 0xDEAD0000)
   to edm_status_address as the very first write in fabric firmware init, before writing
   edm_local_sync_address. Then `terminate_stale_erisc_routers()` can distinguish the two cases:
   0x49706550 = genuine UMD relay, 0xDEAD0000 = crashed mid-init (needs force-reset + retry).
   Currently both are treated identically as "base-UMD" and the skip_soft_reset path is used for
   both.


## Key Files

- `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` — terminate_stale_erisc_routers(), classify/handle 0x49706550
- `tt_metal/impl/context/metal_env.cpp` — teardown_fabric_config(), the F5a timeout-skip path
- `tt_metal/fabric/erisc_datamover_builder.cpp` — edm_status_address allocation starting at 0x18000
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h` — ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE=0x18000
- `tt_metal/third_party/umd/src/firmware/riscv/wormhole/eth_l1_address_map.h` — TILE_HEADER_BUFFER_BASE=0x18000 (UMD relay)
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` — only writes 0xA_B_C_D_ EDMStatus values
- `tt_metal/fabric/fabric_edm_packet_header.hpp` — EDMStatus enum (0x49706550 absent)
- `tt_metal/llrt/tt_cluster.cpp` — is_base_routing_fw_enabled() returns true for T3K
