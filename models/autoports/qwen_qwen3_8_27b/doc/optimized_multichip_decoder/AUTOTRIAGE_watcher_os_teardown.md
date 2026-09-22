# AutoTriage: fabric router leaves packet tags set on kernel exit

## Diagnosis

The full-watcher O3/noinline failure is a proven fabric router cleanup bug.
After traffic, ERISC1 returns from `kernel_main` with a nonzero sticky
`NOC_PACKET_TAG` register. The kernel wrapper asserts at
`tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:51`.
ERISC1's assertion handler cannot return to base firmware and spins forever;
ERISC0 waits for the subordinate's DONE signal. The host heartbeat timeout is
subsequent fanout, not the first failure.

The smallest repair is `noc_clear_packet_tags(noc_index)` in router teardown,
after its existing write/atomic barriers and before its final peer rendezvous.
This restores the owned NoC's configuration after traffic has drained. It
preserves every exit assertion and all synchronization. Do not clear both
NoCs from each ERISC: the other NoC belongs to its peer/base firmware.

The earlier Os run has the same passing-model/failing-close symptom but no
post-abort mailbox capture, so the exact assertion is directly proven for
O3/noinline only. Both optimization settings fit full instrumentation into the
fabric config allocation; changing compiler optimization alone does not
restore packet tags.

## Triage evidence

Parent-owned live captures, taken before reset after the O3/noinline abort:

- `triage_watcher_noinline/l1_manifest.json`: 128 KiB at L1 address zero on
  all 16 active ETH cores, across four chips, with SHA-256 for each binary.
- `triage_watcher_noinline/pc_samples.json`: four nonhalting debug-bus PC
  samples for each of the 32 ERISCs. The capture checked that a PC debug-bus
  signal exists before calling `get_pc`; it did not take the halt fallback.
- `watcher_full_eth_noinline/generated/inspector/kernels.yaml` and its
  actual fresh-cache ELFs/named compile arguments identify the loaded kernels.
- `triage_watcher_noinline/decode_saved_state.py` is an offline stdlib decoder
  using the actual firmware's DWARF layout and saved kernel launch addresses.
  It verifies capture hashes and generates `decoded_state.json`. Its
  subprocesses only run RISC-V `addr2line` on saved ELF files.

Every captured core has exactly this state:

| Field | Value | Meaning |
| --- | --- | --- |
| Watcher assertion line/type/which | 51 / 11 / 1 | ERISC1, `DebugAssertNCriscNOCPacketTagClearedTripped` |
| ERISC0 / ERISC1 waypoint | SEW / NKFW | Subordinate wait / kernel NoC exit checks |
| Subordinate sync | 0x80 | GO, never transitioned to DONE=0 |
| `aerisc_run_flag` | 0 | Host/assert handler requested application exit |
| Go-message signal | 0 | Assert handler reported DONE for the launch; this is distinct from subordinate sync |
| Termination word | 2 | `IMMEDIATELY_TERMINATE` reached every router |
| EDM status | 0xa4b4c4d4 | `TERMINATED`; router teardown reached its final status store |

The enum definitions are in `hw/inc/hostdev/dev_msgs.h:292-301` and
`fabric/fabric_edm_packet_header.hpp:37-63`. The captured 16-bit line number,
type byte, and RISC byte are at L1 addresses 0x620, 0x622, and 0x623.
These offsets were independently checked against the exact compiled
`firmware/active_erisc/active_erisc.elf` DWARF and the subordinate kernel's
assert-handler disassembly. They are specific to this capture's firmware ABI.

For device 0 ETH29-25, saved active-ETH config base is 62256, ERISC1 kernel
text offset is 11792, and the ELF entry is 0xc710. Thus live PC 74676 maps to
ELF 0xc984: `assert_and_hang`, `assert.h:132`, instruction `j 0xc984`.
Samples also include the immediately following address 0xc988; this does not
establish execution of the adjacent sanitizer function. All 16 ERISC1s have
the same self-loop/next-address sampling pattern after their own relocation.
ERISC0 samples map to `wait_subordinate_eriscs`, NoC synchronization, context
switch, and base-firmware addresses. Context-switch progress is compatible
with the permanent subordinate wait and does not mean application teardown
completed. Reset-PC registers in the host error are not live PC samples.

`watcher_full_eth_noinline.log` shows strict model output, cache, and replay
checks passed before watcher detach at 02:38:28.792 UTC, then close aborts
with exit 134. `watcher_empty_mesh.log` is the parent's same watcher/O3/
noinline Ring TP4 open/close control: all checks enabled, no model traffic,
clean exit 0 and driver close. This contrast supports the sticky-tag contract:
initial tags are clear, and workload transactions set them.

The preceding Os run (`watcher_full_eth_os.log`) passed model checks before
watcher detach at 02:33:45.402 and timed out at 02:34:05.404, exit 134.
Its saved watcher sample predated shutdown and reported no assertion.
Its attempted post-abort tt-triage capture had no live Inspector RPC or
serialized capnp state; YAML alone could not reconstruct the failed state.
The later raw capture resolves this evidence gap for the noinline run.

## Source evidence and ownership ledger

1. Fabric receive/local/forward writes use explicit transaction IDs through
   `edm_fabric_utils.hpp:38-63` and `fabric_edm_packet_transmission.hpp`.
   `blackhole/noc_nonblocking_api.h:554-556` writes `NOC_PACKET_TAG` when
   `use_trid` is true. The stateful write API also writes the tag at line 1294.
   These registers retain configuration after transfers complete.
2. Router teardown first rendezvous, drains receiver transaction IDs via
   `WriteTransactionIdTracker::all_buffer_slot_transactions_acked`
   (`fabric_erisc_router_transaction_id_tracker.hpp:109-118`), then reinitializes
   software counters, fans out termination, performs write and atomic
   barriers, rendezvous again, and stores TERMINATED
   (`fabric_erisc_router.cpp:2879-2959`). This entire flow completed in the
   captures. Neither `noc_async_write_barrier_with_trid` (`dataflow_api.h:2603`)
   nor `ncrisc_noc_counters_init` (`blackhole/noc_nonblocking_api.h:831`)
   resets packet-tag configuration. The router contains no tag clear.
3. `ncrisc_noc_packet_tags_cleared` checks WR, WR_REG, and AT command-buffer
   tag registers equal zero (`blackhole/noc_nonblocking_api.h:253-258`).
   This is configuration restoration, not a transaction-flush counter check.
   The earlier read/write/atomic checks passed on ERISC1 before line 51.
4. The wrapper's contract is therefore correctly catching omitted cleanup.
   The adjacent fabric mux already uses the same repair pattern: full barrier,
   `noc_clear_packet_tags(noc_index)`, then TERMINATED
   (`fabric/impl/kernels/tt_fabric_mux.cpp:272-276`).
5. `FabricRiscConfig` assigns ERISC0 to NoC0 and ERISC1 to NoC1
   (`fabric/erisc_datamover_builder.cpp:202-203`); the single-ERISC Blackhole
   special case assigns its configured NoC1 instead (`:288-294`).
   `compute_mesh_router_builder.cpp:983-993` passes that configured NoC into
   `EthernetConfig`. The offline decoder checked all 32 actual generated
   named-argument maps: each of 16 ERISC0s services only NoC0 sender paths;
   each of 16 ERISC1s services only NoC1 receive/local/forward paths. Therefore
   `noc_index`, rather than a hard-coded 1 or all-NoC loop, names the owned
   interface for this repair. Existing dynamic-NoC static assertions remain.
6. Clearing its owned tags after the existing drains and before the final
   local rendezvous cannot overwrite peer-owned command buffers. No further
   NoC transactions are issued after that rendezvous; the remaining status
   update is an L1 store. This also leaves the established single-ERISC
   configured-NoC behavior intact. No claim of comprehensive validation of
   other architectures/topologies is made by this target-specific evidence.

## Downstream effects

`api/debug/assert.h:110-132` writes the launch DONE flag and disables the
Ethernet application, but only ERISC0 can call `erisc_exit`; ERISC1 enters
the infinite loop. Subordinate firmware therefore never reaches its
`signal_subordinate_erisc_completion()` after the kernel call
(`firmware/src/tt-1xx/subordinate_erisc.cc:151-165`). Main firmware stays in
`active_erisc.cc:94-104`, waiting for subordinate sync to change from 0x80
to zero. Writing the application run flag again cannot release this wait.

`MetalContext` destroys watcher before RISC firmware teardown. Therefore
pre-detach clean watcher logs do not clear assertions occurring during exit.
This investigation requires no watcher suppression or timeout increase.
The DCBA heartbeat word is written by the fabric router; the stale value is
consistent with a failed handback. Firmware bundle 19.8.0 is already newer
than the generic minimum 18.10.0 in the error; an upgrade is not the diagnosis.

## Proposed fix and discriminating verification

Patch: `watcher_packet_tags.patch.gz`, isolated in
`/home/mvasiljevic/qwen38-full-rerun/watcher-tags-fix`.
One source file, one call plus explanatory comment in teardown. No native
factory/model changes, config allocation changes, assertion changes, new
waits, or watchdog changes.

The source/triage hypothesis is verified. Hardware verification of the repair
belongs to the parent and is pending at patch handoff. Run serially:

1. Apply the patch only after all device processes close. Use a fresh
   `TT_METAL_CACHE` and full watcher10 O3/noinline, with no disabled features.
   Re-run the exact traffic-bearing failing model command and retain its
   strict output/cache/trace checks. Require process exit 0 and driver close;
   passing JSON before shutdown is insufficient.
2. Re-run a second open/traffic/close cycle to cover handoff/reuse. An empty
   open/close is a control only and was already passing before the fix.
3. If the O3/noinline case passes, full watcher10 Os with a separate fresh
   cache can confirm the earlier alternative also closes. Watcher/profiler
   remain separate. Check generated compile commands to prove the patch's
   new JIT binary was materialized; per-kernel optimization levels are not
   safely distinguished by the existing cache key.
4. If line 51 persists, preserve post-abort L1/assertion/PC state before
   reset. If a new assertion appears, treat it as a new contract failure,
   not grounds to suppress checks or label this repair complete.

No compile, import of TTNN/ExaLens, linked probe, or device command was run by
this offline diagnosis. Source formatting/diff checks and the saved-state
decoder were run. The parent will compile the modified kernel through the
normal device JIT before claiming runtime verification. Raw tag-register
values were not captured, so the exact offending command buffer is unknown;
assert type 11 proves at least one of the three checked registers was nonzero.
