<!--
SUMMARY: Adversarial race-condition audit of TT-Metal EDM fabric firmware and host driver
KEYWORDS: edm, fabric, race-condition, deadlock, hang, erisc, flow-control, firmware
SOURCE: Manual code audit of nsexton/0-racecondition-hunt branch
SCOPE: 13 files covering ERISC router, channels, packet TX, flow control, connection management, firmware
USE WHEN: Investigating EDM fabric hangs, debugging flow control stalls, reviewing fabric changes
-->

# EDM Fabric Race-Condition Audit v2

Branch: `nsexton/0-racecondition-hunt`
Date: 2026-05-10
Auditor: BrAIn (Claude Opus 4.6)

---

## File 1: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`

### Hazard 1.1: Teardown sync `wait_for_other_local_erisc()` has no termination escape  [MEDIUM]
- Location: fabric_erisc_router.cpp:2861-2878
- Mechanism: `wait_for_other_local_erisc()` spins on a scratch register value written by the other ERISC, with NO termination signal check. The code explicitly comments "No termination check is added here intentionally" because both ERISCs share the same termination signal. However, if one ERISC crashes or hangs before reaching the sync point (e.g., during teardown transaction flush at line 2912), the other ERISC will spin forever in `wait_for_other_local_erisc()`.
- Evidence:
```cpp
// Line 2864-2870
if constexpr (IS_TEARDOWN_MASTER()) {
    write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(multi_erisc_sync_start_value);
    while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) !=
           multi_erisc_sync_step2_value) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    }
```
- The teardown function calls `wait_for_other_local_erisc()` THREE times (lines 2909, 2935, 2954), amplifying the probability.
- Fix: This is by-design — adding a termination check would be worse (one ERISC could skip the scratch register write, causing the other to spin forever on a different value). The real fix would be a timeout with watchdog. Not fixing.

### Hazard 1.2: `outbound_to_receiver_channel_pointer` is a local copy, not a reference  [MEDIUM]
- Location: fabric_erisc_router.cpp:2213-2214
- Mechanism: `outbound_to_receiver_channel_pointer_ch0` is initialized as a **copy** of the tuple element, not a reference:
```cpp
auto outbound_to_receiver_channel_pointer_ch0 =
    outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();
```
  This means the `num_free_slots` field tracks credits independently of the original tuple element. This appears intentional ("Workaround the perf regression in RingAsLinear test"), but it means the original `outbound_to_receiver_channel_pointers` tuple never gets updated during the main loop. If any code path reads from the tuple instead of the local copy, it would see stale credit counts.
- Evidence: Grep confirms only the local copy is used in the main loop. However, if future code uses the tuple element directly, it would silently see stale data.
- Fix: Not a current bug, but a fragility. Not fixing.

### Hazard 1.3: Coordinated context switch during pause can race with erisc1 routing  [LOW]
- Location: fabric_erisc_router.cpp:2681-2688
- Mechanism: When erisc0 receives PAUSE command, it calls `coordinated_context_switch_start_as_master()` then `execute_pause_command()`. Meanwhile erisc1 is in `execute_main_loop()` and only checks for the retrain intent via `run_routing_without_noc_sync_coordinated_as_non_master()` at the bottom of each main loop iteration. Between erisc0 writing RETRAIN_INTENT and erisc1 seeing it, erisc1 continues routing packets. This is benign — erisc1's routing is independent — but could be an issue if the pause is meant to quiesce all traffic before a state change.
- Fix: Not fixing — current behavior is by-design for performance.

### CLEAN areas verified:
- **Main loop termination**: `continue_running_main_run_loop()` checks both termination signal and state manager command — cannot infinite-loop.
- **Packet forwarding**: `receiver_forward_packet()` always invokes either `forward_payload_to_downstream_edm`, `execute_chip_unicast_to_local_chip`, or both. No case where a dequeued packet is silently dropped — the only early-return is in `send_next_data` when teardown is requested before `eth_send_packet_bytes_unsafe`.
- **Credit tracking**: `num_free_slots` (uint32_t) is decremented only after `has_space_for_packet()` guard (line 1680-1684) and incremented by `completions_since_last_check` (signed int32_t returned by flow control). The signed return prevents spurious underflow.
- **ETH TXQ wait**: Post-send TXQ drain (line 622-630) correctly spins without teardown escape — the packet is already committed. Pre-send teardown check (line 603-607) handles the cancel case.
- **Context switch**: `run_routing()` is only called in well-defined idle states (SWITCH_INTERVAL exceeded, or during wait loops). No re-entrancy risk since the caller is single-threaded.

---

## File 2: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp`

### CLEAN — No CRITICAL/HIGH/MEDIUM hazards found.

**Analysis summary:**
- `StaticSizedSenderEthChannel`: Circular buffer with `wrap_increment` for index management. Single-core access per channel — no cross-core races.
- `StaticSizedEthChannelBuffer`: Receiver buffer with address array. Same single-core access pattern.
- `EdmChannelWorkerInterface`: Base class with `cached_worker_semaphore_address` (uint64_t) and `connection_live_semaphore` (volatile uint32_t*). The `uint64_t` field on a 32-bit RISC-V could theoretically have torn reads, but it is only written by `cache_producer_noc_addr()` and read by the same core — no cross-core access, so torn reads are impossible.
- `StaticSizedSenderChannelWorkerInterface`: Read/write counter management with `ChannelCounter`. Credit notification to workers via NOC write of read counter. All state is core-local.
- `connection_live_semaphore` is correctly declared `volatile` — the only field that can be modified by remote NOC writes.
- `wrap_increment` handles all edge cases (power-of-2, size 1, size 2, general) correctly.
- Buffer pointer arithmetic in `advance_buffer_slot()` uses pre-cached `next_buffer_address` which is always updated after use — no off-by-one or stale pointer risk.

---

## File 3: `tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp`

### CLEAN — No CRITICAL/HIGH/MEDIUM hazards found.

**Analysis summary:**
- `flush_write_to_noc_pipeline()`: Spins on transaction ID completion with no timeout. Bounded by hardware — NOC must eventually flush. No escape needed since the caller has already committed the write.
- `execute_chip_unicast_to_local_chip_impl()`: Reads packet header from L1 via `tt_l1_ptr` (non-volatile). Correct because the packet is fully received and stable in L1 before dispatch. The switch on `noc_send_type` covers all expected types with `ASSERT(false)` default.
- `forward_payload_to_downstream_edm()`: Modifies packet header (hop decrement) then sends via NOC. Both operations are on the same core — no TOCTOU race. The `ASSERT` for downstream space is "best effort" per comment, but the real check happens in the caller.
- `update_packet_header_for_next_hop()` (LowLatency variant): Uses `const_cast` to write through volatile pointer to `route_buffer`. The cast is necessary because the volatile qualifier on the packet header makes array element access awkward. Safe because only one ERISC touches this buffer at a time.
- Scatter write: `offset` accumulates chunk sizes. If malformed packet has chunk sizes summing past `payload_size_bytes`, the `final_chunk_size = payload_size_bytes - offset` would underflow (uint16_t wraps). This is a packet integrity issue, not a race.

---

## File 4: `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp`

### CLEAN — No CRITICAL/HIGH/MEDIUM hazards found.

**Analysis summary:**
- Two credit system implementations selected at compile time: counter-based (multi-TXQ) and stream-register-based (single TXQ). Both are race-free.
- Counter-based: `send_completion_credit()` increments local copy then writes to volatile L1 then calls `eth_send_packet_bytes_unsafe()`. The local copy avoids an L1 load. The volatile write ensures the sender side sees the update after ETH transfer.
- Stream-register-based: Uses `remote_update_ptr_val` for atomic remote register updates. `increment_local_update_ptr_val` with negative values for local decrement — this is the stream register API convention.
- `get_num_unprocessed_acks_from_receiver()` uses `router_invalidate_l1_cache()` before reading volatile counter. Subtraction `*counter_ptr - processed_count` is safe with uint32_t wraparound as long as difference fits in uint32_t (always true for credit counts).
- `receiver_send_completion_ack()`/`receiver_send_received_ack()` with `CHECK_BUSY=true`: spin on `eth_txq_is_busy()` with no escape. Bounded by hardware ETH TX completion. Empty loop body is intentional (PAUSE hint caused 13.8% BW regression per comment).

---

## File 5: `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp`

### CLEAN — No hazards found.

**Analysis summary:**
- Simple wrapper around `WorkerToFabricEdmSender` for forward/backward connections. All operations are single-threaded worker-side code.
- `open()` correctly separates start/finish phases: sends both open requests first, then waits for both to complete. This allows pipelining the handshakes.
- `build_from_args()` with `BUILD_AND_OPEN_CONNECTION_START_ONLY` documents the requirement to call `open_finish()` manually — clear API contract.
- `close()` similarly separates start/finish. No ordering hazards since both directions are independent.

---

## File 6: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp`

### Hazard 6.1: VC2 speedy path TRID ping-pong produces garbage TRID  [HIGH — FIX ER1]
- Location: fabric_erisc_router_speedy_path.hpp:260
- Mechanism: The TRID ping-pong uses `1 - current_write_trid` to toggle between two TRIDs. This only works when `current_write_trid` is 0 or 1. For VC2, `RX_CH_TRID_STARTS[2] = 2 * NUM_TRANSACTION_IDS = 8`, so `current_write_trid` starts at 8 and the flip produces `1 - 8 = 249` (uint8_t underflow/wrap).
- Evidence:
```cpp
// Line 260 — BROKEN for VC_ID > 0
receiver_state.current_write_trid = 1 - receiver_state.current_write_trid;
```
  With `RX_CH_TRID_STARTS[2] = 8`: first flip → `1-8 = 249`. The NOC TRID 249 was never used, so `ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, 249)` returns true immediately → completion credits sent before NOC writes land → sender reuses buffer → **data corruption**.
- Trigger: Any 2D fabric configuration with `FABRIC_2D_VC2_SERVICED` defined and VC2 receiver channel active in speedy mode. The `RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2 = 1` means the flip fires after the very first received packet.
- Fix: Replace with `2 * RX_CH_TRID_STARTS[VC_ID] + 1 - current_write_trid` to correctly toggle between base and base+1. **FIXED in commit 731764a384d.**

### CLEAN areas verified:
- Sender speedy path credit amortization: `sender_amort_counter` tracks sends, `completion_count` accumulates completions. The subtraction `sender_amort_counter -= completion_count` is safe because completions are a subset of sends.
- `speedy_state_copy_in`/`speedy_state_copy_out`: Copy semantics for register-local copies. The persistent ↔ local copy pattern avoids aliasing issues.
- `increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1)` at line 156: adds 1 to the stream register, signaling the worker that a slot is freed. This is after the send completes (TXQ drain at line 141-149), so the worker won't race ahead.
- Pre-send teardown check at line 124: Single non-spinning check before `eth_send_packet_bytes_unsafe`. Correct — the TXQ busy check already happened in `can_send`.

---

## File 7: `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp`

### CLEAN — No hazards found.

**Analysis summary:**
- V2 of FabricConnectionManager with N-slot connection array (max 4). Same open/close separation pattern. All single-threaded worker-side code.

---

## File 8: `tt_metal/fabric/hw/inc/edm_fabric/datastructures/outbound_channel.hpp`

### CLEAN — No hazards found.

**Analysis summary:**
- 35-line CRTP base class (`SenderEthChannelInterface`). Three pure forwarding methods: `init`, `get_cached_next_buffer_slot_addr`, `advance_to_next_cached_buffer_slot_addr`. No state, no concurrency concerns.

---

## File 9: `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp`

### CLEAN — No hazards found.

**Analysis summary:**
- 27-line constants file. Connection state values (unused=0, open=1, close_request=2) and stream IDs. No logic.

---

## File 10: `tt_metal/hw/firmware/src/tt-1xx/erisc.cc`

### CLEAN — No hazards found.

**Analysis summary:**
- ERISC firmware main loop (170 lines). Startup wait for `routing_enabled`, then dispatch loop checking `go_message_signal`.
- `RUN_MSG_DONE` signal written to local L1 before `notify_dispatch_core_done()` NOC send. NOC ordering guarantees the dispatch core sees the updated signal when it reads back via NOC.
- `launch_msg_rd_ptr` ring buffer uses power-of-2 mask wrapping. Standard pattern, no overflow risk.
- Context switch (`risc_context_switch()`) in idle path — no state corruption since kernel is not running during context switch.

---

## File 11: `tt_metal/hw/firmware/src/tt-1xx/brisc.cc`

### CLEAN — No fabric-specific hazards found.

**Analysis summary:**
- General BRISC firmware (592 lines). Dispatches kernels including fabric kernels. Synchronization with NCRISC/TRISCs via `subordinate_sync` fields with proper `invalidate_l1_cache()` in spin loops.
- Line 305: `while (subordinate_sync->dm1 != RUN_SYNC_MSG_WAITING_FOR_RESET)` lacks explicit cache invalidation but `subordinate_sync` is `tt_l1_ptr` qualified (Wormhole-specific, NCRISC IRAM reset path). This is a Wormhole-only path and the NCRISC writes to L1 which BRISC reads — the `tt_l1_ptr` qualifier ensures volatile-like behavior.
- No direct fabric EDM logic in this file. It's the kernel dispatch infrastructure.

---

## File 12: `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`

### CLEAN — No CRITICAL/HIGH/MEDIUM hazards found.

**Analysis summary:**
- `WorkerToFabricEdmSenderBase` (720 lines): Main worker-to-EDM adapter used by all fabric operations. Single-threaded worker-side code — no cross-core races.
- `open_start()`: Piggybacks readback data onto `worker_teardown_addr` temporarily. Safe because only the worker initiates teardown, and teardown cannot be requested until after `open_finish()` completes.
- `open_finish()`: Calls `noc_async_read_barrier()` to ensure readback of EDM buffer index and read counter are complete before using them to initialize `buffer_slot_write_counter`. Correct ordering.
- `close_finish()`: Bounded wait loop (5M iterations) for teardown ACK — prevents indefinite hang. The bound is generous but finite.
- `get_num_free_write_slots()`: Has explicit `invalidate_l1_cache()` before reading `edm_buffer_local_free_slots_read_ptr`. Comment explains this prevents hangs even when data cache is disabled — the L1 invalidation flushes any stale prefetch. Correct pattern.
- Credit tracking (counter mode): `buffer_slot_write_counter.counter - *edm_buffer_local_free_slots_read_ptr` computes in-flight slots. Safe with uint32_t wraparound as long as in-flight count < 2^32 (always true).
- Credit tracking (stream register mode): `get_ptr_val()` reads stream register directly. No caching issue since stream registers bypass L1.
- `advance_buffer_slot_write_index()`: Increments counter and wraps buffer index. Called exactly once per send in `post_send_payload_increment_pointers()`. No double-advance risk.
- `post_send_payload_increment_pointers()`: Updates remote free slots (NOC write or stream register increment) then advances local index. The NOC write is fire-and-forget — the remote receiver sees the update asynchronously. This is correct because the receiver only uses this to know when new data is available, and the data was already sent before this call.
- No volatile/cache hazards: The only remotely-modified field (`edm_buffer_local_free_slots_read_ptr`) is read through a volatile-qualified pointer with explicit cache invalidation.

---

## File 13: `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_async_datamover.hpp`

### CLEAN — No CRITICAL/HIGH/MEDIUM hazards found.

**Analysis summary:**
- Legacy async datamover (593 lines). Simpler state-machine design compared to the new fabric EDM router. Single ERISC drives all channel state transitions sequentially.
- `ChannelBuffer`: Explicit state machine with sender states (SIGNALING_WORKER → WAITING_FOR_WORKER → READY_FOR_ETH_TRANSFER → WAITING_FOR_ETH) and receiver states (WAITING_FOR_ETH → SIGNALING_WORKER → WAITING_FOR_WORKER). All transitions are guarded by condition checks.
- **First-level / second-level ack race (documented and handled)**: Comment at lines 248-258 describes the race where receiver's first-level ack and second-level ack could alias if sent from the same L1 address. The code avoids this by using a separate `eth_transaction_ack_word_addr` for the first-level ack. Verified that the addresses differ via `ASSERT` at line 263.
- `sender_eth_send_data_sequence()`: `advance_buffer_index()` is called after `eth_send_bytes_over_channel_payload_only()` but before the ETH transfer completes. This is correct — the sender's buffer is committed to the ETH subsystem and won't be re-read.
- `eth_receiver_channel_done()`: Zeroes both `bytes_sent` and `bytes_acked` locally, then sends the zeroed `bytes_sent` via ETH to the remote sender. Local zero before ETH send is safe — only the local ERISC reads these fields, and it's the one executing this code.
- `sender_eth_check_receiver_ack_sequence()`: Uses `eth_is_receiver_channel_send_acked() || eth_is_receiver_channel_send_done()` — handles both explicit ack and implicit "done" signal. No race because both checks read from different L1 addresses (`channel_bytes_acked` vs `channel_bytes_sent`).
- `is_local_semaphore_full()`: Correctly calls `invalidate_l1_cache()` before reading the volatile semaphore. Properly handles both `MESSAGE_COUNT_REACHED` and `WORKER_INITIATED` termination modes.
- `sender_notify_workers_if_buffer_available_sequence()`: Clears semaphore (line 432) then increments worker semaphores (line 433). Order is safe because the increment signals workers to START work — no worker can have completed work between clear and increment.
- `receiver_noc_read_worker_completion_check_sequence()`: Guards `eth_receiver_channel_done()` behind `!eth_txq_is_busy()`. If TXQ is busy, returns false and caller retries. The semaphore state is stable across retries since it wasn't cleared.
- `QueueIndexPointer::distance()`: Wraparound arithmetic assumes pointers don't wrap more than once. Safe given `wrap_around = queue_size * 2` and queue capacity invariant.
- No cross-core data races: All state is accessed by a single ERISC. Worker interaction is through semaphores with proper volatile/cache-invalidation.

---

## Summary

### Hazards found: 4 total
- **CRITICAL**: 0
- **HIGH**: 1 (Hazard 6.1 — FIX ER1: VC2 speedy path TRID ping-pong, FIXED)
- **MEDIUM**: 2 (Hazard 1.1 — teardown sync no escape, Hazard 1.2 — local copy fragility)
- **LOW**: 1 (Hazard 1.3 — pause/context switch timing)

### Fixes applied: 1
- **FIX ER1** (commit 731764a384d): Fixed TRID ping-pong in `fabric_erisc_router_speedy_path.hpp:260`. The expression `1 - current_write_trid` was replaced with `2 * trid_base + 1 - current_write_trid` to correctly toggle between base and base+1 TRIDs for any receiver channel, not just VC0.

### Clean files: 11 of 13
Files 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13 had no CRITICAL/HIGH/MEDIUM hazards.

### Architecture observations:
- The EDM fabric firmware is well-structured with clear single-owner access patterns per channel. Most potential race conditions are avoided by design: each ERISC core owns its sender/receiver channel state, with cross-core communication limited to volatile L1 locations and stream registers.
- The flow control credit system (both counter-based and stream-register-based) is correct — uint32_t subtraction with wraparound handles the credit math, and volatile/cache-invalidation is consistently applied at read sites.
- The main risk area is the dual-ERISC teardown synchronization (Hazard 1.1), which relies on both ERISCs reaching specific sync points. This is by-design and would require a hardware watchdog to improve.
- The speedy path optimization (File 6) was the only source of a real bug, due to the TRID base offset not being accounted for in the ping-pong expression.
