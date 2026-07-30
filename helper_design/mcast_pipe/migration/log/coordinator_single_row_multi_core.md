# coordinator_single_row_multi_core.cpp — DEFERRED (design-gap)

Tier 3. Status: deferred. No code change (no migration attempted on device).

## Role
Sort coordinator = STAR sender over `coordinator_to_cores_sem` (FLAG-ONLY, no data
mcast). Per Ht row: a START broadcast then a per-substage broadcast; workers ack via
`cores_to_coordinator_sem`.

## Why deferred (three independent helper design gaps)
1. **Runtime recipient count.** The mcast recipient count `number_of_dest` is a runtime
   arg (`get_arg_val<uint32_t>(6)` = `core_range.num_cores()`). v7 `SenderPipe`'s
   `NUM_ACTIVE_RECEIVER_CORES` is a compile-time template param. Same gap as
   gn_v2 / welford / conv3d / flash_mla.
2. **Split mcast-dest vs ack count.** The START phase mcasts `number_of_dest` and waits
   `cores_to_coordinator.wait(number_of_dest)`; the per-substage phase mcasts
   `number_of_dest` but waits `cores_to_coordinator.wait(Wt/2)` (`number_of_confirmations`),
   which is NOT the recipient count. v7 `PRE_HANDSHAKE` ties the consumer-ready wait to
   `NUM_ACTIVE_RECEIVER_CORES`; it cannot serve two different wait counts. Same gap class
   as in0_sender_dram_sharded.
3. **Runtime sem ids.** Both sem ids arrive as runtime args (`get_arg_val(4)`/`(5)`) and
   the kernel builds `Semaphore<>(runtime_arg)`. v7 takes `DATA_READY_SEM_ID` /
   `CONSUMER_READY_SEM_ID` as compile-time template params. Same gap as
   group_attn_matmul. (Host-side the ids are constants 0/1, but moving them to
   compile-time args is a host change requiring a rebuild, out of this kernel-only tier —
   and gaps #1/#2 block migration regardless.)

The flag is also inverted-polarity (coordinator's cell stays host-init 0 and is broadcast
as `0`=GO; workers wait(0)), opposite to the helper's VALID(1)-ready convention — a
re-wire, not the binding blocker.

Helper untouched (per conventions). Lines removed: 0.
