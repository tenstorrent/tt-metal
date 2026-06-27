# activation_reader_width_sharded.cpp — MIGRATED v8 DUAL-PIPE (2026-06-20)

- Group: G3 conv (WS round-robin self-mcast). Role: **hybrid** (SenderPipe + ReceiverPipe).
- FINAL: status=migrated, migrated_api_version=8, role=hybrid.
- Validation: full WS matrix `test_conv_features -k WIDTH_SHARDED` = **48 passed / 16 legit RM+bf8 skips / 0 fail**.
  Smoke (bf16/bf16, filter=3, fp32_accum=False, packer_l1_acc=True) under `--dev`: PASS, PCC 0.9999565, no hang.
  JIT-verified: `grep -rl activation_reader_width_sharded generated/` hit (watcher kernel_names + kernel_elf_paths).

## What changed (v7 sender-only + raw receiver  ->  v8 dual-pipe)

This was the v8-unblocked NET-NEW upgrade. The prior committed form (v7, commit b9e23dafb11) used
`SenderPipe<..., num_reader_cores-1, PRE_HANDSHAKE=false>` for the broadcast face only, and a RAW receiver
(`act_mcast_receiver_sem.wait(VALID)` + raw `act_mcast_sender_sem` fan-in counter / `up`). It used the pipe
"partially" — sender face only. v8 lets the divergent counts coexist, so both faces now use the helper.

### Final SenderPipe spelling (built once before the round-robin loop)
```cpp
dataflow_kernel_lib::SenderPipe<
    noc_index,
    get_compile_time_arg_val(13),     // DATA_READY_SEM_ID  = act_mcast_receiver_sem (VALID/INVALID flag)
    /*PRE_HANDSHAKE=*/true,
    get_compile_time_arg_val(12)>     // CONSUMER_READY_SEM_ID = act_mcast_sender_sem (fan-in counter)
    act_send_pipe(
        noc,
        dataflow_kernel_lib::McastRect<>{
            mcast_rect.noc_x_start, mcast_rect.noc_y_start, mcast_rect.noc_x_end, mcast_rect.noc_y_end},
        /*consumer_ack_count=*/num_mcast_cores - 1);
```
`send(tilized_in0_cb.get_read_ptr(), act_cb.get_write_ptr(), act_mcast_sender_size_bytes)` (INCLUDE_SRC
loopback inferred: in-rect && src!=dst). The raw `act_mcast_sender_sem.wait_min(num_mcast_cores-1)` + `set(0)`
ack-wait is now FOLDED into `send()`'s PRE_HANDSHAKE path. The raw pre-loop `act_mcast_receiver_sem.set(VALID)`
is GONE (the ctor owns it; `send()` re-asserts VALID per call — the M12b fix).

### Final ReceiverPipe spelling (built per round inside the loop, in the receiver branch)
```cpp
dataflow_kernel_lib::ReceiverPipe<
    get_compile_time_arg_val(13),     // DATA_READY_SEM_ID
    /*PRE_HANDSHAKE=*/true,
    get_compile_time_arg_val(12)>     // CONSUMER_READY_SEM_ID
    act_recv_pipe(noc);
act_recv_pipe.receive(sender_x, sender_y);   // sender coords from x/y lookup tables
```
`.receive()` folds BOTH the raw fan-in ack (`act_mcast_sender_sem.up(noc, sender_x, sender_y, 1)`) AND the
raw `act_mcast_receiver_sem.wait(VALID)`. The raw top-of-round `act_mcast_receiver_sem.set(INVALID)` is KEPT
right before constructing the ReceiverPipe — it also clears the stale VALID this core's own sender round left
behind (harmlessly redundant with the ctor's own set(INVALID)).

## The consumer_ack_count divergence (THE reason v8 unblocked this kernel)
The mcast fan-out is derived from the rect's `area()` = `num_reader_cores` (the broadcast lands on every reader
core). But only `num_mcast_cores - 1` cores actually ack (`num_mcast_cores = max(num_input_cores, num_output_cores)`;
the round-robin loop runs over `num_input_cores` and the sender doesn't ack itself). At v7 the SenderPipe could
express only ONE count for both the fan-out and the ack-wait (the dropped 3rd template param), so the divergent
case had no correct spelling -> the original hang. v8's runtime `consumer_ack_count` ctor arg decouples them:
fan-out stays rect-derived (`num_reader_cores`), ack-wait is the explicit `num_mcast_cores - 1`.

## Why the quarantine-era hang does NOT recur (the clear-after-wait hazard)
ReceiverPipe uses **clear-after-wait** (H11): the receive cell ends INVALID. A ctor-once-VALID sender would go
stale across rotating rounds and hang (the block_sharded failure). Round 9's M12b fix — `send()` re-asserts the
source cell VALID every call — removes this: when this core becomes the sender, its `send()` refreshes VALID
before broadcasting, so the stale INVALID its own receive rounds left behind never reaches the receivers. Proven
on device here (stable across the full WS matrix + the `--dev` watcher smoke, no hang).

## Diff
Replaced the v7 sender-only block + raw receiver branch (raw `wait_min`/`set(0)`/`up`/`wait(VALID)` plus the
3rd-template-param v7 SenderPipe) with the v8 dual-pipe form above. Net lines removed (raw primitives): ~6.

## Best sibling reference
`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
(commit 5a9d07277df) — the rotating-role self-mcast that uses BOTH faces; copied its spelling/discipline. Its
factory makes fan-out == ack (dense, no explicit count); conv-WS is the divergent sibling that needs the explicit
`consumer_ack_count`.
