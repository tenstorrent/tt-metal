# activation_reader_width_sharded.cpp — MIGRATED v7 (quarantine LIFTED 2026-06-20)

- Group: G3 conv (WS sender, PRE_HANDSHAKE=false round-robin self-mcast)
- FINAL: status=migrated, migrated_api_version=7, commit=b9e23dafb11.
- Validation: full WS matrix `test_conv_features -k WIDTH_SHARDED` = **48 passed / 16 legit RM+bf8 skips / 0 fail**.

## RE-INVESTIGATION (2026-06-20) — the quarantine was a MIGRATION BUG, not a helper gap
The original Tier-0 subagent quarantined this as a HANG (same suspected class as block_sharded). On
re-investigation that conclusion is WRONG. A *correct* v7 migration PASSES:
- `SenderPipe<noc_index, act_mcast_receiver_sem(CTA13), num_reader_cores-1, /*PRE_HANDSHAKE=*/false>(noc, McastRect<>{...})`
- keep the raw readiness wait (`act_mcast_sender_sem.wait_min(num_mcast_cores-1)`) — PRE_HANDSHAKE=false
- `send(tilized_in0_cb.get_read_ptr(), act_cb.get_write_ptr(), size)` (loopback INCLUDE_SRC inferred)
- **REMOVE the raw flag `set(INVALID)` (was before the mcast) and `set(VALID)`** — the helper ctor owns the flag.

This reconstruction was run on device: PASS (control = no hang), stable across reruns, full WS matrix green.

### Why it works here but block_sharded genuinely doesn't (the real distinction)
conv-WS's raw receiver uses **clear-BEFORE-wait**: it sets the flag INVALID *before* `wait(VALID)` and
never clears after, so each receiver round *ends* with the cell at VALID (the sender delivered it). The
v7 `SenderPipe` ctor-once VALID is therefore preserved across rotating rounds — no stale-INVALID broadcast.
block_sharded uses the helper `ReceiverPipe`, which does **clear-AFTER-wait** (H11) → cell ends INVALID →
ctor-once VALID goes stale → hang (confirmed separately). Same self-mcast shape, opposite reset discipline.

### Likely original mistake
The subagent's v7 source was amended away (never committed cleanly), so the exact line is unrecoverable.
Most probable: it left the stale raw `act_mcast_receiver_sem.set(INVALID)` *before* `send()` while removing
the per-send `set(VALID)`, so the helper's flag mcast broadcast INVALID. Its "count-collapse" hypothesis
(num_reader_cores-1 vs num_mcast_cores-1) was a guess, now FALSIFIED on device (the working migration keeps
the raw num_mcast_cores-1 wait and the helper's num_reader_cores fan-out, and they coexist fine).

## Why quarantined
This kernel runs a mixed raw/helper round-robin self-mcast: each core is sender on its
own iteration (helper SenderPipe::send, INCLUDE_SRC loopback) and a RAW receiver on others
(act_mcast_receiver_sem.wait(VALID), act_mcast_sender_sem fan-in counter).

The v7 translation was API-correct (changelog R8 D1 order, PRE_HANDSHAKE=false, consumer omitted):
  SenderPipe<num_reader_cores-1, act_mcast_receiver_sem_id, act_mcast_sender_sem_id, Staging::Flag, false>(noc, McastRect{...})
  -> SenderPipe<noc_index, act_mcast_receiver_sem_id, num_reader_cores-1, /*PRE_HANDSHAKE=*/false>(noc, McastRect<>{...})

But it HANGS on device: RuntimeError TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:757
+ "FDMeshCommandQueue ... completion reader queue is not empty" (device hang). Triage report
(generated/tt-triage/triage.txt) captured no user-kernel callstack (teardown-time throw).

A/B diagnostic (decisive): swapping in the raw pre-helper kernel (git 77b3e62c8a1) made the SAME
WS test PASS (JIT 40/41 hits). So the regression is the helper migration, not the test/topology.

Likely cause: flag-mcast recipient-count mismatch. The pipe broadcasts the data-ready flag to
NUM_ACTIVE_RECEIVER_CORES = num_reader_cores-1, while the kernel's raw fan-in counter waits
num_mcast_cores-1 (num_mcast_cores = max(num_input_cores, num_output_cores)). When these differ,
some raw receivers never see VALID -> hang. The helper send()'s linked data+flag mcast and flush
fence also differ subtly from the raw INCLUDE_SRC mcast + set_multicast + wait(VALID) self-fence.

Action: reverted to raw primitives (un-migrated, matching its pending raw siblings) to keep the
tree green. Re-attempt requires reconciling the pipe's recipient count with the round-robin
counter semantics (a tune-dm-helper concern, not a Tier-0 remigration). diff_lines_removed: full
helper block replaced by raw.
