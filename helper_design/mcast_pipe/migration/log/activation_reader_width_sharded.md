# activation_reader_width_sharded.cpp — QUARANTINED

- Group: G3 conv (WS sender, PRE_HANDSHAKE=false round-robin self-mcast)
- Commit: 9b5d9b5823f4be01adfc074b33091799791aa63a (revert to raw primitives)
- Status: QUARANTINED (status=quarantined, migrated_api_version=null)
- Validation nodeid: test_conv_features WIDTH_SHARDED output_channels=353 input_channels=384 input_height=8 input_width=8 (BFLOAT8_B/BFLOAT8_B HiFi4 fp32_accum=True)
- Result: v7 helper migration = HANG; raw pre-helper version = PASS

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
