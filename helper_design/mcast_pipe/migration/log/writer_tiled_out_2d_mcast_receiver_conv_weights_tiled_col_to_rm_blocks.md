# writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp

Tier 0e (conv weights) — v7 -> v8 re-verify (Round 10, D2 count split).

## Transform
**None.** `ReceiverPipe` is unchanged in v8 (Round 10 only dropped `SenderPipe`'s 3rd template
arg `NUM_ACTIVE_RECEIVER_CORES`). This receiver-only kernel already spells
`ReceiverPipe<weights_mcast_receiver_sem_id, /*PRE_HANDSHAKE=*/true, weights_mcast_sender_sem_id>`,
which is the correct v8 form. No code edit.

## Diff
0 lines changed.

## Validation
Re-verified on the block-sharded weights-mcast path alongside the 2D sender (#1):
`test_conv_features -k BLOCK_SHARDED` -> **48 passed, 16 skipped, 0 failed**, no hang. Smoke
`... BFLOAT16/BFLOAT16 oc128/ic128 32x32 BLOCK_SHARDED ...` PCC 0.9999992.

## JIT-verification method + result
`grep -rl writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks generated/`
returned hits in `generated/inspector/{programs_log.yaml,kernels.yaml}` and
`generated/watcher/kernel_{names.txt,elf_paths.txt}` after the block-sharded runs — the receiver
kernel was JIT-compiled and launched.

## Ledger
status=migrated (unchanged), migrated_api_version 7 -> 8, existing migration commit kept,
last_verified=2026-06-20, coverage_confidence=high.
