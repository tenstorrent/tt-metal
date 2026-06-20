# reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp

Tier 0e (conv weights) — v7 -> v8 re-verify (Round 10, D2 count split).

## Transform
**None.** `ReceiverPipe` is unchanged in v8. This receiver-only kernel already spells
`ReceiverPipe<weights_mcast_receiver_sem_id, /*PRE_HANDSHAKE=*/true, weights_mcast_sender_sem_id>`
(the correct v8 form). No code edit.

Note: the *sender* partner of this 1D path
(`reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`) is a
DIFFERENT kernel, ledger status `deferred` (design-gap: split mcast-dest vs consumer-ack count).
It is not in this tier. The v8 `consumer_ack_count` ctor arg may now unblock that deferred entry,
but that is out of scope here.

## Diff
0 lines changed.

## Validation
Smoke (`--dev`): `test_conv_features[... BFLOAT16/BFLOAT16 oc16/ic16 256x256 HEIGHT_SHARDED
config={'act_block_h': 32} ...]` -> PASSED, no hang.

## JIT-verification method + result
`grep -rl reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks generated/`
returned hits in `generated/inspector/{programs_log.yaml,kernels.yaml}` and
`generated/watcher/kernel_{names.txt,elf_paths.txt}` after the HEIGHT_SHARDED 256x256
act_block_h=32 run — the receiver kernel was JIT-compiled and launched on that path.

## Ledger
status=migrated (unchanged), migrated_api_version 7 -> 8, existing migration commit kept,
last_verified=2026-06-20, coverage_confidence=high.
