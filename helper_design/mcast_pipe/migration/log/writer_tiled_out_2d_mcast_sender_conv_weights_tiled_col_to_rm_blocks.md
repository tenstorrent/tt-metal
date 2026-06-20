# writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp

Tier 0e (conv weights) — v7 -> v8 remigration (Round 10, D2 count split).

## Transform
DENSE site -> **pure deletion** of the 3rd `SenderPipe` template arg
(`NUM_ACTIVE_RECEIVER_CORES` = `weights_mcast_num_dests_ct`). Nothing added to the ctor.

### Dense vs divergent decision (read from the host factory)
`conv2d_op_sharded_program_factory.cpp`:
- The 2D weights sender is dispatched only on the **block-sharded** path.
- mcast rect (non-transpose): `(top_core.x, row1) .. (top_core.x, num_cores_y-1)` — a 1-wide
  strip of the receiver rows 1..N-1. Transpose: a 1-tall strip of cols 1..N-1.
- The sender core sits at logical row/col **0**, i.e. OUTSIDE the rect.
- Therefore `rect.area() == num_cores_y-1` (or `num_cores_x-1`), and since the sender is not in
  its own box, the EXCLUDE fan-out `= area - 0 = num_cores_y-1`.
- The old 3rd template arg (`block_sharded ? (transpose ? num_cores_x-1 : num_cores_y-1) : 0`,
  factory line ~1048) == that same `num_cores_y-1`/`num_cores_x-1`.
- ack count == EXCLUDE fan-out == rect area  =>  **DENSE**. Default `ACK_EQUALS_FANOUT`
  reproduces the dropped count exactly; no `consumer_ack_count` ctor arg needed.

(The 1D weights sender — a different kernel, `reader_writer_tiled_out_1d_mcast_sender_...` — is
the divergent one and is `deferred` in the ledger; it is NOT in this tier.)

## Diff
- ~3 lines removed (the `weights_mcast_num_dests_ct` template arg + reflow). Comment block
  rewritten to v8 (Round-4 narration -> v8 dense rationale).
- The `constexpr ... weights_mcast_num_dests_ct = get_compile_time_arg_val(...)` read is now
  unused but harmless (unused constexpr; host still pushes the CT arg; leaving it avoids any
  CT-index churn). Left in place.

## Validation
- Smoke (`--dev`): `test_conv_features[... BFLOAT16/BFLOAT16 oc128/ic128 32x32 BLOCK_SHARDED ...]`
  -> PASSED, PCC 0.9999992590338381 (threshold 0.997). No hang.
- Family (`--run-all -k BLOCK_SHARDED`): **48 passed, 16 skipped (RM+bf8b incompatible), 0 failed**,
  no hang.

## JIT-verification method + result
After the runs, `grep -rl writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks generated/`
returned hits in `generated/inspector/{programs_log.yaml,kernels.yaml}` and
`generated/watcher/kernel_{names.txt,elf_paths.txt}` — the sender kernel was JIT-compiled and
launched on the block-sharded path. The 2D receiver (#2) shows the same hits.

## Ledger
status=migrated, migrated_api_version=8, commit recorded, last_verified=2026-06-20,
coverage_confidence=high.
