# conv2d-weights-single-sender-rect

- Kind: host integration
- Helper API: `MCAST_PIPE_API_VERSION=9`
- Binding: `weights-mcast:conv2d-sharded:single-sender-rect`
- Status: migrated, fully end-to-end
- Code commit: `75b977e1a04ee7a14df5d8039393c7844f33fdae`
- Verified: 2026-07-30

## Atomic scope

- `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
- `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`
- the height-sharded/default 1D weights branch in
  `conv2d_op_sharded_program_factory.cpp`

The factory now constructs one host `Mcast2D` over `all_cores`, with sender
`(0,0)`, the preferred writer NoC, the existing factory-owned semaphore IDs,
and `num_active=total_active_num_cores-1`. The helper therefore owns the full
rectangle geometry and five-word CT/four-word RT wire while preserving the
smaller active receiver ACK subset. Both kernels consume that wire through the
matching `McastArgs` decoder.

The separate block-sharded fixed-line branch retains its two-word semaphore ABI
and manual line geometry. The `SKIP_MCAST` path still creates no receiver
kernel, executes no sender pipe, and uses the helper's inactive single-core
wire. Split-reader state, activation-reuse offsets, and weight/bias buffer
bindings are preserved.

## Validation

- `./build_metal.sh`: passed.
- Exact compile-focused height-sharded node under
  `scripts/run_safe_pytest.sh --dev`: 1 passed, PCC
  `0.9999993139704398` against threshold `0.997`.
- Isolated `TT_METAL_CACHE` JIT proof:
  - sender hash `7585446677319138702`;
  - receiver hash `15099792235267128693`.
- `test_conv_features -k HEIGHT_SHARDED`: 48 passed, 16 expected skips.
- `test_conv_dram_config -k HEIGHT_SHARDED`: 1 passed.
- Shared `test_conv_dram`: 14 passed.
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed, including split fan-out/ACK-count cases.

## Diff and coverage

- Production diff: 48 insertions, 51 deletions (net 3 lines removed).
- Coverage gap: no.
- Result: PASS; no rollback or quarantine required.
