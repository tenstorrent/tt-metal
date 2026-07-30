# conv2d-weights-fixed-line

- Kind: host integration
- Helper API: `MCAST_PIPE_API_VERSION=9`
- Binding: `weights-mcast:conv2d-sharded:fixed-line`
- Status: migrated, fully end-to-end
- Code commit: `261e322ed2284175e3b4b7b80f98e947b569fe10`
- Verified: 2026-07-30

## Atomic scope

- `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp`
- `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp`
- the block-sharded fixed-line weights branch in
  `conv2d_op_sharded_program_factory.cpp`

The factory now constructs one host `Mcast1D` over `output_cores`: `PerRow`
when `transpose_mcast` is true and `PerColumn` otherwise. It adopts the
existing receiver/sender semaphore IDs and preferred writer NoC. Both kernels
consume the helper's five-word CT/four-word RT wire through `McastArgs`.

The output parallel configuration produces one dense, zero-anchored rectangle,
and the factory asserts both properties before constructing the helper.
Multicast participants are the output cores. Offset input-only split-reader
cores retain `skip_work` behavior and do not participate. The separate
split-reader semaphores, `is_sender_core`, `SKIP_MCAST`, pad-out arguments,
weight/bias buffer bindings, and NoC selection remain unchanged.

## Validation

- `./build_metal.sh`: passed.
- Exact PerColumn block-sharded node under
  `scripts/run_safe_pytest.sh --dev`: passed on retry, PCC
  `0.9998128517571359` against threshold `0.985`.
- Exact PerRow block-sharded node under
  `scripts/run_safe_pytest.sh --dev`: passed, PCC
  `0.9998128517571359` against threshold `0.985`.
- `test_conv_features -k BLOCK_SHARDED`: 48 passed, 16 expected skips.
- `test_conv_dram_config -k BLOCK_SHARDED`: 1 passed.
- Shared `test_conv_dram`: 14 passed.
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed.

The first PerColumn smoke run hung because the sender read its trailing
`is_sender_core` and `skip_work` values before advancing across the helper RT
block. Advancing to `McastArgs::next_runtime_args_offset()` fixed the decoder;
the retry and every subsequent regression passed.

## Diff and coverage

- Production diff: 40 insertions, 96 deletions (net 56 lines removed).
- Coverage gap: no.
- Result: PASS; no rollback or quarantine required.
