# Archived: mcast_pipe reconcile — width-sharded Conv migration, 2026-08-03

## Scope

- Approved Tier-6 plan: `e712d3ab4bb`.
- Reconciled production code commit: `fe866a1d0c4`.
- Apply ledger/log/report commit: `30927931918`.
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`.
- Helper API: `MCAST_PIPE_API_VERSION=9`, unchanged.

## Census and ledger audit

- All 91 kernel paths in `ledger.json` exist in the current tree.
- `census.txt` and `ledger.json` contain the same 91 paths, with no one-sided entry.
- No kernel was added, removed, or renamed by this focused change.
- The width-sharded Conv activation reader moved from pending/refactor to migrated at API v9.
- Required host binding `activation-mcast:conv2d-width-sharded:rotating-rect` moved with the kernel
  and factory as one atomic unit.
- All 13 migrated kernels still include `mcast_pipe.hpp` and construct `McastArgs`.
- Resulting totals: 13 migrated kernels, 78 deferred kernels, 0 pending, 0 quarantined; 12 migrated
  host bindings; 0 open `needs_recheck` flags.

## Recall sweep

The diff from `e712d3ab4bb` through `fe866a1d0c4` introduces no raw multicast primitive callsite.
It removes the width-sharded reader's explicit `async_write_multicast`, semaphore flag/counter
sequence, multicast endpoint, and physical-coordinate lookup arrays, replacing them with the
expected helper surfaces:

- factory: rotating, handshaked Flag `Mcast2D` with adopted semaphore IDs and a distinct active ACK
  count;
- kernel: `McastArgs<12,3,num_input_cores>`, `SenderPipe::send()`, and
  `ReceiverPipe::receive(round)`.

The Conv kernel directories were searched with the recognition family from
`primitive_contracts.md`. Every production invocation is already represented in the census.
`conv_reader_common.hpp` remains a declaration-only hit (`McastRect` / `McastDst`), not a primitive
invocation. No new annotation row or deferred candidate is required.

## Validation write-back

- `./build_metal.sh`: passed.
- Exact BF16/BF16 filter-3 TILE-output node under `--dev` from a fresh isolated cache: passed at PCC
  `0.999956503`; `activation_reader_width_sharded` JIT artifact confirmed.
- Complete `test_conv_features and WIDTH_SHARDED`: 48 passed, 16 legitimate row-major+bfloat8
  skips, 0 failed.
- `test_conv_dram_config and WIDTH_SHARDED`: 1 passed at PCC `0.998234911`; current JIT route
  confirmed.
- Complete `test_mcast_pipe.py`: 72 passed after production integration.

The current ACK-fenced completion rule for a real loopback copy resolves the earlier port's 25
numerical failures. No API bump, quarantine, rollback, or follow-up `needs_recheck` flag is required.
