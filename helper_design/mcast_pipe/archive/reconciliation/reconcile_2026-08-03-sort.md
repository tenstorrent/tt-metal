# Archived: mcast_pipe reconcile — sort migration, 2026-08-03

## Scope

- Prior authoritative rollout commit: `74a3e672551`.
- Reconciled production code commit: `7337302b564`.
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`.
- Helper API: `MCAST_PIPE_API_VERSION=9`, unchanged.

## Census and ledger audit

- All 91 kernel paths in `ledger.json` exist in the current tree.
- No kernel was added, removed, or renamed by this focused change.
- The sort coordinator and reader moved from pending/refactor to migrated at API v9.
- The sort writer remains deferred/helper-neutral: its done-counter `up` is operation protocol, not
  a Pipe face. Its coupled runtime-ABI cleanup is recorded without inflating the migration count.
- Added required host binding `sort:single-row-multi-core:control` at the same production commit.
- Resulting totals: 12 migrated kernels, 79 deferred kernels, 0 pending, 0 quarantined; 11 migrated
  host bindings; 0 open `needs_recheck` flags.

## Recall sweep

The diff from `74a3e672551` through `7337302b564` introduces no raw multicast primitive callsite.
It introduces only the expected helper calls:

- coordinator: `McastArgs` + `send_signal()`;
- reader: `McastArgs` + `receive_signal()`;
- focused helper-test sender/receiver kernels: the same Counter signal-only surface.

The full sort kernel directory was searched. No untracked multicast emitter or new spelling was
found. The raw reader-ready and writer-done atomic operations are intentionally retained and are
documented as operation-owned return channels.

## Validation write-back

- `./build_metal.sh`: passed.
- Exact `test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]` under `--dev`, fresh
  isolated JIT cache: passed with coordinator, reader, and writer artifacts present.
- `test_sort_multi_row_multi_core_no_deadlock`: 2 passed.
- Complete `test_sort_long_tensor`: 7 passed.
- Complete `test_mcast_pipe.py`: 72 passed after production integration.

No API bump, quarantine, rollback, or follow-up `needs_recheck` flag is required.
