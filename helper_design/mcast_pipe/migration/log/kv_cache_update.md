# kv_cache_update.hpp — DEFERRED (design-gap)

Kernel: `models/demos/deepseek_v3_b1/unified_kernels/kv_cache_update.hpp`
Tier: 2d. Status: deferred (design-gap). No code change.

## Why
The NopeSender mcast block (lines 150–221) is built entirely on the
**preprogram-state** primitive family from `dataflow_utils.hpp`:
- `unified_kernels::noc_async_write_multicast_preprogram_all_state<...>(...)` — programs the
  cmd buf state for the data mcast AND (separately) the semaphore mcast.
- `unified_kernels::noc_async_write_multicast_issue_txn<...>(...)` / `multicast_write_with_state<...>` —
  fires the pre-programmed transaction.
- A `mcast_is_shared_write_cmd_buf` compile-time branch picks between a shared-cmd-buf
  `multicast_write_with_state` path and a separate `write_reg_cmd_buf` path.

The set-state is issued BEFORE `cb_wait_front(kv_rmsnorm_output_cb)`, so the cmd-buf programming
overlaps with the producer wait — that overlap is the explicit perf intent.

The v7 `SenderPipe::send()` issues a single fused `Noc::async_write_multicast` (data) +
`Semaphore::set_multicast` (flag) per call. There is no set-state / issue-txn split and no
shared-cmd-buf knob, so the migration would (a) be inexpressible without an issue-txn face on the
Pipe and (b) drop the preprogram-state overlap that motivates these unified_kernels.

`proposed_helpers.md` lists this exact case under Defer/out-of-scope:
> "Preprogram-state perf optimization (deepseek mcast.hpp) — no mcast set-state in object API; future."

## Verdict
DESIGN-GAP. Helper would need a set-state/issue-txn (preprogram-state) mode. Per task rules, do NOT
touch the helper. Deferred. (The including kernel `kv_cache_update_kernel.cpp` / fused
attention/decoder/pre_sdpa kernels JIT-build it; left untouched.)
