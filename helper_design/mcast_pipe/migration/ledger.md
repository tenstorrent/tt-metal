# mcast_pipe migration ledger — API v10, Gate 0 verified 2026-08-06

Source of truth: `ledger.json`. Test inventories are in `test_map.json`;
failure isolation and JIT evidence are in the per-kernel migration logs.
Reconcile history is in `reconcile_<date>.md` (latest: `reconcile_2026-08-03-conv-width.md`).

Current baseline: `origin/llk_helper_library` @ `4a1d6a97ca9`; the reconciled production rollout
now runs through width-sharded Conv code commit `fe866a1d0c4`. The pre-rebase branch is preserved at
`backup/mcast-migration-prerebase-20260803`; every commit hash recorded in this
ledger was remapped from that line to its post-rebase equivalent on 2026-08-03.

The v8 July ledger is historical input, not the current result. Every one of
its 19 production migrations was re-audited against the current v10 helper.
Remediation also added the previously raw Conv 1D sender and both GroupNorm v2
senders; the subsequent sort and width-sharded Conv re-entries produce the
24-row production set. On 2026-08-06, six previously blocked rows moved to
pending for the three approved units, leaving 13 migrated, 6 pending, and 5
still concretely blocked.

## Host-helper re-entry state

The paired `mcast_host` helper and `McastArgs` decoder are materialized and
their 2026-08-06 intake tests are green: `McastHostFixture.*` passed 25/25 and
the complete device/wire helper suite passed 77/77 under `--dev`. Twelve
required bindings across six units are fully current at API v10:

| Unit | Required bindings | Status | Kernel rows | Validation |
|---|---:|---|---:|---|
| `conv2d-weights-single-sender-rect` | 1 | fully end-to-end migrated @ v10 | 2 | `CONV-HEIGHT`: 48 passed, 16 expected skips; DRAM-config 1 passed; shared DRAM: 14 passed |
| `conv2d-weights-fixed-line` | 1 | fully end-to-end migrated @ v10 | 2 | `CONV-BLOCK`: 48 passed, 16 expected skips; DRAM-config 1 passed; shared DRAM: 14 passed |
| `matmul-in1-mcast-padding-host` | 4 | fully end-to-end migrated @ v10 — **re-verified 2026-08-05** | 2 | `MM-IN1-ALL`: 302 passed, 188 expected skips, 490 selected |
| `groupnorm-sharded-v2-mcast-host` | 4 | fully end-to-end migrated @ v10 | 4 | mapped production geometry is zero-edge; synthetic splitter geometry 3/3; legacy perf +0.248%; Welford perf -0.485%; inventories 108/2 each; fixed/default 19/6 |
| `sort-single-row-control` | 1 | fully end-to-end migrated @ v10 | 2 migrated + 1 helper-neutral | handshaked row-start + no-handshake sub-stage Pipes; long 7/7; Ht=2 2/2; helper 77/77; perf +1.195124% |
| `conv2d-activation-width-sharded` | 1 | fully end-to-end migrated @ v10 | 1 hybrid | exact fresh-cache JIT at PCC 0.9999992598; features 48 passed / 16 expected skips; DRAM-config 1 passed; helper 73/73 |

Seven bindings across the next three units are pending:

| Unit | Required bindings | Kernel rows | Gate-A state |
|---|---:|---:|---|
| `topk-multicore-final-readiness` | 1 | 2 | API-008 resolves the old race with existing Counter mode; `TOPK-MULTICORE` coverage high |
| `layernorm-sharded-pre-allgather` | 1 | 2 | API-003 implemented; shared reader-argument builder split required before migration |
| `matmul-in0-mcast-interleaved` | 5 | 2 | API-007 materialization required; sparse binding device-verified from an empty cache with exact sender/receiver JIT artifacts |

The exact binding/dispatch map is in `test_map.json`; the easier-first atomic
order and risk gates are in `tiers.md`. Until a unit's required bindings are
migrated at API v10, its kernel rows are kernel-current but not fully
end-to-end current. The completed Conv2d units use code commits
`991b5b6b6386a90726d15007002fe1f5a77d8487` and
`51dfb1f1ed61045ed10dc679269960b6d2ccac9e`; matmul in1 uses
`aeeb28ff007807c71b1f60842cca85e5c41efa7f`; GroupNorm v2 uses
`bc24a55bf80a8ab2a4d702be2a91b827c1dcbeb0`; sort uses
`7337302b5649b7cd169764cd95c0b0343e88950d`; width-sharded Conv uses
`fe866a1d0c4c32b78aae8a76e875c0da109f51c8`.

## `needs_recheck` — CLOSED 2026-08-03 (0 open)

The 6 rows raised by `reconcile_2026-08-03.md` were cleared by an
`apply-dm-helper --mode=halt` verify-only pass at tree state `eb05b3929a3`.
**No rewrite was performed.** Details in `log/matmul-in1-mcast-padding-host.md`.

| Rows | Why they were flagged | Outcome |
|---|---|---|
| kernels `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`; host bindings `matmul-in1-mcast:reuse-{1d,2d}:{legacy,descriptor}` | The matmul mcast 1d/2d factories were churned upstream (`54d8dfb7bef`→`4a1d6a97ca9`, +203/−13 and +93/−4) touching `mm_in1_sender_writer_args`, then reworked again by `c946da17d29` + `eb05b3929a3`, which postdate the last ledger update (`62f82dd4a64`). | **PASS.** Static pre-check: kernels byte-identical to the pre-rebase verified state; `McastArgs` wire intact on both factories (sender CT idx 10–14, next = 15 = `KtNt`; sender RT idx 2–5; receiver CT idx 4–8; receiver RT idx 0–3; `MCAST_ARGS` at `2d:618`/`1d:1512`). Device: exact `--dev` 2D node PASSED with both kernels JIT-built at the current state; `MM-IN1-ALL` 302 passed / 188 expected skips / 490 selected — **exact baseline match**; `McastHostFixture` 19/19; `test_mcast_pipe.py` 68/68. Flag cleared, `last_verified` = 2026-08-03, `verified_at_commit` = `eb05b3929a3`. |

Each row's `commit` still points at its migration commit (`aeeb28ff007`) — that field's role is the
revert/bisect anchor; "last verified at" is `verified_at_commit`.

## Current migrations under API v10 (13)

| Area | Role | Kernel | Validation |
|---|---|---|---|
| matmul | sender/hybrid | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | fully end-to-end @ v10, **re-verified 2026-08-03 at `eb05b3929a3`**; `MM-IN1-ALL`: 302 passed, 188 expected skips, 490 selected; exact `--dev` 2D node with fresh JIT evidence |
| matmul | receiver | `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | fully end-to-end @ v10, **re-verified 2026-08-03 at `eb05b3929a3`**; `MM-IN1-ALL`: 302 passed, 188 expected skips, 490 selected; exact `--dev` 2D node with fresh JIT evidence |
| Conv | sender | `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | fully end-to-end @ v10; `CONV-HEIGHT`: 49 passed, 16 expected skips; 14 DRAM regressions; exact JIT path |
| Conv | receiver | `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | fully end-to-end @ v10; `CONV-HEIGHT`: 49 passed, 16 expected skips; 14 DRAM regressions; exact JIT path |
| Conv | sender | `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | fully end-to-end @ v10; `CONV-BLOCK`: 49 passed, 16 expected skips; 14 DRAM regressions; exact PerRow/PerColumn paths |
| Conv | receiver | `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | fully end-to-end @ v10; `CONV-BLOCK`: 49 passed, 16 expected skips; 14 DRAM regressions; exact PerRow/PerColumn paths |
| normalization | receiver | `reader_mcast_receiver_unary_sharded_gn_v2.cpp` | fully end-to-end @ v10; legacy inventory: 108 passed, 2 expected skips; exact JIT evidence |
| normalization | sender | `reader_mcast_sender_unary_sharded_gn_v2.cpp` | fully end-to-end @ v10; legacy inventory: 108 passed, 2 expected skips; fixed/default nodes: 19 passed, 6 expected skips |
| normalization | receiver | `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp` | fully end-to-end @ v10; Welford inventory: 108 passed, 2 expected skips; exact JIT evidence |
| normalization | sender | `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` | fully end-to-end @ v10; Welford inventory: 108 passed, 2 expected skips; fixed/default nodes: 19 passed, 6 expected skips |
| sort | sender | `coordinator_single_row_multi_core.cpp` | fully end-to-end @ v10; handshaked row-start + no-handshake sub-stage Counter `send_signal`; exact fresh-cache JIT; long 7/7; Ht=2 2/2 |
| sort | receiver | `reader_single_row_multi_core.cpp` | fully end-to-end @ v10; row-start readiness is Pipe-owned; separate no-handshake sub-stage `receive_signal`; long 7/7; Ht=2 2/2 |
| Conv | hybrid | `activation_reader_width_sharded.cpp` | fully end-to-end @ v10; rotating INCLUDE-source loopback; exact fresh-cache PCC 0.999956503; features 48/16 expected skips; DRAM-config 1/1; helper 72/72 |

The original ten kernel files remain byte-identical to the pre-rebase verified state and retain
`mcast_pipe.hpp` + `McastArgs`. The two sort faces were added atomically in `7337302b564`; the
width-sharded Conv hybrid followed atomically in `fe866a1d0c4`.

The paired `writer_single_row_multi_core.cpp` is deliberately not in this table: it has no Pipe
face. It remains a deferred/helper-neutral ledger row after coupled runtime-ABI cleanup.

## Historical deferred or rolled-back state before the 2026-08-06 re-entry

This table records the prior state. The six TopK, pre-allgather LayerNorm, and
interleaved Matmul in0 rows are now pending as described above; the other rows
remain deferred. No row is a partial migration.

| Area | Kernel(s) | Blocker |
|---|---|---|
| matmul | `reader_bmm_tile_layout_in0_sender_padding.cpp`, `reader_bmm_tile_layout_in0_receiver.cpp` | typed/custom `VALID` / `IGNORE_BATCH` control values |
| matmul | `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | independent data/signal loopback modes and sender-side ready participation |
| normalization | post-allgather LayerNorm sender/receiver | explicit INCLUDE-source loopback when the sender is outside the receiver rectangle, plus host fan-out that may differ from rectangle area; pair restored together |
| normalization | pre-allgather LayerNorm sender/receiver | acknowledged signal-only channel |
| normalization | plain sharded LayerNorm sender/receiver | acknowledged signal-only plus one-gate/multi-block mixed-mode streaming |
| TopK | `reader_final_topk.cpp`, `writer_local_topk.cpp` | race-free no-handshake receiver initialization |

## Fleet totals

The census now contains 91 entries (92 before the 2026-08-03 reconcile; the
deepseek_prefill `reader_dispatch.cpp` row was deleted after the kernel was
removed upstream by `af00262e51d`):

- 13 current production migrations at API v10, **0 carrying `needs_recheck`**;
- all 12 required host bindings current across 6 atomic units, **0 carrying
  `needs_recheck`**;
- 11 production candidates deferred with `v9-port-blocked` and a concrete
  design-gap flag;
- 67 other deferred candidates, including the helper-neutral sort writer;
- 0 kernel rows pending and 0 quarantined.

The dated v8 reconcile reports and per-kernel logs are retained as historical
evidence. `reconcile_2026-08-03-conv-width.md`, this ledger, and `ledger.json` are
authoritative for the current branch; `reconcile_2026-07-29.md` and earlier are
historical.
