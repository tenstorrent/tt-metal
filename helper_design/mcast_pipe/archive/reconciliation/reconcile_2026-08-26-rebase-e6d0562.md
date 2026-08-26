# Reconcile — rebase onto `llk_helper_library` `e6d0562cfaa` (2026-08-26)

| | |
|---|---|
| Old baseline | `dc9282be7d5` (ledger `baseline_ref`, README line 9) |
| New baseline | `e6d0562cfaa` (`origin/llk_helper_library`) |
| Branch | `sjovic/mcast-migration-stable` → `sjovic/mcast-migration-stable-squashed` |
| Pre-rebase HEAD | `9d4cdf5328b` (kept on the original branch + `backup/mcast-migration-stable-2026-08-26-9d4cdf5328b`) |
| Shape | 88 commits squashed to 1, replayed as `767a0588c34` |
| Conflicts | 19 files resolved, 0 markers committed |
| Baseline move | 692 commits, 4071 files; 40 paths overlapped the migration |

## Helper divergence — resolved as "v14 wins" (Option A)

Both lines evolved the helper from v9: upstream to **v11** (refactored down, host split into
`host/mcast_host.cpp`, new `mcast_pipe_spec.hpp` / `mcast_spec.py` / `toy_spec_mcast` surface), the
branch to **v14**. The wire formats are incompatible — upstream 6-word CT / 4-word RT versus v14's
7-word CT (`present, has_receivers, data_ready, consumer_ready, ack_count, flags, rotating_span`,
plus a 1-word absent form) / 6-word RT, and `receive_signal()` lost its `round` parameter.

Decided on scope: **upstream had zero production ops migrated**, the branch has ~49 production
files. v14 therefore wins, and the blast radius is confined to upstream's test/example/spec layer.

Consequences recorded:

- `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.cpp` **deleted** and dropped from `ttnn/sources.cmake`;
  v14's header defines `Mcast1D`/`Mcast2D` inline. Upstream's deliberate header/impl split is
  therefore undone — re-splitting v14 is a `tune-dm-helper` decision, not a rebase edit.
- **OPEN:** `mcast_pipe_spec.hpp`, `ttnn/ttnn/mcast_spec.py` and `toy_spec_mcast` still speak the
  v11 wire (they read `num_active`, which v14 renamed `ack_count`). Neither the host build nor any
  other test exercises them, so they are silently stale. Port required.
- **OPEN:** upstream's 15 new gtest cases and 8 new pytest cases were **parked**, not merged — they
  reference `num_active` and cannot compile. v14 has every construct they exercise
  (`sender_grid`, `sender_placement` Uniform/Diagonal, `rotating_senders`, `Mcast1DShape`), so they
  are portable, mostly a rename. Recoverable verbatim from `e6d0562cfaa`.

## Units reverted to baseline

| unit | reason | blocker flag |
|---|---|---|
| `sort-single-row-control` (3 kernels + factory, 6 files) | upstream Metal-2.0-ported the op (#52528): positional args → `get_arg(args::name)`, factory now builds `SemaphoreSpec`/`TensorParameter`/`DFBBinding`. Re-migration is a re-authoring, not a merge. | `blocked:needs-metal2-named-args` |
| `argmax-multicore-control` (kernel + factory) | upstream added an end-of-kernel semaphore restore for trace replay; git spliced it onto the migrated kernel with no conflict, yielding a kernel that does not compile. The upstream defect applies to the migrated Counter form too. | `blocked:needs-pipe-semaphore-restore` |

Both are `deferred` in `ledger.json` / `test_map.json`, with full detail in
`migration/log/reader_argmax_interleaved_multicore.md` and the three sort logs.

## The finding that matters most

**Both defects introduced by this rebase came from files git auto-merged WITHOUT a conflict — not
from any of the 19 resolved conflicts.**

1. `reader_argmax_interleaved_multicore.cpp` — upstream's new tail referenced `start_sem` and eight
   rectangle-coordinate identifiers the migration had deleted. Hard failure (`ncrisc build failed`),
   caught immediately.
2. `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` — upstream's `compact_output` line arrived
   reading CT slot **32**, correct only in upstream's layout. The migration removed 4 mcast args, so
   every other index was shifted by −4 (`MtNt`=24, `in3_tensor_stride_w`=25, `fuse_op_all_gather`=26,
   `fuse_op_reduce_scatter`=27) but `compact_output` was not. **Silent wrong values, no compile
   error.** Only surfaced because `test_sparse_matmul_indexed.py::test_indices_absent_is_unchanged`
   asserts an inactive expert slot stays zero-filled. Fixed to slot 28; the host emits 29 scalars, so
   the accessor chain also moved 28 → 29.

Arithmetic that closes it: `28 (branch) + 1 (compact_output) + 4 (mcast args) = 33 (upstream)`.

**Action for the next reconcile:** 13 of the 15 behavior-sensitive auto-merged paths remain
unaudited. Counting host CT-arg emission against each kernel's hardcoded base is the check that
found defect 2; a clean auto-merge is not evidence of correctness.

## Submodules

Set to the baseline's pointers per explicit instruction: `tt_ops_code_gen` → `9b5edd63`,
`umd` → `0b263b2c`, `tt-cluster-descriptors` → `ac8a3b5b`. The branch's `tt_ops_code_gen` pointer
`4860704b721` was **orphaned** — absent from all refs with 0 reflog hits in the submodule clone, so
taking upstream's was the only option. (Same class of loss as the 2026-08-03 rebase.)

## Validation

| suite | result |
|---|---|
| `build_metal.sh` | 0 errors |
| `test_mcast_pipe.py` | 80/80 |
| `test_mcast_pipe_source_audit.py` | 31/31 (after removing 2 reverted-unit assertions, repairing 1 stale for #51637) |
| `test_mcast_topology.py` | 3/3 |
| `test_matmul.py` | 860 passed, 310 skipped, 2 xfailed |
| `test_sparse_matmul.py` | 27/27 |
| `test_sparse_matmul_indexed.py` | 18/18 |
| `test_argmax.py` | 68/68 (against the reverted baseline) |

Not run: `toy_spec_mcast` (needs the spec-layer port) and the `test_mcast_host` gtest binary.
Not covered: layernorm, topk, move, conv2d, conv3d, groupnorm, sdpa_decode migrated units.

**Nothing was pushed.**
