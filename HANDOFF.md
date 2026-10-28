# Handoff: GLM sparse KV push (drop this commit before the PR)

Branch `ipotkonjak/sparse-kv-push-subdevice` (local only on `bh-glx-120-c02u02`), 3 commits on main `8d7478b591c`:

1. `[GLM] ttnn.experimental.sparse_kv_push and sparse_push_plan` (ops, kernels, nanobind, op unit test, plan test)
2. `[GLM] sparse_sdpa: run on a core subset` (`sub_core_grids`, default = first sub-device; + nightly
   `test_sparse_sdpa_kv_push.py`)
3. `[GLM] Sparse KV push for the shared layers of sparse MLA prefill` (`mla.py`, `tt/mla/sparse_kv_push.py`
   (`SparseKvPushPlanner`), `tt_ccl.py`, `SPARSE_KV_PUSH.md`, `test_sparse_mla.py`)

`ipotkonjak/sparse-kv-push-rebased` (pushed, tracked by a colleague) still holds the previous, fused version: the
prefix push ran inside sparse_sdpa's program (`SparseSDPAPushFusedFactory`). It was replaced by sub-devices to keep
the sparse_sdpa change small (the sdpa owners review `sparse_kv_push/` anyway, per CODEOWNERS). Warn before
force-pushing over it.

Design overview: `models/demos/deepseek_v3_d_p/tt/mla/SPARSE_KV_PUSH.md`.

## Status

- Push is the only behaviour on `FABRIC_2D_TORUS_XY` meshes >= 3x3 (`sparse_kv_push_supported`); other meshes keep the
  dense gather. No flag.
- A layer whose successor is shared loads `SparseKvPushPlanner.prefetch_manager` (attention cols 0-7 = sub-device 0,
  push cols 8-11 rows 4-9 = sub-device 1), enqueues `sparse_kv_push(part=1)` then `sparse_sdpa`, and clears the manager
  (the join). Under trace the load/clear go through the `SubDeviceTraceController`.
- Bitwise equal to the dense gather at op, MLA and full-model level.
- Op unit test `tests/ttnn/unit_tests/operations/experimental/test_sparse_kv_push.py`: 54 cases pass, now including
  part 1 alone and part 1 + part 2 == part 0. **Not in CI**: adding it (or any CI yaml / perf baseline change) needs
  Iva's OK.
- Pre-commit passes on this machine. Fixed for it: `expect_error` instead of `pytest.raises` in the reject test;
  `sparse_push::validate` renamed `validate_config` (the legacy-device-op hook flags any `validate(` in a header).
- Review fixes 1-14 done. Nits 15-18 deferred to after the PR (17: de Bruijn `lowest_bit()` in the reader vs
  `__builtin_ctz` in the plan kernel).
- Known main breakages are not ours and not to-dos: `test_prefill_transformer_chunked` import of
  `sparse_mla_overlap_gather_rows`, rows2 cases in `test_sparse_mla_cache`, full55k untraced PCC 0.876 (same on the
  dense path).

## Measured (1 BH Galaxy)

1M SC1 e2e (`run_multirank_pcc.sh glm53 sc1`), PCC gate passes; the perf gate fails only for being faster than the
checked-in baseline (update after a CI run, not by hand). Dense column from `c02u08`; fused and sub-devices on
`c02u02` (fused gave the same numbers on both machines).

| | dense gather | push, fused | push, sub-devices |
|---|---|---|---|
| TTFT | 128.9 s | 109.12 s | 109.43 s |
| chunk @1020k | 834 ms | 617 ms | 619 ms |
| throughput @1020k | 6,143 tok/s | 8,284 tok/s | 8,259 tok/s |
| 55k fit (ms) | 419.8 + 2.19/chunk | 423.1 + 1.21/chunk (break-even ~20k) | not re-measured |

Op level, real 1M top-k, attention + next layer's prefix push, us per traced replay (scratch
`tests/nightly/blackhole/sdpa/scratch_push_overlap_perf.py`, untracked, not for the PR):

| | attn 80 cores | prefix push alone | serial | fused | sub-devices |
|---|---|---|---|---|---|
| L14 | 893 | 427 | 1404 | 943 | 947 |
| L58 | 827 | 730 | 1661 | 979 | 990 |
| L62 | 837 | 1448 | 2349 | 1547 | 1568 |

Extra memory: ~220 MiB per chip (push buffer capped at min(cache rows, query rows x top-k) + N x unit; the dense
gather buffer is reused as the other parity).

Grid: attention cols 0-7 (80 cores; attention time steps with ceil(160/cores), more cores only add DRAM contention),
push 24 cores (links x 4 dirs x (2 senders + 1 mux)) on cols 8-11 rows 4-9, away from the Ethernet row.

## Next steps, in order

1. **Readability pass with Iva, file by file, before the draft PR** (propose, agree, then edit; gate every kernel
   change on the 54-case unit test being bit-exact and L14/L58/L62 op timing within noise). Order:
   1. `device/kernels/sparse_push_table_reader.cpp` - proposal agreed:
      - `compact_page(run_base, rank)` as the one named home of the slot formula
        `base + (j % banks) + (j / banks) * unit + i * banks`, shared with the plan kernel via `sparse_push_constants.hpp`;
      - helpers `serves(requester)`, `run_base(counts, owner)`, `owned_blocks(header, part)`, `read_bitmap_window(...)`;
      - a `PacketBuilder` struct replacing the `open_packet`/`close_packet` lambdas and their 6 loose variables;
      - real names (`jb`->`block`, `lane0`->`first_lane`, `take`->`owns_packet`, `sub`->`sender_index`,
        `split_j`->`prefix_blocks`, ...), `kDramReadAlign = 64`, one-line whys for the sender round-robin
        (both senders walk the bitmap, each reads only packets whose turn matches its index) and the block-granular split.
   2. `sparse_push_writer.cpp` (incl. the mux args read by bare offsets `arg + 1` .. `arg + 16`)
   3. plan kernels: `sparse_push_plan.cpp`, `sparse_push_plan_work.hpp`, `sparse_push_plan_compute.cpp` (explain the
      5-processor split; the aliased step constants `kStepScatter = kBucketed`, ...)
   4. `sparse_push_route.hpp`, `sparse_push_constants.hpp`
   5. host: `sparse_push_program.cpp` (its `append` / `Kernels` builder split existed for the fused factory; may now
      fold into the program factory), device ops, factories
   6. Python: `sparse_kv_push.py`, `mla.py` wiring
2. Drop this HANDOFF commit, then **draft PR** (Iva opens it). Perf baselines updated after a CI run, not by hand.
3. Cleanup on the old machine (`bh-glx-120-c02u08`): worktrees under `.claude/worktrees/` (push-core-sweep,
   push-fused-profile, sparse-kv-push-op with uncommitted experiments), local-only branches
   `-presquash`, `push-core-sweep`, `push-fused-profile`, superseded `sparse-kv-push` / `sparse-kv-push-op`.
4. After the PR: restore push resume (progress tensor, quit semaphore, part 3) as op args to hide the leftover push
   behind MoE/FFN and the next layer's q/kv (heavy layers like L62 push ~1.45 ms at 1M; resume was groundwork, not
   dead code); run kv before q; optional data-based buffer cap; nits 15-18; maybe slim the nightly 1M test.

## How to run

- Device tests through `scripts/run_safe_pytest.sh` (device lock).
- Op test: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_sparse_kv_push.py`
- MLA: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k kv_push`
- Nightly: `scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_sparse_sdpa_kv_push.py -k real_topk`
- With `PREFILL_TORUS_XY_CERTIFIED=1`, also export `TT_MESH_GRAPH_DESC_PATH` to the single_bh_galaxy_torus_xy descriptor.
- 1M e2e on `c02u02`: no `/etc/ttop`, so `TTRUN_DIR` points at a dir whose `hostfile` holds `bh-glx-120-c02u02`; also
  `PREFILL_SUMMARIES` (a path containing `prefill_summaries`) and `PREFILL_KEEP_RUN_DIR=1`, under
  `flock /tmp/tt-device.lock` (the lock `run_safe_pytest.sh` uses). Takes ~8 min warm.
- Git has no identity on `c02u02`: commit with `GIT_AUTHOR_*` / `GIT_COMMITTER_*` set to Iva Potkonjak
  <ipotkonjak@tenstorrent.com>.
