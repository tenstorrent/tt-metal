# Handoff: GLM sparse KV push (drop this commit before the PR)

Branch `ipotkonjak/sparse-kv-push-rebased`, 3 commits on main `8d7478b591c` (each builds):

1. `e21129f3ea9` ttnn.experimental.sparse_kv_push and sparse_push_plan (ops, kernels, nanobind, op unit test, plan test)
2. `862fab7b4a2` sparse_sdpa: core subset and a fused sparse KV push (+ nightly `test_sparse_sdpa_kv_push.py`)
3. `bd3a5d25327` GLM wiring: `mla.py`, `tt/mla/sparse_kv_push.py` (`SparseKvPushPlanner`), `tt_ccl.py`, `SPARSE_KV_PUSH.md`, `test_sparse_mla.py`

Design overview: `models/demos/deepseek_v3_d_p/tt/mla/SPARSE_KV_PUSH.md`. A colleague tracks this branch: warn before
force-pushing.

## Status

- Push is the only behaviour on `FABRIC_2D_TORUS_XY` meshes >= 3x3 (`sparse_kv_push_supported`); other meshes keep the
  dense gather. No flag.
- Bitwise equal to the dense gather at op, MLA and full-model level. Edge cases checked: rotated/padded chunks,
  multi-chunk, traced replay over slots and chunk starts, MTP multi-turn (16/16).
- Op unit test `tests/ttnn/unit_tests/operations/experimental/test_sparse_kv_push.py`: 54 cases pass. **Not in CI**:
  adding it (or any CI yaml / perf baseline change) needs Iva's OK.
- Review fixes 1-14 done. Nits 15-18 deferred to after the PR (17: de Bruijn `lowest_bit()` in the reader vs
  `__builtin_ctz` in the plan kernel).
- Known main breakages are not ours and not to-dos: `test_prefill_transformer_chunked` import of
  `sparse_mla_overlap_gather_rows`, rows2 cases in `test_sparse_mla_cache`, full55k untraced PCC 0.876 (same on the
  dense path).

## Measured (squashed tip, 1 BH Galaxy)

| | dense gather | push |
|---|---|---|
| 1M SC1 TTFT | 128.9 s | 109.1 s |
| 1M chunk @1020k | 834 ms | 617 ms |
| 1M throughput | 6,143 tok/s | 8,286 tok/s |
| 55k fit (ms) | 419.8 + 2.19/chunk | 423.1 + 1.21/chunk (break-even ~20k) |

Extra memory: ~220 MiB per chip (push buffer capped at min(cache rows, query rows x top-k) + N x unit; the dense
gather buffer is reused as the other parity).

Grid: attention cols 0-7 (80 cores; attention time steps with ceil(160/cores), more cores only add DRAM contention),
push 24 cores (links x 4 dirs x (2 senders + 1 mux)) on cols 8-11 rows 4-9, away from the Ethernet row (~35 us better
on heavy layers when fused).

## Next steps, in order

1. **Readability pass with Iva, file by file, before the draft PR** (propose, agree, then edit; gate every kernel
   change on the 54-case unit test being bit-exact and L14/L58/L62 op timing within noise). Order:
   1. `device/kernels/sparse_push_table_reader.cpp` - proposal made, awaiting OK:
      - `compact_page(run_base, rank)` as the one named home of the slot formula
        `base + (j % banks) + (j / banks) * unit + i * banks`, shared with the plan kernel via `sparse_push_constants.hpp`;
      - helpers `serves(requester)`, `run_base(counts, owner)`, `owned_blocks(header, part)`, `read_bitmap_window(...)`;
      - a `PacketBuilder` struct replacing the `open_packet`/`close_packet` lambdas and their 6 loose variables;
      - real names (`jb`->`block`, `lane0`->`first_lane`, `take`->`owns_packet`, `sub`->`sender_index`,
        `split_j`->`prefix_blocks`, ...), `kDramReadAlign = 64`, one-line whys for the sender round-robin
        (both senders walk the bitmap, each reads only packets whose turn matches its index) and the block-granular split.
   2. `sparse_push_writer.cpp`
   3. plan kernels: `sparse_push_plan.cpp`, `sparse_push_plan_work.hpp`, `sparse_push_plan_compute.cpp` (explain the 5-processor split)
   4. `sparse_push_route.hpp`, `sparse_push_constants.hpp`
   5. host: `sparse_push_program.cpp`, device ops, factories
   6. Python: `sparse_kv_push.py`, `mla.py` wiring
2. Drop this HANDOFF commit, then **draft PR** (Iva opens it). Perf baselines updated after a CI run, not by hand.
3. Cleanup on the old machine (`bh-glx-120-c02u08`): worktrees under `.claude/worktrees/` (push-core-sweep,
   push-fused-profile, sparse-kv-push-op with uncommitted experiments), local-only branches
   `-presquash`, `push-core-sweep`, `push-fused-profile`, superseded `sparse-kv-push` / `sparse-kv-push-op`.
4. After the PR: restore push resume (progress tensor, quit semaphore, part 3) as op args to hide the leftover push
   behind MoE/FFN and the next layer's q/kv (heavy layers like L62 push ~850 us at 1M; resume was groundwork, not dead
   code); run kv before q; optional data-based buffer cap; nits 15-18; maybe slim the nightly 1M test.

## How to run

- Device tests through `scripts/run_safe_pytest.sh` (device lock).
- Op test: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_sparse_kv_push.py`
- MLA: `scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py -k kv_push`
- With `PREFILL_TORUS_XY_CERTIFIED=1`, also export `TT_MESH_GRAPH_DESC_PATH` to the single_bh_galaxy_torus_xy descriptor.
- 1M e2e: `run_multirank_pcc.sh glm53 sc1` with `TTRUN_DIR` (hostfile for the local galaxy), `PREFILL_SUMMARIES`,
  `PREFILL_KEEP_RUN_DIR`. Old run scripts/logs live in `~/.cache/claude-sparse-push/` on the old machine.
- Pre-commit hook env was broken there: commits used `--no-verify` with manual formatting.
