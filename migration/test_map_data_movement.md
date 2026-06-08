# Static kernel→test map — data_movement + reduction op-family (Phase 2)

STATIC analysis only (no device runs). For each kernel: kernel → program factory (+ exact dispatch
condition) → TTNN op → test params that hit it.

Note on `-k`: pytest `-k` cannot match `name=value` with `=`. Filter on a value substring only
(e.g. a shape token or dtype id), and add `--run-all` to disable the wrapper's implicit `-x` when you
want full pass/fail counts. Confirm a kernel actually built by grepping its basename in the JIT
artifacts under `generated/` after a run.

---

### ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp
- role / tag: receiver (final/reduce core) / clean — cleanest modern receiver. FLAG mcast of
  receiver-ready (`receiver_sem.set_multicast<EXCLUDE_SRC>`), then `sender_sem.wait(Wt_final)` for
  all local cores' unicast data. New API (`Noc`, `Semaphore<>`, `CircularBuffer`).
- factory: `TopKMultiCoreProgramFactory::create_descriptor` —
  `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp:388`
  (kernel wired in); selected by `TopKDeviceOperation::select_program_factory`
  `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:59`.
  Selected when ALL hold: `padded_shape[dim] >= multi_core_min_width(8192)` AND
  `< 65535` AND `is_power_of_two(dim_size)` AND `k <= 64` AND `verify_multi_core_cost(...)`
  passes (work divisible across cores, fits L1, contiguous rect, >1 core). Else single-core.
- op: `ttnn.topk` (largest/smallest, sorted, dim=-1)
- candidate_validation_set:
  - `test_topk[None-True-True-BFLOAT16_B-...-(1, 1, 32, 8192, 3, 6)]` — W=8192 (pow2, == threshold),
    k=6 ≤ 64 → multicore. Smallest fast multicore case in test_topk.
- candidate_regression_set:
  - `test_topk` row `(1, 1, 32, 8192, 3, 50)` (k=50), and `(1, 1, 32, 32*512=16384, 3, 32)`;
    plus bfloat8_b dtype variants of the same rows.
  - `test_topk_sub_core_grids[...-(1, 1, 32, 16*1024, 3, 32)]` — multicore + explicit sub_core_grids +
    preallocated indices path.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/reduce/test_topk.py -k "8192 and BFLOAT16_B"`
  then confirm build: `grep -rl reader_final_topk generated/`
- coverage_confidence: high
- gaps: `-k "8192"` also matches the k=50 row `(…,8192,3,50)`; both are multicore so either validates.
  W=8192 is exactly `multi_core_min_width` — relies on `>=` (line 66), not `>`. The mcast send only
  fires when `num_dests`>0 i.e. genuinely multi-core (guaranteed once verify_multi_core_cost passes,
  which requires >1 core).

---

### ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp
- role / tag: sender (local/worker core) / refactor — unicast scatter of Kt value-tiles + Kt
  index-tiles into the final core's L1 (`noc.async_write` to `UnicastEndpoint`), then `sender_sem.up()`
  (atomic increment on remote) — the up(counter) handshake counterpart to reader_final_topk.
  Needs `async_atomic_barrier` after `up`.
- factory: `TopKMultiCoreProgramFactory::create_descriptor` —
  `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp:413`
  (kernel wired in). Same dispatch condition as reader_final_topk above (paired in the same
  multicore program — local cores run writer_local_topk, final core runs reader_final_topk).
- op: `ttnn.topk`
- candidate_validation_set:
  - `test_topk[None-True-True-BFLOAT16_B-...-(1, 1, 32, 8192, 3, 6)]` — same case; this kernel is
    always co-dispatched with reader_final_topk.
- candidate_regression_set: same rows as reader_final_topk (8192/16384 W, k≤64, bf16 + bf8_b);
  `test_topk_sub_core_grids` for the sub_core_grids + preallocated-indices path.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/reduce/test_topk.py -k "8192 and BFLOAT16_B"`
  then `grep -rl writer_local_topk generated/`
- coverage_confidence: high
- gaps: none beyond the shared "W==8192 boundary" note. Both topk kernels are exercised by the exact
  same param, so one run covers the pair.

---

### ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp
- role / tag: hybrid (worker + reduce roles in one kernel) / refactor — 2-rect (cores0/cores1)
  topology; the reduce core FLAG-mcasts the start signal to all workers in ONE send covering
  INCLUDE+EXCLUDE rects (start_core0/end_core0 and start_core1/end_core1 compile args), workers
  increment a done counter. FLAG-ONLY — no data is multicast; intermediate vals/idxs are gathered, no
  payload broadcast. Legacy raw-NoC + semaphore API.
- factory: `ArgMaxMultiCoreProgramFactory::create_descriptor` —
  `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_multi_core_program_factory.cpp:380`
  (cores0) and `:412` (cores1); selected by `ArgMaxDeviceOperation::select_program_factory`
  `ttnn/cpp/ttnn/operations/reduction/argmax/device/argmax_device_operation.cpp:62`.
  Selected when `args.use_multicore == true` (op default is FALSE —
  `argmax.hpp:14`, `argmax_nanobind.cpp:82`). Multicore additionally REQUIRES ROW_MAJOR input
  (validate `argmax_device_operation.cpp:154`) and ≤2 sub_core_grid ranges. The actual flag-mcast /
  2-rect path only does real cross-core work when `num_total_cores > 1`, which needs
  `red_dim_units` (last dim) >> `min_red_dim_units_per_core` (= dram_align/2, ~16 for bf16 DRAM).
- op: `ttnn.argmax(..., use_multicore=True)`, dim=-1 (last-dim only), output UINT32.
- candidate_validation_set:
  - `test_argmax[...-([64, 128], ROW_MAJOR, -1, True, True, float32)]` — RM, use_multicore=True,
    last-dim=128 → 128/16=8 blocks → 8 worker cores → real multicore flag-mcast. Small & fast.
  - `test_argmax[...-([1, 8, 160], ROW_MAJOR, -1, False, True, bfloat16)]` — RM multicore, tiny.
- candidate_regression_set:
  - `([1, 256, 1024*8], ROW_MAJOR, -1, False, True, float32)` — last-dim 8192, spans many cores +
    likely both core groups (2-rect path).
  - `([16, 32, 64, 128], ROW_MAJOR, -1, True, True, bfloat16/int32)`,
    `([32, 64, 128], ROW_MAJOR, -1, True, True, float32)`,
    `([8, 16, 32, 64], ROW_MAJOR, -1, True, True, float32)`,
    `([50, 100, 200], ROW_MAJOR, -1, True, True, int32)`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/reduce/test_argmax.py -k "64-128 or 1, 8, 160"`
  (filter on shape substring; the `True, True` use_multicore flag can't be `-k`-filtered directly).
  Then `grep -rl reader_argmax_interleaved_multicore generated/`.
- coverage_confidence: med
- gaps: `use_multicore` is a positional param value, not in the test id name, so `-k` selects by SHAPE
  substring only — the chosen shape MUST be one whose row has use_multicore=True (verified above:
  `[64,128]` float32 row and `[1,8,160]` row both have it). Shape substring `128` is non-unique
  (matches many rows); the suggested `"64-128"` token narrows to the `[64,128]` rows but still
  includes the use_multicore=False `[64,128]` float32 row — run `--run-all` and confirm via the
  generated JIT artifact grep that the multicore reader actually built. Whether BOTH 2-rect groups
  (cores1>0) populate depends on the device grid + `split_work_to_cores`; the regression 8192 case is
  the safer bet for exercising the cores1 INCLUDE+EXCLUDE second rect.

---

### ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/coordinator_single_row_multi_core.cpp
- role / tag: sender (coordinator) / refactor — legacy raw API; FLAG-ONLY.
  **DEFERRED: legacy-API prereq** (needs a Noc/Semaphore port before migration).
- factory: `SortProgramFactorySingleRowMultiCore::create_descriptor` —
  `ttnn/cpp/ttnn/operations/data_movement/sort/device/sort_program_factory.cpp:1046`; selected by
  `SortDeviceOperation::select_program_factory` `sort_device_operation.cpp:49`.
  Selected when `Wt > total_number_of_tiles_for_hybrid_approach` (the "DRAM implementation" branch;
  `Wt > SORT_WT_THRESHOLD(64)` is the first gate, then > the hybrid cap). i.e. very large last dim.
- op: `ttnn.sort` (dim=-1, descending)
- candidate_validation_set:
  - `test_sort_long_tensor[...-([1, 16384 * TILE_WIDTH=524288], -1, False)]` — huge Wt → multicore DRAM
    path. (Also `([1, 151936])`, `([1, 128256])` rows are large-Wt candidates.)
- candidate_regression_set: other `test_sort_long_tensor` rows (151936, 128256, 8192*TILE_WIDTH).
- verification_command (defer execution; for when ported):
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/data_movement/test_sort.py -k "test_sort_long_tensor and 16384"`
  then `grep -rl coordinator_single_row_multi_core generated/`.
- coverage_confidence: med
- gaps: DEFERRED — legacy-API prereq. Exact Wt cutoff for the multicore-DRAM vs hybrid branch is
  device-grid dependent (`total_number_of_tiles_for_hybrid_approach`), so the large `test_sort_long_tensor`
  rows are the candidates but which one lands in the DRAM branch must be confirmed by the JIT artifact
  grep once the kernel is run.

---

### ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/reader_single_row_multi_core.cpp
- role / tag: receiver / refactor — legacy. **DEFERRED: legacy-API prereq.**
- factory: `SortProgramFactorySingleRowMultiCore::create_descriptor` —
  `sort_program_factory.cpp:1078`; same dispatch as coordinator above (co-dispatched in the
  multicore-DRAM sort program). Selected when `Wt > hybrid cap` (`sort_device_operation.cpp:49`).
- op: `ttnn.sort`
- candidate_validation_set: `test_sort_long_tensor[...-([1, 16384 * TILE_WIDTH], -1, False)]` (same row).
- candidate_regression_set: same large-Wt `test_sort_long_tensor` rows as coordinator.
- verification_command (defer):
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/data_movement/test_sort.py -k "test_sort_long_tensor and 16384"`
  then `grep -rl reader_single_row_multi_core generated/`.
- coverage_confidence: med
- gaps: DEFERRED — legacy-API prereq; same branch-cutoff caveat as coordinator.

---

### ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/dataflow/writer_single_row_multi_core.cpp
- role / tag: sender / refactor — legacy. **DEFERRED: legacy-API prereq.**
- factory: `SortProgramFactorySingleRowMultiCore::create_descriptor` —
  `sort_program_factory.cpp:1098`; same dispatch as coordinator/reader above.
- op: `ttnn.sort`
- candidate_validation_set: `test_sort_long_tensor[...-([1, 16384 * TILE_WIDTH], -1, False)]` (same row).
- candidate_regression_set: same large-Wt rows.
- verification_command (defer):
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/data_movement/test_sort.py -k "test_sort_long_tensor and 16384"`
  then `grep -rl writer_single_row_multi_core generated/`.
- coverage_confidence: med
- gaps: DEFERRED — legacy-API prereq; same branch-cutoff caveat.

---

### ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_interleaved_with_overlap.cpp
- role / tag: sender / refactor — single-shot go-flag mcast over 2-3 rects (`get_multicast_regions`
  around a controller core at logical {0,0}); legacy API. TILE-layout (tilized) variant.
  **DEFERRED: legacy-API prereq.**
- factory: `MoveOverlapProgramFactory::create_descriptor` —
  `ttnn/cpp/ttnn/operations/data_movement/move/device/move_overlap_program_factory.cpp:125`
  (tilized branch picks this kernel); selected by `MoveDeviceOperation::select_program_factory`
  `move_device_operation.cpp:16` for `MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP`.
  That strategy is chosen in `move.cpp:105` when: input/output buffers OVERLAP in the same mem space
  (`not non_overlap`) AND it `fits_in_cb` AND grid x>1 AND y>1. (Tilized branch ⇒ TILE layout input.)
- op: `ttnn.move` (in-place L1→L1 realloc where src/dst ranges overlap, TILE layout)
- candidate_validation_set:
  - `test_move_op[overlap-TILE-...-in0_L1-out_L1-...]` shape `[1, 1, 32, 32]` —
    `tests/tt_eager/python_api_testing/unit_testing/misc/test_move.py:107`. test_id="overlap" allocates
    a dummy then frees it so the moved tensor's dst overlaps src in L1 → MULTI_CORE_OVERLAP; TILE
    layout → this kernel. (Non-L1 in0 is skipped by the test.)
- candidate_regression_set: `test_move_op_with_program_cache` (shape `[1,3,320,384]`, TILE, L1,
  overlap path) at `test_move.py:117`.
- verification_command (defer):
  `scripts/run_safe_pytest.sh tests/tt_eager/python_api_testing/unit_testing/misc/test_move.py -k "overlap and TILE and in0_L1"`
  then `grep -rl move_interleaved_with_overlap generated/`.
- coverage_confidence: med
- gaps: DEFERRED — legacy-API prereq. The OVERLAP strategy depends on runtime allocator addresses
  (`fits_in_cb` + actual address overlap), not just params — `-k "overlap"` selects the test_id that
  is DESIGNED to force overlap (dummy-alloc-then-free), but confirm via the JIT artifact grep that the
  overlap kernel (not plain MoveProgramFactory) built. `-k "TILE"` disambiguates from the RM stick twin.

---

### ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp
- role / tag: sender / refactor — RM (row-major / stick) twin of the above; same single-shot go-flag
  mcast over 2-3 rects, legacy API. **DEFERRED: legacy-API prereq.**
- factory: `MoveOverlapProgramFactory::create_descriptor` —
  `move_overlap_program_factory.cpp:126-127` (non-tilized branch picks this kernel); same
  MULTI_CORE_OVERLAP dispatch as the tiled twin (`move.cpp:105`), but with ROW_MAJOR input layout.
- op: `ttnn.move` (in-place overlapping L1→L1, ROW_MAJOR layout)
- candidate_validation_set:
  - `test_move_op[overlap-RM-...-in0_L1-out_L1-...]` shape `[1, 1, 32, 32]` —
    `tests/tt_eager/python_api_testing/unit_testing/misc/test_move.py:107` with layout=RM.
- candidate_regression_set: the `[1,3,320,384]` RM overlap variant of `test_move_op`.
- verification_command (defer):
  `scripts/run_safe_pytest.sh tests/tt_eager/python_api_testing/unit_testing/misc/test_move.py -k "overlap and RM and in0_L1"`
  then `grep -rl move_stick_layout_interleaved_with_overlap generated/`.
- coverage_confidence: med
- gaps: DEFERRED — legacy-API prereq. Same runtime-allocator-dependent overlap caveat as the tiled
  twin; `-k "RM"` is the discriminator that routes to the stick kernel vs the tiled one.

---

## Summary of dispatch conditions
- topk multicore (both topk kernels): `dim_size >= 8192` AND pow2 AND `< 65535` AND `k <= 64` AND
  verify_multi_core_cost passes (>1 core, fits L1, divisible).
- argmax multicore reader: `use_multicore=True` (default False) AND ROW_MAJOR input AND last-dim; real
  cross-core flag-mcast only when last-dim spans >1 worker core (last-dim >> ~16).
- sort multicore-DRAM (3 kernels): `Wt > hybrid-cap` (`Wt > 64` first gate, then > grid-dependent cap).
- move overlap (2 kernels): `MULTI_CORE_OVERLAP` ⇒ src/dst overlap in same mem space + fits_in_cb +
  grid x>1,y>1; tilized→interleaved kernel, RM→stick kernel.
