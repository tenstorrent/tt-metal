# AutoFix: issue #55075, hang in the 2D core grid pre-all-gather path

Branch: `fplavec/ln-pre-ag-2d-multirow` (from `main` at `b5a29a3316c`). Not committed.
Scope: Option B from `AUTODEBUG.md`, which adds real support for a core that handles more than one
tile row (`tiles_per_core_x > 1`) in `LayerNormPreAllGather2DProgramFactory`. The SRAM
over-allocation at wide rows (problem 2 in the issue) is not part of this change.

## Diagnosis (confirmed against source before editing)

When a core handles more than one tile row, the 2D path hangs for two reasons:

1. The compute kernel popped the reduce scaler tile once per row. The reader pushes it only once,
   so the second row's reduce waited forever.
2. Each core sent only one partial result to the merge core, and the merge core merged only once.
   The writer on the merge core expected one output tile per row.

The input start offset, the row-to-row input stride, and the output start offset were also wrong
for more than one row per core (#56908). The hang hid these errors.

## Changes

- `kernels/compute/layernorm_pre_allgather_2d.cpp`
  - The merge block now runs once per row, inside the row loop.
  - The scaler pop (and a pop of the zero tile on merge cores) now happens once, after the loop.
  - No new `compute_kernel_hw_startup` call is added. The merge block's existing mid-kernel call
    and its `TODO(#52395)` are unchanged, but the block now runs once per row, so on merge cores
    that call also runs once per row.
  - Nothing restores the statistics setup before the next row. The square and add helpers
    (`eltwise_chain`) and the reduce helper set their own data formats and inits. The reduce helper
    also clears its packer mask when it finishes.
  - Open PR #57495 removes the merge-block call. It touches the same lines, so the two changes
    will conflict.
- `kernels/dataflow/reader_layernorm_preallgather_2d.cpp`
  - Each row's partial is sent to the merge core inside the row loop. Row `r`'s partial is sent
    after row `r + 1`'s input has been read.
  - The gather buffer on the merge core holds one row. For each row, the merge core first does
    `reserve_back(cores_y)` on the gather buffer, which waits until its compute has used the
    previous row. It then increments a new semaphore, `gather_free`, on the other cores in its
    column with `inc_multicast`.
  - Worker cores wait for `gather_free`, reset it, and only then write their partial. So the merge
    core's `reducer` semaphore reset cannot lose an increment that belongs to the next row.
  - The input tile index starts each row at `row_stride` (the full row width in tiles) past the
    previous row's start.
- `layernorm_pre_all_gather_program_factory.cpp`
  - Adds the `PRE2D_GATHER_FREE` semaphore and its reader binding.
  - Adds the reader runtime args `row_stride` and `workers_noc_{x,y}_{start,end}`. The start and
    end are swapped when the reader uses NOC_1.
  - `in_tile_offset = x * tiles_per_core_x * Wt + y * tiles_per_core_y`
  - `out_tile_offset = x * tiles_per_core_x * out0_tiles`
- `tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py`
  - `test_pre_all_gather_non_welford_fp32_precision` now takes a combined `variant, shape`
    parameter. The existing (1,1,32,128) case still runs for all three variants.
  - Four shapes run for `rms_norm_2d` only: (1,1,288,64), (1,1,320,256), (1,1,1024,1024) and
    (1,1,288,32). They give 3, 2, 4 and 3 tile rows per core, with 2, 8, 8 and 1 cores splitting
    each row. With 1 core per row, the merge core has no other cores in its column.
  - The existing parameters and checks apply unchanged: fp32 input, with and without a residual,
    the accurate SFPU and fast FPU reduce, and the fp64 reference with its tolerances.

## Validation (Wormhole n150, single device)

| Command | Result |
|---|---|
| `./build_metal.sh -e --enable-fake-kernels-target` | pass |
| `test_pre_all_gather_non_welford_fp32_precision` (28 cases), `TT_METAL_OPERATION_TIMEOUT_SECONDS=45` | 28 passed |
| Same test, `rms_norm_2d` cases, with the reader's row stride broken on purpose | 12 failed: the 3 multi-row shapes with more than one core per row. 8 passed: (1,1,32,128) and (1,1,288,32). In those, the broken stride gives the same result as the correct one. Reader restored afterwards. |
| Same test, `rms_norm_2d` cases, with the reader ignoring each core's start offset | all 20 failed, including (1,1,288,32). Reader restored afterwards. |
| `test_distributed_layernorm_pre_allgather.py` + `test_distributed_rmsnorm_allgather.py` (run with an earlier version of the test) | 174 passed |
| Issue snippet at (1,1,1024,1024), HiFi2, `packer_l1_acc=True` | completes, max relative error of sum(x²) 1.2% (bf16 output) |
| `pre-commit run` on changed files | pass |
| Original sources (fix removed, rebuilt): issue snippet at (1,1,1024,1024) | `TIMEOUT: device timeout, potential hang detected`; device reset with `tt-smi -r` |
| Original sources: (1,1,32,64) and (1,1,288,64), bf16, no residual, with an earlier version of the test | (1,1,32,64) passed; (1,1,288,64) hit the same device timeout; device reset |
| Fix reapplied and rebuilt: earlier version of the test | 24 passed |
| Both test files, final kernel | 166 passed; the kernel cache files show the 2D compute kernel was rebuilt |
| Final kernel, bf16 input, HiFi2, `packer_l1_acc=True`: (1,1,1024,1024), (1,1,288,64), (1,1,288,32), residual on and off | all complete, max relative error of sum(x²) 1.2% to 1.6% |

## Not verified / remaining risk

- Only Wormhole was tested. Blackhole and Quasar were not.
  - On Quasar the gather buffer now uses `reserve_back`/`push_back` on the merge core, and the
    semaphore may resolve to a different access method. Neither was run.
  - The code assumes that pushing the full buffer depth brings the gather buffer's write pointer
    back to its base, as it does on Wormhole and Blackhole.
- The C++ gtests `DistributedRmsNorm2DGrid*` were not run, because this build has
  `TTNN_BUILD_TESTS=OFF`. They use (1,1,32,64), one row per core, a case the Python tests also
  cover.
- Performance was not measured. Merge cores now run a semaphore handshake once per row.
- Not addressed:
  - The SRAM over-allocation at wide rows (issue problem 2).
  - The same offset errors in the 2D post-all-gather factory, at
    `layernorm_post_all_gather_program_factory.cpp:536`.
