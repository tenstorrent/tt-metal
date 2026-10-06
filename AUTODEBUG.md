# AutoDebug: issue #55075, `LayerNormPreAllGather2DProgramFactory` (2D core grid pre-all-gather)

Scope: `ttnn.rms_norm_pre_all_gather(..., use_2d_core_grid=True)`, which selects
`LayerNormPreAllGather2DProgramFactory`. Inspection only; no hardware run.
Line references are against local `main` at `b5a29a3316c`. `origin/main` (`a30a6a1b15d`, fetched
during this investigation) has no newer commits that touch these files.

## Summary

| # | Problem | Verdict | Root cause |
|---|---|---|---|
| 1a | Hang at `(1,1,1024,1024)` | **Confirmed from code.** The first stop is the reduce scaler. | The compute kernel pops the one reduce scaler tile inside the per-row loop. The reader pushes that tile only once. The second row's reduce waits for it forever. |
| 1b | Same hang, second layer | **Confirmed from code.** | The partial-result handoff, the cross-core merge, and the output write all run once per core, but the writer expects `tiles_per_core_x` tiles. Moving the scaler pop alone does not stop the hang. |
| 2 | SRAM over-allocation at `(1,1,32,4096)` with `fp32_dest_acc_en=True` | **Confirmed; the arithmetic reproduces the reported 1725632 B exactly.** | Input, x², residual and fused buffers are sized from the full row width `Wt`, but every kernel only holds `tiles_per_core_y = Wt / cores_y` tiles per row. |

All of the hang findings share one trigger: **a core owns more than one tile row**
(`tiles_per_core_x > 1`). From the work split in the factory, that happens exactly when
`NC * Ht > compute_with_storage_grid_size().y`. On an 8x8 Wormhole grid this means more than 8 tile
rows (padded `H > 256` for `NC = 1`). The 2D kernels were written for one tile row per core when
the path was added in #25960 (`1bcaf92d8be`), and no current test or model runs it with more than
one.

An independent line-by-line ledger audit (separate subagent, no preferred answer given) reached the
same result for 1a, 1b and problem 2, and found every buffer balanced for `(1,1,32,1024)`.

Same trigger, not a hang cause: the reader's input tile index and the writer's output tile offset
are also wrong when `tiles_per_core_x > 1` (#56908). Any real multi-row fix must correct them too;
see the first entry under "Other potential issues".

## Trigger condition

The work split is at
[layernorm_pre_all_gather_program_factory.cpp:491-503](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L491-L503):

- `cores_x` is the largest divisor of `num_tile_rows = NC * Ht` that is not larger than `grid.y`.
  `tiles_per_core_x = num_tile_rows / cores_x`.
- `cores_y` is the largest divisor of `Wt` that is not larger than `grid.y`.
  `tiles_per_core_y = Wt / cores_y`.

If `num_tile_rows <= grid.y`, `cores_x = num_tile_rows` and `tiles_per_core_x = 1`. If
`num_tile_rows > grid.y`, then `cores_x < num_tile_rows` and `tiles_per_core_x >= 2`.

| Shape (bf16, grid 8x8) | num_tile_rows | cores_x | tiles_per_core_x | Wt | cores_y | tiles_per_core_y | Reported result |
|---|---|---|---|---|---|---|---|
| (1,1,32,1024) | 1 | 1 | 1 | 32 | 8 | 4 | runs |
| (1,1,32,2048) | 1 | 1 | 1 | 64 | 8 | 8 | runs |
| (1,1,32,4096) | 1 | 1 | 1 | 128 | 8 | 16 | allocation failure (problem 2) |
| (1,1,1024,1024) | 32 | 8 | **4** | 32 | 8 | 4 | hang |
| (1,1,288,64) (commenter's smallest case) | 9 | 3 | **3** | 2 | 2 | 1 | hang predicted |

The passing and failing rows in the issue split exactly on `tiles_per_core_x`.

## Finding 1 (headline): hang when a core owns more than one tile row

### 1a. The reduce scaler is popped once per row but pushed once per program

Evidence:

- The reader prepares the scaler once, before its row loop:
  [reader_layernorm_preallgather_2d.cpp:57-62](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L57-L62).
  The helper reserves one tile and pushes one tile:
  [reduce_helpers_dataflow.inl:163-203](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl#L163-L203).
- The scaler buffer holds one tile: `in1_tiles = 1` at
  [layernorm_pre_all_gather_program_factory.cpp:467](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L467-L467),
  used at
  [layernorm_pre_all_gather_program_factory.cpp:536](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L536-L536).
- Every call of `compute_kernel_lib::reduce` waits for one scaler tile and never pops it:
  [reduce_helpers_compute.inl:399](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L399-L399).
  The wait is outside the SFPU/FPU branch, so it applies to both the accurate and the fast reduce.
- The 2D compute kernel pops the scaler **inside** the row loop:
  [layernorm_pre_allgather_2d.cpp:70-112](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp#L70-L112),
  pop at line 111.
- On Wormhole and Blackhole, a compute-side `wait_front` is `llk_wait_tiles` and `pop_front` is
  `llk_pop_tiles`:
  [tt-1xx/dataflow_buffer.inl:100-114](tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl#L100-L114).
  After the pop, the buffer count is 0 and nothing pushes again, so the next wait blocks.
- Both 1D sibling kernels pop the scaler **after** the loop, which is the placement that matches the
  reduce helper's contract:
  [layernorm_pre_allgather.cpp:125](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp#L124-L125)
  and
  [rmsnorm_pre_allgather.cpp:107](ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp#L106-L107).

Ledger for `(1,1,1024,1024)`, bf16, `fp32_dest_acc_en=True`, grid 8x8, any core:
`NCHt = 4`, per-core `Wt = 4`, input buffer depth 64 tiles, scaler depth 1.

| Step | Reader | Compute | Scaler count |
|---|---|---|---|
| start | push scaler (1) | | 1 |
| | pushes all 16 input tiles (depth 64, never blocks), then waits on the partial-result buffer | | 1 |
| row 0 | | square 4 tiles, reduce (waits scaler: OK), push partial, pop input 4, **pop scaler** | 0 |
| | ships row-0 partial to the merge core, semaphore up; merge-core reader pushes the column's 8 partials and exits; other readers exit | | 0 |
| row 1 | | square 4 tiles, reduce: **`scaler_dfb.wait_front(1)` blocks forever** | 0 |

End state: every compute kernel on all 64 cores is stopped in the unpack thread at
[reduce_helpers_compute.inl:399](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L399-L399)
on row index 1. The 8 merge-core writers are stopped at
[writer_unary_interleaved_start_id_blocked.cpp:33](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp#L33-L33),
because the merge block in compute is never reached. All readers finish.
With `tiles_per_core_x == 1` the loop runs once, the pop happens after the only reduce, and nothing
blocks. This matches the passing H=32 cases.

History: the original kernel from #25960 already popped the scaler inside the loop, but it waited on
the scaler only once, before the loop. In the circular-buffer model of that time, a depth-1 buffer
read pointer wraps back to the same tile, so the extra pop did not block. The pop started to block
when the kernel moved to `compute_kernel_lib::reduce` (#41637), which waits for the scaler on every
call. My guess is that this is why the scaler stop is not mentioned in older reports; it does not
change the conclusion, because 1b below blocks in both versions.

This confirms the issue commenter's first claim. Some of the commenter's line numbers come from an
older revision: the scaler wait is now at `reduce_helpers_compute.inl:399` (not 403) and the 1D
layernorm pop is at `layernorm_pre_allgather.cpp:125` (not 137).

### 1b. The partial-result handoff and the merge run once, but the writer expects one tile per row

Moving the scaler pop out of the loop is necessary but not enough. Evidence:

- Compute produces one partial tile per row into the partial-result buffer (`dfb::out`, depth
  `out0_tiles = 1`):
  [layernorm_pre_all_gather_program_factory.cpp:472](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L472-L472),
  [layernorm_pre_all_gather_program_factory.cpp:553](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L553-L553),
  reserve at
  [reduce_helpers_compute.inl:553](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L553-L558).
- The reader consumes exactly one partial tile, after its row loop, and sends it to the merge core:
  [reader_layernorm_preallgather_2d.cpp:105-125](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L105-L125).
  The semaphore handshake and the push of the gathered tiles also run once:
  [reader_layernorm_preallgather_2d.cpp:127-135](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L127-L135).
- The merge in compute runs once, after the row loop, and pushes one `out_final` tile:
  [layernorm_pre_allgather_2d.cpp:117-166](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp#L117-L166).
- The writer on each merge core is told to write `tiles_per_core_x * out0_tiles` tiles:
  [layernorm_pre_all_gather_program_factory.cpp:737-756](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L737-L756),
  and waits for each one:
  [writer_unary_interleaved_start_id_blocked.cpp:32-42](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp#L32-L42).

Ledger with only the scaler pop moved:

- `tiles_per_core_x >= 3` (the `(1,1,1024,1024)` case): row 0 partial is shipped and popped by the
  reader, which then exits. Row 1 fills the depth-1 partial buffer. Row 2's reduce blocks in
  `output_dfb.reserve_back(1)` at
  [reduce_helpers_compute.inl:553](ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl#L553-L553)
  on every core. The merge-core writers block on their first `wait_front`.
- `tiles_per_core_x == 2` (on an 8-row grid: `NC * Ht` in {10, 12, 14, 16}): every compute kernel
  finishes its two rows, and one partial tile is left unconsumed. The merge block runs on row 0's
  partials and pushes one `out_final` tile. The writer writes it and then blocks on the second
  `wait_front`.

So the issue commenter's second claim (one-shot merge, writer expects `tiles_per_core_x` tiles) is
correct. For `tiles_per_core_x >= 3` the stop happens even earlier, in compute, before the writer.

### Smallest fix for the hang

Two options, depending on whether multi-row support is wanted now.

**Option A, smallest change that removes the hang.** Reject `tiles_per_core_x > 1` with a
`TT_FATAL` on the host, for example in the 2D factory right after
[layernorm_pre_all_gather_program_factory.cpp:498](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L498-L498),
or in `validate_on_program_cache_miss` with the device grid. The message should state the condition
(`NC * Ht` must not exceed `grid.y`). Every shape that runs today has `tiles_per_core_x == 1`, so
this changes no working case; it turns a device hang into a host error. The alternative of silently
falling back to the 1D factory changes the result path for a user who asked for the 2D grid, so I
would prefer the explicit error.

**Option B, real support for `tiles_per_core_x > 1`.** All of these are needed; any subset still
hangs or corrupts:

1. Compute: move `dfb_reduce.pop_front(1)` from line 111 to after the row loop, as in the 1D kernels.
2. Reader: per-row input stride and a corrected start offset in the factory (first entry under
   "Other potential issues").
3. Per-row cross-core merge:
   - Reader: move the partial wait, the NoC write to the merge core, and the semaphore increment
     into the row loop.
   - Compute (merge core): move the merge block into the row loop, so it pushes one `out_final`
     tile per row.
   - Add back-pressure, so a worker does not overwrite its slot in the merge core's gather buffer
     before the merge core's compute has popped the previous row. One way: the merge-core reader
     does `reserve_back(cores_y)` on the gather buffer, then signals the column's workers through a
     second semaphore before they write. Today the merge-core reader calls `push_back` without a
     `reserve_back` (reader line 133), which only works because it runs once.
   - Make the reducer semaphore safe for reuse. Today `wait(cores_y)` then `set(0)` (reader lines
     131-135) is safe, because each core increments it once per program. In a per-row loop, an
     increment from a fast worker for the next row could arrive between the wait and the `set(0)`
     and be lost. The back-pressure semaphore above prevents that; a running target
     (`cores_y * (row + 1)`) also works.
4. Writer: corrected output start offset (same entry).

An alternative to item 3 that keeps the one-shot handshake: size the partial buffer at
`tiles_per_core_x` tiles and the gather buffer at `cores_y * tiles_per_core_x` tiles, ship all
partials at once, and merge row by row after the loop. This needs less new synchronization, but the
gather buffer grows with H. For example, H = 8192 on 8x8 gives
`tiles_per_core_x = 32` and an 8 x 32 fp32-tile gather buffer of 1 MiB, so it needs its own SRAM
limit. I would use the per-row merge.

## Finding 2 (headline): SRAM over-allocation at wide rows

Evidence:

- Buffer depths use the full row width, before the work split exists:
  [layernorm_pre_all_gather_program_factory.cpp:465-471](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L465-L471)
  (`in0_tiles`, `res_tiles` and `intermed0_tiles` are `Wt * 2`, `fused_tiles` is `Wt`).
- The work split is computed later, at line 503.
- The kernels only use the per-core width: compute compile-time `Wt = tiles_per_core_y`
  ([layernorm_pre_all_gather_program_factory.cpp:649-654](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L649-L654)),
  reader runtime `Wt = tiles_per_core_y` (line 746). The compute kernel waits for at most `Wt`
  input tiles before popping, and reduces exactly `Wt` x² tiles
  ([layernorm_pre_allgather_2d.cpp:59-111](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp#L59-L111)).
  So the buffers are `cores_y` times larger than needed.
- The size check at
  [layernorm_pre_all_gather_program_factory.cpp:474-479](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L474-L479)
  can never fail, because `in0_tiles` is derived from `W`.
- The sibling 2D post-all-gather factory already sizes its buffers per core
  (`cb_length = tiles_per_core_y`):
  [layernorm_post_all_gather_program_factory.cpp:226-228](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_program_factory.cpp#L226-L228).

Arithmetic for `(1,1,32,4096)`, bf16 input and output, `fp32_dest_acc_en=True`
(intermediate tile 4096 B, bf16 tile 2048 B), `cores_y = 8`, `tiles_per_core_y = 16`:

| Buffer | Current depth | Current bytes | Per-core depth | Per-core bytes |
|---|---|---|---|---|
| input | 256 | 524288 | 32 | 65536 |
| reduce scaler | 1 | 4096 | 1 | 4096 |
| x² | 256 | 1048576 | 32 | 131072 |
| gather (`cores_y`) | 8 | 32768 | 8 | 32768 |
| partial, zero | 1 + 1 | 8192 | 1 + 1 | 8192 |
| out_final | 1 | 2048 | 1 | 2048 |
| **total** | | **1619968** | | **243712** |

`1619968 + 105664 = 1725632`, the reported end address. The same base (105664 B) also reproduces
the issue's 1D number (`1579008 + 105664 = 1684672`), so my guess is that 105664 B is the reserved
SRAM base on that Wormhole setup. With per-core sizing the end address is about 349 KB.

### Smallest fix for the over-allocation

Move the sizing below line 503 and use the per-core width:

- `in0_tiles = res_tiles = intermed0_tiles = tiles_per_core_y * 2`
- `fused_tiles = tiles_per_core_y`
- replace the check at lines 474-479 with one against `tiles_per_core_y`, or drop it.

The kernels need no change for this part: every wait and pop count in the compute kernel is
`tiles_per_core_y` or a block of it.

Remaining limit after this fix: when `Wt` has no divisor between 2 and `grid.y`, `cores_y = 1` and
per-core sizing changes nothing. Example: `W = 4064` (`Wt = 127`) with bf16 input and
`fp32_dest_acc_en=True` still ends at 1684672 B. On an 8-row grid this fails for prime `Wt` of 113
and above. For those widths, add the 1D factory's fallback to a single x² buffer
([layernorm_pre_all_gather_program_factory.cpp:151-171](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L151-L171)),
with a budget that includes the 2D-only buffers, or reject them with a clear message.

## Test coverage

No test or model in the repository runs the 2D pre-all-gather path with `tiles_per_core_x > 1`.
Every live call has `num_tile_rows` of 1 or 4:

- gtests `DistributedRmsNorm2DGridStats` and `DistributedRmsNorm2DGrid`, shape (1,1,32,64):
  [test_normalization.cpp:733-791](tests/ttnn/unit_tests/gtests/test_normalization.cpp#L733-L791).
- Nightly single-device test, per-device shapes (1,1,128,2048) and (1,1,128,1024) with
  `fp32_dest_acc_en=True`:
  [test_distributed_rmsnorm_allgather.py:112-123](tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_rmsnorm_allgather.py#L112-L123).
  W = 2048 is the widest width tested on this path.
- Nightly fp32 precision test, `rms_norm_2d` variant, shape (1,1,32,128):
  [test_distributed_layernorm_pre_allgather.py:1065-1115](tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py#L1065-L1115).
- The exhaustive 2D test is always skipped:
  [test_distributed_layernorm_exhaustive.py:482](tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_exhaustive.py#L482-L482).

The only model call that would reach `tiles_per_core_x > 1` is the Qwen prefill path, behind
`policy["prefill_norm_rectangular"]`, which nothing in the repository sets:
[decoder_tp.py:725-731](models/demos/qwen38_27b_qb2/tt/decoder_tp.py#L725-L731).
With prefill chunks of 2048 or 4096 rows it would have `tiles_per_core_x` of 8 to 32 and would hang.

Suggested regression tests: one shape with `NC * Ht > grid.y` (for example `(1,1,288,64)` and
`(1,1,1024,1024)`), with row-distinct input values so that wrong-tile reads are visible, and
`(1,1,32,4096)` with `fp32_dest_acc_en=True` and `use_2d_core_grid=True`.

## Other potential issues (not needed to explain the report)

- **Wrong input and output tile indices when a core owns more than one row (silent corruption,
  #56908).** These do not cause the hang, because the hang stops the program first. Any fix for 1a
  and 1b exposes them, so Option B must include their fix. Details:

  - Reader start offset is `x * Wt + y * tiles_per_core_y`:
    [layernorm_pre_all_gather_program_factory.cpp:739](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L739-L739).
    Core `x` owns rows `x * tiles_per_core_x ... x * tiles_per_core_x + tiles_per_core_x - 1`, so the
    start should be `x * tiles_per_core_x * Wt + y * tiles_per_core_y`.
  - Inside a core, the reader increments the tile index by 1 across row boundaries:
    [reader_layernorm_preallgather_2d.cpp:67-103](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp#L67-L103).
    After each local row it must skip `Wt - tiles_per_core_y` tiles. The reader only receives the
    per-core width (runtime arg `Wt = tiles_per_core_y`, factory line 746), so it needs the full row
    width as a new argument.
  - Writer start offset is `x * out0_tiles`:
    [layernorm_pre_all_gather_program_factory.cpp:740](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp#L740-L740).
    It should be `x * tiles_per_core_x * out0_tiles`. With `tiles_per_core_x = 4`, merge core `x = 1`
    would write output tiles 1..4, overlapping core 0's tiles 1..3.

  Several open community PRs target #56908 (for example #57156, #57650, #58038). None is merged.
- **LayerNorm through the 2D factory produces the wrong output width.** The factory hard-codes
  `out0_tiles = 1` (line 472) and the 2D compute kernel computes only sum(x²). For
  `LayerNormDistributedType::LAYERNORM`, `compute_output_specs` makes the output 2 tiles wide
  ([layernorm_pre_all_gather_device_operation.cpp:83-89](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp#L83-L89)),
  and validation does not reject LAYERNORM with the 2D grid
  ([layernorm_pre_all_gather_device_operation.cpp:68-76](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp#L68-L76)).
  The public `layer_norm_pre_all_gather` never sets the 2D option
  ([layernorm_pre_all_gather.cpp:41-50](ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/layernorm_pre_all_gather.cpp#L41-L50)),
  so only a direct `ttnn::prim` call can reach this. The issue title names `layer_norm_pre_all_gather`,
  but that op cannot take the 2D path today.
- **The zero tile is never popped** on merge cores (compute waits at line 125 and does not pop).
  This is harmless on hardware: the firmware clears all buffer counters after each kernel run
  ([brisc.cc:556](tt_metal/hw/firmware/src/tt-1xx/brisc.cc#L550-L556) triggers
  [trisc.cc:97-105](tt_metal/hw/firmware/src/tt-1xx/trisc.cc#L97-L105)). A buffer-balance checker
  such as the emulator's would still report it. A per-row merge can keep waiting on the same tile
  each row, as long as nothing pops it inside the loop.
- **Mid-kernel `compute_kernel_hw_startup`** in the merge block (line 130, TODO #52395). With a
  per-row merge this full re-initialization would run once per row; the following square and reduce
  helpers re-init their own operations, but my guess is that a targeted reconfiguration is safer.
- **Grid axis use.** `cores_x` is bounded by `grid.y`, not `grid.x`
  (factory line 493-494), and the core range puts `cores_x` on the x axis (line 505). This is safe
  on Wormhole and Blackhole, where `grid.x >= grid.y`, but it leaves cores unused along x.
- **Stale test comment.** "The 2D-grid path activates when shape[-2] == 128" at
  [test_distributed_rmsnorm_allgather.py:116](tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_rmsnorm_allgather.py#L116-L116)
  no longer matches the Llama 70B Galaxy caller, which never sets the 2D option.

## Remaining uncertainty

- No hardware run. The hang location for `(1,1,1024,1024)` is derived from buffer counts. A
  tt-triage capture should show every compute unpack thread in `llk_wait_tiles` on the reduce scaler
  buffer, and the merge-core writers in `wait_front` on `out_final`.
- The reserved SRAM base of 105664 B is inferred from the two reported totals, not read from the
  hardware abstraction layer.
