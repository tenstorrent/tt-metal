# Plan: topk_xl for BFLOAT16 values and true UINT32 row-major indices

This is the fresh plan after stopping the previous WIP. Treat the current
topk_xl worktree changes as exploratory, not as the implementation baseline.
The source of truth for existing LLK behavior is the pristine Blaze copy under
`/tmp/topk_xl_blaze`.

The chosen implementation path is:

1. Use Blaze's existing fused local sort for each K-sized row chunk.
2. Immediately split each sorted chunk into unfused value/index DST regions.
3. During that split, create true row-major UINT32 indices.
4. Use Blaze's existing unfused merge/rebuild for the rest of the row
   reduction.
5. Pack BFLOAT16 values and UINT32 indices; the writer only writes pages.

No other implementation path is in scope for v1.

## 1. Op requirements

Hard requirements:

- Architecture: Blackhole only.
- Input layout: row-major only.
- Input value dtype: BFLOAT16 only.
- Output layout: row-major only.
- Output value dtype: BFLOAT16.
- Output index dtype: UINT32.
- Index semantics: each output index is the true zero-based row-major position
  along the ranked row.
- Large-row support: rows with `N > 65535` must work.
- Shape semantics: flatten all leading dimensions into `M` logical rows, each
  with length `N`; compute top-K independently along the last dimension.
- Parallelism: one logical row is processed by one Tensix core at a time; cores
  may loop over multiple rows.
- No cross-core reduction is required by this TTNN op.
- Execution model: rows are streamed as row work items through
  reader -> compute -> writer. Use double-buffered CBs so the reader can load
  the next row/chunk while compute sorts/reduces the current work item and the
  writer drains the previous result.
- Writer constraint: the writer should only move already-correct output pages.
  No writer-side RISC-V index decode or semantic reconstruction.

Initial v1 constraints:

- `dim = -1`, `largest = true`, `sorted = true`.
- K uses the proven Blaze LLK sizes. Prefer public support for `K = 1024` and
  `K = 2048`; keep `K = 512` as LLK coverage unless product needs it.
- Require `N % K == 0` in v1.
- Require row-major input and row-major output tensors.
- Do not support tiled input/output tensors in v1.

Correctness contract:

- For distinct BFLOAT16 values, output indices must match the exact top-K
  row-major positions.
- With duplicate BFLOAT16 values, each returned index must point to an input
  value equal to the returned output value. Exact PyTorch tie order is not part
  of v1 unless explicitly added and tested.
- Output values must be BFLOAT16-clean. Low FP32 mantissa remnants from fused
  intermediate words must not leak into the public value output.

## 2. Pristine Blaze sources reviewed

Fetched local reference files:

- `/tmp/topk_xl_blaze/ckernel_sfpu_topk_xl.h`
- `/tmp/topk_xl_blaze/llk_math_eltwise_unary_datacopy_topk_xl_copy.h`
- `/tmp/topk_xl_blaze/llk_unpack_A_topk_xl_copy.h`
- `/tmp/topk_xl_blaze/llk_math_topk_xl_copy_api.h`
- `/tmp/topk_xl_blaze/llk_math_eltwise_unary_sfpu_topk_xl.h`
- `/tmp/topk_xl_blaze/llk_unpack_A_topk_xl_copy_api.h`
- `/tmp/topk_xl_blaze/topk_xl.h`
- `/tmp/topk_xl_blaze/distributed_topk_op.hpp`
- `/tmp/topk_xl_blaze/distributed_topk_op.py`
- `/tmp/topk_xl_blaze/test_distributed_topk.py`
- `/tmp/topk_xl_blaze/test_distributed_topk_keep_values.py`

## 3. What the current Blaze LLKs do

### 3.1 Fused representation

Blaze local sort uses a fused 32-bit DST word:

```text
[ BFLOAT16 value in high 16 bits | u16 index fragment in low 16 bits ]
```

This is not full-precision FP32 value storage. The value contract is BFLOAT16:
the high 16 bits carry the value payload, and the low 16 bits are available for
index/tie-break data. This is acceptable for our op because public values are
BFLOAT16.

The sort compares the 32-bit fused word. The BFLOAT16 value dominates the
comparison; the low index bits act as a deterministic tie-breaker.

### 3.2 FP32 path clarification

Blaze tests pass BFLOAT16 input tensors, then compile with:

```text
fp32_dest_acc_en = True
unpack_to_dest_fp32_cb_indices = [0, 1, 2, 3]
```

That selects 32-bit destination/unpack-to-dest machinery for the copy path.
For full tiles, the copy path clears the low 16 bits, and
`topk_xl_add_lsb_indices` then ORs a 16-bit index into those bits.

So "FP32" in Blaze is mostly a DST/container and transport detail for this op.
It does not mean the fused local sort preserves full FP32 values. The meaningful
value precision remains BFLOAT16.

### 3.3 `topk_xl_copy_tile`

`topk_xl_copy_tile_init(cb)` sets unpack/math state for the special copy path.
`topk_xl_copy_tile<K>(cb, dst_start, in_tile_base, num_elements)` copies up to K
elements from one or two input tiles into DST and pads inactive elements with
negative infinity.

Supported K:

- `K=512`: one half-tile of live data.
- `K=1024`: one tile.
- `K=2048`: two tiles.

The copy path only moves values. It does not create true row-major UINT32
indices.

### 3.4 `topk_xl_add_lsb_indices`

`topk_xl_add_lsb_indices<K, core_id>` ORs a 16-bit index fragment into the low
half of each fused word.

Blaze comments describe the field as:

- bits `[15:11]`: 5-bit core id
- bits `[10:5]`: row/tile-derived field
- bits `[4:0]`: within-row column/lane field

Diagnostics and Blaze tests show this is a tile/SFPU coordinate, not directly
our row-major API index. For `K=1024`, the observed coordinate is equivalent to:

```text
raw = (col << 6) | row
row = raw & 0x1f
col = (raw >> 6) & 0x1f
row_major_idx = ((row >> 4) << 9)
              | ((col >> 4) << 8)
              | ((row & 0xf) << 4)
              |  (col & 0xf)
```

For `K=2048`, the second tile adds a tile-within-sequence bit.

The fused low field is only 16 bits. It can bootstrap local sort, but it cannot
be the final index representation for `N > 65535`.

### 3.5 `topk_xl_local_sort`

The public Blaze local-sort API is fused:

```cpp
topk_xl_local_sort<K>(idst, ascending)
```

It sorts K elements in bitonic order using fused `[bf16 value | u16 index]`
words. This is the local-sort primitive we will use in v1.

We do not need to implement unfused local sort for v1.

### 3.6 `topk_xl_separate_indices`

`topk_xl_separate_indices_init(group_id_bit_shift)` stores a runtime shift in
an SFPU config register. `topk_xl_separate_indices<K, group_id>` converts fused
words into split value/index regions:

- value region keeps the value half;
- index region keeps the low-16 index fragment and ORs in
  `group_id << group_id_bit_shift`.

This is the Blaze bridge from fused to unfused mode. It is close to what we
need, but not sufficient as-is because it does not row-major-decode the tile
coordinate.

### 3.7 Unfused merge/rebuild

`topk_xl_merge<K, false>()` and `topk_xl_rebuild<K, false>()` already support
parallel value/index regions.

The unfused helpers load:

- value rows as FP32 words;
- matching index rows as INT32 words.

Index registers are swapped in lockstep with values, then both regions are
stored back. This is the key existing LLK support we reuse after the immediate
split.

### 3.8 Blaze distributed op behavior

Blaze's `distributed_topk` is a cross-core sharded reduction, not our topology.
Relevant behavior:

- It starts each shard with fused copy, add low-16 indices, and fused local sort.
- Test coverage sweeps `maxN = mult * elements_per_core` with
  `mult in {2, 4, 8, 16, 32, 64, 128}`. For `K=2048, epc=2048`, that reaches
  `maxN = 262144`.
- The host op allows up to `MAX_REDUCTION_STAGES = 7`, i.e. up to 128 shard
  participants in the reduction tree.
- It uses `max_fused_stage = 5`.
- For `elements_per_core = 2048`, the fused threshold is 65536.
- Above that threshold, it calls `topk_xl_separate_indices_init(16)`,
  `topk_xl_separate_indices<K, group_id>`, then continues with
  `topk_xl_init<K, false>()` and unfused merge/rebuild.

Our op should split much earlier: immediately after each local chunk sort. That
prevents the fused low-16 field from becoming the final source of truth.
Blaze proves the split-index/unfused merge path can handle arrays larger than
65535, but it does so with sharded/tile-coordinate semantics. Our op still needs
a row-major split primitive for true row indices.

## 4. What is missing

The missing support for v1 is focused:

Already present in pristine Blaze LLKs:

- Fused BFLOAT16-value local sort:
  `topk_xl_copy_tile`, `topk_xl_add_lsb_indices`, `topk_xl_local_sort<K>`.
- Basic fused-to-unfused split:
  `topk_xl_separate_indices_init`, `topk_xl_separate_indices<K, group_id>`.

Missing LLK functionality to implement:

- `topk_xl_separate_indices_row_major_init(...)`
  - Programs any constants needed for row-major coordinate decode.
  - Programs high-index bit placement for `N > 65535`.
  - Sets ADDR_MODs for the output value/index DST layout expected by
    `topk_xl_merge<K, false>()`.
- `topk_xl_separate_indices_row_major<K>(idst, chunk_id_or_base)`
  - Input: locally sorted fused K-run in DST:
    `[bf16 value hi16 | u16 raw_coord lo16]`.
  - Output: unfused value/index regions:
    `[BFLOAT16/FP32 value words, UINT32 row-major index words]`.
  - Decodes the low-16 tile/SFPU coordinate into row-major within-chunk
    position.
  - Adds the full chunk base, `chunk_id * K`, preserving all bits in UINT32.
  - Stores the index region in the exact layout consumed by
    `load8_rows_x2_unfused` / `store8_rows_x2_unfused`.
  - Leaves the value region clean enough that final public values pack as
    BFLOAT16 with no low-bit index remnants.
- Compute API and LLK API wrappers for the new primitive, mirroring the shape
  of `topk_xl_separate_indices_init` and `topk_xl_separate_indices`.
- A pack/output helper only if generic packing cannot directly emit:
  - BFLOAT16 values from the unfused value region;
  - UINT32 indices from the unfused index region.

Now validated from Blaze LLKs:

- Unfused value/index merge and rebuild for UINT32 indices:
  - `topk_xl_merge<K, false>` and `topk_xl_rebuild<K, false>` preserve
    value/index pairing when fed valid split output in layout
    `[values0, indices0, values1, indices1]`.
  - Focused LLK coverage exists for direct UINT32 companion swaps, two split
    chunks before merge, and merge+rebuild.
  - For K=1024 test stimulus must keep raw bit 5 clear. Raw bit 5 is the
    K=2048 tile-in-sequence bit and is ignored by the K=1024 row-major decoder.

Not needed for v1:

- Unfused local sort. The chosen path deliberately uses the existing fused
  local sort, then splits immediately.
- Writer-side index decode.
- Host-side semantic index decode.

Validation still needed:

- BFLOAT16 `topk_xl_copy_tile` through the same CB contract the TTNN op uses.
- Fused local sort with BFLOAT16 values and low-16 indices.
- New split primitive: values, low index decode, high index bits, output layout.
- Unfused merge/rebuild with UINT32 index regions.
- End-to-end exact indices for `N > 65535`.

## 5. Bottom-up implementation plan

Build from LLK primitives upward. Do not start by wiring the full TTNN op.
Every phase should leave a small test artifact with exact pass/fail criteria.

Use release builds when a full build is needed:

```bash
./build_metal.sh --release
```

Run TTNN op tests through:

```bash
scripts/run_safe_pytest.sh <test path>
```

Use ttsim for SFPU/DST-layout debugging when the silicon result does not explain
which address/layout step is wrong.

### Phase 0: clean baseline and source control

Goal: make sure we are building on the intended architecture.

Actions:

- Do not continue the stopped writer/index-decode WIP.
- Treat `/tmp/topk_xl_blaze` as the pristine LLK reference.
- Keep current WIP only as diagnostic history.
- Confirm the canonical LLK headers in the worktree are either pristine Blaze
  copies or intentionally modified for the new row-major split primitive.
- Remove or ignore exploratory IEEE/FP32-value assumptions. Public values are
  BFLOAT16.

Pass condition:

- The plan and worklog identify the immediate-split architecture as the only v1
  implementation path.
- No final design step depends on writer-side or host-side semantic index
  decode.

### Phase 1: prove existing Blaze LLK blocks in isolation

Goal: establish that the imported blocks we plan to reuse are green before
adding new behavior.

Artifacts:

- `tt_metal/tt-llk/tests/sources/topk_xl_copy_test.cpp`
- `tt_metal/tt-llk/tests/sources/topk_xl_test.cpp`
- `tt_metal/tt-llk/tests/python_tests/test_topk_xl_copy.py`
- `tt_metal/tt-llk/tests/python_tests/test_topk_xl.py`

Tests:

- BFLOAT16 `topk_xl_copy_tile` copies the full active K region through the same
  CB/unpack contract the op will use.
- `topk_xl_add_lsb_indices` creates a deterministic low-16 coordinate.
- `topk_xl_local_sort<K>` produces top-K sorted fused words for K 1024 and
  2048. Keep K 512 coverage if public support remains possible.
- Fused values compare exactly after BFLOAT16 truncation.
- Existing unfused `topk_xl_merge<K, false>` and
  `topk_xl_rebuild<K, false>` preserve value/index pairing when the value and
  UINT32 index regions are manually seeded.

Pass condition:

- Existing fused local sort is trusted for BFLOAT16 values.
- Existing unfused merge/rebuild is trusted with independent UINT32 indices.
- Any failure here blocks new LLK work; do not debug it through the TTNN op.

### Phase 2: add row-major split LLK

Goal: implement the only missing LLK block for v1.

New LLK/API surface:

- SFPU primitive:
  `_topk_xl_separate_indices_row_major_init_(...)`
- SFPU primitive:
  `_topk_xl_separate_indices_row_major_<K, ...>()`
- LLK API wrapper:
  `llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init(...)`
- LLK API wrapper:
  `llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major<K>(...)`
- Compute API wrapper:
  `topk_xl_separate_indices_row_major_init(...)`
- Compute API wrapper:
  `topk_xl_separate_indices_row_major<K>(idst, ...)`

Primitive contract:

```text
input fused word  = [bf16 value hi16 | u16 raw_coord lo16]
output value word = BFLOAT16 value payload in unfused value region
output index word = UINT32 row_major_index in unfused index region
```

Required behavior:

- Decode low-16 tile/SFPU coordinate into row-major within-chunk position.
- Add full chunk base: `chunk_id * K`.
- Preserve all UINT32 bits for `N > 65535`.
- Store index words in the layout consumed by `load8_rows_x2_unfused`.
- Store value words in the matching unfused value region.
- Do not require writer-side or host-side semantic decode.

First LLK tests:

- Seed DST with known fused words, without copy or sort.
- Run only `topk_xl_separate_indices_row_major`.
- Pack value and index regions.
- Check exact BFLOAT16 values and exact UINT32 indices.
- Sweep:
  - K 1024 and 2048;
  - `chunk_id = 0`;
  - `chunk_id` such that `chunk_id * K > 65535`;
  - selected raw coordinates that cover row/col/tile boundary bits.

Pass condition:

- The split primitive alone converts every tested raw coordinate to the exact
  row-major UINT32 index.
- Packed values are BFLOAT16-clean.
- Packed index layout matches generic UINT32 packing, with no postprocessing.

### Phase 3: local chunk LLK pipeline

Goal: prove one complete K chunk bottom-up before any multi-chunk reduction.

Pipeline:

```text
BFLOAT16 input chunk
  -> topk_xl_copy_tile
  -> topk_xl_add_lsb_indices
  -> topk_xl_local_sort<K>
  -> topk_xl_separate_indices_row_major<K>
  -> pack values + indices
```

Tests:

- Distinct BFLOAT16 ramp where value bits uniquely identify original positions.
- Random BFLOAT16 values with duplicate quantized values.
- K 1024 and 2048.
- `chunk_id = 0` and a chunk id above the 65535 boundary.

Checks:

- `indices` are exact for distinct values.
- `gather(input, indices - chunk_base) == values` for the local chunk.
- Duplicate BFLOAT16 cases satisfy value/index self-consistency even if tie
  order differs from PyTorch.

Pass condition:

- One chunk produces correct BFLOAT16 values and true UINT32 row-major indices
  with no host or writer decode.

### Phase 4: two-chunk unfused merge LLK pipeline

Goal: prove that immediate split output feeds existing unfused merge/rebuild.

Pipeline:

1. Run the Phase 3 local chunk pipeline for chunk 0.
2. Run the Phase 3 local chunk pipeline for chunk 1.
3. Place both split chunks in the unfused DST layout:
   `[values0, indices0, values1, indices1]`.
4. Call `topk_xl_init<K, false>()`.
5. Call `topk_xl_merge<K, false>()`.
6. Call `topk_xl_rebuild<K, false>()` if the next stage would need it.
7. Pack the top-K value/index result.

Tests:

- K 1024 and 2048.
- Distinct BFLOAT16 values across both chunks.
- Duplicate BFLOAT16 values across chunk boundaries.
- Second chunk base that crosses 65535.
- For K=1024, host-seeded fused tests must use hardware-valid raw coordinates
  with bit 5 clear; `topk_xl_add_lsb_indices` naturally produces valid
  coordinates in the real pipeline.

Pass condition:

- Final top-K values match the reference.
- Final indices are exact row-major UINT32 positions.
- `gather(row, indices) == values`.

### Phase 5: multi-chunk LLK/device reduction

Goal: prove the single-core row algorithm without TTNN host complexity.

Device-level test shape:

- One logical row.
- `N = chunks * K`.
- chunks in `{4, 8, 32, 64}` as feasible.
- Include at least one case where `N > 65535`.

Pipeline:

- Process each K chunk through fused local sort and immediate split.
- Use only unfused merge/rebuild for all inter-chunk reductions.
- Pack final BFLOAT16 values and UINT32 indices.

Pass condition:

- Exact indices for distinct BFLOAT16 values.
- Self-consistent values/indices for random BFLOAT16 values with duplicates.
- No writer decode.

### Phase 6: TTNN op skeleton

Goal: add host/device integration only after the LLK/device core is proven.

Host validation:

- Blackhole.
- Input layout row-major.
- Input dtype BFLOAT16.
- Output values row-major BFLOAT16.
- Output indices row-major UINT32.
- `dim = -1`.
- `largest = true`.
- `sorted = true`.
- Supported K.
- v1 `N % K == 0`.

Output specs:

- `values` shape `[M, K]` after flattening leading dims.
- `indices` shape `[M, K]`.
- Values dtype BFLOAT16.
- Indices dtype UINT32.
- Both row-major.

Device program:

- Work is row-by-row. A row work item is one logical row assigned to a core.
- Reader streams row-major BFLOAT16 chunk pages for the current row work item
  into double-buffered input CBs.
- Compute consumes input CB pages, runs the proven single-row algorithm, and
  pushes final value/index pages into double-buffered output CBs.
- Writer drains output CB pages to row-major value and index tensors.
- Reader, compute, and writer must be able to overlap:
  - reader fills the next row/chunk page;
  - compute sorts/reduces the current row/chunk;
  - writer stores the previous row result.
- Cores loop over assigned rows, but each row remains a complete independent
  work item. Do not mix rows inside a single reduction.
- CB depth should be at least 2 for the input and output streams used between
  reader/compute/writer. Increase only if profiling shows the pipeline is
  starved by NoC or pack latency.

Pass condition:

- `M=1`, one row, one chunk passes through the TTNN op.
- Then `M=1`, multiple chunks passes.
- Then multi-row `M > grid` row-looping passes.

### Phase 7: TTNN correctness suite

Goal: validate public op semantics.

Required tests:

- Distinct BFLOAT16 bit-pattern ramp with exact indices.
- Random BFLOAT16 data with duplicates.
- Verify `gather(input, indices) == values`.
- Large-row exact-index case, for example `M=1, N=131072, K=2048`.
- Target workload shape, even though it is below 65535:
  `M=640, N=51200, K=2048`.
- Multi-row loop where `M` is larger than the core count and not evenly
  divisible by it.
- K sweep for public supported K values.

Pass condition:

- All public semantic tests pass through `scripts/run_safe_pytest.sh`.

### Phase 8: cleanup and performance

Goal: make the implementation maintainable after correctness is stable.

Actions:

- Confirm L1/DST footprint for K=2048 after immediate split.
- Ensure no writer-side or host-side decode remains.
- Remove exploratory comments and code from the previous WIP.
- Keep ttsim-only instrumentation out of final headers.
- Record LLK cycle counts and TTNN op timings.

Pass condition:

- Code path is minimal: fused local sort, immediate row-major split, unfused
  merge/rebuild, plain writer.
- Plan/worklog match the final implementation.

## 6. Current Missing LLK Block

The local sort, row-major index split, and unfused merge/rebuild LLKs are green
for the internal DST layout, but the public op needs a final output
materialization step that the LLK suite has not validated yet:

```text
sorted/split DST layout
  -> BFLOAT16 values in true row-major output order
  -> UINT32 indices in true row-major output order
```

The existing LLK tests pack all result tiles with a single Float32/raw output
format. That proves value/index pairing, but it does not prove the mixed public
contract: BFLOAT16 values plus UINT32 indices, both row-major. TTNN bring-up
confirmed this gap: plain `pack_tile` gives tile/raw output, while
`pack_rows`/`pack_untilize_dest` do not correctly consume the current topk_xl
sorted/split DST layout as BFLOAT16 values.

Next green milestone:

- Add an LLK-level final-output primitive that rewrites or packs the topk_xl
  result layout into packer-consumable row-major values and indices.
- Validate it in LLK tests for:
  - one chunk, K=1024/2048;
  - two chunks after unfused merge/rebuild;
  - BFLOAT16 value output semantics;
  - UINT32 index output semantics, including chunk bases above 65535.
- Wire TTNN compute to use only that proven final-output primitive before the
  pure-DMA writer.
