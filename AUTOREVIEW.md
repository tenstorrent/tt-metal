# Auto-review report

- **Target:** all commits on branch `fplavec/qsr_untilize_with_unpadding` relative to its merge-base with `origin/main` (`git diff 8d8b1ab0df9 73b83ff731a`, 6 files).
- **BASE:** `8d8b1ab0df92698fe328ff71f7831828f3744561`
- **BASELINE** (commit the fixes are based on): `73b83ff731a90791cb6d7fea4642e04568da7add`
- **Date:** 2026-10-26

## 1. Summary per reviewer

| Reviewer | Reviews | Ended with |
|---|---|---|
| 1 | 3 | `NO FURTHER COMMENTS` |
| 2 | 2 | `NO FURTHER COMMENTS` |
| 3 | 2 | `NO FURTHER COMMENTS` |

No reviewer found a bug, a performance regression or a security problem. All issues were low severity.

## 2. All issues raised

### Reviewer 1

1. **Comment understated when a tile row can start past the output end** (category 5). The block-interleaved writer comment said the padded height can exceed the output height "by more than one tile row", but the main case (padded 64, output 32) differs by exactly one tile row. [writer_unary_stick_layout_wh_multicore_metal2.cpp:80-82](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp#L80-L82)
   **Fixed:** now "by one tile row or more", same lines.
2. **`aligned_page_size` comment applied to both same-shard writers, but only one reads it** (category 5). [untilize_with_unpadding_multi_core_sharded_program_factory.cpp:239-245](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L239-L245)
   **Fixed:** first by naming the batch-rows writer in the comment, then (reviewer 2, issue 1) by passing the compile-time arg only to the batch-rows writer, same lines.
3. **No program-cache test for same-shard-type sharded output** (category 11). The output address now reaches the writer through the tensor binding instead of the borrowed DFB, and no test covered a cache hit with a moved output shard.
   **Fixed:** new `multi_core_sharded_to_sharded` case (HEIGHT_SHARDED in and out) in [test_untilize_with_unpadding.py:1073-1137](tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py#L1073-L1137).
4. **Validator comments named the accessor class used by a distant kernel** (category 9). Three copies in the device operation.
   **Fixed:** they now state the property ("each core writes its own output shard in place, so that shard must be in the core's own SRAM (BufferType::L1)"): [untilize_with_unpadding_device_operation.cpp:167-173](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp#L167-L173), [246-248](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp#L246-L248), [300-302](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp#L300-L302).
5. **Wording against project conventions** (category 13): "only toucher" (same family as terms banned in CLAUDE.md), DFB wording "nothing downstream drains it" for memory that is no longer a DFB, and "L1-to-L1" for Tensix SRAM.
   **Fixed:** [untilize_with_unpadding_multi_core_sharded_program_factory.cpp:51-52](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L51-L52), [226-229](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L226-L229), [writer_unary_unpad_batch_rows_sharded.cpp:26](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp#L26-L26), [writer_unary_unpad_width_16_sharded.cpp:31](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp#L31-L31).
6. **Pronouns lost their antecedents after the fixes for issues 2 and 4** (category 5).
   **Fixed:** the validator comment now says "The same-shard-type writer also copies ..." ([untilize_with_unpadding_device_operation.cpp:169-173](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp#L169-L173)); the factory sentence with the other pronoun was later removed by the fix for reviewer 2, issue 1.

### Reviewer 2

1. **Comment said the width-16 writer does not read `aligned_page_size`, while the code still passed it** (category 5 and 13).
   **Fixed:** the compile-time arg is set only in the batch-rows branch: [untilize_with_unpadding_multi_core_sharded_program_factory.cpp:239-245](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp#L239-L245). The width-16 kernel never reads it.
2. **Comment described the change rather than the code** (category 8): a sentence explaining why `dfb::untilize_out` supplies the tile size, when that is the only DFB the kernel binds.
   **Fixed:** sentence removed: [writer_unary_unpad_width_16_sharded.cpp:19-22](ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp#L19-L22).
3. **The block-crop test depends on a factory-selection heuristic and does not check which factory ran** (category 9 and 11). [test_untilize_with_unpadding.py:1047-1054](tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py#L1047-L1054)
   **Out of scope** for the check: no Python-visible way to observe the selected program factory exists, and adding one goes well beyond this branch. Project tests describe factory selection in comments without checking it ([test_tilize.py:509-515](tests/ttnn/unit_tests/operations/data_movement/test_tilize.py#L509-L515), [test_pad.py:774-776](tests/ttnn/unit_tests/operations/data_movement/test_pad.py#L774-L776)). The kernel cache on Wormhole and on the Quasar simulator showed that these cases compile `writer_unary_stick_layout_wh_multicore_metal2`, so they reach the fixed writer today. The reviewer accepted this.
   Reviewer 3 raised this issue again (its issue 1) and suggested wording the docstring as intent. That part is **Fixed**: the docstring now reads "The inputs are 64 tiles wide so that they take the block-interleaved factory, ...", same lines. Reviewer 3 accepted the out-of-scope verdict for the check.

### Reviewer 3

No new issues. Its only finding repeated reviewer 2, issue 3 (see above).

## 3. Unresolved issues

No unresolved issues.

## 4. Tests and experiments

All runs used the working tree with the review fixes. Formatting was checked with clang-format 19.1.4 and black after each round; both were clean.

| After | Build | Target | Tests | Result |
|---|---|---|---|---|
| Reviewer 1 round 1 fixes | `build_metal.sh -e --enable-fake-kernels-target`, passed | Wormhole n150 | program-cache test (5 cases, including the new sharded-to-sharded case), width-16, height-sharded, width-sharded, block-crop tests | 28 passed |
| Reviewer 1 round 1 fixes | same | Quasar simulator (craq-sim) | program-cache test (5), width-16, block-crop tests | 12 passed |
| Reviewer 2 round 1 fixes (host change) | rebuilt, passed | Wormhole n150 | the 28 above plus `test_untilize_with_unpadding_sharded_multi_batch_unpadding_regression` (8) | 36 passed |
| Reviewer 2 round 1 fixes | same | Wormhole n150 | `test_sharded.py -k untilize_with_unpadding` | 12 passed |
| Reviewer 2 round 1 fixes | same | Quasar simulator | program-cache test (5), width-16, height-sharded, width-sharded tests | 23 passed |
| Reviewer 3 round 1 fix (docstring only) | not needed | Wormhole n150 | block-crop test | 5 passed |

No test failed or was skipped in these runs. The full test files (`test_untilize_with_unpadding.py`, `test_untilize.py`) were not rerun after the review fixes; they were run on the committed code before this review. The review fixes changed comments, one test, and where one compile-time arg is passed.

## 5. Changed files

All fixes are uncommitted edits in the working tree (`git diff --stat 73b83ff731a`); no files were created:

- `tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py`
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_sharded_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore_metal2.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp`
- `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp`
