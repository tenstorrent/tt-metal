# AutoReview: `ttnn.gather` on ROW_MAJOR BLOCK/WIDTH-sharded tensors with an unaligned shard row (issue #62153)

Review of all changes on branch `fplavec/gather-rm-subaligned-shard-width` (commit `96e5e0f8cf8`), relative to `main` at `8ac40e3f904`. Fix and diagnosis are in [AUTOFIX.md](AUTOFIX.md) and [AUTODEBUG.md](AUTODEBUG.md). Three reviewers, each starting with no context, reviewed the changes one after the other. All fixes from the review are uncommitted edits on top of `96e5e0f8cf8`. All results below are from a Wormhole n150 in this machine.

The review found two real bugs that the branch had not fixed, and both are now fixed and tested on the device. All three reviewers ended with NO FURTHER COMMENTS.

## 1. Reviewers

| Reviewer | Reviews | How it ended |
|---|---|---|
| 1 | 4 of 5 | NO FURTHER COMMENTS |
| 2 | 3 of 4 | NO FURTHER COMMENTS |
| 3 | 2 of 3 | NO FURTHER COMMENTS |

One process slip: in the reply to reviewer 3, the parent agent said that earlier reviews had reached the same conclusion on its F2. The review process says not to tell later reviewers about earlier ones. The remark named no finding.

## 2. All issues raised

### Reviewer 1

- **F1. AUTODEBUG.md and AUTOFIX.md are committed at the repo root, and the commit message does not describe the change.** Out of scope: these are working notes. **Remove both files (and this one) and rewrite the commit message before the PR is opened.**
- **F2. Wrong values from `RmSingleRowMultiCore` when the input row is not a multiple of 16 B.** The input CB has two slots, and the second slot started at an unaligned address. **Fixed**: the slot size is now rounded up to the buffer alignment, in [gather_program_factory.cpp:577-589](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L577-L589). On the device before the fix, all-interleaved bf16 with W=1985 gave 3964 of 7936 values wrong. New test: `RM_multicore_interleaved_unaligned_input_row`.
- **F3. The output shard spec was derived again after the index tensor had moved to interleaved memory.** This gave a (32, 32) output shard instead of (32, 4), and dropped the shard orientation. **Fixed**: the wrapper now passes its own spec to the device op, in [gather.cpp:397-418](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L397-L418). New test: [test_gather_rm_width_sharded_index_no_spec_out](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L633-L675). It checks shard shape, orientation and values, and it failed on the old build.
- **F4. The test IDs and comment said "uneven split", but 1984 / 64 is an even split.** **Fixed**: the IDs are now `..._unaligned_column_split`, and the comment is at [test_universal_input_tm_gather.py:527-528](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L527-L528).
- **F5. No test covered the column limit for the last core** ([gather_program_factory.cpp:656](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L656-L656)). **Fixed**: added `RM_multicore_interleaved_odd_index_width`, with an index width of 1985.
- **F6. Tests missing.** Partly fixed: the shard-shape check is in the F3 test. Out of scope:
  - A test of the device-op hard error: it needs a Python binding of `ttnn::prim::gather`, which does not exist.
  - DRAM-sharded RM cases: the test file has no helper code for DRAM shard grids.
- **F7. `has_subaligned_shard_row` existed twice, once in slice and once in gather.** **Fixed**: one shared helper in [common.hpp:443-447](ttnn/cpp/ttnn/operations/data_movement/common/common.hpp#L443-L447) and [common.cpp:911-924](ttnn/cpp/ttnn/operations/data_movement/common/common.cpp#L911-L924). Slice now uses it through [slice.cpp:48](ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp#L48-L48). The parent agent first rejected this finding with a wrong reason; the reviewer showed that both slice call sites already limit the check to BLOCK/WIDTH sharding.
- **F8. Tensors were always moved to SRAM, even DRAM-sharded ones.** **Fixed**: `interleaved_like` keeps each tensor's buffer type ([gather.cpp:394-396](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L394-L396)). No test covers it, for the same DRAM reason as F6.
- **F9. A new test line was over 120 characters.** **Fixed**: black reformatted it.

### Reviewer 2

- **F1. The moved helper comment described kernel internals that do not apply to gather.** **Fixed**: it now states only the requirement ([common.hpp:443-446](ttnn/cpp/ttnn/operations/data_movement/common/common.hpp#L443-L446)).
- **F2. The float32 runs of the shard-width-12 cases have 48 B rows, which are aligned.** **Fixed**: the comment now says those runs are aligned controls ([test_universal_input_tm_gather.py:456-459](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L456-L459)).
- **F3. Test comments described how the factory splits work, not why the inputs were chosen.** **Fixed**: they now describe the inputs ([test_universal_input_tm_gather.py:527-528](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L527-L528), [test_universal_input_tm_gather.py:547-548](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L547-L548)).
- **F4. The wrapper comment had unclear back-references ("that are not").** **Fixed**: [gather.cpp:385-389](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L385-L389).
- **F5. No BLOCK-sharded sub-aligned input or index test.** Partly fixed: added `RM_block_subaligned_shard_row_input_to_interleaved` and `RM_block_subaligned_shard_row_index_to_interleaved`. Out of scope: DRAM cases, a gtest, and gather on a dim other than the last (the issue covers only dim=-1). Reviewer 3 also raised the gtest, as its F2.
- **F6. The comment on the BLOCK index case did not say why it uses a 14-column shard.** **Fixed**: [test_universal_input_tm_gather.py:505-507](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py#L505-L507).

### Reviewer 3

- **F1. Extra host work on every ROW_MAJOR gather.** Gather has no cache-hit validation, so the cache-miss validation, and the output-spec calculation inside it, ran on every call. **Fixed**: the output spec is now computed only when the output can be sharded, in [gather.cpp:397-400](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L397-L400) and [gather_device_operation.cpp:139-147](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L139-L147).
- **F2. No gtest reaches the hard error in the device op.** Out of scope: no current caller can reach that error, and it would need a new C++ test target. Reviewer 2 raised the same point.
- **F3. The comment named only the index tensor as the source of the derived output spec.** **Fixed**: it now names the index and input tensors.

## 3. Unresolved issues

No unresolved issues. The open items are the out-of-scope ones above. The only one that needs action is reviewer 1's F1: remove AUTODEBUG.md, AUTOFIX.md and this file, and rewrite the commit message before the PR.

## 4. Tests and experiments (Wormhole n150)

- **Device experiment, all tensors interleaved, before the fixes:**
  - bf16 input W=1985: 3964 of 7936 values wrong.
  - f32 input and index W=1985: 3970 of 7940 values wrong.
  - Index width 1985: correct.
- **New tests on the old build:** `unaligned_input_row` failed for both dtypes, and the F3 no-spec test failed (wrong shard shape).
- **Final run after the last rebuild:** `test_universal_input_tm_gather.py`, `test_gather_codegen_routing.py`, `tests/ttnn/unit_tests/operations/data_movement/test_gather.py` and `test_universal_input_tm_slice.py`. Result: 312 passed, 1 xfailed. The xfail is the existing codegen trace test for #46533.
- **Formatting:** black and git-clang-format report no changes.
- **Not tested:** DRAM-sharded RM tensors, and Blackhole.
- **AUTOFIX.md is now out of date:** it does not describe the review fixes. This only matters if it is used to draft the PR description.

## 5. Changed files (uncommitted, no new files other than this report)

```
tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py | 106 ++++++---
ttnn/cpp/ttnn/operations/data_movement/common/common.cpp                                  |  15 ++
ttnn/cpp/ttnn/operations/data_movement/common/common.hpp                                  |   6 +
ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp          |  30 +--
ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.hpp          |   7 -
ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp           |   8 +-
ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp                                  |  58 ++---
ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp                                    |  14 +-
```
