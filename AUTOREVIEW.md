# Auto-review: `fplavec/quasar-uplift-slice`

- **Target:** all commits on branch `fplavec/quasar-uplift-slice` that are not on `main`. The net change is the
  Quasar uplift of the slice op: 13 files under `ttnn/cpp/ttnn/operations/data_movement/slice/`.
- **BASE:** `8d8b1ab0df92698fe328ff71f7831828f3744561` (merge base with `origin/main`).
- **BASELINE:** `5f80e7410b25c0a4844cfb8b832985f6664f49f6` (`HEAD` when the review started). All fixes below are
  uncommitted edits on top of it.
- **Date:** 2026-10-26.

## 1. Summary per reviewer

| Reviewer | Reviews done | Ended with |
|---|---|---|
| 1 | 3 of 5 | `NO FURTHER COMMENTS` |
| 2 | 3 of 4 | `NO FURTHER COMMENTS` |
| 3 | 2 of 3 | `NO FURTHER COMMENTS` |

The reviewers raised 20 distinct issues:
- 16 were fixed. For one of these (R1-F2), one part was judged out of scope.
- 1 was fixed and then withdrawn: R1-F1, whose fix also made R1-F9 moot (see below).
- 2 were out of scope.
- 1 became moot (R1-F9).
- 0 were rejected.
- 0 are unresolved.

Five more findings repeated an earlier issue.

## 2. All issues raised

### Reviewer 1

1. **R1-F1 (medium): same-core NoC reads in the sharded reader.** The height-sharded reader read rows over the NoC
   even when the rows were on its own core. The porting guide says such a read can hang or be dropped on the
   physical Quasar emulator.
   **Fixed, then withdrawn after the review:** a Quasar-only copy with `local_l1_copy()` was added for rows held by
   the reading core, and later removed at the user's request. No symptom was ever seen: the plain NoC reads pass all
   7 Quasar sharded cases on craq-sim, both before the copy was added and after it was removed. Adding the copy only
   to this reader conflicted with the guide's rule to apply such fixes only after the symptom appears, which is also
   the reason the other readers have none (R3-F1). The open question is now one follow-up for all slice readers
   (see R3-F1). The reader issues the same NoC reads as at the original port
   ([slice_reader_unary_unpad_dims_rm_sharded.cpp:50-89](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L50-L89)).
2. **R1-F2 (medium): no test for UINT32 or tile-layout UINT16 slices, and no Quasar CI run.**
   **Fixed (tests):** added
   [test_slice_unsigned_int](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L688-L711)
   (tile, row-major, strided 4D and ND) and
   [test_slice_tensor_args_unsigned_int](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L714-L729).
   Both compare exactly against torch.
   **Out of scope (CI):** the Quasar lists in `tests/scripts/quasar/` hold no `tests/ttnn/unit_tests` entries, and
   changing CI configuration is outside an in-place op uplift. The command and node ids for a rerun are in section 4.
3. **R1-F3 (low): stale comments in the sharded factory header.** They still described borrowed DFBs.
   **Fixed:** [slice_program_factory_rm_sharded.hpp:16-23](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp#L16-L23)
   (shortened again for R3-F4).
4. **R1-F4 (low): misleading kernel comment, and a later error for DRAM input.** The comment said "this core's
   shard of the input", which is wrong when the reading core holds no input shard. Also, a DRAM height-sharded input
   now failed at kernel compile instead of on the host. `BASE` rejected such an input on the host through the
   runtime check for borrowed DFBs in
   [dfb.cpp:50-55](tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L50-L55); the port
   replaced the borrowed DFBs with tensor bindings, so that check no longer applied.
   **Fixed:** comment at
   [slice_reader_unary_unpad_dims_rm_sharded.cpp:46-49](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L46-L49),
   and a host check at
   [slice_program_factory_rm_sharded.cpp:273-279](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp#L273-L279)
   (refined for R1-F8 and R2-F4).
5. **R1-F5 (low): format helper comment described other files.** The usage contract was missing from the header.
   **Fixed:** contract at
   [slice_device_operation.hpp:27-30](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.hpp#L27-L30);
   the body comment at
   [slice_device_operation.cpp:95-107](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp#L95-L107)
   now gives only the reason for the branch.
6. **R1-F6 (low): the `OUT_SHARDED` pop in the slice tile writer had no effect.** No slice factory defines
   `OUT_SHARDED`, and the added comment ("sharded in place") was wrong for slice.
   **Fixed** by dropping the edit. The writer is identical to `BASE` again
   ([writer_unary_interleaved_start_id.cpp:25-27](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp#L25-L27)).
7. **R1-F7 (low): test text said "CB addresses".** It described the sharded cache hit as patching CB addresses.
   **Fixed:** [test_slice.py:1990](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L1990-L1990),
   [2035](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L2035-L2035),
   [2079](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L2079-L2079),
   [2099](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L2099-L2099),
   [2109](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L2109-L2109).
8. **R1-F8 (low): the new host check rejected `L1_SMALL`, which `BASE` accepted.**
   **Fixed:** the check uses `Buffer::is_l1()`
   ([slice_program_factory_rm_sharded.cpp:273-279](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp#L273-L279)).
9. **R1-F9 (low): ternary style, and a duplicate helper.** The `l1_uncached()` ternary lacked parentheses, and the
   helper is the third copy in the tree.
   **Moot:** the helper was removed together with the same-core copy (R1-F1). Reviewers 2 (R2-F5) and 3 (R3-F5)
   raised the duplicate again before that.
10. **R1-F10 (low): unsigned test values never set the top bits.**
    **Fixed:** [full_range_unsigned_tensor()](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L681-L685)
    draws over the whole UINT16 or UINT32 range.

### Reviewer 2

11. **R2-F1 (low): the Quasar-only code had no recorded run after the latest edit.**
    **Fixed (verified):** the 69 Quasar cases were rerun on craq-sim after the review's last kernel edit, and the 7
    sharded cases again after the same-core copy was removed (section 4). The CI part repeats R1-F2's CI part (out of
    scope).
12. **R2-F2 (low): more compile variants.** The two tensor bindings add each tensor's accessor compile-time
    arguments, including the shard grid's core coordinates, to the reader
    ([slice_program_factory_rm_sharded.cpp:309-319](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp#L309-L319)).
    Sharded slices that shared one reader binary at `BASE` can now need separate JIT compiles, on the first run only.
    **Out of scope:**
    - `TensorBinding` has no address-only form, and `LocalTensorAccessor` needs the binding.
    - The tilize uplift (`aa97958452b`) took the same cost.
    - Device work at run time is unchanged.

    Note this next to the WH/BH parity claim in the PR. Reviewer 3 (R3-F3) raised this again.
13. **R2-F3 (low): two more test docstrings said "CB".**
    **Fixed:** [test_slice.py:1903](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L1903-L1903),
    [1950-1951](tests/ttnn/unit_tests/operations/data_movement/test_slice.py#L1950-L1951).
14. **R2-F4 (low): the host check did not print the values it found, and no test reached it.**
    **Fixed (message):** it prints both buffer types
    ([slice_program_factory_rm_sharded.cpp:273-279](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp#L273-L279)).
    The test added for this was removed again for R2-F8.
15. **R2-F5:** duplicate `l1_uncached()` helper, raised again (see R1-F9).
16. **R2-F6 (low): ambiguous "it" in the kernel comment.**
    **Fixed:** [slice_reader_unary_unpad_dims_rm_sharded.cpp:46-49](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L46-L49).
17. **R2-F7 (low): the tensor-args factory comment repeated a validation rule from distant code.**
    **Fixed:** [slice_program_factory_tile_tensor_args.cpp:61-63](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp#L61-L63).
18. **R2-F8 (low): the DRAM-rejection test checked an undocumented implementation limit.** The user's CLAUDE.md
    requires tests to check documented behavior.
    **Fixed** by deleting `test_slice_rm_height_sharded_rejects_dram`. The host check stays; `BASE` also rejected
    this case, and that rejection had no test either.

### Reviewer 3

19. **R3-F1 (low): the other slice readers have no same-core copy.** When the input is in SRAM, they can also
    issue NoC reads from their own core, for example
    [reader_unary_unpad_dims_interleaved_start_id.cpp:46-47](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp#L46-L47).
    **Out of scope:** craq-sim runs these readers correctly, and the guide applies such fixes only after the symptom
    appears. After R1-F1 was withdrawn, no slice reader has the copy, so the same reasoning covers all of them.

    **Follow-up:** run slice on the physical Quasar emulator, with an SRAM-interleaved input (for example
    `test_slice_rm`) and with a height-sharded input whose input and output shard grids match. If a slice reader
    hangs or drops data on a read from its own core, fix it then. My guess is that a fix in the NoC or TensorAccessor
    layer would serve every op better than a copy in each reader.
20. **R3-F2 (low): the lock comments described the lock's internals on each architecture.**
    **Fixed:** they now state only the requirement:
    [reader_multicore_slice_4d.cpp:161-162](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp#L161-L162),
    [reader_multicore_slice_nd.cpp:136-137](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp#L136-L137),
    [reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:50](ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp#L50-L50).
21. **R3-F3:** more compile variants, raised again (see R2-F2).
22. **R3-F4 (low): header comment was trivially true and named a hash field in another file.**
    **Fixed:** [slice_program_factory_rm_sharded.hpp:16-17](ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp#L16-L17).
23. **R3-F5:** duplicate `l1_uncached()` helper, raised again (see R1-F9).

## 3. Unresolved issues

No unresolved issues. Every reviewer ended with `NO FURTHER COMMENTS`, and each accepted the out-of-scope reasons
above.

## 4. Tests and experiments

**Wormhole (n150), full `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`.**

| Run | Tests | Passed | Skipped | Xfailed | Failed |
|---|---|---|---|---|---|
| Code at `BASE` (recipe-branch head, which differs from `BASE` only in docs) | 511 | 472 | 38 | 1 | 0 |
| After all fixes | 521 | 482 | 38 | 1 | 0 |

A per-test comparison of the JUnit results shows the 511 tests that exist at `BASE` with the same outcome. The 10
new unsigned cases pass.

**Quasar (craq-sim), after all fixes:** 69 cases, 65 passed, 4 skipped. The skips are the test's own conditions:
"Cannot untilize 5D tensor" twice, and "Cannot slice the last dimension to 1 in row major layout" twice. The cases
cover all five factories, including the 7 height-sharded cases that fit the simulator's 8x4 worker grid. An earlier
run of 59 of these cases with `TT_METAL_LLK_ASSERTS=1`, before the review fixes, gave the same result as without it.

**After removing the same-core copy (R1-F1):** the 7 Quasar height-sharded cases passed on craq-sim, and the 41
Wormhole tests matching `rm_sharded or height_sharded` passed.

**Experiments:**
- **Same-core copy reached.** While the copy existed, setting it temporarily to 0 bytes made
  `test_slice_rm_sharded_shard_grid_order[...ROW_MAJOR-origin_no_gap]` fail on Quasar with an exact-compare
  mismatch. This showed the tests reach the copy; it did not show the copy was needed. The copy was later removed
  (R1-F1).
- **DRAM input rejected on the host.** On Wormhole, a DRAM height-sharded ROW_MAJOR input sliced into an SRAM
  height-sharded output now fails on the host with the new message, which names both buffer types.
- **Unsigned values survive the round trip.** A host round trip through `ttnn.from_torch` and `ttnn.to_torch`
  keeps full-range UINT16 and UINT32 values exactly when they are given as `int64`.

**Not run:**
- Blackhole (no board on this machine).
- The physical Quasar emulator (not available).
- The 8x7-grid sharded tests on Quasar (the simulator grid is 8x4).

**Rerunning the Quasar cases** (for the PR description). From the repo root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD TT_METAL_SIMULATOR=/localdev/fplavec/temp/qsr-sim/libttsim.so
export TT_METAL_SIMULATOR_HOME=/localdev/fplavec/temp/qsr-sim ARCH_NAME=quasar CHIP_ARCH=quasar
export TT_METAL_SLOW_DISPATCH_MODE=1 LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH PYTHONPATH=$PWD/ttnn:$PWD
pytest --timeout 600 <node ids below>
```

## 5. Changed files

All fixes are uncommitted edits on top of `BASELINE` (`5f80e7410b2`). Nothing is staged or committed. No new files
were created.

```
tests/ttnn/unit_tests/operations/data_movement/test_slice.py                                          | 68 +++++++++++++++++++---
ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp    |  5 +-
ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp    |  5 +-
ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp |  4 +-
ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp |  8 +--
ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp |  3 -
ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp                        |  6 +-
ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.hpp                        |  4 +-
ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp              |  7 +++
ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp              | 10 ++--
ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp        |  3 +-
11 files changed, 85 insertions(+), 38 deletions(-)
```

## Appendix: Quasar node ids run

```
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args_device_path[input_shape=[1, 1, 128, 2880]-dim=2-start=[0, 0, 0, 0]-end=[1, 1, 32, 2880]-step=1-layout=Layout.TILE]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_shard_grid_order[w=16-h=128-c=128-n=1-orientation=ShardOrientation.ROW_MAJOR-origin_with_gap]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_shard_grid_order[w=16-h=128-c=128-n=1-orientation=ShardOrientation.ROW_MAJOR-origin_no_gap]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_shard_grid_order[w=16-h=128-c=128-n=1-orientation=ShardOrientation.COL_MAJOR-origin_with_gap]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_shard_grid_order[w=16-h=128-c=128-n=1-orientation=ShardOrientation.COL_MAJOR-origin_no_gap]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_input_and_output_grids_differ[w=16-h=128-c=128-n=1-disjoint_cores]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_input_and_output_grids_differ[w=16-h=128-c=128-n=1-fewer_output_cores]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_rm_sharded_input_and_output_grids_differ[w=16-h=128-c=128-n=1-two_input_rectangles_one_output]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.BFLOAT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.BFLOAT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.BFLOAT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.BFLOAT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.INT32-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.INT32-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.INT32-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.INT32-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT16-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT8-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT8-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=18-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT8-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=18]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_two_dim[dtype=DataType.UINT8-stride_w=2-stride_h=2-begins_w=2-begins_h=2-w=34-h=34]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_three_dim[dtype=DataType.BFLOAT16-stride_w=1-stride_h=1-stride_c=2-begins_w=0-begins_h=0-begins_c=0-w=2-h=2-c=8]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_three_dim[dtype=DataType.INT32-stride_w=1-stride_h=1-stride_c=2-begins_w=0-begins_h=0-begins_c=0-w=2-h=2-c=8]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_three_dim[dtype=DataType.UINT16-stride_w=1-stride_h=1-stride_c=2-begins_w=0-begins_h=0-begins_c=0-w=2-h=2-c=8]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_stride_slice_three_dim[dtype=DataType.UINT8-stride_w=1-stride_h=1-stride_c=2-begins_w=0-begins_h=0-begins_c=0-w=2-h=2-c=8]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.TILE-input_shape=(1, 1, 5, 1, 256)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 1, 1, 234)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.TILE-input_shape=(1, 2, 32, 32, 32)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 32, 32, 1)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.TILE-input_shape=(1, 1, 32, 32, 64)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 32, 1, 32)-input_stride=(1, 1, 2, 1, 2)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.TILE-input_shape=(2, 1, 1, 64, 64)-input_start=(1, 0, 0, 0, 0)-input_ends=(2, 1, 1, 1, 1)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.TILE-input_shape=(4, 3, 2, 1, 18)-input_start=(1, 1, 1, 0, 0)-input_ends=(1, 1, 2, 1, -2)-input_stride=(1, 1, 1, 1, 2)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.ROW_MAJOR-input_shape=(1, 1, 5, 1, 256)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 1, 1, 234)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.ROW_MAJOR-input_shape=(1, 2, 32, 32, 32)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 32, 32, 1)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.ROW_MAJOR-input_shape=(1, 1, 32, 32, 64)-input_start=(0, 0, 0, 0, 0)-input_ends=(1, 1, 32, 1, 32)-input_stride=(1, 1, 2, 1, 2)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.ROW_MAJOR-input_shape=(2, 1, 1, 64, 64)-input_start=(1, 0, 0, 0, 0)-input_ends=(2, 1, 1, 1, 1)-input_stride=(1, 1, 1, 1, 1)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_5d[layout=Layout.ROW_MAJOR-input_shape=(4, 3, 2, 1, 18)-input_start=(1, 1, 1, 0, 0)-input_ends=(1, 1, 2, 1, -2)-input_stride=(1, 1, 1, 1, 2)]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_7d_strided
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_7d
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=0-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=0-layout=Layout.TILE]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=1-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=1-layout=Layout.TILE]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=2-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=2-layout=Layout.TILE]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=3-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_negative_ends[ends=-2-dim=3-layout=Layout.TILE]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 64, 64]-begins=[0, 0, 0, 0]-ends=[1, 1, 16, 16]-layout=Layout.TILE-use_sharding=False-description=within_tile_small]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 64, 64]-begins=[0, 0, 8, 8]-ends=[1, 1, 24, 24]-layout=Layout.TILE-use_sharding=False-description=within_tile_offset]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 64, 64]-begins=[0, 0, 0, 0]-ends=[1, 1, 32, 64]-layout=Layout.TILE-use_sharding=False-description=across_tiles_aligned]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 128, 128]-begins=[0, 0, 32, 32]-ends=[1, 1, 96, 96]-layout=Layout.TILE-use_sharding=False-description=across_tiles_multiple]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 64, 64]-begins=[0, 0, 0, 0]-ends=[1, 1, 32, 32]-layout=Layout.ROW_MAJOR-use_sharding=False-description=row_major_input]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 128, 128]-begins=[0, 0, 16, 16]-ends=[1, 1, 48, 48]-layout=Layout.ROW_MAJOR-use_sharding=False-description=row_major_unaligned]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 256, 32]-begins=[0, 0, 0, 0]-ends=[1, 1, 128, 32]-layout=Layout.TILE-use_sharding=True-description=sharded_tile_boundary]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_within_tile_vs_across_tiles[input_shape=[1, 1, 128, 128]-begins=[0, 0, 32, 0]-ends=[1, 1, 96, 128]-layout=Layout.TILE-use_sharding=True-description=sharded_aligned_slice]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args[input_shape=[4, 4]-dim=1-start=[0, 1]-end=[1, 2]-step=1-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args[input_shape=[1, 28, 56, 96]-dim=2-start=[0, 0, 0, 0]-end=[1, 16, 32, 32]-step=2-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args[input_shape=[10]-dim=1-start=[2]-end=[7]-step=1-layout=Layout.ROW_MAJOR]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.TILE-shape=[1, 2, 64, 64]-begins=[0, 1, 32, 0]-ends=[1, 2, 64, 64]-steps=[1, 1, 1, 1]-dtype=DataType.UINT16]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.TILE-shape=[1, 2, 64, 64]-begins=[0, 1, 32, 0]-ends=[1, 2, 64, 64]-steps=[1, 1, 1, 1]-dtype=DataType.UINT32]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[2, 3, 40, 64]-begins=[0, 1, 4, 8]-ends=[2, 3, 20, 40]-steps=[1, 1, 1, 1]-dtype=DataType.UINT16]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[2, 3, 40, 64]-begins=[0, 1, 4, 8]-ends=[2, 3, 20, 40]-steps=[1, 1, 1, 1]-dtype=DataType.UINT32]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[1, 1, 18, 18]-begins=[0, 0, 2, 2]-ends=[1, 1, 18, 18]-steps=[1, 1, 2, 2]-dtype=DataType.UINT16]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[1, 1, 18, 18]-begins=[0, 0, 2, 2]-ends=[1, 1, 18, 18]-steps=[1, 1, 2, 2]-dtype=DataType.UINT32]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[1, 1, 3, 4, 32]-begins=[0, 0, 1, 0, 0]-ends=[1, 1, 3, 4, 32]-steps=[1, 1, 1, 1, 2]-dtype=DataType.UINT16]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_unsigned_int[layout=Layout.ROW_MAJOR-shape=[1, 1, 3, 4, 32]-begins=[0, 0, 1, 0, 0]-ends=[1, 1, 3, 4, 32]-steps=[1, 1, 1, 1, 2]-dtype=DataType.UINT32]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args_unsigned_int[dtype=DataType.UINT16]
tests/ttnn/unit_tests/operations/data_movement/test_slice.py::test_slice_tensor_args_unsigned_int[dtype=DataType.UINT32]
```
