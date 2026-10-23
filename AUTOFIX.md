# AutoFix: `ttnn.gather` on ROW_MAJOR BLOCK/WIDTH-sharded tensors with an unaligned shard row (issue #62153)

Branch `fplavec/gather-rm-subaligned-shard-width`, based on `main` at `8ac40e3f904`. Diagnosis in [AUTODEBUG.md](AUTODEBUG.md). All results below are from a Wormhole n150 in this machine.

## Diagnosis (confirmed on hardware)

1. **Unaligned shard row (issue cases 1 to 5).** For a ROW_MAJOR BLOCK/WIDTH-sharded buffer, one page is one shard row of `shard_W * element_size` bytes, and the device stores pages at a pitch of that size rounded up to 16 B. The RM gather kernels use the unrounded size as both the page pitch and the NOC chunk size. When the shard row is not a multiple of 16 B, the kernels read and write the wrong SRAM bytes. A sharded index then gives garbage index values that are used as load addresses with no bounds check, and the device hangs.
   - Hardware check: bf16 cases with 24 B shard rows failed; the f32 versions of the same cases (48 B shard rows) passed; the sharded-index case hung until killed after 120 s.
2. **Unaligned per-core offset in `RmSingleRowMultiCore` (also hit by case 5).** The factory gives core `c` the columns starting at `w_start = c * W_index / ncores`. The byte offset `w_start * element_size` is often not 16 B aligned, while the local CB address is. The Wormhole NOC then moves the wrong bytes. This happens with interleaved tensors too.
   - Hardware check, all tensors interleaved, shape (1, 1, 4, W): W = 1984 gave 4156 of 7936 wrong values in bf16 and 384 of 7936 in f32. W = 2048 (32 columns per core, so every offset is aligned) gave 0 wrong values.

## Changes

- [gather_device_operation.hpp](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.hpp), [gather_device_operation.cpp](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp)
  - New `ttnn::prim::has_subaligned_shard_row(memory_config, element_size)`. It has the same name and test as slice's guard, restricted to BLOCK/WIDTH sharding.
  - `validate_on_program_cache_miss` raises a hard error for a ROW_MAJOR input, index, or computed output that fails this test. After this change, `per_shard_page_size_bytes()` equals `buffer->aligned_page_size()` for every tensor the RM factories see, which is what the Metal 2.0 port needs.
- [gather.cpp](ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp), in `gather_dispatch`, for the native ROW_MAJOR path:
  - An input or index tensor that fails the test is moved to interleaved SRAM.
  - If the output spec fails the test, the device op writes to interleaved SRAM, and the result is moved to that spec with `to_memory_config`, into the preallocated `out` tensor when one is given. The output spec is computed before the operands move, because a sharded output with no shard spec derives its spec from the index tensor.
- [gather_program_factory.cpp](ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp), `RmSingleRowMultiCore`: W_index is split in units of `w_unit` columns, chosen so that `w_unit * element_size` is a multiple of the buffer alignment for both the index and the output tensor. The last core takes fewer columns when W_index is not a multiple of `w_unit`. The kernels are unchanged.
- Tests in [test_universal_input_tm_gather.py](tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py):
  - the 5 repro cases from the issue;
  - 2 interleaved multi-core cases (SRAM and DRAM) with W = 1984;
  - a WIDTH-sharded output with no shard spec, where the spec derived from a (32, 4) uint32 index gives an 8 B bf16 shard row;
  - a preallocated WIDTH-sharded and a preallocated BLOCK-sharded `out` tensor, each with a 24 B shard row.
- [test_gather_codegen_routing.py](tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather_codegen_routing.py): removed the skip of the value comparison for index rows wider than 1920. The comment that justified the skip said `RmSingleRowMultiCore` "returns wrong elements", which is no longer true.

## Validation

| Command | Result |
|---|---|
| `./build_metal.sh -e --enable-fake-kernels-target` | pass |
| `pytest test_universal_input_tm_gather.py::test_gather_rm` and the 2 new tests, excluding the index case | 29 passed (before the fix, 5 of the 8 issue cases failed) |
| `pytest ...::test_gather_rm -k "subaligned and index"` | 2 passed, no hang (before the fix: hang) |
| Script: interleaved DRAM RM gather, W = 1984, 2048, 151936, bf16 and f32, exact compare with torch | 0 wrong values for every case |
| `pytest test_universal_input_tm_gather.py test_gather_codegen_routing.py tests/ttnn/unit_tests/operations/data_movement/test_gather.py` | 234 passed, 1 xfailed (the xfail is the existing codegen trace test for issue #46533) |
| `pre-commit run` on changed files | pass |

The backstop in `validate_on_program_cache_miss` raises a hard error for any ROW_MAJOR operand with an unaligned BLOCK/WIDTH shard row. The new sharded tests passed, so the wrapper reroute ran for each of them.

## Remaining risk

- The device-op backstop has no direct test. `ttnn::prim::gather` has no Python binding, and `gather_force_native` goes through the same wrapper.
- The reroute adds one or two `to_memory_config` conversions for affected tensors. This is the same cost that slice and fold already have for such tensors. Performance was not measured.
- The fix was tested on Wormhole only. On Blackhole the DRAM alignment is larger, so `w_unit` can be larger. This is handled by the same formula, but it was not tested on Blackhole.
- Not addressed: the RM gather loops still have no bounds check on index values. An out-of-range index can still read outside the input CB. The documented contract says the result is unspecified, not that the device may hang.
