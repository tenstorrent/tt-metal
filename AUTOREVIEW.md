# Auto review: Quasar top-k uplift

- **Target:** all commits on `fplavec/quasar_topk_2` relative to `origin/main`
- **BASE:** `e63758cb687081e2ec941e6473c6a9956a00a349` (merge-base with `origin/main`)
- **BASELINE:** `aacebc12e573ecc5cfa6cf0cab50aacf5184efe4` (HEAD; all fixes are uncommitted edits on top of it)
- **Date:** 2026-10-28

## 1. Summary per reviewer

| Reviewer | Reviews | Ended with |
|---|---|---|
| 1 | 3 of 5 | `NO FURTHER COMMENTS` |
| 2 | 2 of 4 | `NO FURTHER COMMENTS` |
| 3 | 2 of 3 | `NO FURTHER COMMENTS` |

Totals: 12 fixed (one of them partly, see R1-F1), 2 rejected, 2 out of scope, 0 unresolved.

## 2. All issues raised

### Reviewer 1

- **R1-F1 (test coverage, medium).** No test covered the Quasar behavior: rejections, the Int16 remap, indices with bit 15 set, the factory gate.
  **Fixed (partly).** New file [test_topk_quasar.py:1-121](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L1-L121): BF16 top-k with signed inputs, indices 32768 to 40704 checked exactly, and the Quasar-only rejections of FLOAT32, BFLOAT8_B and 32-bit indices. No Quasar skips were added to `test_topk.py`, because no CI runs ttnn tests on Quasar. The reviewer accepted this. My first claim was that the W=40960 case covers the factory gate. That claim was wrong, because the width is not a power of two. The `(1, 2, 32, 1024)` case covers the gate (see R1-F9).
- **R1-F2 (stale comment, low).** After the `Int16` exemption in [data_format.cpp:72-74](tt_metal/jit_build/data_format.cpp#L72-L74), the typecast test comment said that Int16 trips the format check, which is no longer true.
  **Fixed.** [test_sfpu_compute.cpp:4150](tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp#L4150-L4150) now says only that the narrow-int datapath has not landed. The change to `get_data_exp_precision` (Int16-only kernels now get B) matches the documented default at [data_format.cpp:115-120](tt_metal/jit_build/data_format.cpp#L115-L120), and the reviewer accepted this.
- **R1-F3 (stale comment, low).** The comment "Dimension too small for multi-core" in validation no longer matched once the architecture gate was added.
  **Fixed.** [topk_device_operation.cpp:398](ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp#L398-L398).
- **R1-F4 (duplicate code, low).** The Quasar branch of `generate_index_tile` copied the 32-bit store loop.
  **Fixed.** A new `detail::store_index_tile` at [index_tile_dataflow.hpp:22-42](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L22-L42) is called from both paths ([line 85](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L85-L85), [line 163](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L163-L163)). Its pointer parameter is a template type, because on Quasar `get_ptr` returns a `CoreLocalMem`. The first version did not compile on Quasar, and the craq-sim run caught it.
- **R1-F5 (comment will go stale, low).** The comment claimed, with no issue number, that the Quasar emulator does not handle a NoC self-read reliably.
  **Fixed.** That clause is removed. The comment keeps only the `load_blocking` reason ([index_tile_dataflow.hpp:80-82](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L80-L82)).
- **R1-F6 (comment will go stale, low).** The `pack_init` comment described LLK register details ("format gasket", "BFD").
  **Fixed.** [topk.cpp:35-36](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L35-L36) now states only the rule and cites #58488.
- **R1-F7 (convention, low).** The `#ifdef ARCH_QUASAR` around `dummy_pack` is not needed, because it is a no-op on WH/BH.
  **Rejected.** The guard keeps the `dummy_pack` sites in the same form as the `dummy_unpack` site ([topk.cpp:123](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L123-L123), [topk.cpp:141](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L141-L141)), so a reader sees that both workarounds apply only on Quasar. The reviewer accepted this.
- **R1-F8 (bad practice, low).** After the R1-F4 fix, `tile_faces` and `face_size` were unused in the Quasar build.
  **Fixed.** They moved under `#else` ([index_tile_dataflow.hpp:89-90](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L89-L90)).
- **R1-F9 (unexplained code, low).** Nothing marked the test case that covers the Quasar factory gate.
  **Fixed.** A comment on that case, later shortened (see R3-F5): [test_topk_quasar.py:29-31](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L29-L31).

### Reviewer 2

- **R2-F1 (test coverage, low).** No test covered a caller-supplied UINT16 `indices_tensor` going through the Int16 buffer, or several tile rows on one core.
  **Fixed.** Two tests: [test_topk_several_rows_per_core](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L66-L76) runs 4 tile rows on core (0,0), and [test_topk_uint16_indices_tensor](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L79-L99) uses labels from 65472 to 65535.
- **R2-F2 (comment will go stale, low).** The comment on the factory-gate case depended on the constants in `topk_constants.hpp`.
  **Fixed.** The comment points to `topk_multicore_structurally_eligible`. The shape was kept, because a wider shape would slow every craq-sim run without adding coverage. Reviewer 3 raised the same issue again as R3-F5.

### Reviewer 3

- **R3-F1 (test coverage, medium).** No test used k > 64, so the growing phase of the sort ([topk.cpp:347-354](ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp#L347-L354)) did not run.
  **Fixed.** Added the case `((1, 1, 32, 256), 96)` at [test_topk_quasar.py:32-33](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L32-L33).
- **R3-F2 (stale docs, low).** `TopK.md` and the nanobind docstring did not mention the Quasar limits.
  **Out of scope.** The docs were first updated, then reverted at the user's request: the Quasar limits are temporary, and Quasar is not officially supported yet.
- **R3-F3 (convention, low).** The `Int16` exemption in [data_format.cpp:72-74](tt_metal/jit_build/data_format.cpp#L72-L74) applies only part of what PR #47852 had deferred: the pack-src part is not included.
  **Out of scope.** The suggested fix is a note in the PR description, which the branch owner writes. The reviewer accepted this. Suggested text for the PR description: "Only the `check_consistent_format_across_buffers` exemption for Int16 is applied; the `fp32 -> Int16` pack-src handling deferred in #47852 is not."
- **R3-F4 (convention, low).** The test file was in `experimental/quasar/`, but every other file there tests a `ttnn.experimental.quasar.*` op.
  **Fixed.** Moved to [tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L1-L121).
- **R3-F5 (comment will go stale, low).** The test comment still described the threshold values. This is the same issue as R2-F2.
  **Fixed.** The threshold values are removed ([test_topk_quasar.py:29-31](tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py#L29-L31)).
- **R3-F6 (comment will go stale, low).** The `ckernel.h` guard comment names `ckernel_trisc_id.h`.
  **Rejected.** Naming the `#error` tells the reader what fails and how to check whether the guard is still needed ([index_tile_dataflow.hpp:12-16](ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp#L12-L16)). The reviewer withdrew the finding.

## 3. Unresolved issues

No unresolved issues.

## 4. Tests and experiments

The runs below were made with the docstring change in place (host rebuilt, `build_metal.sh` exit 0). The docstring change was reverted afterwards; it affects only Python help text, so the results still apply. The host build in `build/` still contains the docstring change until the next rebuild.

| Run | Result |
|---|---|
| WH n150: `test_topk_quasar.py` + `tests/ttnn/unit_tests/operations/reduce/test_topk.py` | 321 passed, 46 skipped, 76 xfailed |
| craq-sim, `TT_METAL_LLK_ASSERTS=1`: `test_topk_quasar.py` | 16 passed |
| CI simulator (public `tenstorrent/ttsim` v1.15.0 `libttsim_qsr.so`, CI setup with a 3x2 core grid, LLK asserts off): `test_topk_quasar.py` | 16 passed in 78 s |

- On WH the 3 Quasar-only rejection tests are skipped, as intended.
- One intermediate craq-sim run failed with a Quasar compile error in the first version of `detail::store_index_tile` (R1-F4), which was then fixed.
- A craq-sim run with LLK asserts off was not repeated after these fixes.

## 5. Changed files

All fixes are uncommitted. Nothing was staged or committed.

```
 tests/tt_metal/tt_metal/llk/test_sfpu_compute.cpp  |  3 +-
 ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp   | 58 +++++++++++-----------
 .../reduction/topk/device/kernels/compute/topk.cpp |  5 +-
 .../topk/device/topk_device_operation.cpp          |  2 +-
```

New file: `tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_quasar.py`.
