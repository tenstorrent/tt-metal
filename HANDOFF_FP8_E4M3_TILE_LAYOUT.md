# Handoff: FP8_E4M3 in TILE layout

Branch `feat/fp8-e4m3-tile-layout`, one functional commit (`9300c6e442b`
"Allow FP8_E4M3 tensors in TILE layout") on top of `ba6a65ce40c`. This file
is a separate commit; drop it before opening a PR. No PR has been opened.

## Goal

`DataType::FP8_E4M3` was accepted only in ROW_MAJOR. It is a plain
1-byte-per-datum format (1024 B per 32x32 tile), so nothing in the format
prevents TILE layout. The goal is to allow TILE and make the host and device
paths that assumed ROW_MAJOR work.

## Why it was ROW_MAJOR-only

A scope cut, not a technical limitation.

- PR #43775 (`pmilojevicTT`, May 2026) introduced FP8_E4M3 for the DeepSeek V3
  prefill combine op, which produces row-major data. The PR states that only
  row-major is supported and that full support is tracked in issue #45040.
- In review, the answer to "what about other layouts" was that only the minimal
  support needed for DeepSeek prefill was being implemented. The lock-down tests
  (pad/unpad/to_layout throw) were added at a reviewer's request.
- Issue #45040 (`fplavecTT`, open) lists row-major-only as an open item. This
  branch addresses that item.
- PR #48046 treated the restriction as given. Slack has no rationale either,
  only workarounds.

Related 8-bit tilize bugs on Blackhole that this change can expose:
#50707 (open, narrow tiles), #48315, and #49473 (reverted by #50989).

## What changed (non-test: 20 files, +107 / −124)

Host:
- `tensor_spec.cpp`: accept TILE for FP8_E4M3.
- `tensor_types.hpp` (public header, +3 lines, no struct layout change):
  `convert_to_data_type<float8_e4m3>()` returns `FP8_E4M3`.
- `to_layout`: tilize/untilize `float8_e4m3` like any element-encoded dtype.
  Host pad/unpad are enabled for fp8 (needed for non-tile-aligned shapes).
- Template instantiations for `float8_e4m3`: encode/decode,
  `to_tile_major_layout`, `convert_layout`, `host_buffer::get_as`,
  `HostTensor`/`Tensor::to_vector`. `convert_to_data_type` maps it, and
  `to_vector<float>` handles fp8.
- `to_torch`: single-device FP8 (ROW_MAJOR and TILE) returns
  `torch.float8_e4m3fn` instead of throwing. Mesh compose handles TILE fp8.
- `print`: pivots fp8 through FLOAT32 in either layout.

Device:
- `tilize` / `tilize_with_val_padding` accept an FP8_E4M3 output dtype. With
  no `dtype`, an fp8 input still tilizes to FLOAT32 (backward compatible).
- `untilize` / `untilize_with_unpadding` enable 32-bit DEST for fp8 inputs.
- `copy` with dtype conversion enables 32-bit DEST when fp8 is involved.
- `typecast` needed no change.

The 32-bit DEST enables are required by the Blackhole/Quasar check in
`tt_metal/impl/program/program.cpp` (around line 3171), which rejects any
compute kernel on a core with an FP8 CB unless `fp32_dest_acc_en` is set.

## Test status (Blackhole p150b, all passing)

- Full `tests/ttnn/unit_tests/base_functionality/test_to_layout.py`,
  `test_copy.py` and `tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py`.
- The fp8 subsets of `test_tilize.py`, `test_tilize_with_val_padding.py`,
  `test_print_tensor.py`, to_dtype and to_list.
- 78 C++ host-tensor tests, including the `test_host_tensor_to_layout.cpp`
  matrix, which now covers FP8_E4M3.

Measured device semantics, identical for TILE and ROW_MAJOR, pinned by tests:
- fp8 → bf16/fp32 is exact for every normal value and +0.
- bf16 → fp8 rounds toward zero; torch rounds to nearest even
  (for example 1.1875 → 1.125 on device, 1.25 in torch).

Five strict xfails in `test_eltwise_typecast.py` record the remaining
deviations from torch. They are pre-existing hardware/LLK behaviour, not
introduced here:
- `test_typecast_bf16_to_fp8_e4m3_tile_matches_torch_rne`: round toward zero instead of RNE.
- `test_typecast_bf16_to_fp8_e4m3_tile_negative_zero`: −0 becomes +0.
- `test_typecast_fp8_e4m3_to_bf16_tile_negative_zero`: −0 becomes +0.
- `test_typecast_fp8_e4m3_to_bf16_tile_nan`: fp8 NaN (0x7F) widens to 512.
- `test_typecast_bf16_to_fp8_e4m3_tile_subnormals`: no fp8 subnormals.

Reproduce:

```bash
source source-env.sh
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/base_functionality/test_to_layout.py \
    tests/ttnn/unit_tests/base_functionality/test_copy.py \
    tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/data_movement/test_tilize.py -k fp8
./build/test/tt_metal/unit_tests_api --gtest_filter='*HostTensorToLayout*:*HostTensorToTileLayout*'
```

## Is the 32-bit DEST requirement imposed by the ISA?

No. It is a tt-metal runtime choice; FP8 with 16-bit DEST is not undefined
behaviour according to `tt-isa-documentation`.

- The ISA docs define unpacking FP8 (including Blackhole e4m3) to FP16 in
  SrcA/SrcB and in 16-bit DEST (`BlackholeA0/.../Dst.md`), and packing FP16 to
  e4m3. No UndefinedBehavior entry covers this case.
- The real hardware constraint: one FPU operation cannot mix A-exponent (FP16)
  and B-exponent (BF16) formats in SrcA/SrcB.
- The check comes from PR #43481 (issue #43418). Of four options, the author
  picked "(Laziest way) Have Fp8 ONLY work with Fp32 dest" to avoid reprogramming
  the DEST A/B format across threads.
- LLK `cpack_common.h` handles 16-bit DEST feeding FP8 explicitly. tt-llk's own
  tests (datacopy, pack_untilize, unpack_tilize) run FP8 with 32-bit DEST off at
  atol/rtol 0.2.
- A one-off experiment with the check bypassed (not in this branch): typecast
  gave bit-identical normal values; only −0 was kept and NaN became 510.
- Issue #43418 reports software bugs on the 16-bit path (DEST format inference,
  a race on the e4m3 flag), so relaxing the check needs per-op testing.

## Open decisions / next steps

1. **Default tilize output dtype for fp8 input.** Currently FLOAT32, to keep
   existing callers unchanged. Making it stay FP8_E4M3 is a one-line change in
   each of the two tilize device operations.
2. **Relax the `program.cpp` 32-bit DEST check** (separate PR). This would let
   fp8 kernels use BF16/FP16 DEST and halve DEST usage.
3. **Rounding.** bf16 → fp8 rounding toward zero is a deviation from torch;
   worth an issue if consumers need RNE.
4. Watch the 8-bit tilize bugs above (#50707 especially) when fp8 tensors
   with narrow widths are tilized on device.
5. Before a PR: update issue #45040, drop this file, and run the full
   post-commit tensor and data-movement suites. Only Blackhole p150b was tested;
   Wormhole has no FP8_E4M3 path and was not run.
