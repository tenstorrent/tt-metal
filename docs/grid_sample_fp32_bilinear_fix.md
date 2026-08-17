# grid_sample fp32 bilinear fix

**Op:** `ttnn.grid_sample` — bilinear mode
**Class of bug:** wrong numerical result (silent), input-dtype dependent
**Status:** fixed and verified — all grid_sample suites pass (681 passed / 163 skipped / 0 failed)

---

## Summary

`ttnn.grid_sample` in **bilinear** mode returned badly wrong results when the **input tensor was
`float32`**. The output was numerically uncorrelated with the reference (PCC ≈ 0.10–0.45) and its
magnitude was inflated 2–5×. The exact same op with a `bfloat16` input was correct (PCC ≈ 1.0).

The defect was a single mis-declared circular buffer: the bilinear **weights** are always produced by
the reader as `bfloat16`, but the weights circular buffer was declared using the *input* tensor's data
format. With an fp32 input, the compute engine then read the packed bf16 weight bytes as if they were
fp32, corrupting every interpolation weight.

The fix declares the weights circular buffer as `bfloat16` unconditionally. It is a one-line, contained
change with no effect on the (already correct) bf16 path.

---

## The issue

### Symptoms

| Case (input NCHW / grid / align_corners) | PCC vs reference | Output abs-max vs reference |
|---|---|---|
| (1, 32, 8, 8) / (1, 4, 4, 2) / True  | 0.099 | 5.9 vs 3.5 |
| (1, 32, 8, 8) / (1, 4, 4, 2) / False | 0.451 | 2.4 vs 2.3 |
| (1, 64, 96, 96) / (1, 128, 64, 2) / True | 0.137 | 13.4 vs 3.6 |

The result was not merely noisy — the magnitude was systematically too large, and zeros from
`padding_mode="zeros"` landed in the wrong positions. That signature (scaled/garbled rather than
approximately-right) points at corrupted **weights**, not at accumulation noise.

### How it surfaced

The op is fed from a model-compilation path (ONNX → compiler → TTNN) where the data tensor arrives as
`float32`, is retiled and permuted to the NHWC row-major layout the kernel expects, and then passed to
`grid_sample` with a raw (non-precomputed) grid. Every tensor in that flow is `float32`. Earlier
TTNN-level testing had only exercised `bfloat16` inputs, so the fp32 path was never covered and the bug
went unnoticed.

---

## Root cause

Bilinear `grid_sample` is a gather-plus-weighted-sum. For each output point the reader:

1. computes the 4 corner pixel positions and their 4 bilinear weights from the grid coordinate, and
2. reads the 4 corner input "sticks" into an **input** circular buffer, and writes the 4 weights into a
   separate **scalar (weights)** circular buffer.

The compute kernel then reduces the 4 corners against the 4 weights to produce the output — a weighted
average, so with correct weights the output magnitude can never exceed the input's.

Two independent facts collided:

- **The weights are always `bfloat16`.** The reader truncates each weight to bf16 and packs the four of
  them as 16-bit values into the weights buffer. This is unconditional — it does not depend on the input
  dtype.
- **The weights buffer inherited the input dtype.** The program factory declared the weights circular
  buffer using the *input* tensor's data format.

For a `bfloat16` input these agreed, so the buffer was (accidentally) correct. For a `float32` input the
buffer was declared 4-byte fp32 while its contents were 2-byte bf16. The compute engine unpacked the
packed bf16 weight bytes as fp32 values — reading two bf16 weights as one fp32 number, at the wrong
element stride and face layout. Every weight became garbage, so the weighted sum was garbage and its
magnitude was unbounded relative to the input.

Crucially, the **input** data path was never at fault: the corner sticks are copied byte-for-byte and the
reduce handles fp32 input tiles correctly. Only the weights buffer's declared format was wrong. This is
why the fix is limited to that one buffer.

---

## The fix

**File changed:** `ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp`

Declare the bilinear weights circular buffer as `bfloat16` unconditionally, independent of the input
tensor's dtype, so the buffer's format matches the bf16 weights the reader actually writes. The same
correction is applied to the second weights buffer used on the split-reader (sharded) path.

- **fp32 input:** weights are now unpacked correctly → correct weighted sum.
- **bf16 input:** the weights buffer was already bf16, so behavior is unchanged.

No change was needed in the reader, the compute kernel, or the input/output buffers.

### Diff

```diff
diff --git a/ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp b/ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_bilinear_program_factory.cpp
@@ ProgramDescriptor GridSampleBilinearProgramFactory::create_descriptor(
         });
     }

-    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
+    // The bilinear weights are always produced as bf16 by the reader (fp32_to_bf16_truncate +
+    // fill_four_val, which packs uint16 bf16 values). The scalar CB must therefore be bf16 regardless
+    // of the input dtype — declaring it with the (possibly fp32) input format makes the compute unpack
+    // the bf16 weight bytes as a wider dtype, corrupting every weight and inflating the output.
+    const tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
+    const uint32_t scalar_cb_page_size = tt::tile_size(scalar_cb_data_format);
     const uint32_t scalar_cb_index_0 = cb_idx++;
     desc.cbs.push_back(CBDescriptor{
         .total_size = BUFFERING_FACTOR * scalar_cb_page_size,
         .core_ranges = all_cores,
         .format_descriptors = {{CBFormatDescriptor{
             .buffer_index = static_cast<uint8_t>(scalar_cb_index_0),
-            .data_format = input_cb_data_format,
+            .data_format = scalar_cb_data_format,
             .page_size = scalar_cb_page_size,
             .face_geometry = scalar_face_geometry,
         }}},
@@ (split-reader path)
             .core_ranges = all_cores,
             .format_descriptors = {{CBFormatDescriptor{
                 .buffer_index = static_cast<uint8_t>(scalar_cb_index_1),
-                .data_format = input_cb_data_format,
+                .data_format = scalar_cb_data_format,
                 .page_size = scalar_cb_page_size,
                 .face_geometry = scalar_face_geometry,
             }}},
```

### Why it is safe

- The change only alters the declared format of the weights buffer; it does not touch the input tiles,
  the reduction, or the output.
- For bf16 inputs the declared format is identical to before, so the well-tested bf16 path is untouched.
- The fp32 input tiles were already being read and reduced correctly, confirmed by the fact that fixing
  only the weights buffer restored full accuracy.

---

## Verification

After the fix, the three failing cases match the reference to PCC ≈ 1.0, with output magnitude back in
line with the reference:

| Case (input NCHW / grid / align_corners) | PCC before | PCC after | Output abs-max after vs ref |
|---|---|---|---|
| (1, 32, 8, 8) / (1, 4, 4, 2) / True  | 0.099 | 0.99999 | 3.5 vs 3.5 |
| (1, 32, 8, 8) / (1, 4, 4, 2) / False | 0.451 | 0.99999 | 2.3 vs 2.3 |
| (1, 64, 96, 96) / (1, 128, 64, 2) / True | 0.137 | 0.99999 | 3.6 vs 3.6 |

---

## Regression coverage

**Test file:** `tests/ttnn/unit_tests/operations/pool/test_grid_sample_fp32_repro.py`
(test: `test_grid_sample_fp32_bilinear_repro`)

A dedicated regression test was added that reproduces the compilation flow faithfully — the full op chain
(retile → permute to NHWC → row-major → grid_sample → retile → permute back to NCHW), all tensors
`float32`, DRAM-interleaved memory, and a raw (non-precomputed) grid whose coordinates span slightly
beyond `[-1, 1]` so the zeros-padding path is exercised. Three cases are covered:

- `(1, 32, 8, 8)` data, `(1, 4, 4, 2)` grid, bilinear, `align_corners = True`
- `(1, 32, 8, 8)` data, `(1, 4, 4, 2)` grid, bilinear, `align_corners = False`
- `(1, 64, 96, 96)` data, `(1, 128, 64, 2)` grid, bilinear, `align_corners = True`

The reference is PyTorch's `grid_sample` (identical semantics to the ONNX operator), compared in NCHW at a
PCC threshold of 0.99. These cases failed before the fix and pass after it.

### Full suite results

| Suite (test file) | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/pool/test_grid_sample_fp32_repro.py` (new) | 3 passed |
| `tests/ttnn/unit_tests/operations/pool/test_grid_sample_comprehensive.py` | 329 passed, 44 skipped |
| `tests/ttnn/unit_tests/operations/pool/test_grid_sample.py` | 108 passed |
| `tests/ttnn/nightly/unit_tests/operations/pool/test_grid_sample.py` | 241 passed, 119 skipped |
| **Total** | **681 passed, 163 skipped, 0 failed** |

The comprehensive, unit, and nightly counts are unchanged from before the fix, confirming the bf16 path
was not disturbed.

---

## Takeaways

- **A circular buffer's declared data format must match what is written into it, not the tensor it is
  associated with.** The weights are a derived, always-bf16 quantity; tying their buffer format to the
  input dtype was the latent bug.
- **Cover input dtypes explicitly.** The op silently accepted fp32 input while the shared compute path was
  only validated for bf16; the gap hid a wrong-result bug rather than raising an error. The new fp32
  regression cases close that gap, and adding an fp32 axis to the broader matrix would keep it closed.
- **Magnitude is a diagnostic.** A weighted-average op whose output exceeds its input by 2–5× is almost
  always a weights problem, which is what localized the root cause quickly.
