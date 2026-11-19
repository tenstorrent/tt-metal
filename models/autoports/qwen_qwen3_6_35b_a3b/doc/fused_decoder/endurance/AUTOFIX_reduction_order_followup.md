# CPU recurrence reduction order

**The exact CPU order is reproducible:** sequentially sum each contiguous block of 16 key rows, then sequentially sum the eight block results. This matches PyTorch FP32 codes exactly for all tested HF and TT recurrence products. Four interleaved partial accumulators, one 128-term running sum, and a balanced tree do not.

No TTNN import, hardware run, decoder edit, or test edit was performed. Added a focused CPU evidence script, `reduction_order_cpu_microprobe.py`, and its results, `reduction_order_cpu_microprobe.json`. Run from repository root:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/reduction_order_cpu_microprobe.py
```

## Source and validation

Installed PyTorch reports `2.11.0+cpu`, git `70d99e998b4955e0049d13a98d77ae1b14db1f45`, CPU capability AVX512. Its local `ATen/native/cpu/Reduce.h` is present, but the floating-point sum implementation is not installed. Inspected the [exact-commit upstream SumKernel.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/SumKernel.cpp#L318).

For the `[32,128,128]` product reduced on axis 1, `vectorized_outer_sum` calls `multi_row_sum`: its 16-term cascade gives eight contiguous partial sums. The four vector accumulators cover different **output columns**, not interleaved key rows. Relevant source: `multi_row_sum` lines 318–378, `vectorized_outer_sum` lines 437–470, outer-path selection lines 548–555. This is shape/layout/version specific; transposing to a contiguous last-axis reduction changes the algorithm.

The CPU script verifies exact integer FP32 codes, including signed zero, against `torch.sum`, using:

- Two deterministic random products, including a broad exponent distribution.
- HF and TT canonical memory products at all nine precise-probe positions.
- Products from the nine captured device `decayed_state` tensors.
- HF and TT canonical output contractions at those same nine positions.

All **29 memory/random cases and 18 output cases** match exactly with the 16-row cascade. Each sum yields 4096 values. Alternative orders differ in thousands of FP32 values and sometimes change the resulting next BF16 cache even with all other arithmetic held fixed.

| Position | Next-cache BF16 differences: sequential128 | Four interleaved partials | Balanced tree | Cascade16 |
|---|---:|---:|---:|---:|
|128|6|0|1|0|
|129|1|2|2|0|
|130|6|1|0|0|
|1024|4|0|0|0|
|3979|6|3|2|0|
|5011|6|3|3|0|
|7551|4|4|2|0|
|7703|3|2|2|0|
|8157|5|3|2|0|

These are CPU counterfactuals using captured TT inputs and canonical separate update. They do not establish that matching the reduction order cures the endurance failure. The saved device memory itself differs from the canonical reduction in 2821–3017 FP32 values and 0–2 BF16 values; the earlier intermediate report quantifies its effect on the actual update.

## Feasible model-only TTNN diagnostic

Use separate SFPU FP32 multiply first, then explicit additions in the established order. Process all eight 16-row chunks in parallel: row-strided slices gather one row from each chunk, and 16 sequential additions produce the eight partials. Then perform eight sequential additions of those partials. Do **not** replace either stage with `ttnn.sum`, whose order is the variable being controlled.

For an FP32 `product` of shape `[1,32,128,128]`, a source-supported adapter is:

```python
rm = ttnn.to_layout(product, ttnn.ROW_MAJOR_LAYOUT)
def chunk_rows(i):
    x = ttnn.slice(rm, [0,0,i,0], [1,32,128,128], [1,1,16,1])
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # [1,32,8,128]
partials = ttnn.add(chunk_rows(0), 0.0)
for i in range(1, 16):
    partials = ttnn.add(partials, chunk_rows(i))
prm = ttnn.to_layout(partials, ttnn.ROW_MAJOR_LAYOUT)
def partial_row(i):
    x = ttnn.slice(prm, [0,0,i,0], [1,32,i+1,128])
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # [1,32,1,128]
result = ttnn.add(partial_row(0), 0.0)
for i in range(1, 8):
    result = ttnn.add(result, partial_row(i))
```

The initial scalar-zero additions preserve the CPU zero-initialized accumulation convention; omitting them would require accepting possible signed-zero differences. Every intermediate remains FP32. The adapter needs no runtime torch work or uploads, but has substantial slicing/layout/add dispatch overhead; it is a diagnostic, not a performance recommendation. No hardware validation is claimed.

Current source explicitly supports positive row-major strided slices, including FP32: `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:121,235,377–383`; `slice/device/slice_device_operation.cpp:266–276`; Python `slice/slice_nanobind.cpp:55–66,90–97`. FP32 add selects SFPU at `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–66`.

Apply this first to `memory = reduce(decayed*k)`, keeping the separately rounded state update control. Compare captured memory against the CPU cascade on identical decayed/K inputs, then measure next-cache codes. Matching only memory cannot remove EXP/softplus/frontend differences. The output contraction can use the same helper independently if exact core arithmetic is needed; it does not feed the recurrent cache, so memory is the priority for accumulated state drift.
