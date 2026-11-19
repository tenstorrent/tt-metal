# Existing-op optimizations for exact ordered reductions

Source/CPU only. No device run, decoder edit, shared-native edit, or measured speed claim. The parent owns validation of the sequential prefill-state repair and performance baseline. CPU layout-check results are saved in `ordered_reduction_layout_cpu.json`.

## Main finding: a final-result-only sequential reduction exists

`ttnn.cumsum` has no final-only option: its output spec preserves the input shape, and a supplied output must have that same shape. It packs every prefix tile. Neither `optional_out` nor the compensated-sum flag changes that contract.

However, **`ttnn.experimental.fast_reduce_nc` directly supports FP32 sequential SFPU addition with only the final output stored**, when FP32 input and `fp32_dest_acc_en=True` select `REDUCE_FP32_IN_DEST`. This is a concrete model-only projection candidate after moving the scan axis into an outer dimension and packing independent outputs into tile lanes.

Source proof:

- `ttnn/cpp/ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp:15–22`: all public options; no terminal-only option.
- `.../reduction/accumulation/device/accumulation_device_operation.cpp:22–30,111–128`: preallocated output must match input shape; generated output has input logical shape.
- `.../reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp:40–83`: every scanned tile is packed/pushed.
- `.../experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp:53–82`: FLOAT32 is accepted, rank>2, only outer dimensions (axis <= rank-3).
- `.../fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:286–307`: FP32 input plus FP32 destination sets `REDUCE_FP32_IN_DEST` and explicit `UnpackToDest`.
- `.../fast_reduce_nc/device/kernels/reduce_nc.cpp:42–76`: copy first tile; then sequential `add_binary_tile` for every later tile; pack only once at the end.
- `.../fast_reduce_nc/device/kernels/reader_reduce_nc.cpp:57–87`: increasing reduction index; each successive input uses the next outer plane.
- `.../fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:169–184,345–370`: split **output tiles** across cores. Each owning core processes all reduction inputs. There is no cross-core partial sum or changed association. Granularity groups reads in eights but does not regroup additions.

Call the experimental op directly. Generic `ttnn.sum`'s FLOAT32 route does not automatically take this NC implementation: `generic_reductions.cpp:500–505` limits its fast-NC shortcut to BF16/BFP8, and its other-axis path transposes the reduced axis to H at lines 212–219.

## Decode projection layout proposed by parent

Let `Wordered` be the existing FP32 weight matrix after the established adjacent-K swap, shape `[2048,12352]`, and `xordered` the correspondingly gathered BF16 input widened to FP32.

Setup only:

```python
wp = torch.nn.functional.pad(Wordered.reshape(4,512,12352), (0,960))
wp = wp.permute(1,0,2).contiguous().reshape(512,4,32,416)
# Upload wp as FP32, TILE, interleaved DRAM.
```

Runtime layout:

```python
# Rearrange gathered x, without arithmetic or dtype narrowing:
coefficients = reshape(transpose(reshape(xordered, (4,512)), 0, 1), (512,4,1,1))
products = ttnn.multiply(wp_device, coefficients)  # [512,4,32,416], FP32
partial = ttnn.experimental.fast_reduce_nc(
    products, dims=[0], memory_config=DRAM, compute_kernel_config=self.compute)
# [1,4,32,416] -> [4,1,13312], slice channel dimension to 12352.
# Keep current left-associated addition of the four partials.
```

Use the model's reshape/layout helper, not an assumed metadata-only reshape across padded dimensions. `coefficients` must remain FP32. Padding weights with zero does not enter a valid channel's sum. Keep separate multiply and accumulation, avoiding fused multiply-add. Do not combine the four partials with generic sum.

For each output channel c and block b, the reduction computes the exact established sequence `p[b,0,c] + p[b,1,c] + ... + p[b,511,c]`. Packing c into H/W lanes changes only location, not operands or order. A CPU microprobe with full `[2048,12352]` BF16-representable random weights/inputs and explicit FP32 sequential additions found **zero differing FP32 codes in all four partials** between the current and proposed layouts. This proves the layout mapping, not the native execution.

The candidate product occupies 109,051,904 bytes versus 101,187,584 bytes currently (+7.77%). Its terminal output is 212,992 bytes instead of a 101,187,584-byte prefix tensor. The reduction still reads every product, and product generation still materializes all products. It uses only 52 output tiles, hence at most 52 working cores: fewer writes do not guarantee a speedup relative to a scan with more independent column work units.

A later tighter packing can pad channels to 12544 and pack the four blocks into `[512,32,1568]`, reducing padding to 1.55% and output to 200,704 bytes. This requires each block's input coefficient repeated over eight tile rows. The parent's `[512,4,32,416]` version has simpler broadcasting and is the preferred first control.

Minimal validation: compare all FP32 partial codes and final BF16 projection codes against the current ordered-cumsum path on the same real input (include the known Z/A/B mismatch-target positions 129/315/202). Then compare the full stream. Explicitly inspect tiny/subnormal and signed-zero differences; fast-NC copies the first value while cumsum initializes with +0, so signed-zero equality is the one source-level starting-convention caveat. Neither affects ordinary nonzero finite inputs. No output reuse/aliasing shortcut is required.

## Recurrence: two uncompensated scans preserve the cascade order

Replace the 16-row sequential-add stage and the 8-partial sequential-add stage with **two scans**, keeping the reduction axis as H in each. Do not scan all 128 rows at once: that removes the required rounding boundary between the eight partials.

```python
chunks = reshape(products, (32,8,16,128))
scan16 = ttnn.cumsum(chunks, -2, dtype=ttnn.float32, disable_compensated_sum=True)
partial = scan16[:, :, 15:16, :]
# Repack the 8 partial values as H, preserving increasing chunk order.
partial_rows = reshape(partial, (32,1,8,128))
scan8 = ttnn.cumsum(partial_rows, -2, dtype=ttnn.float32, disable_compensated_sum=True)
result = reshape(scan8[:, :, 7:8, :], (1,32,1,128))
```

Both scans perform a left-associated FP32 sequence. The first resets for each head/chunk and produces the same 16-term partial; the second resets for each head and sums those eight rounded partials in order. The reshape maps `partial[head,chunk,0,value]` to `partial_rows[head,0,chunk,value]`; it does **not** exchange the value and key axes. CPU explicit-sequence mapping validation has zero differing FP32 codes.

Source: `accumulation_tile_axis_compute.cpp:30–36,48–68` carries each column's running total down H, resetting at each independent run. Blackhole `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h:69–99` initializes zero and emits a sequence of SFPADD operations; compensation must remain disabled. Logical height16/8 padding is after the selected final row and cannot affect that row's sum.

This changes roughly 22 additions plus their slices into two scans, two slices and repacking. The first scan's tile-padded tensor is about 4 MiB; the second about 0.5 MiB. The former has 1024 independent H-scan runs and the latter 128, so output-lane parallelism is much less restrictive than the packed projection NC case. These are storage/work-unit bounds, not latency measurements.

## State-only replay

The current `compute_output=False` already skips output contraction, but `_recurrent_step` still repeats/scales/permutes Q before reaching that return. Move Q handling to the output-only branch. For state-only prefill replay, also avoid slicing Q and g each token when supplied `decay_value` makes both unnecessary.

The next model-only simplification is to prepare repeated/permuted K, FP32 V, beta and decay layouts **once for the whole prefill**, then slice the prepared per-token operands. These transformations do not depend on the recurrent state and introduce no arithmetic reassociation. Keep the token loop, separate `state*decay`, separate `(v-memory)*beta`, separate `k*delta`, separate addition, and the two-stage memory reduction. Do not introduce BF16 rounding between internal prefill steps; retain the public cache boundary convention. This also avoids repeatedly casting/repeating K/V for each row.

Native KDA chunk summary/scan ops expose `emit_outputs=False`, but that only suppresses output construction: their state updates still use matrix products and chunk transforms (`experimental/kda/recurrent_chunk_scan/device/kernels/compute/recurrent_chunk_scan.cpp:34–40,86–104,301–308`). They do not provide a switch to the exact per-token SFPU arithmetic described here. EMA is a scalar-alpha scan, not the key-dependent rank-one recurrence. Neither is a source-supported replacement preserving these boundaries.

A sequential replay and HF's chunked prefill reference need not be bit-identical merely because their formulas agree. The parent's state repair experiment remains the correctness discriminator; the optimizations above preserve that sequential candidate's arithmetic rather than independently proving its prefix-state contract.
