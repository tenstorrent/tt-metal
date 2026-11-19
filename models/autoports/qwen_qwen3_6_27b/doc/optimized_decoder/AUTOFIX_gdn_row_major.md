# Direct convolution-output intake

Source-only candidate work, 2026-11-12. The stage owner's accurate profile identifies approximately0.33ms in conversions around convolution outputs: row-major output, conversion to tile layout, and padding to32 sequence rows solely for a recurrence reader that consumes row0. The reported recurrence GenericOp itself is826us. These are parent-reported profile findings, not timings of the new variants.

## Separate row-major inputs

`candidates/gdn_sfpu_row_major.py` adds direct BF16 row-major Q/K `[B,1,2048]` and V `[B,1,6144]` intake. Call the existing `recurrent_sfpu_compact(q,k,v,g,beta,state,normalized=False,...)` interface. Input layout selects the reader mode. Gates remain tile-layout `[B,1,48]` and state remains tile-layout FP32 `[B*48,128,128]`; output is unchanged.

The new reader compile-time flag is argument5, and TensorAccessor argument groups begin at6. A row-major page is one complete channel row:4096 bytes for Q/K and12288 for V. For head h, `user=h//48`, `hv=h%48`, `hk=hv//3`. Each vector is one256-byte read from page`user`, offset`hk*256` for Q/K or`hv*256` for V. Local destinations are scratch offsets0,512,1024. BF16 widening consumes contiguous `src16[i]`. Source and destination addresses are64-byte aligned. Unlike tiled face reads, row-major reads need no inter-face padding or overfetch.

The host validates the exact logical row-major shapes and that padded shapes match, along with DRAM interleaved placement. The original tile-layout interface remains available for comparison. No layout conversion or padding is performed by the helper.

## Packed row-major input

`candidates/gdn_sfpu_packed_row_major.py` additionally supports one BF16 row-major QKV tensor `[B,1,10240]`. Pass the same tensor as q, k, and v and set `packed_qkv=True`. The reader uses20480-byte pages and byte offsets0/4096/8192 for Q/K/V segments, respectively, plus the same per-head256-byte offsets. The host requires all three arguments to refer to the same buffer. This can remove the three preparatory slices as well as tile conversion and padding.

Packed mode adds compile-time flag6 and moves TensorAccessor groups to7. All three accessors receive the same buffer and page geometry; the per-input tensor arguments remain registered with generic_op. Output and stable in-place state behavior are unchanged.

## Integration and validation

Both variants intentionally preserve `COMPUTE` and `WRITER` byte-for-byte from the corrected compact helper. The stage owner can combine their reader/host changes with the already selected optimized compute string. These candidates do not themselves select a different normalization or numerical policy: raw mode invokes the existing internal FP32 normalization. If the selected path uses an externally normalized control, establish equivalence of that normalization before switching it to raw intake.

Host checks passed for Python AST syntax, balanced reader braces, unchanged compute/writer strings, and B1/B3/B32 head-to-row mappings with64-byte alignment and bounds. No hardware or C++ compilation was performed by this helper. Required device checks are the isolated own-input recurrence probe, full decoder short and repeated correctness, trace changed-input/state-allocation checks, watcher, and before/after profiling showing the removed conversion/slice rows and improved full latency. The first implementation/API failure should be diagnosed rather than treated as rejection of direct row-major intake.

Existing source precedent: `ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/slice_write_reader_interleaved.cpp` constructs a TensorAccessor using a row-major `stick_size`, matching this candidate's complete channel-row page interpretation.
