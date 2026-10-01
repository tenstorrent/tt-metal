This model-local binding reuses the installed SDPA descriptor, paged reader,
causal mask, and writer. It changes only accumulator storage and arithmetic in
a generated private compute kernel. No TTNN or Metal source is modified.

Build once during model setup, before measurements or trace capture:

```bash
python -m models.autoports.ifm_k2_horizon_7b.tt.accurate_attention.build
```

The build requires this checkout's existing `build/build.ninja`, TTNN libraries,
nanobind static library, and `clang++-20`. It performs host compilation only.
Generated kernel sources and the extension stay in `.build/`. Exact source
substitutions fail if the upstream call sites change. Build provenance records
source/library SHA256 values and compiler arguments; loading rejects stale
artifacts.

```python
from models.autoports.ifm_k2_horizon_7b.tt.accurate_attention import accurate_attention

output = accurate_attention(
    q, k_cache, v_cache, page_table,
    chunk_start_idx_tensor=offset,
    q_chunk_size=32,
    k_chunk_size=128,
)
```

Either a scalar `chunk_start_idx` or a device `chunk_start_idx_tensor` is required.
The latter retains stock SDPA's device-read offset and replay behavior. Scalar
offsets participate in the program cache key. Input and output tensor contracts
and validation come from the production SDPA operation.

The recurrence keeps numerator and denominator in FP32. Aliases of their
circular buffers enable `UnpackToDestFp32`; SFPU multiply/add therefore avoids
TF32 source-register truncation between key chunks. The alias read pointers
follow the production CBs, as in TTNN's matmul partial-reload implementation.
The final output is converted to BF16 once. Maxima and correction factors retain
the production BF16 format, and the final denominator row reduction and
reciprocal retain two isolated TF32 reloads. These bounded final conversions do
not repeat with context length.

`fp32_output_accumulator=False` is an isolated diagnostic switch: denominator
arithmetic stays repaired while numerator storage remains BF16. Production
callers should use the default `True`.

## Accurate flash decode

`accurate_flash_decode(q, k_cache, v_cache, page_table, cur_pos, max_cores_per_head=..., k_chunk_size=...)`
keeps the stock `paged_scaled_dot_product_attention_decode` tensor contract, reader, writer, split-K work
distribution and tree reduction, and removes the stock path's precision losses:

* half-tile (16x32) CBs are promoted to 32x32 tiles, so the kernel runs `VectorMode::RC` and the accurate,
  clamped exponential (stock half-tile decode takes the approximate branch);
* scores, outputs, sums, correction factors and tree-exchange buffers are FP32, and recurrence CBs are
  `UnpackToDestFp32`. Per-chunk and tree merges run on SFPU (`accurate_decode.hpp`);
* row sums are `P @ col_identity` in FP32 destination registers, because `reduce_tile<SUM>` does not handle FP32
  score tiles.

The generated kernel `.build/<fingerprint>/sdpa_flash_decode.cpp` comes from the stock
`sdpa_flash_decode.cpp` with guarded substitutions. It runs at HiFi4, as the chunked fallback does. The
model uses it beyond the stock decode bound. Evidence: `doc/long_context_perf/FINDINGS.md`.

`accurate_attention(..., grid=(x, y), math_fidelity=...)` selects the worker grid and fidelity for single-offset
calls. Per-request offsets keep the pinned 8x8 grid.
