This model-local binding runs K2's long-context prefill attention. It reuses the
installed SDPA descriptor, paged reader, causal mask, and writer. It changes only
accumulator storage and arithmetic in a generated private compute kernel. No TTNN
or Metal source is modified.

Build once during model setup, before measurements or trace capture:

```bash
python -m models.demos.k2_horizon_7b_qb2.tt.accurate_attention.build
```

The build requires this checkout's existing `build/build.ninja`, TTNN libraries,
nanobind static library, and `clang++-20`. It performs host compilation only.
Generated kernel sources and the extension stay in `.build/`. Exact source
substitutions fail if the upstream call sites change. Build provenance records
source/library SHA256 values and compiler arguments; loading rejects stale
artifacts.

```python
from models.demos.k2_horizon_7b_qb2.tt.accurate_attention import accurate_attention

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

`accurate_attention(..., grid=(x, y), math_fidelity=...)` selects the worker grid and fidelity. Evidence:
`tests/check_accurate_prefill_attention.py` (relative L2 against an FP64 oracle).

Decode no longer uses this binding: beyond the stock decode bound the model runs the stock
`paged_scaled_dot_product_attention_decode` with `SDPAProgramConfig(fp32_accumulation=True)`.
