Accurate attention for the long-context paths.

Prefill (and the chunked decode fallback) run TTNN's paged chunked SDPA with the ACCURATE precision recipe
(`precision=ttnn.SDPAPrecision.ACCURATE`): HiFi4 QK and PV, FP32 scores, row maxima, sums and output
recurrence, an accurate FP32 correction exp, and one final conversion to BF16
(tech_reports/FlashAttention/SDPAPrecisionRecipes.md).

```python
from models.demos.k2_horizon_7b_qb2.tt.accurate_attention import accurate_attention

output = accurate_attention(
    q, k_cache, v_cache, page_table,
    chunk_start_idx=start,  # or chunk_start_idx_tensor=offset (device scalar, trace-safe)
    q_chunk_size=64,
    k_chunk_size=1024,      # 0 / 0 leaves the blocking to the op
    grid=(11, 10),
)
```

Decode uses a model-local binding (below) that reuses the installed SDPA decode descriptor, reader, writer and
work split, and changes only accumulator storage and arithmetic in a generated private compute kernel. No TTNN or
Metal source is modified. Build it once during model setup, before measurements or trace capture:

```bash
python -m models.demos.k2_horizon_7b_qb2.tt.accurate_attention.build
```

The build requires this checkout's existing `build/build.ninja`, TTNN libraries,
nanobind static library, and `clang++-20`. It performs host compilation only.
Generated kernel sources and the extension stay in `.build/`. Exact source
substitutions fail if the upstream call sites change. Build provenance records
source/library SHA256 values and compiler arguments; loading rejects stale
artifacts.

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
model uses it beyond the stock decode bound. Evidence: `tests/check_accurate_flash_decode.py` and
`tests/check_accurate_prefill_attention.py` (relative L2 against an FP64 oracle).

`accurate_attention(..., grid=(x, y))` selects the worker grid of the recipe call.
