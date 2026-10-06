# Native attention DRAM estimate

`tests/summarize_perf.py` estimates traffic from profiler operand metadata. Native
paged decode SDPA reads the current causal or sliding context, rounded to its
read chunk; it does not read every allocated cache page. The denominator remains
the complete layer: first native firmware start through last firmware end,
including normalization, residuals, routing, conversions, and intervening gaps.
Decode uses the mean of all 128 successive trace replay windows. The useful-FLOP
numerator and non-SDPA traffic rules are unchanged.

The source basis is:

- `tt/optimized_decoder.py`, `NativePagedAttention`: batch-one non-MLA GQA,
  interleaved DRAM query, separate paged K/V caches, dynamic `k_chunk_size=0`,
  FP32 destination accumulation and full destination synchronization.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:35`:
  `get_workload_for_core` rounds the window start down and its exclusive end up
  to the K chunk, then partitions chunks between workers. At line 103,
  `get_dynamic_Sk_chunk_t` selects a power-of-two tile count.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:387`:
  dynamic chunks are capped at four 32-token tiles with FP32 accumulation;
  `dst_full_sync_en` does not change this cap. The non-MLA query path retains
  `q_heads_parallel_factor=1` (line 110).
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:279`:
  each assigned head/chunk reads K and V separately. The ordinary read paths in
  `dataflow_common.hpp:703` and `:780` transfer each selected full tile once.
- `tt_metal/impl/data_format/tile.cpp:70`: tile size includes exponent storage.
  A 32x32 BFP8 tile is 1,088 bytes, giving 1.0625 bytes per element; BF16 uses two.

For absolute positions 4096 through 4223, the effective native chunk is 128
tokens. Sliding attention has eight KV heads of width 256 and a 1,024-token
logical window. Its rounded span is 1,152 tokens for the first 127 positions and
1,024 at position 4223. Full attention has two KV heads of width 512 and reads
4,224 rounded tokens at every position. These are source-derived byte estimates,
not device measurements:

| K+V reads per decode | Mean tokens/head | Mean BF16 bytes | Mean BFP8 bytes | BFP8 full-pool bytes at 160 pages |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 1,151 | 9,428,992 | 5,009,152 | 22,282,240 |
| Full | 4,224 | 17,301,504 | 9,191,424 | 11,141,120 |

The byte formula is `2 * rounded_tokens * kv_heads * head_width * storage_bytes`.
Only the native SDPA K/V inputs receive this correction. Cache updates still
count one 32-token page read and write per cache, and other operations retain
their existing padded-operand accounting, including complete-cache conversions
where present. Index/page-table operands retain one generic read each. Additional
per-worker query/page-table rereads, NOC traffic, reduction traffic, and profiler
writes are outside this estimate; the result is not a DRAM-controller counter.

Run from the repository root, using an actual profiler CSV:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/summarize_perf.py \
  /path/to/ops.csv --layer-type sliding_attention --output /path/to/whole_layer.json \
  --decode-start-position 4096 --precision-policy 'State the measured runtime policy here'
```

The parser reads `k_chunk_size`, `fp32_dest_acc_en`, and operand dtypes from the
CSV. If chunk metadata is missing, the caller must provide the effective chunk
explicitly with `--native-sdpa-read-chunk-size 128` for this target configuration;
an override contradicting present metadata fails. Layer type supplies the
model's fixed window and head geometry, checked against available metadata.
Unsupported sharded queries, circular addressing, MLA, or cache shapes fail.
Positions follow replay groups sorted by firmware start, matching
`tests/run_decoder.py:360` onward. JSON and replay CSV expose those positions,
logical/rounded spans, K/V dtypes, and estimated bytes.

CPU verification: synthetic metadata arithmetic covered both layer types, both
cache dtypes, endpoint rounding, missing/conflicting chunk metadata, and rejected
unsupported geometries. On the existing
`doc/fused_decoder/tracy/sliding_verified/ops.csv`, all 22,382 device-row traffic
estimates and the complete-layer latency, traffic, and roofline outputs were
identical to the preceding implementation. Black and Python compilation passed.
No hardware was run for this estimator change.
