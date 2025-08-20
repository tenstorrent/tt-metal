Scaled Dot Product Attention (SDPA) program cache review

Status: Reviewed — no program cache issues found.

Summary
- Program factories reviewed: `sdpa`, `joint_sdpa`, and `ring_joint_sdpa` under `ttnn/cpp/ttnn/operations/transformer/sdpa`.
- All factories provide override callbacks that correctly update per-run tensor buffer base addresses for inputs and outputs.
- Chunked SDPA updates page-table base address and the `chunked_q_chunk_offset` runtime argument on cache hits.
- Custom program hash (`ScaledDotProductAttention::compute_program_hash`) includes operation attributes and both input/optional tensors, covering scale, causal flag, program config, chunked mode, and compute kernel config.

Key references
- `device/sdpa_program_factory.cpp`
  - Reader args order at create-time includes: q, k, v, mask, page_table, per-core ranges, and `chunked_q_chunk_offset` (last). On override, indices 0–4 and 12 are updated.
  - Writer args order includes output base and per-core ranges, with `chunked_q_chunk_offset` last; override updates index 0 and 8.
  - Compute args include per-core ranges with `chunked_q_chunk_offset` last; override updates index 7.
- `device/joint_sdpa_program_factory.cpp`
  - Override updates all input addresses (q, k, v, joint_q/k/v) and both outputs.
- `device/ring_joint_sdpa_program_factory.cpp`
  - Override updates all input addresses (q, gathered k/v, joint_q/k/v) and all outputs (out, joint_out, lse).

Notes
- Compile-time arguments include tensor shapes, chunk sizes, dtype-derived tile sizes, causal/mask usage, and for chunked mode, page table properties. These are part of the hash via attributes and tensors, preventing stale code reuse across incompatible runs.
- No missing CB size/page-size updates were identified for the cache-hit path, as those parameters are compile-time and covered by the hash.

Recommendation
- No changes required. Add targeted cache-hit tests in the future if kernel argument ordering changes.

SDPA — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_op.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_op.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

Findings:
- Uses the old type-erased infra with an override callback.
- Override updates all runtime-only values on cache hits:
  - Reader: Q/K/V and optional mask/page_table buffer base addresses, and chunked Q offset.
  - Writer: output buffer base address and chunked Q offset.
  - Compute: per-core indices and chunked Q offset.
- For paged/chunked mode, page table base address and chunked_q_chunk_offset are recomputed and updated per hit.
- Program hash includes all determinants that affect codegen and program shape: head_dim_v, scale, output_mem_config, program_config (q/k chunk sizes, grid), is_causal, chunked-mode flag, compute_kernel_config, plus input_tensors and optional_input_tensors (hashed by properties, not buffer addresses). This avoids under/over-keying.

Existing cache tests:
- `tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py` includes multiple two-run cache tests:
  - `test_sdpa_tt_with_program_cache` (interleaved causal/noncausal)
  - `test_sdpa_chunked` and `test_sdpa_chunked_iterate_batch` (paged cache / chunked prefill)
  - These assert the cache is reused (`num_program_cache_entries() == 1`) and validate outputs across cache hits.

No program-cache issues identified.

Suggested optional test (not required given existing coverage):
- Minimal two-run cache test that varies only buffer base addresses (and, in chunked mode, chunk_start_idx) to exercise the override path; expect correctness on cache hit.
