# Program cache review: experimental/transformer/rotary_embedding

Findings:
- Program factory (`device/rotary_embedding_program_factory.cpp`) conditionally defines DECODE_MODE and configures CBs/kernels based on `token_idx`.
- Override updates buffer base addresses and token-dependent offsets for both reader and writer paths.
- Custom `compute_program_hash` in `device/rotary_embedding_device_operation.cpp` only includes `seq_len`, `output_mem_config`, and input tensors; it omits `token_idx` and the compute kernel config. This can cause prefill vs decode to alias in the same cache entry.

Tests:
- See `failures/test_rotary_embedding_cachehit_decode_flag_mismatch.py` for a two-run cache test expecting a hang on cache-hit when switching from prefill to decode under the same hash.

Suggested fixes:
- Include `token_idx` and `compute_kernel_config` in `compute_program_hash` for `RotaryEmbedding` (or ensure factory path is unified and purely runtime-controlled).
