# rotary_embedding: failure tests

- test_rotary_embedding_cachehit_decode_flag_mismatch.py: Expected hang on cache-hit due to under-keyed hash.
  - Program factory sets DECODE_MODE defines and different CBs when `token_idx` is provided.
  - Custom hash in `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp` omits `token_idx` and `compute_kernel_config`.
  - Cache entry created by prefill (token_idx=None) is reused for decode (token_idx set), leading to a hang.

Run:

```bash
pytest -q program_cache/experimental/transformer/rotary_embedding/failures/test_rotary_embedding_cachehit_decode_flag_mismatch.py -s --disable-warnings
```

Environment notes:
- Interleaved tensors, TILE layout, DRAM buffers.

Suggested fix:
- Include `token_idx` and `compute_kernel_config` in `compute_program_hash` for `RotaryEmbedding`, or refactor factory to normalize to a single kernel path controlled purely by runtime args.
