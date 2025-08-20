NLP Create QKV Heads Decode — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp`.
- Cache-hit override updates output CB addresses (Q/K/V) and re-applies per-core reader/writer runtime args using the current input buffer base and offsets. No issues found.

Key observations

- Uses multiple program variants (interleaved/sharded, overlapped grids). All variants update CBs and per-core args consistently.
- Input base `q_start_addr` is refreshed from the input tensor buffer on cache hit.

Conclusion

- Runtime-only values are refreshed correctly on cache hits across variants. No cache correctness issues identified.
