NLP Create QKV Heads — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp` and headers.
- Interleaved variant provides an explicit `override_runtime_arguments` that updates all input/output buffer addresses; sharded variants rely on CB dynamic address updates. No issues found.

Key observations

- Reader runtime arg 0 updated to input Q buffer address; arg 1 updated to KV buffer when present.
- Writer runtime args 0/1/2 updated to output Q/K/V buffer addresses per core.
- Arg ordering matches creation-time push order.

Conclusion

- Cache-hit path properly updates runtime-only values. No cache correctness issues identified.
