NLP KV Cache Load Slice — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/device/nlp_kv_cache_load_slice_program_factory.cpp`.
- Override updates output CB base address and fully rebuilds per-core reader/writer runtime args using fresh input/output tensors and slice start. No issues found.

Key observations

- CB for output is set to the output buffer (globally allocated) and refreshed on cache hit.
- Reader args include current input buffer base and dynamic per-core start offsets; recomputed identically to creation logic.

Conclusion

- Cache-hit path correctly refreshes runtime-only values. No cache correctness issues identified.
