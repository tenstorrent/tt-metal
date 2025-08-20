NLP Create QKV Heads Boltz — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_program_factory.cpp`.
- Interleaved variant’s typed override updates input Q and optional KV buffer addresses and all three output addresses. No issues found.

Key observations

- Shared variables expose kernel IDs and core topology; override iterates cores and updates reader/writer args in-place.
- Arg indices preserved across create/override.

Conclusion

- Override path correctly updates all runtime-only addresses. No cache correctness issues identified.
