NLP Concat Heads Boltz — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_boltz/device/nlp_concat_heads_boltz_program_factory.cpp`.
- Cache-hit override updates all buffer base addresses for both interleaved and sharded variants. No issues found.

Key observations

- Interleaved variant: reader arg 0 = input buffer address; writer arg 0 = output buffer address updated per core on cache hit.
- Sharded variant: input/output CBs created with globally allocated addresses; override updates CB addresses via `UpdateDynamicCircularBufferAddress`.
- Counts/offsets derive from hashed properties and stay constant for a given cache key.

Conclusion

- Override path mirrors creation-time arg ordering and correctly updates runtime-only addresses. No cache correctness issues identified.
