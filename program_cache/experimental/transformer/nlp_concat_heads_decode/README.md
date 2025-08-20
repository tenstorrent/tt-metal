NLP Concat Heads Decode — Program Cache Review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp`.
- Cache-hit override updates output CB addresses and recomputes per-core reader/writer runtime args based on the current input buffer address. No issues found.

Key observations

- Output tensors use CBs with globally allocated addresses; override updates them via `UpdateDynamicCircularBufferAddress`.
- Input base address `q_start_addr` is taken from `input_tensors[0].buffer()->address()` on each cache hit.
- Per-core offsets (`in_tile_offset_by_batch`, indices) are rebuilt consistently with creation-time logic.

Conclusion

- Correct runtime updates on cache hit for both overlapped and non-overlapped core-grid variants. No cache correctness issues identified.
