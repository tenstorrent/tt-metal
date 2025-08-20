### slice_write program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/slice_write`

Findings:
- Supports interleaved input and sharded input (RM and TILE). Factories compute per-core runtime args and update them in override.
- Sharded input path updates the input CB address in override via `UpdateDynamicCircularBufferAddress(...)` and recomputes per-core reader/writer args.
- Interleaved path recomputes work split and resets all per-core args on cache hits.

No issues identified.
