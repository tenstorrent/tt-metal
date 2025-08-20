### fast_reduce_nc program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc`

Findings:
- Program sets reader/writer runtime args per core with input/output buffer base addresses; override updates both correctly on cache hits.
- Work partitioning is recomputed in override using the same logic as create(...) ensuring indices remain consistent.
- Hash uses default; attributes include dim, output mem config via tensor specs, and input/output tensors, which drive codegen.

No cache/override issues identified.
