Embedding OP program cache review

Summary

- Searched for `ttnn/cpp/ttnn/operations/embedding` device operation and program factory overrides. No explicit override functions found; likely uses existing data movement/unary infra patterns.
- No cache issues identified in current tree.

Note

- Revisit if/when an embedding-specific program factory lands.
