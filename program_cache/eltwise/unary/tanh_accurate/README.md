Tanh Accurate (unary) OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/unary/tanh_accurate` program factories (regular and sharded).
- No cache-hit issues found. Overrides update CB dynamic addresses for sharded variant and buffer addresses for interleaved path via shared unary infra.

Details

- Uses shared unary factories; override behavior matches `eltwise/unary` review.
