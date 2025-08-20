Complex Unary OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/complex_unary` device ops. This op uses composite kernels following the unary infra; buffer base addresses are updated on cache hits via shared unary override implementations.
- No cache-hit issues found.
