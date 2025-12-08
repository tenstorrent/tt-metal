---
applyTo: "ttnn/**"
---

For TT-NN code:

- Prioritize readability and correctness over micro-optimizations; add brief docstrings/comments.
- API stability matters: changes should not force downstream code churn.
- Ensure Python bindings and C++ APIs stay in sync; add/adjust tests when changing signatures.
- Validate tensor shapes/dtypes early; produce clear error messages.
- When changing program factories, do not capture tensor objects in update_runtime_args lambdas. Capturing tensors can lead to unintended memory retention, stale data, or incorrect program behavior, especially if the lambda is reused or the tensor's lifetime is not managed properly. Instead, pass only metadata (such as shape, dtype, or device) or use weak references if necessary. Avoid capturing actual tensor objects in lambdas to ensure safe and predictable execution.
