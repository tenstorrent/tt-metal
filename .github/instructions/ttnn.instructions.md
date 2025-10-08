---
applyTo: "ttnn/**"
---

For TT-NN code:

- Prioritize readability and correctness over micro-optimizations; add brief docstrings/comments.
- API stability matters: changes should not force downstream code churn.
- Ensure Python bindings and C++ APIs stay in sync; add/adjust tests when changing signatures.
- Validate tensor shapes/dtypes early; produce clear error messages.
- if you change program factories, make sure nobody captures tensor in update_runtime_args lambda
