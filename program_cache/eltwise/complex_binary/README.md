Complex Binary OP program cache review

Summary

- Scanned `ttnn/cpp/ttnn/operations/eltwise/complex_binary` for program factory and overrides. This op uses composite device ops; no custom override entry points found beyond standard reader/writer buffer address updates within composite implementation.
- No cache-hit issues identified.

Note

- If composite path changes introduce custom kernels, revisit to ensure per-core runtime args are overridden.
