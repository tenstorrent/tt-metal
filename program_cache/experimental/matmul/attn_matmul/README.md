# Program cache review — experimental/matmul/attn_matmul

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra; override updates buffer addresses and any runtime-only scalars. Program selection depends on shapes, dtypes, layouts, memory configs, and matmul tiling/program config encoded in attributes.
- Hashing appears to include op config and tensor props; tests exist for program cache behavior.

## References
- Program factory and device op: `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/*`.
- See `tests/tt_eager/python_api_testing/unit_testing/misc/test_attn_matmul.py::test_attn_matmul_with_program_cache`.

## Notes
- Keep matmul program-config parameters in the hash; avoid putting per-invocation scalars in the hash.
