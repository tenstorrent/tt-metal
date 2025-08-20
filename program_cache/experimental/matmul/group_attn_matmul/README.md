# Program cache review — experimental/matmul/group_attn_matmul

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra; override updates input/output buffer addresses and runtime scalars.
- Hash includes group matmul program config (tiling, mcast flags), tensor props, and memory configs; adequate for selecting compiled kernels.

## References
- Program factory and device op: `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/*`.
- Program-cache tests: `tests/tt_eager/python_api_testing/unit_testing/misc/test_attn_matmul.py::test_group_attn_matmul_with_program_cache`.

## Notes
- Ensure any future fused behaviors keep their runtime-only controls out of the hash and are overridden properly.
