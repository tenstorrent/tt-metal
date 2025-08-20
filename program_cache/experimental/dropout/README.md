# Program cache review — experimental/dropout

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra; override updates input/output buffer addresses and (if used) seed/start-id args.
- Hash includes probability, scale, dtype/layout/shape, memory config; seed should be treated as runtime-only and overridden to avoid over-keying.

## References
- Program factory: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`.

## Notes
- Verify tests use varying seeds across runs to exercise the override path.
