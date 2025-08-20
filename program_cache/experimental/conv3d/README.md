# Program cache review — experimental/conv3d

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- New templated infra with `ProgramFactory` and explicit override. Override updates input/weights/bias/output buffer addresses; shape- and config-derived constants compiled in.
- Hash composed from `Conv3dConfig` fields, compute grid, dtype/layout, memory config, and input shapes via attributes and tensors.

## References
- Device operation and factory: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/*`.
- Tests include cache hash/address coverage.

## Notes
- If any runtime-only parameter (e.g., padding values) becomes changeable without affecting codegen, keep it out of hash and ensure it’s passed as runtime arg and overridden.
