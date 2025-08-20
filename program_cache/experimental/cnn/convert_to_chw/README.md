# Program cache review — experimental/cnn/convert_to_chw

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra with override callback. Reader updates input address; writer updates output address. CB sizes and page sizes are compile-time from hashed attrs.
- Hash covers dtype/layout/page config, grid size, and memory config via op attributes and input tensor props.

## References
- Program and override: `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/convert_to_chw_program_factory.cpp`.
- Unit test includes program-cache check.

## Notes
- If grid selection becomes dynamic at runtime, ensure the selector is hashed or encoded into runtime args appropriately.
