# Program cache review — experimental/cnn/convert_to_hwc

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra with override callback. Reader/writer update input/output buffer addresses on cache-hit.
- Hash includes dtype/layout/page config, memory config, and grid selection via attributes and input tensor props.

## References
- Program and override: `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.cpp`.

## Notes
- Maintain runtime-arg ordering alignment when modifying kernels.
