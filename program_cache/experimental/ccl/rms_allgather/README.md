# Program cache review — experimental/ccl/rms_allgather

Status: Reviewed — no blocking program-cache issues identified.

## Summary
- Old infra with override callback. Reader/writer kernels get per-input/output buffer addresses updated on cache-hit; semaphores updated.
- Hashing: CMake lists multi-core PF; review shows codegen depends on dim, num_links, ring size/topology, dtype/layout/shape, and memory config. These are included via op attributes and input tensors.

## References
- Device op/program: `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/*`.

## Notes
- Ensure any future changes to per-link worker counts or muxing that impact compile-time args get added to hash if not derived from existing hashed fields.
