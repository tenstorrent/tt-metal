# Program cache review: experimental/transformer/rotate_half

Findings:
- Old-infra style `ProgramWithCallbacks` program in `device/single_core/rotate_half_program_factory.cpp`.
- Runtime override updates input and output buffer base addresses for reader and writer kernels.
- No sharded paths; single-core interleaved only. No custom compute_program_hash defined, default hashing uses op type + input tensors, which is sufficient.
- No runtime-only scalars varying between runs other than buffer addresses.

Conclusion:
- No program-cache issues identified.
