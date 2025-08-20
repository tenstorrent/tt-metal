## Program cache review — copy/typecast

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with multiple factories (interleaved, sharded, subgrid). Each factory defines `override_runtime_arguments` to update runtime-only values on cache-hit.
- Hashing: custom `compute_program_hash` is declared for the device op; factories use only hashed determinants (dtype/layout/memory config, grid split) to select kernels and CB sizes. Buffer base addresses are overridden at runtime.
- Interleaved variant updates per-core reader/writer base addresses at arg index 0; tile ranges are compile-time derived per core.
  - Reference: `device/typecast_program_factory.cpp:L155-L184`.
- Sharded variant uses globally allocated CBs bound to tensor buffers; on cache-hit, it updates CB dynamic addresses for input and output.
  - Reference: `device/typecast_sharded_program_factory.cpp:L156-L169`.
- Subgrid variant updates base addresses for the subset of cores captured in `cores_with_rtargs`.
  - Reference: `device/typecast_program_factory.cpp:L322-L352`.
