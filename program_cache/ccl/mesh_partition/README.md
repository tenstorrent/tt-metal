## Program cache review — ccl/mesh_partition

Status: Reviewed — no program cache issues found.

Findings
- New-infra mesh op that delegates to `data_movement::SliceDeviceOperation` per mesh coordinate and captures its override callback in shared variables.
- Hashing: default hash for mesh device operations; determinants include `dim`, `cluster_axis`, `output_mem_config`, and input tensor properties. No runtime-only addresses are hashed.
- Overrides: On cache-hit, it invokes the captured slice op override to update buffer addresses and per-run values. Iterates all programs in the mesh workload and applies the callback.
  - References:
    - Create and capture callback: `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp:L115-L129`.
    - Cache-hit override that forwards to slice override: `L133-L145`.
