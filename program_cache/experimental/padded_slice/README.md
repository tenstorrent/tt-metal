### Padded Slice program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/padded_slice`
- Infra: old/type-erased (`operation::ProgramWithCallbacks` with override callback)

Findings:
- The program factory builds separate paths for RM and TILE input layouts and requires sharded outputs.
- On cache-hit, the override updates:
  - Output circular buffer base via `UpdateDynamicCircularBufferAddress(...)`.
  - All per-core runtime args for reader/compute/writer kernels by recomputing them from current tensors:
    - Reader args include input buffer base address and derived offsets.
    - Writer/compute args include counts and indices derived from hashed attributes.
- The default cache key (no custom hash) includes op type, attributes (`padded_slice_start`, `padded_slice_end`, `step`, `output_mem_config`) and tensor args, which together determine codegen (kernels, CB sizes, grid). Buffer addresses are excluded and correctly overridden at runtime.

No issues identified:
- No missing buffer address updates observed.
- Per-core iteration during override mirrors creation.
- No runtime-only scalars outside the hash are compiled-in.

Notes:
- RM path ties CB memory to the output buffer with `set_globally_allocated_address(*output.buffer())`; override updates this per run.
- Steps != 1 are not supported by design (guarded).

Suggested follow-up (optional):
- Add a non-regression two-run cache test to exercise both RM and TILE paths with different allocations to validate override paths, though no defect was found during review.
