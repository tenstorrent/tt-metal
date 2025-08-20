Generic Op — Program Cache Review

Status: Reviewed — potential under-keying risk noted, but override uses no runtime args so no stale-address risk.

Summary
- New infra device op that wraps a prebuilt `ProgramDescriptor` as attributes and expects a preallocated output tensor.
- Program factory `create(...)` constructs an empty program from the descriptor; no kernels or runtime args are set here.
- No custom `compute_program_hash(...)` present, so default hashing includes the `ProgramDescriptor` contents and tensor_args (including output tensor reference). Ensure `ProgramDescriptor` is hash-stable and excludes non-deterministic fields.

Override logic
- `override_runtime_arguments(...)` is empty; since no kernels/runtime args are set in `create(...)`, there’s nothing to update on cache hit. This avoids stale-addr issues but also means the op is effectively a no-op placeholder for generic programs.

Risk and suggestion
- If `ProgramDescriptor` contains unordered maps or non-deterministic fields, default hash may be unstable or under-keyed. Consider implementing a custom `compute_program_hash(...)` that serializes only deterministic, codegen-determinant fields in a stable order.
