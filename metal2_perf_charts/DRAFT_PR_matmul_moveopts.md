# matmul (Metal 2.0): build run-args/spec by move, not init-list copy

## Summary
`MatmulMultiCoreProgramFactory::create_program_spec` assembled the per-node run-args,
the kernel list, the work-units, and the tensor parameters with **initializer-list
assignments** (`run.kernel_run_args = {reader_run, writer_run};`,
`spec.kernels = {...};`, `spec.work_units = {...};`, `spec.tensor_parameters = {...};`).
A `std::initializer_list`'s elements are `const`, so every `= {...}` **deep-copies** its
payload — and for the per-node run-args and the KernelSpec list (source paths + bindings)
that is a large copy paid on **every dispatch** of the spec-as-key cache path.

This PR switches those assignments to `reserve` + `push_back(std::move(...))`.

## Scope
- **ttnn-only.** Touches a single file — the matmul MultiCore factory. **No changes to
  the Metal 2.0 host API** (`tt_metal/.../metal2_host_api/*`).
- Stacks on the spec-as-key matmul M2 port + the ttnn spec-as-key adapter (the
  `metal2_host_api` framework Audrey owns). Draft until that base lands.

## Result (device, MatmulMultiCore 512×256×384, cache-hit host dispatch, 10-run median)
| | host µs/call |
|---|---|
| spec-as-key baseline (init-list copies) | 75.3 |
| **+ move opts (this PR)** | **53.2** |

−22µs, **PCC 0.99996**, stdev 0.37. The std::string init-list copies were the bulk of the
avoidable host cost.

## Why it stops at 53µs (not the 27.1µs descriptor north star)
The remaining gap is **structural to spec-as-key**: the framework builds and hashes the
*whole* `ProgramSpec` every dispatch to determine the cache hit/miss — building it **is**
the lookup, so it can't be skipped on a hit. Going lower would require either:
- cheaper per-node argument tables (heap-free keys / positional run-args) — that lives in
  the Metal 2.0 host API, **out of scope** (framework-owned); or
- splitting `create_program_spec` into immutable-spec vs per-dispatch run-args so the
  framework can reuse the invariant part on a hit — an op-interface change.

Both are deliberately left out of this PR.
