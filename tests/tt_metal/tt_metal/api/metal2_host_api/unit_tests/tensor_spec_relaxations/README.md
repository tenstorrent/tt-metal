# TensorSpecRelaxations tests

Host-only tests, with no device, of the match and hash relation declared in `tensor_spec_relaxations.hpp`.
`SetProgramRunArgs` uses this relation to decide whether a tensor argument fits its TensorParameter.

Suite: `TensorSpecRelaxations`.

| File | Tests |
|---|---|
| `tensor_spec_relaxations.cpp` | 20 |

What is pinned:

- With no flags set, matching is `TensorSpec` equality.
- `dynamic_tensor_shape` frees the logical shape but keeps the rank, unless `relax_logical_rank` is also set.
- `match_padded_shape_only` frees the logical shape within the padded shape. `dynamic_tensor_shape` takes
  precedence when both are set, though the two are not strictly ordered.
- `match_page_size` pins the row-major width; it is a no-op on tiled layouts.
- `relax_logical_rank` and `match_page_size` have no effect without `dynamic_tensor_shape`.
- The shard distribution strategy stays pinned under `dynamic_tensor_shape` and `relax_logical_rank`.
- For all 16 flag combinations, two specs match exactly when their relaxed hashes are equal.

Related: run-time matching in [`../program_run_args/tensor_args.cpp`](../program_run_args/tensor_args.cpp), and
the kernel hash under `dynamic_tensor_shape` in
[`../kernel_hash/tensor_spec_relaxations.cpp`](../kernel_hash/tensor_spec_relaxations.cpp).
