Moreh Arange – Program Cache Review

Findings: Reviewed `create(...)`, `override_runtime_arguments(...)`, and default hashing. No program-cache issues found.

- Override correctness:
  - Writer kernel: updates output buffer base address on cache-hit.
  - Other writer args (`tile_offset`, `num_tiles_per_core`, `start`, `step`, `element_size`) are derived from hashed attributes/tensor shapes and remain constant across cache hits.
- Hashing:
  - Uses default hash including operation attributes (`start`, `end`, `step`, `untilize_out`, `dtype`, `memory_config`) and tensor args (output spec via `any`/`output`).
  - Ensures cache-key stability for identical shapes/attrs while allowing different buffer addresses per run.
- Per-core coverage: Iterates cores identically to `create(...)`.

References:
- `ttnn/cpp/ttnn/operations/moreh/moreh_arange/device/moreh_arange_program_factory.cpp` – create/override and arg indices.

Status: Reviewed – no issues identified.
