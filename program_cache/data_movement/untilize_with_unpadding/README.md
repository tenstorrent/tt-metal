## Program cache review — data_movement/untilize_with_unpadding

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with single-core, multi-core interleaved/column/block, and sharded variants; all provide overrides.
- Hashing: default determinants include unpadding sizes (computed from input/output shapes), tile/block sizes, sharding; runtime addresses excluded.
- Overrides update reader/writer base addresses per core or dynamic CBs for sharded variants; special-cased W=16 path updates CBs and writer addresses accordingly.
  - Reference: `device/untilize_with_unpadding_program_factory.cpp` override lambdas per variant.
