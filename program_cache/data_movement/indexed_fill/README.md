## Program cache review — data_movement/indexed_fill

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra; override callback provided.
- Hashing: default determinants include batch shapes and page size; runtime addresses excluded.
- Override updates reader args with batch_ids/input_a/input_b addresses and writer arg[0] with output address, recomputing per-core local sizes.
  - Reference: `device/indexed_fill_op_multi_core_program_factory.cpp:L100-L136`.
