## Program cache review — data_movement/sharded_partial/sharded_to_interleaved_partial

Status: Reviewed — no program cache issues found.

Findings
- Device op delegates to `sharded_to_interleaved_multi_core` factory with slice args; output comes from input[1] for partial path.
- Hashing: default determinants include slice parameters; runtime addresses excluded.
- Override updates writer output base address (or CB) and, for partial path, updates start index arg; input CB bound to source tensor is refreshed.
  - References: `device/sharded_to_interleaved_partial_op.cpp` and delegated factory override.
