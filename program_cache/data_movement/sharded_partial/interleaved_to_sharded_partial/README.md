## Program cache review — data_movement/sharded_partial/interleaved_to_sharded_partial

Status: Reviewed — no program cache issues found.

Findings
- Device op delegates to the full `interleaved_to_sharded_multi_core` factory and passes `num_slices` and `slice_index`.
- Hashing: default determinants include slice count/index through op attributes; runtime addresses excluded.
- Overrides from the delegated factory update input base address and, when partial, the start index field; writer updates output address or output CB as needed.
  - References: `device/interleaved_to_sharded_partial_op.cpp` and delegated factory override.
