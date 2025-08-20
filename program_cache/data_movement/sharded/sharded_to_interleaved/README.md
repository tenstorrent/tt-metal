## Program cache review — data_movement/sharded/sharded_to_interleaved

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased program factory with override callback; supports partial slicing and dtype conversion.
- Hashing: default determinants include shard spec, layout, dtype conversion flag, grid usage, and slice count; runtime addresses excluded.
- Overrides update:
  - Reader: updates dynamic CB bound to input buffer.
  - Writer: updates output base address (or dynamic CB) and, for partial variants, updates the start index argument.
  - Reference: `device/sharded_to_interleaved_program_factory.cpp:L253-L288`.
