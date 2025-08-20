## Program cache review — data_movement/sharded/reshard

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased factory offering three variants (same-width, same-height, generic). Each returns a program and override.
- Hashing: default determinants include shard specs, layout, page size relationships, and grid; runtime buffer addresses excluded.
- Overrides update:
  - Same-width/height: update remote buffer address in reader/writer runtime args and update dynamic CB bound to local tensor buffer.
  - Generic: update input address in reader runtime args and dynamic CB for output.
  - References: `device/reshard_program_factory.cpp` override lambdas in each variant (`L579-L601`, `L818-L840`, `L698-L718`).
