## Program cache review — data_movement/fill_pad

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra; writer-only kernel for in-place fill with override callback.
- Hashing includes dtype/layout/sharding and fill parameters via compile-time args; buffer base addresses excluded.
- Override updates writer base address per core at arg index 0.
  - Reference: `device/fill_pad_program_factory.cpp:L131-L149`.
