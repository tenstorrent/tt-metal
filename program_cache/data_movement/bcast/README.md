## Program cache review — data_movement/bcast

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with `ProgramWithCallbacks` per parallelization variant; provides an override callback.
- Hashing: custom `compute_program_hash` is present on the op type; determinants include math op, dim, and input/output tensor properties. No runtime buffer addresses are hashed.
- Override path updates per-core reader and writer buffer base addresses; other runtime args (tile counts, offsets, NC/Ht/Wt) are recomputed and set to match creation ordering.
  - Reference (H variant): `device/multi_core_h/bcast_op_multi_core_h.cpp:L185-L289`.
