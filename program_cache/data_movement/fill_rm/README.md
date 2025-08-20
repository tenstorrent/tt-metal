## Program cache review — data_movement/fill_rm

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra; single-core program and override callback.
- Hashing: determinants include N,C,H,W, fill window and values, dtype/layout; addresses excluded.
- Override updates the writer base address at arg index 0 for the single core.
  - Reference: `device/fill_rm_op.cpp:L74-L90`.
