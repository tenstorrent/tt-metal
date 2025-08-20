## Program cache review — data_movement/fold

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with `Fold::MultiCore` factory; uses globally allocated CB addresses for input/output and updates dynamic CB addresses on cache-hit.
- Hashing: default determinants include stride_h/stride_w, sharding grid/shape, layout and dtype; addresses excluded.
- Override updates:
  - Dynamic CB base addresses for input/output; re-sets writer runtime args with derived sizes/strides consistent with creation.
  - Reference: `device/fold_multi_core_program_factory.cpp:L94-L148`.
