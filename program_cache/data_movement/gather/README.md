## Program cache review — data_movement/gather

Status: Reviewed — no program cache issues found.

Findings
- New-infra op with single-row single/multi-core factories; both provide overrides.
- Hashing: default determinants include shapes/tiles, DRAM/L1 flags, grid utilization, etc.; runtime addresses excluded.
- Override updates reader/writer buffer base addresses for each used core; also recomputes loop counts when width residuals exist.
  - References: `device/gather_program_factory.cpp` override methods for single-core and multi-core variants.
