## Program cache review — data_movement/pad

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra with multiple RM/TILE and sharded/non-sharded variants; each returns a program and override callback.
- Overrides update base addresses and dynamic CB addresses per core as needed:
  - RM single-core: reader/writer args[0,1] updated to src/dst addresses.
  - RM multi-core: recomputes per-core args via helper on cache-hit and sets them; dynamic CB recreated at creation time; override updates args only.
  - RM sharded variants: update dynamic CB addresses for input/output.
  - References: `pad_program_factory.cpp` overrides in functions ending with `override_runtime_args_callback` and explicit implementations at `L653-L681`, `L1317-L1331`, etc.
