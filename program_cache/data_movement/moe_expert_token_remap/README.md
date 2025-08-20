## Program cache review — data_movement/moe_expert_token_remap

Status: Reviewed — no program cache issues found.

Findings
- New-infra mesh op. Hashing via default determinants; override iterates mesh programs and updates per-core runtime addresses from mesh buffers.
- Override updates:
  - Reader args[0..2] = mapping, metadata, topk addresses; writer args[0] and [3] = output mapping and reduced addresses.
  - Reference: `device/moe_expert_token_remap_program_factory.cpp:L234-L266`.
