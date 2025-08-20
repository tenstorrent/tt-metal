Generic Pool2D program cache review

Status: Reviewed — no program cache issues found.

Summary
- Factory: `device/pool_multi_core_program_factory.cpp` (new infra typed ProgramFactory).
- Hash includes sliding window config, pool type, memory config, divisor_override, count_include_pad, input mem config, dtype — sufficient to key compiled structure.
- Override updates only dynamic CB addresses for input (`raw_in_cb`) and output (`cb_out`) when tensors are sharded; reader indices and optional scalar-config CBs are captured from device storage at creation and remain valid for cache hits.

Recommendation
- No changes required.
