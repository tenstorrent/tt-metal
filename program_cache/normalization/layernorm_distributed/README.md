LayerNorm Distributed program cache review

Status: Reviewed — no program cache issues found.

Reviewed files
- `device/multi_core/layernorm_pre_all_gather_op_multi_core.cpp`
- `device/multi_core/layernorm_post_all_gather_op_multi_core.cpp`

Findings
- Override updates input and output base addresses per core for both pre/post all-gather variants; addresses are the only runtime-only values. All partitioning params and CB sizes are derived from shapes/config and compiled into the program.
- Reader/writer arg indices in override match creation order; only index 0 (address) is updated.

Recommendation
- No changes required.
