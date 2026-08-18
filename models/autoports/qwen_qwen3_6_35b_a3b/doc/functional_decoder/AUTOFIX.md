# AutoFix Closure

Issue: the first stage review found that linear-attention advertised context was not proven. The token-stepped gated-delta prefill path passed `131073` tokens but timed out at `262144` after 5400 seconds without TT_FATAL, traceback, watcher, NoC, L1, DRAM, or `tt-smi` health evidence.

Diagnostic artifacts:

- `AUTODEBUG_linear_context.md`
- `AUTOFIX_linear_chunked_design.md`
- `logs/context_probe_linear_prefill_131073_sparse.log`
- `logs/context_probe_linear_prefill_262144_long.log`
- `logs/tt_smi_list_after_linear_long_timeout.log`

Fix applied:

- Replaced linear prefill's token-stepped recurrence with a 64-token TTNN chunked gated-delta implementation.
- Kept single-token linear decode on the recurrent `_step` path for trace stability.
- Added static setup masks during module construction; the measured prefill/decode pass remains TTNN-only under the fallback audit.
- Added bounded fan-in concatenation for long streamed prefill outputs.

Post-fix evidence:

- `logs/autofix_chunked_linear_smoke.log`: synthetic linear prefill PCC `0.9994461003286241`, traced decode PCC `0.9994787951042461`, `2 passed`.
- `logs/autofix_chunked_linear_context_65.log`: linear chunk-boundary non-aligned prefill/decode `seq_len=65`, `1 passed`.
- `logs/autofix_chunked_linear_context_1025.log`: linear non-aligned prefill/decode `seq_len=1025`, `1 passed`.
- `logs/context_probe_linear_prefill_262144_chunked.log`: linear advertised prefill/decode `seq_len=262144`, `current_pos=262144`, `1 passed`, call time `315.48s`.
- `logs/context_probe_linear_prefill_262143_chunked.log`: linear near-max non-divisible prefill/decode `seq_len=262143`, `current_pos=262143`, `1 passed`, call time `320.42s`.
- `logs/context_probe_full_prefill_262143_sparse.log`: full-attention near-max non-divisible prefill `seq_len=262143`, `1 passed`, call time `359.25s`.
- `logs/real_weight_multitoken_moe.log`: real-weight representative linear/full layer coverage at `seq_len=1` and `seq_len=5`, including multi-token MoE prefill, `4 passed, 18 deselected`.
- `logs/correctness_full.log`: full post-fix correctness suite with real weights and context probes, `18 passed, 4 deselected`.
- `logs/runtime_fallback_audit.log`: runtime source audit plus live fallback-guarded linear/full prefill/decode, `3 passed, 19 deselected`.

Outcome: the advertised linear-attention functional-decoder context is now proven at `262144`; no capability reduction is recorded.
