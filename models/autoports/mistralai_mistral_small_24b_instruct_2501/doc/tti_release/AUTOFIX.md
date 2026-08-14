# AutoFix: Mistral release-warning investigation

## Final status

The trace allocation defect is fixed and verified on hardware. The tokenizer-regex hypothesis is refuted for the live request path. The release remains `release-workflow-fail` and `readiness-fail` because IFEval is 75.6635% versus 78.755% and GPQA is 38.8889% versus 40.3%; no mask or waiver is valid, and neither warning provides a surviving mechanism that predicts those scores will improve.

## Hypotheses and verdicts

The live-bad-regex hypothesis is refuted. The TT platform's registered renderer/tokenizer constructs HF with `fix_mistral_regex=True`. `tokenizer_regex_ab.json` applies the exact checkpoint chat template to three preserved IFEval and three preserved GPQA message sets. Every live-plugin token sequence equals HF `fix=true`, while every `fix=false` sequence differs from token index 3. The remaining warning comes from a separate default processor/template probe in the APIServer process.

The harmless-trace-warning hypothesis is also refuted. The original order retained model and sampling traces before the first KV-cache reset compiled its in-place `ttnn.fill` programs, violating the allocator invariant. Commit `993aabf2f73b911062c3bb59412af0b5fdb3e456` compiles the reset in the eager pre-capture phase and retains the post-capture reset to clear warmup-written state. A regression proves reset -> capture -> reset ordering.

## Verification

- Autoport full-model tests: 9 passed, 4 hardware-gated skipped.
- Nested vLLM tokenizer tests: 6 passed.
- Exact committed P300x2 server: zero active-trace allocation warnings and no fatal/traceback errors.
- Health, models, deterministic chat, and non-page-aligned benchmark requests returned HTTP 200.
- Deterministic response: exactly `TRACE SMOKE OK`.
- Benchmark: 1 completed, 0 failed; 36 actual prompt tokens, 16 output tokens, mean TTFT 939.16 ms, mean TPOT 27.45 ms.
- Shutdown left no API-server or EngineCore process. Reset completed, all four p300c chips were visible/resettable, and a fresh `MeshShape(1, 4)` opened and closed successfully.

Evidence is preserved beside this report in `server_tracefix_smoke.log`, `chat_tracefix_smoke.json`, `benchmark_tracefix_smoke.log`, `benchmark_tracefix_smoke.json`, and `tokenizer_regex_ab.json`. Fresh-context diagnosis and source citations are in the repository-root `AUTODEBUG.md`.

## Why no full quality rerun

The exact A/B refutes bad regex tokenization on the live path. The trace repair removes a real unsupported startup allocation, but no evidence connects the old warning to changed answers in the completed v9 evaluation. Repeating roughly eight projected hours of unrestricted evaluation, or rerunning the same subset, would not constitute a proven quality fix. The unwaived quality failures are therefore reported unchanged. A future attempt needs a new, falsifiable model-quality hypothesis rather than a threshold mask.
