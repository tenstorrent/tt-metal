# DiffusionGemma live sweep validation (2026-07-10)

This record covers the fixed 256-token-context denoise-step sweep and the
inherited warmed 32/256/1024/2048-token context evidence. The 3072-token
warmed rerun was intentionally omitted at the priority handoff.

## Evidence integrity

- Ten passed step rows are backed by ten per-run JSON files and ten isolated
  server logs. The failed first K=4 startup is a separate excluded JSON/log;
  its one permitted retry is the passed K=4 row.
- Recomputing every referenced log digest from disk returned:
  `OK: recomputed 11 step-log hashes and 1 warmed-context log hash`.
- The warmed-context log digest is
  `757ce2d3af346f48fd33078215b7ab1845bdf741f4c9ef00e80668ce088a895b`;
  the log ends with the FastAPI `Application shutdown complete.` marker.
- The failed K=4 record contains the bounded no-process, list/reset/list,
  1x4 mesh-smoke, and successful single-retry record. It is excluded from all
  performance aggregates.
- A final process scan after validation found no vLLM API server, EngineCore,
  live sweep, serving smoke, `tt-smi`, or `tt-triage` process.

## Final host gates

All commands below exited zero unless a skip is stated.

| Gate | Result |
|---|---|
| `py_compile` on the adapter, serving core, and live harness | passed |
| Black and `py_compile` on the three touched DG test modules | passed |
| Complete touched-DG host pytest suite | 276 passed, 1 skipped in 88.70 s |
| `prefer-expect-error` pre-commit gate | passed |
| Black check on the changed plugin test | 1 unchanged |
| Focused tt-vllm plugin pytest suite | 13 passed in 4.64 s |
| `python -m json.tool` on compact step, compact context, and warmed-context JSON | passed |
| `git diff --check` in tt-metal and tt-vllm | passed |
| Reverse-apply check for model-runner, scheduler, and host-test patches | passed |
| No-shared-edits gate with `DG_BASE_REF=origin/diffusion-gemma-function` | passed |

The single pytest skip is the explicitly opt-in device serving smoke
(`DG_RUN_DEVICE=1`); the requested live OpenAI-server hardware sweep itself
completed separately. The no-shared-edits gate reported:

```text
OK: no shared-directory edits vs origin/diffusion-gemma-function (models/demos/gemma4, models/common, models/tt_transformers clean)
```

The environment did not provide `ruff`; Python compilation, Black checks,
focused tests, JSON parsing, patch checks, and both repository diff checks
were used as the executable host gates.

## Independent review

A fresh read-only stage review after the fixes returned `CLEAN PASS`, with no
release-blocking findings and both repositories judged safe to commit and
push. Its only non-blocking observation was the known nanobind leak warning
printed after clean resource release and server shutdown.
