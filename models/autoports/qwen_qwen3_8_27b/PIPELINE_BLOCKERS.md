# Pipeline blocker log

## 2026-09-12: stage 6 checker could not find `python`

- Symptom: stages 1 through 6 completed, then `06-full-model.check.sh`
  exited 127 with `python: command not found`.
- Cause: the runner invoked its Python interpreter by absolute path but exported
  only the Node directory to `PATH`. Stage check scripts invoke `python` by name.
- Repair: prepend both `python_env/bin` and the pinned Node directory to `PATH`.
- Validation: rerunning the stage 6 check in the container with the corrected
  path passed the non-degenerate-output and 262144-token context checks.
- Resume point: stage 7 (`optimized-full-model`); stages 1 through 6 are retained.

For later blockers, record the UTC date, symptom, root cause, repair, validation,
and resume point here before continuing the pipeline.

## 2026-09-12: resume prompt selection

- Symptom: the no-model dry-run paired stage 7 with `01-functional-decoder`
  and numbered the final prompt as stage 17.
- Cause: `--start-index 7` controls numbering; it does not remove the first six
  prompt files from an all-prompts glob.
- Repair: supply only prompt files 07 through 11 and retain `--start-index 7`.
- Validation: require the corrected dry-run manifest to map stages 7 through 11
  to `optimized-full-model` through `tti-release` before launching Astra.
- Resume point: stage 7.
