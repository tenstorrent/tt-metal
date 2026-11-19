# Raw benchmark artifact commit audit

Source-only review, 2026-11-18 UTC. No hooks, tests, model imports, device operations, live-artifact edits, or plugin edits were performed.

**Recommendation:** after measurements finish, add one narrowly anchored exception to the existing `end-of-file-fixer` hook for the immutable attempt3 vLLM performance JSON files. Keep every hook enabled and all other files subject to the existing policy. This is the least invasive reproducible way to preserve exact bytes through an ordinary local commit; a broad JSON/directory exclusion is unnecessary.

No `AGENTS.md` exists in the checkout root or any ancestor of the target audit directory, including the model/documentation path. Therefore no applicable AGENTS instruction was found to quote. The checked-in `.pre-commit-config.yaml` already documents the directly analogous exception: **“Preserve the hashed, token-aligned book excerpt, including its missing final newline.”** Its `end-of-file-fixer` exclusion names that single fixture. This is precedent, not a requirement to exempt benchmark files. The installed Git pre-commit hook invokes this configuration normally.

The prior `run/raw-formatting-provenance.json` records three EOF-normalized artifacts and separately retained originals: “Repository end-of-file hook added exactly one final newline; JSON content unchanged.” That historical approach does not preserve the current measured file's hash. `tools/collect_roofline.py` hashes `path.read_bytes()` and records `performance_sha256`, so adding even one newline after collection breaks its binding. Do not normalize current raw bytes or rewrite collector hashes to conceal that mutation.

Suggested procedure for the parent:

1. Finish collection and stop all writers before staging evidence or running hooks. Inventory the actual completed raw JSON paths and SHA-256 values. Preserve historical artifacts and existing path references.
2. Extend only `end-of-file-fixer.exclude`, retaining its current alternatives, with exact completed raw basenames under `models/autoports/qwen_qwen3_6_35b_a3b/doc/benchmark/run/attempt3/`. A bounded expression for the intended set is `^models/autoports/qwen_qwen3_6_35b_a3b/doc/benchmark/run/attempt3/perf-b(1|8|16|32)(-warmup)?\.json$` plus `^models/autoports/qwen_qwen3_6_35b_a3b/doc/benchmark/run/attempt3/perf-b32-c1-(functional|long_context)(-warmup)?\.json$`. Prefer enumerating only files actually completed at finalization. Explain that upstream vLLM bytes are bound by collector hashes. Do not exclude server/configuration/roofline JSON or the whole benchmark subtree. This is a recommendation only; this audit did not edit configuration.
3. Stage the narrowly scoped configuration edit together with completed stage-owned evidence. Run the normal configured hooks on the staged change and retain their output. Leave trailing-whitespace, YAML validation, file-size checks, and all other hooks enabled. The exemption is visible and repeatable in later checkouts; no installed plugin change is needed.
4. Verify every exempt raw file still matches its pre-hook SHA-256 and the corresponding collector reference. Also hash each staged blob (`git show :<path>`) and compare it with the source bytes before committing; then verify the committed blob. Run repaired evidence/context checks after canonical promotion and hook completion. Record the resulting local commit SHA.

Risks and limits:

- An exclusion must match the physical tracked attempt3 paths. Canonical symlinks do not protect tracked backing files; avoid promoting copies that acquire different bytes. Do not move the live run while serving continues.
- `trailing-whitespace` can also mutate text. Current vLLM missing-newline output needs only the EOF exception; inspect exact bytes and fail verification if another hook mutates them. Add no speculative broad exemption.
- File-size limits remain applicable. Preserve oversized raw evidence by the already established retention/inventory policy; this EOF exception does not waive those checks.
- No Git attributes were reported for the inspected `attempt3/perf-b32.json`; staged/committed blob verification still protects against filters or future configuration changes.
- `SKIP=end-of-file-fixer`, `--no-verify`, alternate hooks paths/configurations, restoring index bytes after validation, or renaming JSON to `.raw` are not equivalent to this visible narrow exception. They either disable required coverage, evade verification, or do not reliably avoid text classification. If even an exact-file repository exception is disallowed, unchanged raw bytes and the existing EOF fixer cannot both pass for the same tracked file; report that constraint rather than claim both occurred.
