# Proposed HF/TT comparison identity guard

This is a proposal only. It does not modify the active `precision_repro_hf.py` or its tests, and it does not assert that this diagnostic gap caused a model anomaly.

## Verified gap

In `tools/precision_repro_hf.py`, the `--compare-dir` path currently reads `continuations.chosen.positions[:len(hf_rows)]` and checks only each row's `forced_token`. It then reports top-1 agreement and log-probability differences. It never checks TT conditioning IDs, case identity, or scored positions. Repeated target-token sequences can therefore compare across unrelated prompt prefixes or shifted windows.

The adjacent `precision_repro_probe.py::compare_reports` already rejects differing `conditioning_token_ids` and compares `(position, forced_token)` sequences for served/full TT reports. The HF comparator should apply the same conditioning/window discipline against the HF source case.

I audited 415 existing TT run JSONs under `setup/**/runs/` with `continuations.chosen.positions`: all 415 have a `case` path, `conditioning_token_ids`, `tokens`, and per-row `position`; none had top-level `case_sha256` or `window` in the sampled schema census. The patch therefore requires exact conditioning equivalence and equal compared row positions, but does not require a case hash or explicit window object. If an older report omits conditioning IDs, a readable saved TT case can establish the prompt identity. If neither source is available, comparison fails closed rather than silently producing a metric.

The current 2048-suffix Requests5414 case was independently checked: HF input IDs and TT `conditioning_token_ids` both equal the saved fixture, and the saved fixture equals the designated source suffix. The guard addresses future and legacy comparisons; it does not explain that model's branch behavior.

## Proposed patch and coverage

`precision-repro-hf-identity-guard.patch` changes only the comparison path and adds pure host regression tests. The helper checks:

- exact HF/TT `conditioning_token_ids`, with saved TT case-file prompt IDs as a fallback for reports lacking that field;
- TT case-file prompt IDs and task/call/response identity when the referenced file and metadata are present;
- top-level `case_sha256` equality when present;
- actual `(position, forced_token)` pairs for the compared rows, plus optional explicit window metadata when present;
- a comparison-evidence block in new HF output so the identity check is auditable.

Regression coverage in the patch rejects an unrelated prompt even when every forced caret token matches, rejects a mismatched case prompt and shifted repeated-token window, checks optional hash/window fields, and accepts the existing historical shape without hash/window metadata. It also covers using a readable saved case file when a legacy TT record omits `conditioning_token_ids`.

After the patch is reviewed/applied, the focused host-only check is:

```bash
python_env/bin/python -m pytest -q models/autoports/google_gemma_4_31b_it/tests/test_precision_repro_hf.py -k comparison_identity
```

That selection tests only the pure comparison helper; it does not instantiate a model. No test or inference was run while preparing this proposal. `git apply --check` succeeded against the current source tree.

## Source fingerprints

- Current `tools/precision_repro_hf.py`: `aafe67c88bb5a1eb38b8718570bbddd5749e2a9ecbb9dd19cdb100a49572c03e`
- Current `tests/test_precision_repro_hf.py`: `2bf9b446dd41cbabee394d7b4edee0bf9c26f8cb0e1de883d4eb02900a1735cf`
- Proposed patch: `909490f8bbac6549ce8e68109d0170b197aac32c1385705b2107dc333a26d786`
