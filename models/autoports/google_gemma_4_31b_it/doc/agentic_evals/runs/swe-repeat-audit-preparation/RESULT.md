# Prepared retained-repeat audit

`audit_retained.py` is a read-only artifact reader. The coordinator runs it explicitly after the retained repeat finishes:

```bash
python models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/swe-repeat-audit-preparation/audit_retained.py --completed-run-root PATH_TO_COMPLETED_HARBOR_SWE_RUN --output PATH_TO_NEW_COMPACT_AUDIT_JSON --output-cap 32768
```

The input root must contain the suite `result.json` with `finished_at`, and trial subdirectories. Output uses exclusive creation and refuses overwrite. Confirm the trial names/count against the frozen5 cohort separately; the helper reports every available trial, never chooses a favorable subset. Raw aggregate score and Wilson95 remain explicitly ineligible until the coordinator completes validity/limit review.

Four focused tests passed: positive command candidate versus plain source URL, Wilson boundary, rejection of unfinished runs, and retained-ID/FormatError/cap/privacy handling. Existing completed nightly validation reproduced Astropy84APIcalls/81saved native responses and missing FormatError feedback indices102/139/142; xarray message2curl candidate had an adjacent remote HTTP indicator. Only this validation summary was saved; prior audit and transcripts were not modified.

The accepted live retention confirmation records104IDs at `messages[i].extra.response.choices[0].provider_specific_fields.token_ids`, matching completion usage, through actual LiteLLM1.108.1/mini2.2.8 serialization. The helper handles this location and raw vLLM choice.token_ids, reports missing or mismatched coverage, and retains only counts. Native malformed responses remain missing under the current repair; optional failed_response support is reported only if actually present.

Network command matches are candidates for private human review, not automatic quarantine decisions. URLs in source text are not by themselves retrieval. Embedded scripts, aliases and shell wrappers can evade lexical matching; complete private trajectory review remains necessary. A cap/length finish is a candidate, not proof of looping versus progress. The helper emits no response text, token IDs, tool commands, URLs, patches or credentials. Device use0h; no live endpoint, hardware, polling, publication or central-file changes.
