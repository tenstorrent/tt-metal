# CI Ticketing Helpers

This directory contains helpers for the Cursor-driven CI ticketing workflow.

## Extract Repeated Failures

Generate the candidate list used by the create-ticket rule:

```bash
python tools/ci/extract_failing_jobs.py
```

Optional workflow filter (yaml name without extension):

```bash
python tools/ci/extract_failing_jobs.py t3000-unit-tests
```

The output file is written to:

- `.auto_triage/output/ci_ticketing/create_tickets/failing_jobs.json`

## Notes

- Requires `gh` CLI authentication (`gh auth login`) or `GITHUB_TOKEN`.
- The script only prepares candidate failures. Issue creation/closure is intentionally done through guided Cursor rule execution with explicit `gh` commands.
