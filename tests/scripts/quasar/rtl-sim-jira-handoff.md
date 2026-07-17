# Task: forward failed quasar sim tests into the "RTL Sim CI test" GitHub check output

## Context

tt-metal now auto-files a Jira ticket on the `RELEASE` board when the
**`RTL Sim CI test`** GitHub check is non-green on `stable` HEAD *during the
Package-and-release workflow* (`.github/workflows/release-build-test-publish.yaml`,
job `check-rtl-sim-status`).

That job reads per-test failure detail from the check run's **`output.summary`**
and **`output.text`** (joined by newline) and pastes it into the ticket. Today
those fields are empty, so tickets only say "the sim check failed — see
`<GitLab URL>`." **Your job is to populate them with the exact failing tests.**

## What tt-metal already provides

`tests/scripts/quasar/run_quasar_regression.sh`, when run with `--log-dir <DIR>`,
writes a machine-readable manifest at **`<DIR>/<timestamp>/failed_tests.tsv`**.
It is always written when `--log-dir` is set (an empty file means all passed).
One tab-separated row per failed test:

```
config <TAB> group <TAB> filter <TAB> log-file
```

Example:

```
1x3	unit_tests_legacy	*DmLoopback*	/…/logs/…/1x3_unit_tests_legacy__DmLoopback_.log
2x3	unit_tests_api	*TensixSingleCoreDirectDramReaderDatacopyWriter	/…/…log
```

## Changes needed on the GitLab / polling-agent side

1. **Run the regression with `--log-dir`** (if not already) and capture
   `failed_tests.tsv` — upload it as a job artifact so the polling agent can read it.

2. **When the polling agent creates/updates the `RTL Sim CI test` check run**
   (GitHub Checks API `POST`/`PATCH /repos/tenstorrent/tt-metal/check-runs`) on the
   tested `stable` HEAD commit, set the `output` object from the manifest:
   - `output.title`: e.g. `"RTL sim: N test(s) failed"` (or `"RTL sim: all passed"`).
   - `output.summary`: a markdown bullet list, one line per manifest row, formatted
     as `` - `[<config>] <group> --gtest_filter=<filter>` ``. On success or an empty
     manifest, a short "all passed / no per-test detail" message.
   - `output.text` (optional): fuller detail / log links if useful.

3. **Keep the check name exactly `RTL Sim CI test`** — tt-metal filters on it
   verbatim (`check_name=RTL+Sim+CI+test`).

4. **Robustness:** if the manifest is missing (e.g. the run crashed before the
   summary), fall back to a summary noting that and pointing at the GitLab job URL.
   Cap `output.summary` well under GitHub's 65 535-char limit; if there are very many
   failures, list the first N and note the truncation.

**Forward *all* failed tests.** Don't pre-filter — tt-metal keeps its own
relevance mapping (`.github/actions/scripts/rtl_sim_jira_tests.json`) and files
one RELEASE ticket per *relevant* failed test, ignoring the rest. So the check
summary should list every failed test; tt-metal decides which warrant a ticket.

## Acceptance

After a failing quasar sim run on `stable` HEAD, the `RTL Sim CI test` check run on
that commit has `output.summary` listing the failed
`[config] group --gtest_filter=...` lines.

tt-metal's release job then files one `RELEASE` Jira ticket per relevant failed
test automatically — no further coordination needed.
