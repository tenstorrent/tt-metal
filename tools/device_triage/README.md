# device_triage

Given a Blackhole Galaxy host, say whether it is healthy and what is most likely wrong with it, from
Machine HC (`/data/ci-provisioner/workflows/hc_logs`) and Cluster HC (`/data/dcamp`) logs, with the
exact log lines and the DIIM tickets that name it.

Stdlib Python 3.10, no installs. Run everything from this directory.

## Quick start

```bash
cd tools/device_triage
python3 -m device_triage index          # incremental; first run ~10 min (niced, idle I/O), then ~1 min
python3 -m device_triage jira-sync      # optional: tickets, needs a Jira token (below)
python3 -m device_triage triage bh-glx-120-b02u08
python3 -m device_triage triage bh-glx-110-a10u14 --as-of 2026-10-04T15:00 --days 14 -v
python3 -m device_triage fleet          # every host with recent machine HC runs -> docs/fleet.md
python3 -m device_triage categories     # per DIIM category: tickets vs log evidence -> docs/categories.md
python3 -m device_triage validate       # replay at ticket creation -> docs/validation.md
python3 -m device_triage publish --to /data/device_triage    # read-only snapshot + reports for everyone
DEVICE_TRIAGE_CACHE=/data/device_triage python3 -m device_triage triage bh-glx-120-b02u08  # as a reader
```

Shared setup (one scheduled writer, read-only readers): `docs/deployment.md`.

`triage` prints the host's status (FAILING / FLAKY / DEGRADED / HEALTHY / UNKNOWN) and why, its machine HC
and cluster HC history, the likely issue with the affected chips (`u<tray>c<asic>`), the strongest evidence
per category with first/last seen and run counts (`-v` adds file:line), and the tickets about the host.

## What it reads

| Source | Path | Unit |
| --- | --- | --- |
| Machine HC | `/data/ci-provisioner/workflows/hc_logs/<host>/<date>/<job>/` | `diag_report.json` + `logs/*.log` + `triage_host_side.txt` |
| BMC reboot collection | `/data/dcamp/bmc-reboot-data-collection/` | same layout as machine HC |
| Cluster HC records | `/data/dcamp/cluster-health-check/reports{,-archive}/` | one JSON per recover / physical / fabric run |
| Cluster HC logs | `/data/dcamp/cluster-health-check/logs/recover/`, `/data/stackstorm/*/logs/`, validation logs | `[host][HH:MM:SS]`-prefixed logs |
| Jira | `cache/jira/` | DIIM listing, JQL membership, attachments |

Details: `docs/data_sources.md`.

## How it decides

- Every diag_report check maps to a category (`signatures.CHECK_RULES`); every log line is matched
  against ordered regex signatures (`signatures.LINE_SIGNATURES`). Failed tests are re-labelled by the
  strongest error in their log, so a GDDR test that dies on `Query mappings failed` counts as PCIe.
- Chips are `u<tray>c<asic>`; BDFs and driver ordinals are resolved per host from its triage tables.
- Score per category = Σ over runs of the strongest event: severity × source × rarity × recency.
  Rarity down-weights signals most of the fleet shows; recency halves every 7 days. A machine HC failure
  that 3+ hosts hit within 5 minutes counts 4x less (neighbours resetting, driver events).
- Status: FAILING (last machine HC failed, or recent single-host cluster failure), FLAKY (last run
  passed but ≥2 failures in 7 days), DEGRADED (device-level evidence), HEALTHY, UNKNOWN.

## Jira

Put an Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens)) in
`~/.config/device_triage/jira.env`, mode 0600:

```
JIRA_EMAIL=you@tenstorrent.com
JIRA_TOKEN=...
```

```bash
python3 -m device_triage jira-sync      # DIIM + DCAMP issues, all comments, DIIM attachments; incremental
python3 -m device_triage jira-sync --full --project DIIM --max-mb 50
```

Only GET requests. Issues land in `cache/jira/issues/<KEY>.json` (description and comments as plain
text), attachments in `cache/jira/attachments/<KEY>/` (text and zip only; `manifest.tsv` lists all), and
the per-category JQL membership is refreshed. Tickets are reloaded into the index at the end (also by
`index`, or alone with `tickets`). Category queries live in `device_triage/jql_categories.json`.

A ticket is "about" the hosts its title names; if the title names none, the hosts its description names.
Hosts mentioned only in comments are shown by `triage` as `(comment)` but don't count as tracking.

## Handy queries

```bash
python3 -m device_triage stats
python3 -m device_triage sql "SELECT signal, COUNT(DISTINCT host) FROM events WHERE severity='fail' GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
python3 -m device_triage sql "SELECT key, signal, chips, detail FROM ticket_evidence WHERE severity='fail'"
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
