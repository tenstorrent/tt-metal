# device_triage

Given a Blackhole Galaxy host, say whether it is healthy and what is most likely wrong with it, from
Machine HC (`/data/ci-provisioner/workflows/hc_logs`) and Cluster HC (`/data/dcamp`) logs, with the
exact log lines and the DIIM / DCAMP tickets that name it.

Everything is read live: nothing to set up, nothing stored. Stdlib Python 3.10, no installs. Run it on a
host that mounts `/data`.

```bash
cd tools/device_triage
python3 -m device_triage triage bh-glx-120-b02u08                      # ~10-30 s per host
python3 -m device_triage triage bh-glx-120-b02u08 -v                   # + file:line for each piece of evidence
python3 -m device_triage triage bh-glx-110-a10u14 --as-of 2026-10-04T15:00 --days 14
python3 -m device_triage triage bh-glx-120-b10u02 bh-glx-120-b10u08    # several hosts
python3 -m device_triage triage bh-glx-120-b10u02 --json
```

`triage` prints the host's status (FAILING / FLAKY / DEGRADED / HEALTHY / UNKNOWN) and why, its machine HC
and cluster HC history, the likely issue with the affected chips (`u<tray>c<asic>`), the strongest evidence
per category with first/last seen and run counts, and the tickets about the host.

## What it reads

| Source | Path | Unit |
| --- | --- | --- |
| Machine HC | `/data/ci-provisioner/workflows/hc_logs/<host>/<date>/<job>/` | `diag_report.json` + `logs/*.log` + `triage_host_side.txt` |
| BMC reboot collection | `/data/dcamp/bmc-reboot-data-collection/<host>/` | same layout as machine HC |
| Cluster HC records | `/data/dcamp/cluster-health-check/reports{,-archive}/<kind>/<date>/` | one JSON per recover / physical / fabric run |
| Cluster HC logs | `/data/dcamp/cluster-health-check/logs/recover/`, `/data/stackstorm/*/logs/`, validation logs | `[host][HH:MM:SS]`-prefixed logs |
| Jira (optional) | DIIM and DCAMP, JQL over the REST API | tickets naming the host |

Machine HC runs are found by the host's directory; cluster HC files by the host name in them (`rg`, else
`grep`), limited to directories that can hold runs from the window. Details: `docs/data_sources.md`.

## How it decides

- Every diag_report check maps to a category (`signatures.CHECK_RULES`); every log line is matched
  against ordered regex signatures (`signatures.LINE_SIGNATURES`). Failed tests are re-labelled by the
  strongest error in their log, so a GDDR test that dies on `Query mappings failed` counts as PCIe.
- Chips are `u<tray>c<asic>`; BDFs and driver ordinals are resolved per host from its triage tables.
- Score per category = Σ over runs of the strongest event: severity × source × rarity × recency.
  Rarity down-weights signals most of the fleet shows (`device_triage/signal_rarity.json`, a fleet snapshot);
  recency halves every 7 days.
- Status: FAILING (last machine HC failed, or recent single-host cluster failure), FLAKY (last run
  passed but ≥2 failures in 7 days), DEGRADED (device-level evidence), HEALTHY, UNKNOWN.
- PCIe and reset count as one issue when matching tickets: a chip that drops off the bus and the reset that
  can't bring it back are the same incident.

It looks at one host, so a failure that hit many hosts in the same minutes (neighbours resetting their ETH
links during a fleet-wide HC run) still counts against this one.

## Jira

Optional. Put an Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens))
in `~/.config/device_triage/jira.env`, mode 0600:

```
JIRA_EMAIL=you@tenstorrent.com
JIRA_TOKEN=...
```

Each `triage` then searches DIIM and DCAMP for the host (GET only), and runs the category queries in
`device_triage/jql_categories.json` against those tickets. A ticket is "about" the hosts its title names; if
the title names none, the hosts its description names (at most four). Hosts mentioned only in comments, or
in longer lists, are listed as "also mentioned in". HC bot auto-tickets (`[Fabric Health Check] ...`) are
counted separately.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
