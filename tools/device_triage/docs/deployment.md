# Making device_triage a shared tool

Today it runs from one checkout with a personal Jira token. The target: anyone on an exabox host (or in Slack /
a dashboard later) can ask "what's wrong with bh-glx-X" without indexing anything or holding a token.

Nothing here is deployed. The scheduled job needs a k8s request (see "Ask").

## Shape: one writer, many readers

```
k8s CronJob (every 30 min, niced, 1 pod)            readers (anyone with /data mounted)
  jira-sync        incremental, GET only              DEVICE_TRIAGE_CACHE=/data/device_triage \
  index            incremental, ~1 min if quiet         python3 -m device_triage triage bh-glx-120-b02u08
  fleet / categories / validate -> docs/*.md
  publish --to /data/device_triage                 /data/device_triage/docs/fleet.md, categories.md, ...
      index.sqlite (read-only snapshot, atomic swap)
```

- **Writer state** (`DEVICE_TRIAGE_CACHE`, ~650 MB): index, Jira cache, attachments, rebuilt DCAMP runs. Lives
  on the writer's volume or a private dir on /data.
- **Published dir** (`publish --to`): `index.sqlite` snapshot (~100 MB, mode 0444, swapped atomically with
  `VACUUM INTO` + rename, so readers never see a half-written file) and the markdown reports.
- **Readers** open a database they can't write in read-only, immutable mode: no locks, no migrations, safe
  on NFS. Evidence paths point at `/data/ci-provisioner` and `/data/dcamp`, readable from any exabox host.

## Needs

| What | Why |
| --- | --- |
| Read access to `/data/ci-provisioner/workflows/hc_logs`, `/data/dcamp` | the logs (same mounts HC jobs use) |
| A Jira service account token (k8s Secret -> `jira.env`, 0600) | not a personal token; read-only scopes `read:jira-work`, `read:jira-user` |
| Write access to one shared dir, group-readable | the published snapshot and reports |
| Python 3.10, no packages | stdlib only |
| ~1 CPU, 1 GB RAM; first run ~15 min with 12 workers, then ~1-2 min | parsing is CPU-bound; NFS reads niced + idle I/O class |

## Code home

Stdlib-only package with unit tests (`python3 -m unittest discover -s tests -t .`), so it can live anywhere:
`tt-metal/tools/device_triage` or the exabox tooling repo next to the HC scripts (more natural: it parses their
output, and HC format changes should update its signatures in the same PR). To decide with the team.

## Later

- `triage --json` already exists: a Slack slash command or a small read-only web page can wrap it.
- Ticket hygiene report: failing hosts without a human ticket (fleet.md top table) as a daily digest.
- Fix upstream: the HC bot's base64/truncated `diag_report` uploads (see findings.md).

## Ask (for the k8s side)

One CronJob, schedule `*/30 * * * *`, concurrencyPolicy Forbid, image with Python 3.10, mounts: hc_logs and
dcamp read-only, one RW volume for writer state, the shared publish dir, the Jira Secret. Command:

```bash
python3 -m device_triage jira-sync && python3 -m device_triage index --workers 4 \
  && python3 -m device_triage fleet && python3 -m device_triage categories && python3 -m device_triage validate \
  && python3 -m device_triage publish --to /data/device_triage
```
