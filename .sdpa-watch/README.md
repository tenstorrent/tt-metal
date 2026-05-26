# SDPA Pipeline Watcher

Monitors selected `tenstorrent/tt-metal` GitHub Actions workflows on `main` for
SDPA-related failures and posts a digest to Slack on a cron schedule.

- **Green pipelines** are collapsed into one `✅` line (just names).
- **Red pipelines** get a full block: every failing in-scope test, a short cause for each, and a likely-cause commit if attributable.
- **Out-of-scope failures** (anything not SDPA-related) are completely ignored, no count, no mention.

Currently watches 7 pipelines (see `config.sh`). Cron currently fires every 3 minutes (test cadence — change when ready).

---

## Daily use

| Command | What it does |
|---|---|
| `sdpa` | Run a check now and post to Slack |
| `sdpa-dry` | Print the digest to terminal, don't post |
| `tail -f ~/.sdpa-watch/watch.log` | Watch cron ticks live |
| `crontab -l` | See the active schedule |

Aliases live in `~/.bashrc`. The cron job calls `watch.sh` directly.

---

## What to change

### Add / remove / edit a watched pipeline

Edit `~/.sdpa-watch/config.sh`. Each entry in `PIPELINES=()` is one line:

```
"workflow_filename.yaml|Display Name|in-scope hint"
```

- **workflow_filename** — file under `.github/workflows/` in `tenstorrent/tt-metal`
- **Display Name** — short label for the Slack header
- **in-scope hint** — describes what counts as in-scope. Usually a pytest path like `tests/ttnn/unit_tests/operations/sdpa`. For multi-constraint scope, plain English works (e.g. `job_a (any failure) + job_b (only test_foo.py)`).

Changes take effect on the next tick. No restart needed.

Tip: to immediately re-analyze with the new hint, drop that pipeline from the cache:
```bash
jq 'del(."new-workflow.yaml")' ~/.sdpa-watch/state.json > /tmp/s && mv /tmp/s ~/.sdpa-watch/state.json
```

### Change cron cadence

```bash
crontab -e
```

Edit the `*/3 * * * *` line:

| Cadence | Cron spec |
|---|---|
| Every 3 minutes (current — test) | `*/3 * * * *` |
| Every 15 minutes | `*/15 * * * *` |
| Every hour | `0 * * * *` |
| Every 4 hours | `0 */4 * * *` |
| Twice a day at 08:00 and 20:00 UTC | `0 8,20 * * *` |

Save and exit. The cron daemon picks it up immediately.

### Pause / resume / uninstall

```bash
# pause (comment out the cron line)
crontab -e   # prefix the schedule line with '#'

# resume (uncomment)
crontab -e

# uninstall entirely
crontab -l | grep -v sdpa-watch | crontab -
```

### Force a fresh analysis (next tick)

```bash
rm ~/.sdpa-watch/state.json
```

Drops all cached summaries. The next tick re-analyzes every pipeline (one LLM call per failing pipeline; greens are cheap).

### Switch the LLM model (save cost)

Edit `MODEL=` in `config.sh`:

| Model | Cost relative | Output quality |
|---|---|---|
| `claude-opus-4-7` (current) | 1× | best diagnoses |
| `claude-sonnet-4-6` | ~0.2× | very good |
| `claude-haiku-4-5-20251001` | ~0.05× | acceptable for triage |

### Time zone in the digest title

`watch.sh` uses a POSIX TZ string for Belgrade (`CET-1CEST,M3.5.0,M10.5.0/3`). DST handled automatically. Edit the `ts_human` line in `watch.sh` to switch zones.

### Tweak the agent's global policy

`agent_prompt.txt` defines the cross-pipeline rules (SDPA-only scope, ignore out-of-scope failures, list every in-scope failure, broad in-scope including compile / kernel / runtime, emoji keyed on in-scope subset). Edit this file to change behavior for **all** pipelines at once. After editing, drop the cache (`rm ~/.sdpa-watch/state.json`) so the new policy applies on next tick.

---

## Files

| Path | Purpose |
|---|---|
| `~/.sdpa-watch/config.sh` | Pipelines list + model. **Edit this most.** |
| `~/.sdpa-watch/watch.sh` | The driver. Runs once per tick. |
| `~/.sdpa-watch/agent_prompt.txt` | Global LLM policy. |
| `~/.sdpa-watch/state.json` | Per-pipeline cache. |
| `~/.sdpa-watch/api_key` | Anthropic API key (chmod 600). |
| `~/.sdpa-watch/slack_webhook` | Slack webhook URL (chmod 600). |
| `~/.sdpa-watch/watch.log` | Append-only tick log. |
| `~/.sdpa-watch/agent_errors.log` | Stderr from `claude -p` (only useful on agent errors). |
| `~/.sdpa-watch/SETUP.md` | Setup history, rebuild-from-scratch, design notes. |

A snapshot of the source files (no secrets, no cache) is committed under `/localdev/skrstic/tt-metal/.sdpa-watch/` on the `skrstic/sdpa-pipeline-watcher` branch. The runtime always reads from `~/.sdpa-watch/` — re-sync the branch manually if you want to update the snapshot.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No Slack message at expected time | cron daemon stopped | `service cron status`; `sudo service cron start` |
| Slack response not "ok" in log | Webhook revoked or rate-limited | Reinstall the app, replace `~/.sdpa-watch/slack_webhook` |
| Block says `(agent error — see <url>)` | `claude -p` failed | `cat ~/.sdpa-watch/agent_errors.log`; usually means API key invalid |
| `gh: HTTP 401` in `watch.log` | GitHub PAT expired | `gh auth login --with-token` with a fresh PAT (scope `public_repo`) |
| Same message every tick | Cache is doing its job; nothing changed upstream | Expected. Use `rm state.json` if you want to force fresh. |
| Digest is huge | Too many pipelines or too many failures listed | Trim `PIPELINES=()` or tighten an in-scope hint |
| Cron stops after a reboot | systemd is broken here; cron doesn't auto-start | `sudo service cron start` after reboot |

---

## Architecture & history

See [`SETUP.md`](SETUP.md) for: caching design, log-fetcher implementation, agent-policy decisions, original setup plan, divergences from the original prototype, full rebuild-from-scratch instructions, and cost reference.
