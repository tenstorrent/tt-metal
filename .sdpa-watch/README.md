# SDPA Pipeline Watcher

Monitors selected `tenstorrent/tt-metal` GitHub Actions workflows on `main` for
SDPA-related failures and posts a digest to Slack on a cron schedule.

- **Green pipelines** are collapsed into one `✅` line (just names).
- **Red pipelines** get a full block: every failing in-scope test, a short cause for each, and a likely-cause commit if attributable.
- **Out-of-scope failures** (anything not SDPA-related) are completely ignored, no count, no mention.

Currently watches 7 pipelines (see `config.sh`). Cron currently fires every hour.

---

## Where things live (read this once)

The watcher has two locations, and the distinction matters:

- **Runtime** = `~/.sdpa-watch/`. The cron job, `watch.sh`, the cache, secrets, and logs all live here. **This is the source of truth.** If this directory is intact and cron is running, the service is working — regardless of anything else.
- **Repo snapshot** = `/localdev/skrstic/tt-metal/.sdpa-watch/` on branch `skrstic/sdpa-pipeline-watcher`. A point-in-time **copy** of the source files (no secrets, no cache) committed for review and portability. The runtime never reads from here.

You can delete the repo snapshot and the service keeps running. You cannot delete `~/.sdpa-watch/` without breaking it.

---

## First-time setup on a new machine

### What you need to provide (2 secrets + 1 login + 1 path)

| Input | Where to get it | Goes into |
|---|---|---|
| Long-lived Claude OAuth token | Run `claude setup-token` (requires an active Claude subscription; SSO in browser) | `~/.sdpa-watch/oauth_token` (chmod 600) |
| GitHub PAT (`ghp_...`, scope `public_repo`) | https://github.com/settings/tokens | `gh auth login --with-token` |
| Slack incoming webhook URL | https://api.slack.com/apps → existing `sdpa-watch` app → Incoming Webhooks (or create a new app + admin approval) | `~/.sdpa-watch/slack_webhook` (chmod 600) |
| Path to your tt-metal clone | `pwd` inside the clone | `TT_METAL_DIR=` in `~/.sdpa-watch/config.sh` |

### What you run manually

```bash
# 1. Install OS deps (once)
sudo apt install -y cron jq curl gh    # plus Node.js (nvm) for `claude` CLI
npm install -g @anthropic-ai/claude-code

# 2. Set up the runtime dir from the repo snapshot
mkdir -p ~/.sdpa-watch && chmod 700 ~/.sdpa-watch
cp /path/to/tt-metal/.sdpa-watch/{config.sh,watch.sh,agent_prompt.txt,README.md,SETUP.md} ~/.sdpa-watch/
chmod +x ~/.sdpa-watch/watch.sh
echo '{}' > ~/.sdpa-watch/state.json

# 3. Mint a long-lived Claude OAuth token + drop in the 2 secrets
umask 077
claude setup-token   # SSO in browser; prints an sk-ant-oat... token
printf '%s' 'sk-ant-oat-YOUR-TOKEN' > ~/.sdpa-watch/oauth_token   # long-lived, headless auth
printf 'https://hooks.slack.com/...' > ~/.sdpa-watch/slack_webhook && chmod 600 ~/.sdpa-watch/slack_webhook
echo 'ghp_YOUR-PAT' | gh auth login --with-token

# 4. Edit `TT_METAL_DIR=` in ~/.sdpa-watch/config.sh to point at your clone

# 5. Smoke test (no Slack post)
DRY_RUN=1 ~/.sdpa-watch/watch.sh

# 6. Real post
~/.sdpa-watch/watch.sh

# 7. Install cron (hourly; adjust schedule + PATH if `claude` lives elsewhere)
( crontab -l 2>/dev/null | grep -vE 'sdpa-watch|^PATH='; cat <<CRON_EOF
PATH=$(dirname "$(which claude)"):/usr/local/bin:/usr/bin:/bin
0 * * * * $HOME/.sdpa-watch/watch.sh >> $HOME/.sdpa-watch/watch.log 2>&1
CRON_EOF
) | crontab -
sudo service cron start

# 8. (Optional) shell aliases — append to ~/.bashrc
echo "alias sdpa='~/.sdpa-watch/watch.sh'"            >> ~/.bashrc
echo "alias sdpa-dry='DRY_RUN=1 ~/.sdpa-watch/watch.sh'" >> ~/.bashrc
```

For host-specific tweaks (POSIX time zone string, copying from an existing machine via rsync, file-by-file porting reference), see [`SETUP.md`](SETUP.md) → "Port to another machine".

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
| `claude-opus-4-8` (current) | 1× | best diagnoses |
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
| `~/.sdpa-watch/oauth_token` | Long-lived Claude OAuth token from `claude setup-token` (chmod 600); exported as `CLAUDE_CODE_OAUTH_TOKEN`. |
| `~/.sdpa-watch/slack_webhook` | Slack webhook URL (chmod 600). |
| `~/.sdpa-watch/watch.log` | Append-only tick log. |
| `~/.sdpa-watch/agent_errors.log` | Stderr from `claude -p` (only useful on agent errors). |
| `~/.sdpa-watch/SETUP.md` | Setup history, rebuild-from-scratch, design notes. |

See "Where things live" at the top of this README for the runtime-vs-snapshot distinction.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No Slack message at expected time | cron daemon stopped | `service cron status`; `sudo service cron start` |
| Slack response not "ok" in log | Webhook revoked or rate-limited | Reinstall the app, replace `~/.sdpa-watch/slack_webhook` |
| `🟡 agent error` block, or `FATAL: OAuth token preflight failed` in `watch.log` | `claude -p` failed, usually auth | `cat ~/.sdpa-watch/agent_errors.log`. Most common: the long-lived token expired/revoked — re-mint with `claude setup-token` and overwrite `~/.sdpa-watch/oauth_token`. Test: `CLAUDE_CODE_OAUTH_TOKEN="$(cat ~/.sdpa-watch/oauth_token)" env -u ANTHROPIC_API_KEY claude -p <<<hi`. NOTE: do **not** rely on `~/.claude/.credentials.json` (interactive `/login`) for cron — that token is ~8h and headless `claude -p` never refreshes it, so ticks fail once it lapses. |
| `gh: HTTP 401` in `watch.log` | GitHub PAT expired | `gh auth login --with-token` with a fresh PAT (scope `public_repo`) |
| Same message every tick | Cache is doing its job; nothing changed upstream | Expected. Use `rm state.json` if you want to force fresh. |
| Digest is huge | Too many pipelines or too many failures listed | Trim `PIPELINES=()` or tighten an in-scope hint |
| Cron stops after a reboot | systemd is broken here; cron doesn't auto-start | `sudo service cron start` after reboot |

---

## Architecture & history

See [`SETUP.md`](SETUP.md) for: caching design, log-fetcher implementation, agent-policy decisions, original setup plan, divergences from the original prototype, full rebuild-from-scratch instructions, and cost reference.
