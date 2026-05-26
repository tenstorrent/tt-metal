# SDPA Pipeline Watcher — Setup & Architecture

Historical setup plan, design decisions, and rebuild-from-scratch instructions.
For daily usage, see [`README.md`](README.md).

---

## Status as of last update (2026-05-26)

- All initial setup phases complete: Slack app approved, webhook saved, 7 SDPA-relevant pipelines configured, dry-runs validated, live posts verified, cron installed.
- Cron currently runs every 3 minutes (test cadence). Switch to a slower production cadence (e.g. `0 */4 * * *`) once you've watched a few ticks and confirmed everything looks right.
- **Outstanding:** rotate the Anthropic API key and GitHub PAT — both were exposed in the original Claude Code conversation transcript during setup. See "Security cleanup" below.

---

## Architecture

### How the caching works

For each watched pipeline, on every cron tick:

1. Fetch the latest **completed** run on `main` via `gh api`.
2. If `run_id` matches the cached entry in `state.json` → **reuse cached summary verbatim**. No LLM call, no log fetch.
3. If `run_id` differs → fetch failed-job logs, fetch commit range vs prior SHA, invoke `claude -p` with the agent prompt, replace the cached summary.

In-progress / queued runs are intentionally ignored — the digest always reflects only the latest *completed* run per pipeline.

LLM cost is roughly `(new completed runs / day) × (LLM call cost)`, not `ticks × pipelines × LLM call cost`. Idle pipelines contribute zero LLM cost regardless of cron cadence.

### Agent policy (global, applies to every pipeline)

Lives in `agent_prompt.txt`. Key rules:

- **SDPA-only scope.** The per-pipeline hint defines the in-scope subset (typically a pytest path). Anything not SDPA-related is completely ignored — no name, no count, no mention.
- **Broad in-scope.** Not just pytest FAILED lines — also compile/link errors on SDPA files, kernel build failures for SDPA kernels (`scaled_dot_product`, `ring_sdpa`, `mla`, etc.), runtime hangs/crashes during SDPA tests, and ring-attention / MLA failures.
- **Emoji keyed on in-scope subset, not workflow conclusion.** A workflow that fails for unrelated reasons but has no SDPA failures shows as ✅. Workflow red but SDPA passed = the watcher doesn't care.
- **List every in-scope failure.** No "and N more" truncation. Each gets a label, a one-line cause from the log, and the block ends with a single likely-cause line.

### Log fetching

`gh run view --log-failed` silently returns 0 bytes for some repos / runs (confirmed for `tenstorrent/tt-metal`). The watcher instead iterates failed jobs via the API:

```
repos/$REPO/actions/runs/$run_id/jobs   →  list failed job IDs
repos/$REPO/actions/jobs/<id>/logs       →  raw log per job (tail 12k chars, concat)
```

Each failed job gets a `=== JOB: <name> ===` header line so the agent can attribute failures correctly across jobs.

### Slack digest format

Block Kit payload (not plain mrkdwn):

- `header` block — slightly larger title, includes Belgrade timestamp via POSIX TZ string.
- One `section` mrkdwn block for the collapsed `✅ <names>` success line (if any).
- For each failing pipeline: `divider` + `section` block with the full per-pipeline summary.

This keeps the digest compact when most pipelines are green and noisy only when there's something real to look at.

---

## Divergences from the initial prototype

The shipped scripts differ from the original setup plan in three meaningful ways. If anyone rebuilds this watcher from notes or memory, apply these:

1. **Log fetcher rewrite.** Don't use `gh run view --log-failed`. Use the per-job API endpoint (see "Log fetching" above).
2. **Slack payload is Block Kit, not plain text.** The header + dividers + collapsed-success layout requires a structured `blocks` array. The original prototype concatenated mrkdwn into a single text field.
3. **Agent prompt rewrote the scope policy.** The original asked the agent to *count* out-of-scope failures and emit `(N other failures not in scope)`. The current policy ignores them entirely and keys the emoji on the in-scope subset.

---

## Original phase plan (now complete)

Recorded for reference. Phases 1–5 were the setup; Phase 6 (cron) is done; Phase 7 (verify auto-tick) was observed live during this session.

1. **Slack app install + webhook saved** — `~/.sdpa-watch/slack_webhook` (perms 600).
2. **Real-Slack smoke test** — `hello world` posted manually, confirmed visible.
3. **Configure pipelines** — 7 entries in `config.sh`, minimal per-line hints, global policy in `agent_prompt.txt`.
4. **Dry-run validation** — `DRY_RUN=1 ~/.sdpa-watch/watch.sh` against current state + a replay against run #847 (Blackhole E2E with real SDPA failures) to validate the failure-path formatting.
5. **Real post** — confirmed Slack rendering of both all-green ("✅ <names>" one-line) and mixed state (header + collapsed greens + per-failure block).
6. **Cron installed** — `*/3 * * * *` for test cadence. Daemon started via `sudo service cron start`.
7. **Verify first auto-tick** — observed live in `watch.log` and Slack.
8. **Security cleanup (PENDING)** — see below.

---

## Security cleanup (still pending)

Both secrets used during setup were pasted into the Claude Code conversation. Rotate them.

### Anthropic API key

1. https://console.anthropic.com/settings/keys → revoke the old key → create a new one → copy.
2. Replace the file:
   ```bash
   umask 077 && cat > ~/.sdpa-watch/api_key <<'EOF'
   sk-ant-NEW-KEY-HERE
   EOF
   chmod 600 ~/.sdpa-watch/api_key
   ```
3. Verify: `~/.sdpa-watch/watch.sh` runs cleanly (agent block produced for any red pipeline).

### GitHub PAT

1. https://github.com/settings/tokens → delete old → generate new (classic), scope `public_repo`, "No expiration" → copy.
2. Re-auth `gh`:
   ```bash
   echo 'ghp_NEW_TOKEN' | gh auth login --with-token
   gh auth status
   ```
3. Verify: `~/.sdpa-watch/watch.sh` runs cleanly (no `HTTP 401` in `watch.log`).

---

## Cost reference

At a reasonable production cadence (5–7 pipelines, every 4h, ~10–20 new completed runs per day across the set):

| Model | $/M input | $/M output | Est. monthly cost |
|---|---|---|---|
| `claude-opus-4-7` (current) | $15 | $75 | **~$20–40** |
| `claude-sonnet-4-6` | $3 | $15 | ~$4–8 |
| `claude-haiku-4-5-20251001` | $1 | $5 | ~$1–3 |

At the current test cadence (every 3 minutes), cost is still bounded by *new* completed runs, not ticks — because of the cache. Cron cadence affects only how often the cron daemon evaluates "is there a new run?", not how often the LLM is called.

---

## Rebuild from scratch

If `~/.sdpa-watch/` is wiped or you're setting this up on a new machine:

### Prerequisites on the machine

- `bash`, `jq`, `curl`, `git` (standard Linux)
- `gh` (GitHub CLI) — `apt install gh` or download from cli.github.com
- `claude` (Claude Code CLI) — `npm install -g @anthropic-ai/claude-code`
- `cron` (and a way to start the daemon — `sudo service cron start` works on most distros)
- A local clone of `tenstorrent/tt-metal`

### What you need in hand

- **Anthropic API key** (`sk-ant-...`) — https://console.anthropic.com/settings/keys
- **GitHub PAT** (`ghp_...`) — https://github.com/settings/tokens, scope `public_repo`, no expiration
- **Slack incoming webhook URL** — either reuse the existing `sdpa-watch` app at https://api.slack.com/apps (Incoming Webhooks → copy URL), or create a new Slack app and request workspace-admin install

### Steps

1. **Create directory and save secrets:**
   ```bash
   mkdir -p ~/.sdpa-watch && chmod 700 ~/.sdpa-watch
   umask 077 && printf 'sk-ant-...'  > ~/.sdpa-watch/api_key && chmod 600 ~/.sdpa-watch/api_key
   umask 077 && printf 'https://hooks.slack.com/...' > ~/.sdpa-watch/slack_webhook && chmod 600 ~/.sdpa-watch/slack_webhook
   echo 'ghp_...' | gh auth login --with-token
   echo '{}' > ~/.sdpa-watch/state.json
   ```

2. **Copy the four source files** from the mirror at `/localdev/skrstic/tt-metal/.sdpa-watch/` (or wherever your last working copy lives):
   - `config.sh` — pipelines + model
   - `watch.sh` — driver (`chmod +x`)
   - `agent_prompt.txt` — global agent policy
   - `README.md` + `SETUP.md` — docs

3. **Verify `claude` works in a cron-mimicking env:**
   ```bash
   env -i HOME="$HOME" \
     ANTHROPIC_API_KEY="$(cat ~/.sdpa-watch/api_key)" \
     PATH=/home/skrstic/.nvm/versions/node/v22.18.0/bin:/usr/bin:/usr/local/bin \
     "$(which claude)" --model claude-opus-4-7 -p "say hi" </dev/null
   ```

4. **Smoke test:**
   ```bash
   DRY_RUN=1 ~/.sdpa-watch/watch.sh
   ~/.sdpa-watch/watch.sh
   ```

5. **Install cron:**
   ```bash
   ( crontab -l 2>/dev/null | grep -v sdpa-watch; cat <<'CRON_EOF'
   PATH=/home/skrstic/.nvm/versions/node/v22.18.0/bin:/usr/local/bin:/usr/bin:/bin
   0 */4 * * * /home/skrstic/.sdpa-watch/watch.sh >> /home/skrstic/.sdpa-watch/watch.log 2>&1
   CRON_EOF
   ) | crontab -
   sudo service cron start
   ```

6. **Add shell aliases** in `~/.bashrc`:
   ```bash
   alias sdpa='~/.sdpa-watch/watch.sh'
   alias sdpa-dry='DRY_RUN=1 ~/.sdpa-watch/watch.sh'
   ```

Adjust the `PATH` in the crontab if `claude` / `node` live elsewhere (`which claude` and `dirname "$(which claude)"`).
