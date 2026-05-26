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

## Port to another machine

Two scenarios:

- **A.** You have the old machine accessible — copy the running setup over.
- **B.** Old machine is gone, you have only this repo branch — rebuild from the snapshot.

Both end at the same place: a working `~/.sdpa-watch/` on the new host with cron firing on schedule.

### Prerequisites on the new machine

Install once:

- `bash`, `jq`, `curl`, `git` (standard Linux)
- `gh` (GitHub CLI) — `apt install gh` or grab from cli.github.com
- `claude` (Claude Code CLI) — `npm install -g @anthropic-ai/claude-code` (needs Node.js, typically via nvm)
- `cron` — `apt install cron`; start with `sudo service cron start`
- A local clone of `tenstorrent/tt-metal` (needed for commit-range lookups during failure diagnosis)

### Scenario A — copy from a working machine

The 7 files below are the entire watcher. Five are non-secret, two are secrets.

```bash
# On the NEW machine, create the dir
mkdir -p ~/.sdpa-watch && chmod 700 ~/.sdpa-watch

# From the OLD machine, transfer all runtime files (secrets included)
rsync -av --include='*.sh' --include='*.txt' --include='*.md' \
          --include='*.json' --include='api_key' --include='slack_webhook' \
          --exclude='*' \
          OLD-HOST:~/.sdpa-watch/  ~/.sdpa-watch/

# Re-tighten secret perms (rsync may not preserve)
chmod 600 ~/.sdpa-watch/api_key ~/.sdpa-watch/slack_webhook
chmod +x ~/.sdpa-watch/watch.sh
```

Then jump to "Adjust host-specific bits" below.

### Scenario B — rebuild from the repo snapshot (no old machine)

```bash
# Get the source files from the branch
mkdir -p ~/.sdpa-watch && chmod 700 ~/.sdpa-watch
git -C /path/to/tt-metal checkout skrstic/sdpa-pipeline-watcher
cp /path/to/tt-metal/.sdpa-watch/{config.sh,watch.sh,agent_prompt.txt,README.md,SETUP.md} ~/.sdpa-watch/
chmod +x ~/.sdpa-watch/watch.sh
echo '{}' > ~/.sdpa-watch/state.json

# Recreate secrets (you'll need fresh ones — the originals are not in the repo)
umask 077
printf 'sk-ant-YOUR-NEW-KEY' > ~/.sdpa-watch/api_key  && chmod 600 ~/.sdpa-watch/api_key
printf 'https://hooks.slack.com/services/...' > ~/.sdpa-watch/slack_webhook && chmod 600 ~/.sdpa-watch/slack_webhook
echo 'ghp_YOUR-NEW-PAT' | gh auth login --with-token
```

You need three secrets you don't get from the repo:

- **Anthropic API key** (`sk-ant-...`) — https://console.anthropic.com/settings/keys
- **GitHub PAT** (`ghp_...`) — https://github.com/settings/tokens, scope `public_repo`, "No expiration"
- **Slack incoming webhook URL** — reuse the existing `sdpa-watch` app at https://api.slack.com/apps (Incoming Webhooks → copy URL), or create a new Slack app + request workspace-admin install

Then jump to "Adjust host-specific bits" below.

### Adjust host-specific bits (applies to both scenarios)

Some values in the source files are tied to the *previous* host. Update them on the new host:

1. **`~/.sdpa-watch/config.sh` → `TT_METAL_DIR`** — point at the new clone's path.
2. **`~/.sdpa-watch/watch.sh` → `ts_human` POSIX TZ string** — update if you're not in CEST/CET (`CET-1CEST,M3.5.0,M10.5.0/3`). For a list of POSIX TZ strings, see e.g. https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html. Or install `tzdata` and use `TZ='Region/City'` instead.
3. **Cron PATH** — the `PATH=...` line in the crontab below must include the dir where `claude` lives on the new host. Find it with `dirname "$(which claude)"`.

### Verify and install cron

```bash
# 1. Sanity-check claude works in a cron-like minimal env
env -i HOME="$HOME" \
  ANTHROPIC_API_KEY="$(cat ~/.sdpa-watch/api_key)" \
  PATH="$(dirname "$(which claude)"):/usr/bin:/usr/local/bin" \
  "$(which claude)" --model claude-opus-4-7 -p "say hi" </dev/null
# Expect a one-line "Hi!" reply.

# 2. Dry-run the watcher (no Slack post)
DRY_RUN=1 ~/.sdpa-watch/watch.sh

# 3. Real run (posts to Slack)
~/.sdpa-watch/watch.sh

# 4. Install cron (replace PATH first line with `dirname "$(which claude)"`-prefixed value if needed)
( crontab -l 2>/dev/null | grep -vE 'sdpa-watch|^PATH='; cat <<CRON_EOF
PATH=$(dirname "$(which claude)"):/usr/local/bin:/usr/bin:/bin
0 * * * * $HOME/.sdpa-watch/watch.sh >> $HOME/.sdpa-watch/watch.log 2>&1
CRON_EOF
) | crontab -
sudo service cron start
crontab -l   # verify

# 5. (Optional) Shell aliases
cat >> ~/.bashrc <<'ALIASES'

# sdpa-watch shortcuts
alias sdpa='~/.sdpa-watch/watch.sh'
alias sdpa-dry='DRY_RUN=1 ~/.sdpa-watch/watch.sh'
ALIASES
```

### File-by-file purpose (porting reference)

| File | Required? | Notes |
|---|---|---|
| `~/.sdpa-watch/watch.sh` | **Required** | The driver. Must be executable. |
| `~/.sdpa-watch/config.sh` | **Required** | Pipelines + model + `TT_METAL_DIR`. Edit `TT_METAL_DIR` per host. |
| `~/.sdpa-watch/agent_prompt.txt` | **Required** | LLM policy. Edit only if changing behavior. |
| `~/.sdpa-watch/api_key` | **Required** | Anthropic key (chmod 600). Per-host or shared at your discretion. |
| `~/.sdpa-watch/slack_webhook` | **Required** | Slack URL (chmod 600). Reusable across hosts if you want all of them posting to one channel. |
| `~/.sdpa-watch/state.json` | Optional | Per-pipeline cache. Safe to omit — initialize with `echo '{}' > state.json` (first cron tick will be expensive though, since nothing is cached). |
| `~/.sdpa-watch/watch.log` | Skip | Local tick log; not portable. |
| `~/.sdpa-watch/agent_errors.log` | Skip | Stderr of failed agent calls; not portable. |
| `~/.sdpa-watch/README.md` | Optional | Docs. |
| `~/.sdpa-watch/SETUP.md` | Optional | This file. |

External state that lives outside `~/.sdpa-watch/`:

- `gh auth status` (token in `~/.config/gh/hosts.yml`) — required for log fetching.
- Crontab — must be installed per-host.
- `~/.bashrc` aliases — optional, per-host.
