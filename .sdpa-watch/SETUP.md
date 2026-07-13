# SDPA Pipeline Watcher — Setup & Architecture

Historical setup plan, design decisions, and rebuild-from-scratch instructions.
For daily usage, see [`README.md`](README.md).

---

## Status as of last update (2026-05-26)

- All initial setup phases complete: Slack app approved, webhook saved, 7 SDPA-relevant pipelines configured, dry-runs validated, live posts verified, cron installed.
- Cron currently runs every 3 minutes (test cadence). Switch to a slower production cadence (e.g. `0 */4 * * *`) once you've watched a few ticks and confirmed everything looks right.
- **2026-07 update:** the org retired console API keys. The watcher first moved to the interactive Enterprise OAuth credential (`~/.claude/.credentials.json`), but that ~8h token is never refreshed by headless `claude -p`, so cron degraded to constant 🟡 "(agent error)" blocks from 2026-07-02. It now uses a **long-lived `claude setup-token`** in `~/.sdpa-watch/oauth_token` (exported as `CLAUDE_CODE_OAUTH_TOKEN`). See "Claude auth" under Security cleanup below.

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

### Claude auth (long-lived setup-token)

The org retired console API keys (2026-07). The watcher authenticates with a
**long-lived Claude Code OAuth token** minted via `claude setup-token`, stored
in `~/.sdpa-watch/oauth_token` and exported by `watch.sh` as
`CLAUDE_CODE_OAUTH_TOKEN`. `watch.sh` unsets `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN` first so a stale one can't override it.

**Why not the interactive `~/.claude/.credentials.json`:** that credential is a
short-lived (~8h) OAuth token that is ONLY refreshed by an interactive
`claude`/`/login`. Headless `claude -p` does **not** refresh it — so once more
than ~8h passed with no interactive session, every hourly cron tick failed with
rc=1 and degraded to a 🟡 "(agent error)" block (observed constantly from
2026-07-02 after the credentials.json-based auth was deployed). The setup-token
is long-lived and inference-only, which is exactly what an unattended cron needs.

1. `umask 077 && claude setup-token` — SSO in browser; prints an `sk-ant-oat...`
   token. Save it: `printf '%s' 'sk-ant-oat-...' > ~/.sdpa-watch/oauth_token`.
2. Nothing auto-rotates. Re-mint and overwrite the file if the token is ever
   revoked or expires (watch.sh's preflight will log a FATAL and skip the tick).
3. Verify: `CLAUDE_CODE_OAUTH_TOKEN="$(cat ~/.sdpa-watch/oauth_token)" \
   env -u ANTHROPIC_API_KEY claude --model "$MODEL" -p <<<"say hi"` returns a
   reply (rc=0), then `~/.sdpa-watch/watch.sh` runs cleanly.

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
| `claude-opus-4-8` (current) | $15 | $75 | **~$20–40** |
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

The files below are the entire watcher. Two secrets live under `.sdpa-watch/`:
`slack_webhook` and `oauth_token` (the long-lived `claude setup-token`). The
GitHub PAT lives in `gh`'s own store.

```bash
# On the NEW machine, create the dir
mkdir -p ~/.sdpa-watch && chmod 700 ~/.sdpa-watch

# From the OLD machine, transfer all runtime files (slack_webhook included)
rsync -av --include='*.sh' --include='*.txt' --include='*.md' \
          --include='*.json' --include='slack_webhook' \
          --exclude='*' \
          OLD-HOST:~/.sdpa-watch/  ~/.sdpa-watch/

# Re-tighten secret perms (rsync may not preserve)
chmod 600 ~/.sdpa-watch/slack_webhook
chmod +x ~/.sdpa-watch/watch.sh

# Mint a long-lived Claude token on the new host (tokens are host-agnostic, but
# minting is interactive — do it locally). rsync copies oauth_token too; re-mint
# only if you want to rotate.
umask 077 && claude setup-token   # prints sk-ant-oat...; save to ~/.sdpa-watch/oauth_token
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
printf 'https://hooks.slack.com/services/...' > ~/.sdpa-watch/slack_webhook && chmod 600 ~/.sdpa-watch/slack_webhook
echo 'ghp_YOUR-NEW-PAT' | gh auth login --with-token
claude setup-token   # SSO; save the sk-ant-oat... token:
printf '%s' 'sk-ant-oat-...' > ~/.sdpa-watch/oauth_token
```

You need these you don't get from the repo:

- **Long-lived Claude token** — `claude setup-token` (needs an active Claude subscription); save the `sk-ant-oat...` value to `~/.sdpa-watch/oauth_token`. Long-lived; re-mint only on revoke/expiry.
- **GitHub PAT** (`ghp_...`) — https://github.com/settings/tokens, scope `public_repo`, "No expiration"
- **Slack incoming webhook URL** — reuse the existing `sdpa-watch` app at https://api.slack.com/apps (Incoming Webhooks → copy URL), or create a new Slack app + request workspace-admin install

Then jump to "Adjust host-specific bits" below.

### Adjust host-specific bits (applies to both scenarios)

Some values in the source files are tied to the *previous* host. Update them on the new host:

1. **`~/.sdpa-watch/config.sh` → `TT_METAL_DIR`** — point at the new clone's path.
2. **`~/.sdpa-watch/watch.sh` → `ts_human` POSIX TZ string** — update if you're not in CEST/CET (`CET-1CEST,M3.5.0,M10.5.0/3`). For a list of POSIX TZ strings, see e.g. https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html. Or install `tzdata` and use `TZ='Region/City'` instead.
3. **Cron PATH** — no action needed. `config.sh` now self-heals `PATH`: if `claude` isn't already resolvable it splices in the newest nvm node bin dir that has it. (Historically the crontab carried a hardcoded `PATH=` line; that broke after a reboot because the daemon came up with a bare `/usr/bin:/bin` PATH and the plain crontab line — see README — omitted it. Resolving in `config.sh` is host- and node-version-agnostic.)

### Verify and install cron

```bash
# 1. Sanity-check claude works in a cron-like minimal env with the setup-token
env -i HOME="$HOME" \
  PATH="$(dirname "$(which claude)"):/usr/bin:/usr/local/bin" \
  CLAUDE_CODE_OAUTH_TOKEN="$(cat ~/.sdpa-watch/oauth_token)" \
  "$(which claude)" --model claude-opus-4-8 -p "say hi" </dev/null
# Expect a one-line "Hi!" reply. (ANTHROPIC_API_KEY is intentionally NOT set —
# a stale one would override the token and 401.)

# 2. Dry-run the watcher (no Slack post)
DRY_RUN=1 ~/.sdpa-watch/watch.sh

# 3. Real run (posts to Slack)
~/.sdpa-watch/watch.sh

# 4. Install cron. The crontab line is plain — no PATH= prefix — because
#    config.sh resolves `claude` itself (see "Cron PATH" above).
sudo apt install -y cron
( crontab -l 2>/dev/null | grep -vF '.sdpa-watch/watch.sh'; \
  echo '0 * * * * $HOME/.sdpa-watch/watch.sh >> $HOME/.sdpa-watch/watch.log 2>&1' ) | crontab -
sudo service cron start
crontab -l   # verify

# 5. Shell aliases + reboot-durability hook (see "Reboot durability" below)
cat >> ~/.bashrc <<'ALIASES'

# sdpa-watch shortcuts
alias sdpa='~/.sdpa-watch/watch.sh'
alias sdpa-dry='DRY_RUN=1 ~/.sdpa-watch/watch.sh'
# Reboot-durability: this container wipes the cron package on reboot (only $HOME
# persists) and has no systemd, so the login shell is the only boot hook. Fast
# no-op when the scheduler is healthy; self-repairs in the background otherwise.
[ -x "$HOME/.sdpa-watch/ensure-cron.sh" ] && "$HOME/.sdpa-watch/ensure-cron.sh" >/dev/null 2>&1
ALIASES
```

### Reboot durability

This host is a container: a reboot wipes installed OS packages (`cron` among
them) while `$HOME` and the clone under `/localdev` persist, and there is no
systemd — so nothing auto-starts at boot. The **login shell is the only hook
that always fires**, so `~/.bashrc` calls `~/.sdpa-watch/ensure-cron.sh`, which:

1. fast-path exits (two cheap checks) when the cron daemon is up **and** the
   watcher's crontab line is present — the normal case, adds ~nothing to shell
   startup;
2. otherwise self-repairs, **detached and `flock`-single-flighted** (so a burst
   of logins right after a reboot can't race on the dpkg lock): `apt install`s
   cron if the package is gone, `service cron start`s the daemon, and reinstalls
   the crontab line (dedup-safe). Needs passwordless `sudo` + network. Progress
   is logged to `~/.sdpa-watch/bootstrap.log`.

So after a reboot the watcher self-restores the first time you log in. To force
it without waiting for a login: `~/.sdpa-watch/ensure-cron.sh`. There is no
boot-time mechanism that avoids the login trigger on this host.

### File-by-file purpose (porting reference)

| File | Required? | Notes |
|---|---|---|
| `~/.sdpa-watch/watch.sh` | **Required** | The driver. Must be executable. |
| `~/.sdpa-watch/config.sh` | **Required** | Pipelines + model + `TT_METAL_DIR`. Edit `TT_METAL_DIR` per host. Also self-heals `PATH` so cron can find `claude`. |
| `~/.sdpa-watch/ensure-cron.sh` | **Required** | Reboot-durability hook (must be executable); called from `~/.bashrc`. See "Reboot durability". |
| `~/.sdpa-watch/agent_prompt.txt` | **Required** | LLM policy. Edit only if changing behavior. |
| `~/.sdpa-watch/oauth_token` | **Required** | Long-lived `claude setup-token` (chmod 600); exported as `CLAUDE_CODE_OAUTH_TOKEN`. Re-mint on revoke/expiry. |
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
