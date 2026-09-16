# Kernel Lib pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
# Supply this in the host-local runtime config or environment. Keep the repo
# snapshot free of usernames and machine-specific checkout paths.
TT_METAL_DIR="${TT_METAL_DIR:-}"
DISPLAY_TZ="${DISPLAY_TZ:-UTC}"
SLACK_WEBHOOK_FILE="$HOME/.kernel-lib-watch/slack_webhook"

# ---- Slack bot API (edit-in-place digest) ----------------------------------
# When a bot token (xoxb-, scopes: chat:write) and channel ID are present the
# watcher posts via chat.postMessage ONCE and then chat.update's that same
# message while the status stays the same (tick times accumulate on a
# "checked:" line); a status CHANGE posts a fresh message. The message ts
# lives in state.json under "_slack". If token or channel ID is missing the
# watcher falls back to the legacy webhook (new message per tick). The bot
# user must be a MEMBER of the channel (private channels especially):
# /invite @sdpawatch.
SLACK_BOT_TOKEN_FILE="$HOME/.kernel-lib-watch/slack_bot_token"
# UNSET ON PURPOSE — fill in before the first real (non-DRY_RUN) tick.
# Get it from the target channel: Slack → channel name → About → the C0…
# ID at the bottom. Leaving it empty keeps the watcher in webhook mode if
# $SLACK_WEBHOOK_FILE exists, and makes a real run exit 1 if it doesn't;
# either way it can never post into the sdpa/conv channels by accident.
SLACK_CHANNEL_ID=""

# ---- Auth ----------------------------------------------------------------
# The org retired console API keys (2026-07), so there is no api_key file.
# watch.sh authenticates the headless agent one of two ways, in this order:
#
#   MODE A (optional, most robust): a LONG-LIVED token from `claude setup-token`
#   saved to $OAUTH_TOKEN_FILE. Exported as CLAUDE_CODE_OAUTH_TOKEN; never needs
#   refreshing. Drop one in if you ever want to stop relying on the credential
#   below. Leave the file absent to use MODE B.
#
#   MODE B (default, zero-manual): the interactive OAuth credential at
#   $CLAUDE_CREDS_FILE, seeded once by logging into `claude`. Its access token
#   is short-lived (~8h) and — critically — headless `claude -p` does NOT
#   refresh it (that's what made every cron tick fail rc=1 from 2026-07-02).
#   So watch.sh refreshes it ITSELF before each tick via the OAuth endpoint,
#   using the rotating refresh token. An hourly cron then keeps the credential
#   alive indefinitely with no human in the loop.
OAUTH_TOKEN_FILE="$HOME/.kernel-lib-watch/oauth_token"          # MODE A (optional)
CLAUDE_CREDS_FILE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json"  # MODE B
# MODE B refresh parameters (Claude Code's public OAuth client). Update these
# only if Anthropic changes the endpoint/client_id.
OAUTH_TOKEN_ENDPOINT="https://platform.claude.com/v1/oauth/token"
OAUTH_CLIENT_ID="9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_REFRESH_MARGIN_SEC=1800   # refresh when <30 min of validity remain

# Model for the LLM agent. Heavier = better diagnosis, more $$$.
# Inherited from the sdpa/conv watchers for parity; safe to bump.
MODEL="claude-opus-4-8"

# Pin the Claude Code binary so its self-updater can't rewrite claude.exe
# out from under a cron tick (caused intermittent rc=127 "command not
# found" → "(agent error)" blocks). Update deliberately instead:
#   npm i -g @anthropic-ai/claude-code
export DISABLE_AUTOUPDATER=1

# Cron starts with a minimal PATH (/usr/bin:/bin) that omits the nvm-installed
# `claude` CLI. Pre-reboot this happened to work only because `service cron
# start` had inherited an interactive shell's PATH; a cron daemon that comes up
# fresh after a reboot does not, so the preflight died with "claude: command
# not found" — which the FATAL handler then misreported as an expired token.
# Self-heal PATH here (config.sh is sourced before the preflight): if `claude`
# isn't already resolvable, splice in the newest nvm node bin dir that has it
# (matches nvm's LTS default and survives node-version bumps).
if ! command -v claude >/dev/null 2>&1; then
  _newest_claude=""
  for _c in "$HOME"/.nvm/versions/node/*/bin/claude; do
    [[ -x "$_c" ]] || continue
    if [[ -z "$_newest_claude" || "$_c" -nt "$_newest_claude" ]]; then
      _newest_claude="$_c"
    fi
  done
  [[ -n "$_newest_claude" ]] && { PATH="$(dirname "$_newest_claude"):$PATH"; export PATH; }
  unset _newest_claude _c
fi

# Pipelines to watch.
# Format per entry:
#   "state_key|workflow_filename.yml|Display Name|event|run_title_pattern|test focus hint|job_name_pattern"
# - state_key uniquely identifies one workflow configuration. State and the
#   stale-run guard are isolated by this key, so a newer LLK-assert run cannot
#   hide an older-created watcher run that completes later.
# - workflow_filename is the .yml/.yaml file under .github/workflows in REPO
# - event is the GitHub trigger event to select (push, schedule, ...)
# - run_title_pattern is an extended regex matched against display_title
# - test focus hint is free text injected into the agent prompt; tell it
#   which failures count as in-scope (e.g. only kernel_lib tests).
# - job_name_pattern is an extended-regex (grep -E -i) applied to each
#   failed job's `.name` field BEFORE log fetch. Only matching jobs'
#   logs are sent to the agent — keeps the prompt focused on in-scope
#   failures and prevents context overflow on noisy nightly runs.
#   It is the LAST field, so `|` inside the regex is fine (bash `read`
#   hands the whole remainder of the line to it).
#   Leave empty to fetch logs from every failed job (match-all).
# Edit this list. Restart not required — next cron tick picks up changes.
#
# Every entry matches both the current "kernel lib tests" spelling and the old
# "ttnn llk helper library tests" spelling so historical runs remain dry-runnable.
# Job names carry the SKU, e.g. "ttnn-sanity-tests / kernel lib tests
# [wh_n300_civ2]" (prepare_test_matrix.py appends "[<sku>]" unconditionally).

PIPELINES=(
  "sanity-push|sanity-tests.yaml|Sanity WH push|push|^Sanity tests \\(push\\) SKUs\\[WH,Sim\\]$|In-scope = the ttnn kernel library test group from the push-to-main sanity configuration: pytest tests/ttnn/unit_tests/kernel_lib on wh_n300_civ2 (Wormhole N300). The suite runs all functional cases and deselects models_device_performance_bare_metal. Sim is present elsewhere in the push pipeline but this group has no sim SKU. All other Sanity jobs are out of scope, including tt-llk smoke jobs.|(ttnn llk helper library|kernel lib) tests"
  "sanity-scheduled-bh|sanity-tests.yaml|Sanity BH scheduled|schedule|^Sanity tests \\(scheduled\\) SKUs\\[BH\\]$|In-scope = the ttnn kernel library test group from the two-hour scheduled sanity configuration: pytest tests/ttnn/unit_tests/kernel_lib on bh_p150b_civ2 (Blackhole P150b). The suite runs all functional cases and deselects models_device_performance_bare_metal. All other Sanity jobs are out of scope, including tt-llk smoke jobs.|(ttnn llk helper library|kernel lib) tests"
  "debug-plain|sanity-tests-debug.yaml|Debug Sanity plain|schedule|^Sanity tests nightly debug run$|In-scope = the ttnn kernel library group (pytest tests/ttnn/unit_tests/kernel_lib) in the 00:00 UTC plain RelWithDebInfo sanity run. It runs Ubuntu 22.04 and 24.04 on wh_n300_civ2 and bh_p150b_civ2, for four in-scope jobs. ALWAYS name the SKU and Ubuntu version for failures. Everything else in the run is out of scope.|(ttnn llk helper library|kernel lib) tests"
  "debug-watcher|sanity-tests-debug.yaml|Debug Sanity with watcher|schedule|^Sanity tests nightly debug run with watcher$|In-scope = the ttnn kernel library group (pytest tests/ttnn/unit_tests/kernel_lib) in the 01:00 UTC RelWithDebInfo sanity run with watcher enabled. It runs Ubuntu 22.04 and 24.04 on wh_n300_civ2 and bh_p150b_civ2, for four in-scope jobs. ALWAYS name the SKU, watcher configuration, and Ubuntu version for failures. Everything else in the run is out of scope.|(ttnn llk helper library|kernel lib) tests"
  "debug-llk-asserts|sanity-tests-debug.yaml|Debug Sanity with LLK asserts|schedule|^Sanity tests nightly debug run with LLK asserts$|In-scope = the ttnn kernel library group (pytest tests/ttnn/unit_tests/kernel_lib) in the 02:00 UTC RelWithDebInfo sanity run with LLK asserts enabled. It runs Ubuntu 22.04 and 24.04 on wh_n300_civ2 and bh_p150b_civ2, for four in-scope jobs. ALWAYS name the SKU, LLK-assert configuration, and Ubuntu version for failures. An LLK assert is a real kernel-invariant violation. Everything else in the run is out of scope.|(ttnn llk helper library|kernel lib) tests"
)
