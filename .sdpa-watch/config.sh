# SDPA pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
TT_METAL_DIR="/localdev/skrstic/tt-metal"
SLACK_WEBHOOK_FILE="$HOME/.sdpa-watch/slack_webhook"

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
OAUTH_TOKEN_FILE="$HOME/.sdpa-watch/oauth_token"           # MODE A (optional)
CLAUDE_CREDS_FILE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json"  # MODE B
# MODE B refresh parameters (Claude Code's public OAuth client). Update these
# only if Anthropic changes the endpoint/client_id.
OAUTH_TOKEN_ENDPOINT="https://platform.claude.com/v1/oauth/token"
OAUTH_CLIENT_ID="9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_REFRESH_MARGIN_SEC=1800   # refresh when <30 min of validity remain

# Model for the LLM agent. Heavier = better diagnosis, more $$$.
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
# Format per entry: "workflow_filename.yml|Display Name|test focus hint|job_name_pattern"
# - workflow_filename is the .yml/.yaml file under .github/workflows in REPO
# - test focus hint is free text injected into the agent prompt; tell it
#   which failures count as in-scope (e.g. only SDPA tests).
# - job_name_pattern is an extended-regex (grep -E -i) applied to each
#   failed job's `.name` field BEFORE log fetch. Only matching jobs'
#   logs are sent to the agent — keeps the prompt focused on in-scope
#   failures and prevents context overflow on noisy nightly runs.
#   Leave empty to fetch logs from every failed job (match-all).
# Edit this list. Restart not required — next cron tick picks up changes.

PIPELINES=(
  "sanity-tests.yaml|Sanity|tests/ttnn/unit_tests/operations/sdpa|"
  "tt-metal-l2-nightly.yaml|L2 Nightly|In-scope = SDPA nightly tests (tests/ttnn/nightly/unit_tests/operations/sdpa, run by the 'ttnn nightly sdpa' jobs) PLUS two experimental ops run inside the 'ttnn nightly experimental' job: tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py and test_topk_large_indices.py (both experimental). Every OTHER experimental test sharing that job (deepseek_prefill, minimal_matmul, mla_wo, moe, etc.) is OUT of scope — ignore it even though its log lives in the same job.|sdpa|experimental"
  "blackhole-post-commit.yaml|Blackhole Post-Commit|tests/ttnn/unit_tests/operations/sdpa (ttnn-unit-tests / ttnn sdpa group only — deepseek blitz / per-core allocation jobs under ops-unit-tests are out-of-scope even if their logs mention test_sdpa_tail)|ttnn sdpa group"
  "perf-device-models.yaml|Perf Device Models|In-scope = two gated perf checks in the 'ops perf tests' step (blackhole only): the SDPA_PERF_CHECKS=1 check (tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_perf_check) and the INDEXER_SCORE_PERF_CHECKS=1 check (tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::test_indexer_score_perf_check). Other ops-perf-tests in that step (conv, etc.) are OUT of scope.|"
  "t3000-e2e-tests.yaml|T3K E2E|t3k_ccl_tests only — in-scope failures are limited to tests/nightly/t3000/ccl/test_ring_joint_attention.py (ring-joint SDPA). Other CCL tests in that job are out-of-scope; DeepSeek MLA / prefill failures are not ours.|t3k_ccl_tests"
  "t3000-integration-tests.yaml|T3K Integration|Any failure in t3k_sd35_large_tests, t3k_flux1_tests, t3k_motif_tests, t3k_wan2.2_tests, t3k_mochi_tests (all exercise ring-joint SDPA indirectly via DiT attention)|t3k_sd35_large|t3k_flux1|t3k_motif|t3k_wan2|t3k_mochi"
)
