# SDPA pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
TT_METAL_DIR="/localdev/skrstic/tt-metal"
SLACK_WEBHOOK_FILE="$HOME/.sdpa-watch/slack_webhook"
API_KEY_FILE="$HOME/.sdpa-watch/api_key"

# Model for the LLM agent. Heavier = better diagnosis, more $$$.
MODEL="claude-opus-4-7"

# Pin the Claude Code binary so its self-updater can't rewrite claude.exe
# out from under a cron tick (caused intermittent rc=127 "command not
# found" → "(agent error)" blocks). Update deliberately instead:
#   npm i -g @anthropic-ai/claude-code
export DISABLE_AUTOUPDATER=1

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
  "tt-metal-l2-nightly.yaml|L2 Nightly|tests/ttnn/nightly/unit_tests/operations/sdpa|sdpa"
  "blackhole-e2e-tests.yaml|Blackhole E2E|tests/nightly/blackhole/sdpa|sdpa"
  "blackhole-post-commit.yaml|Blackhole Post-Commit|tests/ttnn/unit_tests/operations/sdpa (ttnn-unit-tests / ttnn sdpa group only — deepseek blitz / per-core allocation jobs under ops-unit-tests are out-of-scope even if their logs mention test_sdpa_tail)|ttnn sdpa group"
  "perf-device-models.yaml|Perf Device Models|tests/nightly/blackhole/sdpa (SDPA_PERF_CHECKS=1 gated perf checks)|"
  "t3000-e2e-tests.yaml|T3K E2E|t3k_ccl_tests only — in-scope failures are limited to tests/nightly/t3000/ccl/test_ring_joint_attention.py (ring-joint SDPA). Other CCL tests in that job are out-of-scope; DeepSeek MLA / prefill failures are not ours.|t3k_ccl_tests"
  "t3000-integration-tests.yaml|T3K Integration|Any failure in t3k_sd35_large_tests, t3k_flux1_tests, t3k_motif_tests, t3k_wan2.2_tests, t3k_mochi_tests (all exercise ring-joint SDPA indirectly via DiT attention)|t3k_sd35_large|t3k_flux1|t3k_motif|t3k_wan2|t3k_mochi"
)
