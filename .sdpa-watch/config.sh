# SDPA pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
TT_METAL_DIR="/localdev/skrstic/tt-metal"
SLACK_WEBHOOK_FILE="$HOME/.sdpa-watch/slack_webhook"
API_KEY_FILE="$HOME/.sdpa-watch/api_key"

# Model for the LLM agent. Heavier = better diagnosis, more $$$.
MODEL="claude-opus-4-7"

# Pipelines to watch.
# Format per entry: "workflow_filename.yml|Display Name|test focus hint"
# - workflow_filename is the .yml/.yaml file under .github/workflows in REPO
# - test focus hint is free text injected into the agent prompt; tell it
#   which failures count as in-scope (e.g. only SDPA tests).
# Edit this list. Restart not required — next cron tick picks up changes.

PIPELINES=(
  "sanity-tests.yaml|Sanity|tests/ttnn/unit_tests/operations/sdpa"
  "tt-metal-l2-nightly.yaml|L2 Nightly|tests/ttnn/nightly/unit_tests/operations/sdpa"
  "blackhole-e2e-tests.yaml|Blackhole E2E|tests/nightly/blackhole/sdpa"
  "blackhole-post-commit.yaml|Blackhole Post-Commit|tests/ttnn/unit_tests/operations/sdpa"
  "perf-device-models.yaml|Perf Device Models|tests/nightly/blackhole/sdpa (SDPA_PERF_CHECKS=1 gated perf checks)"
  "t3000-e2e-tests.yaml|T3K E2E|t3k_DeepSeek_PREFILL (any failure in this job) + t3k_ccl_tests (only failures in tests/nightly/t3000/ccl/test_ring_joint_attention.py; other CCL tests in that job are out-of-scope)"
  "t3000-integration-tests.yaml|T3K Integration|Any failure in t3k_sd35_large_tests, t3k_flux1_tests, t3k_motif_tests, t3k_wan2.2_tests, t3k_mochi_tests (all exercise ring-joint SDPA indirectly via DiT attention)"
)
