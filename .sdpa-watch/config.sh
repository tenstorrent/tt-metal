# SDPA pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
TT_METAL_DIR="/localdev/skrstic/tt-metal"
SLACK_WEBHOOK_FILE="$HOME/.sdpa-watch/slack_webhook"

# ---- Slack bot API (edit-in-place digest) ----------------------------------
# When a bot token (xoxb-, scopes: chat:write) and channel ID are present the
# watcher posts via chat.postMessage ONCE and then chat.update's that same
# message on every subsequent tick, instead of posting a new message hourly.
# The message ts lives in state.json under "_slack". If the token file is
# missing/empty the watcher falls back to the legacy webhook (new message per
# tick). The bot user must be a MEMBER of the channel (private channels
# especially): /invite @sdpawatch.
SLACK_BOT_TOKEN_FILE="$HOME/.sdpa-watch/slack_bot_token"
SLACK_CHANNEL_ID="C0B72MVR88G"   # #sdpa-watch (private)

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
#
# Same class of breakage for `gh`/`jq`: they used to be system packages in
# /usr/bin (which cron does see), but when they are only user-installed in
# ~/.local/bin, cron's bare PATH misses them and every tick silently degrades
# — `gh` failures are swallowed into "null" runs, so the digest goes blank
# rather than erroring. Prepend ~/.local/bin unconditionally so a user-local
# install of either tool is always visible.
if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  PATH="$HOME/.local/bin:$PATH"; export PATH
fi
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
  "sanity-tests.yaml|Sanity|In-scope = ALL sanity-pipeline SDPA coverage, on every SKU. Since PR #48943 (2026-08-13) the retired blackhole-sanity-tests.yaml was folded into this workflow, so Wormhole, Blackhole and ttsim SDPA now report here. Four job families: (a) 'ttnn sdpa group [sku]' — tests/ttnn/unit_tests/operations/sdpa whole-dir sweep on wh_n300_civ2, bh_p100, bh_p150b_civ2, sim_wormhole_b0, sim_blackhole; includes the sparse SDPA op tests test_sparse_sdpa.py and test_sparse_sdpa_msa.py, both in-scope. (b) 'ttnn indexer_score group [sku]' — the rank7 accuracy subset of tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py. (c) 'sdpa nightly tests (QB2 only) [bh_quietbox_2]' — the multi-card suite tests/nightly/blackhole/sdpa/ (ring_joint, exp_ring_joint, sparse_sdpa_multidevice, scaled_dot_product_attention_sprint). (d) 'TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' — a grouped job where ONLY the ring-joint/MLA perf checks (test_ring_joint_attention_perf_check, test_ring_mla_chunked_perf_check, test_exp_ring_joint_attention_perf_check) and indexer_score_qb are ours; the test_high_bw_all_gather CCL check sharing that job is NOT. ALWAYS name the SKU when reporting, and call it out when a failure hits only one arch. Out of scope: ttnn reduce group, conv and pool groups, ops-unit-tests, UMD, profiler and runtime sanity, and all build-artifact jobs.|ttnn sdpa group|ttnn indexer_score group|sdpa nightly tests|SDPA PERF"
  "sanity-tests-debug.yaml|Sanity Debug|In-scope = the SDPA groups in this nightly DEBUG run. Since PR #48943 this workflow just re-invokes sanity-tests.yaml under debug flags, so it carries the same job set: 'ttnn sdpa group' on Wormhole (wh_n300_civ2) and Blackhole (bh_p100, bh_p150b_civ2) — tests/ttnn/unit_tests/operations/sdpa, including test_sparse_sdpa.py and test_sparse_sdpa_msa.py — plus 'ttnn indexer_score group' and the QuietBox jobs 'sdpa nightly tests (QB2 only)' and the SDPA PERF checks inside 'TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' on bh_quietbox_2. IMPORTANT: this workflow has three nightly variants (plain 00:00 UTC, watcher-enabled 01:00, LLK-asserts-enabled 02:00) and each job name carries the mode plus the Ubuntu version, e.g. '(Ubuntu 24.04 with LLK asserts)'. ALWAYS state which debug mode the analyzed run used, and say so if only one of Ubuntu 22.04 / 24.04 failed: an SDPA failure that reproduces ONLY under LLK asserts or ONLY under watcher is a race / ordering / uninitialized-state signal rather than a plain functional regression, and is worth calling out as such. Out of scope: 'ttnn reduce group', blackhole deepseek per-core allocation tests, ttsim / runtime-sim sanity, and all build-artifact jobs.|sdpa|ttnn indexer_score group"
  "tt-metal-l2-nightly.yaml|L2 Nightly|In-scope = SDPA nightly tests (tests/ttnn/nightly/unit_tests/operations/sdpa, run by the 'ttnn nightly sdpa' jobs) PLUS two experimental ops run inside the 'ttnn nightly experimental' job: tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py and test_topk_large_indices.py (both experimental). Every OTHER experimental test sharing that job (deepseek_prefill, minimal_matmul, mla_wo, moe, etc.) is OUT of scope — ignore it even though its log lives in the same job. The nightly sdpa dir-sweep also runs the sparse SDPA op tests (test_sparse_sdpa.py and test_sparse_sdpa_msa.py, plus _perf and _block_cyclic_multidevice) — all in-scope.|sdpa|experimental"
  "perf-device-models.yaml|Perf Device Models|In-scope = the INDEXER_SCORE_PERF_CHECKS=1 gated check on Blackhole: tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py::test_indexer_score_math_util (renamed from test_indexer_score_perf_check). NOTE: the former SDPA_PERF_CHECKS gate and test_sdpa_perf_check were removed from this workflow. Other ops-perf-tests (conv, etc.) are OUT of scope.|P150 BH device perf"
  "t3000-e2e-tests.yaml|T3K E2E|t3k_ccl_tests only — in-scope failures are limited to tests/nightly/t3000/ccl/test_ring_joint_attention.py (ring-joint SDPA). Other CCL tests in that job are out-of-scope; DeepSeek MLA / prefill failures are not ours.|t3k_ccl_tests"
  "t3000-integration-tests.yaml|T3K Integration|Any failure in t3k_sd35_large_tests, t3k_flux1_tests, t3k_motif_tests, t3k_wan2.2_tests, t3k_mochi_tests, t3k_qwenimage_tests (all exercise ring-joint SDPA indirectly via DiT attention)|t3k_sd35_large|t3k_flux1|t3k_motif|t3k_wan2|t3k_mochi|t3k_qwenimage"
  "galaxy-e2e-tests.yaml|Galaxy E2E|In-scope = ring-joint SDPA in the 'Galaxy CCL tests' job: tests/nightly/tg/ccl/test_ring_joint_attention.py (the TG analogue of T3K E2E). All other CCL tests (all_gather, reduce_scatter, all_to_all, moe) are out of scope.|Galaxy CCL tests"
  "blaze-models-prefill-tests.yaml|Blaze Prefill|In-scope = ring-joint SDPA, sparse-MLA (DSA), and MLA attention. Jobs: 'Blaze - Ring Joint SDPA perf checks' (tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check and ::test_ring_mla_chunked_perf_check); the 'Blaze - DSA MLA' jobs (V3.2 and GLM-5.1/5.2 = indexer + sparse SDPA, sparse_mla/test_sparse_mla.py); and the MLA attention module tests 'Blaze - MLA', 'Blaze - MLA Chunked', 'Blaze - Kimi MLA' (test_mla.py — drive ring/sparse SDPA via tt/mla/mla.py). Other Blaze jobs (MoE gate, KV cache, prefill block, transformer, GLM MoE, Kimi MoE/prefill, determinism, kimi chunked) are out of scope. NOTE: renamed from galaxy-deepseek-prefill-tests.yaml in PR #49462.|Ring Joint SDPA|DSA MLA|Blaze - MLA|Kimi MLA"
  "blackhole-e2e-tests.yaml|Blackhole E2E MLA|In-scope = sparse-MLA / ring-joint-MLA (SDPA family). The bh_lb_DeepSeek_DSA job runs models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py (indexer + sparse SDPA); the DeepSeek_PREFILL_OP_TESTS jobs run test_ring_joint_mla. Everything else in those jobs (rope, kv cache, moe, non-mla ops) is out of scope.|DeepSeek_DSA|DeepSeek_PREFILL_OP_TESTS"
)

# Per-workflow trigger-event filter (optional). When a workflow is listed
# here, watch.sh only considers runs with this trigger event when picking
# the run to analyze (the API's `event=` filter). Use for nightlies where
# manual workflow_dispatch re-runs interleave with the schedule: a green
# manual run (often a job subset, or a fix-attempt branch of the same sha)
# would otherwise flip the digest ✅ while the scheduled nightly still fails
# — seen on L2 2026-08-15: dispatch #9053 success on the SAME sha as
# scheduled #9056 failure. Workflows not listed here consider every event.
declare -A PIPELINE_EVENT=(
  ["tt-metal-l2-nightly.yaml"]="schedule"
)
