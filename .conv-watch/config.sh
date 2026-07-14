# SDPA pipeline watcher config
# Sourced by watch.sh. Edit this file to change what gets watched.

REPO="tenstorrent/tt-metal"
BRANCH="main"
TT_METAL_DIR="/localdev/skrstic/tt-metal"
SLACK_WEBHOOK_FILE="$HOME/.conv-watch/slack_webhook"

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
OAUTH_TOKEN_FILE="$HOME/.conv-watch/oauth_token"           # MODE A (optional)
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
  "sanity-tests.yaml|Sanity|In-scope = ttnn conv and pool operation unit tests run on push to main: the 'ttnn conv group' job (pytest tests/ttnn/unit_tests/operations/conv — conv2d, conv1d, conv3d, conv_transpose2d, fold, convert_to_chw, prepare_conv_weights) and the 'ttnn pool group' job (pytest tests/ttnn/unit_tests/operations/pool — maxpool2d, avgpool2d, global/adaptive avg pool, upsample, grid_sample). All other Sanity jobs are out of scope.|ttnn (conv|pool) group"
  "blackhole-sanity-tests.yaml|Blackhole Sanity|In-scope = Blackhole ttnn conv and pool operation unit tests: the 'ttnn conv group [bh_...]' job (tests/ttnn/unit_tests/operations/conv) and the 'ttnn pool group [bh_...]' job (tests/ttnn/unit_tests/operations/pool). All other Blackhole sanity jobs (sdpa, other ttnn groups, ops-unit-tests, multi-card) are out of scope.|ttnn (conv|pool) group"
  "sanity-tests-debug.yaml|Debug Sanity|In-scope = ttnn conv and pool op unit tests in the nightly DEBUG sanity run (Debug build, plus the with-watcher and with-LLK-asserts scheduled variants; it re-runs sanity-tests and blackhole-sanity under debug flags). Only the 'ttnn conv group' and 'ttnn pool group' jobs are in scope; all other sanity jobs are out of scope.|ttnn (conv|pool) group"
  "merge-gate.yaml|Merge Gate C++ smoke|In-scope = the conv2d C++ gtest (tests/ttnn/unit_tests/gtests/conv/test_conv2d.cpp) compiled into the tt-nn validation smoke binary (/usr/bin/tt-nn-validation-smoke) and run on every merge to main. Only conv/pool-related smoke failures are in scope; other smoke tests are out of scope.|smoke"
  "tt-metal-l2-nightly.yaml|L2 Nightly Conv/Pool|In-scope = nightly conv and pool op unit tests ('ttnn nightly conv tests', 'ttnn nightly pool tests' — the deeper suite incl. _sweeps and _ulp variants under tests/ttnn/nightly/unit_tests/operations/conv and /pool), the tt_cnn pipeline and builder unit tests ('tt-cnn pipeline unit tests', 'tt-cnn builder unit tests' — models/tt_cnn/tests), and the di/dt conv power-stress jobs (didt_resnet_conv, didt_sdxl_conv, didt_sdxl_conv_1280x1280_upsample). Everything else in this nightly is out of scope.|ttnn nightly (conv|pool) tests|tt-cnn (pipeline|builder) unit tests|didt_.*conv"
  "perf-device-models.yaml|Device Perf|In-scope = conv2d device-perf regression (tests/ttnn/perf_tests/operations/conv/test_conv2d_device_perf.py, the 'ops perf tests' step) plus the resnet50 and stable_diffusion 1.4 device-perf steps. These run inside shared multi-model jobs (N300 WH B0 Set 1 / Set 2 / P150 BH device perf), so ONLY attribute failures whose step or error text is conv2d, resnet50, or stable_diffusion; ignore other models sharing the job.|(N300 WH B0 Set [12]|P150 BH) device perf"
  "fast-dispatch-full-regressions-and-models.yaml|Frequent Models|In-scope = nightly per-block PCC accuracy for the two conv-team-owned models: resnet50 (Nightly N150/N300 resnet50 — large functional plus performant) and stable diffusion 1.4 (Nightly N150/N300 stable_diffusion — UNet resnet/downsample/upsample/transformer blocks). Other models in this workflow are out of scope.|Nightly .* (resnet50|stable_diffusion)"
  "single-card-demo-tests.yaml|Single-card Demos|In-scope = single-card demo/functional runs for the conv-owned models: resnet50 (resnet-N150-func, resnet-N300-func, resnet-N150-perf stability) and stable diffusion 1.4 (stable_diffusion-N150-func e2e demo, noisier). Other CNN demos (vgg, unet, segformer, mobilenet, etc.) are NOT conv-team-owned and are out of scope.|resnet-N(150|300)-(func|perf)|stable_diffusion-.*-func"
  "perf-models.yaml|Model Perf|In-scope = stable diffusion 1.4 end-to-end model perf (stable_diffusion N300 WH B0, stable_diffusion P150 BH — UNet throughput/latency). Note the resnet50 e2e perf group is currently disabled here. Other models are out of scope.|stable_diffusion (N300 WH B0|P150 BH)"
  "t3000-demo-tests.yaml|T3K Demo|In-scope = resnet50 8-chip demo (t3k_resnet50_tests, models/demos/vision/classification/resnet50 ttnn_resnet test_demo.py). Other T3K demos are out of scope.|t3k_resnet50_tests"
  "t3000-perf-tests.yaml|T3K Perf|In-scope = resnet50 e2e model perf (t3k_CNN_resnet50_model_perf_tests, model_perf_t3000). Other T3K perf tests are out of scope.|t3k_CNN_resnet50_model_perf_tests"
  "t3000-integration-tests.yaml|T3K Integration|In-scope = resnet50 performant trace/2cq integration (t3k_resnet_tests, test_resnet50_performant.py). DiT/SDPA and other integration jobs are out of scope.|t3k_resnet_tests"
  "t3000-profiler-tests.yaml|T3K Profiler|In-scope = resnet50 profiling runs (trace-only resnet50, async tracing resnet50). Dispatch-only workflow so runs may be infrequent. Other profiler targets are out of scope.|(trace-only|async tracing) resnet50"
  "galaxy-integration-tests.yaml|Galaxy Integration|In-scope = resnet50 performant on Galaxy/TG (Galaxy resnet50 integration tests, via run_tg_frequent_tests.sh --model resnet50). Other galaxy integration jobs are out of scope.|Galaxy resnet50 integration tests"
  "ttnn-run-sweeps.yaml|Sweeps|In-scope = conv and pool op sweeps (conv2d, conv_transpose2d, pool2d modules under tests/sweep_framework/sweeps). Scheduled/on-demand batch, not per-commit; sweep job names are dynamic so this is a coarse workflow-level watch. Non-conv/pool sweeps are out of scope.|(conv2d|conv_transpose2d|pool2d)"
)
