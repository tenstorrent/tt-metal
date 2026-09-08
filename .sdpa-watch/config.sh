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
#   refresh it (that's what silently made every cron tick fail rc=1).
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
  "sanity-tests.yaml|Sanity|In-scope = ALL sanity-pipeline SDPA coverage, on every SKU. The retired blackhole-sanity-tests.yaml was folded into this workflow, so Wormhole, Blackhole and ttsim SDPA now report here. Three job families: (a) 'ttnn sdpa group [sku]' — tests/ttnn/unit_tests/operations/sdpa whole-dir sweep on wh_n300_civ2, bh_p100, bh_p150b_civ2, sim_wormhole_b0, sim_blackhole; includes the sparse SDPA op tests test_sparse_sdpa.py and test_sparse_sdpa_msa.py, both in-scope. (b) 'ttnn indexer_score group [sku]' — the rank7 accuracy subset of tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py. (c) 'TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' [bh_quietbox_2] — this ONE job absorbed the retired 'sdpa nightly tests (QB2 only)' lane and is owned by Pavle Josipovic. It runs the WHOLE multi-card suite tests/nightly/blackhole/sdpa/ (ring_joint, exp_ring_joint, sparse_sdpa_multidevice, scaled_dot_product_attention_sprint, and the ring-joint/MLA perf checks) AND test_ring_indexer_score_dsa_perf.py AND test_ring_indexer_score_dsa_4d.py — ALL of that is ours, so do NOT dismiss a failure here just because it is not a perf check. There is no '-k indexer_score_qb' selection any more; those multi-device cases are first-class test_indexer_score_qb_* tests in the 4d suite (Blackhole, exactly four devices). The ONE line in this job that is not an SDPA regression is test_high_bw_all_gather.py::test_high_bw_all_gather_quietbox_ci_perf — CODEOWNERS assigns high_bw_all_gather to metalium-developers-sdpa, so it IS our team's code, but it is a marginal bandwidth-threshold gate that trips often: report it as a footnote, never escalate it as an SDPA regression. ALWAYS name the SKU when reporting, and call it out when a failure hits only one arch. Out of scope: ttnn reduce group, conv and pool groups, ops-unit-tests, UMD, profiler and runtime sanity, and all build-artifact jobs.|ttnn sdpa group|ttnn indexer_score group|SDPA PERF"
  "sanity-tests-debug.yaml|Sanity Debug|In-scope = the SDPA groups in this nightly DEBUG run. This workflow just re-invokes sanity-tests.yaml under debug flags, so it carries the same job set: 'ttnn sdpa group' on Wormhole (wh_n300_civ2) and Blackhole (bh_p100, bh_p150b_civ2) — tests/ttnn/unit_tests/operations/sdpa, including test_sparse_sdpa.py and test_sparse_sdpa_msa.py — plus 'ttnn indexer_score group' and the single QuietBox job 'TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' on bh_quietbox_2. There is no separate 'sdpa nightly tests (QB2 only)' lane any more — that one job now carries the whole tests/nightly/blackhole/sdpa/ sweep plus test_ring_indexer_score_dsa_perf.py and test_ring_indexer_score_dsa_4d.py, and all of it is ours except test_high_bw_all_gather (SDPA-owned per CODEOWNERS, but a marginal bandwidth gate — footnote only, never escalate). IMPORTANT: this workflow has three nightly variants (plain, watcher-enabled, LLK-asserts-enabled) and each job name carries the mode plus the Ubuntu version, e.g. '(Ubuntu 24.04 with LLK asserts)'. ALWAYS state which debug mode the analyzed run used, and say so if only one of Ubuntu 22.04 / 24.04 failed: an SDPA failure that reproduces ONLY under LLK asserts or ONLY under watcher is a race / ordering / uninitialized-state signal rather than a plain functional regression, and is worth calling out as such. Out of scope: 'ttnn reduce group', blackhole deepseek per-core allocation tests, ttsim / runtime-sim sanity, and all build-artifact jobs.|sdpa|ttnn indexer_score group"
  "tt-metal-l2-nightly.yaml|L2 Nightly|In-scope = SDPA nightly tests (tests/ttnn/nightly/unit_tests/operations/sdpa, run by the 'ttnn nightly sdpa' jobs) PLUS three experimental areas run inside the 'ttnn nightly experimental' job: (i) tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/ — note the indexer_score/ SUBDIR, test_indexer_score.py lives there now; the ring_indexer_score_dsa*.py files beside it are Blackhole-only and need exactly 4 devices, so they SKIP on these single-card SKUs and a hard failure rather than a skip is itself the signal; (ii) test_topk_large_indices.py; and (iii) tests/ttnn/nightly/unit_tests/operations/experimental/kda/, in-scope because CODEOWNERS puts experimental/kda/ under metalium-developers-sdpa. KDA is Kimi delta attention: test_sigmoid_gated_rms_norm, test_recurrent_chunk_scan, test_qkv_causal_conv1d_silu, test_affine_exclusive_scan, test_prepare_chunk_recurrence, test_summarize_chunk_recurrence, test_reduce_affine_transforms, test_kda_performance_model. These have NO device-count guard, so they genuinely run on wh_n150_civ2 / bh_p100 / bh_p150b_civ2 and their failures ARE ours. Every OTHER experimental test sharing that job (deepseek_prefill, minimal_matmul, mla_wo, moe, etc.) is OUT of scope — ignore it even though its log lives in the same job. The nightly sdpa dir-sweep also runs the sparse SDPA op tests (test_sparse_sdpa.py and test_sparse_sdpa_msa.py, plus _perf and _block_cyclic_multidevice) — all in-scope. ALSO in-scope: the 'ops-unit-tests / Galaxy CCL tests [wh_galaxy]' job, which absorbed the TG ring-joint coverage that used to live in galaxy-e2e-tests.yaml (that watcher entry is retired). Its cmd is a whole-dir sweep 'pytest tests/nightly/tg/ccl --ignore tests/nightly/tg/ccl/moe', of which ONLY test_ring_joint_attention.py is ours — all_gather, reduce_scatter, all_to_all, broadcast, send_recv and the rest of that dir are out of scope even though they share the job. 'Galaxy MoE tests' is NOT ours, and neither is 'BH Galaxy CCL tests' (it sweeps tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/galaxy_nightly, a different dir with no ring-joint coverage) — the pattern is anchored as '/ Galaxy CCL tests' precisely to exclude it.|sdpa|experimental|/ Galaxy CCL tests"
  "perf-device-models.yaml|Perf Device Models|In-scope = the INDEXER_SCORE_PERF_CHECKS=1 gated check on Blackhole: tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py::test_indexer_score_math_util (renamed from test_indexer_score_perf_check; note the indexer_score/ subdir in the path). NOTE: the former SDPA_PERF_CHECKS gate and test_sdpa_perf_check were removed from this workflow. Other ops-perf-tests (conv, etc.) are OUT of scope.|P150 BH device perf"
  "t3000-e2e-tests.yaml|T3K E2E|t3k_ccl_tests only — in-scope failures are limited to tests/nightly/t3000/ccl/test_ring_joint_attention.py (ring-joint SDPA). Other CCL tests in that job are out-of-scope; DeepSeek MLA / prefill failures are not ours.|t3k_ccl_tests"
  # RETIRED — T3K Integration. Its DiT jobs (t3k_sd35_large,
  # t3k_flux1, t3k_motif, t3k_wan2.2, t3k_mochi, t3k_qwenimage) are GONE from
  # tests/pipeline_reorg/t3k_integration_tests.yaml, which is now just
  # t3k_tteager_tests + t3k_trace_stress_tests — the filter matched no jobs.
  # The DiT coverage moved to tests/pipeline_reorg/models_{unit,e2e}_tests.yaml
  # under a 'TT-DiT ' name prefix, split across the scheduled workflows
  # models-t{1,2,3}-{unit,e2e}-tests.yaml, one set per tier.
  # Deliberately NOT repointed at one tier — that would silently watch a third
  # of the coverage while reading as full. Left dormant pending a decision on
  # whether indirect ring-joint-via-DiT-attention is worth three new entries.
  #
  # RETIRED — Galaxy E2E. The 'Galaxy CCL tests' job no longer exists
  # in galaxy-e2e-tests.yaml (the filter matched no jobs); the job moved into the
  # tests/pipeline_reorg/ops_unit_tests.yaml matrix, which is consumed by
  # ops-unit-tests-impl.yaml and scheduled by tt-metal-l2-nightly.yaml. Coverage
  # is folded into the L2 Nightly entry above ('Galaxy CCL tests' pattern), so
  # this entry is dropped rather than repointed — watching two workflows for one
  # job would double-report it.
  "blaze-models-prefill-tests.yaml|Blaze Prefill|In-scope = ring-joint SDPA, sparse-MLA (DSA), and MLA attention. The jobs were renamed to '<Model>-<version>-<size> <unit> <accuracy/perf/determinism> <shape>', so the old 'Blaze - ' names are dead. The seven in-scope jobs are now: 'Ring joint SDPA perf + ring MLA chunked 55k@5k perf, accuracy, determinism' (tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check and ::test_ring_mla_chunked_perf_check); '(GLM-5.2-744B) sparse MLA accuracy + determinism, incl. chunked 5k@1k' (indexer + sparse SDPA, sparse_mla/test_sparse_mla.py); the four MLA attention module jobs '(Kimi-K2.7-1T) MLA chunked accuracy 10k@5k + determinism 5k', '(Kimi-K3-2.8T) MLA chunked accuracy 10k@5k + determinism 5k@5k', '(Kimi-K3-2.8T) MLA chunked accuracy 55k@5k vs golden trace', '(Kimi-K3-2.8T) KDA perf 5k + MLA chunked perf 55k@5k' (test_mla.py — drive ring/sparse SDPA via tt/mla/mla.py; note the last one absorbed a KDA perf leg and was renamed from a plain MLA-chunked-perf name); and '(Kimi-K3-2.8T) KDA accuracy, determinism 5k SP8xTP4' (models/demos/deepseek_v3_d_p/tests/kda/layer/test_acceptance.py::test_synthetic_kimi_k3_accuracy_and_determinism). KDA is in-scope because CODEOWNERS puts ttnn/cpp/ttnn/operations/experimental/kda/ under metalium-developers-sdpa — but TRIAGE it: the KDA *op* is ours while this model-level harness under models/demos/deepseek_v3_d_p/tests/kda/ has no SDPA CODEOWNERS line and is owned by the Kimi-K3 model folks, so say whether the traceback points at the KDA op/kernels (ours) or the model glue (theirs). TRAP: the 'high_bw_all_gather CCL perf + accuracy' job is grouped under the GitHub job prefix 'Galaxy-Blaze-models-prefill-SDPA-Perf-tests' because the matrix gives it test_group=sdpa_perf. CODEOWNERS assigns high_bw_all_gather to metalium-developers-sdpa (it was added for sparse MLA prefill), so it IS our team's code — do not write it off as another team's. Still: it is a marginal bandwidth-threshold gate that trips often, so report it as a footnote and never escalate it as an SDPA regression. NOTE: DeepSeek-V3/V3.2 MLA and DSA-MLA jobs no longer exist here at all, so Blaze sparse-MLA coverage is GLM-5.2 only. Other Blaze jobs (MoE gate, KV cache, prefill block, transformer, GLM MoE, Kimi MoE/prefill/chunked, disaggregated SC4 runner/producer, HCA, MiniMax) are out of scope. Renamed from galaxy-deepseek-prefill-tests.yaml.|Ring joint SDPA|MLA chunked|sparse MLA|MLA accuracy|KDA"
  "blackhole-e2e-tests.yaml|Blackhole E2E MLA|In-scope = sparse-MLA / ring-joint-MLA (SDPA family). The old DeepSeek_DSA / DeepSeek_PREFILL_OP_TESTS job names are dead — every leg is now 'Disaggregated prefill <...> tests [sku]', and the GitHub job prefix carries the box, e.g. 'blackhole-e2e-tests (LoudBox (8xP150), bh_loudbox) / ...'. ALWAYS name the box/SKU when reporting. In-scope legs, by matrix entry in tests/pipeline_reorg/blackhole_e2e_tests.yaml — identify them by job name and SKU, never by line number, since the matrix shifts constantly: 'Disaggregated prefill sparse MLA accuracy tests' (bh_loudbox — GLM-5.2 sparse MLA vs MLACPU, sparse_mla/test_sparse_mla.py, indexer + sparse SDPA); 'Disaggregated prefill accuracy tests' (bh_loudbox — includes test_mla.py::test_ds_mla AND kda/layer/test_acceptance.py::test_synthetic_kimi_k3_accuracy_and_determinism -k SP2xTP4); and the four 'Disaggregated prefill op tests' legs plus 'Disaggregated prefill D2D socket sync op tests' (on bh_p300, bh_loudbox, bh_quietbox_2, bh_p150b_civ2), which sweep models/demos/deepseek_v3_d_p/tests/op_unit_tests/ where ONLY test_ring_joint_mla is ours. 'Disaggregated prefill accuracy & weight cache tests' (bh_quietbox_2) sweeps tests/pcc/ + tests/cache/ — treat only its MLA/attention pcc cases as ours. 'Disaggregated prefill perf tests' (bh_loudbox) is mostly a MoE/e2e throughput proxy and NOT ours, with ONE exception: its 'KDA_PERF_SKU=bh_loudbox ... kda/perf/test_layer_perf.py::test_synthetic_kimi_k3_perf -k SP2xTP4' leg IS in-scope, because CODEOWNERS puts experimental/kda/ under metalium-developers-sdpa — ignore every other line in that job. For BOTH KDA legs, triage rather than assume: the KDA op/kernels are ours, but the harness under models/demos/deepseek_v3_d_p/tests/kda/ carries no SDPA CODEOWNERS line and belongs to the Kimi-K3 model owners, so state which side the traceback implicates. Everything else sharing these jobs (rope, kv cache, moe, prefill_dispatch, weight cache, fabric/UDM/CCL, ttnn stress and sharding) is out of scope even though the logs land together.|sparse MLA|Disaggregated prefill op tests|Disaggregated prefill accuracy|Disaggregated prefill perf tests|D2D socket sync"
)

# Per-workflow trigger-event filter (optional). When a workflow is listed
# here, watch.sh only considers runs with this trigger event when picking
# the run to analyze (the API's `event=` filter). Use for nightlies where
# manual workflow_dispatch re-runs interleave with the schedule: a green
# manual run (often a job subset, or a fix-attempt branch of the same sha)
# would otherwise flip the digest ✅ while the scheduled nightly still fails
# — seen on L2, where a dispatch run went green on the SAME sha as a
# failing scheduled run. Workflows not listed here consider every event.
declare -A PIPELINE_EVENT=(
  ["tt-metal-l2-nightly.yaml"]="schedule"
)
