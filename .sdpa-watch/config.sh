# SDPA + Kimi K3 pipeline watcher config
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

# ---- Slack emoji for PR states ---------------------------------------------
# Workspace custom emoji, used by the digest notes (watch.sh) and the autofix
# bot's messages (~/.sdpa-fix). An unknown name shows as plain ":name:" text.
EMOJI_PR_OPENED=":git-opened:"
EMOJI_PR_MERGED=":git-merged:"

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
  "sanity-tests.yaml|Sanity|In-scope = ALL sanity-pipeline SDPA coverage, on every SKU. The retired blackhole-sanity-tests.yaml was folded into this workflow. Trigger matters: a PUSH run carries the Wormhole + simulator legs (wh_n300_civ2, sim_wh_n150, sim_bh_p150); the 2-hourly SCHEDULE run carries the Blackhole legs (bh_p150b_civ2 and the QuietBox job). bh_p100 sanity does NOT run here (it runs inside L2 Nightly's ttnn-p100-tests). Three job families: (a) 'ttnn-sanity-tests / ttnn sdpa group [sku]' — tests/ttnn/unit_tests/operations/sdpa whole-dir sweep, including the sparse SDPA op tests test_sparse_sdpa.py and test_sparse_sdpa_msa.py and the reduced windowed / spec-multi-pos subset (the full windowed and spec-multi-pos matrices moved to the L2 nightly sdpa dir in #60755). (b) 'ttnn indexer_score group [sku]' — the rank7 accuracy subset of tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py. (c) 'blackhole-multi-card-fast-sanity-tests / TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' [bh_quietbox_2], owned by Pavle Josipovic. It runs the WHOLE multi-card suite tests/nightly/blackhole/sdpa/ (ring_joint incl. the kimi_k3 presets, exp_ring_joint, sparse_sdpa_multidevice, scaled_dot_product_attention_sprint, ring-joint/MLA perf checks), test_ring_mla_fused_gather.py -k 'production_dimensions or comparison_mode' (split-KV ring MLA, includes kimi_k3), test_ring_indexer_score_dsa_perf.py, test_ring_indexer_score_dsa_4d.py (first-class test_indexer_score_qb_* cases, Blackhole, exactly four devices) and test_ring_indexer_score_dsa.py -k 'full_mesh or sptp'. ALL of that is ours, so do NOT dismiss a failure here just because it is not a perf check. The ONE line in this job that is not an SDPA regression is test_high_bw_all_gather.py::test_high_bw_all_gather_quietbox_ci_perf: CODEOWNERS assigns high_bw_all_gather to metalium-developers-sdpa, but it is a marginal bandwidth-threshold gate that trips often: footnote only, never escalate. ALWAYS name the SKU, and call it out when a failure hits only one arch or only the simulator. Out of scope: ttnn reduce group, ttnn deepseek_prefill group, 'deepseek_prefill d_p multi-chip op unit tests (QB2 only)' (dispatch/combine/rope/kv-cache, no SDPA or K3-specific op), conv and pool groups, ops-unit-tests, UMD, profiler and runtime sanity, and all build-artifact jobs.|ttnn sdpa group|ttnn indexer_score group|SDPA PERF"
  "sanity-tests-debug.yaml|Sanity Debug|In-scope = the SDPA groups in this nightly DEBUG run. This workflow re-invokes sanity-tests.yaml under debug flags, so it carries the same SDPA job set: 'ttnn sdpa group' and 'ttnn indexer_score group' on Wormhole (wh_n300_civ2), Blackhole (bh_p150b_civ2) AND the simulator legs (sim_wh_n150, sim_bh_p150 — these SDPA sim lanes ARE in scope; only the non-SDPA ttsim / runtime-sim sanity jobs are not), plus the single QuietBox job 'TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)' on bh_quietbox_2, which carries the whole tests/nightly/blackhole/sdpa/ sweep, test_ring_mla_fused_gather.py, and the ring_indexer_score_dsa perf/4d/full_mesh-sptp tests — all ours except test_high_bw_all_gather (SDPA-owned per CODEOWNERS, but a marginal bandwidth gate: footnote only, never escalate). IMPORTANT: this workflow has three nightly variants (plain, watcher-enabled, LLK-asserts-enabled) and each job name carries the mode plus the Ubuntu version, e.g. '(Ubuntu 24.04 with LLK asserts)'. ALWAYS state which debug mode the analyzed run used, and say so if only one of Ubuntu 22.04 / 24.04 failed: an SDPA failure that reproduces ONLY under LLK asserts or ONLY under watcher is a debug-check signal rather than a plain functional regression: a silent hang there usually means a device assert (LLK or watcher) fired and halted the core, so say that and name the case; do not guess a race or uninitialized state. A run often stays 'queued' for hours because unrelated bh_p300 jobs wait for a runner; judge it on its finished SDPA jobs. Out of scope: 'ttnn reduce group', 'ttnn deepseek_prefill group', blackhole deepseek per-core allocation tests, non-SDPA ttsim / runtime-sim sanity, and all build-artifact jobs.|sdpa|ttnn indexer_score group"
  "tt-metal-l2-nightly.yaml|L2 Nightly|In-scope = SDPA-family AND Kimi-K3 op coverage in the L2 nightly. (1) SDPA nightly: 'ops-unit-tests / ttnn nightly sdpa tests [wh_n150_civ2 / wh_n300_civ2 / bh_p150b_civ2]' (tests/ttnn/nightly/unit_tests/operations/sdpa — incl. the full windowed and spec-multi-pos matrices moved here by #60755, and the sparse SDPA op tests test_sparse_sdpa.py, test_sparse_sdpa_msa.py, _perf and _block_cyclic_multidevice) and 'ttnn nightly sdpa stress tests' (MLA stress). (2) 'ttnn-p100-tests / ttnn sdpa group [bh_p100]' and 'ttnn-p100-tests / ttnn indexer_score group [bh_p100]': the bh_p100 copies of the sanity sdpa dir sweep and the indexer_score rank7 subset — this is the ONLY place bh_p100 runs them. (3) 'ttnn nightly experimental tests (wormhole) / (blackhole)': ONLY these parts are ours: (i) experimental/indexer_score/ (test_indexer_score.py; the ring_indexer_score_dsa*.py files need exactly 4 Blackhole devices and SKIP here, so a hard failure rather than a skip is itself the signal); (ii) test_topk_large_indices.py; (iii) the WHOLE experimental/kda/ dir (Kimi delta attention ops used by Kimi-K3, CODEOWNERS metalium-developers-sdpa; no device-count guard, so failures on wh_n150_civ2 / bh_p100 / bh_p150b_civ2 are real); (iv) the Kimi-K3 MoE op cases inside experimental/deepseek_prefill/: every test id carrying k3 / kimi_k3 / k3_situ (test_moe_fused_swiglu*_k3_situ, test_single_routed_expert*_k3*, test_routed_expert_hybrid / test_hybrid_routed_expert K3 cases, test_routed_expert_bias k3, and their *_perf files). Every OTHER experimental test sharing those jobs (non-K3 deepseek_prefill ids, minimal_matmul, mla_wo, generic moe, quasar, etc.) is OUT of scope even though its log lives in the same job. (4) 'ttnn nightly experimental tests (blackhole - viommu)' [bh_p150b_civ2_viommu]: runs ONLY the @pytest.mark.requires_host_iommu tests across the experimental AND sdpa nightly dirs, the only CI home of our device-profiler perf gates: indexer_score math-util +/-2% (test_indexer_score_math_util), sparse_sdpa / sparse_sdpa_msa +/-5% device-duration bands, the KDA op perf gates, the topk_large_indices perf gates and the K3 MoE op perf gates. A failure there is a perf regression on our ops, not flake-by-default: report measured vs expected. (5) 'ttnn attn_res_weighted_reduce_nc unit tests [bh_p150b_civ2]': the Kimi-K3 AttnRes weighted-reduce op, ALL of it ours (K3). (6) 'Disaggregated prefill op unit tests (LB / QB2 / P150 / BH Galaxy)': runs models/demos/deepseek_v3_d_p/tests/op_unit_tests/ (moved here from Blackhole E2E by #56889); ONLY test_ring_joint_mla (SDPA family) and the Kimi-K3 ids (test_dispatch_fabric2d topk16 / k3 routing shape) are ours; the rest (generic dispatch/combine/rope/kv-cache/reduce) is out of scope. (7) 'ops-unit-tests / Galaxy CCL tests [wh_galaxy]' sweeps tests/nightly/tg/ccl (minus moe); ONLY test_ring_joint_attention.py is ours. 'Galaxy MoE tests' and 'BH Galaxy CCL tests' are NOT ours (the pattern is anchored as '/ Galaxy CCL tests' to exclude the latter). (8) 'syseng-didt-tests / didt_mla_sdpa [sku]': tests/didt/test_mla_sdpa.py is a di/dt power stress owned by syseng that drives our flash_mla_prefill op; report a failure with owner 'syseng (exercises our MLA SDPA op)', and treat it as ours only if the log shows an op error (PCC, assert, hang in the op) rather than a power/di-dt or board issue.|sdpa|experimental|/ Galaxy CCL tests|indexer_score group|attn_res|Disaggregated prefill op unit tests"
  # RETIRED — Perf Device Models. Its one in-scope item, the
  # INDEXER_SCORE_PERF_CHECKS=1 gate on test_indexer_score_math_util, no longer
  # exists in perf-device-models-impl.yaml: #49878 (2026-07-22) moved the
  # indexer_score math-util band and the sparse_sdpa / sparse_sdpa_msa duration
  # bands off Tracy onto the realtime device profiler, behind the
  # @pytest.mark.requires_host_iommu marker. They now run in the L2 Nightly job
  # 'ttnn nightly experimental tests (blackhole - viommu)', which the L2 entry
  # above already matches on its 'experimental' pattern — so this entry is
  # dropped rather than repointed, and the workflow itself (conv and other ops
  # perf) carries nothing of ours. Left as a comment because the entry read
  # green unconditionally for two months, which is the failure mode to avoid.
  "t3000-tests.yaml|T3K E2E|t3k_ccl_tests only — in-scope failures are limited to tests/nightly/t3000/ccl/test_ring_joint_attention.py (ring-joint SDPA). Other CCL tests in that job are out-of-scope; DeepSeek MLA / prefill failures are not ours. NOTE: #56637 (2026-09-22) consolidated t3000-{e2e,integration,unit}-tests.yaml into this single 'T3K tests' workflow, so the job now arrives under it; the matrix entry in tests/pipeline_reorg/t3k_e2e_tests.yaml still runs 'pytest tests/nightly/t3000/ccl' under the name t3k_ccl_tests. The e2e suite only runs on the 6am-UTC schedule (run-e2e-tests defaults to false on a dispatch), hence the schedule pin below.|t3k_ccl_tests"
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
  "blaze-models-prefill-tests.yaml|Blaze Prefill|In-scope = (A) SDPA family: ring-joint SDPA, split-KV ring MLA, sparse-MLA (DSA), MLA attention; and (B) ALL Kimi-K3 coverage, every K3 job end to end (model, MoE, KDA, AttnRes, MLA, transformer, disaggregated runner, DSpark drafter, perf gates), not only its attention. (A) jobs: 'Galaxy-Blaze-models-prefill-SDPA-Perf-tests / Ring joint SDPA perf + ring MLA chunked 55k@5k perf, accuracy, determinism' (tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py checks incl. -k kimi_k3, plus test_ring_mla_fused_gather.py::test_ring_mla_split_kv_perf_check); the GLM '(GLM-5.x-744B) sparse MLA ...' job, only its sparse_mla/test_sparse_mla.py part; the combined '(Kimi-K2.7-1T) MLA chunked, MoE, transformer ...' and '(Mistral-Small-4-119B) MLA, MLA chunked, block, MoE, transformer ...' jobs, only their test_mla.py part (it drives ring/sparse SDPA via tt/mla/mla.py); the rest of those combined jobs is out of scope. (B) K3 jobs, all in-scope in full: '(Kimi-K3-2.8T) contract gates, KDA 5k, MLA chunked 10k@5k and 55k@5k' (host contract tests under tests/kimi_k3/, KDA acceptance, test_mla.py k3 chunked); '(Kimi-K3-2.8T) KDA perf 5k + MLA chunked perf 55k@5k'; '(Kimi-K3-2.8T) MoE accuracy 5k' (pcc/test_ttnn_moe.py::test_kimi_k3_moe); '(Kimi-K3-2.8T) prefill transformer L24, chunked L24 and traced L24 vs golden trace' (tests/kimi_k3/test_transformer_depth.py, test_chunked_prefill.py, test_prefill_transformer_chunked.py); '(Kimi-K3-2.8T) DSpark drafter accuracy 5k + 55k vs GPU trace, and K3 taps L25' (the K3 DSpark drafter: it runs the dflash_prefill/test_dflash*.py tests on the DSpark checkpoint plus test_dspark_taps — this is K3 and IN scope even though the test files are named dflash); '(Kimi-K3-2.8T) disaggregated prefill L24 KV accuracy 55k@5k + perf 250k@5k, SC1', '(Kimi-K3-2.8T) disaggregated prefill KV accuracy 55k@5k + perf 1M@5k, SC4' and '(Kimi-K3-2.8T) DSpark drafter KV accuracy, eager 34 users, 55k@5k + perf 1M@5k, SC4' (run_multirank_pcc.sh kimi_k3 sc1 or sc4 [--variant dspark], not pytest; the perf gate prints measured vs baseline from perf_baselines.json at the log tail); '(Kimi-K3-2.8T) disaggregated prefill SC4 perf' (scaling_summary.py --sc4-only on ubuntu-latest; NOTE it can report success when the SC4 leg itself was skipped or cancelled — say so rather than calling SC4 green). Two SHARED jobs where only the K3 part is ours: '(Kimi-K2.7-1T, Kimi-K3-2.8T) MoE perf 5k' (only the k3 ids, e.g. k3-galaxy-torus-xy-8x4, of test_kimi_moe_perf run with -k galaxy) and '(all prefill gate models) MoE gate accuracy' (only the kimi_k3 ids of test_moe_gate_prefill2d). TRAP: 'high_bw_all_gather CCL perf + accuracy' is grouped under the job prefix 'Galaxy-Blaze-models-prefill-SDPA-Perf-tests'; CODEOWNERS assigns it to metalium-developers-sdpa, but it is a marginal bandwidth gate that trips often: footnote only, never escalate. Out of scope: all non-K3 Kimi-K2.7 jobs other than its MLA part (components, chunked prefill, MoE, rmsnorm, KV cache table, prefill runner, the K2.7 DFlash drafter, K2.7 disaggregated legs), GLM MoE / MTP, MiniMax, HCA and the two '(DeepSeek-V4-Flash) SWA chunked accuracy/perf' jobs (they call scaled_dot_product_attention directly but are deliberately not ours).|Ring joint SDPA|MLA chunked|sparse MLA|MLA accuracy|KDA|Kimi-K3|DSpark|prefill gate models"
  "blackhole-e2e-tests.yaml|Blackhole E2E|In-scope = (A) SDPA family (sparse MLA, ring MLA perf) and (B) ALL Kimi-K3 coverage. Job names are 'blackhole-e2e-tests (<box>, <sku>) / <name>'; ALWAYS name the box/SKU, and identify legs by name + SKU, never by line number. (A) 'Disaggregated prefill sparse MLA accuracy tests' [bh_loudbox] (sparse_mla/test_sparse_mla.py), and in 'Disaggregated prefill perf tests' [bh_loudbox] the Mistral4 ring-MLA perf gate test_mla_perf.py::test_mistral4_mla_chunked_perf_loudbox (a device-duration band on our ring MLA op: report measured vs band). (B) K3, in full: 'bh_lb_Kimi_K3_ATTN_RES' [bh_loudbox] (whole models/demos/deepseek_v3_d_p/tests/attn_res/ dir); 'ttnn attn_res_gather_softmax unit tests' [bh_loudbox] (the K3 AttnRes op); 'Disaggregated prefill Kimi-K3 MLA perf tests (LoudBox)' [bh_loudbox] and '(QuietBox)' [bh_quietbox_2] (kimi_k3/test_mla_perf.py -m perf, profiles our ring-joint SDPA op on K3 shapes); 'Disaggregated prefill Kimi-K3 KDA perf tests (QuietBox)' [bh_quietbox_2] (KDA layer perf SP4xTP1, test_kimi_k3_tp_reduce_scatter_perf, test_qkv_convolution_exchange_kimi_k3_perf); in 'Disaggregated prefill accuracy tests' [bh_loudbox] the KDA acceptance leg kda/layer/test_acceptance.py::test_synthetic_kimi_k3_accuracy_and_determinism (SP8xTP1), test_qkv_convolution_exchange_matches_single_sequence (the fused KDA QKV-conv exchange op, K3), and every kimi_k3 / k3 test id in the tests/pcc/ sweep (test_kimi_k3_moe, test_moe_gate_prefill2d kimi_k3, test_ffn and test_shared_expert k3 ids); in 'Disaggregated prefill perf tests' [bh_loudbox] the K3 legs kda/perf/test_layer_perf.py::test_synthetic_kimi_k3_perf (SP8xTP1), test_kimi_k3_tp_reduce_scatter_perf and test_qkv_convolution_exchange_kimi_k3_perf; in 'Disaggregated prefill accuracy & weight cache tests' [bh_quietbox_2] the KDA acceptance SP4xTP1 leg, the qkv-conv exchange test and every kimi_k3 / k3 test id. Out of scope: everything else sharing these jobs (Kimi-K2.7 ids, rope, kv cache, non-K3 moe, prefill_dispatch, weight cache, fabric/UDM/CCL, ttnn stress and sharding, MoE/e2e throughput lines, D2D socket sync) even though the logs land together. The disaggregated-prefill op unit tests (test_ring_joint_mla) no longer run here; they moved to L2 Nightly.|sparse MLA|Disaggregated prefill accuracy|Disaggregated prefill perf tests|Kimi[-_]K3|attn_res_gather_softmax"
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
  # T3K tests bundles unit + integration + e2e, but a workflow_dispatch runs
  # the e2e suite only when run-e2e-tests is explicitly ticked (it defaults to
  # false — the suite is 195 of the 279 machine-minutes of a full run). A
  # dispatch therefore carries no t3k_ccl_tests job at all, which would read as
  # a clean run. Only scheduled runs cover it.
  ["t3000-tests.yaml"]="schedule"
)
