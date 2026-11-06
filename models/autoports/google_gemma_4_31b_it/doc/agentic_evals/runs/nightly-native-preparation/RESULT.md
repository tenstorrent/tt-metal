# Native-parser nightly candidates

Prepared independent patches against a087be086b6e2cd6edf215dddb4c47c48463f13a: r1_gpqa_diamond.patch (10), swe_bench_verified.patch (5), terminal_bench_2_1.patch (10). These supersede the older bb321bf preparation. Each enables only its named CI_NIGHTLY suite, preserves two clients, output/time limits, native parser registration and all existing revision pins. Model YAML/registration files are absent from the diffs, so parser flags remain exactly the base version. Frozen cohorts nest over smoke; no cohort has been scored here.

Each candidate passed exact stock common/Harbor selection verification, 23 model-specific tests, Ruff format/check. Patches contain only model-specific eval configuration and corresponding tests. Actual device hours zero. No active publication checkout edits, commit, push, dispatch or hardware work. Detached preparation worktree restored clean. Patch hashes are in manifest.json.

Apply one alternative patch to a fresh worktree at the base SHA; do not stack alternatives:

```bash
nightly_suite=r1_gpqa_diamond
nightly_packet=/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-native-preparation
nightly_host_bin=/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin
git apply --check "$nightly_packet/$nightly_suite.patch"
git apply "$nightly_packet/$nightly_suite.patch"
"$nightly_host_bin/python" "$nightly_packet/verify.py" "$nightly_suite"
"$nightly_host_bin/pytest" -q tests/workflows/test_gemma4_autoport_smoke_suites.py tests/workflows/test_eval_config_mode_reference.py
```

Full publication hooks still precede committing. Refresh/reconfirm if the base changes. Stock GPQA smoke-test uses the first three doc_ids once the nightly list exists; the immutable smoke publication remains the exact smoke2 artifact. Dispatch these candidates in CI_NIGHTLY mode, only after valid smoke and measured projection/admission.

## Dispatch recipe (not executed)

The coordinator fills immutable source-only published SHAs and records run/projection/queue admission before launch. Explicitly pin the stock plugin on every dispatch; its resolved value in a previous run does not pin future default-main resolution:

```bash
# Required: nightly_metal_sha and nightly_tti_sha are immutable published commit SHAs.
case "$nightly_suite" in
  r1_gpqa_diamond) nightly_workflow=evals ;;
  swe_bench_verified|terminal_bench_2_1) nightly_workflow=agentic ;;
  *) echo "Unknown nightly suite" >&2; exit 1 ;;
esac
gh workflow run manual-tt-shield-dispatch.yml -R tenstorrent/tt-agentic-bringup-qb2 \
  -f model=gemma-4-31B-it -f runner-label=bh-qb-ge -f device-type=p300x2 -f workflow="$nightly_workflow" \
  -f tt-metal-git-ref="$nightly_metal_sha" -f inference-server-git-ref="$nightly_tti_sha" \
  -f vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf \
  -f run-full-evals=false -f run-ai-summary=false -f create-issue-comment=false
```

Do not launch before the suite's smoke is valid, the measured next-rung projection fits, and the coordinator has a free granted CI queue slot. Record the returned run identity from workflow listing before another dispatch.
