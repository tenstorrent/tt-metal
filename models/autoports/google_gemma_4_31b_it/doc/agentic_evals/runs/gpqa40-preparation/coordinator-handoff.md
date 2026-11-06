# Held GPQA40 coordinator handoff

Preparation only: no publication, dispatch, image pull or new evidence audit performed. Reuse proof is `../repaired-image-reuse/RESULT.md`; original host acceptance is `RESULT.json`. Await valid preceding20, schedule admission and proven repaired image before dispatch.

Local publication checkout `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12` is clean on dedicated branch `mvasiljevic/gemma4-31b-goal12-v3`, at37ee508969c730320c01259fafe340e6c2c14d4e. GPQA40 candidate6efe6739e6513d0edbf49a8ef39527b80a23784d is its descendant; full-tree changed paths are only model eval configuration and the two previously tested workflow tests. TT publication checkout is already at accepted ab2730c27e557b09b2eafe3494e47790adc210aa. No TT publication change needed. Plugin remains ca7a18246e4111629ed6cb4a7d1192002749fccf.

Coordinator-only safe publication path (not executed; recheck clean state and remote branch before mutation):

```bash
cd /home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = mvasiljevic/gemma4-31b-goal12-v3
git fetch origin mvasiljevic/gemma4-31b-goal12-v3
git merge-base --is-ancestor FETCH_HEAD 6efe6739e6513d0edbf49a8ef39527b80a23784d
git merge --ff-only 6efe6739e6513d0edbf49a8ef39527b80a23784d
test "$(git rev-parse HEAD)" = 6efe6739e6513d0edbf49a8ef39527b80a23784d
git push origin HEAD:refs/heads/mvasiljevic/gemma4-31b-goal12-v3
```

Run each command with failure stopping the sequence. If remote ancestry or checkout cleanliness changed, stop and reconcile; never force push. This execution checkout is not the publication checkout and must stay remote-free.

Image is **pending**: run42273660905/build job128084494246 must prove successful immutable image publication from TTab2730, TTI37ee508 and pluginca7a182, followed by actual normal pull/startup confirmation. Insert its proven repository@sha256 digest below; do not substitute an older image or mutable tag. Explicit TT/plugin inputs do not rebuild or authenticate an overridden image. Existing reuse proof covers only this config/test-only TTI delta.

Coordinator dispatch template (not executed):

```bash
GPQA40_IMAGE='<PROVEN_REPOSITORY@sha256:DIGEST_FROM_RUN_42273660905>'
gh workflow run manual-tt-shield-dispatch.yml \
  -R tenstorrent/tt-agentic-bringup-qb2 \
  -f model=gemma-4-31B-it \
  -f runner-label=bh-qb-ge \
  -f device-type=p300x2 \
  -f workflow=evals \
  -f impl-of-model=gemma4-31b-autoport-v3 \
  -f tt-metal-git-ref=ab2730c27e557b09b2eafe3494e47790adc210aa \
  -f inference-server-git-ref=6efe6739e6513d0edbf49a8ef39527b80a23784d \
  -f vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf \
  -f docker-image="$GPQA40_IMAGE" \
  -f run-full-evals=false \
  -f run-ai-summary=false \
  -f create-issue-comment=false
```

Before launch register frozen40 from existing `cohort.json` centrally and record planned_clients4/B8, immutable source/image and schedule decision in coordinator run manifest. Normal run must show candidate checkout, exact40 task selection, C4, retained diagnostics and accepted repaired runtime. Preserve release output126976 and timeout14400; no request/runtime changes. Dispatch is an eval rung, not spec tests or an automatically authorized release. No new measurements or readiness claims are made here.
