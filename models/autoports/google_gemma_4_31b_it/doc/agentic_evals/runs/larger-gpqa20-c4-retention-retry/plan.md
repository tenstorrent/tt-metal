# Same20 C4 archive-repair retry admission

Preparation only; no dispatch. Coordinator must first accept the local idle-export/archive confirmation and supply the immutable published TT repair SHA. Keep TTI37ee508969c730320c01259fafe340e6c2c14d4e and stock pluginca7a18246e4111629ed6cb4a7d1192002749fccf. Preserve the entire prior20 cohort, C4/B8/maxseq8/context262144, seed42/temperature1/top_k20/top_p.95, output126976, request14400s, run12h and normal evals route. This is a validity repair of unbanked20, not a new rung or score selection. Existing SWE5 remains in flight; count it and every other in-flight job against the six-run grant.

The preceding run measured evaluator1.5119826687h and setup/device overhead0.1371839979h, total device-job1.6491666667h. Add new-image build0.9h and shared queue0.5h, both guesses: retry forecast **3.0491666667h**. Image-build time must not be credited away before confirmed immutable compatibility. This estimate is a measured point forecast with uncertain queue/build and stochastic tails, not a cap; preserve all release limits and update after completion. Record actual queue/build/evaluator/device/workflow components separately. No extra admission wrapper is added to stock CI.

After coordinator supplies the accepted immutable SHA, use the unchanged stock workflow. Deliberately omit docker-image so the changed source builds normally; reuse only if the coordinator independently proves an immutable compatible image. Shell variables below are task-specific placeholders, and this document is not executable:

```bash
retry_tt_sha='<accepted published full TT SHA>'
gh workflow run manual-tt-shield-dispatch.yml -R tenstorrent/tt-agentic-bringup-qb2 \
  -f model=gemma-4-31B-it -f runner-label=bh-qb-ge -f device-type=p300x2 -f workflow=evals \
  -f impl-of-model=gemma4-31b-autoport-v3 \
  -f tt-metal-git-ref="$retry_tt_sha" \
  -f inference-server-git-ref=37ee508969c730320c01259fafe340e6c2c14d4e \
  -f vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf \
  -f run-full-evals=false -f run-ai-summary=false -f create-issue-comment=false
gh run list -R tenstorrent/tt-agentic-bringup-qb2 --workflow manual-tt-shield-dispatch.yml \
  --limit 10 --json databaseId,status,conclusion,createdAt,displayTitle
```

Bind the returned new run by unique immutable refs and creation time before any next dispatch; record the new arm/run in coordinator-owned compute/plan. At completion use the existing monitor/collector, avoiding duplicate collection. If manual collection is needed, use `gh run download <confirmed-run-id> -R tenstorrent/tt-agentic-bringup-qb2 -D models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/larger-gpqa20-c4-retention-retry/collected`. Confirm actual startup pins and repaired source, and require private phase export survives normal stock collection with complete retention integrity. Apply the preceding retained20 audit plan, including independent prompt/physical-chain association; no row-order shortcut. Scores remain reported, not gated. Bank20 only after failures at limits are classified from the actual collected run.

## Next40 schedule against first dry

First-dry deadline is **2026-11-07 02:28:49 UTC**. At planning time **Nov6 18:42 UTC**, retry point finish is approximately **21:45 UTC** if accepted and dispatched immediately. Forty cannot dispatch until20 is valid. Scaling measured20 evaluator time linearly gives40 evaluator **3.0239653374h** (a guess: newly added questions may have different tails), plus0.1371839979h setup and0.5h queue = **3.6611493353h**, only if the retry image can be proven reusable with the frozen40 TTI source. Otherwise add0.9h image build = **4.5611493353h**.

Sequential retry→40 point costs are therefore **6.7103160020h** with proven reuse and **7.6103160020h** without. Latest retry dispatch to merely finish40 by first dry is about **19:46:12 UTC** with reuse or **18:52:12 UTC** without; these leave zero audit/freeze/dry-preparation margin and are not recommended launch deadlines. From18:42, predicted40 finishes about **Nov7 01:25 UTC** with reuse (roughly64min left) or **02:19 UTC** without (roughly10min left). Add actual repair acceptance delay and audits to these projections. A full4h tail contingency for40 alone would miss the dry deadline; do not claim robust schedule fit from linear scaling. Recompute immediately after valid20 and retain measured stop/functional_at20 if40 no longer fits with required dry preparation. Never lower caps or skip the20 validity check to fit40, and do not delay the next rung for unrelated efforts once admitted.

## Dispatch argv verification

Corrected the preparation command to explicitly include `-f impl-of-model=gemma4-31b-autoport-v3`; omission could select a stock implementation and is unacceptable. Checked every supplied field against prior20 manifest and actual workflow input block in `bringup/artifacts/stage12-gpqa20-c4-retained-complete.log:40–69`: model, runner label, p300x2 device type, evals workflow, explicit autoport implementation, TTI37ee508, pluginca7a182 and the three false flags match. The only intended differences are the pending repaired TT SHA and omission of the old immutable image to obtain a new source build. Prior additional-args and schedule were empty, throttle-perf0, timeout1080minutes; leave these stock defaults unchanged. C4/B8/cohort/output/timeout settings are owned by the unchanged TTI SHA, not invented dispatch flags. The prior run matrix explicitly selected this same implementation and startup registered its autoport adapter. Before launch substitute only the accepted full TT SHA; record and verify the new actual input block and autoport registration. No dispatch was executed during verification.
