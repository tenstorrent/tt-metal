# Gemma 4 final benchmark attempt — incomplete

| Candidate upstream task | Completed / frozen subset / full population | Score | Published score / delta |
| --- | --- | --- | --- |
| MMLU-Pro (`mmlu_pro`) | 0 / 280 / 12032 | Unavailable | [82.6%](https://huggingface.co/google/gemma-4-26B-A4B-it#benchmark-results) / unavailable |
| GSM8K-CoT (`gsm8k_cot`) | 0 / 256 / 1319 | Unavailable | Unavailable |
| IFEval (`ifeval`) | 0 / 256 / 541 | Unavailable | Unavailable |

| Required profile | Concurrent requests | Server slots | Actual input / output target | TTFT / TPOT / ITL / end-to-end / percentiles | Request / output throughput | Prefill FLOP / decode DRAM roofline |
| --- | ---: | ---: | --- | --- | --- | --- |
| Single user | 1 | 1 | 4096 / 128 | Not measured | Not measured | Not measured |
| 32 users | 32 | 32 | 4096 / 128 | Not measured | Not measured | Not measured |

This attempt did not complete. The required upstream lm-evaluation-harness client
is absent and dependency installation is prohibited by the supplied repository
instructions. Stage 10 had already stopped its server. No timed runner was
invoked and no Stage 11 inference or phase measurements were collected.
There is no accuracy verdict or score threshold.

The manifest retains the packaged common subsets and document/few-shot hashes;
these are intended sample counts, not evaluated samples. Dataset verification,
live chat-template/thinking policy validation and upstream scoring remain undone.
Published MMLU-Pro82.6% and alternative GPQA Diamond82.3% reference figures are
retained in published_references.json; protocol/subset equivalence is unproven. All response, stop and truncation counts are
unavailable. The preparation configuration is explicitly incomplete.

Stage 10 used the generated 30-layer model, selected
head4_inner_all4_shared_down4 precision, P300x2 mesh 1x4 and 262144 context,
according to its historical handoff. Current imports, tokenizer revision and
running configuration could not be proved because no server was running.
Historical commands and cleanup are retained without presenting them as current
measurements. Its concurrency-one result used 32 server slots and is excluded
from the required one-slot row.

No full-phase collector was validated. Both complete-phase host timing and work
accounting remain required; no host latency is substituted for device time.
Live vLLM profiling was not enabled. The benchmark stage remains incomplete.

See [run notes](../RUN_NOTES.md), [identity](../identity.json),
[setup evidence](setup_inventory.json), [summary](summary.json),
[manifest](manifest.json) and the AutoDebug/AutoFix reports in the parent directory.

Context check passed (262144 preserved). Evidence check failed with missing generated implementation identity (exit2); this is not a completed benchmark. Check logs are retained.

Offline setup progress: the pinned tokenizer template passed one-BOS checks in
both thinking modes, and the installed vLLM benchmark CLI starts. Stage 10's
startup log reports approximately190 seconds to readiness, before this attempt;
its server subsequently stopped. No Stage11 launch has occurred.

Host tests cover the phase-timeline reducer and the planned server-configuration
validator. These tools are not yet connected to a live collector/control hook.
The checkpoint's top_k64 policy is capped to32 by the current device formatter;
an existing request-specific host-sampling route is under validation. No claim
is made that live benchmark protocol or roofline collection is ready.

Final setup audit: the client prerequisite remains unavailable across three
consecutive goal turns; see [blocked audit](blocked_audit.json). Supply a
provisioned upstream lm-eval client or explicitly authorize its isolated setup
under the repository's no-install constraint. Offline tools are preserved, but
live collector integration, both profiles and accuracy measurements remain
unverified and incomplete.
