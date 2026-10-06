# Independent decode-retention checkpoint review

Reviewer: fresh xhigh subagent `decode_reuse_review`, read-only stage-review.
Verdict: **clean-pass**, bounded opt-in synchronous retention checkpoint only.
This is not closure of the overall TSU optimization task.

## Required work

None identified for this checkpoint.

## Independently verified evidence

Matched commands, generated texts and token lengths agree across both primary
cohorts. TSU42.954→47.982 and43.030→47.788; E2EL lower261/292ms. Median ITL
is essentially unchanged, supporting removal of request-level capture overhead.
Candidate TTFT rises49ms in the first cohort and2ms in the second; retain both.
Reversing two formatting changes exactly reconstructs the launch adapter hash.
The generator's recorded configurable bound and current hard bound are1024.

Source guards release incompatible traces before eager prefill, preserve nested
cache ownership, and force request-boundary token/position refresh. Changed
page values refresh retained decode tables. Reduced31-row Watcher evidence
validates changed tokens/pages and4096→4097→4096 transitions without warmed
program-cache growth. Sampler inspection confirms eager sampling does not
replace the captured slot and compatible greedy refresh does not reset it.

## Hard-check gaps and residual risk

- Reduced evidence records model trace identity but not sampler trace identity
  directly; adding the latter would strengthen future artifacts.
- Watcher JSON predates runner configuration/source manifests; console and
  Watcher logs are consistent, but provenance is less self-contained.
- Shared qualitative prompts are21–35 tokens. They establish preserved short
  behavior; changed-input/page controls and4K/8K exact serving outputs cover
  the newly retained long path.
- Broader scheduler overlap, final source/image reproduction and remaining
  matrix/context cases are outside this checkpoint pass.

## Classified anomalies

- Random4K/8K benchmark refusals exactly match baseline output: controlled.
- Long qualitative answers truncate at256 tokens with `finish_reason=length`
  and match the prior accepted suite: controlled, not corruption.
- Sync counters show zero token readbacks, but the uncounted finalizer performs
  blocking conversion: documented diagnostic limitation, not zero transfers.

## Scope

Complete generator/adapter, new CPU contracts/reduced runner, model/sliding-tail,
common sampler and allocation-tracker paths; worklog/AutoDebug/AutoFix/topology;
31-row Watcher data; matched benchmark commands/JSON; launch/server metadata;
qualitative texts and independently checked prior suite;119-test CPU log.
Skills included stage-review, optimize, tracing, vLLM integration, qualitative,
review core/router, trace and serving reviews. Only read-only git/rg/sed/jq,
hashing and small artifact analysis ran; no device access or file edits.

Follow-up inspection confirmed `GEMMA4_PREFILL_TRACE=0` already disables eager
decode reuse through `serving_prefill_eligible()`. No redundant production
guard was needed. CPU cases were added for both128- and4096-token disabled-mode
requests; their next test run is recorded separately.
