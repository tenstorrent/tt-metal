# Qwen3.8 SWE optimization — 27 October 2026

Status: local decode/prefill improvements validated; isolated SWE8 and frozen SWE25 controls audited. Candidate scaling and SWE100/500 remain pending.

Target: useful SWE-bench Verified throughput on one QB2 (four Blackhole chips),
with correctness and small scored samples established before expansion to 100
and 500 tasks. The user authorizes local device tests and QB2 dispatch CI,
including changes to the autoport, vLLM integration, TTI, and sampler experiments.

Source baseline: Metal `ed3804e8afd3475a1fb74054a346d662cc8b4844`,
autoport `models/autoports/qwen_qwen3_8_27b`, all 64 layers, native 262144 context.
The separate `models/demos/qwen38_27b_qb2` is not the implementation under test.
Worktree: `/home/mvasiljev/build/tt-metal-qwen38-swe`, branch
`mvasiljevic/qwen38-swe-opt`; the existing Granite checkout is preserved.

## Current measured state

- Isolated SWE8 control: **7/8 solved**, all eight normal submissions, 5718.577 s
  trial envelope. See [compact control](isolated-swe8-control.json).
- Frozen SWE25 B8/256/C8 medium control: **16/25 solved (64%)**, all 25 normal
  submissions, no exceptions, 9323.502 s trial envelope. All isolation phases and
  stable audit hashes pass. All 923 server responses are accounted: 922 tool-call
  endings and one capped 16384-token response omitted from the saved trajectory,
  recovered on successful Sphinx7889. Observed network retrieval attempts failed
  DNS; inspected Git history is historical/setup content. This is not a matched
  GPU parity or absence-of-pretraining-memorization verdict. See
  [audited SWE25](isolated-swe25-control.json). The deterministic every-fourth
  selection was fixed before this25-task run; earlier Qwen pilot outcomes were
  known. It is not a claim of blind selection before every Qwen experiment.
- Grouped B16/512/C8 SWE8 candidate: 6/8 solved, two agent timeouts, 6363.251 s
  envelope. This does **not** establish accuracy parity or a SWE speedup. See
  [complete accounting](isolated-swe8-grouped.json).
- Masked-state C1 integration smoke: normal submission and reward1 on one known
  task, complete9/9 API responses and isolation audit. This is integration
  evidence, not an accuracy estimate. See [smoke](masked-serving-smoke.json).
- Flat-prefill local stage: independent **clean-pass**, exact full-model states,
  logits and outputs; clean median TTFT **8771.253→7727.073 ms (11.9046% lower)**,
  unchanged decode. L1-input and larger-grid alternatives were adapted, measured
  and retained as slower/capacity results. The feature is default off. See
  [implementation/evidence](FLAT_PREFILL.md) and
  [independent review](FLAT_PREFILL_FINAL_REVIEW.md).
- Isolated low-effort SWE8 pilot: 7/8, all normal submissions, but 6057.211 s
  versus the medium control5718.577 s (+5.92%) and319360 versus204224 completed
  output tokens (+56.38%). All432 HTTP responses and actual isolation pass.
  Retain medium; this single pilot does not show low is universally slower. See
  [matched low comparison](isolated-swe8-low-comparison.json).
- Flat-prefill serving smoke: normal submission and reward1 on the same known
  task, all16 API responses complete with tool-call endings,5325 server/saved
  output tokens, actual isolation and stable export hashes pass. Startup logs
  confirm64 layers, native262144 context, grouped64/masked48/flat64. Trial
  envelope710.233s includes task setup; agent384.260s. This is integration
  evidence, not C16 qualification or a measured SWE speedup. See
  [audited flat smoke](flat-serving-smoke.json).
- C16 grouped+masked SWE25 candidate40585466415 attempt2: **14/25 solved**,
  22 normal submissions, three agent timeouts;8194.275s trial envelope versus
  control16/25,25 normal and9323.502s. The12.11% shorter envelope does not
  establish an acceptable accuracy/runtime tradeoff. Paired outcomes are12
  both-solved,4 control-only,2 candidate-only,7 neither. All744 requests are
  reconciled:741 complete (736 tool_calls,5 capped) and3 incomplete timeout
  bodies;552462 completed output tokens include81920 omitted capped tokens.
  All25 isolation checks and stable exports pass. See
  [audited C16 candidate](isolated-swe25-c16-candidate.json).
  Attempt1 failed before model loading in fabric topology setup; attempt2 used
  identical sources on a different runner. No task score comes from attempt1.
- Hold the prepared C16 SWE100/500 arms. Qualify the combined grouped/masked/flat
  image at C8 on the same frozen25 tasks and individual budgets. This whole
  configuration comparison also adds independently validated flat prefill; it
  cannot isolate concurrency alone. TTI source649c4342d6a1aab5f99c7c6ece672721182a98e9
  retains resident16/block512, medium, agent6300/request3600/output16384/native262144.
  Actual SWE100/500 remain pending.

User steering28October: investigate residentB32 with32 or more task permits
for SWE100/SWE500 andTerminal89 before selecting final concurrency. The C8
50-task partitions remain prepared fallback only. See the reconciled
[C32 memory ledger and optimization targets](C32_MEMORY.md), including
checkpoint geometry, duplicate MLP owners and a group-exclusive storage design.

Raw CSV/Tracy, token arrays, logs and archives remain local or in CI. Only code
and compact documentation are published. Keep the distinct cohorts, resident
batch sizes, HTTP concurrency and policy labels when comparing these rows.

## References and existing evidence

- [Experiment 26 snapshot](https://github.com/tenstorrent/agentic-research/tree/mvasiljevic/experiment-26-qwen38-autoport/experiment-26),
  research commit `817b96e7`.
- [Model card](https://huggingface.co/Qwen/Qwen3.8-27B), pinned checkpoint
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: SWE-bench **Pro 61.7**,
  not a reference for Verified. No matched Verified GPU reference has been found.
  The TTI provisional Pro-as-Verified reference is removed on the experiment ref.
- Historical B32/C8 SWE8 repeats
  [40395063128](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40395063128)
  and [40395139503](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40395139503):
  4/8 and 5/8 raw verifier successes, 2/8 and 5/8 normal endings, respectively;
  1.769823/1.768001 evaluation hours. Nine of sixteen attempts hit the agent
  deadline. Prior trajectory reviews disclose upstream-patch exposure in some
  successes. These are contextual measured controls, not accepted accuracy parity.
- Archived 32-task SWE score 0/32 used a 900-second outer agent deadline;
  every task timed out. Do not interpret it as an uncensored model score.
- Gemma experiment 29 supplies the measurement method: hold task IDs, harness,
  verifiers, sampling and deadlines fixed; report scored outcomes, normal endings,
  malformed/capped responses, generated work, cache reuse, startup and tails.
  A cached-token fraction or sampler replay gain is not a SWE score improvement.

## First scored candidate

Hypothesis: fixed B32 decode wastes work at C8; B8 should lower decode latency
and allow more of the same tasks to complete, preserving model/policy/context.
This changes resident capacity, including its derived sampling CCL default;
it is a whole serving configuration comparison, not isolated padding evidence.

TTI `6264049eb089ac94988c7279110420c859b9035d`, published experiment branch
`mvasiljevic/qwen38-swe-b8-20261027`. This retains the immutable image
`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:182a91996c8104cb1373503bf9326deb5d98e01347655bb1305840165262950e`
and plugin input `1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf` (image includes
the archived autoport deployment patch). No model source overlay is applied.

Tasks: astropy 13398, 13453, 14309, 7166; scikit-learn 13135; sphinx 8459;
sympy 18698, 21612. Frozen 500-task registry, mini-swe-agent 2.2.8,
seed1234, temperature1/top-p.95/top-k20, 16384 output, 6300-second agent,
3600-second request, 7200-second suite, eight clients, prefix caching,
shared recurrent snapshots and 256-token checkpoint geometry are preserved.
Server resident capacity changes 32→8. No claim of supported-context reduction.

Forecast before collection: 1–2.4 CI hours, depending on task tails; prior B8
samples used shorter deadlines and cannot guarantee completion at this protocol.
Do not extrapolate a full-suite go from censored eight-task wall time.

First dispatch 40516088399 failed during source checkout because GitHub checkout
does not resolve abbreviated commit `6264049e`. No model ran. Corrected dispatch
[40516531644](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40516531644)
uses the published experiment branch, whose immutable HEAD is recorded above.
Collect the resolved TTI commit, runtime spec and all outcomes before comparison.

Host validation: 11 existing audit/mount tests passed; importing the actual TTI
evaluation registry confirmed only SWE, C8, 6300 seconds and the intended policy.

## Sampler experiment

User-requested comparison: current Sampling1D versus TTSampling. The standalone
`tests/compare_samplers.py` uses full Qwen vocabulary 248320 split over TP4,
identical BF16 input logits, explicit per-slot seeds, k20/p.95/temp1, matching
power-of-two padding and collective configuration. It records five warmed
traced replay windows, 32 matched stochastic draws, and six global vocabulary
boundary maxima. Batch8/32 and baseline/async gathers are separate strata.
No model weights or hidden dimensions are reduced in the sampler operation.
This is a sampler microbenchmark, not a model quality or SWE measurement.

Local runtime: owned container `qwen38-swe-local`, Ubuntu24 dev image
`8767a9a60bed`, compiled TTNN from the retained experiment checkout
`/data/mvasiljev/tt-metal-exp26-20261023-0822` at `3fbf3dbf1e88`,
Python packages from the existing host Metal venv, model/sampler Python source
from this worktree. Record this difference from the exact CI image.
Added container-local libnsl2; original Ubuntu22 image could not load the newer
binary's GLIBCXX requirement. Four local devices are discoverable.

Probe development failures are retained outside the repository under
`/home/mvasiljev/build/qwen38-swe/`: initial parameter vector length32 failed
the B8 sampling contract; corrected to B8. A missing plus_one warmup failed
trace capture; warmup now includes every captured operation. Neither failure
is a sampler performance measurement.

### Measured sampler decision

Final common-source repeats (five 200-replay windows per row):

| Logical batch / collective | Sampling1D µs | TTSampling µs | Result |
|---|---:|---:|---|
| 8 / existing gather | 474.379 | 557.621 | TTSampling 17.55% slower; pads to32 |
| 8 / existing gather, compact TT offsets/masks | 474.379 | 521.567 | TTSampling 9.95% slower; actual8 rows |
| 32 / async gather | 463.956 | 533.586 | TTSampling 15.01% slower |

All three comparisons have matching source and input hashes. The256/256 active
draws for each B8 candidate and1024/1024 B32 draws exactly match Sampling1D.
Every row passes six unique-maximum vocabulary-boundary cases on every active
slot. The compact variant stages TTSampling's offsets and greedy mask for B8
before capture, avoiding its stock fixed32-row offset broadcast failure. No
sampler replacement is selected. These are traced sampler-only host wall times;
they establish neither model-task accuracy nor a SWE runtime gain. Independent
process repeats of the stock rows agree closely (see local raw artifacts).
The compact measured comparison is in [samplers](samplers/comparison.json). Raw per-draw JSON, CSVs, Tracy captures and logs stay in local/CI evidence and are not published with the source.

The earlier failed capture left its process alive in teardown; a subsequent
probe waited on its device lock. Evidence was retained, triage attempted (installed
0.3.32 versus required tt-exalens0.4.1), and a version-bypassed diagnostic saved.
That mismatched diagnostic's core warnings are not a model defect diagnosis.
Stopped/restarted only the owned Qwen container, reset the four local devices once
with tt-smi, and verified all four were discoverable. All final sampler processes
then opened, ran and closed the mesh successfully. Capture cleanup now ends the
capture even when a captured operation raises.

## Separate reasoning policy pilot

The model card and pinned chat template support medium reasoning. Its instruction
differs from default xhigh, so this is explicitly a policy experiment with its
own measured score, not a kernel speedup. Thinking remains enabled and preserved
by the checkpoint defaults. Same eight tasks, B8/C8, generation sampling and
all deadlines as the first candidate; output budget remains16384.

TTI `a6258f1645e649c82a76e8a2cf63b7f9ed6d0617`, experiment branch
`mvasiljevic/qwen38-swe-b8-medium-20261027`,
[CI40518335859](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40518335859).
Its optional SWE-only control forwards reasoning_effort via chat_template_kwargs;
public validated policy selectors are recorded by the audit middleware without
logging arbitrary template values. Twenty host tests plus four subtests pass,
including actual catalog task/deadline/policy preservation and request observation.
Prediction: potentially less per-turn reasoning; overall task latency and solve
rate may worsen, as the model card cautions. Retain the same1–2.4 CI-hour planning
window until actual completion, output work, caps, failures and verifier scores
are collected. No default-policy promotion or full-suite decision yet.

Raw run artifacts and local logs stay under `/home/mvasiljev/build/qwen38-swe/`.
Copy only compact evidence and provenance into this report after validation.

## Collected evidence and current gates

The B8 xhigh pilot finished with5/8 raw verifier successes, three agent timeouts,
6363.34 seconds trial envelope and251874 saved output tokens. Medium finished
with6/8 raw successes, all eight normal endings,3102.96 seconds and124375 saved
output tokens. However, three successful medium tasks retrieved upstream fixes
(sphinx8459, sympy18698/21612); failed astropy13398 also downloaded the exact fix.
These network-enabled scores do not establish clean accuracy. Do not remove
exposed tasks after observing outcomes to create a new denominator.

An opt-in isolated-agent protocol now keeps setup/verifier egress while removing
external connectivity during agent execution and retaining the host model API.
A local Docker proof passed. The first CI smoke40536968443 is rejected: saved
Harbor config proved that the canonical Docker driver discarded the isolation
flag, despite a successful verifier result. Driver forwarding and provider/type
guards are fixed. Repeat smoke40543287655 passes with TTI18dc75300434736c2a3df1a23d54709c6ffe0f91:
actual internal primary, separate public setup, transitions [true,false,true],
settled workloads and removed networks. Astropy14309 submitted normally and
passed its verifier in308.155s,18responses/zero missing calls, with no upstream-
fix retrieval. This is a protocol smoke, not an accuracy estimate. The auxiliary
post-cleanup exporter rejected the new exact overlay digest; its narrow allowlist
fix passes59 focused tests. Clean B8/C8 SWE8 now runs as CI40554052639 at
TTIb3c86734fd2311fd0be73afc307be15bffbd0ed0 and has completed:7/8
verifier successes, all8 normal submissions,5718.577s trial envelope. All8
actual network-phase audits pass; the fixed stable exporter succeeds with
verified exact files. All349 HTTP requests completed200/tool_calls, with
204224 outputtokens. One472-token response was omitted from the saved agent
trajectory after a recovered tool-format error; the server audit accounts for
it. No upstream-fix retrieval appears in the actual command review. See the
[compact control receipt](isolated-swe8-control.json). This fixed8 result is
not a general accuracy estimate or matched GPU-parity claim.

The default-off grouped recurrence passes paired local all64-layer token tests:
B16 decode13.8087→17.5110 tokens/s/user (1.2681×), B32 clean8.88722→12.81941
(1.44245×). Strict B32 state/logit continuation tests also pass exactly. See
[details](GROUPED_RECURRENCE.md). These fixed-work measurements are independent
of the untrusted SWE accuracy runs; no SWE throughput gain is claimed.

The immutable source-overlay image receipt binds Metal805f873d19d8c21c90757420c8da76ad0308cd69:
`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:47319d8b11749575cc921bae9c03458d37be2667433d7e8e7e0faa3be7a3d895`.
The prepared TTI candidate `mvasiljevic/qwen38-swe-grouped-b16-p512-20261027`
uses resident16, block512, C8, grouped1, medium, and the same isolated protocol.
CI40559423498 is running that candidate at TTI32bf15405b77b02765d424e5509fd496705eb02d; it has not been selected.
Matched clean8 precedes the frozen25-task pilot, then100/500. Native262144
admission and pinned weights/tokenizer are retained throughout.

The [fresh independent local review](GROUPED_FINAL_REVIEW.md) passes all grouped
correctness/quality/trace/watcher and matched repaired-clock profiler gates.
The artifact-only tests/audit_swe_isolation.py fails closed on absent flags,
wrong/missing task identities, public primaries, incomplete phase transitions
or unsettled/unclean resources. It passes the actual repeat smoke and rejects
the earlier nonisolated one; protocol success remains separate from rewards.

The default-off [masked recurrent state commit](MASKED_RECURRENCE.md) passes
full64 strict state/logit/token/qualitative checks and a separate three-repeat
clean timing pair at8 active requests in16 resident rows:13.87818→15.71310
tokens/s/user (1.132216×), with unchanged first-token latency. Its matched reduced
profile and [independent final review](MASKED_FINAL_REVIEW.md) pass. The final
generic-op output declaration correction has focused strict revalidation;
original full/profile source hashes are retained. Neither running SWE8 image
includes this experiment. Large CSV/Tracy captures and raw token arrays remain
local or in CI artifacts; only code and compact documentation are published.
