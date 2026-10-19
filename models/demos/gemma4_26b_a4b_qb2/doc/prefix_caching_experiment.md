# Gemma 4 QB2 prefix caching experiment

Status: contract analysis; no hardware correctness or speed claim yet.

Isolated branch in both repositories: `mvasiljevic/gemma4-qb2-prefix-caching-experiment`.
Metal base: `75c0c0c44c5127a1a9ce5d2b59f3990258a9abd2` (PR 59471).
Inference base: `59c121407e93b9efd221e4ba8fbd3eafb16b862e` (PR 5315).
Worktrees: `/home/mvasiljevic/tt-metal-gemma4-prefix-caching` and
`/home/mvasiljevic/tt-inference-server-gemma4-prefix-caching`.
Plugin inspected: `0a76258e14ca7237067ca9a22ec02e257bee570a`.

## Contract and predicted headroom

The pinned plugin passes complete token prefixes, `start_pos` as the cached-token
count, and `prompt_lens` as the absolute end position (model_runner.py:1971).
The generator must slice `[start_pos:prompt_lens]`, preserve absolute RoPE/cache
positions, and use scheduler-owned per-layer tables and cache without clearing it.
Only complete 32-token blocks can be reused; partial final blocks must be computed.
Sliding and full attention use distinct scheduler cache groups.

The adapter currently rejects nonzero start positions; the generator and model
do not forward them. The decoder already accepts continuation but implements it
as sequential one-token decode calls per layer. This is a correctness baseline,
not evidence of acceleration. Approximate break-even is
`suffix_tokens * continuation_cost_per_token < full_prefill_time(total_tokens)`.
High cache hit rates may help long growing conversations; large tool-result
suffixes may regress. Measure this before expensive agentic CI.

## Checks and isolation

1. Wire suffix slicing and absolute positions through the existing model API;
   avoid prefill trace reuse for continuation until its signature supports offsets.
2. Run focused host contract checks and representative layer/device comparisons:
   partial blocks, sliding boundaries, divergent suffixes, repeats, eviction/reuse.
3. Compare full-model logits and host-sampled outputs, physical batch one.
4. Measure repeated/growing-context TTFT, wall time, hit counters, computed prefill
   tokens, and decode throughput using the existing serving harness.
5. Independently compare explicit agentic request seed 9472 against omitted seed.
   GPQA default seed 42 is separate. Speed alone cannot justify seed removal.
   Retain standard Harbor GPQA10/Terminal5/SWE5 cohorts and existing gates.

No existing PR branch, draft status, kernel, or diagnostic file is changed.
Shield run 39221429342 and unrelated run 39133813650 must remain untouched.
Reuse the supplied 0.23.0 image; no image builds or repeated CI dispatches.

Initial read-only local inspection found four device nodes and a running bringup
container owned by another experiment (`bringup-e96881c64642421baf94289e90faa16a`).
No local hardware operation has been performed; ownership must be resolved before use.

User subsequently required remote CI for all executable validation. No local
test, build, server, Python import, benchmark, reset, or device operation had been
started; there was no owned process to stop. From this point local work is source
inspection, edits, and git preparation only. Do not use the local devices even
if idle. The other user's container remains untouched.

The implementation forwards absolute start positions through adapter/generator/
model, slices only uncached tokens at the generator boundary, and bypasses
start-zero prefill trace reuse for cached continuations. Sampling still sees the
complete prompt, preserving penalty and seed-offset semantics. The experiment
branch advertises caching with `GEMMA4_PREFIX_CACHING_EXPERIMENT=1` and enables it
only in this model's development profile;
this is an experimental configuration, not a passed production capability gate.

Other inexpensive runtime improvements may be investigated, but must remain
separately measurable and model scoped. None has been claimed or measured yet.

## Prepared remote checks (not yet executed)

`tests/test_prefix_caching.py` includes host contracts for absolute positions,
suffix-only upload, invalid intervals before cache access, and host-sampling
adapter translation. Live API cases use lengths 33, 63, 65, 1023, 1025, and 1057,
unique cache salts for cold references, hit-counter assertions, greedy token and
top-five logprob comparisons (absolute tolerance 0.05), and divergent suffixes.
One-token response wall time and cache hit deltas are recorded in JUnit properties.
The shared API helper now preserves an explicit `logprobs` request while enabling
token-ID output; previously it unconditionally replaced the requested count with
zero. The new check requires at least five entries to prevent a vacuous comparison.
This is a focused initial gate, not full-logit, eviction, device-sampling,
growing-conversation, or agentic accuracy proof.

Remote commands after launching the existing serving stack at physical batch one:

```bash
python -m pytest models/demos/gemma4_26b_a4b_qb2/tests/test_prefix_caching.py -k 'not live' --junitxml=prefix-host.xml
python -m pytest models/demos/gemma4_26b_a4b_qb2/tests/test_prefix_caching.py -k live --gemma-server-url http://127.0.0.1:8000 --junitxml=prefix-live.xml
```

Use the supplied Docker image and exact experimental refs. The server must set
`GEMMA4_PREFIX_CACHING_EXPERIMENT=1` and `--enable-prefix-caching`; leave chunked
prefill disabled. No local execution of these commands is permitted.

The user explicitly authorized iterative Actions runs in `tenstorrent/tt-agentic-bringup-qb2`;
final release validation must use Shield main. A narrow workflow on new QB2 branch
`mvasiljevic/gemma4-prefix-cache-ci` invokes the model-owned
`tests/run_prefix_cache_ci.sh` in the supplied prebuilt image. It binds only this
model directory from the exact experimental Metal ref over the image's source,
preserving compiled libraries and plugin revision. No image build is performed.
The existing workflow filename is reused only on that isolated branch so GitHub
can dispatch it without changing the default branch. It runs host contracts first,
then the full 30-layer server at physical B1 and the focused API cases.

Remote run [39226809560](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39226809560)
used Metal `57552912e65862378d9cade7b21c52e57d9451da`, workflow `daa02d5`,
runner `120-qb2-p05t01`. All 11 host contracts passed in 2.22 seconds. Server
startup failed before opening devices: `/mnt/MLPerf/huggingface/hub` was not
writable by the image user. This is infrastructure setup evidence, not model
correctness or performance evidence. The workflow now checks the runner-local
Shield pinned checkpoint cache identified in successful run 39217198324, mounts
it read-only when present, and otherwise uses a writable experiment-specific cache.

Remote run [39227511399](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39227511399)
used Metal `589b160eff576d95b42a1f06f9776212d0b0a92b`, workflow `0324431`,
runner `qb2-120-p02t02`. Host contracts passed again. No complete prior cache was
present; downloading into the experiment workspace failed with ENOSPC during
checkpoint loading. Devices closed cleanly in the failure log. No prefix requests
ran. The automatic download fallback is removed. The next run is a short remote
storage/cache discovery probe; subsequent model runs require every indexed shard
to exist in the selected pinned cache and use offline read-only access.

The remote runner now explicitly matches the profile's `OMP_NUM_THREADS=8` and
attempts startup triage before terminating a live server that exceeds the startup
bound. Neither adjustment has been measured as a runtime improvement.

Remote discovery run [39230215308](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39230215308)
passed without executing the model. It found the complete pinned checkpoint at
`/mnt/models/huggingface/hub` and reported 1.7 TiB free on the local NVMe volume.
All shard names from the index were checked for readability. The workflow now
prioritizes this shared cache, mounts it read-only, sets `HF_HUB_OFFLINE=1`, and
requires 100 GiB free locally before pulling the existing image or running the model.
The failed worker's standard infrastructure cleanup reclaimed 913.7 GB of Docker
storage; the experiment did not issue broad cleanup or reset commands.
# Remote correctness finding and diagnostic follow-up

## Independent suffix variants

### Variant results and next scope

Shared run `39235780379` and traced run `39235783153` both passed 16 host
tests and all six full-logit decode-contract cases. Median wall seconds for
the two 31-token suffix cases:

| Mode | Prompt 63 cached / cold | Prompt 1023 cached / cold |
|---|---:|---:|
| Serial | 2.686 / 0.111 | 2.691 / 0.444 |
| Shared MLP | 2.269 / 0.112 | 2.291 / 0.446 |
| Traced suffix | 0.890 / 0.117 | 0.898 / 0.453 |

Each row has three warmed repetitions and its own cold control. Runners are
distinct as recorded below; do not interpret small cross-machine differences
as causal effects. Both variants preserve exact tested decode logits, but
even the traced path loses to cold prefill at these short contexts. The
one-token suffix cases remain approximately 0.12 s.

Next checks use the traced candidate: (1) fresh growing conversations with
4096/8192/16384 cached-prefix tokens and exactly 127 newly computed tokens,
with the first diagnostic/compile conversation excluded; (2) a separate
4096-context server to force eviction with 12 distinct conversations, test
divergent partial-block suffixes, and verify repeated device-sampled outputs
at physical B1. The latter retains seed9472 solely as a deterministic
sampling-state check, not evidence that the seed is necessary or removable.
No standard-Harbor accuracy or final Shield release has been dispatched for
the caching candidate yet.

Those next checks were dispatched with exact Metal
`7d5c3e5cb8438989e36a6a35c704f1154f9a4584`, workflow `38d4d6f`, traced mode:
growing run `39238201191` on `qb2-120-p01t03`; lifetime run `39238204679`
on `qb2-120-p03t09`. Both remain separate from the baseline Shield release.

### Serial control result

[39235204816](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39235204816)
passed 11 host tests and all six separately named decode-contract cases.
Token-major ordinary decode, layer-major continuation, and scheduler-reused
prefix KV produced exactly equal full-vocabulary logits in those cases.
This does not establish cold-prefill equality: its greedy token differed at
lengths 33, 63, and 65, and matching greedy tokens at the other lengths did
not imply matching logits.

Three warmed pairs per shape, physical B1, one output token, host sampling,
identical selected precision configuration. The first diagnostic/compilation
requests are excluded. Values below are medians in seconds; TTFT is the
server histogram-sum delta and wall time includes the HTTP response.
Counters asserted full recomputation for cold salts and the listed suffix
length for cached requests. The precision *configuration* is fixed, but its
prefill/decode arithmetic differs as documented above.

| Prompt | Computed suffix | Cold wall | Cached wall | Cold TTFT | Cached TTFT |
|---:|---:|---:|---:|---:|---:|
| 33 | 1 | 0.109 | 0.115 | 0.094 | 0.098 |
| 63 | 31 | 0.111 | 2.686 | 0.096 | 2.667 |
| 65 | 1 | 0.126 | 0.122 | 0.110 | 0.102 |
| 1023 | 31 | 0.444 | 2.691 | 0.427 | 2.673 |
| 1025 | 1 | 0.774 | 0.119 | 0.755 | 0.100 |
| 1057 | 1 | 0.780 | 0.119 | 0.762 | 0.102 |

Within these shapes, serial continuation helps the nearly complete ~1K
cache hits but substantially regresses 31-token suffixes. This is not a
growing-conversation or agentic-runtime result. Eviction, divergent suffixes,
device sampling, longer contexts, functional accuracy, and seed independence
remain unqualified. The performance-claim review guidance keeps these
serving-level results distinct from device-kernel or end-to-end eval claims.

The user authorized concurrent jobs on the four-machine QB2 pool. No tests
run locally. Serial control run `39235204816` keeps Metal
`8c594566c40526540337c005cc100e3725f56074` on runner `qb2-120-p02t02`.
The separate branch `mvasiljevic/gemma4-qb2-prefix-suffix-variants` at
`fccedf77af1dfd8d9b80d5a057c7098c0a30e727` adds two default-off alternatives:

- `GEMMA4_PREFIX_SHARED_BATCH=1`: reuse the existing shared-MLP batch path
  for 8..32 suffix tokens; sequential cache updates/attention retain absolute
  positions and repeated request tables. Run `39235780379`, runner
  `120-qb2-p05t01`.
- `GEMMA4_PREFIX_TRACE_SUFFIX=1`: reuse the existing logits-only model decode
  trace for B1 suffixes of at least eight tokens; explicit tokens/positions,
  no sampler draws, and an owned final-logits copy. Run `39235783153`, runner
  `qb2-120-p01t03`.

Workflow `492b0fd` selects exactly one mode, records runner/mode, and refuses
device access when another device container or process owns it. Both jobs
use the same image and inputs as the serial control. These are not duplicate
retries, and neither variant is enabled in the inference profile. Results
are pending; no optimization claim is made from source inspection. The
tracing skill informed reuse of exact-signature warmup/capture and output
ownership instead of introducing a new capture lifecycle.

Follow-up [39233122287](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39233122287)
on Metal `be19cbffcc745b04836cf26c98cc04e897da204f`, workflow `7436b68`,
completed with the expected cold-parity failure. Ordinary decode and
continuation on identical cold prefix KV were exactly equal over the entire
vocabulary (maximum and RMS difference zero). Warm HMA continuation also
exactly matched its ordinary-decode replay. Cold prefill versus ordinary
decode differed by maximum 5.12890625 and RMS 0.8913413 logits. The cold
winner was 128 (8.5625 versus token 130 at 8.125); decode/continuation chose
130 (9.4375 versus 128 at 8.5625). This is not a near-tie explanation.
Full logits and JSON summaries are retained as CI artifacts. These results
prove only the one-token replay contract, not all cache lifetimes or accuracy.

The next focused run retains the original strict cold-parity test unchanged
but selects a separately named decode-contract diagnostic. It compares
token-major ordinary decode with layer-major continuation for 1/31-token
suffixes at page/window boundaries, and compares scheduler-reused KV with
the cold-cache decode reference. Full-vocabulary equality is required there.
Three later cold/cached pairs per shape bypass diagnostic replay after
warmup, recording wall time, server TTFT, cache hits, and computed prefill
tokens. A passing decode-contract test must not be described as passing
cold-prefill equality or agentic accuracy. No seed has been removed.

Run [39230676369](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39230676369)
used Metal `c56f74cc7f93165dbfcfe1a1e1202c0b4dbc6fa4` and workflow `75a859e`.
All 11 host contracts passed. The full 30-layer physical-B1 server started,
but the first live 33-token case failed: cold greedy token 128, cached token
130, with 32 confirmed cache hits and only one computed suffix token. Cold
37.33 s and cached 30.21 s include different first-use compilation and are
not speedup evidence. Remaining live cases were stopped at this failure.

Source-only diagnosis is recorded in `prefix_caching_autodebug.md`. The
leading hypothesis is prefill/decode arithmetic differences (including BF16
direct prefill KV versus BFP8 cached KV); scheduler ownership is not yet
proven. The next remote diagnostic replays the final token through ordinary
decode and continuation on exactly the same physical cache, saves complete
logits and top candidates, then restores the original path. It uses the full
selected policy; no relaxed correctness assertion. Its timings are invalid
for performance because explicit readbacks and replay are enabled via
`GEMMA4_PREFIX_DIAGNOSE=1`. All validation remains remote. No local validation
or device process has been started by this experiment.

# Completed lifetime and growing-context evidence

## Standard-Harbor bootstrap provenance

The seeded Astropy inference ref is
`5c53b2ffca0b8857b700872a6635206155922414`; unseeded is
`c94227c39636b6b102f21695f1d2ec0d486b975b`. Only the CI-nightly task list
and, in the latter branch, the SWE request seed differ from the initial
experimental profile. Server seed9472 remains set in both arms.

- 39240243118 failed before model access because an abbreviated inference
  SHA was interpreted as a branch by actions/checkout. Subsequent dispatches
  use full SHAs.
- 39240553380 reached server readiness, but the client failed because
  run.py requires its repository working directory for VERSION. The run
  was cancelled after source inspection also identified the need for a
  host-accessible bind address for Harbor task containers. No task result.
  Its artifact directory contains stale earlier test files and is not
  usable as new correctness evidence.
- 39240897270 passed the device-free client/cache preflight on
  qb2-120-p05t05.
- 39241097259 was cancelled after discovering the working-directory issue,
  before accepting any task result.
- 39241180817 uses Metal `0aa4df2867dc3c8b709e04a8c6f089234033c524`,
  workflow `fe31b8a`, full seeded inference SHA, cache off. The workflow
  now separates client progress, preserves old output outside the current
  results directory, records precision.json, and runs the client from its
  own repository directory. All four planned arms reuse the same image.

These are experiment-entrypoint failures, not model accuracy failures.
Only this experiment's runs/containers were stopped; baseline Shield
39221429342 and other users' work were not modified.

The corrected run reached the standard Harbor task step. The exploratory
2x2 uses identical Metal `0aa4df2867dc3c8b709e04a8c6f089234033c524`,
workflow `fe31b8a`, image, physical B1, 262144 context and traced suffix mode.
The cache-off arm disables prefix caching explicitly at server startup;
the client-only inference profile does not launch or override this server.

| Request seed9472 | Prefix caching | Run | Runner |
| --- | --- | --- | --- |
| Present | Off | 39241180817 | qb2-120-p01t03 |
| Present | On | 39241807530 | qb2-120-p03t09 |
| Absent | Off | 39241811717 | qb2-120-p02t02 |
| Absent | On | 39241815211 | qb2-120-p05t06 |

These four concurrent jobs have distinct physical workers. The baseline
Shield worker qb2-120-p05t09 is not used. Runs are pending measurement;
dispatch/reaching the client step is not evidence of task completion.

Run 39241180817 completed with workflow exit0 but an invalid task result:
Harbor `1da0bfd8c71cadbff17413fac984b8e391d2afc2`, mini-swe-agent2.2.8
raised `ModuleNotFoundError: No module named 'mcp'` in the LiteLLM tools
path before any HTTP inference request. Prompt, generation and successful
request counters were all zero. Agent execution spent 251.46 seconds
retrying the import; total Harbor runtime was 303.51 seconds. Reward0 and
`NonZeroAgentExitCodeError` are infrastructure evidence, not model accuracy.
The client nevertheless printed `Acceptance: PASS`; green CI must not be
used as the functional gate. The other three arms were cancelled when this
shared dependency failure became known, to avoid duplicate retries.

Source inspection finds Harbor installs the isolated agent tool with
`--with litellm --with orjson --with fastapi`, omitting mcp. Installing a
host dependency would not repair that task-container tool environment.
Any dependency repair must preserve the standard agent and be identical
across all four arms; no accuracy or seed-removal conclusion is available.

The authorized experiment-only repair pins Harbor1da0bfd8 and replaces its
task installer `--with litellm` with `--with 'litellm[mcp]==1.106.0'
--with mcp==2.2.0`, gated by `GEMMA4_PREFIX_HARBOR_DEPS=1`. No shared
Harbor fork or production branch is modified. New inference refs:
seeded `17dbce4dfc4141efb78bda0bd60a9e8dd5f42f19`, unseeded
`6be1ea66e722f305f328042bfe10785c9f3e018f`.

Device-free remote preflight39246470559 passed in a fresh Astropy task-base
container on120-qb2-p05t01. Image digest
`sha256:29e2270fdba28472dd3aa4cc4a4d5fc80e39c5e10b6abcf1e17ce7358cd0cc9e`.
The previously failing LiteLLM handler and mcp CacheableResult imported.
Inventory: mini-swe-agent2.2.8, LiteLLM1.106.0, mcp2.2.0,
fastapi0.147.0, orjson3.13.0, httpx2 2.14.0. No devices were mounted.

Repaired seeded/cache-off bootstrap39246694017 uses Metal
`7fe17051c4417a01ab7d72d414731145edd6f010`, workflowca53de0 and the
new seeded inference ref. It explicitly waits for a nonzero server prompt
counter before additional parallel arms may launch. Standard agent behavior,
task, timeout, sampling and repetition settings are unchanged.

The repaired bootstrap passed the explicit nonzero server prompt-token
gate. The remaining controls were then dispatched, using workflow666769b
(only an additional stale-PID ownership guard relative to ca53de0):

| Request seed9472 | Prefix caching | Repaired run | Runner |
| --- | --- | --- | --- |
| Present | Off | 39246694017 | qb2-120-p01t03 |
| Present | On | 39247598502 | qb2-120-p03t09 |
| Absent | Off | 39247602550 | qb2-120-p02t02 |
| Absent | On | 39247606568 | 120-qb2-p05t01 |

All use Metal7fe17051c4417a01ab7d72d414731145edd6f010 and the same
repaired dependencies. Real inference is confirmed for the bootstrap;
task reward, errors and trajectory remain pending, not inferred from CI status.

### First repaired trajectory result: unseeded, cache on

Run39247606568 completed with official SWE-bench verifier reward1.0 and
no agent exception. It made35 normally stopped requests (zero length,
error or repetition finishes), then submitted via the standard completion
command. Agent execution702.065s; client wall762s; trial751.767s.
Input521,676 tokens, server cache hits512,064, computed prefill9,612,
output21,260. TTFT sum279.138s; decode sum415.620s. Harbor's own cached
token field was0 despite confirmed server hits, so cache accounting uses
server counters. Final request context27,865, cached27,776, computed89.

Trajectory review found35 agent tool-call turns and a final submission.
The target subclass-property exception test passed. The test log also has
an unrelated leap-second update warning failure; the standard SWE-bench
grading output is PASSED/reward1.0, not a claim that every pytest case passed.
This single successful unseeded cached trajectory does not establish seed
removal safety or a speedup over different trajectories. Other arms pending.

### Seeded, cache on

Run39247598502 also passed official Astropy reward1.0 with no terminal
agent exception. Agent1303.737s, client wall1364s, trial1353.644s.
Server totals:59 stop requests, input1,004,807, cached987,008,
computed17,799, output42,830; TTFT sum451.250s and decode842.080s.
The trajectory records58 agent turns and recovers from one malformed bash
tool call. Harbor omits that failed-format response's usage and reports
989,532 input/32,023 output, so server totals represent actual work.
Both cached arms passed this one task. Their different turn/output counts
prevent interpreting the shorter unseeded run as an isolated speed effect.

Standard mini-swe also embeds host `uname` in its user prompt. Completed
seeded/cached worker reports Linux7.0.0-38; unseeded/cached reports
Linux7.0.0-34. Thus these standard task runs use the same task/image/config
but not byte-identical complete prompts. This is another confound for
causal seed/runtime comparisons. The focused synthetic timing controls
used identical token inputs and are unaffected. No prompt was changed
mid-study to conceal or remove this difference.

### Unseeded, cache off

Run39247602550 submitted normally but failed the official verifier
(reward0.0), with no terminal agent exception. The target subclass-property
test and29 other cases failed. Agent2428.358s, client wall2489s,
trial2478.611s. All77 requests stopped normally: input/computed2,098,765,
cached0, output36,459; TTFT1699.761s, decode718.462s. The final context
was51,758 tokens. This is a real task-quality failure, not infrastructure.

Its uname is Linux7.0.0-34, matching the unseeded cached arm. Canonical
initial system/user messages in this pair have identical SHA256
`e609e00bf27fdef5e0686097b43b80a60d3bdd3334878400b1b44d27877d1213`.
The uname confound is therefore between seed groups, not this cache pair.
Random sampling and divergent trajectories still prevent attributing the
full wall-time/quality difference to caching from one sample per arm.

### Seeded, cache off: final control

Run39246694017 passed official reward1.0, no terminal agent exception.
Agent3042.632s, client wall3104s, trial3093.849s. Server totals:
93 requests (92 stop, one repetition finish), input/computed2,204,852,
cached0, output63,417; TTFT1776.696s, decode1253.430s. Four malformed
tool calls were recovered. Harbor reports only2,066,712 input/57,880 output;
server counters include the discarded/error-corrected responses.

The seeded pair's canonical initial system/user messages have identical
SHA256 `dfb94c33d44e031d768ef129bc8d46e3ab113007ce503701cb3bea18271d850d`.
Their first response is also identical (1125 prompt/279 completion tokens,
same reproduction-script tool call). Later trajectories diverge. Both cache
pairs therefore have matched initial prompts; the host-kernel prompt
difference applies to comparisons between seed groups.

## Consolidated exploratory standard-Harbor result

All four runs completed on distinct workers; no experiment job remains active.
All used the exact model overlay, BFP8 release policy and same prebuilt image.
Client wall includes agent/dependency setup and verification, excludes server
startup. These are cold-server task runs, not warmed microbenchmarks; initial
program/trace compilation is included in request timing.

| Request seed9472 | Cache | Reward | Agent min | Client wall min | Server requests | Input tokens | Cached tokens | Computed prefill | Output tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Present | Off | 1 | 50.71 | 51.73 | 93 | 2,204,852 | 0 | 2,204,852 | 63,417 |
| Present | On | 1 | 21.73 | 22.73 | 59 | 1,004,807 | 987,008 | 17,799 | 42,830 |
| Absent | Off | 0 | 40.47 | 41.48 | 77 | 2,098,765 | 0 | 2,098,765 | 36,459 |
| Absent | On | 1 | 11.70 | 12.70 | 35 | 521,676 | 512,064 | 9,612 | 21,260 |

| Seed/cache | TTFT sum seconds | Mean TTFT seconds | Decode seconds | Decode output tokens/s, excluding first token per request |
| --- | --- | --- | --- | --- |
| Present/off | 1776.696 | 19.104 | 1253.430 | 50.52 |
| Present/on | 451.250 | 7.648 | 842.080 | 50.79 |
| Absent/off | 1699.761 | 22.075 | 718.462 | 50.64 |
| Absent/on | 279.138 | 7.975 | 415.620 | 51.07 |

The cached trajectories were shorter and computed dramatically fewer prefill
tokens, but also used fewer requests/output tokens. Their total wall ratios
are not isolated cache-kernel speedups. The matched warmed synthetic tests
provide clean evidence: 127-token suffix caching loses at4K, wins1.42x at8K
and2.71x at16K. The separate shared-batch variant is slower than traced suffix
replay and is not the measured agentic candidate.

Both cached settings passed this one real task, including the no-request-seed
setting. That is evidence of feasibility, not sufficient evidence to remove
the fixed seed or claim broad accuracy. Cold-prefill numerical equality is
still not satisfied; caching follows the proven ordinary-decode contract.
Before promotion, carry `GEMMA4_PREFIX_TRACE_SUFFIX=1` explicitly in the
candidate profile, preserve exact source/image/dependency provenance, and
run the requested GPQA10/Terminal5/SWE5 gates through Shield main. Additional
unseeded trials are needed to assess variability. No production PR was
modified, no local tests/builds/servers/devices were run, and no final Shield
qualification was dispatched by this experiment.

Remote runs 39238204679 (qb2-120-p03t09) and 39238201191
(qb2-120-p01t03) passed concurrently on distinct machines, using Metal
7d5c3e5cb8438989e36a6a35c704f1154f9a4584 and workflow 38d4d6f.
Lifetime checks covered eviction/reconstruction, divergent suffixes and
three exact seeded device-sampling repeats. The raw continuation output
was whitespace, so this is state/repeatability evidence, not quality.

For growing conversations with 127 computed suffix tokens, three warmed
pair medians (cold wall -> cached wall seconds) were 2.109 -> 2.737 at
4,096 cached tokens, 3.891 -> 2.748 at 8,192, and 7.529 -> 2.781 at
16,384. Corresponding TTFT was 2.087 -> 2.712, 3.862 -> 2.720, and
7.489 -> 2.741 seconds. Counter assertions verified cached/computed tokens.
Full-vocabulary cached suffix logits equaled ordinary decode in artifacts;
cold-prefill numerical equality remains a known failure.

The next exploratory standard-Harbor experiment selects only Astropy14096
from the existing CI-nightly list on separate inference branches, preserving
the standard mini-swe-agent, timeouts and repetition guard. Four independent
arms vary caching and presence of request seed9472; GPQA42 and server9472
remain untouched. One task cannot qualify accuracy or justify seed removal.
All execution remains remote; no local validation/device process was run.

### Full-cohort release preparation (2026-10-19)

The clean inference candidate branch `mvasiljevic/gemma4-prefix-full5-release`
at `29f1648bff455a330bafdcaedcf7fd37178cb54d` starts from the original full
five-task configuration, retaining request seed9472. Relative to PR5315,
only the Gemma26 blocks in `workflows/model_specs/dev/llm.yaml` and
`reference_config/evals/eval_config.py` differ: prefix caching and traced
suffix replay are explicit, and SWE uses the existing `agent_env` interface
to supply `UV_OVERRIDE`. There is no `workflow_venvs.py` source patch in this
candidate. No production PR branch was changed.

The override URL is immutable Metal commit
`6b303feaa5f46aa8f4ef84db816104a04e840306`, model-owned file
`tests/harbor_dependency_overrides.txt`, requesting `litellm[mcp]==1.106.0`
and `mcp==2.2.0`. Upstream Harbor main and the current Tenstorrent Harbor
fork still use the original installer without MCP dependencies. The
supported uv override preserves that installer command; it changes only
the isolated task dependency resolution, not the agent or verifier.

Device-free fresh Astropy task-image preflight
[39255415691](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39255415691)
passed on `qb2-120-p05t02`: mini-swe-agent2.2.8, LiteLLM1.106.0, mcp2.2.0,
fastapi0.148.0, orjson3.13.0, httpx2 2.14.0; both the failing MCP handler
and CacheableResult imports succeeded. This preflight used a mounted
override file; the full Harbor run additionally validates immutable URL
loading and the existing per-agent environment propagation.

Full5 SWE run
[39255759103](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39255759103)
was dispatched with those exact Metal/inference refs and workflow `bddb4d6`.
It reuses the existing prebuilt image with the exact BFP8 model directory
mounted, caching/traced suffix enabled, and seed9472 retained. Its workflow
no longer exports the experimental Harbor source-patch switch. The job
timeout is 780 minutes to accommodate five sequential two-hour task limits;
task limits themselves were not changed. Results are pending, and this is
not final Shield qualification.

The full5 job on `qb2-120-p01t03` passed server startup and the explicit
nonzero prompt-token gate, entering `Await standard Harbor result and
collect counters`. Thus the clean standard Harbor path has reached actual
model inference using the URL override, without the source-replacement
hook. This establishes bootstrap success only; individual verifier rewards
and full-cohort completion remain pending.

Shield main's normal dispatch with an empty `docker-image` builds the
specified Metal/inference/plugin refs; supplying the old prebuilt image
does not overlay Metal. Final qualification therefore needs an exact-model
image (one build, subsequently reused). No build has been dispatched yet.
The tested plugin ref to retain is
`0a76258e14ca7237067ca9a22ec02e257bee570a`.

### Independent review and trace-lifetime follow-up

Independent source/artifact review returned **more-work-needed** before
release qualification. It confirmed repeated `LOGITS_TRACE_READY` /
`SPLIT_TRACE_READY` transitions in lifetime server logs: suffix logits mode
and device-sampled decode mode retire and recapture the model trace. The
adapter also reconfigures sampling at request boundaries in a way that
retires traces. Allocation warnings at traced-boundary server.log:491 and
lifetime server.log:641 require tracker-enabled evidence. The whitespace
raw-completion repeat test also needs a prompt-correct chat control.

Diagnostic control
[39258199619](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39258199619)
uses Metal `3d9f4f016e34c912f1c82039c56c96e59fb4ba6a`, new branch
`mvasiljevic/gemma4-prefix-trace-lifetime`, on `120-qb2-p05t01`.
It changes tests only, enabling allocation tracking with allocation
tracebacks and program-cache allocations included. The chat test renders
the instruct template, selects a partial-page suffix of 16–24 tokens,
checks meaningful output, repeats device-sampled requests, grows the
conversation, and includes an independent cold control.

Candidate
[39258540816](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39258540816)
uses Metal `3b8e197545fbd9d703a577ddfd8b5d12b84bf7da`, new branch
`mvasiljevic/gemma4-prefix-persistent-trace`, with the same tracked check.
This model-only variant moves seed advancement from model replay into
feedback sampling, retains the model trace for logits-only suffixes,
prepares suffix output storage before capture, avoids an unused page-table
upload, and retains compatible adapter sampling state across cached
requests. It adds no broad allocation exemptions. Trace IDs and capture /
release counters are logged. Both results remain pending; these changes
are not promoted to the clean full5 candidate. The current full5 control
continues unchanged, and final image construction is held pending the
safety and repeatability evidence.

Control run 39258199619 passed 16 host checks and the tracked chat test
(90.486 seconds including first-request compilation). Rendered prompt length
was 80 tokens: 64 cached, 16 computed on repeats. The cold answer was a
coherent 31-token explanation of compiler name lookup; the cached answer
was a coherent 34-token explanation and matched three subsequent sampled
repeats exactly. A 138-token growing conversation produced a sensible
52-token example (`userAge`, integer type, local scope, memory address).
The independent cold control matched the original cold answer. No unsafe
allocation error occurred with tracking enabled and program-cache buffers
included. Logs still show two captures per cached sampled request, so this
does not close the recapture finding. Prior warning artifacts also had
eager numerical diagnostics enabled, unlike this serving-lifetime check.

Variant run 39258540816 passed all 17 host checks, then failed before model
initialization on `qb2-120-p02t02`: physical chips 2–3 exposed one Ethernet
channel while the existing fabric descriptor requires two (`server.log:83`,
`:110`). This is an infrastructure failure, not a tracker or model result.
No manual reset, topology reduction, or blind repeated retry was performed.
The runner-admin startup hook had already run its standard reset.
Source review found a separate diagnostic-only `table=None` issue from lazy
upload; commit `400585ae5a90787e5135a9bb139e66cdca32dfc6` supplies the bound
table and retires traces before eager numerical probes compile other paths.
Production diagnostics remain disabled; the tracecheck suite is unaffected.

Persistent variant run [39260126298](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39260126298)
passed 17 host checks and the tracked chat test on `120-qb2-p05t01`, the
same worker as the control. Metal was `400585ae5a90787e5135a9bb139e66cdca32dfc6`.
All seven answer texts matched the control. Cached repeats and the growing
conversation retained `MeshTraceId(0)` with one capture and zero releases.
The independent cold request still retired that trace and produced a second
capture; this is not generator-wide trace-lifetime compliance. The tracked
test took 87.427 seconds including compilation: comparison with the control's
90.486 seconds is not a performance claim.

Two independent follow-ups use the same existing image and workflow `6e415f0`:

- [39261166389](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39261166389),
  Metal `f20185936091f47452add16ab7c8ad26d8df4163`, on `qb2-120-p03t09`:
  extended allocation-tracked chat and host-sampling suffix lengths 1–7.
- [39261174410](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/39261174410),
  Metal `d92bc6e087a5003c5f11d61cac79cd5bf9deb963`, on `120-qb2-p05t01`:
  tracking-disabled old-implementation timing control with per-request
  wall, TTFT, cache, computed-token and output counters. A matching variant
  timing run follows correctness acceptance.

The full5 standard Harbor control `39255759103` remains active on
`qb2-120-p01t03` after its real-inference gate. These jobs do not share a
physical worker. The known fabric-fault worker `qb2-120-p02t02` is refused
before model access. A scheduling-only attempt `39259454295` was cancelled
without obtaining a runner; it executed no model work.

