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

