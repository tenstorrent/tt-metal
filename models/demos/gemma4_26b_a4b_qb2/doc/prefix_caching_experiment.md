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
