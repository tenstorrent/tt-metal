# AutoFix: capture public-token formatting

## Starting evidence

`AUTODEBUG.md` identifies an eager public-token slice after every pair of
nonblocking model/sampler trace replays. Existing persistent output tensors
already provide the correct asynchronous ownership boundary. The inherited
serving/standalone throughput difference does not establish causation.

## Hypothesis experiment

Hypothesis: the public copy can run after canonical sampling inside the existing
sampler trace, eliminating eager dispatch without changing public tensor identity,
batch shape, canonical sampling, seeds, penalties, dtype, or trace count.

Before changing implementation, added and ran:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q -k decode_formats_public_tokens --disable-warnings --tb=short
```

Result: one failed, one passed. Device sampling performed one eager formatting
call where the target contract requires zero; host sampling correctly retained
one call. This proves the source-level extra dispatch, not its throughput cost.

## Change

`tt/generator.py` now constructs `_Gemma4SamplingGenerator`, a local subclass
which delegates sampling and penalty bookkeeping to `SamplingGenerator` and
then formats only its explicitly bound feedback token tensor. Binding stores
the feedback and public tensors directly; no generator reference or bound
generator callback creates a reference cycle. `_bind` installs the new binding
after releasing prior traces and creating the persistent public view.

The existing exact-shape warmup is retained. The common sampler's existing
`capture_trace` records the appended slice. `decode_forward` formats eagerly
only for host sampling, whose replay bypasses the device sampler. There is no
third trace. Serving prefill with no bound output remains untouched; standalone
prefill explicitly writing the bound token tensor also performs the public copy.
The latter adds one eager copy at prefill, outside the steady decode loop.

No edits were made to the adapter, plugin, page-table refresh policy, common
sampler, or precision policy.

## Verification

Focused checks after implementation:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q -k 'decode_formats_public_tokens or sampler_trace_formats or sampler_prefill_formats or logits_decode_refreshes' --disable-warnings --tb=short
```

Result: six passed at that point. The final suite additionally covers an
unrelated output binding, which must not format the public tensor.

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short > models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/host_sampling_contract.log 2>&1
python -m black --check models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py
git diff --check
```

Result: **84 host tests passed** in 36.33 seconds; Black and diff whitespace
checks passed. Black emitted its existing Python 3.10/target-3.14 safety-check
warning. `python -m isort --check-only` could not run because isort is not
installed. This is a Python-only change, so no C++ build is required.

The capture-order test exercises the common `capture_trace` and `_run_sampling`
methods with mocked device operations, verifying:
`begin -> apply penalties -> sample -> count token -> public copy -> end`.
Replay invokes only the existing nonblocking trace, returns the original
sampler result, and performs no eager public copy. Other regressions retain
device/host mode behavior, unchanged/changed page-table paths, seed restoration,
resumed prefill, and logical/padded row handling.

## Final status

The source-level dispatch hypothesis is verified and the bounded candidate
passes host contracts. Hardware correctness and performance remain unverified
in this subtask. Parent-owned checks must prove repeated/queued async reads,
persistent tensor identity, stale host token/position feedback, unchanged and
changed pages, nonaligned prompts, and the same-configuration serving result.
Do not retain or claim a throughput improvement without that evidence.

No hardware, server, reset, or profiler commands were run by this investigator.

## Queued-read regression follow-up

`tests/check_vllm_adapter.py` now extends the existing token chain from six to
nine tokens. Two decode/read pairs are submitted before either event is waited
or either output is converted on host. Each result is checked against its
standalone token, the two host tensors have independent storage, and feedback,
position, public-output, page-table tensor identities and refresh counts remain
unchanged across the queued pair.

A subsequent decode rebind moves the same request from slot 0 to slot 1, with
slot 0 inactive and the original cache mapping retained. This changes the
logical public output from one to two rows without introducing another live
request or moving KV data. The old host tensors are reprocessed after the
rebind and must still return their original one-row outputs and values.

The original page-table change swaps unused logical columns 5 and 6. The test
and report now state this scope explicitly: it verifies refresh and trace
retention, while the serving workload covers live allocator page growth.

Test-only checks run:

```bash
python -m black --check models/autoports/google_gemma_4_26b_a4b_it/tests/check_vllm_adapter.py
python -m py_compile models/autoports/google_gemma_4_26b_a4b_it/tests/check_vllm_adapter.py
git diff --check
```

All passed. The parent owns execution of the extended device regression after
the candidate server exits. Model implementation was left frozen throughout
this test-only follow-up.

## Parent hardware and serving verification

Reduced stale-input/allocation tracking and queued-read/rebind probes pass;
scoped compute Watcher multirow probe passes (ETH instrumentation limitation
recorded in anomaly_ledger.md). Full serving requests match prior controls;
shared full sampling72passed/1documentedskip, qualitative and both benchmarks
pass. Decode is effectively flat; no first-use throughput speedup is claimed.
Final evidence: final_validation.json, comparison.json, adapter_queued_reads.json.
