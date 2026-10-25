# Stage 09 AutoFix closure evidence

This records targeted repairs following `AUTODEBUG.md`; it is not a full-server
readiness verdict. Reduced-model proofs below do not establish full-model quality,
headline performance, or the final sampling/lifecycle gates.

## Repairs and focused evidence

- Continuing prefill state ownership: the plugin allocator now reserves existing
  owners before assigning new requests. Finished peers, reordered continuations,
  and new-before-existing requests no longer overwrite continuing state. Three
  regression cases fail on the original allocator; 74 focused tests pass after
  repair (`readiness_vllm/plugin_state_before.log`, `plugin_state_after.log`).
  Details and commands: `plugin_state_autofix.md`.
- Generator/adapter resident decode contract: stale token/current position,
  changed page table, delayed asynchronous read and slot-remap checks pass on the
  reduced two-layer generator and adapter. Exact cache/token comparisons and
  unchanged capture counts are in `generator_device_contract.json` and
  `adapter_device_contract.json`; the adapter reports zero host-logit reads.
  Details: `generator_autofix.md`. This is direct adapter proof, not proof of every
  scheduler transition on the final server.
- Scheduler-owned hybrid cache: the plugin accepts native MambaSpec and preserves
  raw layer indexing; recurrent snapshots use the allocated cache tensors.
  `mamba_cache_bridge.md` records sizing, focused tests and physical-byte caveats.
- Mixed intermediate/final prefill: the plugin submits intermediate rows separately,
  without sampling parameters, and final rows with their selected sampling route.
  Original state slots, per-layer page tables, start/end positions and sampling
  parameters remain aligned. Returned final tokens/logprobs are scattered back
  into packed request order. Intermediate outputs are ignored for legacy models;
  this adapter skips both sampling and logits readback. No new protocol capability,
  environment variable or tuning option is required. Optional final host sampling
  still requires explicit `QWEN38_HOST_SAMPLING=1` and is excluded from performance.
  Focused plugin tests pass 133 cases (one optional vllm_omni serialization skip),
  and five adapter host tests pass. Logs: `intermediate_prefill_plugin_tests.log`,
  `intermediate_prefill_host_tests.log`; plugin lint passes in
  `intermediate_prefill_precommit.log`.
- MRoPE output attribution now uses submitted request IDs, preserving concurrent
  decode peers and reordered prefill subsets; a focused host regression passes.
- Final-prefill RNG reset and sampling now pass only the actual final-group slots
  into the generator-owned subset API. Intermediate groups invoke neither API;
  exact peer-counter preservation requires the generator device test evidence.
- Trace instrumentation: generator-owned preparation events identify actual trace
  IDs and execution signatures, with host preparation/warm/capture elapsed time.
  Failed attempts remain marked failed. These are not device-profile latencies.
  Details: `preparation_events.md`; host checks:
  `preparation_events_host_check.json`.

All bare evidence filenames above are under the model's `readiness_vllm/`.
The adapter host command was:

```sh
docker exec -w /workspace -e USER=mvasiljev -e PYTHONPATH=/workspace:/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/runtime qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=models/autoports/qwen_qwen3_8_27b/tests --rootdir=models/autoports/qwen_qwen3_8_27b models/autoports/qwen_qwen3_8_27b/tests/test_serving_prefill_contract.py -q
```

The plugin test command was:

```sh
docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PYTHONPATH=tests/host-stubs qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=. --rootdir=. tests/test_prefill_output_suppression.py tests/test_model_runner.py tests/test_lane_model_runner.py tests/test_lane_input_batch.py tests/test_seeded_sampling_policy.py tests/test_stage_inputs.py -q
```

## Runtime audit and outstanding proof

The selected plugin version compares current page tables against cloned last
submitted tables, reloads changed tables, and invalidates residency across prefill.
It drains pending asynchronous work at mode transitions. Direct reduced adapter
checks exercise stale tokens and page changes; scheduler-driven multi-page growth
and the final full-layer serving run must still supply their own evidence.

`reduced_lifecycle.json` records 22 requests, actual chunk starts including 0,
4096, 8192 and 12288, concurrent decode, four captures before/after and zero trace
retirements. Multi-turn reuse is 8192/8260 tokens with recurrent saves/restores.
`reduced_sampling_tests.log` records three passing tests and one skipped all-vocab
logprob request (248320 exceeds server max20). These reduced runs predate the final
mixed-prefill repair and cannot substitute for its running-server validation.

Per-token JSONL event logging must be disabled for headline performance. Page-table
comparison remains required by the resident-input contract. Repeated per-layer
padding was confirmed and removed: each distinct cache group is padded once per
submission and shared across its mapped layers, matching the existing read-only
aliasing contract for already-shaped tables. No device profiler was used for these serving repairs.
Final full-server sampling, qualitative judgment, mixed-shape reuse, single-user
4096/128/B1 measurements, CI burst, process cleanup and independent clean-pass
remain stage-owned gates; their absence here is not a waiver.

## Page-table padding overhead repair

A host allocation-count regression failed twice on the original helper (padding
and cropping cases: 64 allocations instead of four); exact-shape and non-hybrid
cases already passed. The repaired helper preserves values, zeros, dtype, None
entries and original input contents, with one output tensor per used cache group.
The focused helper/model-runner/split-prefill/hybrid-cache suite passes all 42 cases;
pre-commit passes. Artifacts: `page_table_padding_before.log`,
`page_table_padding_after.log`, `page_table_padding_precommit.log`.

`page_table_padding_microbenchmark.json` records an AST-extracted before/after
helper comparison, 100 warmups followed by seven repeats of 1000 calls, median
`perf_counter_ns` host wall time per call. Four groups map to 64 layers, input
width 3 and target width 9. Batch 1: 254.205 to 16.879 microseconds; batch 32: 266.181
to 16.566 microseconds. Allocations fall 64 to 4 in both cases. These numbers are
synthetic host-helper timings, not device latencies or vLLM throughput, and do
not establish the size of a full-server improvement. The current running server
can retain its previously imported helper; subsequent performance servers must
load the repaired source. No device was opened.

Validation command:

```sh
docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PYTHONPATH=tests/host-stubs qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=. --rootdir=. tests/test_block_table_padding.py tests/test_model_runner.py tests/test_prefill_output_suppression.py tests/test_mamba_cache_bridge.py -q
```

## Synchronous output finalization repair

The B8 synchronous server exposed a distinct finalization route: the plugin passes
raw replicated device token output directly to `process_decode_output_host`.
Converting that multi-device tensor with `ttnn.to_torch` requires a mesh composer
and failed. The adapter now detects device storage and invokes its existing
single-replica sampled-token read before conversion. Already-read async host
outputs incur no extra read; plain torch outputs are simply formatted. This does
not introduce host sampling, logits readback or a token-feedback loop.

Three host regression cases pass (`output_finalization_host_tests.log`): raw
output reads one replica synchronously and increments the readback counter once;
async host finalization performs no read; torch output uses no TT conversion.
`probe_serving_adapter_contract.py` now also checks direct-device versus async-host
versus last-token equality and exact readback deltas on its reduced device run.
Root executed the extended reduced hardware probe successfully; see `adapter_device_contract.json` and `adapter_sync_finalization_device.log`. Full64/B8 synchronous allocator-growth control also passed (`page_growth_sync_control.json`): three concurrent requests crossed both4096/8192 boundaries with stable captures and no host-logit reads. Matching asynchronous comparison remains pending.

```sh
docker exec -w /workspace -e USER=mvasiljev qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m pytest models/autoports/qwen_qwen3_8_27b/tests/test_serving_output_contract.py --rootdir=/workspace/models/autoports/qwen_qwen3_8_27b/tests --confcutdir=/workspace/models/autoports/qwen_qwen3_8_27b/tests -q
```
