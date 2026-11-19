# Unapplied serving-overhead proposal

Prepared source-only, 2026-11-17. Production source files and live processes were not changed. No hardware or HTTP execution occurred, and no latency improvement is claimed.

The per-step `QWEN_DECODE sampling_mode=...` line is required evidence even when a worker dies before teardown. It is preserved byte-for-byte by this proposal. This corrects the transition-only logging suggestion in `SERVING_OVERHEAD_AUDIT.md`; that suggestion must not be applied.

## Patch

`serving_overhead.patch` is a unified patch against the current working-tree files, built by `build_overhead_patch.py`. `git apply --check` passes; the patch has **not** been applied.

- `tt/serving_state.py`: introduces an immutable, validated internal permutation tuple. External list/tensor/ordinary-tuple payloads retain boundary validation; the same validated object passes through the command decoder, reload executor and state owner without repeated CPU tensor conversion/sort. Capacity mismatches still reject. No remap, token, position, sampling or device operation changes.
- `tt/serving_state.py` and `tt/generator.py`: add the model-local optional `events_synchronized=False` processing argument. The default retains generator-owned event waiting. Published synchronous outputs still perform their blocking read.
- `tt/generator_vllm.py`: the existing plugin processing hook passes `events_synchronized=True`, because pinned `docs/MODEL_EXECUTION.md:175` and `src/vllm_tt_plugin/async_decode.py:1653`–`:1677` require the plugin to synchronize returned read events before invoking it. Direct adapter callers must obey that same documented sequence; the existing hardware command probe already does. This removes the second wait on an already-completed event, not the transfer or its first completion wait.
- `tests/run_serving_commands_cpu.py`: includes the new internal permutation class in its existing AST-only loader. No shared plugin, vLLM, capability, registration or default scheduling interface changes.

The patch does not alter per-step logs, sampling arithmetic, canonical trace contents, output snapshot copying, pool depth, cache identity, reload commands, or trace selection. It is a narrow model-owned reduction of redundant host work. The second event wait is already complete, and the permutation has at most32 slots; source evidence does not support assigning the historical B1/B2 throughput gap to these changes.

CPU verification, already run:

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/build_overhead_patch.py
git apply --check models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/serving_overhead.patch
python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/test_overhead_patch.py
```

Six tests pass against the proposed source **in memory**: one external remap conversion across repeated internal normalization; capacity/malformed input rejection; two pending outputs and independent returned storage; default generator-owned event waits; exactly one wait at the actual adapter post-plugin boundary; synchronous blocking-read preservation; existing stale-token/current-position/page-only command suite; unchanged per-step ledger AST. These test cases are grouped into six unittest methods. Python compilation, Black with Python3.12 target, and probe `--help` also pass. The patch builder intentionally rejects a source drift rather than silently applying fuzzy edits.

## Matched current-source B2 probe

`run_matched_b2_overhead.py` is a standalone diagnostic entrypoint stored here, not a production module. It can run either before or after applying the patch; each artifact hashes every model source plus the repaired native fused reader. It uses the real `QwenForCausalLM` factory and external cache allocation hook, full40 layers, selected precision, capacity32, context262144 and retained B2/B32 traces.

It parses exact cache specifications from a serving log. The existing final server recorded **ten BF16 pools of `[8224,1,32,256]` per chip**, whereas a standalone default allocation would use8192 pages. A CPU-only parser check confirmed these exact ten serving specifications, full40 and max-num-seqs32/max-model-len262144. No TTNN import was needed for this check.

All modes share the same model, cache, sampler, fixture and retained traces, resetting and prefilling between requests:

1. `native_serial`: direct generator token-out, synchronous token observation each step.
2. `adapter_serial`: actual adapter v1 commands, owned output snapshot, asynchronous read followed immediately by the documented event wait/process hook.
3. `adapter_async2`: the same adapter path with up to two outstanding leased outputs, consumed in submission order.

Each request has128 raw input IDs and128 greedy outputs with EOS ignored. Prefill is excluded; exactly127 decode steps are timed. One full warmup request per mode is excluded; two measured repetitions per mode are the default. Every output sequence must match exactly, every timed model replay must be `model_2`, every decode must sample on device, retained traces must stay unchanged, and only the first authoritative decode may refresh tokens. The generated fixture is saved next to the result for an identical later HTTP request. Native mode does not feed observed tokens back during resident steps; adapter resident calls use unreadable stale-input sentinels.

Owner-run command after exclusive hardware ownership becomes available, from repository root in the lane environment:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN_VLLM_AUDIT=0 QWEN_VLLM_REDUCED=0 timeout 1800 \
  python models/autoports/qwen_qwen3_6_35b_a3b/doc/vllm_integration/sampling_debug/run_matched_b2_overhead.py \
  --server-log models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/final_server/server.log \
  --repetitions 2 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/serving_overhead/baseline.json
```

Use a distinct output path after applying the candidate, and `--prompt-json .../baseline.prompt.json` to reuse the exact input. The probe does not apply the patch or modify source. Normal cleanup releases traces before closing the mesh; an actual hang still requires the owner's external timeout, evidence preservation and cleanup workflow.

`adapter_async2` elapsed/127 is queued throughput, not serial token latency; its per-output timing is submission-to-consumption latency. Keep those fields separate from the serial measurements and from HTTP TPOT. The probe reproduces served capacity/context/cache geometry, but not allocator history, concurrent request admission, HTTP or network overhead. One later idle-server request using the saved128-token fixture can quantify that remaining layer. This is not a B1 substitution, maximum-context acceptance rerun, or a performance acceptance result until actually executed.
