# Optimized vLLM work log

Started from vLLM integration commit `a871e282fbfee748cb07c12750eed42c6b45e7a7`
and datatype-sweep checkpoint `a60cab960bc02bba4b9f5c846a7819ee25a42c41`.
Plugin registration checkpoint: `938c45ed71f3f669ffd38e4c9a033c3391cec961`.

## Commands

Primary and CI before/after harness:

```text
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' python -m models.common.readiness_check.run_vllm_server --stages serve,benchmark --model-dir models/autoports/google_gemma_4_26b_a4b_it --hf-model google/gemma-4-26B-A4B-it --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --tt-config '{"trace_region_size": 220000000, "fabric_config": "FABRIC_1D_RING"}' --benchmark-prompt-len 128 --benchmark-output-len 128 --benchmark-num-requests 1 --benchmark-concurrency 1 --ci-benchmark-prompt-len 100 --ci-benchmark-output-len 100 --ci-benchmark-num-requests 32
```

For the warmed primary comparison, the server was held with `--stages serve`.
The same command was then run twice with `--stages benchmark --server-url
http://127.0.0.1:8000 --no-benchmark-ci-serving`; run one was the explicit
128/128/1 warmup and run two was preserved as the measurement. The baseline
server ran from detached worktree `/tmp/gemma4-vllm-baseline` at `d9980cd2a43`;
the optimized server ran from this checkout. All other launch and benchmark
arguments above were unchanged.

Final sampling and qualitative:

```text
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' python -m models.common.readiness_check.run_vllm_server --stages serve,sampling,qualitative --model-dir models/autoports/google_gemma_4_26b_a4b_it --hf-model google/gemma-4-26B-A4B-it --mesh-device P300x2 --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --tt-config '{"trace_region_size": 220000000, "fabric_config": "FABRIC_1D_RING"}'
```

Async overlap against a held server with the same launch configuration:

```text
python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_async_overlap.py --model google/gemma-4-26B-A4B-it --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/async_overlap_state_test.json
```

Direct persistent trace-state and deferred-read probe:

```text
GEMMA4_MIXED_PROBE=1 GEMMA4_ASYNC_STATE_OUTPUT=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/after/async_trace_state_test.json pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_mixed_prompt_and_inactive_slot_probe -s
```

Degeneracy gate:

```text
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/google_gemma_4_26b_a4b_it --scope vllm --missing-artifacts critical --json models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/after/qualitative_degeneracy.json
```

## Decisions and results

1. Baseline artifacts were copied unchanged into `before/`: primary 45.9175 ms
   mean TPOT (21.7782 t/s/u), 20.1111 tok/s aggregate; CI burst 69.5707 tok/s.
2. Rejected retaining `force_argmax_on_sharded_logits`: it violates the serving
   sampler contract and its full-vocabulary gather/argmax dominated prior
   token-out evidence.
3. Rejected the existing generic split TopK: prior exact-shape evidence measured
   10.7359 ms versus 2.3434 ms for force-argmax because a 65,536-wide local
   vocabulary selected the slow TopK factory.
4. Ported the proven two-stage structure into `Sampling1D`: two 32,768-wide
   multi-core TopKs per TP shard, chunk-relative ID restoration, a 64-to-32
   local candidate merge, candidate-only all-gather, and semantic greedy
   sampling. Gemma's vocabulary is exactly 262,144, so there are no invalid
   padded vocabulary IDs to mask.
5. Disabled `allow_force_argmax`, made greedy requests materialize explicit
   semantic k/p/temperature/seed tensors, and updated the selected policy and
   model validation string. The adapter remains thin; its only stage change is
   a page-table refresh counter used by the direct contract probe.
6. Explicitly warmed primary, before: TTFT 214.35 ms P50/P99, mean/P99 TPOT
   46.4824 ms, ITL 44.0706/49.8756 ms P50/P99, aggregate 20.9222 tok/s, and
   21.5135 TPOT-derived t/s/u for 128/128/1 concurrency 1. After: TTFT 201.59
   ms, TPOT 35.6679 ms, ITL 33.9054/34.6876 ms, aggregate 27.0512 tok/s, and
   28.0364 t/s/u. Decode improves 30.3% and TTFT improves 6.0%. First-measured
   runs remain as separate artifacts and are not used as the headline.
7. Final CI: 32/32 complete; TTFT 4270.71/6178.79 ms P50/P99, mean/P99 TPOT
   401.6145/435.2772 ms, ITL 359.0662/1294.9163 ms P50/P99, aggregate 71.8121
   tok/s for 100/100/32 burst.
8. Full sampling passed 72/72 runnable tests with one skip. Six correct
   chat-template qualitative pairs passed manual inspection and the machine
   degeneracy gate.
9. Async overlap proved vLLM scheduling overlap with byte-identical controls.
   A separate focused adapter/generator hardware probe supplied stale token 0
   and position 999 and directly recorded: token/position refreshes 0, aliased
   persistent feedback/input addresses, position 33→34→35, adapter page-table
   copies 1 initial / still 1 unchanged / 2 after mutation / still 2 on reuse,
   stable device addresses with exact mutated contents, and one deferred-read
   event synchronized before formatting. This closes the review findings.
10. Full-model comparison: optimized vLLM 28.0364 t/s/u versus optimized
    full-model host-visible 26.2128 t/s/u and no-host token-out 28.0151 t/s/u,
    all batch 1 with 128 prompt / 128 generated where applicable.

## Hardware recovery

After multiple clean server shutdowns, the next immediate mesh open failed before
model code with `Device 0: Timed out while waiting for active ethernet core
29-25 to become active again`. In both cases:

```text
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

returned all four P300C devices, and a 1x4 mesh open/close passed before work
resumed. No profiler or watcher was active. Final held-server shutdown was clean,
no vLLM/EngineCore process remained, and `tt-smi -ls --local` showed four chips.

## Artifacts

- `before/vllm_{result,benchmark}.json`
- `before/vllm_{warmed,first_measured}_{result,benchmark}.json`
- `before/vllm_ci_serving_{result,benchmark}.json`
- `after/vllm_{result,benchmark}.json`
- `after/vllm_{warmed,first_measured}_{result,benchmark}.json`
- `after/vllm_ci_serving_{result,benchmark}.json`
- `after/sampling_tests.log`
- `after/vllm_qualitative_outputs.json`
- `after/qualitative_degeneracy.json`
- `after/six_prompt_control_comparison.json`
- `after/async_overlap_state_test.json`
- `after/async_trace_state_test.json`
- `readiness_vllm/server.log` and benchmark logs

Stage review and final commit SHA are appended after independent closure.
