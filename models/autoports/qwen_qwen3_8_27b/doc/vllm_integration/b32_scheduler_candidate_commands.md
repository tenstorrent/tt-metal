# B32 scheduler configuration — lifecycle validated

The scheduler budget/threshold `131072/4096` passed the full64 B32 lifecycle: `readiness_vllm/full_b32_budget131072_lifecycle.json`. Cold32 ×4130 time improved524.390→33.763 seconds with stable traces and99.1768% prior-token reuse. The root agent executed the server validation; this analysis only read completed artifacts. A tracking-disabled performance benchmark is separate. Pinned vLLM0.30 has no `max_num_partial_prefills` or `max_long_partial_prefills` flags. The commands below document reproduction; the hardware owner must first finish/stop any current server and audit device ownership.

Launch from the repository root, retaining the foreground process for cleanup. Use fresh artifact filenames; do not erase prior evidence. The environment wrapper supplies existing device/fabric setup and allocation tracking. Explicitly clear reduced-layer and host-sampling overrides.

```bash
docker exec -w /workspace qwen38-functional-build \
  env -u QWEN38_SERVING_LAYERS -u QWEN38_HOST_SAMPLING \
  MESH_DEVICE=P300x2 \
  QWEN38_CHECKPOINT=/workspace/.cache/qwen38-full \
  QWEN38_SERVING_EVENTS=/workspace/models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_events.jsonl \
  TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=models.autoports.qwen_qwen3_8_27b.tt.generator_vllm:Qwen38ForCausalLM \
  bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh \
  /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/.cache/qwen38-full --port 8000 \
  --block-size 4096 --max-num-seqs 32 --max-model-len 262144 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}}' \
  --language-model-only --mamba-cache-mode align --mamba-block-size 4096 \
  --max-num-batched-tokens 131072 --long-prefill-token-threshold 4096 \
  --async-scheduling \
  > models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_server.log 2>&1
```

After the server is ready and its `prepared` event records all 64 layers, B32, selected precision and four traces, run the current HTTP-only lifecycle runner against that same server. The current CLI defaults multi-turn prompts to 4130 even without `--quick`; spelling out the length prevents ambiguity. Do not pass `--reduced`, `--quick` or `--skip-agent-turns` for this comparison.

```bash
docker exec -w /workspace qwen38-functional-build /opt/venv/bin/python \
  models/autoports/qwen_qwen3_8_27b/tests/run_serving_lifecycle.py \
  --server-url http://localhost:8000 --model /workspace/.cache/qwen38-full \
  --max-concurrency 32 --agent-turn-length 4130 \
  --events-path models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_events.jsonl \
  --output models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_lifecycle.json
```

Require pass, unchanged capture/retirement counts and PID, actual nonzero chunk starts, concurrent decode evidence, exact completion counts, prefix reuse and stable greedy repeats. Compare cold-wave adapter row counts, prefill host spans, waiting requests and SSE pauses with the old 8192-budget run. A larger budget should coalesce compatible requests; it does not make different offsets/finalness compatible automatically. Record allocator checkpoints and inspect runtime errors before any performance claim. This lifecycle is synthetic correctness/latency evidence, not the headline 4096/128/B1 benchmark or qualitative evidence.

The larger planning workload can be exercised only as a separate explicitly labeled pool-pressure run on the same stable server:

```bash
docker exec -w /workspace qwen38-functional-build /opt/venv/bin/python \
  models/autoports/qwen_qwen3_8_27b/tests/run_serving_lifecycle.py \
  --server-url http://localhost:8000 --model /workspace/.cache/qwen38-full \
  --max-concurrency 32 --agent-turn-length 32769 \
  --events-path models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_events.jsonl \
  --output models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_budget131072_pool_pressure.json
```

The extra token permits the previous 32768-token aligned boundary to be reused. The configured 289-ID pool is expected to fall short of retaining all 32 such prefixes; report measured admissions, reuse and preemption/eviction evidence. A predicted shortfall is not a completed larger-workload measurement. Preserve final server command/config, then let the hardware owner stop the server and audit EngineCore/device cleanup.
