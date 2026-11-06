# Repair: use stock direct benchmark launcher with runtime override

The previous top-level custom-weights command was insufficiently validated: its parser/resolver passed but its child parser failed. Source diagnosis is in tokenizer-AUTODEBUG.md. No generic source changed.

Supported command from TTI bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb:

```bash
ONLY_BENCHMARK_TARGETS=1 timeout 900 python launchers/run_llm_bench.py \
  --model google/gemma-4-31B-it --workflow benchmarks --device p300x2 \
  --service-port 8000 --server-url http://127.0.0.1:8000 --tools vllm \
  --runtime-model-spec-json /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-local-tti-preflight/workflow_logs/runtime_model_specs/runtime_model_spec_2026-11-06_05-40-28_id_gemma4-31b-autoport-v3_842da3794eaa0b77d5f08bae87a17459d91ff475_p300x2_svrlIeuM.json \
  --output-dir /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-local-tti-preflight/stock-direct-launcher
```

Host validation: run_workflows.parse_args accepted the exact canonical model/device/workflow/runtime spec combination; _load_model_spec_override returned the pinned absolute snapshot identity; get_llm_configs returned only128/128/C1/prompts8. This invokes the same stock launcher/engine/vLLM client and reads the same emitted runtime spec as the failed attempt, using its documented runtime override precedence. Coordinator alone launches real client and records live result; no endpoint accessed by this subagent. Remaining uncertainty: live client reaches server and tokenizer library accepts snapshot. Previous command failure preserved unchanged.
