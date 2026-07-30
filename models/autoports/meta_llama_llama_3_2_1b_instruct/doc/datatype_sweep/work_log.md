# Datatype Sweep Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.2-1B-Instruct`

Mesh: T3K `1x8`, `FABRIC_1D_RING`

## Thresholds

- Top-1 >= 90%.
- Top-5 >= 98%.
- Top-100 recorded, not separately gated.

## Baseline Refresh

Prefill command:

```bash
python -m models.common.readiness_check.run_prefill_check --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 88/100, top5 100/100, top100 100/100.

Teacher-forcing command:

```bash
python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 86/100, top5 100/100, top100 100/100, TTFT 245.25 ms, decode 98.50 t/s/u.

## Candidate Commands

Each candidate used:

```bash
python models/autoports/meta_llama_llama_3_2_1b_instruct/tools/run_datatype_candidate.py \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt \
  --config-id <config_id> \
  --precision-config models/autoports/meta_llama_llama_3_2_1b_instruct/doc/datatype_sweep/configs/<config_id>.json \
  --output models/autoports/meta_llama_llama_3_2_1b_instruct/doc/datatype_sweep/runs/<config_id>/result.json \
  --mesh-device T3K --fabric-config FABRIC_1D_RING
```

`cfg08` was selected because it is the fastest passing trace-verified teacher-forcing result: top1 96/100, top5 100/100, top100 100/100, TTFT 183.03 ms, decode 92.68 t/s/u.

## Post-Selection Token-Out

Command:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/datatype_sweep/post_selection_token_out \
MD_OPT_FULL_MODEL_PROMPT_LEN=60 \
MD_OPT_FULL_MODEL_DECODE_STEPS=128 \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_token_out_no_readback_benchmark
```

Result: selected config `cfg08_bfp8_weights_bfp8_kv_bf16_ccl`, TTFT 1407.02 ms, decode 146.75 t/s/u, mean latency 6.8143 ms/token. Measured loop had 0 readbacks and 0 token refreshes.

## Propagation Check

`propagation_check.json` records:

- `build_generator` default path matched `selected_precision_config.json` in the post-selection token-out artifact.
- `Llama32FullModel.from_pretrained` default resolver matched the selected policy for the autoport `.ttnn_cache` path.
- No vLLM adapter exists yet; no vLLM work was started.

## Exact Artifacts

- `README.md`
- `work_log.md`
- `sweep_results.json`
- `sweep_results.csv`
- `selected_precision_config.json`
- `top1_perf_pareto.png`
- `top5_perf_pareto.png`
- `candidate_manifest.json`
- `configs/*.json`
- `runs/*/result.json`
- `runs/*/run.log`
- `post_selection_token_out/token_out_no_readback_perf.json`
- `post_selection_token_out/run.log`
- `propagation_check.json`
