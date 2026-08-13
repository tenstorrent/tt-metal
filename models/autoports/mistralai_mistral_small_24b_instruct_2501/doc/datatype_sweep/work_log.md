# Datatype sweep work log

## 2026-08-13

- Confirmed four local Blackhole p300c devices, firmware 19.9.0, and a healthy 1x4 mesh before starting serialized hardware work.
- Refreshed the main AIME24 chat-template 100-token optimized-full-model baseline: prefill 99/100 top-1 and 100/100 top-5/top-100; traced decode 97/100, 100/100, 100/100 at 52.9074 t/s/u.
- Added `tt/precision_policy.py` and wired the complete policy through generator, full-model, decoder, cache, collectives, program configs, LM head, and sampling. Added a runtime summary and static completeness test.
- Proved policy propagation with a one-layer smoke, including a 7-token non-aligned prompt.
- Evaluated seven full-model policies. Adapted BFP8 MLP policies to legal block-8 geometry after block-16 exceeded L1, then measured both BFP8 LoFi and HiFi2 rather than recording a blocker.
- Repeated the policy-backed BFP4/LoFi/BFP8-KV/BFP8-CCL baseline: median 229.367625 ms TTFT and 52.829349 traced t/s/u. Repeated the BF16-CCL candidate: median 226.993467 ms TTFT and 54.212248 traced t/s/u. Both were 97/100 top-1 and 100/100 top-5/top-100.
- Selected `bfp4_lofi_bfp8kv_bf16ccl`, the fastest passing trace-ranked candidate. Installed it as `selected_precision_config.json`, which `build_generator` consumes automatically in the absence of an environment override.
- Recomputed KV capacity. BFP8_B retains the 32,768-token contract (34,464 calculated ceiling); BF16 KV is rejected at an 18,304-token ceiling. Updated `doc/context_contract.json` for the selected BF16 decode CCL payload without reducing context.
- Re-ran default-path non-aligned prefill and traced decode with no precision environment variable; the constructed runtime reported the selected policy and passed.
- Ran the normal 40-layer warmed token-out no-readback benchmark: selected BF16 CCL 55.930356 t/s/u versus matched BFP8 CCL 54.456046 t/s/u. Recorded this separately from teacher-forcing selection.
- Ran and manually reviewed the six-prompt shared chat-template qualitative suite; all HF/TT outputs were coherent and topical, with no format or language anomaly.
- Generated normalized JSON/CSV results and pyplot top-1/top-5 Pareto charts.

## Commands

The exact full-model candidate commands are stored per row in `sweep_results.json` and `sweep_results.csv`. Principal command families were:

```bash
/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls --local
pytest -q models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_full_model.py -k precision
MISTRAL_SMALL_24B_PRECISION_CONFIG=<candidate.json> HF_HUB_OFFLINE=1 \
  python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --reference models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/full_model/artifacts/aime24_chat_100.refpt \
  --mesh-device P300_QUAD --fabric-config FABRIC_1D --trace-region-size 200000000
python models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/generate_artifacts.py
```

Selected post-token-out command:

```bash
MISTRAL_SMALL_24B_PRECISION_CONFIG=models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/selected_precision_config.json \
MISTRAL_SMALL_24B_OPTIMIZED_FULL_MODEL_BENCHMARK=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724 \
MISTRAL_SMALL_24B_OPT_FULL_LAYERS=40 MISTRAL_SMALL_24B_OPT_FULL_STEPS=128 \
pytest -q -s models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_full_model.py::test_optimized_full_model_token_out_benchmark
```

Selected qualitative command:

```bash
MISTRAL_SMALL_24B_PRECISION_CONFIG=models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/selected_precision_config.json \
HF_HUB_OFFLINE=1 python models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/full_model/run_qualitative_suite.py \
  --snapshot /home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724 \
  --prompts models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/qualitative_suite \
  --max-new-tokens 128
```

## Gate and review state

- Accuracy gates: top-1 >= 90%, top-5 >= 98%; top-100 recorded and required at 100% for this reference.
- Selection metric: internal trace-verified full-model teacher-forcing decode t/s/u only.
- Selected: `bfp4_lofi_bfp8kv_bf16ccl`, 54.212248 t/s/u, pass.
- Post-selection token-out: 55.930356 t/s/u, exact correctness checks pass.
- Context: advertised 32,768 retained.
- Non-aligned prompt: pass.
- vLLM: deliberately not started.
- Stage review: independent rereview `clean-pass`; see `stage_review.md`.
- Stage-owned implementation/evidence commit SHA: recorded after checkpoint below.
