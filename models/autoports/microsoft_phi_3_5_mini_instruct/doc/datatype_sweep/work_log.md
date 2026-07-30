# Datatype Sweep Work Log

Date: 2026-06-15
Model: microsoft/Phi-3.5-mini-instruct
Commit: 3f9459fb070 on branch agentic-research/experiment-17-phi35-mini
Mesh: T3K 1x8 ring, FABRIC_1D_RING

## Thresholds

- top-1 >= 90%
- top-5 >= 98%
- top-100 kept at the readiness expectation; all teacher-forcing candidates reached 100/100 top-100.

## Baseline Refresh

Reference: `models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt`. Metadata: AIME24 prompt source, chat template, gen_len=100, top_k=100, one entry with prompt length 161.

Baseline prefill (`c000_default_optimized`): 96/100 top-1, 100/100 top-5, 100/100 top-100. Log: `logs/c000_prefill.log`.

Baseline traced teacher forcing (`c000_default_optimized`): 91/100 top-1, 100/100 top-5, 100/100 top-100, TTFT 258.55 ms, decode 40.17 t/s/u. Log: `logs/c000_teacher.log`.

## Candidate Outcomes

| `c000_default_optimized` | 91/100 | 100/100 | 100/100 | 258.55 | 40.17 | pass |
| `c001_safe_bf16` | 93/100 | 100/100 | 100/100 | 3272.85 | 12.66 | pass |
| `c002_bf8_weights_kv_ccl` | 93/100 | 100/100 | 100/100 | 257.25 | 26.17 | pass |
| `c003_default_weights_bf16_kv` | 92/100 | 100/100 | 100/100 | 220.16 | 26.31 | pass |
| `c004_default_weights_bf16_ccl` | 91/100 | 100/100 | 100/100 | 226.88 | 40.34 | pass |
| `c005_inner_mlp_bf4_edges_bf8` | 90/100 | 100/100 | 100/100 | 248.23 | 40.23 | pass |
| `c006_attention_bf4_all` | 90/100 | 100/100 | 100/100 | 225.76 | 26.61 | pass |
| `c007_activation_bf8` | 87/100 | 100/100 | 100/100 | 11833.38 | 9.03 | fail_accuracy |

Rejected configs:

- `c001_safe_bf16`: passes, but 12.66 t/s/u is far slower.
- `c002_bf8_weights_kv_ccl`: passes, but 26.17 t/s/u is slower than BF4 decode MLP.
- `c003_default_weights_bf16_kv`: passes, but BF16 KV cache drops decode to 26.31 t/s/u.
- `c005_inner_mlp_bf4_edges_bf8`: passes exactly at top-1 gate, but is not faster than selected.
- `c006_attention_bf4_all`: passes exactly at top-1 gate, but attention BF4 is slower in this full path.
- `c007_activation_bf8`: fails top-1 at 87/100 and is slow.

Selected config: `c004_default_weights_bf16_ccl`, because it is the fastest evaluated passing traced teacher-forcing result at 40.34 t/s/u with 91/100 top-1 and 100/100 top-5.

## Propagation Check

- `tt/precision.py` loads `doc/datatype_sweep/selected_precision_config.json` by default for this model directory.
- `build_generator` passes `model_dir` into `Phi35MiniForCausalLMTT.from_hf`.
- `Phi35MiniForCausalLMTT.from_hf` resolves the selected policy, applies embedding/norm/LM-head/logits dtypes, passes it into each `MultichipDecoder.from_state_dict`, and allocates KV cache with the selected dtype.
- The post-selection token-out run used `build_generator` with no `PHI35_PRECISION_CONFIG` override and recorded `construction_path: build_generator default path` in `perf/post_selection_token_out_no_readback_prompt128_gen128.json`.
- vLLM integration was not started. A later adapter should use the same full-model construction path or `load_precision_policy` helper.

## Post-Selection Token-Out

Artifact: `perf/post_selection_token_out_no_readback_prompt128_gen128.json`.

- Workload: prompt128/gen128, 8 warmup decode steps.
- TTFT: 214.10 ms.
- Decode: 56.37 t/s/u.
- E2E: 39.79 t/s/u.
- Trace counters: model trace captures 1, model trace replays 135, sampling trace captures 1, sampled-token readbacks 0, full-logit readbacks 0.

## Commands And Artifacts

Exact teacher-forcing commands are stored per result in `sweep_results.json` and `sweep_results.csv`. Raw logs are under `logs/`. Pareto plots are `top1_perf_pareto.png` and `top5_perf_pareto.png`.

Pytest note: direct pytest invocation is currently blocked by top-level conftest importing missing `models.tt_transformers`. The sweep therefore used the readiness runners directly and a direct Python token-out benchmark using the repo `python_env` interpreter.
