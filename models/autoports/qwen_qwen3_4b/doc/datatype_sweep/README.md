# Qwen3-4B Datatype Sweep

Stage: datatype-sweep for `Qwen/Qwen3-4B` in `models/autoports/qwen_qwen3_4b`.

## Gate and Selection

Accuracy thresholds used for this sweep:

- Top-1 >= 0.90
- Top-5 >= 0.98
- Top-100 == 1.00

Ranking metric: traced teacher-forcing decode `t/s/u` from `run_teacher_forcing` with `enable_trace=True` on the AIME24 chat-template readiness reference with 100 generated tokens. Eager and untraced decode numbers were not used for selection.

Selected config: `baseline_bfp4_lofi_bf16kv_bf16ccl`.

- Top-1/top-5/top-100: 94/100, 100/100, 100/100.
- Teacher-forcing TTFT: 872.85 ms.
- Teacher-forcing traced decode: 64.30 t/s/u.
- Post-selection warmed token-out no-readback TTFT: 332.73 ms.
- Post-selection warmed token-out no-readback decode: 65.78 t/s/u, steady 96.91 t/s/u.

This was the fastest evaluated config that satisfied the full-model accuracy gates. All evaluated configs passed the accuracy gate, but each alternative was slower on traced teacher-forcing decode.

## Selected Policy

The selected precision config is `selected_precision_config.json` and is consumed by default construction through `build_generator(model_dir=..., mesh_device=...)` when no explicit `model_config` or `precision_config_path` is supplied.

- Attention weights: `bfloat4_b`, all layers, qkv/o-proj.
- MLP weights: `bfloat4_b`, all layers, gate/up/packed gate-up/down.
- LM head weights: `bfloat4_b`.
- Embedding and norm weights: `bfloat16`.
- Layer exceptions: none.
- Compute fidelities: attention/MLP/LM-head/auxiliary all `LoFi`.
- Activation/residual dtype: `bfloat16` / `bfloat16`.
- CCL dtype: `bfloat16`.
- KV-cache dtype: `bfloat16`, block size 32, max blocks 1280, max sequence length 40960.
- Logits/sampling assumptions: BF16 vocab-sharded logits consumed by the greedy TP4 sampler, sampler emits UINT32 token ids.

Runtime consumption proof is in `policy_propagation_check.json`, `post_selection_token_out_no_readback_benchmark.json`, `logs/post_selection_token_out_no_readback.log`, and `logs/selected_qualitative.log`.

## Candidate Results

Full results are in `sweep_results.json` and `sweep_results.csv`.

- `bfp4_lofi_bfp8kv_bfp8ccl`: pass=pass, top1=0.94, top5=1.00, top100=1.00, traced teacher-forcing decode=62.99 t/s/u, TTFT=7150.14 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_lofi_bfp8kv_bf16ccl`: pass=pass, top1=0.93, top5=0.99, top100=1.00, traced teacher-forcing decode=26.42 t/s/u, TTFT=7677.82 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_hifi2_bf16kv_bf16ccl`: pass=pass, top1=0.94, top5=1.00, top100=1.00, traced teacher-forcing decode=23.40 t/s/u, TTFT=4261.78 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp8_hifi2_bf16kv_bf16ccl`: pass=pass, top1=0.98, top5=1.00, top100=1.00, traced teacher-forcing decode=23.02 t/s/u, TTFT=4232.39 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp8_lofi_bf16kv_bf16ccl`: pass=pass, top1=0.98, top5=1.00, top100=1.00, traced teacher-forcing decode=22.86 t/s/u, TTFT=4212.84 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.
- `bfp4_lofi_bf16kv_bfp8ccl`: pass=pass, top1=0.93, top5=1.00, top100=1.00, traced teacher-forcing decode=19.41 t/s/u, TTFT=5589.06 ms. Rejected because it was slower than `baseline_bfp4_lofi_bf16kv_bf16ccl` on the ranking metric.

The BFP4 material matmul policy was evaluated with LoFi in the selected baseline, satisfying the BFP4+LoFi coverage requirement without needing an autofix/runtime-blocker path. Canonical BF16-KV/BF16-CCL, BFP8-KV, BFP8-CCL, BFP4 HiFi2, BFP8 LoFi, and BFP8 HiFi2 policies were included.

## Pareto Plots

- `top1_perf_pareto.png`: top-1 versus traced teacher-forcing decode, Pareto frontier drawn, selected point marked red, minimum top-1 gate drawn as a vertical dotted line.
- `top5_perf_pareto.png`: top-5 versus traced teacher-forcing decode, Pareto frontier drawn, selected point marked red, minimum top-5 gate drawn as a vertical dotted line.

The selected point is the top performance point on both accuracy views, so the Pareto interpretation is straightforward: no evaluated point with equal-or-better gate compliance had higher traced teacher-forcing decode throughput.

## Context and Non-Aligned Prompts

The selected KV-cache dtype remains BF16, so the final selected config does not change the advertised memory-capacity path. The context contract was recomputed and remains `target=40960, supported=40960` with full HF context retained; see `logs/context_contract_check.log` and `../context_contract.json`.

Non-aligned prompt support is preserved. The baseline prefill refresh and post-selection token-out benchmark both used the AIME24 chat prompt with prompt length 158, which is not aligned to the KV block size 32. No selected KV-cache or trace-buffer dtype/layout change required an additional chunking-specific rerun.

## Qualitative Check

Selected default-path qualitative check passed all six shared prompts from `models/common/readiness_check/vllm_prompts.txt` with 64 generated tokens. The HF controls come from the optimized-full-model qualitative artifact and TT outputs were regenerated through the selected default path. Artifacts are under `qualitative/`; log is `logs/selected_qualitative.log`.

## Commands

- Baseline prefill refresh: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/baseline_bfp4_lofi_bf16kv_bf16ccl.json python -m models.common.readiness_check.run_prefill_check --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `baseline_bfp4_lofi_bf16kv_bf16ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/baseline_bfp4_lofi_bf16kv_bf16ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp4_lofi_bfp8kv_bf16ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp4_lofi_bfp8kv_bf16ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp4_lofi_bf16kv_bfp8ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp4_lofi_bf16kv_bfp8ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp4_lofi_bfp8kv_bfp8ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp4_lofi_bfp8kv_bfp8ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp4_hifi2_bf16kv_bf16ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp4_hifi2_bf16kv_bf16ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp8_lofi_bf16kv_bf16ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp8_lofi_bf16kv_bf16ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- `bfp8_hifi2_bf16kv_bf16ccl` teacher forcing: `QWEN3_4B_PRECISION_CONFIG=models/autoports/qwen_qwen3_4b/doc/datatype_sweep/configs/bfp8_hifi2_bf16kv_bf16ccl.json python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING`
- Context contract: `python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b`
- Post-selection token-out: inline Python script in `logs/post_selection_token_out_no_readback.log` using `build_generator(model_dir=model_dir, mesh_device=mesh)` with no precision override.
- Selected qualitative: inline Python script in `logs/selected_qualitative.log` using `build_generator(model_dir=model_dir, mesh_device=mesh)` with no precision override and `FABRIC_1D_RING` fabric.
- Host contract tests: `pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short`
- Syntax check: `python -m py_compile models/autoports/qwen_qwen3_4b/tt/model.py models/autoports/qwen_qwen3_4b/tt/generator.py models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`

## Limitations

The sweep measured one AIME24 chat-template readiness reference with 100 generated tokens for final ranking, not a broad benchmark suite. Some lower-precision policies improved or preserved accuracy on that reference but were materially slower in traced teacher-forcing throughput. The selected post-selection token-out benchmark is the number later reports and vLLM comparisons should prefer when available; teacher-forcing remains the ranking metric for this datatype-sweep stage.
