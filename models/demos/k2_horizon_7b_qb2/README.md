# K2-Horizon-7B on QB2

[IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) (Apache-2.0) served on one
Blackhole QuietBox 2 (P300x2, four chips) as a TP4 model with its full 524,288-token context.
The implementation was produced by the tt-model-bringup pipeline and then tuned by hand for
long contexts. It is pinned to checkpoint `036114ce8d46c32b24c15423211069abb9c5d25e`
(`tt/model.py`); the hub's `main` moved to a different checkpoint on 2026-10-01, which has not
been qualified against this implementation.

Model facts that shape the implementation: 36 decoder layers, hidden size 4096, 32 query and
8 KV heads of 128, grouped RMSNorm (4 groups), an untied 250,624-entry vocabulary (the embedding
table and LM head are 1.03B parameters each of the 9.0B total), and a reasoning chat template
whose `<think>` segments need the `k2_horizon` reasoning and tool-call parsers that
[vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) provides.

## Design

- **Tensor parallel over the 1x4 mesh** with the shared `models/common` CCL and sampling
  modules (`TT_CCL`, `Sampling1D`, `LMHead1D`). Decode runs as one traced step per batch bucket;
  the paged BFP8 KV cache uses 32-token pages.
- **Per-layer precision policy** (`doc/datatype_sweep/selected_precision_config.json`, loaded by
  `tt/precision_config.py`): BFP8/HiFi2 and BFP4/LoFi are mixed per layer and per operator.
  The policy was selected against HF logits with a teacher-forced gate (top-1/5/100 agreement
  98/100/100 on the AIME fixture) and a repeated-boundary stress fixture (PCC 0.995).
- **Attention in two regimes.** Prompts up to 64K tokens use the stock chunked SDPA with 64-row
  Q chunks (bitwise identical to the 128-row default, 1.28x faster). Beyond that, the model's own
  kernels in `tt/accurate_attention` take over: an FP32 online-softmax prefill kernel on the full
  11x10 core grid and a split-K flash-decode kernel generated from the stock decode program with
  FP32 statistics. They exist because the stock long-context paths lose accuracy (approximate
  exponentials on half tiles and BF16 running statistics gave 7 % error at 512K) and the fallback
  "accurate" path only used 8 of 64 cores. The host binding and kernels build on first use
  (`python -m models.demos.k2_horizon_7b_qb2.tt.accurate_attention.build`).
- **Prefill in 8192-token chunks**, so causal work balances across the grid; output tokens are
  identical to 4096-token chunks.
- **Sampling.** Requests inside the device sampler's domain (`top_k` in 1..32, no penalties or
  logit controls) sample on device; the model-card settings (temperature 1.0, top-p 0.95, no
  top-k) use the model's numerically equivalent host sampler (`tt/host_sampling.py`) when the
  server starts with `K2_VLLM_ALLOW_HOST_SAMPLING=1`; otherwise such requests are rejected at
  admission instead of failing inside the engine.

## Run

vLLM serving goes through [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). The
plugin registers `TTK2HorizonForCausalLM` from this directory's `vllm_metadata.json` when
`EXTRA_MODELS_DIR` points at `models/demos`; the server selects it with
`--hf-overrides '{"architectures":["TTK2HorizonForCausalLM"]}'`. The tt-inference-server dev
catalogue (`workflows/model_specs/dev/llm.yaml`, impl `k2_horizon_7b_qb2`) holds the serving
configuration that the release CI uses: `MESH_DEVICE=P300x2`, `--block-size 32`,
`--max-num-seqs 32`, `--max-model-len 524288`, async scheduling, no prefix caching,
`--reasoning-parser k2_horizon`, `--tool-call-parser k2_horizon`, `sample_on_device_mode: all`,
`trace_region_size: 200000000`, `fabric_config: FABRIC_1D_RING`.

Model-local checks (all need the four chips and the pinned weights in the HF cache):

```bash
pytest models/demos/k2_horizon_7b_qb2/tests/test_functional_decoder.py \
       models/demos/k2_horizon_7b_qb2/tests/test_fused_decoder.py \
       models/demos/k2_horizon_7b_qb2/tests/test_optimized_decoder.py
python -m models.demos.k2_horizon_7b_qb2.tests.check_accurate_prefill_attention   # vs FP64
python -m models.demos.k2_horizon_7b_qb2.tests.check_accurate_flash_decode
python -m models.demos.k2_horizon_7b_qb2.tests.run_multichip_long_context  # 524K stream vs HF
python -m models.demos.k2_horizon_7b_qb2.tests.benchmark_long_context      # TTFT/TPOT by length
```

`doc/context_contract.json` records the supported context (524,288) and the pinned revision.

## Validation

- Decode logits agree with HF at PCC 0.9998 on TP4, checked through position 524,288; the
  524,288-token streamed comparison of the attention layer against HF gives PCC 0.9989 at the
  last position and is unchanged below 64K.
- Teacher-forced AIME fixture: 98/100 top-1 agreement with HF under the selected precision policy
  (bring-up pipeline gate; its runner depends on the pipeline's `readiness_check` package and is
  not part of this directory).
- tt-inference-server release runs on `bh-qb-ge` (GPQA Diamond and MMLU-Pro CI subsets, 30
  benchmark points to 512K, API conformance) pass acceptance; GPQA 60-62.5 and MMLU-Pro 64.3-66.7
  on the 40- and 42-question subsets, where most misses are reasoning traces that hit the 32K
  output cap.

## Measured serving

tt-inference-server release run 36834542524 (tt-agentic-bringup-qb2, 2026-10-01, runner
qb2-120-p01t03), 128 output tokens, greedy:

| Input tokens | Concurrency | TTFT | TPOT | Output tokens/s |
|---:|---:|---:|---:|---:|
| 128 | 1 | 27.6 ms | 10.2 ms | 96.3 |
| 128 | 32 | 970 ms | 13.9 ms | 1494 |
| 4,096 | 1 | 210 ms | 10.9 ms | 79.9 |
| 4,096 | 32 | 5.34 s | 38.7 ms | 399 |
| 32,768 | 1 | 4.16 s | 13.2 ms | 21.9 |
| 65,536 | 1 | 14.1 s | 17.4 ms | 7.9 |
| 131,072 | 1 | 55.6 s | 22.4 ms | 2.2 |
| 262,016 | 1 | 217 s | 29.3 ms | 0.6 |
| 524,160 | 1 | 853 s | 45.7 ms | 0.15 |

Decode stays between 10 and 46 ms per token across the whole context window. Prefill above 64K
is bound by the accurate attention kernels' K/V re-reads and by board power throttling (AICLK
drops to 1.1-1.3 GHz with all four chips busy); the softmax exponential and chunk-size changes
in this directory's history record the measured gains.
