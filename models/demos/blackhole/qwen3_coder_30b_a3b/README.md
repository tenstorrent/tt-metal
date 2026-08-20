# Qwen3-Coder-30B-A3B-Instruct on Blackhole

This directory implements Tenstorrent Blackhole inference for
**`Qwen/Qwen3-Coder-30B-A3B-Instruct`** — a 48-layer `Qwen3MoeForCausalLM`
sparse mixture-of-experts model (30B total / ~3B active parameters) with an
advertised context of 262144 tokens.

| Model | `HF_MODEL` | Mesh / `--mesh-device` | Parallelism |
| ----- | ---------- | ---------------------- | ----------- |
| Qwen3-Coder-30B-A3B-Instruct | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 4 Blackhole dies — `P300x2` (a `(1, 4)` mesh) | 4-way tensor parallel |

The 4-die path needs `FABRIC_1D_RING` for the cross-device collectives and a
trace region for the captured decode and chunked-prefill traces
(`DEFAULT_TRACE_REGION_SIZE = 300_000_000`, see [tt/model.py](tt/model.py)).

## Architecture

Assembly: `tok_embeddings → 48 × decoder layer → RMSNorm → LM head → on-device
sampling`.

Each decoder layer is GQA attention (rotary, per-head QK-norm) followed by a
sparse MoE block — a router plus top-k expert MLPs — replacing the dense MLP.
Everything shape-related (layer count, expert count, top-k, head dims, vocab,
rope base) is read from the parsed HF config, so the code follows the
checkpoint rather than hard-coding it.

| File | Role |
| ---- | ---- |
| [tt/model.py](tt/model.py) | Full 48-layer model, weight load, KV cache, trace capture, LM head + sampling |
| [tt/functional_decoder.py](tt/functional_decoder.py) | Reference-shaped single-layer decoder (attention + MoE) — the correctness baseline |
| [tt/optimized_decoder.py](tt/optimized_decoder.py) | Single-device optimized layer (fused QKV, sharded matmuls, program configs) |
| [tt/multichip_decoder.py](tt/multichip_decoder.py) | Tensor-parallel layer and the CCL schedule for the 4-die mesh |
| [tt/precision.py](tt/precision.py) | `PrecisionConfig` — per-tensor dtypes and math fidelity, overridable via `QWEN3_PRECISION_CONFIG` |
| [config/](config/) | Runtime policy the serving path reads: the selected precision config and the served-context contract |
| [tt/weight_mapping.py](tt/weight_mapping.py) | HF checkpoint → device tensor layout (QKV permutes, expert stacking) |
| [tt/generator.py](tt/generator.py) | `build_generator()` + the high-level `generate()` loop (owns KV cache and page table) |
| [tt/generator_vllm.py](tt/generator_vllm.py) | vLLM adapter — `prefill_forward` / `decode_forward` against a caller-owned cache |
| [vllm_bundle/](vllm_bundle/) | `EXTRA_MODELS_DIR` bundle that registers the adapter with the TT vLLM plugin |

The generator implements the `Generator` ABC in
[models/common/readiness_check/contract.py](../../../common/readiness_check/contract.py),
so the same object serves the host-side demo/readiness path and vLLM.

## Precision

A 29-row datatype sweep over the attention, MoE, KV, CCL, norm and LM-head
tensors selected [config/selected_precision_config.json](config/selected_precision_config.json),
which the vLLM path loads on every serve so that serving and readiness cannot
run different numerics. `DEFAULT_PRECISION` in [tt/precision.py](tt/precision.py)
is the equivalent in-code default. Override for experiments with
`QWEN3_PRECISION_CONFIG=<path-to-json>`.

Served context is capped by
[config/context_contract.json](config/context_contract.json) rather than by the
`--max-model-len` you pass, so a request for more context than has been
validated fails loudly instead of serving a quietly-clipped model.

## Running the tests

All device tests target the 4-die mesh. From the repository root:

```bash
source python_env/bin/activate
export HF_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
D=models/demos/blackhole/qwen3_coder_30b_a3b

# module + model correctness (excludes the perf-only tests)
pytest $D/tests/ -m "not models_performance_bare_metal" -q

# perf tests (decode/prefill timings; writes CSVs under doc/)
pytest $D/tests/test_perf.py -q
```

`test_full_model.py` runs a 2-layer model by default so it stays cheap; set
`QWEN3_FULL_MODEL_LAYERS=48` for the complete model.

Host-only tests (no device): `tests/test_reference.py`,
`tests/test_precision_config.py`.

## Readiness checks and vLLM serving

The shared harness in [models/common/readiness_check/](../../../common/readiness_check/)
drives this model through `tt/generator.py`:

```bash
# teacher-forced accuracy against a reference completion
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/demos/blackhole/qwen3_coder_30b_a3b \
  --reference <reference.refpt> \
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000
```

vLLM serving registers through the plugin's `EXTRA_MODELS_DIR` hook — no edit
to the vLLM checkout is required:

```bash
export EXTRA_MODELS_DIR=$PWD/models/demos/blackhole/qwen3_coder_30b_a3b/vllm_bundle

python -m models.common.readiness_check.run_vllm_server \
  --model-dir <output-dir> --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 1 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
```

`--generation-config vllm` matters: this checkpoint's `generation_config.json`
injects `repetition_penalty=1.05` into every request that does not override it,
and a penalised request costs ~14% TPOT because the penalty operands are staged
per step on the host.

## Measured performance

4 Blackhole dies, 48 layers, traced decode, on-device sampling, greedy,
128-token input / 128-token output, batch 1.

| Path | TTFT | Decode |
| ---- | ---- | ------ |
| Standalone traced (`generate()`) | 129.9 ms | 19.213 ms — **52.05 t/s/u** |
| Through vLLM (`max_num_seqs=1`, `--max-concurrency 1`) | 307–312 ms | 19.78 ms — **50.3–50.6 t/s/u** |

vLLM adds ~0.57 ms per decoded token (2.9%) over the standalone traced path.
The TTFT gap is request handling, tokenisation and detokenisation, not decode.

Evaluated end to end by tt-inference-server 0.20.0 against a live server:
mbpp 77.2%, humaneval 92.7%, ifeval 81.1%/87.1%, gpqa_diamond_cot 56.1%.

## Known limitations

- **Long-prefill scaling is bad.** A 131072-token prefill completes and returns
  valid output but takes 94.4 minutes — 3.05× worse per token than 65536, and
  well off what the single-layer sweep predicts. The suspected cause is
  per-chunk tensor accumulation in the MoE prefill path; this is an unproven
  hypothesis, not a measured root cause.
- **No full-model 262144 prefill has been verified.** 262144 is allocated,
  page-tabled and served, and prefills through a single layer, but the largest
  48-layer prefill measured end to end is 131072. The advertised context is
  left at 262144.
- **Not yet registered in the tiered models CI.** No entry exists in
  `models/model_ci_tiers.md`, `tests/pipeline_reorg/models_*_tests.yaml`,
  `models/model_targets.yaml` or the vLLM test registry, and there is no
  `demo/` entry point in the shape those pipelines invoke. See
  [models/MIGRATING_TO_TIERED_CI.md](../../../MIGRATING_TO_TIERED_CI.md).
