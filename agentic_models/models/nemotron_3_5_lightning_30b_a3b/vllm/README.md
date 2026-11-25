# nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 on Blackhole QB2 (P300x2)

TTNN implementation of [NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
(Nemotron-H hybrid: 52 layers = 23 Mamba-2, 23 MoE with 128 routed experts top-6 plus one shared expert, 6 attention
layers with 32 query / 2 KV heads of 128; hidden 2688, 131K vocabulary, 30B total / ~3B active parameters) for the QB2
1x4 Blackhole mesh (TP4), served through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin).
Native 262,144-token context. Produced by the tt-model-bringup agentic pipeline. The bring-up repository is not public;
`provenance/upstream.json` records the source (host, path, commit `f1dd78ee`, branch `perf/multiuser`), the hash of
every vendored file and the files changed since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 52-layer model: embedding, decoder stack, final norm, LM head; checkpoint loading; slot-major Mamba state, prefix snapshots |
| `tt/multichip_decoder.py` | TP4 attention, Mamba-2 and MoE layers (grouped-matmul experts, `per_expert_lanes` decode) |
| `tt/optimized_decoder.py` | Single-chip layer programs, policies and constants the TP4 layer builds on |
| `tt/generator.py` | Traced bucketed decode (1, 2, 4, 8, 16, 32 rows), chunked prefill, on-device sampling, prefix-cache restore |
| `tt/generator_vllm.py` | vLLM adapter (`NemotronForCausalLM`), vLLM-owned paged KV cache for the 6 attention layers, decode-input reload contract v1 |
| `tt/wide_sampling.py` | Model-owned traced 64-candidate sampler for requests with top_k unset or above 32 |
| `tt/capacity.py`, `tt/prefill_plan.py` | KV pool sizing (`NEMOTRON_KV_POOL_TOKENS`) and prefill chunk plan |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision (below). `NEMOTRON_PRECISION_CONFIG=<path>` loads another config |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup runtime |
| `tt/tt_ccl.py` | `models.common.tt_ccl.TT_CCL` with lazy semaphores (`eager=False`) for the sampler, as a subclass of the pinned class |
| `vllm_metadata.json` | The checkpoint architecture and the adapter class |
| `model/entry.py` | `register()`: registers `TTNemotronHForCausalLM` with vLLM, overwriting any earlier entry (used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`; no bring-up paths remain; the metadata names the adapter; `register()` overwrites |
| `model/tests/test_vllm_adapter.py` | CPU: capabilities, the plugin's call shapes, serving geometry and precision loading, KV cache spec, the vendored `TT_CCL` |

## Precision

`bfp8_hifi2_ccl8`: BFP8 weights with HiFi2 for every projection (attention, Mamba in/out, routed and shared experts),
BFP8 LM head with HiFi4, BFP8 KV cache and CCL payloads, BF16 activations and residual, FP32 router and recurrent state.
BFP4 experts were tried and rejected in the bring-up. Bring-up accuracy against an FP32 reference: prefill KL 0.0023
at 2K / 0.0035 at 8K / 0.0018 at 32K, decode 0.0036 at 8K.

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` with the `patches.json` chain, vllm-tt-plugin `vllm_tt_plugin`
  with its `patches.json` patches, and vLLM `vllm`, with transformers 5.16 as vLLM 0.32 requires.
- tt-metal patches this model reaches at run time:
  - `tt-metal-op-matmul-reduce-scatter-prefill.patch`: `ttnn.experimental.matmul_reduce_scatter_prefill` (decode
    o/out projections), copied from tt-metal 8c48b4e4.
  - `tt-metal-op-nomic-moe-grouped-matmul-routing.patch`: `ttnn.experimental.nomic_embed_text_v2_moe.{routing_metadata,
    grouped_matmul, routed_combine}` (the MoE), copied from 8c48b4e4.
  - `nemotron_3_5_lightning_30b_a3b-grouped-matmul-per-expert-lanes.patch`: `grouped_matmul(per_expert_lanes=True)` for
    more experts than core rows (32 local experts). Fallback without it: `NEMOTRON_MOE_LANES=0` (slower decode).
  - `nemotron_3_5_lightning_30b_a3b-batch-prefix-view.patch`: the `batch_prefix_view` binding (zero-copy decode-bucket
    view of the Mamba state).
- vllm-tt-plugin patches:
  - `vllm-tt-plugin-prefill-state-slot-continuation.patch` (capability `preserve_prefill_state_slots`): a continuing
    chunked prefill keeps its state slot. Required: without it concurrent long prompts can corrupt the Mamba state of
    a continuing prefill.
  - `vllm-tt-plugin-prefix-cache-model-recurrent-state.patch` (capability `prefix_cache_recurrent_state_in_model`):
    with `--enable-prefix-caching`, vLLM caches attention blocks only (`mamba_cache_mode` none), and prefix hits are
    clipped to the Mamba snapshots the model holds. Needed only with prefix caching.
- Shared modules used unchanged from the pinned tt-metal: `models.common.lightweightmodule`, `models.common.sampling`,
  `models.common.tt_ccl` (base class and `get_tt_ccl`).
- Install this repository into the same Python environment (`pip install -e .`).
- The checkpoint at revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`: `MODEL_WEIGHTS_DIR` (or `HF_MODEL`) when it
  is a local checkpoint directory (tt-inference-server exports it), else the HF cache (`local_files_only=True`).

## Serving

The checkpoint architecture `NemotronHForCausalLM` is kept (vLLM's Nemotron-H hybrid handling keys on it), and the
adapter is selected with `TT_MODEL_CLASS_OVERRIDES`; neither `EXTRA_MODELS_DIR` nor `--hf-overrides` is used. Check
that the server log shows `Applied TT_MODEL_CLASS_OVERRIDES: TTNemotronHForCausalLM -> agentic_models...` at start-up.

```bash
export TT_MODEL_CLASS_OVERRIDES='TTNemotronHForCausalLM=agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.generator_vllm:NemotronForCausalLM'
export MESH_DEVICE=P300x2 VLLM_RPC_TIMEOUT=300000
export NEMOTRON_KV_POOL_TOKENS=4194304 NEMOTRON_VLLM_HOST_COMPAT=1 NEMOTRON_VLLM_SERIAL_PREFILL=1 NEMOTRON_PREFIX_CACHE=1
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  --block-size 32 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --enable-prefix-caching --async-scheduling --trust-remote-code \
  --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-logprobs -1 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 256000000, "fabric_config": "FABRIC_1D", "l1_small_size": 16384}}'
```

- `NEMOTRON_VLLM_HOST_COMPAT=1` is mandatory: log-probs, penalties and intermediate prefill chunks use host logits;
  without it the adapter rejects them.
- `NEMOTRON_KV_POOL_TOKENS` sizes the shared KV pool; the code default is 262,144 tokens.
- `NEMOTRON_PREFIX_CACHE=1` with `--enable-prefix-caching` turns on prefix caching (attention KV by vLLM, Mamba state
  snapshots by the model, `NEMOTRON_PREFIX_SNAPSHOTS` rows, default 32). Drop both to serve without it.
- The bring-up served with `"seeded_sampling_policy": "request_stable"`, a plugin feature that is not carried; the
  policy is the plugin default `auto`.
- Block size 32, the 1x4 mesh, `max_num_seqs` <= 32 and `max_model_len` <= 262,144 are enforced by the adapter.

Requests with top_k 1..32 sample with the common device sampler; top_k unset or above 32 samples from the top-64
candidates with the model's own traced sampler (`NEMOTRON_UNBOUNDED_TOP_K=64`; total-variation distance 6e-5 against
exact sampling in the bring-up). `NEMOTRON_UNBOUNDED_TOP_K=0` advertises `max_device_top_k` 64 instead, so the plugin
samples those requests on the host.

## Results

Validation at the repository pins is pending (`manifest.json` `validation.status`). Bring-up references, on the
bring-up tt-metal (a later base than the pin, plus the bring-up's changes) and vllm-tt-plugin 042a9486 with local
commits (vLLM 0.31):

- GPQA Diamond (198 questions, thinking on, T 1.0 / top-p 0.95 / top-k unset, budget 262,144 minus prompt): 77.27,
  75.76, 75.25 over three runs (mean 76.09; card 75.44), no truncations. These runs used tt-metal fd950cc8 / df3b4414;
  the final perf code (f1dd78ee) was re-checked with the KL gates and a prefix-cache restore check only.
- SWE-bench Verified, 5 tasks: 2/5. Terminal-Bench 2, 5 tasks: 2/5.
- Per-user decode (TTFT): 128/128 1 user 70.3 tok/s (0.10 s), 16 users 29.8; 1K/1K 1 user 70.0 (0.25 s), 16 users 29.8
  (444 tok/s total); 4K/512 1 user 69.6 (0.57 s), 16 users 22.0; 32K/512 1 user 68.4 (TTFT 4.99 s).

## Known limitations

- **Not yet validated at the pins.** The model was brought up 658 tt-metal commits after the pin and on vLLM 0.31;
  op drift (matmul, SDPA, CCL, sampling RNG) can move numerics and perf. The bring-up numbers above are not the CI bar.
- **Prompts are prefilled one user at a time** (`NEMOTRON_VLLM_SERIAL_PREFILL=1`): packed MoE prefill did not match
  solo prefill exactly. At 4K/512 with 16 users the p90 TTFT is 7.7 s.
- **Intermediate prefill chunks read full-vocabulary logits to the host.** The adapter declares
  `supports_prefill_no_output`, but the plugin at the pin does not use it.
- **Seeded requests are not reproducible across different batch companions** (no `request_stable` policy), and the
  64-candidate sampler draws from a counter hash, not the common sampler's random stream.
- **Penalties are not applied on device** (`supports_device_penalties` False): such requests sample on the host.
- **Prefix-cache hits are clipped** to the Mamba snapshots the model holds, so a hit can be shorter than vLLM's
  attention-block match.

## Test

```bash
agentic-models check --model nemotron_3_5_lightning_30b_a3b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model nemotron_3_5_lightning_30b_a3b --phase vllm    # CPU package and adapter tests
```
