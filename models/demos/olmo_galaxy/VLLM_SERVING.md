# Serving OLMo-3.1-32B-Think on Galaxy (vLLM + vllm-tt-plugin)

Tested 2026-10-14 on a Wormhole Galaxy (8x4 mesh), tt-metal `main` + this branch, vLLM 0.30.0,
`tenstorrent/vllm-tt-plugin` `main` (b16250b). The old `tenstorrent/vllm` fork is deprecated and is not used.

## Install (into the tt-metal python_env)

```bash
source tt-metal/python_env/bin/activate
cd vllm-tt-plugin && source docs/install-vllm-tt.sh      # vLLM 0.30.0 (VLLM_TARGET_DEVICE=empty) + the plugin
```

## How OLMo plugs in

* `tt/generator_vllm.py: OLMo3ForCausalLM` is the model class; `vllm_models/olmo3/vllm_metadata.json` registers it as
  `TTOlmo3ForCausalLM` through the plugin's `EXTRA_MODELS_DIR` hook:
  `export EXTRA_MODELS_DIR=$TT_METAL_HOME/models/demos/olmo_galaxy/vllm_models`.
  Do **not** use `TT_MODEL_CLASS_OVERRIDES` for this model: it also replaces the bare `Olmo3ForCausalLM`, whose class
  then fails vLLM's text-generation interface check ("This model does not support `--runner generate`").
* The class uses the plugin's *legacy* decode contract (no `decode_input_update_contract`), so scheduling is
  synchronous. `decode_forward` adapts the plugin / warmup-mixin arguments to the generator's own
  `reset_inputs`/`reset_batch` (see the docstring).
* KV cache is allocated **bf16**, like the demo (`allocate_vllm_kv_cache_bf16`); the shared helper allocates bfloat8_b.
* The KV pool is 160K tokens (`get_max_tokens_all_users`): with the generic 131072 a batch of 32 users x 4K tokens fills
  ~98% of the pool and the device timed out.
* Prefill returns host-sampled first tokens (honouring the request's temperature / top_k / top_p / seed) when the plugin
  asks for device sampling, and the raw logits when it asks for host sampling.

## Run

Direct (no tt-inference-server):

```bash
MESH_DEVICE=TG HF_MODEL=<weights dir> TT_CACHE_PATH=<cache dir> TT_LEAN_DECODE_WARMUP=1 \
EXTRA_MODELS_DIR=$TT_METAL_HOME/models/demos/olmo_galaxy/vllm_models \
vllm serve <weights dir> --served-model-name allenai/Olmo-3.1-32B-Think --max-model-len 65536 \
  --max-num-seqs 32 --block-size 64 --override-generation-config '{"top_k": 32}' \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "worker_l1_size": 1345000, "trace_region_size": 184915840}}'
```

Through tt-inference-server (branch `ssinghal/olmo3-galaxy`, impl `olmo3_32b_galaxy`):

```bash
python3 run.py --model allenai/Olmo-3.1-32B-Think --workflow server --local-server --tt-device galaxy --dev-mode \
  --no-auth --tt-metal-home $TT_METAL_HOME --tt-metal-python-venv-dir $TT_METAL_HOME/python_env --host-hf-cache <hf cache>
```

`--host-hf-cache` needs the HF cache layout (`hub/models--allenai--Olmo-3.1-32B-Think/snapshots/<rev>/` can be a
symlink to a local download). Set `HF_HUB_OFFLINE=1` when the weights are local.

## Sampling

The device sampler draws from the 32 most probable tokens (`max_device_top_k = 32`). The model card's settings
(temperature 0.6, top_p 0.95) carry no `top_k`, so without a default the plugin samples on host: ~3x slower
(TPOT 127 ms vs 43 ms at 32 users). The server therefore defaults `top_k` to 32; a request that sets `top_k` keeps
it, and `top_k: 0` requests the full distribution (host sampling).

## Measured (single unpaired runs, `vllm bench serve --dataset-name random --ignore-eos`, 32 concurrent)

| workload | TPOT mean | notes |
|---|---|---|
| 128 in / 128 out, greedy | 42.4 ms (23.6 tok/s/user) | demo `batch-32`: 38.3 ms (26.1 tok/s/user) |
| 128 / 128, temperature 0.6, top_p 0.95 (default top_k 32) | 42.8 ms | device sampling |
| same with top_k unset and no server default | 127 ms | host sampling |
| 2048 / 128 | 51 ms | |
| 4096 / 128 | P99 ITL 55 ms | 64 requests, no failures |

The ~4 ms/token gap to the demo is the synchronous scheduling of the legacy decode contract (the demo overlaps host
work with the device step). Closing it needs the explicit reload contract (`decode_input_update_contract = 1`,
`supports_async_decode`) in `tt/generator.py`; see `models/demos/llama3_70b_galaxy/tt/generator.py: decode_forward`.

Long context through the server: needle retrieval passes at ~28K and ~57K prompt tokens.

## Known issues

* Batch-32 decode is not bit-reproducible run to run on the test host; localized to one chip's ring matmul
  (`bringup/olmo3/LOG.md`). The effect is ~1 bf16 ulp on one output column.
* Penalties are not applied to the first token (it is sampled on host from the prefill logits).
* Async scheduling / speculative decoding / prefix caching are not supported.
