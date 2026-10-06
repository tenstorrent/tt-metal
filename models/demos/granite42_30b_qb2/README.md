# Granite 4.2 30B on QuietBox 2

Run `ibm-granite/granite-4.2-30b` on two P300 cards (four Blackhole devices,
1×4 tensor-parallel mesh). The checkpoint revision is
`9e668ce1c538387ef24d3644e9b0606647762636`.

The implementation supports 1–16 requests, decode buckets 1/8/16, and a shared
131,072-token KV pool. That pool is shared across requests; it is not 131,072
tokens per request at batch 16. Prefill uses chunks up to 1024 tokens. Prefix
caching is not supported. Configured capacity does not establish accuracy at
maximum context.

The selected precision policy is in `tt/precision_config.json`: BFP4 decoder
weights, BFP8 KV and LM head, BF16 activations and collectives. The model owns
persistent decode and prefill traces and accepts scheduler-owned paged KV storage.
The custom token-history kernel lives in this directory.

## Serving

Install TT-Metal and the current vLLM TT plugin using their supported procedures.
Download the pinned checkpoint into the Hugging Face cache before startup:

```sh
hf download ibm-granite/granite-4.2-30b --revision 9e668ce1c538387ef24d3644e9b0606647762636
```

Run from the TT-Metal checkout, with its Python environment active:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export EXTRA_MODELS_DIR="$PWD/models/demos"
export MESH_DEVICE=P300x2
vllm serve ibm-granite/granite-4.2-30b \
  --revision 9e668ce1c538387ef24d3644e9b0606647762636 \
  --tokenizer-revision 9e668ce1c538387ef24d3644e9b0606647762636 \
  --tensor-parallel-size 4 --max-model-len 131072 --max-num-seqs 16 \
  --block-size 32 --max-num-batched-tokens 1024 --enable-chunked-prefill \
  --no-enable-prefix-caching --hf-overrides '{"architectures":["TTGraniteForCausalLM"]}' \
  --reasoning-parser-plugin "$PWD/models/demos/granite42_30b_qb2/tt/granite_thinking_parser.py" \
  --reasoning-parser granite_thinking_parser --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --additional-config '{"tt":{"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all","trace_region_size":134217728}}'
```

`vllm_metadata.json` supplies registration through `EXTRA_MODELS_DIR`; no plugin
fork or built-in model registry edit is needed. The adapter declares ring fabric
before mesh creation. Greedy and bounded top-k sampling (up to 32) run on device.
The plugin routes full-vocabulary sampling, penalties, and unsupported device
parameters through host logits. Host and device RNG streams are independent.

## Tests

Host contract checks use CPU Torch and no device:

```sh
python -m unittest models.demos.granite42_30b_qb2.tests.test_host_contract -v
```

After reserving an idle QB2, run the focused device checks from the checkout:

```sh
pytest models/demos/granite42_30b_qb2/tests/test_token_history.py \
       models/demos/granite42_30b_qb2/tests/test_external_cache.py -v
```

The first check covers traced history writes, all decode buckets, and saturation.
The second uses one real checkpoint layer and verifies that preparation and reset
preserve an externally owned cache on every rank. It does not replace full-model
accuracy or serving qualification. Run the matching tt-inference-server release
workflow through Shield CI for accuracy, performance, API, and concurrent-request
qualification. Include partial batches 5 and 10 and batch transitions in serving
checks. Only Shield CI runs qualify this published implementation.

Shared weekly CI registration is deferred to preserve the requested
`models/demos`-only change scope.
