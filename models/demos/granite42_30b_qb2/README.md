# Granite 4.2 30B on QuietBox 2

Run `ibm-granite/granite-4.2-30b` on two P300 cards (four Blackhole devices,
1×4 tensor-parallel mesh). The checkpoint revision is
`9e668ce1c538387ef24d3644e9b0606647762636`.

The implementation supports 1–16 requests, decode buckets 1/8/16, and a shared
655,360-token KV pool. That pool is shared across requests; it is not 131,072
tokens per request at batch 16. Prefill uses chunks up to 1024 tokens. Prefix
caching is supported and enabled in the release configuration. Configured capacity does not establish accuracy at
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

TTI supplies `HF_MODEL` as a local checkpoint directory. The model loads its
weights, config, and tokenizer from that same directory. The directory must
contain the original Granite config, indexed safetensor shards, and tokenizer
files. Its contents are trusted: a local directory does not prove a Hugging Face
revision. TTI must populate it from the pinned revision above. With `HF_MODEL`
unset or equal to the model repository ID, the model resolves only that pinned
snapshot from the offline Hugging Face cache.

Run from the TT-Metal checkout, with its Python environment active:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export EXTRA_MODELS_DIR="$PWD/models/demos"
export MESH_DEVICE=P300x2
vllm serve ibm-granite/granite-4.2-30b \
  --revision 9e668ce1c538387ef24d3644e9b0606647762636 \
  --tokenizer-revision 9e668ce1c538387ef24d3644e9b0606647762636 \
  --max-model-len 131072 --max-num-seqs 16 \
  --block-size 32 --max-num-batched-tokens 1024 --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details --hf-overrides '{"architectures":["TTGraniteForCausalLM"]}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --additional-config '{"tt":{"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all","trace_region_size":134217728}}'
```

`vllm_metadata.json` supplies registration through `EXTRA_MODELS_DIR`; no plugin
fork or built-in model registry edit is needed. The adapter declares ring fabric
before mesh creation. Greedy and bounded top-k sampling (up to 32) run on device.
The plugin routes full-vocabulary sampling, penalties, and unsupported device
parameters through host logits. Host and device RNG streams are independent.
Granite 4.2 uses the same `<think>` delimiters as vLLM's built-in `qwen3`
reasoning parser, so the server does not need a model-local parser copy.

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

Weekly CI uses the Agentic Research model-tests pipeline.

## Agentic Research CI

The `granite42-30b-qb2` entry in `Agentic Research Model Tests` runs
`tests/run_ci.sh` on `bh_quietbox_2`. It runs the host contracts, token-history
trace and external-cache device regressions, reasoning-channel and partial-batch
HF decoder comparisons for prefill and traced decode, API checks, cold-versus-cached comparisons for ten concurrent requests with aligned
and unaligned cached prefixes, warmed serving at concurrency 1/8/16, and the first ten GPQA Diamond
questions with a 32K output budget and thinking enabled. The subset gate is 6/10;
this is CI regression coverage, not a full-set accuracy or GPU-parity claim.
Fixed-length performance uses raw token continuations and checks completion;
Shield separately grades release performance targets and Terminal/SWE results.

Dispatch the workflow on this PR branch with model `granite42-30b-qb2`,
sku `bh_quietbox_2`, tier `3`, and `vllm-tt-plugin-ref` set to the default pin in
`tests/run_ci.sh`. The shared workflow otherwise overrides the script with `main`.
For direct script runs, set `VLLM_TT_PLUGIN_REF` only to test another revision. The 45-minute allowance
includes device tests, model startup, reasoning and setup variance; phase timings
in the artifacts allow recalibration after a complete run. The weights must be
available in the runner's pinned Hugging Face cache.
