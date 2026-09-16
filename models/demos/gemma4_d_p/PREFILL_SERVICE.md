# Gemma4 prefill service

The service supports **Gemma4-31B-it, Blackhole 8×4, 262144 tokens per slot, 8192-token chunks, and batch 1**, with up to six resident KV slots. Defaults are in `tt/runners/manifest.json`. See the [environment variable reference](PREFILL_ENV_VARS.md) for defaults, fixed values, unused settings, and producer arguments.

Run this setup from the repository root in both terminals:

```bash
source python_env/bin/activate
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export \
       HF_MODEL=google/gemma-4-31B-it \
       HF_HOME=/mnt/models/huggingface \
       TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
       HF_HUB_OFFLINE=1
```

Start the service:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prefill_runner
```

Send six interleaved 256K prompts, one different Gutenberg book per slot:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prefill_producer --results /tmp/gemma4-prefill-results.json
```

The producer downloads and caches the books under `/tmp/gemma4_prefill_text`. Use one `--text /path/to/book.txt` per slot for local text. It waits for all 60 device layer acknowledgments after each chunk and leaves the service available for another producer run. Pass `--shutdown` to send the shutdown sentinel after completion. `--tokens 8193` exercises a padded final chunk. Starting at position zero replaces a slot's prompt; subsequent chunks must be contiguous.

Set `PREFILL_NUM_USERS` to 1–6 in both terminals to change slot capacity. `PREFILL_H2D_SERVICE_ID` selects the shared service name. `PREFILL_HF_MODEL` overrides `HF_MODEL`; `PREFILL_TTNN_CACHE` overrides the `TT_CACHE_PATH` root. The runner reuses `tensor_cache_bf16_mesh8x4` beneath that root.

The service captures one trace and stages tokens, slot metadata, and absolute RoPE positions before each replay. Device acknowledgments follow each layer's KV writes. The engine owns the caches and sockets. The populated caches remain resident until shutdown.

Run the end-to-end hardware check, which starts both the runner and producer:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_service.py -sv --basetemp=/tmp/gemma4-service-test
```

It checks all six full-context prompts, final-chunk trace/eager PCC above 0.999, finite hidden states, nonzero first/last KV rows from every layer, distinct slot contents, and preservation of completed slots. It then reuses all six slots with 8193-token prompts to check padded final chunks. Producer timings and logs are saved in the pytest temporary directory. For the canonical demo, add `--timeout=3600` if loading weights exceeds the repository's default 300-second timeout.
