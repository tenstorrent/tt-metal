# Gemma4 prefill service

The service supports **Gemma4-31B-it, Blackhole 8×4, 262144 tokens per slot, 8192-token chunks, and batch 1**, with up to six resident KV slots. Defaults are in [the model manifest](tt/runners/manifest.json).

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

Prepare six 256K prompts, one different Gutenberg book per slot, then send them with the shared producer:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prepare_prefill_inputs
python -m models.demos.common.prefill.runners.prefill_producer --manifest /tmp/gemma4_prefill_inputs/producer.json
```

The preparation helper downloads and caches the books under `/tmp/gemma4_prefill_text`, then writes per-slot `metadata.json` token files and a producer manifest under `/tmp/gemma4_prefill_inputs`. Use one `--text /path/to/book.txt` per slot for local text, `--slots` for fewer prompts, or `--tokens 8193` to exercise a partial final chunk.

The shared producer uses round-robin scheduling and leaves the service running by default. Set `PREFILL_SEND_SHUTDOWN=1` when launching it to stop the service after the requests. These inputs contain tokens only, so golden-KV verification is disabled. In this mode the shared producer does not wait for layer acknowledgments; the hardware test below checks them separately.

Set `PREFILL_NUM_USERS` to 1–6 for the runner and prepare the same number of prompts with `--slots`. `PREFILL_H2D_SERVICE_ID` selects the shared service name and must match in both terminals. The service uses `google/gemma-4-31B-it`. `PREFILL_TTNN_CACHE` overrides the `TT_CACHE_PATH` root. One of these cache variables must be set. The runner reuses `tensor_cache_bf16_mesh8x4` beneath that root; no cache path is derived from the HF model ID or `HF_HOME`. Starting at position zero replaces a slot's prompt; subsequent chunks must be contiguous.

The service captures one trace and stages tokens, slot metadata, and absolute RoPE positions before each replay. Device acknowledgments follow each layer's KV writes. The engine owns the caches and sockets. The populated caches remain resident until shutdown.

Run the end-to-end hardware check, which starts both the runner and producer:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_service.py -sv --basetemp=/tmp/gemma4-service-test
```

It checks all six full-context prompts, final-chunk trace/eager PCC above 0.999, finite hidden states, nonzero first/last KV rows from every layer, distinct slot contents, and preservation of completed slots. It then reuses all six slots with 8193-token prompts to check partial final chunks. Producer logs, including timings, are saved in the pytest temporary directory. For the canonical demo, add `--timeout=3600` if loading weights exceeds the repository's default 300-second timeout.

For source KV correctness and real loopback migration, see [Migration tests](PREFILL_MIGRATION.md).
