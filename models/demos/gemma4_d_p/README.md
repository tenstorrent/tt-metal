# Gemma4-31B-it disaggregated prefill on Galaxy

This package supports only the dense **Gemma4-31B-it** variant and owns its context-parallel prefill implementation and its migration-ready KV caches. It runs on one 32-device Blackhole Galaxy. Supported layouts are **8×4 (CP8/TP4)** and **4×8 (CP4/TP8)**; the migration address table currently supports **8×4** only.

The original `models/demos/gemma4` implementation is independent of this package. Model, attention, weight-loading, and test helpers are local to `gemma4_d_p`. TTNN and model-independent utilities under `models/common`, `models/demos/common/prefill`, and `models/tt_transformers` remain shared. The prefill service uses the shared engine under `models/demos/common/prefill`.

## Run

Use the existing checkpoint and tensor-cache configuration:

```bash
export \
       HF_MODEL=google/gemma-4-31B-it \
       HF_HOME=/mnt/models/huggingface \
       TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
       HF_HUB_OFFLINE=1
```

Full prefill:

```bash
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-readback_final-ctx_256k-chunk8192-text-8x4] -sv
```

Layer performance:

```bash
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-global-sz8192-ctx_256k-8x4] -sv
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-local-sz8192-ctx_256k-8x4] -sv
```

Both tests support chunk sizes 4096, 8192, 16384, and 32768. A CP-local chunk must cover the 1024-token sliding window, so 4096 skips on 8×4. Layer tests compile and capture once per layer type, initialize the ring caches with random values, and measure each selected chunk once.

Global layers use tied QK projection; sliding layers use QKV. Weight caches are separated by dtype and mesh geometry. A valid completion marker permits cache-only loading; otherwise weights are loaded from the checkpoint. Set `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` to force checkpoint loading. Offline text input also needs the demo's cached corpus.

## Prefill service

The service supports **Gemma4-31B-it, Blackhole 8×4, 262144 tokens per slot, 8192-token chunks, and batch 1**, with up to six resident KV slots. Defaults are in `tt/runners/manifest.json`.

Activate `python_env` and set `PYTHONPATH`, `TT_METAL_HOME`, and the checkpoint variables above in both terminals.

Start the service:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prefill_runner
```

Send six interleaved 256K prompts, one different Gutenberg book per slot:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prefill_producer --results /tmp/gemma4-prefill-results.json
```

The producer downloads and caches the books under `/tmp/gemma4_prefill_text`. Use one `--text /path/to/book.txt` per slot for local text. It waits for all 60 device layer acknowledgments after each chunk and sends the shutdown sentinel after completion. `--keep-serving` leaves the service available for another producer run. `--tokens 8193` exercises a padded final chunk. Starting at position zero replaces a slot's prompt; subsequent chunks must be contiguous.

Set `PREFILL_NUM_USERS` to 1–6 in both terminals to change slot capacity. `PREFILL_H2D_SERVICE_ID` selects the shared service name. `PREFILL_HF_MODEL` overrides `HF_MODEL`; `PREFILL_TTNN_CACHE` overrides the `TT_CACHE_PATH` root. The runner reuses `tensor_cache_bf16_mesh8x4` beneath that root.

The service captures one trace and stages tokens, slot metadata, and absolute RoPE positions before each replay. Device acknowledgments follow each layer's KV writes. The engine owns the caches and sockets. The populated caches remain resident until shutdown.

Run the end-to-end hardware check, which starts both the runner and producer:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_service.py -sv --basetemp=/tmp/gemma4-service-test
```

It checks all six full-context prompts, finite final hidden states, nonzero first/last KV rows from every layer, distinct slot contents, and preservation of completed slots. Producer timings and logs are saved in the pytest temporary directory. For the canonical demo, add `--timeout=3600` if loading weights exceeds the repository's default 300-second timeout.

## Model and cache interfaces

- `tt/common.py::create_tt_model` constructs the prefill model, validates the 31B architecture and Galaxy/chunk geometry, and accepts externally allocated `ring_kv_caches`.
- `tt/model.py::Gemma4Model` supports prefill and last-token output processing. Decode entry points and speculative assistants are excluded.
- `tt/runners/kv_caches.py::allocate_ring_kv_caches` allocates one durable cache per semantic layer. Global layers store 640 channels (`Krot128 | V512`); local layers store separate 256-channel K and V caches.
- `tt/runners/kv_chunk_table.py::build_kv_chunk_address_table` exposes those same cache buffers for CP8/TP4 migration. It uses the shared migration utilities under `models/demos/common/prefill`.

Each model call prefills one user's chunk and returns post-norm hidden states. `max_batch_size` controls the number of durable user cache slots; `user_id` selects a slot. The constructor returns the ring caches, also exposed through `model.tt_kv_cache`. Physical capacity is at least two chunks so single-chunk prompts use the same ring SDPA path. External allocations accept `prefill_chunk_size`; `Gemma4KvCaches.max_seq_len` reports physical capacity for migration offsets. Callers can supply external caches and receive per-layer migration acknowledgements through callbacks, segmented traces, or a D2H socket service. Traced callers stage ring metadata and absolute RoPE positions before replay. This prefill model does not expose a logits projection API.

## Host verification

```bash
python_env/bin/python3 -m pytest models/demos/gemma4_d_p/tests/unit -k 'not device' -q
```

These checks cover packed-cache algebra, projection/RoPE permutations, migration addresses, supported mesh/chunk geometry, and independence from the original Gemma4 package. Device correctness and performance require a Galaxy run; host checks do not establish numerical trace-replay equivalence.
