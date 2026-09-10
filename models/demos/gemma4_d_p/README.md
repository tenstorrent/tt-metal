# Gemma4-31B-it disaggregated prefill on Galaxy

This package supports only the dense **Gemma4-31B-it** variant and owns its context-parallel prefill implementation and its migration-ready KV caches. It runs on one 32-device Blackhole Galaxy. Supported layouts are **8×4 (CP8/TP4)** and **4×8 (CP4/TP8)**; the migration address table currently supports **8×4** only.

The original `models/demos/gemma4` implementation is independent of this package. Model, attention, weight-loading, and test helpers are local to `gemma4_d_p`. TTNN and model-independent utilities under `models/common`, `models/demos/common/prefill`, and `models/tt_transformers` remain shared. This port includes the model and migration-cache interface, not a serving scheduler or a remote decode worker.

## Run

Use the existing checkpoint and tensor-cache configuration:

```bash
# TODO: update this to /mnt/models path and update this
export HF_HUB_OFFLINE=1 \
       HF_HOME=/localdev/svuckovic/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/localdev/svuckovic/huggingface/tt_cache/google--gemma-4-31B-it
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

## Model and cache interfaces

- `tt/common.py::create_tt_model` constructs the prefill model, validates the 31B architecture and Galaxy/chunk geometry, and accepts externally allocated `ring_kv_caches`.
- `tt/model.py::Gemma4Model` supports prefill and last-token output processing. Decode entry points and speculative assistants are excluded.
- `tt/runners/kv_caches.py::allocate_ring_kv_caches` allocates one durable cache per semantic layer. Global layers store 640 channels (`Krot128 | V512`); local layers store separate 256-channel K and V caches.
- `tt/runners/kv_chunk_table.py::build_kv_chunk_address_table` exposes those same cache buffers for CP8/TP4 migration. It uses the shared migration utilities under `models/demos/common/prefill`.

Each model call prefills one user's chunk and returns post-norm hidden states. `max_batch_size` controls the number of durable user cache slots; `user_id` selects a slot. The constructor returns the ring caches, also exposed through `model.tt_kv_cache`. Physical capacity is at least two chunks so single-chunk prompts use the same ring SDPA path. External allocations accept `prefill_chunk_size`; `Gemma4KvCaches.max_seq_len` reports physical capacity for migration offsets. Callers can supply external caches and receive per-layer migration acknowledgements through callbacks, segmented traces, or a D2H socket service. Traced callers stage ring metadata and absolute RoPE positions before replay. Last-token logits are computed separately with `process_logits_after_prefill_trace`.

## Host verification

```bash
python_env/bin/python3 -m pytest models/demos/gemma4_d_p/tests/unit -k 'not device' -q
```

These checks cover packed-cache algebra, projection/RoPE permutations, migration addresses, supported mesh/chunk geometry, and independence from the original Gemma4 package. Device correctness and performance require a Galaxy run; host checks do not establish numerical trace-replay equivalence.

## Prefill service

The common runner supports one CP8/TP4 Galaxy, multiple resident user slots,
chunk-aligned requests, and traced or eager execution. It loads input/decoder
weights only; the LM head and final norm are omitted. The manifest reserves
32 MiB per device for traces and enables device-side layer acknowledgements.

Use a populated **gemma4_d_p** weight cache (see the cache population script),
not the original `gemma4` cache:

```bash
HF_HOME=/localdev/svuckovic/huggingface \
HF_MODEL=google/gemma-4-31B-it \
TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
PREFILL_MANIFEST=models/demos/gemma4_d_p/tt/runners/manifests/gemma4_31b.json \
  python -m models.demos.common.prefill.runners.prefill_runner
```

After `setup complete, entering request loop`, run the producer in another shell:

```bash
PREFILL_MODEL=gemma4_31b PREFILL_SP=8 PREFILL_TP=4 \
PREFILL_CHUNK_SIZE=8192 PREFILL_MAX_SEQ_LEN=262144 \
PREFILL_NUM_LAYERS=60 PREFILL_NUM_USERS=2 \
PREFILL_H2D_SERVICE_ID=gemma4_prefill \
PREFILL_PRODUCER_SYNTHETIC_TOKENS=1 PREFILL_PRODUCER_WAIT_FOR_ACK=1 \
PREFILL_PRODUCER_CHUNKS=7 PREFILL_PRODUCER_MAX_REQUESTS=2 \
PREFILL_PRODUCER_INTERLEAVE=round_robin PREFILL_SEND_SHUTDOWN=1 \
  python -m models.demos.common.prefill.runners.prefill_producer
```

For a single-user run, set producer `PREFILL_NUM_USERS=1`,
`PREFILL_PRODUCER_MAX_REQUESTS=1`, and `PREFILL_PRODUCER_CHUNKS=2`.
Set `PREFILL_SEND_SHUTDOWN=0` to keep the server available after a run.
Match context and chunk configuration between processes; server environment
variables override manifest defaults. `REQUESTS_COMPLETE` confirms all layer
acknowledgements arrived; synthetic tokens do not establish numerical accuracy.
With the same model/cache environment, run the device regression with:

```bash
pytest models/demos/gemma4_d_p/tests/test_common_prefill_runtime.py -sv
```

It exercises slot changes, eager/traced KV equivalence, migration-table export,
and D2H acknowledgement metadata. It is not a comparison against an HF model.
