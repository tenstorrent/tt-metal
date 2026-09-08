# Gemma4 disaggregated prefill on Galaxy

This package owns the Gemma4 context-parallel prefill implementation and its migration-ready KV caches. It runs on one 32-device Blackhole Galaxy. Supported layouts are **8×4 (CP8/TP4)** and **4×8 (CP4/TP8)**; the migration address table currently supports **8×4** only.

The original `models/demos/gemma4` implementation is independent of this package. Model, attention, weight-loading, and test helpers are local to `gemma4_d_p`. TTNN and model-independent utilities under `models/common`, `models/demos/common/prefill`, and `models/tt_transformers` remain shared. This port includes the model and migration-cache interface, not a serving scheduler or a remote decode worker.

## Run

Use the existing checkpoint and tensor-cache configuration:

```bash
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

Cache filenames and `GEMMA4_*` environment settings are preserved. Existing compatible Galaxy weight caches can be reused. For a cold cache, set `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1`; `GEMMA4_WEIGHT_CACHE_MESH_ONLY=1` disables the legacy unqualified cache fallback. Offline text input also needs the demo's cached corpus, or use the full-prefill test's random token source.

## Model and cache interfaces

- `tt/common.py::create_tt_model` constructs the prefill model, validates Galaxy/chunk geometry, and accepts externally allocated `ring_kv_caches`.
- `tt/model.py::Gemma4Model` supports prefill and last-token output processing. Decode entry points and speculative assistants are excluded.
- `tt/runners/kv_caches.py::allocate_ring_kv_caches` allocates one durable cache per semantic layer. Global layers store 640 channels (`Krot128 | V512`); local layers store separate 256-channel K and V caches.
- `tt/runners/kv_chunk_table.py::build_kv_chunk_address_table` exposes those same cache buffers for CP8/TP4 migration. It uses the shared migration utilities under `models/demos/common/prefill`.

The constructor's legacy cache return slot is retained for the demo call interface; ring caches live on each layer's `self_attn.ring_kv_cache`, or in the `Gemma4KvCaches` supplied by the caller. No paged cache is allocated by the standalone constructor. Internal tensor helpers retain generic shape handling, but the model entry points reject smaller meshes and decode execution.

## Host verification

```bash
python_env/bin/python3 -m pytest models/demos/gemma4_d_p/tests/unit -k 'not device' -q
```

These checks cover packed-cache algebra, projection/RoPE permutations, migration addresses, supported mesh/chunk geometry, and independence from the original Gemma4 package. Device correctness and performance require a Galaxy run; host checks do not establish numerical trace-replay equivalence.
