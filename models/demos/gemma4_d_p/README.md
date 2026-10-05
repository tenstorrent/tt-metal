# Gemma4-31B-it disaggregated prefill on Galaxy

This package supports only the dense **Gemma4-31B-it** variant and owns its context-parallel prefill implementation and its migration-ready KV caches. It runs on one 32-device Blackhole Galaxy. Supported layouts are **8×4 (CP8/TP4)** and **4×8 (CP4/TP8)**; the migration address table currently supports **8×4** only.

The `models/demos/gemma4` implementation is independent of this package. Model, attention, weight-loading, and test helpers are local to `gemma4_d_p`. TTNN and model-independent utilities under `models/common`, `models/demos/common/prefill`, and `models/tt_transformers` are shared. The prefill service uses the shared engine under `models/demos/common/prefill`.

For the runner, producer, and service validation, see [Prefill service](docs/PREFILL_SERVICE.md).

## Run

Set the checkpoint and tensor-cache paths:

```bash
export \
       HF_MODEL=google/gemma-4-31B-it \
       HF_HOME=/mnt/models/huggingface \
       TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
       HF_HUB_OFFLINE=1
```

Full prefill:

```bash
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk8192-text-8x4] -sv
```

Layer performance:

```bash
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-global-sz8192-ctx_256k-8x4] -sv
pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkall-local-sz8192-ctx_256k-8x4] -sv
```

Both tests take chunk sizes 2048, 4096, 8192, 16384, and 32768; a chunk must split into whole 32-token tiles per CP rank. Layer tests compile and capture once per layer type, initialize the ring caches with random values, and measure each selected chunk once.

### Layer perf in CI

The **Blaze Models Prefill tests** workflow runs the `gemma4_d_p_layer_perf` stage with Tracy on a 14kW Galaxy. Dispatch it with `test-type=gemma4_d_p_layer_perf`; the regular nightly callers exclude this group. It measures `chunk_idx=ci` for 2048, 4096, and 8192 chunks at 256k context on 8×4. `layer_perf_ci_cells` derives the cells from the chunk count: the first, second, middle, and last global chunk (0/1/63/127, 0/1/31/63, and 0/1/15/31) and sliding chunks 0 and 1. All three chunk sizes run in one Tracy session, so each signpost names its chunk size. The job summary shows device-kernel time, span, and host time for each cell, plus each cell's full `tt-perf-report` output: the op table, advice, and stacked summary. The gap before each device's first replayed op is idle time before the replay, so it is left out of span and of `tt-perf-report`'s totals. The `layer-perf-*` artifact holds the raw `ops_perf_results_*.csv` and, for each cell, the slice of it that was reported (`*_ops.csv`) with `tt-perf-report`'s CSV, text output, stacked CSV/PNG, and log.

To reproduce it locally:

```bash
python -m tracy -p -r -v -o generated/profiler --op-support-count 20000 \
  -m "pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkci-both-sz2048-ctx_256k-8x4] models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkci-both-sz4096-ctx_256k-8x4] models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkci-both-sz8192-ctx_256k-8x4]"
pip install tt-perf-report
python models/demos/gemma4_d_p/scripts/layer_perf_report.py --profiler-dir generated/profiler
```

The test writes a manifest of its cells to `$PREFILL_SUMMARIES/layer_perf`, which defaults to `/tmp/prefill_summaries_$USER`. The report script slices the ops CSV by each cell's signpost pair and writes `$PREFILL_SUMMARIES/perf/gemma4_d_p_layer_perf.md`. It only parses files. To slice one cell of a downloaded artifact by hand:

```bash
tt-perf-report --start-signpost gemma4-layer-global-sz8192-chunk15-start \
               --end-signpost gemma4-layer-global-sz8192-chunk15-stop ops_perf_results_<ts>.csv
```

Global layers use tied QK projection; sliding layers use QKV. Weight caches are separated by dtype and mesh geometry. A valid completion marker permits cache-only loading; otherwise weights are loaded from the checkpoint. Set `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` to force checkpoint loading. Offline text input also needs the demo's cached corpus.

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

These checks cover packed-cache algebra, projection/RoPE permutations, migration addresses, supported mesh/chunk geometry, and independence from `models/demos/gemma4`. Device correctness and performance require a Galaxy run; host checks do not establish numerical trace-replay equivalence.
