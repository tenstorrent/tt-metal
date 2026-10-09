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

The tests parameterize chunk sizes 2048, 4096, 8192, 16384, and 32768. Chunks must divide the context capacity and contain whole CP-local tiles. Layer tests compile and capture once per layer type, initialize the ring caches with random values, and measure each selected chunk once by default.

### Layer perf in CI

The **Blaze Models Prefill tests** workflow runs the `gemma4_d_p_layer_perf` stage with Tracy on a 14kW Galaxy. Dispatch it with `test-type=gemma4_d_p_layer_perf`; the regular nightly callers exclude this group. It measures `chunk_idx=ci`, which covers the cells in `LAYER_PERF_CI_CELLS`: the global layer at chunks 0, 1, 15, and 31, and the sliding layer at chunks 0 and 1, all at 256k@8k on 8×4. The job summary shows device-kernel time, span, and host time for each cell, plus each cell's full `tt-perf-report` output: the op table, advice, and stacked summary. The gap before each device's first replayed op is idle time before the replay, so it is left out of span and of `tt-perf-report`'s totals. The `layer-perf-*` artifact holds the raw `ops_perf_results_*.csv` and, for each cell, the slice of it that was reported (`*_ops.csv`) with `tt-perf-report`'s CSV, text output, stacked CSV/PNG, and log.

To reproduce it locally:

```bash
python -m tracy -p -r -v -o generated/profiler --op-support-count 20000 \
  -m "pytest models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunkci-both-sz8192-ctx_256k-8x4]"
pip install tt-perf-report
python models/demos/gemma4_d_p/scripts/layer_perf_report.py --profiler-dir generated/profiler
```

The test writes a manifest of its cells to `$PREFILL_SUMMARIES/layer_perf`, which defaults to `/tmp/prefill_summaries_$USER`. The report script slices the ops CSV by each cell's signpost pair and writes `$PREFILL_SUMMARIES/perf/gemma4_d_p_layer_perf.md`. It only parses files. To slice one cell of a downloaded artifact by hand:

```bash
tt-perf-report --start-signpost gemma4-layer-global-chunk15-start \
               --end-signpost gemma4-layer-global-chunk15-stop ops_perf_results_<ts>.csv
```

Global layers use tied QK projection; sliding layers use QKV. Weight caches are separated by dtype and mesh geometry. A valid completion marker permits cache-only loading; otherwise weights are loaded from the checkpoint. Set `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` to force checkpoint loading. Offline text input also needs the demo's cached corpus.

## Model and cache interfaces

- `tt/common.py::create_tt_model` constructs the prefill model, validates the 31B architecture and Galaxy/chunk geometry, and accepts externally allocated `ring_kv_caches`.
- `tt/model.py::Gemma4Model` supports prefill and last-token output processing. Decode entry points and speculative assistants are excluded.
- `tt/runners/kv_caches.py::allocate_ring_kv_caches` allocates one durable cache per semantic layer. Global layers store 640 channels (`Krot128 | V512`); local layers store separate 256-channel K and V caches.
- `tt/runners/kv_chunk_table.py::build_kv_chunk_address_table` exposes those same cache buffers for CP8/TP4 migration. It uses the shared migration utilities under `models/demos/common/prefill`.

Each model call prefills one user's chunk and returns post-norm hidden states. `max_batch_size` controls the number of durable user cache slots; `user_id` selects a slot. The constructor returns the ring caches, also exposed through `model.tt_kv_cache`. Physical capacity is at least two chunks so single-chunk prompts use the same ring SDPA path. External allocations accept `prefill_chunk_size`; `Gemma4KvCaches.max_seq_len` reports physical capacity for migration offsets. Callers can supply external caches and receive per-layer migration acknowledgements through callbacks, segmented traces, or a D2H socket service. Traced callers stage ring metadata and absolute RoPE positions before replay. This prefill model does not expose a logits projection API.

## Fixed 2×4K chunked batching

`tt/runners/chunked_batch_runtime.py::ChunkedBatchRuntime` defaults to exactly two
requests per batch on CP8/TP4. Create its model with `prefill_chunk_size=4096`
and at least two KV slots. Each rank holds 512 consecutive token rows from each
request, in lane order. Embedding, projections, norms, RoPE and MLP operate on
the combined 8K rows. Attention locally slices each lane, writes its cache slot,
executes the existing 4K ring-attention operation, and concatenates the local
outputs. The two attention calls execute sequentially within the trace.

```python
from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedRequest
from models.demos.gemma4_d_p.tt.runners.chunked_batch_runtime import ChunkedBatchRuntime

# model was created with 4K chunks and two or more cache slots.
runtime = ChunkedBatchRuntime(model, num_slots=2)
runtime.capture()  # Warm up and capture before populating request histories.
requests = tuple(ChunkedRequest(i, i, 0, tuple(prompts[i][:4096])) for i in range(2))
runtime.prefill_batch(requests)
outputs = runtime.to_torch()  # Independent host copies, keyed by request_id.
runtime.close()
```

Each lane supplies 1–4096 valid tokens. Short final chunks are zero-padded to 4K;
only valid output rows are returned. KV writes stop at the last 32-token page
containing valid tokens, preserving subsequent cache pages. Both lanes are
active: requests wait until a full batch is available. Starting at zero
replaces a slot's request; continuations require the same request identity and
the preceding full chunk's end. Lane order and prefix positions can change
without recapture. Absolute RoPE positions follow the tokens' CP-major order.

This cache uses **4K chunk geometry**. Histories placed with 8K or 1K chunks
cannot be reused. Migration tables must use `chunk_size=4096`. The runtime can
emit `(layer_index, request_id)` callbacks after a completed batch through
`layer_completion_sink`. Complete migration reads before overwriting a slot.
The earlier 4×1K shape remains available by passing
`plan=ChunkedBatchPlan(batch_size=4, chunk_size=1024)` to a runtime with 1K model
geometry and at least four slots.

With the usual model/cache environment, run the checks and performance modes
in separate processes:

```bash
pytest models/demos/gemma4_d_p/tests/test_chunked_batch.py -sv --timeout=7200
GEMMA4_BATCH_PERF_MODE=canonical pytest models/demos/gemma4_d_p/tests/test_chunked_batch_perf.py -sv --timeout=7200
GEMMA4_BATCH_PERF_MODE=chunked2 pytest models/demos/gemma4_d_p/tests/test_chunked_batch_perf.py -sv --timeout=7200
```

The perf test defaults to 60 layers and 256K context, populates histories using
real model calls, and measures five replays at selected prefix positions.
Both paths process **8,192 useful tokens per call**: canonical handles one 8K
chunk; the fixed batch handles two 4K chunks. Compare call latency or useful
tokens/s directly. `GEMMA4_BATCH_PERF_CONTEXT` controls capacity and
`GEMMA4_BATCH_PERF_OUTPUT` selects JSON output. Host staging is reported
separately from trace execution. Two final-batch samples retain the fixed 8K
execution shape with fewer useful tokens. Two full 256K request caches use
7.06 GiB/device, versus 14.11 GiB/device for four slots; weights and working
buffers are additional. Math settings are unchanged. The six-layer check
covers outputs, KV writes, isolation, migration addressing and trace replay.

Measured results: [2×4K vs canonical 8K report](docs/perf/chunked_batch_2x4k_vs_8k_2026_10_09/report.html)
and [PDF](docs/perf/chunked_batch_2x4k_vs_8k_2026_10_09/comparison.pdf).
The previous [4×1K report](docs/perf/chunked_batch_4x1k_vs_4k_2026_10_08/report.html)
is retained with its original measurements.

The combined PDF includes first/final global and sliding-layer operation
comparisons and detailed per-call tables generated with `tt-perf-report` 1.4.1
(main commit `cb9407747a28`). The isolated-layer test accepts:

- `GEMMA4_LAYER_BATCH_MODE=canonical|chunked2|chunked4` (default `canonical`).
- `GEMMA4_LAYER_PERF_CHUNKS=0,31` for canonical 8K; `0,62,63` for 2×4K.
- `GEMMA4_LAYER_PERF_REPEATS=5` to warm four replays before the signposted replay.

Use `test_prefill_layer_perf_chunk_n[blackhole-chunkall-both-sz8192-ctx_256k-8x4]`
for both current modes: `sz8192` is the total useful token count. A batch request
advances by 4K. The final canonical chunk is `[248K,256K)`; final batch requests
are each `[252K,256K)`. Batch chunk 62 supplies a matching-prefix control at
248K. Embedding and RoPE preparation are excluded from layer timing. These
isolated tests use random KV histories; full-model runs populate actual model
histories. The [reproduction script](docs/perf/chunked_batch_2x4k_vs_8k_2026_10_09/reproduce.sh)
includes Tracy capture and PDF generation. It omits the device timeline from
Tracy's GUI export while retaining the device CSV used for operation timings.

## Host verification

```bash
python_env/bin/python3 -m pytest models/demos/gemma4_d_p/tests/unit -k 'not device' -q
```

These checks cover packed-cache algebra, projection/RoPE permutations, migration addresses, supported mesh/chunk geometry, and independence from `models/demos/gemma4`. Device correctness and performance require a Galaxy run; host checks do not establish numerical trace-replay equivalence.
