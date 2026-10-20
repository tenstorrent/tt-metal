# Gemma4 prefill service

The service packs multiple requests into **8192 total tokens** on Gemma4-31B-it, Blackhole CP8/TP4, with up to six resident KV slots and 262144 tokens per slot. Each request is padded to a multiple of 256 tokens. Rank `r` owns `[32r, 32r+32)`, `[256+32r, 256+32r+32)`, and so on, regardless of request length. Defaults are in [the model manifest](../tt/runners/manifest.json).

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

Prepare six prompts of 256, 512, 768, 1024, 1536, and 2048 tokens, then send them with the shared producer:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prepare_prefill_inputs
python -m models.demos.common.prefill.runners.prefill_producer --manifest /tmp/gemma4_prefill_inputs/producer.json
```

The preparation helper downloads and caches the books under `/tmp/gemma4_prefill_text`, then writes per-slot `metadata.json` token files and a producer manifest under `/tmp/gemma4_prefill_inputs`. Use one `--text /path/to/book.txt` per slot for local text, `--slots` for fewer prompts, or `--tokens 1600,512,768,1024,1536,2048` for custom lengths. A single count applies to every slot; `--tokens 262144` restores the full-context workload.

The shared producer coalesces ready requests without exceeding the padded token budget or using a KV slot twice in one batch. It flushes before idle gaps and at the end of the workload. Round-robin scheduling leaves the service running by default. Set `PREFILL_SEND_SHUTDOWN=1` when launching it to stop the service after the requests. These inputs contain tokens only, so golden-KV verification is disabled. In this mode the shared producer does not wait for layer acknowledgments; the hardware test below checks them separately.

Set `PREFILL_NUM_USERS` to 1–6 for the runner and prepare the same number of prompts with `--slots`. `PREFILL_H2D_SERVICE_ID` selects the shared service name and must match in both terminals. The service uses `google/gemma-4-31B-it`. `PREFILL_TTNN_CACHE` overrides the `TT_CACHE_PATH` root. One of these cache variables must be set. The runner reuses `tensor_cache_bf16_mesh8x4` beneath that root; no cache path is derived from the HF model ID or `HF_HOME`. Starting at position zero replaces a slot's prompt; continuation starts must equal that slot's previous end and align to 256 tokens. A partial final chunk is supported, but cannot be continued at an unaligned position. Set `PREFILL_CHUNK_SIZE` identically in runner and producer to change the total budget; it must be a positive multiple of 1024 because the residual tensor is split across CP8 and TP4.

Projection matmuls, normalization, RoPE, output projection, and MLP operate on the full packed tensor. Only KV updates and attention split into per-request slices. Global attention uses `ttnn.experimental.gemma4.rjsdpa(stripe_size=32)`; sliding attention uses the same op with `communication="halo"`. Absolute RoPE positions restart independently for each request.

Striped global attention shares legacy's SDPA tuning policy for request lengths with a 128-token-aligned local Q slab (multiples of 1024 tokens at CP8). For a full 8192-token request on CP8/TP4, it uses Q=96, K=256, one K split, and segmented accumulation. Intermediate 256-token request sizes retain their existing stripe-specific tuning: applying the policy indiscriminately reduced the synthetic 4352-token request's PCC below its 0.98 acceptance threshold. K blocks remain aligned to the cache capacity. Sliding attention keeps its separate stripe-specific policy.

A trace captures a fixed tuple of padded request sizes. Slot IDs, prefix lengths, valid ends, token IDs, and RoPE positions are staged in persistent device buffers before replay. New size tuples are compiled and captured lazily; `GEMMA4_PREFILL_MAX_TRACES` bounds the LRU cache (default 4). All traces share persistent input/output buffers. Shape changes incur a warmup and capture; steady replays do not. `PREFILL_LAYER_ACK_D2H=0` is required: host acknowledgments are emitted per request/layer after the completed batch, so warmup cannot generate false completions. Migration may start later than with per-layer device acknowledgments.

The H2D metadata format is now uint32 `[request_count, slot, start, end, ...]`, padded to `1 + 3 * min(num_users, chunk_size / 256)` words. Token payloads concatenate padded requests and unused batch padding, then transpose 32-token stripes into CP-major order. Use `packing.py` or the shared producer; old clients sending three metadata words must be updated. A metadata record filled with `0xffffffff` shuts down the service. KV migration tables and command-queue validation use the same fixed stripe ownership. The engine owns the caches and sockets; populated caches remain resident until shutdown.

Run the short integration check (two synthetic decoder layers, global and sliding attention, full 8k projection batch):

```bash
pytest models/demos/gemma4_d_p/tests/test_packed_prefill.py -sv
```

It compares mixed requests and continuation chunks against Hugging Face, checks trace reuse with changed slots/prefixes, partial final requests, trace eviction, and completion counts. It does not load the 31B checkpoint.

Run the full-checkpoint packed sanity test beside the canonical long-context demo, using the same environment above:

```bash
pytest 'models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_packed_traced[blackhole-ctx_256k-chunk8192-text-8x4]' -sv
```

This loads all 60 layers and allocates six 256k slots. It checks 17 mixed request segments (including 1600- and 257-token final requests), continuation, slot reuse, trace reuse/eviction, and completion counts. It then replays those schedules separately for each slot, replacing every other request's tokens with zeros, and requires every valid output row of the selected request to remain bit-for-bit identical. Keeping packed row positions fixed makes this a strict request-isolation check despite reduced-precision differences across layouts. Trace/eager PCC must be at least 0.999.

The test also extends one prompt to 256k in 6144-token pieces, batching 512- and 1536-token requests alongside it. Its final output is compared with legacy 8192-token chunked prefill using the same checkpoint and token history, requiring PCC ≥ 0.98 and relative RMSE < 0.21. Results are written to `packed_prefill.json` in the pytest temporary directory. The test allows two hours for cold weight loading and kernel compilation. The short synthetic test provides the separate Hugging Face comparison.

Compare one full chunk through legacy and striped prefill, using the same environment above:

```bash
GEMMA4_PERF_CHUNK_SIZE=8192 pytest 'models/demos/gemma4_d_p/demo/prefill_perf.py::test_prefill_single_request_comparison[blackhole-8x4]' -sv
```

The default chunk size is 8192; change `GEMMA4_PERF_CHUNK_SIZE` to a multiple of 1024, such as 4096. Both paths process the same real tokens through all 60 layers with one resident KV slot. The default cached prefixes are 0 and `floor((262144 - chunk_size) / chunk_size) * chunk_size`, aligned to the legacy chunk boundaries. Cache capacity rounds up to a complete chunk. `GEMMA4_PERF_PREFIXES=0` measures only the initial chunk. These are timing-only tests: caches start with zero-initialized synthetic KV, and metadata selects the desired prefix directly. The measured forwards update their current chunks, but historical tokens are not run through the model. Inputs use the same Gutenberg token helper as the canonical test. Device time, microseconds per real token, and speedups are reported; no outputs are downloaded and no PCC or relative RMSE is computed. Use the packed sanity test above for numerical acceptance checks.

Results are saved to `investigations/gemma4_single_prefill_<chunk_size>/results.json`; `GEMMA4_PERF_OUTPUT` overrides the directory. Set `GEMMA4_PERF_PROFILE_PROGRAMS=1` to retain kernel source names and individual program timestamps from the measured samples on the chip with the longest execution span. This adds host-side record processing, but no extra replays.

For debugging, replay one 4k packed batch with requests `[256, 512, 768, 1024, 1536]`:

```bash
pytest 'models/demos/gemma4_d_p/demo/packed_prefill_debug.py::test_prefill_packed_4k_traced[blackhole-8x4]' -sv
```

This uses all 60 layers, the canonical Gutenberg token source and trace reservation, and the same five 256k cache slots as the 4k benchmark. The requests use fixed cached prefixes `[27904, 12800, 25088, 29184, 13824]`, all below 32k and aligned to 256 tokens. They are embedded in the test and logged before model loading. Historical KV is zero-initialized synthetic data, as in the performance tests. Like the canonical test, it warms up the forward, captures it with the output kept alive, then executes the trace. It performs exactly one replay, with no performance sweep or output comparison. Pytest timeout is disabled for breakpoints. The `ttnn.execute_trace(...)` call is the replay entry point; Python attention code runs during warmup and capture.

Compare full requests and equally sized mixed batches at 4k, 6k, and 8k in one test:

```bash
pytest 'models/demos/gemma4_d_p/demo/prefill_perf.py::test_prefill_comprehensive_comparison[blackhole-8x4]' -sv
```

All three chunk sizes run sequentially on the same mesh, sharing device weights and releasing traces, caches, and cached programs between sizes. There is no chunk-size parametrization; `GEMMA4_PERF_CHUNK_SIZE` does not select a subset. Each size measures both the single request and mixed batch through legacy and packed prefill:

| Projection chunk | Single request | Mixed requests | Resident KV slots | Default long prefix |
|---:|---:|---|---:|---:|
| 4096 | 4096 | 256, 512, 768, 1024, 1536 | 5 | 258048 |
| 6144 | 6144 | 256, 512, 768, 1024, 1536, 2048 | 6 | 251904 |
| 8192 | 8192 | 512, 512, 1024, 1536, 2048, 2560 | 6 | 253952 |

Each comparison test prints a final table of median device milliseconds, microseconds per real token, and packed speedup, grouped by chunk size, request lengths, and cached prefix. The comprehensive test prints all three chunk sizes together.

The default run measures prefix 0 and the aligned long prefix for each size. Legacy pads each mixed request to the projection chunk size and executes requests sequentially; packed executes one shared projection/MLP pass for the whole batch. The 8k mix uses six requests because seven full-context KV slots exceed device DRAM with the full checkpoint. Results are saved after each size to `investigations/gemma4_comprehensive_prefill/results.json`, with raw per-chip samples in `chunk_4096/results.json`, `chunk_6144/results.json`, and `chunk_8192/results.json`. `GEMMA4_PERF_OUTPUT` overrides the parent directory. `GEMMA4_PERF_PREFIXES=0` shortens the run; explicit nonzero prefixes must align to all three chunk sizes.

The original version, which reconstructed real token histories and checked outputs, passed in 694.74 seconds on 2026-10-19, collecting all 24 measurements in one invocation. The [comparison table](../../../../investigations/gemma4_comprehensive_prefill/comparison.csv) and [sample distributions](../../../../investigations/gemma4_comprehensive_prefill/measurements.json) retain the run. Single-request output PCC ranged from 0.9948 to 0.9961 across the six size/prefix combinations.

The timing-only version passed all 24 measurements in 266.72 seconds on the same day. Time after the last weight load fell from about 651 to 217 seconds. The [timing-only comparison](../../../../investigations/gemma4_comprehensive_prefill/perf_only_comparison.csv) and [samples](../../../../investigations/gemma4_comprehensive_prefill/perf_only_measurements.json) retain this run. Most device medians were within 1% of the original run; the largest difference was 4.1%. These samples use synthetic KV history, so they measure execution cost without validating model outputs.

The initial single-request investigation (2026-10-19, model revision `3d19910c295`) measured 111.0 → 130.1 ms at prefix 0 and 274.0 → 331.7 ms at prefix 253952, legacy → striped. At chunk 4096 the corresponding measurements were 75.1 → 83.1 ms and 144.5 → 153.7 ms. [Recorded samples and program breakdowns](../../../../investigations/gemma4_single_prefill_diagnosis/measurements.json) include the baseline and isolated tuning experiments.

For chunk 8192, one diagnostic replay gave the following sums of device-program durations on the chip with the longest full-pass span. These are attribution samples, separate from the latency medians above.

| Component | Prefix 0: legacy / striped (ms) | Prefix 253952: legacy / striped (ms) |
| --- | ---: | ---: |
| 10 global attention layers | 5.3 / 7.4 | 160.0 / 205.5 |
| 50 sliding attention layers | 15.4 / 30.4 | 21.7 / 32.4 |
| Projection and MLP matmuls | 46.1 / 46.2 | 46.1 / 46.1 |
| Copies, KV typecasts and updates | 1.4 / 3.8 | 1.4 / 3.8 |

At that revision, the global settings differed: legacy used Q=96, K=256, one K split, and segmented accumulation; striped used Q=128, K=256, one K split, and ordinary accumulation. Q=96 produces 88 head/query work units versus 64 at Q=128 on the 110-core compute grid. An isolated override applying legacy's global settings to striped ownership reduced the long-prefix full pass from 331.7 to 293.2 ms. It also raised output PCC against legacy from 0.9530 to 0.9952 (the initial-prefix PCC was already about 0.995). Those global settings are now the default for 8192-token striped prefill; other supported lengths follow the policy described above. These historical numerical metrics were diagnostic; the current performance tests omit output comparisons.

The [production-tuning rerun](../../../../investigations/gemma4_global_tuning_fix/measurements.json) measured 129.3 ms at prefix 0 and 293.0 ms at prefix 253952, with output PCC 0.9949 and 0.9952 against legacy. Legacy measured 111.1 and 275.3 ms in the same run. These are medians of three full-model device spans; weight loading, warmup, trace capture, and history initialization are excluded. The long-prefix improvement over the old striped default is 11.7%, with a remaining 6.4% gap against legacy.

[Validation](../../../../investigations/gemma4_global_tuning_fix/validation.json) passed all 45 configuration tests, the expanded synthetic mixed-request comparison against Hugging Face, and the full 60-layer packed sanity test. The latter executed 248 request segments, preserved bit-exact request isolation, and reached PCC 0.9927 with relative RMSE 0.1216 against legacy after a 256k continuation.

Sliding attention loses locality under stripes. At this aligned long prefix, the current halo range calculation requests 7840 remote KV token rows per rank per KV head with stripe=32 and K=32, versus legacy's 1024-row boundary halo. Those are received-row counts, not an estimate of aggregate fabric-link bytes. Increasing K to 128 raises the striped receive count to 8064 and therefore trades coarser compute blocks against slightly more transfer padding. Separately, the full-request slices copy 110 L1 K/V tensors into DRAM; subsequent typecasts and cache updates also cost more. Their combined additional cost is about 2.4 ms per pass. The common matmuls are essentially unchanged, and a single full request has no cross-request weight-sharing benefit.

Changing only sliding K to 128 measured 128.5 ms at prefix 0 and 330.2 ms at prefix 253952, saving about 1.6 and 1.5 ms respectively. It did not resolve the long-prefix numerical gap (PCC 0.9516). Global tuning was the more effective change in these experiments.

The [diagnostic plugin](../../../../investigations/gemma4_single_prefill_diagnosis/config_override.py) was used for the isolated experiments above. Its `global_legacy` variant now matches the default global tuning for single full chunks. To experiment with sliding K, add `PYTHONPATH=investigations/gemma4_single_prefill_diagnosis:.`, `GEMMA4_PERF_VARIANT=swa_k128`, a distinct `GEMMA4_PERF_OUTPUT`, and `-p config_override` to the single-request command. The [summary helper](../../../../investigations/gemma4_single_prefill_diagnosis/summarize.py) compacts raw results and groups the diagnostic program timings by kernel source.

Compare full-model steady-state prefill performance across multiple batch shapes:

```bash
pytest 'models/demos/gemma4_d_p/demo/prefill_perf.py::test_prefill_packed_perf[blackhole-8x4]' -sv
```

This measures one 8192-token request, one 1600-token request, four 1600-token requests, and six requests of 256/512/768/1024/1536/2048 tokens. Legacy processes each request separately with 8192-token padding; packed processes the whole batch with one shared 8192-row projection/MLP pass. Both use all 60 layers, CP8/TP4, the same checkpoint, and six resident KV slots. This baseline represents the existing fixed-8192 path, without tuning legacy to smaller chunks.

Each case runs at cached-prefix lengths 0 and 253952. Prefixes are identical across requests and paths and aligned to legacy cache ownership. Cache buffers are zero-initialized and the requested prefix is staged directly, without reconstructing real histories. Weight loading, compilation, trace capture, metadata staging, and warmup are excluded from device timing. There are no output checks or separate wall/replay timing passes.

Each case uses one warmup and three measured replays per request (legacy) or batch (packed), collected in one profiler window. Device samples use real-time profiler timestamps: the maximum per-chip span from the first program's start to the last program's end, including gaps between operations. Legacy batch device time sums its separate request spans. Per-token metrics divide by real tokens, excluding padding. The real-time device profiler must be enabled; the benchmark rejects dropped records, missing chips, incomplete per-chip program counts, and inconsistent program sequences across replays. Use the packed sanity test above for numerical comparisons.

Results, individual samples, and per-chip timestamps are saved after each case to `investigations/gemma4_varlen_perf/results.json`. The [initial comparison](../../../../investigations/gemma4_varlen_perf/baseline_2026-10-19.csv) and [sample distributions](../../../../investigations/gemma4_varlen_perf/baseline_2026-10-19.json) are checked in. Set `GEMMA4_PERF_OUTPUT` to retain separate runs, `GEMMA4_PERF_PREFIXES=0` for a short initial-prefix comparison, or `GEMMA4_PERF_DEVICE_REPEATS` to change the number of measured replays (default 3). `GEMMA4_PERF_REPEATS` is no longer used because the separate host-timing loops were removed. Reports identify the synthetic cache history explicitly; historical results linked here used real token histories.

After the global tuning fix at `be99be86ffd`, the [comparison rerun](../../../../investigations/gemma4_varlen_perf_global_tuned/comparison.csv) measured the following full-model device times in milliseconds, shown as legacy → packed. Both paths were remeasured with six resident slots; [individual samples](../../../../investigations/gemma4_varlen_perf_global_tuned/measurements.json) are retained.

| Requests | No cached prefix | 253,952-token prefix |
|---|---:|---:|
| 1 × 8192 | 111.4 → 129.4 | 277.4 → 296.6 |
| 1 × 1600 | 111.2 → 104.5 | 272.5 → 249.2 |
| 4 × 1600 | 444.9 → 136.9 | 1094.1 → 717.2 |
| 256, 512, 768, 1024, 1536, 2048 | 667.7 → 142.7 | 1644.7 → 465.8 |

Run the end-to-end hardware check, which starts both the runner and producer:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_service.py -sv --basetemp=/tmp/gemma4-service-test
```

It checks all six full-context prompts, final-chunk trace/eager PCC above 0.999, finite hidden states, nonzero first/last KV rows from every layer, distinct slot contents, and preservation of completed slots. It then reuses all six slots with 8193-token prompts to check partial final chunks. Producer logs, including timings, are saved in the pytest temporary directory. For the canonical demo, add `--timeout=3600` if loading weights exceeds the repository's default 300-second timeout.

For source KV correctness and real loopback migration, see [Migration tests](PREFILL_MIGRATION.md).
