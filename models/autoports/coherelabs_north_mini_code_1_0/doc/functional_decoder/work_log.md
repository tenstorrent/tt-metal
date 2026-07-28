# Functional decoder work log

## 2026-07-28

- Resolved model revision `d11e61a842617a22dc328552fa5bb86231ee4f37`
  and inspected Transformers 5.10.2 `Cohere2MoeDecoderLayer`, attention, rotary,
  router, expert, RMSNorm, and dense MLP sources.
- Recorded target config: hidden 2048, Q heads 32, KV heads 4, head dim 128,
  49 layers, context 500000, sliding window 4096, 128 experts, sigmoid top-8,
  dense prefix intermediate 3072, sparse intermediate 768.
- Device health: `tt-smi` is not installed in this environment. A bounded 1x1
  TTNN mesh smoke passed on Blackhole. Four local `/dev/tenstorrent` devices were
  visible; this stage opened only one.
- Queried empty-device DRAM through `ttnn.get_memory_view`: 8 banks ×
  4,272,341,376 bytes = 34,178,731,008 usable bytes.
- Implemented functional decoder, paged BF16 cache, reversed/permuted page-table
  handling, Cohere interleaved RoPE equivalence, parallel residual order, dense
  SwiGLU, and sigmoid top-8 MoE.
- Dense smoke at prefill 32 and eager decode position 32 passed.
- Full dense decode compile/capture/replay passed with trace id 0.
- Manual HF-equivalent reference at logical prefill length 33:
  prefill PCC 0.9997368842; decode position 33 PCC 0.9997677531.
- Downloaded only official checkpoint shards 1 and 2 (2.80 GB on disk) for
  layer 1. Loaded 390 real layer tensors totaling 1,246,236,672 bytes.
- Official layer-1 real-weight decode at position 0 passed with PCC
  0.9997981557. Selected experts were `[107, 119, 126, 14, 61, 79, 18, 20]`.
- Added pytest coverage and ran:

```text
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py \
  -k contract_and_runtime_fallback_audit
Result: 1 passed, 10 deselected

pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py \
  -k 'prefill_non_aligned and 33 or decode_trace_replay'
Result: 2 passed, 9 deselected
```

- Full pytest suite before added boundary tests: 12 passed in 12.98 seconds.
- Added batch-2 prefill PCC and physical reversed-page cache checks, batch-4
  randomized nonzero position slot checks, bitwise decode determinism, paged
  prefill PCC for layers 1 and 4, and a controlled sliding-window reference at
  length 4097. The added semantic subset passed 4/4; the window test passed.
- Full layer-1 decoder probes passed at MoE chunk lengths 1023/1024/1025 and
  sliding-window lengths 4095/4096/4097. Dense long non-divisible prefill passed
  at 8193 (50.99 ms warmed) and 65537 (2867.65 ms warmed).
- Advertised-context prefill at 500000 passed with finite output in
  159557.12 ms for a single complete dense/full decoder pass.
- The initial 500000 prefill attempt ran two passes under a 300-second bound and
  was externally terminated while device 3 still had active cores. A bounded
  mesh smoke reported the expected firmware-init timeout. Located tt-smi at
  `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi`, listed four boards, reset only
  logical device 3, waited for reinitialization, relisted all four boards, and
  passed a 1x1 mesh smoke. The single-pass retry above then succeeded.
- Advertised-context traced decode at position 499999 passed for batch 1
  (1,024,000,000-byte KV cache, 44.28 ms replay) and batch 32
  (32,768,000,000-byte KV cache, 137.30 ms replay).
- Installed `tt-perf-report==1.2.8` into the active `python_env` with
  `uv pip` because that interpreter does not expose `python -m pip`.
- Warming/Tracy evidence at sequence 128:

| Kind | Wall prefill | Filtered device prefill | Wall traced decode | Filtered device decode |
|---|---:|---:|---:|---:|
| layer 0 dense/full | 0.636 ms | 586 us | 0.358 ms | 338 us |
| layer 1 sliding/MoE | 14.822 ms | 14644 us | 9.520 ms | 9452 us |
| layer 4 full/MoE | 14.712 ms | 14567 us | 9.520 ms | 9439 us |
| layer 0 batch-32 | n/a | n/a | 6.644 ms | 6614 us |

  Every filtered report contains zero host ops. Raw ops CSVs, filtered CSVs,
  human-readable tables, and command logs live below `tracy/`.
- Generated official layer-1 statistics for all 390 tensors consumed by the
  decoder: shape, dtype, mean, and standard deviation.
- Final watcher run used a unique `TT_METAL_LOGS_PATH` and no profiler:

```text
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<artifact>/watcher/full_suite \
  pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py
Result: 17 passed in 73.13 seconds
```

  The watcher attached/detached cleanly. A case-insensitive scan found no fatal,
  assert, invalid-NOC, CB bounds, overflow, sanitizer, or timeout signature in
  the watcher log.

Validation gates are complete. Independent stage review and the stage-local
commit remain.

## Independent review remediation

The first independent review returned `more-work-needed` for non-degenerate MoE
PCC/trace evidence, statistics-derived synthetic experts, updated stable trace
inputs, and near-limit nonaligned/MoE context evidence. Remediation:

- Sparse synthetic states now use deterministic BF16 weights at the recorded
  official layer-1 standard deviations (including attention, norm, router,
  gate/up/down experts) instead of generic scale or zero expert weights.
- Layer-1 multi-token prefill at length 1025 compares tokens 0/1023/1024 against
  active top-8 CPU expert references: PCC 0.9998267501.
- Layer-4 nonzero expert prefill: PCC 0.9997632031.
- Layer-1 decode uses populated, reversed-table paged K/V history and replays at
  position 4097 after copying new hidden/current-position/cos/sin values into
  the stable captured buffers: PCC 0.9998490619.
- Layer-1 batch-32 nonzero MoE replay updates the stable buffers before replay:
  PCC 0.9981929746.
- Layer-4 batch-1 nonzero MoE replay updates stable buffers: PCC
  0.9998228561.
- `_assert_pcc` now emits exact values, and the official real-weight test emits
  selected experts, so the preserved pytest logs are machine-readable evidence.
- Pinned the explicit model revision in both performance and capacity harnesses.
  Every regenerated JSON includes revision and layer identity.
- Near-limit nonaligned prefill at 499999 passed for every meaningful kind:
  dense/full/forced-RoPE 159650.64 ms, sliding/RoPE/MoE 22634.40 ms, and
  full/no-RoPE/MoE 176343.69 ms. Dense advertised 500000 was regenerated with
  full metadata and passed in 159559.74 ms.
- Batch-1 traced decode at advertised context passed for all three kinds;
  batch-32 dense advertised-context traced decode also passed.
- Served batch-32 warmed/filtered traced-decode performance is recorded for
  every kind: dense 6.652/6.614 ms, sliding-MoE 11.122/11.084 ms, and full-MoE
  11.129/11.077 ms (wall/filtered device).
- Non-watcher expanded suite:

```text
pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py
Result: 22 passed in 59.62 seconds
Artifact: pytest_full_remediation.log
```

- Separate expanded watcher suite:

```text
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<artifact>/watcher/remediation_full_suite \
  pytest -q -s models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py
Result: 22 passed in 76.71 seconds
```

  Its watcher log has no fatal, assert, invalid NOC, CB bounds, overflow,
  sanitizer, or timeout signature.

Fresh independent re-review inspected the remediated source, tests, raw PCC and
watcher logs, capacity JSONs, Tracy CSVs/reports, HF semantics, and sparse API.
It returned `clean-pass` with no required work. Non-blocking notes were:

- the batch-32 trace test updates hidden content while the populated-history
  test separately proves dynamic position/cos/sin updates;
- future watcher evidence would retain every per-test watcher dump more clearly
  with `TT_METAL_WATCHER_APPEND=1`;
- older batch-1 Tracy command logs predate the explicit config revision pin,
  although the cached reference and pinned wall reruns resolve the exact target.

These do not contradict current correctness, profiler, or watcher evidence.
Stage checkpoint commit: `4e45a256771` (`Add North Mini functional decoder`).
