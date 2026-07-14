# Multichip decoder work log

## 2026-07-14: baseline and hardware inventory

- `timeout 60 tt-smi -ls --local`: four Blackhole P150b devices, firmware
  19.9.0.
- First mesh smoke omitted the repo build paths and failed at import with
  `_ttnncpp.so: cannot open shared object file`; this was an environment
  invocation error, not device evidence.
- Corrected smoke with `build_Release` on `PYTHONPATH`/`LD_LIBRARY_PATH`:
  `MeshShape(2,2)` opened and closed successfully; architecture Blackhole,
  compute grid 11x10, DRAM grid 8x1 per device.
- Stage 03 checkpoint is local commit `5e21925512d`.  Its frozen single-P150
  baseline is 1.186 ms sliding / 1.332 ms full traced decode and 2.672 ms
  sliding / 3.395 ms full prefill-128, with PCC above 0.998 for the main
  real-weight rows.

## Mesh decision before final-path coding

Selected `MeshShape(1,4)`, TP=4, BF16 replicated layer residual, local-head
attention and BFP8 paged KV cache, column-parallel BFP8 QKV, row-parallel BFP8
O, column-parallel BFP4 gate/up, and row-parallel BFP4 down.  The exact tensor,
cache, collective, memory, and rejected-alternative calculations are in
`README.md`.  The chosen path is dense; no MoE/expert strategy applies because
the target HF config has `enable_moe_block=false`.

The selected residual/collective contract was subsequently earned by the
adapted fractured-boundary comparison recorded below and in `README.md`.

## Implementation and correctness

- Added `tt/multichip_decoder.py` with a fixed 1x4 TP path derived from
  `OptimizedDecoder`: local Q/K/V heads and cache, column-parallel QKV/gate/up,
  row-parallel O/down, and replicated BF16 layer boundaries.
- The first construction attempt accidentally entered the inherited fused
  constructor and expected a different state-dict prefix. The direct
  `FunctionalDecoder` composition fixed that wiring error.
- The first BFP8 cache fill passed BF16 K/V directly to `paged_fill_cache`.
  The final TP prefill explicitly casts bulk fills and preserves BF16 token-tail
  updates for bounded sliding caches.
- Initial real-weight prefill passed at PCC 0.999849 sliding and 0.999759 full.
  Traced decode after the final DRAM-sharded attention change passes at PCC
  0.999954 sliding and 0.999886 full.
- Non-aligned sliding lengths 1,025 and 1,057, including cache wrap followed by
  decode, pass at PCC 0.999678 and 0.999504. Batch-32 non-aligned prefill and
  traced decode pass for both layer kinds at PCC 0.999907 / 0.999890.
- The acceptance test's first batch-32 attempt had a host-only shape helper
  error (`[B,1,H]` was reshaped as batch one). Correcting it to the documented
  `[1,1,B,H]` decode layout made the real device test pass; no decoder fix was
  needed.
- The page-table test uses a non-identity mapping. Trace tests replay eight
  times, mutate stable token and both position buffers, verify changed output,
  restore the buffers, and verify bitwise restoration across all replicas.

Acceptance commands:

```bash
pytest -q -s models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py --junitxml=.../evidence/final_suite.xml
GEMMA4_MULTICHIP_EXACT_CONTEXT=1 pytest -q -s ... -k advertised_context_traced_decode --junitxml=.../evidence/exact_context.xml
pytest -q -s ... -k paged_decode_trace_matches --junitxml=.../evidence/mutable_trace.xml
```

## Performance candidates and final selection

- Replicated-v1 with demo interleaved/default decode attention: 0.5999 ms
  sliding and 0.6702 ms full traced decode.
- Replaced it with TP-local packed QKV and row-local O, BFP8/LoFi,
  DRAM-width-sharded over all eight banks. After the final MLP sweep, decode is
  0.5268 ms sliding and 0.5768 ms full.
- QKV 8-core candidate: 0.5318 / 0.5821 ms. QKV 32-core candidate:
  0.5302 / 0.5824 ms. Kept 32 cores.
- Linear one-link candidate: 0.5558 ms sliding. Linear two-link final:
  0.5302 ms.
- The local square MLP traced-layer sweep measured 0.5324 / 0.5294 / 0.5413 /
  0.5261 ms for 8/12/21/24 cores; the final decode MLP uses 24 cores and block
  width 7 for all three projections.
- The prefill MLP sweep measured 0.7253 ms for auto-programming and 0.7937 /
  0.6241 / 0.6226 / 0.5830 ms for explicit 8/12/21/24-core 1-D programs. The
  final M<=128 path uses the 24-core 4x7 per-core geometry and 1x7 output
  subblock. DRAM-sharded prefill was actually attempted and rejected by the
  kernel's hard `M == 1` validation.
- The adapted fractured path used real RS, distributed RMSNorm, fractured
  residual, delayed AG, and the next gate projection. Decode trace measured
  0.2734 versus 0.1616 ms and prefill-128 measured 0.8545 versus 0.6796 ms, so
  the replicated boundary remains final. All outputs passed PCC >=0.99987.
- Final same-harness prefill-128 is 2.4044 / 2.4387 ms versus single-chip
  2.6721 / 3.4453 ms. Final traced decode is 0.5268 / 0.5768 ms versus
  single-chip 1.1862 / 1.3520 ms.

Evidence:

- `evidence/replicated_v1_latency.log`
- `evidence/dram_sharded_attention_latency.log`
- `evidence/candidate_qkv8_latency.log`
- `evidence/candidate_one_link_latency.log`
- `evidence/mlp_geometry_sweep.{log,xml}`
- `evidence/mlp_prefill_sweep.{log,xml}` and the DRAM-sharded rejected artifact
- `evidence/fractured_boundary.{log,xml}`
- `tracy/replicated_v1/sliding_decode/{filtered.csv,report.txt}`
- final four-window Tracy/`tt-perf-report` artifacts under `tracy/final`

## Context and physical memory correction

The initial plan counted unique weights once. The final implementation keeps
separate prefill and decode placements, so physical per-device stack accounting
is 9,498,818,560 B weights plus 2,789,212,160 B advertised-context KV. A 12 GiB
activation/trace/allocator reserve produces a conservative accounted total of
25,172,932,608 B/device. No capability was reduced: both layer kinds allocate
their local advertised-context cache and trace decode at absolute position
262,143. `doc/context_contract.json` contains the corrected figures and exact
artifact paths.

## Final Tracy, watcher, and acceptance gates

Four isolated Tracy invocations used the direct 1x4 profile fixture so that
single-chip baseline setup could not overflow or contaminate profiler markers:

```bash
GEMMA4_MULTICHIP_PROFILE=1 LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
  python -m tracy -r -p -v -o .../tracy/final/<mode> -m pytest \
  '.../test_multichip_decoder.py::test_multichip_profile[<mode>-<layer>]'
```

All four invocations passed and produced `ops_perf_results_*.csv`. Each CSV was
then filtered between its `MC_<layer>_<mode>` and `_END` signposts with
`tt-perf-report`; the exact files, hashes, and analysis command are recorded in
`tracy/final/README.md`. The final reports show 20-22 us decode RS/AG phases,
40-45 us prefill RS/AG phases, 156/172 GB/s modeled decode DRAM traffic, and
77/75 GB/s modeled prefill DRAM traffic. The explicit 24-core prefill MLP rows
are compute-bound at 69-71% modeled FLOPs rather than flagged slow.

Watcher sequence:

1. Full watcher instrumentation failed before test execution because active-
   Ethernet firmware was 27,792 B versus the 25,600 B config buffer.
2. `TT_METAL_WATCHER_NOINLINE=1` let all four mutable-trace and batch-32 stress
   tests pass, but the instrumented Ethernet router timed out during teardown.
3. `tt-smi -r all` restored a coherent four-board topology after the stale
   router invalidated a partial-reset attempt.
4. `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1
   TT_METAL_WATCHER_DISABLE_ETH=1` passed mutable traced decode for sliding and
   full attention, exited zero under shell `pipefail`, and left a worker/NoC-
   clean watcher log at `evidence/watcher_device.log`. Ethernet is the only
   disabled watcher feature. The final-source refresh passed both rows in
   125.63 s and the device log contains no error-pattern matches.

The final hash-binding command was:

```bash
pytest -q -s models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py \
  --junitxml=.../evidence/final_suite.xml
```

Result: `11 passed, 15 skipped` in 336.75 s. The opt-in exact-context,
benchmark, geometry, fractured-boundary, and profiler skips have separate
passing artifacts. Final source
hashes before documentation-only review updates:

- `multichip_decoder.py`: `79aaaa757efd1ac5955925bd6874678bc731242d1118c66c0a5d9f03bebf988d`
- `test_multichip_decoder.py`: `31e9dc8d740e32fd7568a01e7e013805228e040cfc61431bf5b904eb224c5b6a`

## Independent stage rereview

The fresh `$stage-review` rereview returned `clean-pass` after inspecting the
final source, tests, context and memory contract, manifest hashes, correctness
and trace XML, geometry/fractured-boundary closure, watcher evidence, and all
four final profiler windows. It found no required work or hard-check gaps.
The remaining decoder-MLP and prefill-attention profiler flags are controlled
by the measured winning geometry and like-for-like whole-layer benchmarks;
the Ethernet watcher limitation remains isolated to instrumentation.

## Local stage commit

The stage-owned implementation, tests, documentation, and evidence were
committed locally as `683adda7a3d12cc060df9ab3a36f1fd506eef234`. Nothing was
pushed. The follow-up commit only records this checkpoint in stage metadata.
