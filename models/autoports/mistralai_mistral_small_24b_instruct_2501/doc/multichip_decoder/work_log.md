# Multichip decoder work log

## 2026-07-17: baseline and plan freeze

- Starting repository checkpoint: `64608f66cd8`. Completed optimized-decoder checkpoint: `92f5a3` (as recorded by the preceding stage).
- Read the functional and optimized decoder code/tests/evidence, `doc/context_contract.json`, and layer-20 compiler multichip provenance before choosing the target.
- `timeout 60 tt-smi -ls --local` found four Blackhole p300c devices. A real `ttnn.MeshShape(1, 4)`/`FABRIC_1D` smoke opened logical device IDs `[3, 2, 1, 0]`; every rank reported compute grid `11x10`, DRAM grid `8x1`, and 34,178,731,008 DRAM bytes.
- Froze TP4 on mesh axis 1 with replicated residual state and two all-reduces per layer. `mesh_plan.md` was written before the final path and records exact tensor/cache shapes, tile-aware capacity arithmetic, padding, and rejected strategies.

## Implementation

- Added `MultichipDecoder(OptimizedDecoder)`. Weight packing is per-rank Q/K/V and gate/up; QKV, gate, and up are column sharded; output/down are row sharded. Local shapes are Q 8 heads, K/V 2 heads, attention width 1024, QKV width 1536, and intermediate width 8192.
- Kept hidden input/output replicated, cache KV-head-sharded, and page tables/positions replicated. Both contiguous and paged cache paths validate local layout. The public logical sequence is never rounded; internal tile/chunk padding is sliced away.
- Reused optimized BFP4/LoFi matmuls, BF16 activations, BFP8 caches, bounded prefill MLP chunks, DRAM-sharded weights, and 11-core advisor RMSNorm layouts. The runtime owns its graph and never calls the optimized forward path or host conversion.
- Added trace, direct optimized-baseline, page-table/physical cache, 32K capacity, performance/determinism, collective-consumer, and runtime-fallback tests.
- A fresh stage review found that the first trace API accepted both scalar and tensor positions but still used the scalar for RoPE. AutoFix replaced that split contract with one persistent replicated INT32 `[batch]` tensor that drives device-side RoPE embedding, cache updates, and SDPA. Three changing nonuniform vectors pass in one trace for contiguous and reversed-page-table modes.
- The same review found that the first capacity estimate counted only one layer's cache and omitted duplicate prefill weights/RoPE. The final path shares immutable RoPE across layers, provides `release_prefill_weights()` for decode handoff, and physically validates all 40 weight/cache lifetimes plus a 4 GiB/rank reserve.
- Final rereview found that checking only `get_num_devices()==4` admitted 2x2/4x1 meshes even though collectives use axis 1. AutoFix now requires `tuple(mesh_device.shape)==(1,4)` before touching model state. A hardware-free parameterized regression proves both wrong four-device shapes are rejected.

## Correctness and capacity commands

Here `$MODEL_DIR` is `models/autoports/mistralai_mistral_small_24b_instruct_2501` and `$SNAPSHOT` is `/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724`.

```bash
tt-smi -r
pytest -q -s --junitxml=$MODEL_DIR/doc/multichip_decoder/evidence/synthetic_pcc_trace.xml \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_synthetic_prefill_decode_pcc_layout_and_trace

tt-smi -r
pytest -q -s --junitxml=$MODEL_DIR/doc/multichip_decoder/evidence/paged_cache.xml \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_paged_cache_page_table_and_positions

tt-smi -r
MISTRAL_SMALL_24B_MULTICHIP_CAPACITY=1 pytest -q -s \
  --junitxml=$MODEL_DIR/doc/multichip_decoder/evidence/capacity_32768.xml \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_full_context_paged_cache_capacity
```

Final HF PCCs were 0.995336/0.995353/0.995555 for prefill lengths 17/18/32, 0.995057 for final-geometry decode, and about 0.9937 for written K/V. The matched paged/contiguous control is exactly 1.0 for prefill, logical K/V, and decode; both decode modes are 0.995011 versus HF. Physical paged K/V mapping is 0.993746/0.993669. The mismatch reviewed in the first result was caused by paged SDPA dynamically choosing `k_chunk=32` while the contiguous control used 128; fixing both to 128 removed it.

The real optimized comparison ran the same test twice, first with `MISTRAL_SMALL_24B_MULTICHIP_BASELINE_IMPL=optimized`, then `multichip`, with `MISTRAL_SMALL_24B_MULTICHIP_BASELINE_REAL=1`, `MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT`, and a shared temporary artifact path. Final prefill/decode/K/V PCCs were 0.999994/0.999990/1.0/1.0. The 38 MiB temporary tensor file was deleted after comparison; the two command logs and JUnit files reproduce it.

The full capacity gate keeps two real decoders sharing RoPE, releases their prefill matrices, allocates exact decode constants for the other 38 layers, all 40 cache pairs, TP embedding/head, final norm, page table, positions, and eight BF16 reserve tensors totaling 4 GiB/rank. With everything live, paged decode passes at position 32767. Exact steady-state arithmetic and the 37,344-token calculated ceiling are in `mesh_plan.md` and `context_contract.json`.

## Collective selection

- A 327,680-byte BF16 decode-payload microbenchmark measured Linear 0.114246 ms and Ring 0.117872 ms; selected Linear with two links.
- The required consumer-chain comparison used 50 traced iterations. Replicated `all_reduce → advisor RMSNorm → QKV` was 0.102029 ms. Hidden-sharded `reduce_scatter → distributed RMSNorm stats all_gather → hidden all_gather → QKV` was 0.155569 ms, 1.524752× slower, with per-rank PCC at least 0.99974.
- The 53.54 us deficit exceeds the final all-gather's roughly 20 us, so even perfect gather/QKV overlap cannot recover it. Ring was already slower. The synchronous all-reduce API's internal scratch has no user-visible persistent buffer requirement and is trace safe.

## Performance and profiler

Real-weight timing command, run once with `optimized` and once with `multichip`:

```bash
tt-smi -r
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_IMPL=multichip \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=50 pytest -q -s \
  --junitxml=$MODEL_DIR/doc/multichip_decoder/evidence/perf_multichip.xml \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_warmed_single_chip_and_multichip_perf
```

Final exact-code warmed results: TP1 5.399418 ms prefill and 1.288697 ms decode; TP4 3.717828 ms and 0.579850 ms. Speedups are 1.452304× and 2.222466×; efficiencies are 36.308% and 55.562%. The test also verifies eager/trace PCC 1.0 and exact output/K/V equality after 50 replay writes.

The TP-local sweep screened attention and MLP core/block variants at fixed precision with real weights and PCC >= 0.9999. Three fresh runs of the original geometry were 0.610086/0.609471/0.609677 ms (median 0.609677). The selected combined geometry was 0.579591/0.579596/0.579498 ms (median 0.579591), a 4.934% improvement at output PCC 0.999989. Wider attention grids, 80-core MLP, and alternate intermediate layouts were slower. `evidence/geometry_sweep.csv` and `geometry_*.log` retain every result.

Accepted profiler capture and report generation:

```bash
tt-smi -r
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_IMPL=multichip \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=1 python -m tracy -r -p -v \
  -o $MODEL_DIR/doc/multichip_decoder/tracy/final -m pytest -q -s \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_warmed_single_chip_and_multichip_perf

gzip -dc $MODEL_DIR/doc/multichip_decoder/tracy/final/ops.csv.gz >/tmp/mistral-small-24b-ops.csv
tt-perf-report --no-color --start-signpost MULTICHIP_DECODE \
  --end-signpost MULTICHIP_DECODE_END \
  --csv $MODEL_DIR/doc/multichip_decoder/tracy/final/decode_perf_report.csv \
  --summary-file $MODEL_DIR/doc/multichip_decoder/tracy/final/decode_summary \
  /tmp/mistral-small-24b-ops.csv
tt-perf-report --no-color --no-merge-devices --start-signpost MULTICHIP_DECODE \
  --end-signpost MULTICHIP_DECODE_END \
  --csv $MODEL_DIR/doc/multichip_decoder/tracy/final/decode_perf_report_per_device.csv \
  /tmp/mistral-small-24b-ops.csv
```

Equivalent prefill reports use `MULTICHIP_PREFILL`/`MULTICHIP_PREFILL_END`. The final profile contains 42 representative device ops, zero host ops, and 575 us device time. Matmuls total 265.56 us, CCL 83.08 us, and explicit data movement about 27.57 us; modeled DRAM is 121 GB/s. The merged 96.6 ms cross-device row-order gap is a known row-order artifact contradicted by the per-device table and 0.579850 ms warmed wall result; all processed raw rows are retained.

Superseded profiler captures, duplicated tool output directories, and the temporary baseline artifact were removed to keep the stage evidence around 1.5 MiB. They are not recoverable in place; all retained reports and exact commands permit regeneration.

## Watcher and recovery

Default `TT_METAL_WATCHER=10` failed before model execution because an active-Ethernet watcher kernel was 27,776 bytes versus Blackhole's 25,600-byte configuration region. `TT_METAL_WATCHER_NOINLINE=1` fits and passes the model, but firmware 19.8.0 then times out returning active Ethernet core 29-25 to base firmware during process teardown and exits 134. The final clean worker-watcher command was:

```bash
tt-smi -r
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_LOGGER_LEVEL=info timeout 900 pytest -q -s \
  --junitxml=$MODEL_DIR/doc/multichip_decoder/evidence/watcher_final_worker.xml \
  $MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_synthetic_prefill_decode_pcc_layout_and_trace
```

It passed in 23.61 s, reported only `disabled features: ETH`, produced no worker watcher/NoC/assert markers, detached devices 0–3, and exited 0. Fabric/CCL remained enabled and exercised; only active-ETH watcher inspection was disabled. The default-size and full-ETH teardown failures are retained, and `tt-smi -r` recovered the board after each fault.

The final exact-code consolidated suite passed 5 required gates (3 separately parameterized artifact tests skipped by design) in 102.64 s. A post-suite `tt-smi -ls --local` showed all four p300c devices visible and reset-capable.

The inspector permission warning and `/dev/shm` warning are machine-environment limitations, not model failures. `runtime_fallback_audit.xml`, Python compilation, JSON parsing, and `git diff --check` complete the non-hardware checks.

## Review and commits

The first independent review returned `more-work-needed`; all five findings were repaired through AutoFix. A rereview then found the missing exact `(1,4)` mesh-shape guard; the constructor and two wrong-shape regressions were added. The next fresh independent rereview returned `clean-pass` with no required work. `evidence/stage_review.md` records the sequence and final verification scope.

Implementation, tests, documentation, and retained evidence were committed locally as
`f26e3ae24e68b3e3e1517f4b8bb9da31f8f5b58f`. This work-log checkpoint is committed as
its immediate successor at the final stage `HEAD`; neither commit is pushed. The unrelated
dirty `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` remains explicitly excluded.
