# TP4 multichip decoder

This stage implements the dense text decoder for `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The baseline is this run's completed `OptimizedDecoder` at commit `87140921cfc65695b74f0981b3c19b4182e46e61`. No full model, generator, or serving adapter is introduced.

## Execution contract

Open a **1×4 Blackhole mesh** after setting `ttnn.FabricConfig.FABRIC_1D_RING`. The selected collectives use `Topology.Ring` and two links. This implementation targets the four available P300_X2 ASICs (11×10 usable worker grid, eight DRAM banks each). Smaller meshes are rejected.

Construct `MultichipDecoder.from_state_dict(..., hf_config=config, layer_idx=index, mesh_device=mesh)`. The caller pins HF configuration, weights, and tokenizer to the revision above. Both phases use replicated BF16 hidden residuals; prefill input/output is `[B,1,S,5376]`, decode is `[1,1,B,5376]`. Decode returns the same eight-core width-sharded L1 contract consumed by the next layer. Prefill returns DRAM. Entry conversion is performed once; the layer stack needs no inter-layer gather. The inherited prefill interface accepts valid logical lengths without tile/page/chunk alignment, with internal padding and output slicing.

Weights use BFP4/LoFi, residuals/norms use BF16, KV uses BFP8, and prefill gate/up/intermediate uses BFP8. Prefill CCL uses BFP8 with BF16 restored before norms; decode CCL uses BF16. Q/K/V and MLP intermediate channels are column-parallel, O/down are row-parallel. Each ASIC owns eight Q heads and four sliding KV heads (D256) or one global KV head (D512). Gate/up shards are padded5376→5632 with matching zero down rows. Decode RS staging/output and AG output buffers are preallocated independently for each projection and B1/8/16/32 signature. They remain owned by the decoder while traces are live; every invocation overwrites the relevant buffer before reading it. See `mesh_plan.md` for dimensions, shard specs, collective placement, alternatives, and configuration selection.

## Cache and context

The advertised **262144-token context is preserved**. Use `sliding_cache_tokens=2048` for a bounded sliding cache; full attention retains the complete logical prefix. Allocate at least64 private physical32-token pages per active sliding sequence. The supplied logical page table has at least64 columns and enough columns for the logical context; its entries repeat each sequence's64-page pool. Pools must not alias other live sequences. Permuting physical page IDs is supported. Prefill uses slices of this logical table; decode uses generic modulo-aware cache update and SDPA operations. RoPE and mask positions remain absolute. The2048-token capacity retains the previous1024-token window and the current1024-token prefill chunk.

The generic fused cache update cannot accept a logical page count larger than the physical page count. The bounded path therefore uses two modulo-aware updates. Unbounded caches retain the faster fused update inherited from the optimized baseline. Both paths are tested; final performance profiles use the bounded sliding policy. Cache metadata, current positions and RoPE tensors remain device-resident through trace replay; caller-owned host copies in tests occur outside forward/capture.

`memory_plan.json` accounts for all60 decoder layers, TP4 embedding/head reserves, BFP tile headers, BF16 norm tile padding, and an activation/trace reserve. Keeping full-prefix sliding caches would exceed one device's DRAM across the entire model. Cyclic sliding storage avoids that without reducing logical context. The2GiB activation reserve assumes streamed prefill chunks, not all users' full-context embedded activations retained simultaneously. Full-prompt B1 validation has a larger transient input/output footprint; even an8GiB reserve fits with the planned weights and cyclic/full KV. Concurrent32768-token working sets atB32 fit analytically with streamed prefill. This is a layer-stage capacity plan, not a claim that a full model or serving engine has been built.

## Validation and artifacts

The acceptance runners compare against the single-chip optimized TTNN decoder using identical pinned real weights and recorded HF activations. PCC threshold is0.995. They check local KV ownership, shuffled page IDs, untouched unused pages, logical positions, non-aligned sequence boundaries, warmed decode traces, and runtime host-conversion guards. The stress run uses32 distinct input rows,128 advancing/per-user positions, page-table row permutations, one trace capture per signature, and another warmed cache's prefill interleaved while that trace remains live. Trace allocation tracking stays enabled, including program-cache allocations. No allocation exemptions are installed. Callers warm each intended execution signature before capturing it; all supported decode CCL buffers already exist at construction. The layer does not manage a serving trace cache.

Maximum-context tests execute the complete prefill and compare its final32 rows (the full output remains on device), then decode at position262143. Boundary and4096-token tests compare the entire logical output. `check_multichip_stack.py` feeds sliding output directly into global attention in both phases; it is a two-layer interface test only.

Commands (run device jobs serially):

```bash
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation --group boundaries
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation --group batches
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation --group stress
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation --group context
TT_METAL_WATCHER_DISABLE_ETH=1 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_validation --group watcher
TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_multichip_stack --ring --result models/autoports/google_gemma_4_31b_it/doc/multichip_decoder/stack.json
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_multichip_profiles
```

`final_layer*.json` and adjacent manifests identify exact commands, source hashes, PCC, cache checks, and host timing. Raw command logs remain local. The watcher run is separate from profiling. `tracy_final/<layer_type>/batch<batch>/` contains complete-window device metrics, raw CSV provenance, median-replay selection metadata, and human-readable/CSV tt-perf-report tables for both phases. Whole-layer roofline denominators include all operations and internal device gaps; decode averages128 complete advancing replay windows, never matmul-only subtotals or per-op utilization averages. Host trace latencies are labeled separately from device latency. Baseline device results come from `../optimized_decoder/tracy_final/` for sliding attention and the selected `../optimized_decoder/tracy_final_k32/` for full attention, with the same4096/128 workload.

The workload is a single-layer teacher-forced input stream, not token-out model generation. The model is dense; active-expert execution is not applicable. `work_log.md`, `mesh_plan.md`, `AUTODEBUG*.md`, `AUTOFIX.md`, `AUTOTRIAGE.md`, and `recovery_*.json` retain investigations, rejected candidates, and device recovery provenance. Final stage review and commit records are maintained in the work log.

Worker-core watcher checks pass for both layer types with `TT_METAL_WATCHER_DISABLE_ETH=1`. This build’s instrumented ACTIVE_ETH kernel requires 29,264 bytes against a 26,624-byte configuration buffer, so Ethernet watcher instrumentation cannot run. `watcher_configuration.json`, `watcher_eth_limit.exit.json`, and `recovery_watcher_eth.json` record the failure, scoped mitigation, and recovery. Ethernet communication remains covered by numerical, trace, cache, and stress checks.

See [performance.md](performance.md) for baseline/TP4 device latency, speedup and four-device efficiency at batches 1/8/16/32; [profile_analysis.md](profile_analysis.md) maps profiler advice to measured controls. [acceptance_summary.json](acceptance_summary.json) collects the 34 passing cases. [available_hardware.json](available_hardware.json) records machine capacity separately from the selected mesh.

For the required 4096-input/128-decode/B1 workload, whole-layer TP4 device times are **11.423 ms prefill / 484.8 us decode** for sliding attention and **13.748 ms / 534.8 us** for full attention. Against the completed single-chip baseline, prefill/decode speedups are **1.92×/1.98×** and **2.30×/2.01×**, respectively. Four-device efficiencies are **48.0%/49.5%** and **57.5%/50.2%**. Each layer type is reported separately.
