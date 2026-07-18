# Llama 3.1 8B multichip decoder

This stage implements one dense Llama decoder layer on the exact hardware in this checkout: a fixed `1x2` Blackhole P300c mesh. `OptimizedDecoder` is the single-chip TTNN baseline. Full-model, generator, and vLLM work are intentionally outside this stage.

## Final runtime contract

- TP degree is fixed at two. Packed QKV and the separate gate/up projections are output-column sharded; O and down are input-row sharded.
- Each rank owns 16 query heads, four KV heads, 2,048 attention features, and 7,168 MLP features. Each row-parallel projection produces a full BF16 partial followed by `all_reduce_async`.
- All decoder layers on a mesh resolve through the common per-mesh `TT_CCL` owner (or accept that owner explicitly). A hardware-free 32-layer construction test proves one shared set of 36 persistent semaphore handles rather than 32 per-layer sets.
- The stacked boundary is a replicated logical `[1, batch, sequence, 4096]` tensor. Decode keeps each replica width-sharded in L1 on 32 cores; the selected O projection uses eight cores.
- Weights use the optimized baseline's BFP4/LoFi policy. The final collective dtype is BF16; measured BFP8 collectives were slower.
- Contiguous caches have local shape `[batch, 4, max_cache_len, 128]`. Paged caches have local shape `[blocks, 4, 64, 128]`; page tables are replicated and physical cache heads remain rank-local. BF16 and BFP8 cache storage work.
- Logical sequence lengths are not required to align to a tile or page. Sequences 7 and 31 are covered, and ordered paged decode crosses positions 63, 64, and 65; padding remains internal.
- This model is dense, so MoE/expert strategy is not applicable.

The requested fabric configuration/topology is `FABRIC_1D_RING`/`Topology.Ring` with two links. The physical P300 pair has no wrap edge, so TTNN resolves topology-sensitive operations to the usable two-device line. This matters for fused all-gather–matmul, which rejects linear topology; the selected generic all-reduce path is supported and trace-safe.

`doc/context_contract.json` retains the advertised 131,072-token logical context. TP2 leaves four KV heads per device. Across 32 layers the per-device cache is exactly 8.0 GiB BF16 or 4.25 GiB BFP8, and projection weights are 1.828 GiB/device. The allocator reports 34,178,731,008 bytes/device, so no context reduction is required. A 2,048-block aggregate pool with 64-token pages can be assigned to one maximum-length request or shared across requests.

## Correctness and trace evidence

Final command:

```bash
pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k 'runtime_path_is_real_multichip or context_capacity_contract or stack_shares_one_ccl_owner or multichip_correctness'
```

All five items pass with production-shape BFP4 weights. PCC against the actual single-chip `OptimizedDecoder` is:

| Contract | PCC |
| --- | ---: |
| non-aligned prefill | 0.999999174 |
| follow-on decode | 0.999987356 |
| contiguous key / value cache | 0.999987048 / 0.999988071 |
| paged decode, three runs | 0.999986244 each |
| paged physical key / value blocks | 0.999986861 / 0.999983236 |
| non-aligned 31-token prefill before boundary walk | 0.999999547 |
| paged decode at positions 63 / 64 / 65 | 0.999966946 / 0.999967534 / 0.999965821 |
| physical page-0 K, ranks 0 / 1 | 0.999934224 / 0.999934638 |
| physical page-1 K, ranks 0 / 1 | 0.999886938 / 0.999888267 |
| physical page-0 V, ranks 0 / 1 | 0.999953822 / 0.999953993 |
| physical page-1 V, ranks 0 / 1 | 0.999919675 / 0.999919991 |
| five warmed page-1 BFP8 decode trace replays | 1.0 versus eager; bitwise identical |

Batch 32, nonidentity/disjoint physical page mappings, both page-table columns, page-1 unwritten suffixes, per-rank local cache heads, stacked output layout, repeated determinism, and trace replay are all checked. The boundary case uses a non-aligned 31-token prefill followed by ordered decode writes through position 65 because the optimized baseline's padded-64 prefill MLP exceeds Blackhole L1; this does not align or truncate the multichip public API. The static runtime audit rejects host conversions, Torch math, and parent/single-chip forward delegation in the multichip forward path.

## Performance and topology decision

Final default command uses 50 warmed prefill iterations and 1,000 decode trace replays:

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_BATCH=1 \
MULTICHIP_DECODER_SEQ_LEN=18 MULTICHIP_DECODER_PREFILL_REPEATS=50 \
MULTICHIP_DECODER_TRACE_REPLAYS=1000 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

| Path | Single chip | TP2 | Speedup | TP efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill | 1.243528 ms | 0.638309 ms | 1.948161x | 97.4081% |
| traced decode | 0.582049 ms | 0.401595 ms | 1.449345x | 72.4672% |

The TP2 output PCC in the performance run is 0.999999745. `candidate_results.csv` records the output-core, geometry, link-count, packed gate/up, and collective-dtype sweep.

A shape-faithful fractured-residual boundary was also measured with BFP4 weights and 1,000 trace replays. The supported persistent `reduce-scatter -> local add -> distributed RMSNorm -> all-gather -> QKV matmul` path has PCC 0.999976 on both ranks but takes 0.099587 ms versus 0.083456 ms for `all-reduce -> replicated add/RMSNorm -> QKV matmul`: 19.3% slower. Fused AGMM was adapted through its rank, shard-count, and weight-layout requirements, then hit the hard usable-topology restriction described above. The replicated residual is therefore the measured winner, not just the simplest implementation.

## Profiler, watcher, and artifacts

- `tracy/README.md` gives the exact Tracy/`tt-perf-report` command and the DRAM, compute, CCL, and data-movement interpretation. The canonical source CSV and all filtered human-readable/CSV tables are committed.
- `watcher_summary.txt` records the clean target-only watcher stress run, rerun after shared-CCL wiring. Worker, NoC, CB, stack, dispatch, and assert checks stayed enabled; active-Ethernet instrumentation was disabled because firmware 19.8.0 otherwise fails only while restoring Ethernet firmware after a fully passing test.
- `topology_probe_plan.md` records every residual/topology family, exact shapes, persistent-buffer plan, and rejected alternatives.
- `shard_advise/report.txt` records the fresh advisor blocker: the installed prebuilt tt-mlir toolchain has an ABI mismatch with this checkout. Per skill guidance, no local tt-mlir rebuild was substituted.
- `triage/AUTODEBUG.md` distinguishes pytest traceback tensor reads, real L1 CB overflows in BF16/BFP8 experimental policies, the repaired prefill CCL memory mismatch, and the fixture lifetime constraint.
- `work_log.md` is the detailed command/provenance log and records the stage-review verdict and local commit SHA.
- `final_gate_results.txt` is the concise retained post-review correctness, trace, stack-ownership, watcher, and performance transcript.
