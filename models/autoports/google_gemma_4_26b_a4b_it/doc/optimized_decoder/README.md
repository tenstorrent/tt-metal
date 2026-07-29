# Gemma 4 26B-A4B optimized decoder

This directory contains the single-device optimized-decoder evidence for
`google/gemma-4-26B-A4B-it` on one Blackhole P300C. The implementation is
`../../tt/optimized_decoder.py`; it inherits the functional decoder's public
prefill, decode, paged-KV, trace, and context interfaces, but overrides every
material measured attention, dense-MLP, and routed-expert path. The optimized
tests assert those overrides were entered, so a functional fallback cannot
pass the stage.

## Selected policy

- BF16 activations, residuals, norms, attention weights, and KV cache.
- HiFi4 sliding-attention, HiFi2 full-attention, and HiFi2 dense-MLP
  compute.
- BFP8 dense-MLP weights.
- BFP8 routed-expert weights with LoFi compute and sparse gate/down
  `in0_block_w=11`; exact `nnz=8` is retained from the router contract.
- BFP8 prefill-expert weights loaded directly from the HF tensors. Expert
  prefill is split into proven one-tile (32-token) sparse invocations; larger
  grouped invocations remain diagnostic-only after failing the complete
  non-aligned boundary matrix.
- Interleaved dense decode projections. The faster-looking DRAM-sharded family
  is available as a setup-time candidate, but is off by default because the
  serving-batch HF oracle found a deterministic single-user accuracy cliff.
- Separate dense gate/up projections. The packed candidate is retained behind
  a setup-time option but rejected by real-weight PCC.

No measured hot method contains Torch, `from_torch`, `to_torch`, or host
fallback. All host tensor work is setup-only. The remaining layout conversions
are required by QKV/head, paged-SDPA, concat-head, sparse-expert, or rejected
DRAM-sharded projection contracts.

## Correctness and capability

The final real-weight PCC results (functional bar `0.995`) are:

| Layer/cache case | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| sliding attention, shared physical cache | 0.998631 | 0.999636 |
| full attention, natural cache | 0.997686 | 0.999836 |
| full attention, shared-cache view | 0.997686 | 0.999836 |

Logical prefill lengths `1, 31, 32, 33, 63/64/65` or `127/128/129`, and
`1023/1024/1025` pass for the applicable layer kind. Physical real-weight
prefill also passes at the non-aligned advertised length `262143` for both
layer kinds, and traced decode passes at current position `262143`. Therefore
`doc/context_contract.json` remains unchanged at 262144; cache dtype, cache
layout, and allocation topology were not changed.

Traced serving-batch tests cover batch 1 and 32, require eager/replay and
repeat/replay PCC `>=0.9999`, and add a batch-32 per-user tail guard. The final
sliding batch-32 aggregate HF PCC is `0.999538`, minimum user PCC is `0.998238`,
and user matching is identity. Mutable stable-buffer replay, batch-2 prefill,
natural/shared paged-cache cases, and a watcher-clean seven-case run also pass.

## Performance

All figures are warmed host latency for one real decoder layer at sequence or
current position 1024. Decode is trace replay. Negative delta is faster.

| Phase / layer kind | Batch | Functional | Optimized | Delta |
| --- | ---: | ---: | ---: | ---: |
| prefill / sliding | 1 | 680.955 ms | 120.697 ms | -82.27% |
| prefill / full | 1 | 681.880 ms | 121.500 ms | -82.18% |
| traced decode / sliding | 1 | 3.019 ms | 1.883 ms | -37.63% |
| traced decode / full | 1 | 3.201 ms | 2.051 ms | -35.93% |
| prefill / sliding | 32 | 21780.254 ms | 3856.548 ms | -82.29% |
| prefill / full | 32 | 21818.995 ms | 3884.307 ms | -82.20% |
| traced decode / sliding | 32 | 68.879 ms | 32.202 ms | -53.25% |
| traced decode / full | 32 | 68.646 ms | 31.994 ms | -53.39% |

The primary batch-1 decode target beats the best correct functional baseline,
and serving batch 32 does not regress.

## Profiler conclusion

`final_profile_tracy_midrun/ops_perf_results.csv` is the retained reduced
one-layer Tracy capture. Its advice-enabled `prefill_perf_report.*` and
`decode_perf_report.*` were produced by modern Blackhole-aware enrichment
with mid-run device dumps; the initial non-dumped attempt is retained
in `work_log.md` as a documented failed profiling attempt.

The final prefill window contains 119.993 ms of device work plus 0.482 ms of
op gaps, matching the 120.697 ms warmed host result. Sparse expert matmuls are
99.627 ms (83.0% of device time); QKV is 485 us, SDPA 717 us, and dense
gate/up about 75 us each. The traced-decode window contains 1.822 ms of device
work plus 0.072 ms of gaps, matching the 1.883 ms warmed replay. Its three
sparse expert matmuls total 742 us (40.7%); QKV is 114 us at 80.4% modeled
DRAM bandwidth. The report verifies BF16 attention, BFP8 dense/expert weights,
and the selected fidelity policy.

See `work_log.md` for the topology audit, candidate ledger, exact commands,
AutoFix investigation, watcher/profiler details, limitations, and completed
optimization checklist.
