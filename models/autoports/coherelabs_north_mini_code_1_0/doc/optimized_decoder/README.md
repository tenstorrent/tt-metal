# North-Mini-Code-1.0 optimized decoder

Status: complete; independent review clean-pass. Implementation checkpoint:
`26ffee67839`.

This stage optimizes the single-device decoder implemented by
`tt/functional_decoder.py`.  It preserves the public prefill/decode, paged
KV-cache, logical sequence-length, and context-capacity contracts.  The
optimized implementation and tests are intentionally separate from the
functional stage; measured optimized tests must instantiate
`tt.optimized_decoder.OptimizedDecoder`.

## Starting baseline

The functional stage is commit `4e45a256771` with context-contract follow-up
`78dbd88bec7`.  Its filtered `tt-perf-report` windows use real target tensor
shapes, BF16 weights/activations/cache, default matmul kernels, and zero host
ops.

| Kind / workload | Warm wall latency | Filtered device time |
|---|---:|---:|
| dense layer 0, prefill 128, batch 1 | 0.636 ms | 0.586 ms |
| dense layer 0, traced decode, batch 1 | 0.356 ms | 0.338 ms |
| sliding-MoE layer 1, prefill 128, batch 1 | 14.908 ms | 14.644 ms |
| sliding-MoE layer 1, traced decode, batch 1 | 9.528 ms | 9.452 ms |
| full-MoE layer 4, prefill 128, batch 1 | 14.655 ms | 14.567 ms |
| full-MoE layer 4, traced decode, batch 1 | 9.524 ms | 9.439 ms |
| dense layer 0, traced decode, batch 32 | 6.652 ms | 6.614 ms |
| sliding-MoE layer 1, traced decode, batch 32 | 11.122 ms | 11.084 ms |
| full-MoE layer 4, traced decode, batch 32 | 11.129 ms | 11.077 ms |

The requested serving batch is 32 according to `doc/context_contract.json`.
Batch 1 remains the primary latency target; every material layout/matmul family
will therefore be measured independently at logical batches 1 and 32.

## Operation-topology audit

This is the source-plus-profiler audit made before tuning.  Percentages are
from the functional batch-1 sliding-MoE decode window unless stated otherwise.

| Current sequence / boundary | Current evidence | Candidate | Action |
|---|---|---|---|
| RMSNorm -> packed QKV | one-core BF16 norm 26 us; one packed `2048 x 5120` projection 54 us | width-sharded L1 residual/norm; BFP8/BFP4 attention weights; DRAM-sharded decode matmul | Sweep after the dominant expert path; retain packed QKV topology |
| packed QKV -> create heads -> RoPE -> paged update -> paged SDPA | already one QKV matmul; two 1-us RoPE metadata interleaved-to-sharded conversions; SDPA 9 us | explicit SDPA config; BFP8 KV cache; persistent pre-sharded RoPE inputs if caller contract permits | Measure total attention contract, not projection alone |
| concat heads -> O projection | one layout conversion, 1 us; BF16 O matmul 41 us | DRAM-sharded BFP8/BFP4 LoFi matmul with L1 output | Sweep with QKV precision family |
| normalized input -> router -> TopK -> scatter | router 25 us; TopK 25 us; scatter/pad/layout cluster is small but feeds an avoidable dense expert expansion | retain exact sigmoid top-8 routing; produce sparse-matmul sparsity directly; verify exact nonzero count before setting `nnz` | Replace dense all-expert topology; omit static `nnz` unless exact post-BF16 count is proved |
| normalized input -> repeat across 128 experts | repeat plus tilize cluster about 0.2 ms decode and 0.4 ms prefill | routed `ttnn.sparse_matmul` consumes the unreplicated token and sparsity | Remove repeat and dense all-expert activation materialization |
| repeated input -> separate expert gate/up matmuls | 3.477 + 3.476 ms, 73.0% decode; BF16 HiFi2; 24 cores; 122 GB/s | routed sparse gate/up; BFP8 baseline; mandatory BFP4/LoFi; packed gate-up versus best separate sparse family; geometry/in0-block sweep | Highest priority |
| gate -> SiLU -> multiply | 52 + 48 us | fused activation/binary form where legal; keep L1 intermediates | Include in whole-MLP comparison |
| activated experts -> dense expert down matmul | 1.727 ms, 18.1% decode; BF16 HiFi2; 64 cores; 247 GB/s | sparse-input routed down; BFP8 then guarded BFP4/LoFi; geometry/in0-block sweep | Highest priority |
| expert outputs -> routing multiply -> expert reduce | about 0.23 ms, including fill/reduce in decode | keep expert intermediates in L1; eliminate DRAM round trips; reduce only active/routed results | Measure with sparse family |
| hidden + attention + MLP | two BF16 DRAM-interleaved adds | width-sharded L1 residual stream carried through norm/add/projections where legal | Measure as a coherent layer chain |
| prefill MoE chunks | public arbitrary logical length; internal chunks of at most 1024 | phase-specific large-M program configs; preserve short final chunk and internal padding | No public alignment restriction |

There are no collectives in this single-chip stage, so CCL topology, fused
matmul-CCL, and persistent CCL-buffer checklist items are not applicable.
There is no LM head or sampling in decoder scope.

## Evidence index

Candidate tables, commands, PCC, traced timings, profiler conclusions, watcher
results, limitations, and commit SHAs will be appended to `work_log.md`.

## Current selected topology

The construction-time default is phase and batch specific, with no runtime
fallback:

| Path | Selected topology / precision |
|---|---|
| dense layer | packed gate/up, BFP8 weights, HiFi2 |
| MoE prefill | packed all-expert gate/up, BFP8 weights |
| MoE routing, all phases | FP32 on-device accumulation, BF16 top-k input; preserves stable expert choice at the functional PCC bar |
| MoE decode batch 1 | one packed routed gate/up `sparse_matmul` plus routed sparse down, BFP4/LoFi; 24-core gate/up and 64-core down |
| MoE decode batch 32 | packed all-expert gate/up, BFP4/LoFi; explicit 48-core gate/up and 64-core down configs |
| attention/cache batch 1 | packed QKV, explicit SDPA, interleaved BFP8 weights, BFP8 paged KV |
| attention/cache batch 32 | packed QKV, explicit SDPA, DRAM-width-sharded BFP8/LoFi weights, BFP8 paged KV |

## Final performance

All values are warmed wall latency. Decode is trace-captured and replayed.
Batch 32 is the serving batch from the context contract. No selected row
regresses its like-for-like functional baseline.

| Kind | Workload | Functional | Optimized final | Change |
|---|---|---:|---:|---:|
| dense layer 0 | prefill 128, batch 1 | 0.636 ms | 0.543 ms | 14.6% faster |
| sliding-MoE layer 1 | prefill 128, batch 1 | 14.908 ms | 10.458 ms | 29.8% faster |
| full-MoE layer 4 | prefill 128, batch 1 | 14.655 ms | 10.230 ms | 30.2% faster |
| dense layer 0 | traced decode, batch 1 | 0.356 ms | 0.293 ms | 17.7% faster |
| sliding-MoE layer 1 | traced decode, batch 1 | 9.528 ms | 0.847 ms | 91.1% faster |
| full-MoE layer 4 | traced decode, batch 1 | 9.524 ms | 0.846 ms | 91.1% faster |
| dense layer 0 | prefill 128, batch 32 | 13.732 ms | 5.322 ms | 61.2% faster |
| sliding-MoE layer 1 | prefill 128, batch 32 | 146.050 ms | 105.219 ms | 28.0% faster |
| full-MoE layer 4 | prefill 128, batch 32 | 145.694 ms | 104.994 ms | 27.9% faster |
| dense layer 0 | traced decode, batch 32 | 6.652 ms | 5.007 ms | 24.7% faster |
| sliding-MoE layer 1 | traced decode, batch 32 | 11.122 ms | 2.542 ms | 77.1% faster |
| full-MoE layer 4 | traced decode, batch 32 | 11.129 ms | 2.533 ms | 77.2% faster |

The operation-level Tracy windows under `tracy/` cover every layer kind at
batch 1 prefill/decode and serving-batch decode. Filtered reports contain no
host operations. The final runtime methods contain no Torch conversion or host
fallback; Torch is imported only during construction to pack checkpoint
weights. `tt-perf-report` reports modeled DRAM roofline utilization of 17.4%
for dense prefill, 16.2–16.3% for MoE prefill, 27.9% for dense batch-1 decode,
7.2% for sparse MoE batch-1 decode (modeled with four active expert groups),
3.1% for dense batch-32 decode, and 29.3% for the remediated MoE batch-32
decode.

The official layer-1 checkpoint passes the optimized BFP4 routed decode at PCC
0.995917.  That run also exposed and fixed singleton-rank elision in real
`sparse_matmul` down outputs.  Repeated nonzero-position decode is bitwise
deterministic, while physical BFP8 paged-cache slots have key/value PCC
0.999878–0.999897.

Packed MoE prefill initially appeared to have a layer-kind-dependent floor
even with BF16 experts. Component isolation instead found the discontinuity in
router top-k: BF16 accumulation changed the selected expert set for 7 of 33
synthetic tokens. FP32 router accumulation on device, followed by a BF16 cast
required by the top-k kernel, restores the functional `0.995` bar without
changing expert topology. Final selected PCC spans `0.995368–0.996838` for
non-aligned prefill and `0.997540–0.997681` for dynamic trace replay.

## Validation and limitations

The exact final-source optimized-only suite passes 28 tests under Watcher. It covers dense,
sliding-window MoE, and full-attention MoE layers; logical lengths 31 and 33;
paged prefill followed by traced decode; nonidentity physical page slots;
bitwise repeated-run determinism; dynamic trace input replay at batches 1 and
32; explicit selected packed candidates; and official checkpoint layer-1
weights. The preserved console log, JUnit report, and 3,248-line Watcher log
are under `watcher/final_full_suite`; the run completed in 287.22 seconds
and the Watcher signature scan was clean.

The BFP8 cache halves the functional BF16 capacity cost, so
`doc/context_contract.json` now records 512,000,000 bytes at batch 1 and
16,384,000,000 bytes at batch 32 for the unchanged advertised 500,000-token
context. Functional near-limit execution remains the capability floor; the
optimized tests additionally verify physical BFP8 cache writes and reads.

Two one-microsecond RoPE metadata interleaved-to-sharded conversions remain in
decode. Keeping the residual/norm stream width-sharded would require converting
back before the selected interleaved batch-1 attention projections and before
the sparse expert input, adding boundaries rather than removing them. At batch
32, DRAM-width-sharded QKV/O is retained only across each legal matmul boundary;
SDPA and the packed expert family consume different layouts. This is the
lowest-movement coherent layout contract found in the measured families, not a
public layout restriction.
