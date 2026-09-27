# Gemma4 optimized multichip decoder — Stage 05

Status: **accepted — independent [clean-pass](stage_review.md)**. Model
`google/gemma-4-26B-A4B-it`, baseline commit `adcb0e8f21`, target TP4 on a
1×4 mesh of four Blackhole P300c ASICs. This stage changes the decoder in
place. It does not implement a full model or serving integration.

## Workload and current measurements

The headline workload is 4096 input tokens, 128 advancing traced decode
positions, batch 1, one concurrent request. Real checkpoint weights and recorded
model activations are used. TP1 is the correctness reference; TP4 is the timed
path. The unchanged output/cache PCC gate is 0.995.

| Path | Sliding prefill / decode µs | Full prefill / decode µs |
| --- | ---: | ---: |
| Reproduced Stage04 baseline | 93454.934 / 650.322 | 79042.991 / 724.813 |
| Current final-default run | 93346.919 / 653.344 | 79004.977 / 701.861 |

These are warmed **host-wall** medians, not device time. The full layer's decode
is 3.17% faster; sliding is 0.46% slower than the original lower-precision
baseline. Sliding now passes the expanded real adjacent-layer accuracy gate;
it is 0.48% faster than the accuracy-matched unoptimized control (656.489 µs).
Do not substitute an earlier faster candidate's latency for the final default.
Final device profiles are complete. Whole-layer prefill/decode device times are
329194.676/710.513 µs for sliding and339373.403/744.896 µs for full attention.
See [device findings and advice disposition](final_perf_findings.md) for baseline
comparisons, complete-window rooflines and the profiling-overhead distinction.

Current final single-layer minimum decode PCC is 0.997643887 sliding and
0.996542932 full. The batch-32 decode minima are 0.997691369 and 0.995505438.
Actual adjacent layers 0→1 and 4→5 pass at both 33 and 4096 input tokens with
128 advancing decode steps. The layer-4 fixtures are recomputed through HF
layers 0–3 for each exact token history. Diagnostic 0→5 compositions skip four
real layers and are not used to veto precision. The 33-token 0→1 minimum,
0.995049733, has little numerical margin; this is a limitation of this tested
policy, not a relaxed gate.

## Selected decoder contract

| Group | Sliding attention | Full attention |
| --- | --- | --- |
| Decode QKV weights | BFP8 | BFP4; retained BFP8 prefill |
| Output projection weights | BFP8 | BFP8 |
| Routed expert gate/up / down | BFP8 / BFP4 | BFP4 / BFP4 |
| Shared gate/up / down | BFP4 / BFP8 | BFP4 / BFP4 |
| Attention / paired-MoE CCL payload | BF16 / BFP8 | BFP8 / BF16 |
| KV cache | BFP8, paged local heads | BFP8, paged local heads |

The final path retains gate-selected top-8 indexed sparse expert execution;
prefill uses EP4 active expert unions. Tensor-parallel attention, weights and
cache heads remain distributed. Replicated residuals do not mean replicated
layer computation. Packed QKV and gate/up projections, role-specific matmul
geometry, asynchronous Linear RS/AG, persistent L1 collective buffers and
private full-worker-grid semaphores are retained.

The [inter-layer residual contract](residual_contract.md) is replicated BF16,
TILE, DRAM-interleaved `[1,1,S,2816]`, passed directly into the next decoder.
There is no inter-layer collective or reshard. Public logical lengths need not
be aligned; padding, masks and slicing belong to the decoder. A caller-owned
`CollectiveBufferPool` may be shared by serial layers on the same mesh and
command queue. It must outlive their traces and must not be shared by concurrent
requests, threads or queues. Semaphores remain private to each layer.

## Optimization and repairs

The initial operation-topology audit is in [work_log.md](work_log.md).
[Candidate comparisons](candidate_comparisons.md), `candidate_summary.csv`,
and each result/command JSON record the geometry, fidelity, dtype, placement,
packed/separate projection, DRAM-sharding and persistent-buffer comparisons.
Carried mesh-sharded residual candidates consume local hidden width 704 through
distributed norms and residual updates. Their harness-only gather is excluded
from timing; the family is not measured with an immediate replicated restore.
Fused QKV AGMM, output-column WO AGMM and output MMRS are separate comparisons.

- [DRAM mesh repair](AUTOFIX_dram_mesh.md): secondary-reader placement passed a
  multi-device mesh to a physical-device hop query. The complete changed C++
  translation unit compiled and the shared library linked using the existing
  native toolchain. The required Docker wrapper was attempted but Docker is
  unavailable. All three reader-count hardware regressions passed. Exact build
  commands, native binary hash and limitations are retained in the report.
- [L1 residual repair](AUTOFIX_l1_residual.md): the mixed FP32/BF16 final add
  exceeded the fast path's effective register capacity. The supported SFPU add
  restores correctness. The adapted full-layer L1 residual family was slower.
- [CCL semaphore coverage](AUTODEBUG_ccl_semaphore_grid.md): imported 8×8
  semaphore storage omitted workers chosen on the actual 11×10 grid. Full-grid
  storage repairs replica divergence and adjacent-layer prefill. A host
  regression verifies coverage for native worker configurations.
- [Precision investigation](AUTODEBUG_stack_precision.md): real adjacent-stack
  controls and captured router boundaries justify the sliding BFP8 expert
  gate/up and QKV policy. Full BFP4 WO fails batch-32 and is rejected. The
  selected full BFP4 QKV survives actual adjacent 4→5 tests.
- [Fused gather geometry](AUTODEBUG_sharded_regression.md): the optimized K44
  block exceeds the fused gather's 22-tile readiness slice. The K22-only control repairs
  all128 decode/cache comparisons, and all six final-policy fused-family
  comparisons pass but remain slower. The failed run is retained as diagnosis.

- [Watcher endpoint repair](AUTOFIX_watcher_ag.md): the one-worker all-gather
  writer requested a nonexistent outward connection at Linear endpoints.
  Guarding only unused endpoint lookups preserves required-route assertions.
  Exact BF16/BFP8 payload probes and the two-case durable regression pass
  Watcher with eight traced replays. Both layer kinds and both actual adjacent
  stacks also pass; see `final_watcher_summary.json`.

## Capacity, checks and reproduction

The context target remains 262,144 tokens. The updated
[context contract](../context_contract.json) and `final_memory_plan.json` count
1,754,480,640 extra resident DRAM weight bytes per device and a 2,690,688-byte
shared CCL payload pool. The conservative DRAM bound is 29,206,137,856 bytes
per device. Capacity tests prime the real shared pool and allocate all 30
private CCL semaphore sets before prefill, plus anonymous other-resident DRAM
reservations. Both kinds pass 262144 prefill and 262143 plus final-position
decode. These exercise capacity, not a complete model's allocation order.
Batch-32 short-context support does not imply 32 maximum-context requests.

From the repository root, using the existing environment:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/validate_defaults.py
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/profile_defaults.py
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/write_packet.py
```

Run hardware commands serially. The validation script separates Watcher from
profiling and records exact argv, environment, exit status and source hashes.
It covers batch 32, cache/page ownership, repeated traces, non-aligned lengths,
actual adjacent stacks, and maximum context. Runtime fallback audits are clean
in the completed batch and stack checks. Final Watcher checks and device profiles pass. Independent review returned `clean-pass`; local checkpoint SHAs are recorded
in the work log.
No push is authorized or performed.

The independent review also found a NaN-sensitive host acceptance predicate.
All output/cache/batch/stack PCC gates now explicitly require finite values.
`finite_gate_audit.json` rechecks every saved accepted vector against the actual
new predicate; the hardware measurements are unchanged. Exact source-equivalence
records retain the original hashes for this host-only edit and the fused-only
K-block and unused endpoint-pointer edits.

The compact local evidence packet is written to
`bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/1a0f19aa-a217-42b9-acac-1a58fe787838.json`.
Archive paths and original hashes are mapped in `preserved_evidence_manifest.json`;
raw native captures and fixture tensors remain local.

Local implementation/evidence commit: `dc254cbde8997f0ec8f8f42bccb5aaf1f0e8d69e`.
