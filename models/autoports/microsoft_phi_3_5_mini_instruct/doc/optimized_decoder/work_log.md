# Work log

Date: 2026-07-30 UTC

- Started from `8b71c7819dc`.
- Read the complete `optimize`, `tt-device-usage`, and `shard-advise` skill
  instructions before device or implementation work.
- Preserved unrelated untracked GPT-OSS artifacts already present in the
  checkout.
- Hardware health: `timeout 60 tt-smi -ls --local` listed four Blackhole p300c
  boards. No live pytest, serving, or device-profiler job owned the device.
  The persistent Tracy web viewer and two defunct capture processes do not own
  TT devices.
- Fresh functional baseline command:
  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/functional_decoder_perf.py`
- Fresh baseline result: 3 passed in 5.35 s. Prefill B1/S128 was 2.007797 ms;
  traced decode B1/context128 was 1.052189 ms mean and 1.050918 ms min; traced
  decode B32/context128 was 1.215247 ms mean and 1.211395 ms min.
- Completed the initial operation-topology audit in `README.md`. It records
  current ops, repeated-input packing, layout movement, composite-op
  candidates, precision/fidelity constraints, actions, and baseline evidence.
- Advisor installation preflight: `TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir`;
  checkout commit is the required `618cd4e75d`.
- Added the initial `OptimizedDecoder` candidate with BFP8 attention/output
  weights, BFP4 packed gate/up, BFP8 down, packed-gate/up LoFi, and down HiFi2.
  The optimized-only test constructs exactly `OptimizedDecoder` and checks its
  owned MLP runtime method.
- First real-weight run exposed an API namespace mismatch:
  `ttnn.BlackholeComputeKernelConfig` does not exist in this checkout. This was
  repaired to `ttnn.types.BlackholeComputeKernelConfig` and rerun rather than
  treating the first error as a rejected optimization.
- Real-weight initial-candidate command:
  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py`
- Initial-candidate result: 2 passed in 3.34 s; prefill-33 PCC 0.99927393 and
  decode-33 PCC 0.99931367, both above the functional 0.995 bar. This proves
  the reduced MLP/attention weight policy is viable on target weights, but it
  is not yet final because timing, geometry, trace, cache, and runtime-row
  verification remain.

## Cumulative candidate contract

| Contract | Strongest correct candidate |
| --- | --- |
| Projection topology | packed QKV; packed gate/up |
| Attention / MLP weights | BF16, HiFi2 baseline |
| Activations / norms | BF16 |
| KV cache | BF16, DRAM paged, page size 32 |
| Logical decode batch | 1 primary and 32 serving; physical tile rows are not active users |
| Attention | paged SDPA decode; explicit Phi rotate-half for head dim 96 |
| Residual / norms | DRAM interleaved baseline; one-core norms |
| Context | prefill/decode through 131072; non-aligned prefill preserved |
| Runtime | TTNN-only forward, trace safe |

## Shard advisor hard gate

- Used `TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir` at pinned commit
  `618cd4e75d`, sourced `scripts/bootstrap.sh` in the capture shell, and ran
  `ttnn-advise capture` on the rewritten packed dense block at B1 and B32.
- Required artifacts are `shard_advise/report.json` and `final_ir.mlir`;
  supplemental B32 artifacts are `report_b32.json` and `final_ir_b32.mlir`.
  The report contains 40 ops/37 choices at B1 and 39/36 at B32, and records
  `dram_sharded_considered=4`, `advised=4`.
- Applied all four DRAM-sharded seeds: QKV `(in0_block_w=6, per_core_M=1,
  per_core_N=3)`, output `(12,1,1)`, gate/up `(6,1,5)`, down `(16,1,1)`.
  Applied the residual-chain intent as an 8-core width-sharded RMSNorm whose
  output is exactly the next DRAM-sharded matmul input layout. This adapted
  the advisor's irregular block-sharded norm suggestion to dimensions that
  divide 3072 cleanly and removed the intervening DRAM restore.
- Rejected the advisor's unrelated per-element irregular shard changes around
  Phi rotate-half because the 48-wide half-slices cannot consume those tile
  shards; the retained rectangular one-core-per-user cache layout is required
  by `nlp_concat_heads_decode`. The whole adapted norm/matmul chain, rather
  than an isolated sharded norm, was measured and won.

## Geometry, precision, and topology evidence

The advisor geometry was crossed with the final BFP4/LoFi policy at both
batches. Larger `in0_block_w` values were retried after the first API errors:

| Role/candidate | B1 result | B32 result | Decision |
| --- | ---: | ---: | --- |
| Advisor seed, final BFP4/LoFi | 0.491 ms | 0.635 ms | viable seed |
| QKV `in0_block_w=12` | 0.489 ms | 0.633 ms | **apply** |
| Gate/up `in0_block_w=8` | width 12 tiles not divisible by block 8 | same | reject, exact shape constraint |
| Gate/up six-core input, block 8 | 0.490 ms | 0.635 ms | reject: no material B1 win, B32 regresses |
| Gate/up six-core input, block 16 | L1 need 2,077,440 > 1,572,864 | same | reject, exact hard limit |
| Gate/up `in0_block_w=12` | L1 need 1,618,688 > 1,572,864 | same | reject, exact hard limit |
| Down `in0_block_w=32` | 0.490 ms | 0.634 ms | **apply** |
| Output `per_core_N=2` | 0.491 ms | 0.635 ms | reject |
| Output `per_core_N=3` | 0.493 ms | 0.635 ms | reject |
| Gate/up `per_core_N=8` | 0.492 ms | 0.634 ms | reject |
| Down `per_core_N=2/3` | 0.492 ms | 0.635/0.634 ms | reject |

Precision/fidelity sweeps were independently timed at B1/B32: attention
BFP8/LoFi `0.622/0.778`, down BFP8/LoFi `0.636/0.797`, attention BFP4/LoFi
`0.606/0.779`, down BFP4/LoFi `0.626/0.784`, and cumulative all-projection
BFP4/LoFi `0.564/0.723` before sharded norms. The cumulative real-weight
result is prefill PCC `0.99927393`, decode PCC `0.99895715`, so it wins.

Post-review, the exact final BFP4 geometry was rerun and preserved in
`sweeps/geometry_bfp4_lofi.log`; this exposed that QKV block 12 and down block
32 become legal after BFP4 reduces L1 pressure, and both were promoted.
Gate/up block 12 remains an exact L1 failure at both batches. The DRAM-sharded
program type exposes no output-subblock or core-grid arguments; the API
signature/error is preserved in `sweeps/dram_sharded_output_subblock_repro.txt`.
Thus the generic profiler `SLOW / No output subblock size found` warning has no
actionable subblock control in the selected family, while every control that
the family does expose was swept.

The alternate-shard follow-up in `sweeps/gate_adapted_shard.log` adapts the
complete post-norm gate/up boundary from eight to six input cores, giving 16
input tiles per shard. Block 8 is then legal but measures `0.489746/0.634601`
ms versus the selected full path's `0.489858/0.631896`: the batch-1 difference
is noise-sized and batch 32 regresses, so it is rejected. Block 16 reaches an
exact static-CB L1 limit of 2,077,440 bytes versus 1,572,864 at both batches.
The runner classifies this expected negative result and exits cleanly.

Same-dtype fidelity evidence is in `sweeps/policy_hifi2.log`. Against final
all-BFP4/LoFi `0.491/0.634` ms, setting only QKV HiFi2 gives `0.534/0.676`,
output `0.505/0.649`, gate/up `0.568/0.711`, and down `0.530/0.672`; all-HiFi2
is `0.666/0.807`. LoFi wins every projection role at both batches.

The 8-core sharded norm/residual chain improved cumulative BFP4/LoFi decode
from `0.566/0.723` to `0.490/0.652` ms. The final same-run topology sweep
measured BFP8 cache at `0.489/0.633` versus BF16 cache at `0.489/0.649`;
consuming-cache real-weight PCC was `0.99894647`, so BFP8 is final. Exact raw
results are in `sweeps/topology_cache_split.log`. The first BFP8 fill/update errors were
adapted according to each API contract: fill inputs are explicitly cast while
decode update retains BF16 and lets the update kernel repack.

Packed versus fully adapted split projections used BFP4/LoFi DRAM-sharded
weights, independent Q/K/V and gate/up program configs, legal rank-2 weight
slices, and included output concatenation/layout costs. Split measured
`0.567/0.713` ms versus packed `0.489/0.633`; packed QKV and gate/up remain.
The first rank mismatch was repaired and rerun, not treated as rejection.

Prefill remains DRAM interleaved and uses TTNN's selected large 2D programs;
the profiler proves 96/103-core matmuls with output subblocks `(2,3)`,
`(4,1)`, `(1,5)`, `(4,1)`. Explicit HiFi2/LoFi compute policies reduced
warmed S128 prefill from 2.007797 to 1.682877 ms. SDPA is the optimized TTNN
composite in both phases. Native rotary remains invalid for head dimension 96
(its width contract is divisible by 64), so explicit on-device rotate-half is
retained; no host fallback is introduced.

## Final correctness, performance, and profiling

- Optimized-owned suite: non-aligned S31/S33/S65, batch-2 paged routing,
  real-weight cache-consuming prefill/decode, B1/B32 deterministic trace,
  LongRoPE trace at position 4096, and advertised-context decode at 131072 all
  pass. Advertised-context real-weight PCC is `0.99883851`; LongRoPE trace PCC
  is `0.99999190`.
- Final unprofiled performance after the post-review geometry promotion:
  prefill B1/S128 `1.654356 ms`; traced decode B1 `0.489858 ms` and B32
  `0.631896 ms`. The same-session preserved functional baseline in
  `sweeps/baseline_functional.log` is `1.791829`, `1.051599`, and `1.216610`
  ms, so final improvements are 7.7%, 53.4%, and 48.1% respectively.
- `tracy_final/ops.csv` and signpost-derived reports prove all four decode
  matmuls are BFP4/LoFi with DRAM-sharded weights, norms are 8-core L1
  width-sharded, and decode SDPA consumes BFP8 cache. Representative device
  rows are QKV 55.5 us, output 20.5 us, gate/up 93.3 us, down 48.2 us; B1
  decode modeled DRAM roofline is 24.1% (123 GB/s). Remaining time is the
  required cache/head/explicit-RoPE topology plus trace dispatch.
- Separate watcher command used `TT_METAL_WATCHER=10` with no skipped asserts.
  Four real-weight/trace gates passed and
  `watcher_final/generated/watcher/watcher.log` contains no error/assert/hang.
- No runtime method contains `torch`, `from_torch`, `to_torch`, `.cpu`, or a
  call back through `FunctionalDecoder`; setup-only state loading is outside
  measured forwards.
- `sweeps/correctness_final.log` preserves the complete 13-test final pass.
  `sweeps/stress_final.log` preserves 50 trace replays each at B1 and B32;
  sampled replays 1/25/50 are finite and bitwise stable.

## Optimize checklist

- [x] Functional semantics/PCC, paged cache, trace, determinism, stress, B1/B32.
- [x] Optimized-owned traced decoder with no host fallback.
- [x] Decode norms and dominant projection boundaries use L1 width shards;
  unavoidable RoPE/cache composite boundaries are documented.
- [x] Prefill remains DRAM interleaved with large multicore 2D programs.
- [x] Operation-topology audit completed.
- [x] Shard advisor run this pass; required artifacts and apply/reject evidence.
- [x] Best candidate and final default reproduced against strongest baseline.
- [x] Runtime profiler proves dtype/fidelity/layout policy.
- [x] SDPA composites and packed repeated-input projections retained with an
  adapted legal split comparison.
- [x] Important memory/program/compute configs are explicit; geometry,
  precision, fidelity, output blocks, cores, and cache policy swept at B1/B32.
- [x] DRAM-sharded decode matmuls used for every dominant projection.
- [x] Multi-device/CCL, MoE, LM-head, sampling, and serving-only checklist
  entries are not applicable to this single-device decoder-layer stage.
- [x] Device-time rows, roofline, and warmed traced host latency reconciled.

## Independent stage review

The final `$stage-review` verdict is `clean-pass`. Its last requested
alternate-shard gate/up search is preserved in
`sweeps/gate_adapted_shard.log`; there are no unresolved correctness,
capability, performance-evidence, fallback, or stage-contract findings.

The first independent review returned more-work-needed for raw sweep evidence,
same-dtype fidelity, output-subblock investigation, and stress evidence. Those
findings produced the preserved sweep logs, a faster final geometry, the API
minimal repro, and the 50-replay tests above. Pending rereview and local commit.
