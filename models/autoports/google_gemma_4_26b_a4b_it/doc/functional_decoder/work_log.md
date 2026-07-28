# Functional decoder work log

Date: 2026-07-28 UTC

Base checkout: `b9e6c242a34011e3daeebab9207fbb5b79750f39`

Scope remained under
`models/autoports/google_gemma_4_26b_a4b_it/{tt,tests,doc}`. No optimized
decoder, multichip decoder, full model, or vLLM work was started.

## Implementation record

1. Implemented both target decoder kinds from the repo Gemma-4 config: layer 0
   sliding attention and layer 5 full attention.
2. Used setup-only Torch/Hugging Face weight transformation. A single runtime
   prefill/decode pass is TTNN-only.
3. Implemented paged prefill/decode, current-position tensors, page tables,
   shared physical cache views, tile padding/slicing, and exact bounded-modulo
   tail updates.
4. Replaced dense all-expert MoE execution with canonical top-8 sparse TTNN
   execution. Decode batch 32 uses 32 trace-safe single-user sparse calls
   because the sparse kernel accepts one independent 128-expert mask.
5. Added a multi-user prefill wrapper and real-shape batch-2 coverage.
6. Routed prefill above the 32768-token non-chunked SDPA limit through bounded
   sliding/full chunked attention. Whole-layer full-context runs use bounded
   1024-token MoE chunks to avoid an avoidable 11.8 GB intermediate request.
7. Removed the QKV DRAM-to-L1 workaround after exact-shape op-level DRAM/L1
   A/B tests showed PCC 1.0. Post-removal real-weight and traced batch-32
   regressions pass with zero promotion hits; no pre-removal whole-decoder
   contrast is claimed.

## Precision and program-config A/B

The runtime was narrowed from a blanket HiFi4/fp32 policy.

- Framework defaults except sparse gate produced decode PCC 0.993275 sliding
  and 0.990723 full.
- High-accumulation attention head norms plus sparse gate produced 0.994512
  sliding and 0.999856 full.
- Adding decode SDPA was refuted at 0.994412 sliding.
- High accumulation for all RMS norms plus sparse gate produced final PCC
  0.999655 sliding and 0.999861 full.
- Full chunked prefill SDPA at framework defaults produced PCC 0.975739 at
  position 32768; its explicit correctness config raises selected-row minimum
  PCC to 0.998736.

Decode SDPA `program_config=None` silently failed batch 32 at PCC 0.630007
sliding and 0.958089 full. The canonical full-grid Q32/K64 sliding control was
also 0.629103. A workload-derived 32-core cap (up to 8x4), Q32, K64 passed both
layer kinds at PCC 0.999283 and 0.999861. These are correctness requirements,
not optimization-stage tuning.

Dense matmuls use no program config. `ttnn.sparse_matmul` requires a
`MatmulMultiCoreReuseMultiCast1DProgramConfig` in its API; gate/up/down use the
canonical Gemma-4 shape-derived builder rather than a performance sweep.

Artifacts: `precision_exception_ab.json`, `precision_policy_isolation.json`,
and `sdpa_program_config_ab.json`.

## Final correctness and capability evidence

- Real-weight whole-layer PCC:
  - layer 0 prefill/decode: 0.998617/0.999655;
  - layer 5 prefill/decode: 0.997773/0.999861 for natural and shared cache
    views.
- Real-shape batch-2 prefill PCC: 0.997502 sliding, 0.998994 full.
- Traced decode HF PCC:
  - sliding batch 1/32: 0.999418/0.999283;
  - full batch 1/32: 0.999861/0.999861.
- Eager/replay and repeated replay PCC: 1.0 for every traced batch contract.
- Mutable A/B/A batch-32 buffers:
  - eager-control/replay PCC 1.0 for A, B, A;
  - repeated A PCC 1.0;
  - nonzero A/B maximum difference for both kinds.
- Logical boundary minimum PCC: 0.996049 sliding, 0.998098 full.
- Long-attention selected-row minimum PCC: 0.996791 sliding, 0.998736 full.
- Bounded modulo length 1025 preserves every live cache row except the exact
  wrapped tail destination.
- Real-weight prefill passed at 262144 and 262143 for both layer kinds. Final
  artifact host elapsed:
  - sliding: 175.017 s / 175.017 s;
  - full: 262.357 s / 262.378 s.
- Traced batch-1 decode passed at current position 262143 for both kinds with
  repeat PCC 1.0. Nonzero baselines and distinct logical history sentinels
  were device-read back through rolled page tables and preserved after replay.
  Context artifacts bind decoder/test/binary hashes and exact commands. No
  context capability reduction is recorded.

## Final performance

Sequence/current position 1024, one Blackhole P300:

- warmed prefill synchronized host:
  - sliding: 681.667 ms;
  - full: 682.521 ms;
- measured traced decode device/host:
  - sliding batch 1: 2.991/3.038 ms;
  - sliding batch 32: 68.860/68.969 ms;
  - full batch 1: 3.177/3.204 ms;
  - full batch 32: 68.628/68.723 ms.

The warmed host prefill window is authoritative. Device marker buffers filled,
so the supported legacy parser's human-readable per-op tables are diagnostic
and are not reported as a full device window. Dedicated device trace profiling
captured both decode replays without dropped markers. `tracy/provenance.json`
records commands, hashes, hardware, parser limitations, and every retained
artifact hash.

## Runtime and device gates

- `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'` was used for
  correctness and performance.
- Static source audit excludes Torch and TTNN host conversion from measured
  hot paths.
- No-download default suite: 23 passed, including both real-shape synthetic
  HF-vs-TTNN numerical cases; 28 real-weight/opt-in cases skipped as designed.
- Complete directory with the repo-local real-layer cache: 40 passed, 11
  explicit opt-in cases skipped in 57.66 s.
- Final real-weight/batch-2/traced contract bundle: 9 passed.
- Boundary/context/mutable/long bundle: all repaired cases passed; the full
  long-attention default-precision failure was fixed and rerun.
- Final watcher command used `TT_METAL_WATCHER=10`: 9 passed, 0 failed, 0
  errors, 0 skipped in 49.701 s. The 2173-line watcher log has no
  error/fatal/assert/hang match.
- Final performance profiler harness: all four batch/layer cases passed.

## Review and commits

The initial stage review returned `more-work-needed`. Its findings drove:
long-attention chunking, exact modulo tails, mutable trace-buffer tests,
precision isolation, removal of the QKV L1 workaround, real statistical
synthetic shapes, batch-2 prefill, watcher interval 10, context refresh, and
performance refresh.

The fresh xhigh stage rereview returned `clean-pass`: no required work and no
hard-check gaps. It verified the current-source context sentinels/provenance,
current-source batch-1/32 performance, normal-CI synthetic numerical coverage,
and the narrowed QKV evidence claim. The local checkpoint SHA is recorded
below after commit. No push is performed.
