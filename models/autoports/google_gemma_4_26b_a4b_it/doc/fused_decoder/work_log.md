# Fused decoder work log

## Scope and baseline

- Model: `google/gemma-4-26B-A4B-it`.
- Stage: 02 fused decoder, single device only.
- Functional checkpoint: `f3f2cfd7033` plus environment record
  `00cade41989`.
- Hardware health: `timeout 60 tt-smi -ls --local` listed four P300 chips.
- Context/layout/cache dtype are unchanged, so `doc/context_contract.json`
  remains valid without a capability reduction.

## Operation topology

| Region | Functional sequence | Movement | Final fused sequence |
| --- | --- | --- | --- |
| attention | RMSNorm → packed QKV linear → create heads → per-head RMSNorm → RoPE → cache fill/update → SDPA → concat heads → O linear | required DRAM/L1 sharding at decode kernel boundaries | unchanged; already dedicated ops |
| dense MLP | gate linear + up linear → GELU → multiply → down linear | DRAM interleaved | gate linear + up linear → multiply(input-A GELU) → down linear |
| router | RMSNorm → scale tensor → scalar multiply → FP32 linear → TopK → softmax → scatter | DRAM interleaved | RMSNorm → pre-folded scale multiply → FP32 linear → TopK → softmax → scatter |
| prefill MoE | tiled sparse gate + up → GELU → multiply → sparse down → weighted reduce | DRAM, canonical 32-token chunks | tiled sparse gate + up → multiply(input-A GELU) → sparse down → weighted reduce |
| decode MoE | sparse gate + sparse up → GELU → multiply → sparse down → weighted reduce | L1 sparse layouts | sparse gate + sparse up → multiply(input-A GELU) → sparse down → weighted reduce |

## Candidate ledger

- Dedicated fused GeGLU binary: accepted for dense MLP and decode MoE. Initial
  use of the generic `activations` argument failed real-weight PCC
  (sliding 0.712586, full 0.911095); operand-specific
  `input_tensor_a_activations` fixed it and restored the baseline PCC.
- Prefill MoE GeGLU: accepted after adapting the canonical prefill helper
  locally. A first copy omitted its internal 32-token split and failed seq33
  with `16 blocks > 8 cores`; preserving split/concat and applying a no-op
  TILE-layout normalization restored every non-aligned boundary while retaining
  the fused binary.
- Router constant folding: accepted. The two immutable scale factors are
  combined once at setup, removing a runtime broadcast multiply.
- Dedicated router family: assessed and rejected on the target P300.
  `generalized_moe_gate` is exact only for the middle global-top-8 plus
  selected-logit-softmax subgraph (`generalized_moe_gate.hpp:16-27` and its
  golden in `models/common/tests/modules/moe/test_generalized_moe_gate.py:24-43`).
  Its validator requires BF16/UINT16 L1-sharded inputs, a 256-element
  32x32-tiled block, and one token per core
  (`generalized_moe_gate_device_operation.cpp:67-95,129-145`); it emits only
  compact values and indices, not the dense 128-entry sparsity consumed by
  `sparse_matmul`. Gemma would therefore retain 128-to-256 negative-sentinel
  padding, shard/layout adapters, caller chunking for long/non-aligned
  prefill, dense scatter, and its nonuniform per-expert post-scale. More
  decisively, the focused current-hardware probe below failed before execution:
  the Blackhole JIT cannot find
  `experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h`.
  This matches the op test's explicit Blackhole skip at
  `test_generalized_moe_gate.py:70`; no Stage-02-local adapter can repair a
  missing architecture LLK implementation.
- `deepseek_moe_gate`: structural rejection. Its API has no global/softmax
  mode and the kernel implements grouped 8x32 selection with linear
  renormalization, while Gemma requires global top-8 and selected-logit
  softmax (`deepseek_moe_gate.hpp:13-23`;
  `deepseek_moe_gate_device_operation.cpp:57-70,94-103`). It also emits
  compact values/indices rather than dense post-scaled routing.
- `moe_gate_mm`: structural and architecture rejection. Its program factory
  requires exactly 12 DRAM-aligned cores and calls the ring algorithm
  Wormhole-only (`moe_gate_mm_program_factory.cpp:24-37`); the checked test
  shape is M=32, K=7168, N=256, not Gemma's hidden 2816 and 128 experts, and
  the test is disabled on Blackhole. Its packed collector output cannot feed
  Gemma's dense 128-entry `sparse_matmul` sparsity contract.
- `ttnn.experimental.moe_compute` / `TTMoEDecode`: structural rejection for
  this stage. The primitive accepts selected UINT16 indices and K scores,
  rank-6 specially packed expert weights, and a row-major BF16 activation;
  the delivered router instead exposes dense 128-entry post-scaled routing.
  A correct adapter would have to retain indices, gather the nonuniform
  per-expert scales, repack the delivered BF16 weights (the common module
  documents BF4_B packing in `models/common/modules/moe/tt_moe_decode.py:58-68`),
  and qualify the resulting precision. The direct single-card route is
  demonstrated only through `compute_only=True`; `TTMoEDecode` itself
  orchestrates all-to-all dispatch and full combine
  (`tt_moe_decode.py:840-923`) and is decode-only. The primitive caps tokens
  per invocation and therefore needs a new chunk/pad/unpad plus local-combine
  implementation for valid prefill lengths 1/31/33/.../1025. That is a new
  expert execution subsystem, not an available fused substitution preserving
  this decoder's BF16 PCC and prefill/decode contracts.
- QKV/create-heads, SDPA, cache, concat-heads, RoPE, RMSNorm, TopK, sparse MoE:
  already dedicated fused ops; no primitive spelling remains.
- Residual-add + RMSNorm: rejected by dataflow contract. The materialized sum
  is independently consumed by router, MoE normalization, and later residual.
- Dense gate/up shared-LHS packing: assessed; two matmuls do not meet the
  skill's >=3 peer threshold, and output slicing adds dispatch/movement.
- QKV peer packing: already present in the functional weights and one linear.
- Permute/reshape identities: none; remaining reshapes/transposes change head,
  expert, or decode layouts required by consumers.
- Matmul bias/activation/transpose/slice merging: no bias or adjacent eligible
  operation. GELU belongs on the consuming multiply and is fused there.
- Conv/BN/pad/pool/spatial-mean/scaled-sum/numeric-softmax patterns: absent.
- Distributed RMSNorm/fused CCL-matmul: multichip-only and outside stage 02.
- LM-head/top-k sampling: full-model-only and outside stage 02.

## Evidence

- Correctness: `pcc_layer*.json`, `prefill_boundaries_*.json`,
  `prefill_batch2_*.json`, `trace_*.json`.
- Perf: functional and fused `layer*_host_timings.json`; fused files include
  every sample and the median.
- Profiler: `tracy/sliding_b1/`; modern join failed on dropped markers, legacy
  parsing produced `reports/ops_perf_results.csv` and
  `prefill_sliding_batch1.{txt,csv}`.
- Watcher: `watcher.log`, 9 passed, no sanitizer error match.

### Router-family AutoFix experiment

- Hypothesis: the checkout's generalized dedicated gate can replace the
  explicit `topk` plus selected-score `softmax` on the target Blackhole.
- Experiment:
  `timeout 180 python -c 'import ttnn; from
  models.common.tests.modules.moe.test_generalized_moe_gate import
  test_generalized_moe_gate; d=ttnn.open_device(device_id=3);
  test_generalized_moe_gate(d,32,False,42,8,True,1.0);
  ttnn.close_device(d)'`, after `timeout 60 tt-smi -ls --local` showed all
  four P300 devices.
- Result: failed in 4.16 seconds during Blackhole TRISC1 compilation with
  `fatal error:
  experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h:
  No such file or directory`; the device closed normally.
- Verdict: refuted for the current hardware/checkout. No runtime candidate was
  retained and no performance run is meaningful because the op cannot build.
- Fix: none. The final fused runtime remains the already-correct and faster
  TopK/softmax/scatter route. The exact API/semantic rejections for the other
  dedicated gate and MoE-compute families are recorded above.

## Final status

Implementation and evidence are ready for independent stage review. Commit SHA
is recorded after a clean-pass review.

## Stage-review remediation

- All P1/P2 findings in `stage_review.md` were treated as work.
- Final host fused tests: 7 passed, including prefill dispatch, provenance, and all
  overridden graph source audits.
- Final hardware correctness: 11 passed for PCC/cache views, batch-2,
  eager/replay, b1/b32 determinism, and bounded tail cache; both complete
  non-aligned boundary cases passed; both advertised-context decode cases
  passed.
- Separate final watcher run: 9 passed, with no sanitizer fault.
- Controlled final performance: 7 prefill and 101 trace-replay samples; fused
  wins both prefill rows and all four decode rows. Exact numbers are in README.
- Profiler: both prefill layer kinds have valid Blackhole/110-worker
  `tt-perf-report` CSVs. All decode layer/batch cases have nonzero Blackhole
  device-trace timings. The modern join's missing-op assertion is documented
  as a tool limitation in `profiler_summary.md`.
- Immutable final evidence identity and artifact hashes are in
  `final_manifest.md` and `final_manifest.sha256`.

Stage-owned files include:

- `tt/fused_decoder.py`;
- `tests/test_fused_decoder.py` and the shared
  `tests/test_functional_decoder.py` evidence harness;
- all files under `doc/fused_decoder/`;
- the four functional baseline `layer*_host_timings.json` files under
  `doc/functional_decoder/`.

Unrelated `.agents`, `.skillexp-STAGE-RUNNING`, and GPT-OSS paths are excluded.

## AutoFix report: performance/evidence gap

### Starting evidence

- Fresh source-only diagnosis: `AUTODEBUG.md`. The repo AutoDebug launcher was
  attempted first but could not run because the `codex` executable is absent;
  a fresh inspection-only fork wrote the equivalent report before edits.
- Original failure: full-attention b1 fused traced decode was 3.21499 ms versus
  functional 3.21154 ms (11 unpaired samples), and its prefill result was
  680.696 ms versus 680.615 ms (3 unpaired samples). Both sample ranges
  overlapped.
- The empty decode report was traced to signposts around trace replay: TTNN op
  rows are emitted during capture, before the decode replay signpost.

### Hypothesis experiments

- Hypothesis: forcing the fused decode-MoE GeGLU multiply output to L1 removes
  hidden movement.
  Experiment: explicit `memory_config=ttnn.L1_MEMORY_CONFIG`, then 31 full-b1
  replays; compare with the immediately following default-placement run.
  Result: 3.21054 ms versus 3.21177 ms (0.038%), below the observed run
  variation; prefill, which cannot be affected by this change, moved in the
  opposite direction.
  Verdict: refuted; the edit was removed.
- Hypothesis: the immutable `router.scale` and `HIDDEN_SIZE**-0.5` factors can
  be folded once during setup, removing one router broadcast multiply per
  invocation.
  Experiment: compare 201 replay medians on the same device/regime.
  Result: folded 3.20765 ms versus original fused 3.21792 ms. A repeated final
  folded run was 3.21365 ms; a current-checkout functional control was
  3.21538 ms. Final folded prefill was 680.599 ms versus the functional control
  680.699 ms.
  Verdict: verified. Real-weight full-attention natural/shared-cache PCC passed:
  prefill 0.998483 and decode 0.999865.
  Fix: `FusedDecoder.__init__` materializes `fused_router_scale`; the overridden
  router path consumes it in one multiply.
- Hypothesis: decode `tt-perf-report` is empty because replay has no TTNN op
  rows.
  Experiment: extract the capture rows immediately after the prefill end
  signpost and feed those rows to `tt-perf-report`.
  Result: `tracy/decode_capture_sliding_batch1.csv` contains 73 decode-capture
  op rows. The raw capture slice contains fused GELU attributes; replay-only
  signpost filtering remains correctly empty.
  Verdict: verified; capture-graph and replay-latency evidence are now labeled
  separately.

### Final status

- Fixed: the final path contains both GeGLU fusion and router constant folding,
  passes real-weight PCC, and beats the current-checkout functional full-b1
  control in both 201-replay traced decode and 7-repeat prefill.
- Commands:
  `GEMMA4_DECODER_IMPL=fused GEMMA4_RANGE_DOWNLOAD=1
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q
  .../test_functional_decoder.py -k 'real_weights_prefill_decode and
  full_attention and not shared_physical'`;
  `GEMMA4_PERF_PREFILL_REPEATS=7 GEMMA4_PERF_DECODE_REPEATS=201 ... pytest -q
  ...::test_functional_decoder_perf_profile -k 'full_attention_1024 and
  batch1'`.
- Remaining uncertainty: process-to-process host timing jitter is several
  microseconds. The 201-replay control is materially stronger than the original
  11-sample evidence, but it is still sequential rather than an in-process
  paired benchmark.

## Stage checkpoint

- Independent final rereview: `stage_clean_rereview.md`, verdict `clean-pass`.
- Stage-owned implementation, tests, documentation, and retained evidence
  checkpoint: `18e7ad76781`.
- The checkpoint is local only; it was not pushed.
