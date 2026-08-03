# AutoFix Report

## Starting Evidence

- Source: `doc/optimized_decoder/AUTODEBUG.md`, hypothesis H1.
- Symptom: default BFP8 down projection `32x8192x3072` measured about 183 us,
  `SLOW`, with about 27.7% reported DRAM utilization.
- Initial source state: BFP8 storage dtype reached runtime, but
  `FusedDecoder._mlp` used default interleaved `ttnn.linear` without an
  activation shard, DRAM-sharded weight, program config, or compute config.

## Hypothesis Experiment

- Hypothesis: the default interleaved down path is bandwidth-starved; a
  coherent width-sharded L1 activation plus DRAM-sharded BFP8 weight and
  explicit program config will materially reduce latency.
- Experiment: `tests/optimized_down_h1_experiment.py` measures the isolated
  real-weight down projection at logical B1 and B32. It crosses 16/32 cores,
  every selected legal `in0_block_w`, and BFP8 HiFi2/LoFi. All paths produce
  BF16 output and compare to a Torch matmul.
- Command:

  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_down_h1_experiment.py`

- Evidence: `doc/optimized_decoder/h1_down_experiment.log`.
- Baseline:
  - B1: 0.200110 ms, PCC 0.99986798
  - B32: 0.199745 ms, PCC 0.99986541
- Selected candidate: 16 cores, `in0_block_w=8`, LoFi:
  - B1: 0.070320 ms, PCC 0.99982256
  - B32: 0.070274 ms, PCC 0.99983132
- HiFi2 control at the same geometry:
  - B1: 0.105027 ms, PCC 0.99988532
  - B32: 0.105123 ms, PCC 0.99989098
- Result: the selected candidate is about 64.9% faster than the isolated
  baseline and passes the 0.995 real-weight PCC requirement.
- Verdict: verified.

## Fix

- `tt/optimized_decoder.py` now materializes a separate BFP8 DRAM-width-sharded
  decode down weight.
- Decode MLP reshards the activated tensor over 16 L1 width shards, executes
  the selected `in0_block_w=8` LoFi DRAM-sharded matmul, restores DRAM
  interleaving, and performs the inherited residual add.
- Prefill retains the previous interleaved down path.

## Verification

Real-reference correctness command:

`pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py -k "prefill_real_reference_non_aligned or decode_real_reference_and_determinism"`

Evidence: `doc/optimized_decoder/h1_default_correctness.log`.

- 5 passed.
- Decode B1 PCC: 0.9991490165
- Decode B32 PCC: 0.9992328431
- Non-aligned prefill remains unchanged:
  - S31: 0.9992170142
  - S33: 0.9992864108
  - S65: 0.9992565117
- Repeated decode determinism assertions passed.

Whole-decoder performance command:

`pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py`

Evidence: `doc/optimized_decoder/h1_default_perf.log`.

- Traced B1: fused 1.047423 ms, optimized 0.667841 ms
- Traced B32: fused 1.211761 ms, optimized 0.830889 ms
- Relative to the prior optimized default (0.791344/0.939042 ms), the fix
  improves B1 by about 15.6% and B32 by about 11.5%.
- Warmed prefill also passed and retained its dtype-only improvement because
  prefill does not dispatch through the new down config.

## H1 Status

- H1 fixed and verified.
- Hardware was healthy (four Blackhole p300c devices visible); commands ran
  serially and devices closed after each run.

## H2/H4 Composite Experiment

- Hypothesis: width-sharded decode RMSNorm output can remain sharded through
  the consuming QKV and gate/up projections, replacing the two 44-us
  single-core norm rows and reducing whole-layer latency.
- Experiment: a temporary candidate policy converted each decode norm input
  to a 32-core `8x4` width shard (`3072/32=96` elements, three tiles per
  shard), ran `LayerNormShardedMultiCoreProgramConfig(block_w=3,
  subblock_w=3)`, and passed the sharded result directly to the existing
  attention/MLP projections. Both H1 and H2/H4 decoders were fully constructed
  before traces were captured. Real layer-0 weights and the Torch layer oracle
  were used.
- Command:

  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_norm_chain_experiment.py`

- Evidence: `doc/optimized_decoder/h24_norm_chain_experiment.log`.
- Results:
  - B1 PCC: H1 0.9990386963, sharded norm 0.9990386963
  - B1 traced: H1 0.667442 ms, sharded norm 0.667264 ms
  - B32 PCC: H1 0.9992697239, sharded norm 0.9992697239
  - B32 traced: H1 0.873325 ms, sharded norm 0.874050 ms
- Verdict: refuted as a default win. B1 differs by only 0.000178 ms and B32
  regresses by 0.000725 ms. The default projection path absorbs or offsets the
  isolated norm benefit; this is not a measurable whole-layer improvement.
- Fix: none. The candidate policy, norm override, and temporary test were
  removed. The log is retained as the refutation artifact.

## Trace Allocation Warning

- Cause: the benchmark constructed and compiled the optimized candidate after
  capturing the fused candidate trace. Device tensor/program allocation while
  a trace remained active produced an explicit corruption warning.
- Fix: `tests/optimized_decoder_perf.py` now constructs, warms, and stores all
  candidate closures first, then captures their traces in a separate loop.
- Verification command:

  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py -k candidate_traced_decode`

- Evidence: `doc/optimized_decoder/trace_ordering_verification.log`.
- Results:
  - 2 passed; no active-trace allocation/corruption warning.
  - B1: fused 1.047274 ms, optimized 0.667609 ms
  - B32: fused 1.210779 ms, optimized 0.830255 ms

## Final Status

- H1 remains fixed and verified.
- H2/H4 is refuted for the tested coherent norm-to-projection contract; no
  speculative production code was retained.
- Trace resource ordering is fixed and verified without the warning.
- Remaining uncertainty: a deeper topology rewrite that also changes QKV,
  output projection, residual adds, and gate/up program configs could make a
  persistent residual shard profitable, but that is a distinct composite
  hypothesis rather than evidence for retaining this candidate.

## Remaining Projection Matrix

- Hypothesis: packed QKV (BFP8), output (BFP8), and packed gate/up (BFP4)
  benefit from the same explicit DRAM-sharded/LoFi family as the verified down
  projection.
- Focused command:

  `pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_projection_matrix_experiment.py`

- Evidence: `doc/optimized_decoder/projection_matrix.log`.
- Matrix: exact real weights at logical B1/B32, padded decode M=32, 16- and
  32-core activation/output grids, HiFi2 and LoFi, and all selected legal K
  blocks. For K=3072 (96 tiles), the 16-core sweep included non-power-of-two
  `in0_block_w=6` and `3`, plus `2` and `1`; the 32-core sweep included `3`
  and `1`.
- Best isolated rows (B1/B32):
  - QKV baseline 0.103650/0.103342 ms; 16-core block-3 LoFi
    0.078474/0.078629 ms, PCC 0.99983239/0.99985766.
  - Output baseline 0.086405/0.086814 ms; 16-core block-6 LoFi
    0.036812/0.036859 ms, PCC 0.99985945/0.99984908.
  - Gate/up baseline 0.119416/0.119979 ms; 16-core block-6 LoFi
    0.106462/0.106378 ms. Its isolated BFP4-to-BF16-oracle PCC
    0.99281788/0.99304563 is essentially the same as the BFP4 baseline
    0.99280089/0.99301869, so whole-layer PCC—not this synthetic boundary
    threshold—was the acceptance gate.
- The cumulative all-role candidate passed real whole-layer PCC:
  B1 0.9991608353 and B32 0.9992698047
  (`projection_cumulative_correctness.log`). A separate run appeared to improve
  traced latency to 0.642772/0.806150 ms
  (`projection_cumulative_perf.log`), but that apparent gain did not reproduce
  in the required same-process cumulative control.
- Same-process cumulative ablation evidence:
  `doc/optimized_decoder/projection_ablation.log`.
  - B1: H1 0.642650, +QKV 0.642832, +output 0.642731, all 0.642582 ms.
  - B32: H1 0.805800, +QKV 0.805604, +output 0.805346, all 0.805576 ms.
- Verdict: refuted as material cumulative winners. Every difference is below
  0.0005 ms; QKV regresses B1, output does not recover the H1 control at B1,
  and gate/up regresses B32 relative to QKV+output. The cross-run apparent
  gain is not accepted over the same-run A/B.
- Fix: none retained. The temporary QKV/output/gate DRAM weights, decode
  override, policy switches, and ablation test were removed. The reusable
  isolated matrix and all logs remain as evidence. The verified H1 down
  projection is unchanged.
