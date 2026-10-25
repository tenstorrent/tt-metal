# Full-path topology and selected optimizations

Target: TP4,1x4 ring, four Blackhole p300c ASICs, two links. Batch1 regression;
agent target32 conversations x32768 tokens; supporting batches8/16. Decoder policy
and rejected alternatives remain in `../optimized_multichip_decoder/` and stage06
`rejection_ledger.json`. No new precision frontier is being selected.

|Boundary|Current sequence/layout|Candidate/action|Constraints/evidence|
|---|---|---|---|
|Embedding|BF16 hidden-sharded table; one gather at B1 entry, sharded B8+|Retain existing ownership|Input token IDs stay on device;1280 channels/device matches stacked layout|
|Norms/residuals|Stage05 BF16 norms/residual exceptions; B1 replicated L1, B8+ native fractured stream|Retain cumulative selected contract|Stage05 family/geometry/precision rejection tables; no layer-boundary gather added|
|Decoder projections/CCL|BFP4 LoFi; BF8 transfer/KV; FP32 recurrence; shared persistent workspace|Preserve selected packed/separate projections and nativeRS|Stage06 fixes native B1 prefill outK4, preserved; stage05 outK48 is rejected for full-model quality|
|Terminal hidden|One gather if sharded, packed user rows, local final RMSNorm|Keep compact user packing; assess head geometry end-to-end|Stage06 B32 packing removed53ms terminal anomaly; BF16/HiFi4 FP32 policy retained|
|LM head|Vocab-sharded BF16 generic32x5120x62080 matmul|Common LMHead1D explicit DRAM-sharded chunks, legal reader/K-block/input-grid candidates|Baseline reduced device row1650us,75.8% DRAM peak; preserve local shard vocabulary offsets and slice inert physical padding before sampling|
|Logits/sampler|Power-of-two local padding;32local candidates, small candidate gather, semantic k1/p0/temp1|Retain canonical Sampling1D; repeat semantically greedy force-argmax control|No full-vocabulary gather or generic one-core TopK; stage06 normal sampler~452us vsargmax1625us B1|
|Feedback/state|Persistent tt_out_tok, positions/RoPE increment in model trace; page map changed-only|Preserve exact buffers|No host token reconstruction; explicit low-level scheduler-owned state preserved|
|Output|Per-token token readback blocks next enqueue|Traced indexed-fill into128-step persistent token window; one boundary read|Same greedy/sampled token sequence required; first-token read remains TTFT; no per-token host work other than nonblocking trace submission|
|Prefill|Eager prepared physical buckets1,32..4096 with internal tail masks and continuation|Preserve arbitrary logical lengths and context; measure short/long/tail|No advertised capability reduction; profiling remains two real layer kinds only|

Stage06 timeline: request reset/setup -> eager prefill chunks -> terminal/sampler
-> first-token read -> [model trace -> sampler trace -> per-token read] repeated.
Selected stage07 timeline replaces repeated reads with a trace-owned token window and one
read at window completion. Cache/persistent state ownership is unchanged; collector
adds128 aligned token rows plus one index and a sampling-trace variant, not a copy
of weights or KV. Allocation accounting and final repeated-trace tracking are gates.

Common LMHead1D was inspected and selected for the candidate rather than inventing
another projection helper. Its `input_memcfg` converts LazyWeight inputs only;
already-device inputs require explicit to_memory_config in the caller. Both common
samplers were inspected: Sampling1D accepts the existing caller-owned buffers and
traces; SamplingGenerator adds trace/penalty/logprob ownership not advertised by this
generator. Keep Sampling1D and the stage06 comparison rationale.

Head geometry screening keeps BF16 weights/output and HiFi4 FP32 math fixed.
Physical local chunk width is a multiple of256×readers (eight DRAM banks), with
zero-padded channels sliced before concatenation. Input working grids8/16/10/5
provide20/10/16/32 K-tiles per core respectively. Candidate blocks5/10/16/20/32
are legal divisors rather than a fixed small K block. Reader1/2/3 controls use
separate per-role configs; no generic matmul defaults change. Every reported latency
includes reshards, all chunk matmuls, padding slices, concat and output copy. These
are terminal projections on real reduced-stack hidden, not full-model metrics.

Final selection (full-stack accuracy/performance and diagnostic gates passed):

- Pack the last-token prefill terminal across user rows, as already done for decode.
  `prefill_terminal_b32.json` proves exact logits on33/65/4097-token cases including
  mixed inactive slots. All-logits teacher-forcing prefill keeps its existing layout.
- B32 uses async ring candidate gathers in Sampling1D; B1/B8/B16 retain the faster
  baseline gather. `sampling_*b*_v2.json` and intermediate-batch JSONs record same
  semantic greedy settings and shard-boundary maxima. Persistent candidate output
  buffers were measured with required transient ownership copies and were slower.
  Decoder persistent buffers and pooled CCL remain enabled.
- Keep generic BF16/HiFi4 head after19 adapted controls (`head_candidates.csv`).
  Large K16/20/32 DRAM-sharded and interleaved split candidates run correctly but
  lose after reshard/slice/concat costs. Explicit smallK4 changes only~3us, so the
  existing generic configuration remains selected. Lower-fidelity micro-controls
  are not a policy selection; the full-model datatype frontier belongs to stage08.
- Default autoregression uses the traced128-step output collector. Explicit
  `buffered_output=False` retains per-token readout for comparison/compatibility;
  teacher-forcing callbacks and host-sampling compatibility select that path.
  Fixed-length4096/128 measurements have no intermediate window cursor refresh and
  no per-token readback. First-token and final output reads are request boundaries.
- EOS stops returned text at its first occurrence; already queued device work may
  finish the current16-step EOS-aware window (128 for fixed-length throughput). Longer outputs reset the collector cursor
  between windows. This trades prompt EOS return latency for uninterrupted replay;
  it does not provide a streaming API or introduce host token feedback.
