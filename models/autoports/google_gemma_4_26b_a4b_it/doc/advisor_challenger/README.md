# Full-model estimate: 54,633.6 ± 32.4 us before; 47,528.2 ± 32.4 us after

At decode batch 1, `$shard-advise` adds an estimated **7,105.4 us/model
(13.01%)** to the already-optimized Gemma-4 26B A4B decoder. The shipped
change is the advisor-directed 88-core width-sharded implementation of the
hidden-width RMSNorms during decode. The headline is a derived full-model
estimate, not an end-to-end full-model measurement: it combines measured
one-layer deltas using the model-config counts of 25 sliding-attention and 5
full-attention layers.

## Measured decision

Everything ran with `decode_batch = requested_decode_batch = capture_batch =
1`. The frozen sliding control was 1.824021 ms, versus 1.587259 ms for the
candidate. Its five candidate repeats (1.586423–1.588189 ms) all beat all five
control repeats (1.823523–1.824110 ms). The frozen full-attention control was
2.013929 ms, versus 1.776660 ms; again every candidate repeat
(1.775702–1.777690 ms) beat every control repeat (2.012077–2.015620 ms).
Fresh-process confirmations preserved strict separation. A final run with the
environment override removed verified the shipped default itself: all repeats
were below every corresponding incumbent repeat.

The full-model arithmetic is:

- sliding: 25 × (1.824021 − 1.587259) ms = 5,919.044 us saved;
- full attention: 5 × (2.013929 − 1.776660) ms = 1,186.343 us saved;
- total: 7,105.387 us, taking the frozen 54,633.600 us profile-window estimate
  to 47,528.213 us.

The conservative ±32.390 us band is unchanged and comes from linearly summing
the model-scaled incumbent repeat spreads: 14.675 + 17.715 us. It describes
uncertainty in the frozen full-model estimate; it is not a statistical
confidence interval for a directly measured full model.

## Correctness, capture, and attribution

The shipped default passed the real-weight HuggingFace oracle for prefill and
decode: sliding PCC was 0.998810/0.999629 and full-attention PCC was
0.998598/0.999787, all above 0.995. Both cache-sharing modes passed for full
attention. Candidate op-level reports are bounded by the fixed harness's
`PERF_DECODE` signposts in `tracy/norm88_sliding_ops.csv` and
`tracy/norm88_full_ops.csv`.

The pinned advisor was `618cd4e75d`; it was not rebuilt during measurement.
Controls were frozen before capture, and captures used the fixed capture
template hook with executed BF16 attention/dense and BF8 expert weights. Both
generated reconciliations close at 100% and are not `DEGRADED`. Sparse expert
suffixes are uncapturable, leaving 64.70% of the sliding window and 58.51% of
the full-attention window untraced; no contribution is claimed for that time.

Boundary-chain attribution is zero: `advisor_removes_us` is 0.000 for both
kinds. That does not suppress the material advisor disagreement exposed by
reconciliation: four hidden RMSNorms ran on one core while the advisor placed
them across 88 cores. Measuring that doubtful candidate produced the shipped
win. Decisions across kinds use the model-scaled savings above, not per-layer
figures.

## Reported, not screened

The advisor agrees with shipped comparable boundaries costing 7.882 us per
sliding layer and 8.826 us per full-attention layer. Those are real times but
are neither screened nor credited. `model_estimate.layer_handoff` likewise
reports DRAM layer entry with no captured L1 exit and is not screened.

The above-advice 110-core norm attempt is geometrically illegal because 2,816
does not form an integral tile-aligned shard, so 88 is a measured candidate,
not a general core-count recommendation. The DRAM rotary candidate hard-fails
because decode rotary requires sharded input and remains default-off. Existing
DRAM-sharded dense candidate subclasses also remain non-default.
