# Wan2.2 findings: 2026-09-16

The [full report](suite-01/REPORT.md) contains paired videos, frame sheets,
CLIP, block timings and real-input L2/PCC. This is a two-prompt, one-seed
exploratory evaluation, not a model-quality qualification.

- D and C retained their accuracy bands on the four sampled real Wan inputs:
  D 0.167–0.199% L2, C 0.193–0.493%. They took 264.4 and 237.1 seconds/video
  on average, versus 169.5 for same-suite stock.
- E took 162.9 seconds/video, versus B's 188.5. The sampled videos broadly
  preserve the scene and human detail, but E's worst captured operator L2
  was 6.08%, versus B's 5.02%. Two videos do not settle model quality.
- Do not discard F yet: it took 175.6 seconds/video and had lower L2 than B
  on each of the four identical QKV captures (maximum 4.70% versus 5.02%).
  It also beat B on all four initial full-block timings. This is a different
  tradeoff from the short-context FLUX observations, not proof of universal
  dominance. Block timing warmup and ordering caveats remain relevant.
- G achieved 159.5 seconds/video and genuine BFP4_B KV transport, but its
  operator L2 was 6.09–23.69%. Sampled butterfly frames have visibly smeared
  wing/flower detail; the human frames also have softer, smeared detail and
  a larger change in appearance. Its competitive CLIP scores do not capture
  this degradation. Keep G as a compression research option; this result
  argues for the conditioning/preprocessing investigation before treating
  it as a transparent model drop-in.

The experimental kernels use prepared-format KV all-gather; stock uses ring
attention. These video times are integration-level comparisons, not isolated
numerical-kernel speedups or ring-scaling measurements. G's KV uses 576 bytes
per tile versus 2048 for BF16 (including block-format exponent overhead),
but this experiment does not measure scaling along the ring axis.

All 14 videos and 112 PNGs passed hash/shape validation; all 28 sampled block
replays were exact. The accuracy sweep completed all 24 measurements, not
24 passes of a 0.5% gate. Original BF16 inputs evaluated in FP64 provide the
operator reference; D is only a video comparator, not ground truth. Captures
cover four blocks in a two-step butterfly pilot, not the whole trajectory
or the human prompt. CLIP uses eight sampled frames and does not assess
temporal quality. Visual observations above refer to inspected sampled PNGs,
not a comprehensive video-quality study.

The generation test passed in 48m27s, with 45.2 minutes inside timed video
calls. All converted weights came from cache. The native partial-tile mask
fix needed by F changes only unpack format, not exp arithmetic. The full
generation log and initial padded-mask qualification log are retained.
