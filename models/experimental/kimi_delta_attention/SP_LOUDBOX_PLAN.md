# KDA LoudBox sequence-parallel prototype

## Objective

Turn the existing `TP=8` whole-head KDA implementation into a correctness-first
sequence-parallel prototype on the available eight Blackhole devices, without
regressing the existing path.  The production scale-out shape remains
`SP=8 x TP=4` on 32 Galaxy devices; LoudBox is the place to validate the
state-transfer protocol and measure its components.

## Baseline (verified on this branch)

* Branch: `pjosipovic/kda-loudbox-sp-prototype`, based on Momcilo's
  `mvasilijevic/codex/kimi-linear-kda` at `671b0c9acb6`.
* Hardware: one eight-device Blackhole LoudBox.
* `test_ttnn_layer.py`: 7 passed (T=1, 4, 32, state continuity and external
  state tests).
* `test_tp_weights.py`: 2 passed; TP=8 output PCC 0.999965, recurrent-state
  PCC 0.999910, convolution-state PCC 0.999997.
* Existing target-shape TP=8 measurements, from the branch reports, are
  619.594 us at T=640 and 3183.263 us at T=5120.  At T=5120 the output
  projection's fused matmul/reduce-scatter is about 1.04 ms, so its CCL is a
  primary reason that merely tuning local KDA will not meet the next goal.

## Milestones

1. **Protocol reference and tests.** Add a partitioned PyTorch reference that
   exactly composes KDA spans.  It must show that each sequence boundary needs
   the incoming recurrent state plus the preceding three projected Q/K/V
   samples for the causal short convolution.
2. **LoudBox SP=8, TP=1 protocol probe.** Each chip owns one contiguous T/8
   span and all 32 heads.  Add explicit point-to-point handoff of the state and
   three-value convolution carry, with PCC >= 0.98 at T=256, 640, and 5120.
   This is a correctness/control-plane experiment, not a promised speedup.
3. **LoudBox SP=2, TP=4 topology.** Refactor logical TP from physical mesh
   size, preserve TP=4 output reduce-scatter inside each group, and transfer
   the per-TP-rank carry at the single SP boundary.  Validate the same outputs
   and state criterion.
4. **Galaxy-ready affine scan.** Replace serialized state relay by a
   log-depth scan over each span's affine summary `(A, B)`.  For K=V=128 and
   TP=4, a rank owns eight heads: `A` is 512 KiB and `B` is 512 KiB, so one
   uncompressed summary is 1 MiB.  Keep the actual recurrent state FP32; do
   not silently quantize it.
5. **Profiler gate.** Add/profile harnesses for TP=8, SP=8/TP=1 and
   SP=2/TP=4.  Record operation timing and transfer payloads before proceeding
   to Galaxy integration.

## Performance goals and decision gates

* **LoudBox M2 gate:** no end-to-end speedup is required.  It must establish
  the handoff cost and retain at least 80% of the sum-of-local-span throughput;
  the resulting trace must contain no host-mediated tensor round trip.
* **Galaxy target (T >= 5120):** reduce KDA layer latency by at least 2.5x
  versus the TP=8 3.183 ms control, to <= 1.273 ms.  The aspirational target
  is 3.0x, <= 1.061 ms.  These are end-to-end KDA targets, not a claim that
  LoudBox has the 32-device resources to demonstrate them.
* **Stop/adjust condition:** if the measured affine-summary scan cannot be
  hidden behind local span work, do not proceed with a full `SP=8 x TP=4`
  model integration; first fuse summary generation/consumption or reduce the
  transport representation with an accuracy study.

## Non-goals for this first slice

* No FP32 recurrent-state downgrade.
* No all-reduce used in place of the ordered causal carry.
* No unrelated TP=8 micro-tuning before the cross-device SP bottleneck is
  measured.
