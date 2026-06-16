# Plan: make LTX audio decode SCALE and finish << 600ms (NOT YET IMPLEMENTED)

Status: design only. Current best ~672ms (1x4 submesh); 4x8 ~842ms. Audio scales NEGATIVELY today
(more chips = slower). This is the plan to fix both the absolute latency and the scaling sign.

## Diagnosis (measured, see audio-perf.md / PLAN_PROGRESS.md)
- The vocoder+BWE is **device-side-latency bound**, NOT compute or host-dispatch bound: warm wall
  ~790-865ms vs device-active ~362ms → ~500ms is per-op gaps. Trace replay does NOT shrink it
  (confirmed +17ms), so it's device-side (per-op sync/scheduling), not host dispatch.
- The gaps are **cross-chip causal-halo barriers**: under T-sharding each conv calls
  `neighbor_pad_async` (a fabric/ethernet CCL op) to fetch boundary rows from neighbor T-shards.
  **Barrier count ∝ T-shard factor.** 4x8 → T-shard=8 → most barriers; 1x4 → T-shard=4 → fewer.
- ~3100 small device ops per decode → high per-op latency sum + the one-time device graph-assembly
  cost scales with chip count (496s on 32 chips vs 5.6s on 4).
- **The warm-decode curve is U-SHAPED in T-shard count (measured): 1x1(T=1) 1463ms · 1x2(T=2) 1051ms
  · 1x4(T=4) 675ms[min] · 2x4(T=4) 688ms · 4x8(T=8) 842ms.** Two opposing forces:
  (i) cross-chip halo barriers ∝ T-shard (hurts high T: 4x8), (ii) per-chip SERIAL conv compute
  ∝ 1/T-shard (hurts low T: 1x1 has ZERO halos yet is SLOWEST). T-shard=4 is the balance point.
- WHY it doesn't scale: it's not "fewer chips = faster" (1x1 is worst). Adding T-shards past 4 adds
  halo barriers faster than it removes per-chip work; dropping below 4 serializes conv compute on
  too few chips. Either way the 6s workload is too small to amortize.

## Target
- **< 600ms** warm decode (stretch: ~300-400ms) from the 1x4 min of 675ms. The U-shape means you
  can't get there by just changing chip count — both ends are worse. Must attack BOTH forces.

## Levers (ranked; the target needs op-count cut + breaking the U-shape, not a chip-count pick)
1. **Op-count reduction (biggest absolute win, helps at every T-shard).** Fuse the ~3100 ops:
   - Fuse per-block elementwise chains (snake/residual/bias) — fewer BinaryNg/Ternary dispatches.
   - Collapse per-conv Slice/Concat/NeighborPad scaffolding (polyphase already cut; more remain).
   - Fewer ops → less device-side per-op latency AND less graph-assembly. Effort: med, PCC-safe.
2. **Break the U-shape: get T-locality WITHOUT per-chip serialization.** Note 1x1 (zero halos) is the
   SLOWEST → simply de-sharding T is wrong. Want halo-free AND parallel:
   - Parallelize across chips by CHANNEL or BATCH (stereo / mel-chunks / clips) while keeping T LOCAL
     per chip — convs become local (no neighbor_pad) and the extra chips do real parallel work instead
     of adding halos OR serializing. This is the design that could scale positively.
   - Trade-off: a gather/scatter at stage boundaries (once, not per-conv); memory for replicated state.
   - Structural; effort: high; PCC-gated. This is the crux of "scale correctly."
3. **Overlap/async any residual halos**: double-buffer the halo exchange so the fabric barrier overlaps
   adjacent compute instead of sitting on the critical path. Effort: med-high.
4. **Batch halos per AMP block** instead of per conv (one exchange feeds several local convs). Effort: med.
5. NOT a lever: trace (device-bound, net-negative); blindly changing chip count (U-shaped, both ends worse).

## Sequencing
A. Re-profile the 1x4 config per-op (counters + tt-npe) to get the exact post-submesh op/halo budget.
B. Lever 1 (op-count) first — safest, biggest absolute drop toward <600ms.
C. Lever 2 (de-shard T / channel-or-batch parallel) — the scaling fix; validate it beats 1x4 at 2x4/4x8.
D. Levers 3/4 if a halo residue remains on the critical path.
Each step: PCC gate (test_audio_decode_girl) + warm-decode + cold-assembly + per-RISC measurement.

## Open questions
- Does channel/batch parallelism for the vocoder fit L1/DRAM without the T-shard split? (memory check)
- Is the mel-VAE (already 56ms) worth re-parallelizing or leave as-is?
- Can the BWE half (~449ms, a full 2nd vocoder) share the de-sharded layout?
