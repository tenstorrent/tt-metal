# LTX-2 audio decode — submesh perf profiling + lessons

Blackhole Galaxy. Branch `smarton/optimizer/ltx-audio-kernel` worktree at `ltx-perf`
HEAD `6182a3b34c0` (all audio wins incl env-gated `LTX_AUDIO_SUBMESH` routing) plus a
harness-only commit `0d27007987c` (adds the 1x1/1x2 sweep cases). Timestamps UTC / PT.
All numbers below are observed on device, not estimated.

The audio decode is torch-in / torch-out and self-contained, so `LTX_AUDIO_SUBMESH=RxC`
routes ONLY `decode_audio` onto an `RxC` create_submesh slice of the full mesh while the
video pipeline keeps the whole mesh. The decode's waveform tensor is sharded along the
time (T) axis by `t_factor = max(submesh_axes)`: 1x1 → T-shard=1 (replicated, no halo),
1x2 → 2, 1x4 → 4, 2x4 → 4, 4x8 → 8.

## (b) WHICH numbers show it — the full warm-decode-vs-chips curve

Warm steady-state decode, EAGER, conv1d path, audio routed to the submesh
(`audio_warm_harness.py`, device-synced per stage, median of 5 reps). The 1x1/1x2 row is
the new sweep (job 080840-40, 2026-06-14 08:24-08:29 UTC / 01:24-01:29 PT); 1x4 re-measured
this run (job 080305-38); 2x4/4x8 from the prior Phase-8 sweep on the same code.

| config | chips | T-shard | mel_vae | vocoder+bwe | total warm | cold assembly |
|--------|-------|---------|---------|-------------|------------|---------------|
| 1x1    | 1     | 1       | 39.0ms  | **1427.8ms**| **1462.7ms** | 102.8s\* |
| 1x2    | 2     | 2       | 40.3ms  | **1010.7ms**| **1051.0ms** | 120.6s\* |
| 1x4    | 4     | 4       | 39.9ms  | **632.9ms** | **675.3ms**  | 5.6s (prior, fully-primed) |
| 2x4    | 8     | 4       | 40.9ms  | 646.8ms     | 687.8ms      | 141.4s (prior) |
| 4x8    | 32    | 8       | 54.9ms  | 786.9ms     | 841.8ms      | 495.8s (prior) |

\* The 1x1/1x2 cold here is confounded by one-time JIT kernel recompile into the
worktree-keyed cache (JIT cache hits 46.1% on 1x1, 58.9% on 1x2 — the cache was still
priming), so it is recompile + assembly, NOT clean device-side assembly. The clean
cold-assembly numbers (fully-primed kernel cache, prior Phase-8) are the 4x8/2x4/1x4 column.

### The curve is U-shaped, NOT monotonic — this corrects the prior hypothesis
The prior sweep (4x8 → 2x4 → 1x4, T-shard 8 → 4 → 4) only ever halved T-shard from 8 down
to 4 and saw monotonic speedups, which read as "smaller is always faster." Extending the
sweep BELOW T-shard=4 (to 1x2 T-shard=2 and 1x1 T-shard=1) reveals the real shape:

```
vocoder+bwe warm:  1x1 1428ms  >  1x2 1011ms  >  1x4 633ms  <  2x4 647ms  <  4x8 787ms
T-shard:               1            2             4 (min)       4             8
```

The minimum sits at **T-shard=4 (1x4)**. Going above it (4x8, T=8) and going below it
(1x2 T=2, 1x1 T=1) are both slower. So audio decode does not simply prefer the smallest
mesh — it prefers the T-shard count that balances two opposing device-side costs.

### The cold-assembly collapse (fully-primed cache, prior Phase-8)
One-time device-side program / mesh-workload assembly for the ~4400-op audio graph
(NOT JIT compile — measured at 100% kernel-cache hits): **496s (4x8) → 141s (2x4) →
5.6s (1x4)**. Fewer chips = fewer cross-chip program barriers to assemble, so the
one-time cold cost collapses ~88x from 4x8 to 1x4.

### The ~503ms device-side vocoder gap + per-RISC split (prior, 4x8 warm)
On 4x8 the vocoder+bwe stage is ~58% device-bound, not host-dispatch-bound: 865ms wall vs
362ms device-active = ~503ms gap. The PARADOX-RESOLVED probe (eager vs traced, device-
synced, one warm process) showed the gap does NOT shrink under trace (main_vocoder
436.0ms eager → 453.7ms traced, +17.7ms SLOWER) — a host-dispatch-bound graph would have
collapsed toward its device floor. So the gap is per-op DEVICE-side latency. Phase-1 per-RISC
device-FW (BRISC / NCRISC / TRISCmax): Conv3d 12588 / 12477 / 12578ms (balanced, ~saturated);
the data-movement ops NeighborPad / Slice / Concat run TRISC=0 (pure DM) — these are the
cross-chip halo + reshape overhead that scales with T-shard count.

## (a) WHY smaller-but-not-too-small is faster — the exact mechanism

The vocoder convolutions are stride-1 causal in time. With the waveform T-sharded across
`t_factor` chips, every per-conv causal context that straddles a shard seam must be fetched
from the NEIGHBOR chip. That fetch is `neighbor_pad_async` — a FABRIC (ethernet) CCL op
(`neighbor_pad_async_program_factory.cpp` includes `fabric.hpp`: fabric writers send boundary
rows to the neighbor chip, fabric readers receive the halo into L1, gated by GlobalSemaphores
+ barrier sems). Each such halo is a cross-chip exchange with a receive-before-compute barrier.

Two opposing device-side costs trade off as T-shard count changes:

1. **Cross-chip halo-barrier cost — grows with T-shard count.** More T-shards = more shard
   seams = more `neighbor_pad_async` fabric exchanges and semaphore barriers per decode. This
   is the dominant device-side cost on the large mesh: it is exactly why 4x8 (T=8, 787ms) is
   slower than 1x4 (T=4, 633ms) — halving the T-shard count halves the seam count and cut
   vocoder+bwe ~150ms. Each per-conv halo is a genuine device-side sync point that trace
   cannot remove (it is not host dispatch — confirmed by the paradox probe), and the cold
   assembly of all those cross-chip program barriers is what makes 4x8 cold 496s vs 1x4 5.6s.

2. **Per-chip serial conv compute — grows as T-shard count shrinks.** With fewer T-shards
   each chip owns a LONGER T-segment, so each conv does proportionally more sequential work
   on that one chip; the T-parallelism that overlaps conv compute across chips is lost. At
   T-shard=1 (1x1) there are ZERO halos (replicated, no fabric, no barrier) — yet it is the
   SLOWEST (1428ms) because all of the vocoder's conv compute is serialized onto a single
   chip. T-shard=2 (1x2, 1011ms) splits the compute over 2 chips but still leaves each chip
   with twice the 1x4 segment.

The sum of (1) + (2) is minimized at **T-shard=4 (1x4)**: enough T-parallelism to keep the
per-chip compute small, few enough seams to keep the fabric-halo barrier count low. Below 4,
compute dominates; above 4, halo barriers dominate. This is why 1x1 (zero halos) is NOT the
winner — the lever is not "fewest chips," it is "the T-shard count that balances fabric-halo
barriers against per-chip serial compute," and on this graph that optimum is 4.

(Carry-forward: the only way to push below the T=4 floor without paying per-chip compute is
to cut the halo barriers themselves — e.g. run the vocoder UNSHARDED on T within the submesh,
gather once / many local convs / partition once, trading CCL for redundant local compute. The
conv-kernel halo-fold is closed as infeasible: the halo is a fabric CCL receive, and the conv
readers address only local pages via TensorAccessor — folding it would re-implement
neighbor_pad in-kernel with the same barriers, no op-count or sync-point reduction.)

## E2E accuracy gate for the pushable {1x1, 1x2} configs

Push rule: may land to ltx-perf ONLY a 1-chip (1x1) or 2-chip (1x2) config, and only after
full E2E accuracy is confirmed; pick the fastest that passes. Of {1x1 1463ms, 1x2 1051ms},
**1x2 is the faster**, so it is the candidate.

### (a) Audio PCC — 1x2 PASS (job 083228-41, 2026-06-14 08:32-08:34 UTC / 01:32-01:34 PT)
`test_audio_decode_girl -k bh_4x8sp1tp0`, `LTX_TRACED=0`, `LTX_AUDIO_SUBMESH=1x2`:
- "Audio decode routed onto 1x2 submesh of 4x8" confirmed.
- **conv1d-vs-torch PCC = 0.99664** (> baseline 0.99379, gate PASS > 0.95); mac-vs-torch 0.99936.
- Audio decoded on device to **(2, 288480) = 6.01s @ 48000Hz** — valid audio. warm 1044.3ms.
- The pytest fixture then SIGABRTs at teardown (`MeshDevice cq ID 0 is in use by child
  submesh ID 2 during close of mesh ID 1`) — the documented cq-sharing-child teardown
  limitation. The decode + PCC produced valid results BEFORE teardown; the crash is the
  fixture closing the parent mesh while the child submesh is alive, not the decode.

### (b) Full E2E — TRACED path HANGS at audio-submesh warmup
`test_pipeline_distilled -k bh_4x8sp1tp0_ring`, `LTX_TRACED=1`, `LTX_AUDIO_SUBMESH=1x2`
(job 083228-41): Stage 1 + Stage 2 denoise completed, video VAE loaded, then HUNG at
`warmup_buffers: warmup audio decode (on-device)` — capturing the audio TRACE on the
cq-sharing 1x2 child submesh. No log progress for 8.5 min, no mp4 produced; killed.

### (b) Full E2E — EAGER path ALSO HANGS at audio-on-submesh (job 085025-44)
`test_pipeline_distilled -k bh_4x8sp1tp0_ring`, `LTX_TRACED=0` (eager, no trace warmup),
`LTX_AUDIO_SUBMESH=1x2`: the full video pipeline completed cleanly on the parent 4x8 mesh —
Stage 1 denoise 71.6s, latent upsample 13.5s, Stage 2 denoise 42.2s, video VAE decode 35.8s
→ valid (1, 3, 145, 1088, 1920) video tensor. Then the FIRST audio decode on the 1x2 child
submesh HUNG (log frozen 7+ min, no mp4). So 1x2 audio decode works STANDALONE (gate a) but
DEADLOCKS when it runs on a child submesh AFTER the video pipeline ran on the parent mesh —
in BOTH traced and eager E2E. **1x2 FAILS the full-E2E gate (never produces audio+video).**

### 1x1 — INDETERMINATE (device firmware-init stuck, cannot reset per safety rule)
SIGKILLing the hung 1x2 eager E2E mid-fabric-CCL corrupted the board: every subsequent job —
the 1x1 gate (jobs 090929-47, 091141-48) and even a bare 4x8 device-open probe (091759-50) —
fails at device init with `Device 0: Timeout (10000 ms) waiting for physical cores to finish …
failed to initialize FW! Try resetting the board.` The board needs a `tt-smi` reset, which the
safety rules forbid me from issuing. So 1x1's full-E2E behavior could NOT be evaluated. (1x1
routes audio onto a 1x1 child submesh too, so the same child-submesh-after-parent-video deadlock
is the likely failure mode, but this is unverified.)

## DECISION — push NOTHING to ltx-perf
The strict rule: land the FASTEST of {1x1, 1x2} that passes BOTH (a) audio PCC and (b) a clean
full E2E; if neither passes, push nothing and report why.
- **1x2 fails (b)** — the full E2E hangs at the submesh audio decode (no audio+video produced),
  in both traced and eager. PCC (a) passed standalone, but a clean E2E is mandatory.
- **1x1 is indeterminate** — board stuck before it could be evaluated; no clean E2E pass.
Neither config achieved a clean full-E2E pass, so **NOTHING was pushed to ltx-perf.** The
env-gated `LTX_AUDIO_SUBMESH` routing already on ltx-perf (`6182a3b`) is unchanged — it stays the
opt-in capability it was. The standalone audio-only win (1x4 warm 675ms vs 4x8 842ms, PCC clean)
is real and measured, but 1x4 is not pushable per the rule, and the small pushable configs do not
survive a full AV E2E on the cq-sharing child submesh.

### Root cause of the E2E failure (grounded)
A `create_submesh` child SHARES the parent mesh's command queue. The full AV E2E runs video on
the parent 4x8 mesh, then decode_audio on the child submesh that shares the parent's CQ — and
that shared-CQ child audio decode deadlocks after the parent has issued/serialized video work on
the same CQ. This is the same ttnn limitation already documented for TEARDOWN (parent-close-with-
live-child throws; child-close-while-parent-alive hangs), now shown to ALSO break the audio
decode itself in the live AV pipeline, not just at teardown. The audio-only test (decode_audio
with NO preceding parent-mesh video work) does not hit it, which is why gate (a) and the warm-
harness sweep pass. Default-on small-submesh audio routing in the full AV pipeline needs the
ttnn fix the memory note calls out: a SEPARATE command queue per submesh (or cascade close-
children-first). Until then the routing is usable only in an audio-only / one-shot context.

## Files / commits
- `models/tt_dit/tests/models/ltx/audio_warm_harness.py` — 1x1/1x2 sweep cases
  (commit `0d27007987c` on `smarton/optimizer/ltx-audio-kernel`, NOT ltx-perf).
- Scripts (worktree-targeted, in /home/smarton): `run_submesh_sweep_acc.sh`,
  `run_submesh_small_acc.sh`, `run_gate_1x2_acc.sh`, `run_e2e_1x2_eager_acc.sh`,
  `run_gate_1x1_acc.sh`, `probe_device.sh`; env `ltx_wt_acc_env.yaml`.
- **Pushed to ltx-perf: NOTHING** (no {1x1,1x2} config achieved a clean full-E2E pass).
- DEVICE LEFT NEEDING A RESET: jobs after 085025-44 fail FW init — operator `tt-smi` reset
  required (I am not permitted to reset the board).
