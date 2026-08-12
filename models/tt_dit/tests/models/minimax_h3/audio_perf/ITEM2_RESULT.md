# Item 2 result: trace is the lever, 280 ms is reachable, and T-sharding returns the wrong audio

Measured 2026-08-12 on `bh-glx-110-a09u02`, 4x8 Galaxy (32 chips), T=207, fusion off unless stated.
Scripts: `factor_scan.py`, `halo_cost.py`, `trace_on_mesh.py`, `fusion_on_mesh.py`,
`divergence_probe.py`.

**Three things, in order of importance.** (1) Trace is worth **3.14x** on a sharded mesh against 1.04x
on one chip, which makes **280 ms** the measured floor at factor 8 and 191-281 ms the projection at 32
-- the target is reachable. (2) **T-sharding is numerically wrong at every factor**, so no sharded
timing here is a result yet; that, not performance, is the remaining work. (3) Chip count on its own is
worth almost nothing untraced -- 32 chips project to 822 ms -- so goal.md's "only 10x-shaped lever"
framing is wrong about the mechanism as well as the size.

## The numbers

| config | median | vs single-device |
|---|---|---|
| single-device mesh, factor 1, fused | 0.9304 s | — |
| single-device mesh, factor 1, unfused | 1.0980 s | — |
| **32-chip mesh, factor 1** | **1.4409 s** | **+343 ms for being on the mesh** |
| 32-chip mesh, factor 4 (axis 0) | 0.9469 s | |
| 32-chip mesh, factor 8 (axis 1) | 0.8820 s | |

Same-axis fit on axis 1 (factor 1 and 8, the only two valid factors there):

    t = 802 ms + 639 ms / factor
    projected factor 32: 822 ms

**Sharding plateaus at ~800 ms.** 32 chips is projected at 822 ms against 882 ms measured at 8, so
going from 8 to 32 chips buys ~60 ms. goal.md's "the only 10x-shaped lever" is off by more than an
order of magnitude: measured 8-chip speedup is 1.63x against the on-mesh baseline and 1.24x against
single-device.

## Why: the 802 ms is mostly not sharding's fault

    ~343 ms   being on a 32-chip mesh at all (factor=1 on mesh vs single-device mesh)
    ~260 ms   the single-chip per-op device floor from item 1 -- op count per chip does not change
    ~200 ms   sharding-specific remainder (halo, ups gather/re-partition, tpad masks, extra dispatch)

The +343 ms is the headline. It is measured with **no sharding at all** -- same graph, same T, same
op count, `parallel_config=None`, just 32 devices open instead of 1. Per-op dispatch has to reach 32
devices instead of one, which is what that looks like. So the honest baseline for judging a shard
factor is 1.4409 s, not 1.0980 s.

## Trace is the whole game on a mesh -- 3.14x, and it changes the answer

    TRACE factor= 1: untraced 1.4447s | traced 1.1141s -> 1.30x  (tracers=1, PSNR inf dB)
    TRACE factor= 4: untraced 0.9380s | traced 0.4690s -> 2.00x  (tracers=1, PSNR inf dB)   axis 0
    TRACE factor= 8: untraced 0.8790s | traced 0.2800s -> 3.14x  (tracers=1, PSNR inf dB)   axis 1

**469 ms at factor 4 -- the number that can be quoted today**, since factor 4 is the correct config and
factor 8 is `KNOWN_BROKEN`. That is 1.98x on the single-device fused 0.9304 s. **280 ms at factor 8**,
once its divergence is fixed. Every traced output is bit-identical to its untraced counterpart with a
tracer confirmed captured, so these are real replays, not silent fall-throughs.

Fitting traced against untraced:

    traced,   axis 1 pair (f=1, f=8):   t = 161 ms + 953 ms / factor   -> factor 32 ~191 ms
    traced,   axis 0 pair (f=1, f=4):   t = 254 ms + 860 ms / factor   -> factor 32 ~281 ms
    untraced, axis 1 pair (f=1, f=8):   t = 802 ms + 639 ms / factor   -> factor 32 ~822 ms

The two traced fits disagree because the factors sit on different mesh axes -- axis 1 is the 8-wide
one, axis 0 the 4-wide -- and the factor is forced to equal the axis length (see blockers). So take
factor 32 as **191-281 ms** rather than a single figure. Both ends clear 300 ms.

Two things follow. The traced fixed cost is **161 ms, not the 260 ms** item 1 attributed to a
single-chip per-op device floor -- so part of that "floor" was dispatch even on one chip, sitting
underneath device time where a 1.04x trace result could not expose it. And **sharding works far better
than the untraced numbers suggest**: traced, 953 ms of the work divides by the factor, against only
639 ms untraced. Dispatch was masking the parallelism.

This is the reconciliation item 1 could not do from single-device data alone. `vocoder_ltx.Vocoder`'s
"~70% host-bound" docstring and the `trace 1.00x` dead-end entry are **both right, about different
configurations**: on one chip the decode is device-bound and trace is worth 1.04x; on 32 chips
dispatch has to reach 32 devices, dispatch becomes the bottleneck, and trace is worth 1.30x unsharded
and 3.14x sharded. **Every untraced multi-chip number in this file understates its configuration by
~3x**, and goal.md's dead-ends list must stop reading "trace (1.00x)" without the single-device
qualifier.

### The caveat that keeps every sharded number from being a result

`PSNR inf` above is traced-vs-plain at the *same* factor -- it proves trace is faithful and nothing
more. **T-sharding is numerically broken at every factor, not just the one marked `KNOWN_BROKEN`.**
With fusion off so the `t_factor=1` baseline actually runs:

    t_factor= 1 axis=1: 1.4556 s   1.00x  PSNR    inf dB
    t_factor= 4 axis=0: 0.9457 s   1.54x  PSNR  -10.1 dB     <- also broken
    t_factor= 8 axis=1: 0.8848 s   1.65x  PSNR  -11.0 dB

An earlier run in this session appeared to show factor 4 as correct. It was not: the fusion-on-mesh
crash (below) killed `t_factor=1`, so factor 4 *became* `baseline_out` and scored PSNR inf against
itself. Any run where the baseline is skipped will do this -- the test guards against all factors
failing, but not against the baseline failing while a later factor silently takes its place.

So **469 ms and 280 ms are both timings of a wrong computation**, and `KNOWN_BROKEN = {(8, 1)}`
understates the problem: `(4, 0)` belongs there too until this is fixed.

## Two suspects killed by measurement

Both were the obvious explanations and both are wrong, which is why they are recorded here rather than
left as plausible.

* **Halo CCLs are not the cost.** `halo_cost.py` times `_t_neighbor_pad` at every decoder stage shape:
  ~202-289 us/call, flat across C=512..8 and identical at pad=1 and pad=25. All ~126 per-conv halo
  exchanges are **~27 ms per decode**, not the ~500 ms they were suspected of.
* **Sharding does not push convs onto the chunked fallback.** The DRAM auto-slice C-chunking warnings
  are **18 per config sharded and 18 unsharded** -- identical. (18/decode is the pre-existing count
  goal.md already documents.)
* **Replicated `ups` work is not the cost either.** `row_model.py`: the unsharded upsamples hold 2.7%
  of rows against the halo-sharded resblocks' 97.3%.

## Route to the target

| route | result |
|---|---|
| sharding alone, factor 32 | ~822 ms (fit) — plateau, do not bother |
| **trace + factor 4 (correct today)** | **0.4690 s measured**, 1.98x on single-device fused |
| **trace + factor 8** | **0.2800 s measured**, but −4.0 dB — fix first |
| trace + factor 32 | 191–281 ms (fit, axis-dependent) |
| any of the above + fusion under sharding | up to a further 1.18x — fusion is off in every number here |

**Trace plus T-sharding already clears the 300 ms bar and projects under the original 200 ms one.**
That reorders the plan completely: item 2's value is not the chip count, it is that a mesh makes trace
worth 3.14x, and item 3 becomes margin rather than a prerequisite.

Order of work from here:

1. **Fix T-sharding correctness -- this is the whole critical path.** Not "fix factor 8": every factor
   is wrong. Start from fault 2 above (shards 1+ saturate, shard 0 does not), because fault 1 is only
   worth 22 dB of a ~50 dB gap. Then re-enable `t_pad` and fix fault 1.
2. **Add a baseline-substitution guard to the test.** `assert baseline_ran` catches "everything
   failed" but not "the baseline failed and factor 4 became the baseline", which is how factor 4 read
   as correct for most of this session.
3. **Make trace the default for multi-chip.** Untraced multi-chip understates by ~3x, so any sharded
   measurement taken without it will mislead the next person the way it misled this session.
4. **Then factor 32** via `AudioTParallelConfig` for the 191-281 ms projection.
5. **Item 3 last**, as goal.md sequences it -- it is now margin, not the critical path.

Performance is no longer the risk: 280 ms is already measured at factor 8 and the projection at 32
clears the original 200 ms target. Correctness is the risk, and it is a bigger job than goal.md's
"why does the existing T-parallel path hang" suggests, because the answer is that it does not hang --
it runs, fast, and returns the wrong audio.

## Where the sharding bug is -- localized, not yet fixed

`divergence_probe.py` splits it into two independent faults.

**Fault 1: the T-padding tail, worth ~22 dB.**

    PROBE T= 207  t_pad= 49  32 rows/chip  fully-padding shards=1  PSNR -11.0 dB
    PROBE T= 256  t_pad=  0  32 rows/chip  fully-padding shards=0  PSNR +11.0 dB

Removing the padding improves PSNR by 22 dB, so the `_set_tpad_tail` path is genuinely wrong. The
mechanism is not the tile edge the test comment blames ("256/8 = 32 exactly one tile per shard"): at
factor 8, `t_pad = 49` over 32-row shards leaves **shard 7 holding nothing but padding**, and
`mode="replicate"` has to materialize "the real last row", which then lives on shard 6. factor 4's
64-row shards never fully pad, which is why the two factors differ. But note factor 4 is *also* broken,
so this is not the whole story.

**Fault 2: something structural, present with no padding at all.** T=256 still diverges (+11 dB, far
below the 40 dB gate). `_localize_divergence` on factor 4 says what it looks like:

    per-shard mean error: ['0.270666', '1.000112', '1.000085', '1.000107']
    boundary bands (+-128): ['1.004378', '0.993146', '1.004494']  interior 0.816890  ratio 1.23
    correlation at lag 0: 0.0426; best 0.0426 at lag 0

Shards 1-3 sit at mean error ~**1.000** against a reference whose absmax is 0.2842 -- they are
**saturated** at the closing `tanh`/`clamp(-1, 1)`, i.e. producing garbage, while shard 0 is different
(0.271). The boundary-to-interior ratio is only 1.23, so the error is spread through the interior
rather than concentrated at shard edges, and correlation is 0.04 at lag 0 with the best lag *at* 0 --
so it is **not** a shift or trim bug, and not a halo-width bug. At factor 8 the correlation is `nan`
(constant output), consistent with full saturation.

That shape -- only the first shard computing anything sensible, the rest saturating -- points at
per-shard state rather than at boundary math: weights or activations not valid beyond shard 0. Ruled
out already by reading: halo width does account for dilation (`eff_k = (kernel_size - 1) * dilation +
1`, `same_pad = eff_k // 2`, so k=11 d=5 correctly asks for 25); the live conv path does halo when
sharded; the resample up/down paths use `_t_neighbor_pad` when sharded and `_replicate_pad_t` only
when not; `_forward_tap_matmul` pads with the local `_zero_pad_t` and would be wrong under sharding,
but it is gated off by default (`MINIMAX_H3_AUDIO_TAP_MATMUL`, on only in accurate mode) -- **a latent
sharding bug for accurate mode regardless**.

## Blockers a 32-chip number has to clear first

1. **factor 8 returns the wrong signal** -- PSNR -4.0 dB against the factor-4 output, divergence mean
   0.200 against baseline absmax 0.284, i.e. the output is unrelated rather than slightly off. It is
   `KNOWN_BROKEN` in the test for this reason. A fast wrong answer is not a result.
2. **factor 32 needs `AudioTParallelConfig`** (both mesh axes), which is a different, less-exercised
   code path -- its halo does one CCL per axis, so two per conv.
3. **The T-shard factor must equal the mesh axis length.** factor=2 and factor=4 on the 8-wide axis 1
   both die in `_partition_t` at `slice_device_operation.cpp:164` ("height begin index aligned to
   tiles"). Only factor=4 on axis 0 and factor=8 on axis 1 run. This is why no single-axis scaling
   curve exists on a 4x8 mesh and why the two measured factors differ in axis as well as factor.
4. **Fusion is off for every sharded number above**, so the sharded path gives back the week's 1.181x.
   The gate is legitimate: `_forward_fused` builds boundary padding from the *local* tensor's
   first/last rows, which are not the global ones on a T-shard. Lifting it means moving that padding
   onto `_t_neighbor_pad`.

## Fixed here: fusion could not run on a mesh at all

`_snake_conv_params` read alpha/beta with a bare `ttnn.to_torch`. Those parameters are replicated
across the mesh, so `MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV=1` plus any multi-device mesh died at
`buffers.size() == 1` (pytensor.cpp:299). Because the gate above declines when `factor > 1`, the only
configuration that reached the bad line was **unsharded-on-a-mesh** -- exactly
`test_audio_decode_t_parallel`'s `t_factor=1` baseline. And since the test asserts the baseline ran,
one dead readback presented as "the T-parallel path is entirely unavailable". Now reads one shard, the
same shape as `Vocoder._device_to_host` and `_project_latents_device`. `fusion_on_mesh.py` is the
cheap regression for it.

## Correction to goal.md's framing of the hang

"Its one test (`mesh4x8`, line 719) times out at 300 s, pre-existing" and "why does the existing
T-parallel path hang" both assume a hang. **There is no hang.** The test takes **~12 minutes**
(13:24:41 -> 13:36:54 for three factors, each rebuilding the decoder and running warm + 3 iters) and
CI kills it at 300 s. `reset_cluster.sh` and the fabric-flakiness branch of that question are not
needed.
