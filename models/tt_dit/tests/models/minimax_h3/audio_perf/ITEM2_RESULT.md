# Item 2 result: sharding plateaus at ~800 ms, and the mesh itself costs 343 ms

Measured 2026-08-12 on `bh-glx-110-a09u02`, 4x8 Galaxy (32 chips), T=207, fusion off unless stated.
Scripts: `factor_scan.py`, `halo_cost.py`, `trace_on_mesh.py`, `fusion_on_mesh.py`.

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

### The caveat that keeps 280 ms from being a result

`PSNR inf` above is traced-vs-plain at the *same* factor -- it proves trace is faithful and nothing
more. factor 8 is the `KNOWN_BROKEN` config, **-4.0 dB against the single-device path**, so 280 ms is
currently a fast wrong answer. The quotable number is a traced factor 4, and 280 ms only becomes real
once factor 8's divergence is fixed.

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

1. **Fix factor 8's -4.0 dB divergence.** This is the only thing between 280 ms and a shippable
   number. The test's own comment names the suspect: 207 frames pad to 256, and 256/8 = 32 is exactly
   one tile per shard, so a boundary is landing on a tile edge.
2. **Make trace the default for multi-chip**, and re-measure factor 4 traced as the correctness-clean
   datapoint.
3. **Then factor 32** via `AudioTParallelConfig` for the ~191 ms projection.
4. **Item 3 last**, as goal.md sequences it -- it is now margin, not the critical path.

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
