# Item 1 result: kernel-bound, with a ~260 ms floor that neither trace nor sharding removes

Measured 2026-08-12 on `bh-glx-110-a09u02`, branch `rouzbeh/minimax-audio` at `b30d1792f6a`, both
fusion flags on. Baselines reproduced first, so everything below sits on the same ground as the
week's numbers: **fused 0.9318 s** (goal.md says 0.9304) and **default 1.0980 s** (goal.md says
1.0982).

## Verdict

**Kernel-bound. Items 2 and 3 are the right work.** But the decode splits into two parts with
different fixes, and the plan's framing only covered one of them.

| | |
|---|---|
| decode, fused | 0.9304 s |
| T-independent component | **~260 ms** |
| data-proportional component | **~670 ms** |
| host dispatch | **<= 37 ms** |

## How it was settled -- not with the profiler

The profiler cannot answer item 1, and the attempt to make it do so is where the contradiction came
from. Two totals are on record for this same stage: **224 ms** (amendment 66 calls it a ~6x
undercount) and **1401 ms** (`PROFILE_2026_08_06.txt`). The second exceeds the untraced wall clock,
which is impossible for device time. `test_tracy_audio_decode`'s own docstring already says to treat
the per-op ranking as the product and not the absolute total -- that warning is correct and should be
obeyed. `occupancy.py` re-reads that CSV with a denominator and shows why the sum is unusable: the
host-clock span over the window is 507.5 ms, i.e. the host finished enqueuing all 6955 ops in 507 ms
and then waited. That span is the enqueue window, not the decode, so neither it nor the FW sum is a
valid basis for a host-vs-kernel claim.

Two profiler-free measurements settle it instead.

**Trace (`test_audio_decode_traced`).** A trace replay issues the whole graph as one pre-recorded
command, so it removes essentially all host dispatch. Removing all of it buys 37 ms of 945:

    PERF audio_decode_5s untraced 0.9450 s | traced 0.9081 s -> 1.04x
    PERF split: dec_in_proj 0.0020 s | vocoder 0.9415 s | vocoder traced 0.9071 s

So device time is ~0.90 s, not ~0.2 s. This also kills two beliefs: `vocoder_ltx.Vocoder`'s "~70%
host-bound" docstring is **wrong** for this path, and the `dec_in_proj` host round-trip that docstring
blames is **2 ms**. The dead-ends list entry "trace (1.00x)" is right; 1.04x reproduced.

**T-sweep (`t_sweep.py`).** Every loop in `Vocoder._forward_device` is over `num_upsamples` /
`num_kernels` / `num_branches` -- none over T -- so op count is the same 6955 at any sequence length
while the data per op scales with T. Sequence length is not tile-padded when unsharded
(`_upload_BCT` pads only when `factor > 1`), so T is a clean knob:

| T | median |
|---|---|
| 207 | 0.9304 s |
| 160 | 0.7295 s |
| 104 | 0.5049 s |
| 52 | 0.3500 s |
| 26 | 0.2980 s |
| 13 | 0.2794 s |

The data is convex, so a single least-squares line understates the intercept -- it reports 199.5 ms
fixed + 3.373 ms/latent, and underpredicts T=207 by 3.5%. The local slope near the bottom
(T=13 -> 26) is only 1.43 ms/latent, which puts the true T -> 0 limit at **~260 ms**. Take 260 ms as
the floor and ~670 ms as data work.

**`op_pipeline.py`** closes the last loose end -- the "6955 ops x 180 us = 1254 ms of floor" quote:

      chained           7.10 ms total  ->    141.9 us/op
      independent       6.29 ms total  ->    125.7 us/op
      per-op-sync       6.94 ms total  ->    138.8 us/op

All three are equal, so that microbenchmark is host-*issue*-bound and cannot measure a device floor
at all. `op_floor.py`'s 180 us/op was host issue cost. **Retire the 1254 ms floor number.** The real
per-op device cost is ~260 ms / 6955 = ~37 us.

## Why this matters more than the host/kernel answer

The target is <= 200 ms (relaxed by the user on 2026-08-12 to "222 ms good, happy with 300 ms"). The
~260 ms floor is T-independent, so **T-sharding does not divide it and trace does not remove it.**
That caps every sharding-only route:

    factor 32:  260 + 670/32  = ~298 ms      <- and only if sharding were free
    factor  8:  260 + 670/8   = ~344 ms
    factor  4:  260 + 670/4   = ~428 ms

So item 2 is a **~3x lever with a hard floor at the target**, not the 10x lever goal.md describes.
Any route to 300 ms with margin has to cut the floor too, which means deleting ops -- exactly the rule
the week already established ("removing a full-tensor pass wins; trading one for another does not").

## Corrections to goal.md

* **"There is no 8-chip configuration in that file"** -- there is. `test_audio_minimax_h3.py:729`
  has `FACTORS = [(1, 1), (4, 0), (8, 1)]` with `KNOWN_BROKEN = {(8, 1)}`, and the comment above it
  records t_factor=8 at **0.898 s** with PSNR **-6.3 dB** vs single device. goal.md asks for this to
  be reconciled first; this is the reconciliation. 8-way T-sharding is measured at 1.04x, so the
  10x premise was already contradicted in-tree before item 2 starts.
* **The ~70% host-bound docstring** in `vocoder_ltx.py:361` should be corrected or deleted; it is the
  origin of the trace-is-the-dominant-lever belief.
* **Do not quote absolute Tracy totals for this stage** -- 224 ms and 1401 ms are both on record and
  neither is device time.

## Where the sharding overhead actually is (`row_model.py`)

Sizing the two levers before spending device time on them. A row costs ~4.2 ns almost regardless of
width, so rows per stage is a first-order cost model:

    ups rows            327,060    2.7%  (NOT sharded)
    resblock rows    11,774,160   97.3%  (sharded via halo)

`ConvTranspose1dViaConv3d` does **not** shard -- `audio_ops.py:1678` says "Inner conv stays UNSHARDED;
forward gathers T, runs unsharded, then re-partitions", so under T-sharding every chip redoes all 7
upsamples in full and pays an all-gather + re-partition + 4 layout conversions each. That reads like
the reason sharding fails, but it is not: the ups are only **2.7% of rows**. The resblocks, which are
97.3%, do shard correctly through the `_t_neighbor_pad` halo.

So the row model predicts factor=8 at ~360 ms against a **measured 0.898 s**. The ~540 ms gap is
sharding *overhead*, not replicated work. Candidates, in the order worth profiling:

1. **Halo CCL per conv.** 7 stages x 3 branches x 6 convs = 126 convs, each taking a
   `neighbor_pad_persistent_buffer` round trip.
2. **`_set_tpad_tail` machinery.** Called per stage and per branch; each call is a masked pass over
   the full local tensor.
3. **Lost snake fusion.** The fusion declines when `parallel_config.factor > 1`, so the sharded path
   silently gives back the week's 1.181x. goal.md already flags that this gate must be lifted before
   multi-chip and fusion can be measured together.
4. **A larger floor.** Sharding *adds* ops (CCLs, layout conversions, tpad masks), and the floor is
   per-op, so the 260 ms grows rather than shrinks.

## Scripts added

| script | does |
|---|---|
| `t_sweep.py` | the T-sweep and the affine fit; the floor/data split |
| `occupancy.py` | re-reads a profiler CSV with a denominator (span, gaps, dispatch), instead of summing FW |
| `row_model.py` | rows per stage, ups-vs-resblock split, sharding projection vs the measured 0.898 s |
| `dw_layout_check.py` | item 3 step 2's standalone weight-layout check -- CPU only, no build |

## Item 3 is smaller than scoped

`dw_layout_check.py` models both halves of the item 3 contract (the weight prep mapping, and the
"input column becomes c/k" activation index) against `F.conv1d(groups=C)` and matches **bit-exact in
float64** for k=1 (must not regress), k=2/3/4, C=8/16/32/224, kw=3/7, and with broadcast repeats 2/4.

The useful finding: `conv_depthwise_weight_bcast_helper` indexes axis 0 by
`original_weight_shape[0]`, which is `out_channels` -- it never assumes `out_channels == groups`. So
**goal.md step 2 needs no mapping change**, only the output shape derivation `(K*C, C) -> (K*C, k*C)`.
That removes the riskiest of the four edits.

`k*C` also stays tile-clean at every shape the decode uses (C=8 k=2 -> 16 pads to 32; C=8 k=4 -> 32;
C=32 k=2 -> 64; C=224 k=2 -> 448).

## Build trap for whoever does item 3

`build_Release` in the `audio-kernels` worktree is a **symlink to the main checkout's** build dir, and
that CMake cache has `CMAKE_HOME_DIRECTORY=/data/rshirvani/tt-metal`. So `ninja -C build_Release ttnn`
run from a worktree compiles **main's** sources, not the worktree's -- C++ edits in a worktree are
silently not built, which is the same class of failure as goal.md's "ninja alone is not enough". Either
make the C++ edits in the main checkout, or configure a build dir against the worktree source.
