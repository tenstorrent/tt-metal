# Goal: MiniMax-H3 audio decode — 0.93 s to sub 200 ms

**Target:** decode 207 latents (5.17 s of audio, batch 2) in **<= 200 ms** at **>= 49.45 dB PSNR vs
the CPU reference**. Today: **0.9304 s**. That is ~15x still to find.

Relaxed by the user on 2026-08-12: **222 ms is good, 300 ms is acceptable.** Item 1 measured a ~260 ms
T-independent floor, so this relaxation is what makes any route feasible at all — see below.

**Do item 1 before anything else.** It decides whether items 2 and 3 are the right work at all.

---

## Where things stand

| | |
|---|---|
| branch | `rouzbeh/minimax-audio` (checked out at `/data/rshirvani/tt-metal`) |
| decode, default flags | **1.0982 s** |
| decode, both fusion flags | **0.9304 s** (1.181x) |
| pre-week baseline `bd12ad2aeb2` | 1.2506 s — so the week bought 1.344x total |
| PSNR vs CPU | **49.45 dB** mean (47.87 / 47.82 / 52.83 / 49.28 across 4 clips) |
| gate | ~~24 passed, 1 failed — the failure is inherited~~ → **25 passed, 0 failed** as of 2026-08-12; the T-parallel failure was not inherited, it was `conv_pre` under sharding, now fixed |

Nothing is on by default. The 0.9304 s needs:

    MINIMAX_H3_AUDIO_FUSE_BAND=1 MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV=1

The one gate failure is `test_audio_decode_t_parallel[blackhole-mesh4x8]`, a 300 s timeout.
**Confirmed pre-existing** — it fails identically on pristine `cglagovich/minimax-h3` with none of
this work applied.

---

## Environment (not guessable — copy verbatim)

Run device jobs **only on `bh-glx-110-a09u02`**. The `bh-glx-110-c09*` boxes belong to other
sessions; `c09u14` is wedged and needs `tt-smi -r 0`. `/data/rshirvani` is shared across hosts.

    cd /data/rshirvani/tt-metal
    source /data/rshirvani/tt-metal/python_env/bin/activate
    export TT_METAL_HOME=/data/rshirvani/tt-metal
    export PYTHONPATH=/data/rshirvani/audio_ref_pkgs:$TT_METAL_HOME
    export TMPDIR=$HOME/ttcc && mkdir -p $TMPDIR     # /tmp is shared; parallel jobs clobber the assembler
    export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers

`audio_ref_pkgs` is a diffusers-main overlay; without it the reference model will not import.

**Never export `TT_CONV1D_SNAKE_PARAMS` by hand.** `snake_fused_verify.py` owns it and sets it only
for its second run; exporting it makes the baseline conv read past an un-widened weight and return
`inf`.

### Building host C++ (two traps)

Device kernels are JIT-compiled from `TT_METAL_HOME` and need no build. Host C++ does:

    ninja -C build_Release ttnn \
      && cmake --install build_Release --component tt_pybinds \
      && cmake --install build_Release --component tar

**`ninja` alone is not enough.** It leaves `ttnn/ttnn/_ttnn.so` and `build_Release/lib/` stale, so
you silently run the old binary and measure nothing. Both `--component` installs are required.
A `.cpp`-only change is ~4 min; a header change ~10 min (ccache absorbs most of it).

**Third trap, same class: a worktree's `build_Release` is a symlink to the main checkout's.** Its
CMake cache has `CMAKE_HOME_DIRECTORY=/data/rshirvani/tt-metal`, so `ninja -C build_Release ttnn` run
from a worktree compiles **main's** sources — C++ edits made in a worktree are silently not built, and
you measure the unmodified binary. Either make C++ edits in the main checkout, or configure a separate
build dir against the worktree source (2.6 GB, and ccache is warm).

---

## Scripts

| script (in `models/tt_dit/tests/models/minimax_h3/audio_perf/`) | produces |
|---|---|
| **`decode_bench.py`** | timing medians — `BENCH_N=5 python .../decode_bench.py <label>` |
| **`cpu_vs_device.py`** | **PSNR vs CPU + .wav files** into `/data/rshirvani/audio_compare/clips/` |
| `run_snake_verify.sh` | one-shot: device health -> regression -> snake golden -> full gate |
| `snake_fused_verify.py` | fused snake vs float64 golden (7.649e-08) |
| `band_grouped_multiplier.py` | acceptance test for item 3 |
| `band_{concat,wide_conv,reduce}_cost.py` | evidence for the rejected options |

Gate: `pytest models/tt_dit/tests/models/minimax_h3/test_audio_minimax_h3.py -q`
(renamed from `test_audio_vae_minimax_h3.py` by h3's suite consolidation).

Do **not** quote timings from `decode_accuracy.py` — it times one call per process, which is where
~1% spread came from. Use `decode_bench.py`.

Stale WAVs: `*_3_device_prefix.wav` are from 08-07 and a different config. Delete before any
listening test.

---

## Item 1 — Settle host-bound vs kernel-bound  ✅ DONE 2026-08-12

**Answer: kernel-bound.** Device ~0.90 s of the 0.9304 s. Items 2 and 3 are the right work.
Full write-up and scripts: `audio_perf/ITEM1_RESULT.md`.

Settled **without** the profiler, because the profiler is what produced the contradiction — two
totals are on record for this stage (224 ms and 1401 ms) and neither is device time. Instead:

* **Trace.** `untraced 0.9450 s | traced 0.9081 s -> 1.04x`, and `dec_in_proj 0.0020 s`. Removing
  *all* host dispatch buys 37 ms of 945, so the ~70%-host-bound docstring is wrong and the
  `trace 1.00x` dead-end entry is right.
* **T-sweep** (`t_sweep.py`, op count is T-independent): the decode is **~260 ms fixed + ~670 ms
  data-proportional**.
* **`op_pipeline.py`**: chained 141.9 / independent 125.7 / per-op-sync 138.8 us/op — all equal, so
  that microbenchmark is host-*issue*-bound and never measured a device floor. Real per-op device
  cost is ~37 us. **The "6955 x 180 us = 1254 ms floor" is retired.**

**The consequence that changes the plan:** the ~260 ms floor is T-independent, so sharding does not
divide it and trace does not remove it. Sharding-only routes cap at `260 + 670/factor` — ~298 ms at
32 chips, ~344 ms at 8, and that ignores CCL cost. Item 2 is a **~3x lever with a hard floor at the
target**, not the 10x lever described below. Reaching 300 ms with margin needs the floor cut too,
which means deleting ops.

**Two corrections to this file, both load-bearing:**
1. Item 2's "there is no 8-chip configuration in that file" is **wrong** — `FACTORS = [(1,1), (4,0),
   (8,1)]` at `test_audio_minimax_h3.py:729`, with t_factor=8 recorded at **0.898 s** (1.04x) and
   PSNR −6.3 dB. This file asks for that to be reconciled first; it is now reconciled, and the 10x
   premise was already contradicted in-tree.
2. `row_model.py` sizes the sharding problem: the unsharded `ups` are only **2.7% of rows** (the
   resblocks, 97.3%, shard correctly via the halo). So the 898 ms is **~540 ms of sharding overhead**,
   not replicated work — profile the 126 per-conv halo CCLs, `_set_tpad_tail`, and the disabled snake
   fusion, in that order.

<details><summary>original item 1 brief</summary>

**Why:** the whole week assumed kernel-bound. The evidence is contradictory and nobody has closed it.

* `PROFILE_2026_08_06.txt` device times sum to **~1.3-1.4 s** against ~1.1 s end-to-end. Device time
  cannot exceed wall time, so one of those is measured wrong.
* `op_floor.py`'s **180 us/op was measured with a synchronize between ops**, so it folds host
  round-trip into every sample. "6955 ops x 180 us = 1254 ms of floor" is therefore **not** a
  device-time floor and must stop being quoted as one.
* `decoder_minimax_h3_audio.py`'s docstring says the vocoder is ~70% host-bound and tracing is "its
  dominant lever". The dead-ends list says trace measured 1.00x. Both cannot be true.
* `op_pipeline.py` was written to answer exactly this. No recorded result was found.

**Do:**

    TT_METAL_DEVICE_PROFILER=1 pytest models/tt_dit/tests/models/minimax_h3/test_audio_minimax_h3.py \
        -q -k "decode and single_device"
    pytest models/tt_dit/tests/models/minimax_h3/test_audio_minimax_h3.py -q -k traced   # already exists, line 928

Sum device op time from the profiler CSV; compare against `decode_bench.py`'s 0.9304 s median.

**Acceptance / branch:**
* device ~= 0.9 s -> **kernel-bound**. Items 2 and 3 are correct work. Proceed.
* device ~= 0.2 s -> **host-bound**. Stop all kernel work. Trace / multi-CQ / dispatch is the whole
  game, and this week's layer was the wrong one.

</details>

---

## Item 2 — 32 chips  (~~the only 10x-shaped lever~~ a ~3x lever, floored at ~260 ms — see item 1)

> ## ✅ TARGET MET 2026-08-12 — **283 ms at 49.45 dB**, from 0.9304 s (3.29x)
>
> `mesh 4x8, t_factor=8 axis=1, traced` scores **49.45 dB mean vs the CPU reference** (47.87 / 47.82 /
> 52.83 / 49.28) — *identical* to the single-device baseline — at **283.1 ms**. Reproduce with:
>
>     CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 \
>       python models/tt_dit/tests/models/minimax_h3/audio_perf/cpu_vs_device.py
>
> Two things got it there, neither of which is item 2 as written below. **Trace**, worth 3.06x on a
> sharded mesh against 1.04x on one chip. And a one-conv correctness fix: `conv_pre` returned
> uninitialized memory under T-sharding while every other conv was bit-exact, which is why every
> sharded decode was wrong. Item 3 was never needed.
>
> **RESULT 2026-08-12 — read `audio_perf/ITEM2_RESULT.md` before touching this section.**
> The lever is **trace, not chip count**. Traced on the mesh: factor 8 = **0.2800 s** (3.14x), factor 4
> = 0.4690 s (2.00x), factor 32 projects to 191–281 ms. Untraced, 32 chips project to 822 ms — a
> plateau — because `factor=1` on 32 chips already costs 1.4409 s against 1.0980 s on one chip: +343 ms
> of pure 32-wide dispatch, of which trace removes 331 ms.
>
> **But T-sharding returns the wrong audio at every factor** — factor 4 is −10.1 dB and factor 8 is
> −11.0 dB against the single-device path, so both traced timings above are timings of a wrong
> computation. `KNOWN_BROKEN = {(8,1)}` should include `(4,0)`. Localized in ITEM2_RESULT.md: shards 1+
> saturate at the closing tanh/clamp while shard 0 does not, error spread through the interior (not
> boundaries), correlation 0.04 at lag 0 — so not a shift, trim, or halo-width bug. Correctness is now
> the critical path; performance is not.
>
> Also: **the mesh4x8 test does not hang.** It takes ~12 min and CI kills it at 300 s, so
> `reset_cluster.sh` and the fabric-flakiness branch below are not needed.
>
> **Corrected 2026-08-12.** The claim below is false and the reconciliation it asks for is done.
> `test_audio_minimax_h3.py:729` has `FACTORS = [(1, 1), (4, 0), (8, 1)]`, `KNOWN_BROKEN = {(8, 1)}`,
> and the comment above it records **t_factor=8 at 0.898 s (1.04x) with PSNR −6.3 dB**. So multi-chip
> configurations exist and 8-way T-sharding is already measured at no speedup.
>
> `row_model.py` says why that is not a replicated-work problem: the unsharded `ups` hold only **2.7%
> of rows**, the halo-sharded resblocks hold 97.3%. The row model predicts ~360 ms at factor=8 against
> the measured 898 ms, so **~540 ms is sharding overhead**. Profile in this order: (1) the 126
> per-conv halo CCLs (7 stages x 3 branches x 6 convs), (2) `_set_tpad_tail`, called per stage *and*
> per branch as a masked pass over the full local tensor, (3) the snake fusion, which declines when
> `factor > 1` and silently gives back the week's 1.181x.
>
> Ceiling with item 1's floor: `260 + 670/32` ≈ **298 ms** at 32 chips even if sharding were free. That
> clears the relaxed 300 ms bar but leaves no margin, so pair this with op-count reduction.

Every timing and PSNR test in `test_audio_minimax_h3.py` is `SINGLE_DEVICE = (1, 1)`. ~~**There is no
8-chip configuration in that file**~~ — if someone believes there is, reconcile that first.

T-sharding is already implemented: `_t_neighbor_pad`, the halo exchange, `AudioTParallelConfig`.
Its one test (`mesh4x8`, line 719) times out at 300 s, pre-existing.

**The first question is not "can we use 32 chips" — it is "why does the existing T-parallel path
hang".** Fabric flakiness, or unfinished code? Reproduce with:

    pytest models/tt_dit/tests/models/minimax_h3/test_audio_minimax_h3.py -q \
        -k test_audio_decode_t_parallel --tb=line

If fabric: `reset_cluster.sh` (coordinated all-shelf reset; a per-shelf reset leaves the cross-shelf
fabric half-trained — see its header).

**Acceptance:** `t_parallel` green on mesh4x8, then a `decode_bench.py` median at 32 chips.
Note the snake fusion **declines** when `parallel_config.factor > 1`, so it will silently switch off
under T-sharding. That gate has to be lifted before multi-chip and fusion can be measured together.

---

## Item 3 — Depthwise channel multiplier  (ready, but sequence it last)

Let the 1D depthwise path accept `out_channels == k * in_channels` with `groups == in_channels`.
Worth **2.94x on the conv pair at C=8**, ~1.2-1.5x end to end. Four `.cpp` edits, no header change.

1. `conv2d_utils.cpp:542` — `is_depthwise_conv = groups == input_channels && groups ==
   output_channels`. Relax to `output_channels % groups == 0 && groups == input_channels`, carry
   `k = output_channels / groups`. **Env-gate this first** — it routes every grouped conv in the
   repo, and widening it silently re-routes callers the depthwise factory has never seen.
2. `prepare_conv2d_weights.cpp` — the weight matrix goes `(K*C, C)` -> `(K*C, k*C)`; output column
   `k*c + j` needs group `c`'s tap set `j`. ~~**Write a standalone check for this before touching
   anything downstream**~~ — **done 2026-08-12: `audio_perf/dw_layout_check.py`**, bit-exact in
   float64 vs `F.conv1d(groups=C)` for k=1 (no regression), k=2/3/4, C=8/16/32/224, kw=3/7, and
   broadcast repeats 2/4.
   **This edit is smaller than scoped:** `conv_depthwise_weight_bcast_helper` already indexes axis 0
   by `original_weight_shape[0]` = `out_channels` and never assumes `out_channels == groups`, so the
   mapping needs no change — only the output shape derivation. `k*C` stays tile-clean at every shape
   the decode uses.
3. `conv2d_op_program_factory_common.cpp` — mostly verification; CB widths already key off
   `per_core_out_matrix_width_ntiles`, and SNAKE_PARAMS is already `2 *` that.
4. `compute_depthwise_conv1d.cpp` — tile indexing. `block_w` is already threaded into the flat loops;
   what changes is that the **input** column becomes `c / k` while the output column stays `c`.

**Acceptance:** `band_grouped_multiplier.py` keeps its 2.94x **and** reaches ~5e-08 (it measures
1.4e-03 today). Then `decode_bench.py` against 0.9304 s, and the gate at 24 passed.

The snake rides it for free — the parameter CB is per output column and sized from the output width.

---

## Facts — do not re-derive

* **Row effect.** A row costs ~4.2 ns almost regardless of width; the same elements cost 11.6x more
  at C=8 than C=224. Long-T narrow-C tail stages dominate the decode.
* **`copy_tile_to_dst_init_short` does not reconfigure SrcA's data format**, despite taking a CB id.
  The operand is unpacked with whatever format SrcA last held. Symptom: an fp32 tile read as two
  16-bit datums — correct value in odd channel columns, exactly zero in even ones.
* **Any fp32 CB read with `copy_tile` must be in `unpack_to_dest_mode`**, or it arrives TF32: 10
  mantissa bits, **truncated toward zero, not rounded**. A round-to-nearest error model will not
  reproduce it and will look like it exonerates precision.
* **C=512 is excluded from the snake fusion for L1, not accuracy** — 2 tiles per channel-tile is
  32 tiles / 128 KB and the DRAM auto-slice cannot fit it. `MINIMAX_H3_AUDIO_SNAKE_CONV_MAX_C=256`.
* **18 `DRAM Auto slice` criticals per decode are pre-existing** — identical count with all fusion
  off. conv1d's slice search probing, caught upstream. Not a regression, but 18 convs are taking a
  fallback path.
* **A wedged card is indistinguishable from a kernel hang.** Stage 0 of `run_snake_verify.sh` is a
  bare `ttnn.add` for exactly this reason. This cost ~2 h once.

## Do not retry — all measured

`trace (1.00x` **on a single device only — 1.30x unsharded and 3.14x sharded on a 32-chip mesh, see
`audio_perf/ITEM2_RESULT.md`; this entry sent the whole week down the kernel path**`)` · conv1d
L1_FULL (1 of 42 shapes fit) · operand splitting · algebraic
band fusion alone (neutral) · L1-sharded intermediates · conv3d UnpackToDestFp32 · act_block_h ·
merging the paired convs (2C conv is 0.52x the pair at C=8, but the duplicate and channel-halves
reduce cost it all back) · removing the 252 per-band concats (nothing in ttnn adds rows without
copying; `ttnn.slice_write` is not exposed) · grouped channel multiplier **as-is** (2.94x but
1.4e-03 — that is item 3, once the depthwise path accepts it).

**The rule these establish:** removing a full-tensor pass wins; trading one for another does not.
The snake fusion won 1.18x because it deleted a pass outright. Every conv-merging variant just moved
the cost somewhere else.
