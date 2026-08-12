# Goal: MiniMax-H3 audio decode — 0.93 s to 60 ms

**Target:** decode 207 latents (5.17 s of audio, batch 2) in **<= 60 ms** at **>= 49.45 dB PSNR vs
the CPU reference**. Today: **0.9304 s**. That is ~15x still to find.

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
| gate | 24 passed, 1 failed — the failure is inherited, see below |

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

## Item 1 — Settle host-bound vs kernel-bound  (BLOCKING, ~half a day)

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

---

## Item 2 — 32 chips  (the only 10x-shaped lever)

Every timing and PSNR test in `test_audio_minimax_h3.py` is `SINGLE_DEVICE = (1, 1)`. **There is no
8-chip configuration in that file** — if someone believes there is, reconcile that first.

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
   `k*c + j` needs group `c`'s tap set `j`. **Write a standalone check for this before touching
   anything downstream** — everything downstream reads this layout.
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

`trace (1.00x, but see item 1)` · conv1d L1_FULL (1 of 42 shapes fit) · operand splitting · algebraic
band fusion alone (neutral) · L1-sharded intermediates · conv3d UnpackToDestFp32 · act_block_h ·
merging the paired convs (2C conv is 0.52x the pair at C=8, but the duplicate and channel-halves
reduce cost it all back) · removing the 252 per-band concats (nothing in ttnn adds rows without
copying; `ttnn.slice_write` is not exposed) · grouped channel multiplier **as-is** (2.94x but
1.4e-03 — that is item 3, once the depthwise path accepts it).

**The rule these establish:** removing a full-tensor pass wins; trading one for another does not.
The snake fusion won 1.18x because it deleted a pass outright. Every conv-merging variant just moved
the cost somewhere else.
