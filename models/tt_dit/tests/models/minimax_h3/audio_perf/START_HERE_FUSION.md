# Fusion implementation — start here, build only

For a fresh session. **Do not re-derive anything in this file.** The design is settled and the maths is
verified; what is missing is build-and-verify cycles. The previous session produced ~1.16x against a
5-100x goal because it kept converting implementation into analysis. Don't repeat that.

## The framing to work from

From `plan-audio-encoder.md` (the original meeting): *"Fuse multiple ops to keep intermediate values in
SRAM and eliminate redundant memory traffic"*, and crucially *"tiling strategy may allow full fusion
without requiring entire tensor to be resident."*

Tiling is not a workaround for L1 being small -- **it is the thing that makes full fusion possible.**
Scope the resident chunk as widely as L1 allows and run the whole chain on it, rather than fusing one
band and stopping. 53->4 ops per band is the timid version; the chain the doc points at is
upsample + resblock branches, where the arithmetic is nearer 500->4.

## Facts already established -- trust these, do not re-measure

| fact | value | source |
|---|---|---|
| per-op floor | **180.3 us fp32**, 127.6 us bf16, flat to ~16k rows | `op_floor.py` |
| decode op count | 6955 -> 1254 ms of pure floor vs ~1.1 s actual | same |
| 60 ms budget | **~332 ops** fp32, ~470 bf16 | same |
| Concat | 469 calls, 285.3 ms, **20.4 % -- largest single line item** | PROFILE_2026_08_06.txt |
| all convolution | 292 ms / 21 % | same |
| data movement total | ~900 ms of ~1400 ms | same |
| conv invocations per band | **6.85** for 2 logical convs (DRAM slicing, 5-op wrapper per slice) | same + audio_ops.py:375 |
| arithmetic cost | free -- `sin` cheaper than `add` | `compute_intensity.py` |
| row effect | same elements cost **11.6x** more at C=8 than C=224 | same |
| current decode | 1.107 s fp32, PSNR 49.45 dB vs CPU | `cpu_vs_device.py` |

## Dead ends -- do not retry, all measured

trace (1.00x) · conv1d L1_FULL (1 of 42 shapes fit) · operand splitting · algebraic band fusion
(**-2.4 %**) · L1-sharded intermediates · `_zero_pad_t` concat->pad (2.2x on the op, **0 % end to
end**) · conv3d UnpackToDestFp32 · 32-chip hypothesis · act_block_h · grouped-conv branch batching
(**0.94-1.11x**, and lossy at C>=64).

They fail for one reason: they optimise ops that are already at the floor, or merge ops far above it.
Only removing large numbers of floor-bound ops moves the decode.

## Verified maths -- transcribe, don't re-derive

`band_stencil_cpu.py`, **maxdiff 3.331e-16** against the literal band in float64. With `xp` the
once-replicate-padded input:

    u[m], m even -> sum_j h[K-1-2j] * xp[m/2 + j]        odd-indexed taps, reversed
    u[m], m odd  -> sum_j h[K-2-2j] * xp[(m+1)/2 + j]    even-indexed taps, reversed
    s            =  u + inv_beta * sin(alpha*u)^2
    y[n]         =  sum_k g[k] * s[2n + K-1-k]

The phase-to-tap-subset mapping is **the reverse of the obvious guess** -- getting it backwards costs
maxdiff 0.598. Replicate padding does **not** decompose per phase (the pad region's parity
alternates), so chunk edges take halo from real neighbours; only the outermost chunks apply the
replicate rule.

## What is already built and verified

* `apply_snake_beta` in `compute_depthwise_conv1d.cpp` -- per-channel snake in DST, **3 DST slots**
  (reuses DST_A/DST_B after the last tap), **does not pop** (params are per-block, popping destroys
  them for following tiles).
* `Conv2dCb::SNAKE_PARAMS` + `CBInfo` entry + `SNAKE_PARAMS_CB_ID` define. **0 pages unless
  `TT_CONV1D_SNAKE_PARAMS` is set**, so `post_conv2d_op_memory_checks` is untouched on the default
  path -- that check is a known hard-failure source (tt-metal #35207) and this sidesteps it.
* Built and regression-tested: **17/17, `conv1d=8.082e-08 mac=8.082e-08`**, bit-identical.

## Status after the mcast session (2026-08-11) -- read this before the section below

The hang is **fixed** and the fusion **runs on every channel**. Remaining gap: rel_rmse **4.4e-04**
against the float64 golden, versus the ~1e-07 bar.

What was done, all on `rouzbeh/audio-decode-exact-fp32`:

1. **Mcast the parameters** (option 1 below, as recommended). Sender fills the CB, waits on the
   weights sender semaphore, mcasts both tiles, multicasts VALID; receiver reserves 2 pages, signals,
   waits, pushes. Both sides do it once, ahead of their block loops, so it stays paired with the
   weights handshakes. The receiver also needed `SNAKE_PARAMS` + the CB id in **`writer_defines`** --
   the sender reads a *separate* map (`writer_mcast_sender_defines`), which is why it had nothing.
2. **`reconfig_data_format_srca` before each parameter `copy_tile`.** This was the real bug behind the
   wrong numbers, and it is worth remembering generally: `copy_tile_to_dst_init_short` is the *short*
   init -- it re-inits the datacopy MOP but does **not** reconfigure SrcA's data format. The fp32
   parameter tile was being unpacked with whatever format SrcA last held, so each 4-byte datum was
   read as two 2-byte ones: high half into the odd channel column, zero low half into the even one.

   Signature to recognise it again: with `alpha = inv_beta = 1.0`, **odd columns 100% correct, even
   columns exactly untouched**. Proof it is a 16-bit read and nothing else: set the parameters to
   `0x3F803F80`, whose two 16-bit halves are both bf16 1.0, and the even columns go to 100%.

### The remaining 4.4e-04 -- characterised, do not re-measure

Recovering the parameters the kernel actually used, per channel column (`snake_diag9.py`):

| quantity | finding |
|---|---|
| alpha | good to ~1e-5; the recovered ratio does **not** drift with x |
| inv_beta | a median **9.3e-04 low**, and *always* low -- never high |
| effective precision | one-sided ~2^-10 ⇒ **truncation toward zero at ~10 mantissa bits** |

That is TF32 through SrcA, i.e. the parameters are not taking the unpack-to-dest fp32 path. Note
round-to-*nearest* at 10 bits does **not** reproduce it (a mantissa sweep bottoms out at 4.4e-04 from
12 bits up and never explains the floor) -- the one-sidedness is the tell, and it is why a rounding
model looks like it exonerates precision when it does not.

Ruled out, measured, do not retry: parameter bf16 rounding (every bf16 golden is *worse* than the
exact one); the SFPU sine approximation (`math_approx_mode=False` gives bit-identical output);
addressing (`weight_matrix_height` is the widened 288 ⇒ 9 tiles ⇒ alpha at tile 7, inv_beta at 8, and
no "weights not properly prepared" warning appears, so the widened rows survive).

Next step is to get these two copies onto a genuine fp32 path -- `UnpackToDestEn` is `true` on
Blackhole, so the mechanism exists; what is missing is that this particular copy is not using it.

### Two traps that cost hours this session

* **`snake_fused_verify.py` owns `TT_CONV1D_SNAKE_PARAMS`.** It runs the plain conv *first* to capture
  the prepared weight and only then sets the var. Exporting it around the whole process makes the
  baseline conv read two tile-rows past an un-widened weight matrix, so it returns `inf` and every
  number downstream is meaningless. Use `run_snake_verify.sh`, which gets this right.
* **A wedged card looks exactly like a kernel hang.** `c09u14` was left wedged and every program --
  including a bare `ttnn.add` -- hung after the device opened. Stage 0 of `run_snake_verify.sh` is a
  bare eltwise add for precisely this reason. Recover with `tt-smi -r 0`.

## Original blocker (now resolved): the mcast receivers never get the parameters

The snake now reaches the compute kernel on all four accumulate paths, and the op **hangs**.

`compute_defines` reach every core, so every core waits on `SNAKE_PARAMS`. But the fetch was added to
`reader_writer_tiled_out_1d_mcast_sender_...`, and under HEIGHT_SHARDED only one core runs that; the
rest run `..._mcast_receiver_...` and never fill the CB. Every receiver blocks on `wait_front(2)`.

The receiver **has no `s_weight` TensorAccessor** -- it only receives weights over mcast -- so it
cannot simply read the two tiles itself. Two ways out:

1. **Mcast the parameters** from the sender, mirroring the weights handshake that already exists
   (semaphores are already set up for weights on both sides). Kernel-only, and correct by
   construction since the parameters are identical on every core. **Preferred.**
2. Give the receiver a weight `TensorAccessor` and address so each core reads its own copy. Needs a
   new runtime arg from the program factory, so a host change and a rebuild.

Take option 1. Fill the CB on the sender as now, then mcast that CB region to the receiver grid and
have receivers wait on the semaphore before their own `push_back(2)` -- the weights path in the same
two kernels is the template to copy.

## Next three steps to make the snake fusion live

1. Fill the CB once in the weights reader, **outside** the block loop.
2. Host: alpha and `inv_beta = 1/(beta+eps)` as tiles, per-channel value replicated down all 32 rows.
3. Check against `band_stencil_cpu.py`'s float64 golden, target ~1e-07 (GELU got 7.6e-08 through the
   scalar seam).

Then widen the resident chunk from the band to the resblock chain, which is where the order of
magnitude is.

## Build procedure -- this bites

**Host C++ must be built from the main checkout.** The worktree's `build_Release` is a symlink to
`/data/rshirvani/tt-metal`'s, whose cmake cache is configured against the main checkout, so
`build_metal.sh` from the worktree refuses outright -- and a build that did run would compile the
wrong sources. Copy the files across, build there, keep backups.

Device kernels are JIT-compiled from `TT_METAL_HOME` and **do** take effect from the worktree. That
asymmetry is why kernel changes were verifiable in place all session and host changes were not.

**Cost:** a header change (e.g. `conv2d_op_program_factory_common.hpp`) plus unity builds invalidates
most of ttnn -- **~2 h**. A .cpp-only change is minutes. Prefer .cpp-only edits while iterating.

After any kernel edit, immediately re-run
`pytest test_audio_vae_minimax_h3.py -k "depthwise_mac or channel_padding"` (~20 s). A template
defined outside an `#ifdef` still needs its callees declared -- "gated off" does not mean "cannot
break the build", and that check catches it in seconds.

## Gates

`pytest models/tt_dit/tests/models/minimax_h3/test_audio_vae_minimax_h3.py` -- 17 must pass.
`cpu_vs_device.py` for timing and PSNR. `full_table.py` for the whole comparison from saved WAVs.
Commit and push to `rouzbeh/audio-decode-exact-fp32`; the user has asked for this explicitly, which
supersedes the "don't commit" line in plan-audio-encoder.md.
