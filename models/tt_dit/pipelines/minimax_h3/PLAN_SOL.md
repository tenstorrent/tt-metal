# Porting Sol-H3 to the 4x8 Blackhole Galaxy

Sol-H3 is NVIDIA Research's MiniMax-H3 inference engine (project page
`nvlabs.github.io/Sana/Sol-Engine/Sol-H3/`, source `NVlabs/Sana@sol-engine`
`models/minimax_h3/Sol-H3/`, Apache-2.0). It reports **6.612 s for a 15 s / 1344x768 / 24 fps
generation with stereo audio on 8x B300**. This file is the plan for what of it transfers here, and
the honest accounting of what does not.

Companion docs: `OPTIMIZATIONS.md` (the measured lever list this plan feeds into),
`models/MiniMaxH3.md`, `models/transformers/minimax_h3/VSA_README.md`, `VSA_STREAM_DESIGN.md`.

## What the headline number is and is not

Sol-H3's own README states the comparison: the SGLang and Diffusers baselines run **49 DiT
forwards**, Sol-H3 runs **4**. That step reduction is ~12x of the reported ~11x. The engine's own
contribution on top is roughly 2.1x on attention (Sol-Attn) and 12x on VAE decode (tiling +
batching).

**We already run the 4-forward FastH3 adapter.** So the comparison that matters is at matched
duration, shape and adapter:

| | Sol-H3, 8x B300 | here, 4x8 BH Galaxy |
|---|---|---|
| 15 s / 1344x768 / 24 fps + audio | **6.612 s** | **18.9 s** at the current serving point |

(19.9 s is job 886's figure, which this plan was first written against. 18.9 s is job 247, measured
here; **18.5 s** is the same configuration with the S1 gather narrowing.)

A 3x hardware-and-engine gap, not an 11x algorithm gap. Nothing in this plan closes 3x. S1 is now
measured at 0.41 s; what is left is a few hundred ms of cleanup plus a fidelity result we currently
cannot measure at all.

## The six Sol techniques, against this tree

| Sol technique | State here |
|---|---|
| Fused residual+RMSNorm+modulation, QKNorm+partial RoPE, SwiGLU | **Done.** `DitFusedDistributedRmsnorm` takes the modulation as `dynamic_weight`/`dynamic_bias`; the gated residual is folded into `to_out`'s matmul epilogue; `ff1` packs gate+up for a fused SwiGLU; `forward_fused_addcmul`. Sol's own `fusions.py` header rates this whole line at ~3% and "worth taking because it is lossless, not because it is large." |
| AdaLN precomputation | **Done.** `adaln_precompute.py` + `adaln_cache_minimax_h3.py`. Our residual cost is O7's six `Embeddings` gathers per block (1.06 s). Nothing to import. |
| Parallel VAE decoding (tile · batch · compile) | **Done.** 196 work units over waves of 32 (`vae_minimax_h3.py`). Sol's "batch" step is our wave batching; the leftover is O10 (repack the padded last wave). |
| Exact-KV sink + dense prefix query rows | **Done, under another name.** Sol's `sink_start`/`sink_tokens` plus dense recompute of prefix query rows is our `exempt_prefix` / `exempt_ids` / `dense_row_mask`. Same mechanism, same 64-token granularity. |
| Sparse setup fusion (1.206 -> 0.285 ms) | **Not done**, and the same diagnosis as O3's remainder. See S3 below. |
| INT8 QKV / FP8 output transport on the Ulysses all-to-all | **Landed as `bfloat8_b` K/V, measured -3.7% of the block.** See S1. |

Five of six already exist, several measured further than Sol's. The port that matters is narrow.

## S1 - Compress the collectives (landed, measured: -3.7% of the block)

`OPTIMIZATIONS.md` O6 establishes that the VSA K/V all-gather is link-bound: a device receives
7 x 51.8 MB = 363 MB per gather and the best of every swept configuration moves it at 87-93 GB/s,
~90% of 2 x 50 GB/s. O6's conclusion was "the serial time cannot be tuned away; only overlap is
left" - and then overlap turned out to be blocked by sub-device ownership (`TT_FATAL: Sub device id
0 currently in use by cq 1`).

Sol's answer is not overlap. It is **bytes**: INT8 QKV plus FP8 output transport at 51.6% of the
BF16 payload. The TT analog is `bfloat8_b` at 56% of bf16, and it needs no new infrastructure -
`CCLManager.get_ag_ping_pong_buffer` already keys its persistent buffer on `tensor.get_dtype()`,
and `vsa_sdpa` already validates `k`/`v` as `BFLOAT16 || BFLOAT8_B` with `vsa_sdpa_stream_program_factory.cpp`
deriving per-tensor `tile_size` from the data format.

**But nothing in the tree exercised that permissive dtype gate.** That is why this step is a knob
plus two tests, not a one-line dtype change.

### What landed

* `MiniMaxH3VSAConfig.gather_dtype: str = "bfloat16"`, accepting `"bfloat8_b"`.
* `MiniMaxH3Attention._narrow_for_gather`: typecast K/V immediately before the SP all-gather. The
  coarse stage runs *before* the gather and scores the local bf16 K, so **selection is bit-identical**
  and only the values the fine stage attends over lose precision. Q stays bf16 (`vsa_sdpa` requires it).
* `tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa.py::test_vsa_sdpa_bfp8_kv` - op-level, both
  kernels, reference built from the *dequantized* K/V so the PCC measures kernel error and not input
  quantization.
* `test_vsa_attention_production_shape` parametrized on `gather_dtype`, both arms held to the same
  0.985 floor (`_GATHER_DTYPE_PCC`) once the measurement showed they do not separate.
* `VSA_GATHER_DTYPE` env knob on the block-perf, e2e-perf and pipeline paths for A/B without edits.

### Measured

Jobs 223-226 on `c46d3176990`, 4x8 BH Galaxy. Op level (job 223): 4/4 pass, PCC >= 0.999 against the
dequantized reference on both kernels -- the permissive dtype gate is real and the reader honours the
bfp8 tile size. Fidelity (job 224, `39f_1280x768`, zero gate): **99.3249 % against a 99.3308 %
control**, 0.006 pp apart, so the narrowing does not separate from the selection-noise floor.

Block profile, jobs 225/226, `test_minimax_h3_vsa_block_perf -k 15s_768p`, Tracy signpost window,
32-device merge:

| row | bf16 control | bfp8_b | delta |
|---|---|---|---|
| K/V `AllGatherAsync` (the two large ones) | 4,029 + 4,071 = 8,100 us | 2,854 + 2,853 = 5,707 us | **-2,393 us (-29.5%)** |
| pooled coarse gathers (two) | 158 + 136 us | 159 + 137 us | unchanged, as intended |
| `Typecast` BF16 -> BFP8 (two, new) | - | 203 + 203 = 406 us | **+406 us** |
| `VsaSdpaOperation` | 17,821 us | 17,753 us | -68 us (noise) |
| `AllGatherMinimalMatmulAsyncOp` x4 | 14,588 us | 14,568 us | -20 us (noise) |
| **block device time** | **55,640 us** | **53,585 us** | **-2,055 us (-3.7%)** |

Scaled `x50 blocks x4 forwards`: **-0.41 s of an 18.9 s run (-2.2%)**. Take the lever -- it is free of
fidelity cost and costs nothing to keep -- but it is **half the predicted 0.74 s**. It does survive to
the e2e number (below), which the VSA port's block-level delta did not.

### The e2e pair

Jobs 245/247, `test_zz_yuv_timing.py -k 15s` (FastH3 LoRA + VSA + `vae_output_type="yuv420"`, 5
scheduler points / 4 forwards, 1344x768 / 362 frames), fully warm on g14blx03:

| stage | bf16 control | bfp8_b | delta |
|---|---|---|---|
| Encoder | 0.3 s | 0.3 s | - |
| **Denoise (4 fwd)** | **12.5 s** | **12.1 s** | **-0.4 s** |
| VAE decode | 4.6 s | 4.7 s | +0.1 s (noise) |
| Audio decode | 1.4 s | 1.4 s | - |
| **Total (compute)** | **18.9 s** | **18.5 s** | **-0.4 s (-2.1%)** |
| per forward | 3132.6 ms | 3015.7 ms | **-116.9 ms (-3.7%)** |
| steady ms/step | 3113 | 2996 | **-117 ms (-3.8%)** |

**-3.7% per forward against the block profile's independently measured -3.7%.** The whole delta sits
in denoise; Encoder and Audio match exactly and VAE moves +0.1 s in the wrong direction, which is the
correct signature for a change that only touches attention. The control also reproduces the 18.9 s
serving figure exactly, so that baseline is now attached to a job id.

The same pair at 10 s / 243 frames (jobs 243/244) agrees: total 12.5 -> 12.2 s, steady 1947 -> 1873
ms/step (**-3.8%**), again entirely in denoise.

One pair per duration against a +-8% band is not a confirmation on its own. What makes it credible is
that four instruments -- block profile, 10 s per-step, 15 s per-forward, 15 s per-step -- land within
0.1 pp of each other.

### The 15 s decode and the g14blx03 tray

Getting that pair took four attempts, and the failures are worth recording because they are not about
the lever.

Jobs 227 (bf16) and 234 (bfp8) were both killed by the broker with `[MCP killed: chips left PCIe]` at
100.6 s and 115.4 s -- **the same 8 chips (8-15)**, the same point in the stage, immediately after
`decoder weights uploaded` and the same two
`No known best blocking for (M, K, N) = (1792, 32, 2048) / (1824, 2048, 3072) on 11x10 core grid;
using default 8x8x8` warnings. Each time the broker's bridge reset failed and escalated to a **full
power cycle**, which takes the host off the network -- so the fault costs the whole box, not just the
job.

Chips 8-15 are one row of the 4x8: one UBB tray, one power rail. `69e6a5d195b`'s own message records
the history -- *"Every chip-drop incident on blx04 lands in the VAE or audio decode and none in the
denoise... 21 of 21 prior incidents sit on one UBB tray, so the rail is marginal either way."* These
were incidents 22 and 23. The cap that commit added is already active (`get_matmul_core_grid` clamps
32-chip Blackhole to 11x10, and 11x10 is the grid in the warnings), so the shapes that trip are
*inside* it.

**Both 15 s runs succeeded after the power cycle**, on the same box, same command. So the rail is
intermittent, not a hard limit at this shape, and a 15 s e2e is obtainable -- just not reliably.
Registering those two matmul shapes in the VAE decoder blocking table (O4's work) lowers the draw and
is worth doing; on this evidence it is a mitigation, not a fix.

Naming note: the host these jobs ran on reports `g14blx03`. `g03blx04` is a stale `~/.ssh/config`
alias pointing at it, and `OPTIMIZATIONS.md` plus `69e6a5d195b` both say "blx04" for the same machine.
The sibling Galaxy, `g14blx02` (alias `g03blx02`), was unreachable for this entire session.

### Follow-ons once S1 is proven

* **Kill the typecast**, worth the measured **406 us/block (0.08 s)**. The QKV projection writes K/V
  in bf16 because QK-norm, RoPE and the coarse pooling all consume them. If the fine stage is the only
  consumer of the *gathered* copy, the narrowing can move into the gather's own output dtype
  (persistent buffer allocated bfp8_b, input bf16) if `all_gather_async` will accept a dtype change -
  check; if not, this is the op request that makes S1 strictly free.
* **The four AGMMs (O1).** Measured **14,568 us/block, 27.2% of the block** and completely untouched
  by this change - the largest remaining row after the SDPA. `all_gather_minimal_matmul_async` fuses
  the TP gather into the matmul, so its input dtype is the lever on the same 2.96 s O1 already flags
  as `SLOW` for a different reason
  (`DEV_0_DRAM_INTERLEAVED` input 0, no `program_config`). Do the dtype arm *after* the
  `program_config` arm so the two are attributable separately.
* **The pooled coarse gathers.** Already tiny (`[heads, head_dim, n_tiles]`); leave bf16.

## S2 - Sol-Attn routing and the LSE residual

The genuinely new algorithm. Reference to read: `h3_runtime/third_party/sol_attn/triton_ref/preprocess.py`
and `.../fwd.py`. `BLOCK_SIZE=64`, `HEAD_DIM=128` - identical tiling to our VSA, which is why this is
a port and not a rewrite.

### The algorithm

1. **Summaries per 64-token KV block:** `k_c` = **mean** of K, `v_c` = **sum** of V.
2. **Per-(q_block, head) threshold, computed analytically and not from real scores.** With `q_bar`
   the query-block centroid, over the population of KV blocks:

       mu    = q_bar . E[k_c]
       sigma^2 = q_bar^T Cov[k_c] q_bar
       threshold = mu + tau * sigma

   Two variants ship: `_diag_threshold_kernel` (diagonal variance, one `D`-vector of per-dim
   variance) and `_exact_fused_threshold_kernel` (full `D x D` second moment). Both fold the
   `scale * log2(e)` into the statistics so the kernel works in base 2.
3. **`tau = 1.0`, no calibration.** Sol's `sparse_attention.py` header records that per-shape
   calibration returned an empty route set on H3's shape and the configuration *silently fell back to
   dense*. Do not build a calibration pass.
4. **Routing:** block is exact if `mean_col(q @ k_c^T) > threshold` **OR** `|q_block - kv_block| <= 1`
   (local window) **OR** inside the sink range.
5. **Non-exact blocks are not dropped.** They are folded into the *same* online softmax as one pooled
   token using `k_c`/`v_c`, length-weighted in the denominator (`row_sum += p * block_len`).

### What this changes here, and the two honest problems

**Problem 1: variable list length** (largely settled -- the oracle measured the spread at +-12%, so a fixed width plus a count costs ~13% and no dynamic-count path is needed)**.** Threshold routing yields a data-dependent count per row instead
of a fixed `k = compute_topk(sparsity, n)`. `vsa_sdpa` already carries `list_len`, `block_counts`,
`dense_row_mask` and `dense_row_hint` and handles rows from 1 to 1024 listed blocks, so this is not a
wall - but dynamic counts are a *worse* fit for TT static shapes than for CUDA. **Port it as
fixed-width-plus-count**: allocate the worst-case width, write a count, let the kernel's existing
per-row length handling do the rest. `topk_large_indices` is replaced by a threshold compare plus a
prefix count; `vsa_stages_minimax_h3.py` already computes the full score matrix, so the compare is
free and the count is the new device work.

**Problem 2: our coarse output is not Sol's residual.** `MiniMaxH3VSACoarseStage.__call__` already
computes `o_c = softmax(Q_c @ K_c^T) @ V_c`, pools V, all-gathers it and broadcasts tile -> 64 tokens.
Two differences:

* Our pooled softmax includes the **selected** blocks too, so it double-counts - which is exactly why
  it needs a learned `to_gate_compress` to blend.
* Sol excludes selected blocks from the pooled branch and merges by **exact log-sum-exp**, no gate.

Converting is: mask the exact blocks out of the coarse scores (a sibling of `cand_mask`), then merge
the coarse and fine outputs by LSE instead of `ttnn.addcmul`. **That requires `vsa_sdpa` to return its
row LSE**, which it does not today - check whether the exact row sums already computed on the writer
core (`vsa_sum_service.hpp`) can be exported cheaply. That op change is the main new C++ work in this
plan.

### Step 1 result: the oracle is in, and it moves two of the estimates

`sol_attn_reference/` vendors the Triton reference (Sana ``sol-engine`` @ ``8e0db4fa562d``) as the
algorithm of record -- it cannot run here, no triton and no CUDA -- and `sol_attn_oracle.py` is the
pure-torch transcription. `test_sol_attn_oracle_minimax_h3.py` is host-only, 15 tests, 2.7 s, no
device. The load-bearing one is `tau -> -inf`: every block routes exact, the approximate branch
empties, and the result must equal dense attention. That pins the softmax shift, the base-2 scores
and the shared denominator at once.

Routing at H3's block geometry, `tau=1.0`:

| shape | thresh | blocks | per-row selected (min/mean/max) | density |
|---|---|---|---|---|
| 15s_768p | diag | 1808 | 256 / 288.8 / 326 | **15.97%** |
| 15s_768p | exact | 1808 | 261 / 289.0 / 318 | **15.99%** |
| 10s_768p | diag | 1216 | 161 / 195.1 / 228 | 16.04% |
| 10s_768p | exact | 1216 | 163 / 195.3 / 220 | 16.06% |

**These are Gaussian pooled keys and query centroids, not real activations.** They characterise the
threshold formula, not this model -- real Q/K carry RoPE and spatial structure that will move both
numbers. Re-run them against pooled tensors dumped from a generation (the `VSA_DUMP_INDICES=1`
machinery is the place to hang that) before treating any of it as an H3 property. With that said,
three things follow:

* **Sol-Attn at the validated `tau` is denser than our VSA, not sparser: ~16% against 10%.** Porting
  it as-is gives the fine stage ~60% more work. It is a fidelity-at-higher-cost trade, and anyone
  expecting a latency win from it is reading Sol's "2.1x on attention" against a *dense* baseline we
  do not run.
* **The per-row spread is narrow (+-12%).** This substantially weakens "Problem 1" below: a fixed
  width sized at the max wastes ~13% over the mean, so fixed-width-plus-count is comfortable and no
  dynamic-count device path is needed.
* **`diag` and `exact` agree to 0.02 pp.** The `D x D` second moment buys nothing at this geometry, so
  the port only needs the diagonal-variance threshold -- one `D`-vector of per-dim variance per
  (batch, head), not a matrix.

### What it is worth: measured, and it is a delivery question

`test_vsa_sdpa_sparsity_scaling` sweeps the real 15 s shard (14464 queries, 1808 blocks) and carries
its own dense reference at `k=1808`. Job 248 (full op) and job 249 (`TT_VSA_PROBE=1`, consume visits
and skip the math), `model` order:

| k / 1808 | density | full op | util | delivery floor | `topk` full op |
|---|---|---|---|---|---|
| 1808 | 100.0% | 103.73 ms | 34.86% | 18.81 ms | 104.55 ms |
| 904 | 50.0% | 55.49 ms | 32.58% | 15.59 ms | 55.32 ms |
| 452 | 25.0% | 30.03 ms | 30.10% | 11.94 ms | 30.52 ms |
| **289 (Sol, tau=1.0)** | **16.0%** | **20.98 ms** | **27.56%** | **10.68 ms** | 21.52 ms |
| **197 (VSA, sparsity 0.9)** | **10.9%** | **16.09 ms** | **24.49%** | **10.25 ms** | 16.86 ms |
| 128 | 7.1% | 12.37 ms | 20.70% | 10.01 ms | 12.99 ms |
| 64 | 3.5% | 8.98 ms | 14.26% | 9.78 ms | 8.79 ms |
| 32 | 1.8% | 7.36 ms | 8.70% | 9.66 ms | 7.29 ms |

At the sparse end the probe exceeds the full op, so the probe path is not a strict subset of the real
one -- read the floor, not the difference, there. Utilization under the probe is meaningless (the
listed math is not done) and is omitted.

**1. Sol's density costs +30% on this kernel, not the +60% a key count implies.** 16.09 -> 20.98 ms:
1.47x the blocks for 1.30x the time, because utilization climbs 24.49% -> 27.56%. Density partly
finances itself, and key counts should not be quoted as cost.

**2. There is a ~9.6 ms k-independent delivery floor, and it is the biggest item in this op.**
1.8% -> 16% density moves delivery 9.66 -> 10.68 ms: ~1 ms for 9x the keys. That is fixed cost --
index walk, stream setup, per-row -- and at production 10.9% it is **64% of the whole op**, which is
64% of 32% of the block. It is entirely independent of what the routing selects. On the 15 s shard's
226 q-rows it works out to ~42 us per row, so `TT_VSA_PROBE=9`'s per-TRISC phase timers are the next
instrument, not a kernel rewrite.

**3. The kernel is ~2x off the dense kernel at identical math, but that headroom is not reachable.**
`k=1808` is the same 14464 x 115712 attention the dense ring SDPA does in **51.4 ms**
(`VSA_README.md`); this path takes **103.7 ms**, and utilization saturates at ~35% even with
everything listed, consistent with `VSA_STREAM_DESIGN.md` 4's 26-30% ceiling. Dense efficiency at
Sol's density would be `0.16 x 51.4 ~ 8.2 ms` -- **below the measured 10.68 ms floor, so it cannot be
had.** The reachable ceiling at Sol's density is **~1.96x** (20.98 -> 10.68 ms) and it is NoC-bound.
Going under it means attacking the floor first.

**4. Sol's density is nearly free on delivery**: +0.43 ms from 10.9% to 16%. Its entire +4.9 ms cost
is math. A kernel with efficient math would therefore take Sol's routing at almost no latency cost --
the strongest case for the routing change, and an argument about the kernel rather than the routing.

**5. Locality is free today and would not stay free.** At 289 blocks `model` vs `topk` differ by 2.6%
on the full op but **35% on delivery** (10.68 vs 14.39 ms). Today's math hides the gather's locality
penalty; a delivery-bound kernel would expose it. Sol-Attn's threshold has different spatial
structure than top-k, so that becomes a question needing its own measurement.

### Order of work

1. ~~Delivery floor at Sol's density~~ **done (job 249)**: ~9.6 ms fixed, 10.68 ms at 16%. The
   dense-math projection is dead and the reachable ceiling is ~1.96x, delivery-bound.
2. **Find where the 9.6 ms goes** (`TT_VSA_PROBE=9` per-TRISC phase timers, single device). This is
   now the highest-value question in the op and it is not a Sol-Attn question at all. Everything below
   is worth less.
3. **Threshold routing in the coarse stage**, fixed width plus count. The oracle settled the sizing:
   289 +- 12% at the 15 s shard, so a fixed width at the max wastes ~13%, and `diag` suffices -- one
   `D`-vector of per-dim variance per (batch, head), no `D x D` matrix. `topk_large_indices` becomes a
   threshold compare plus a prefix count; the coarse stage already computes the full score matrix.
4. **Row LSE out of `vsa_sdpa`**, so the coarse output merges by log-sum-exp instead of the learned
   gate. Check whether the writer core's exact row sums (`vsa_sum_service.hpp`) can be exported cheaply.
5. **Fidelity A/B** against top-k at matched measured density, through S4's gate.

Steps 3-4 alone buy fidelity at +4.9 ms/block (~+1.2 s). They are worth doing only once step 2 says
whether the math can be hidden under the delivery floor -- if it can, Sol's routing is close to free
and the pair improves both axes.

## S3 - Fuse the coarse-stage setup

Sol fuses K/V summaries, block statistics and the threshold into one preprocessing kernel:
**1.206 -> 0.285 ms, -76.4%**. Our coarse stage is ~3.9 ms/block of unfused `ttnn` ops, and O3's
remainder is the same diagnosis: 9 `Transpose` at 1.20 ms, `NlpCreateHeads` 0.61 / `Ternary` 0.53 /
`NLPConcatHeads` 0.32, and 31 sub-20 us ops still paying full dispatch latency.

Nothing new to learn from Sol here beyond confirmation that the ceiling is real and modest.
`VSA_README.md` already names the fix (fold the transposes into the pooling layout). Worth ~1 ms/block,
~0.2 s of the run. Keep it as O3's cleanup, not a project. S2 step 2 adds the threshold compute to
this stage, so do the fusion *after* S2 lands or the work gets done twice.

## S4 - The quality gate (prerequisite, not a footnote)

`OPTIMIZATIONS.md`' standing caveat: **nobody has watched a FastH3 video.** CLIP is -2.4 to -2.9
against base and cannot separate detail from artifact. VBench has never run (`~/vbench_env` absent on
g03blx04). `check_spatial_seams` logs horizontal 0.666 against vertical 1.033 at three VAE tile
boundaries, diagnostic only, never asserted. Every number in that file is latency.

This blocks four separate things at once:

* **S1** - bfp8 K/V is lossy; the PCC floors in `_GATHER_DTYPE_PCC` are the only guard and PCC on a
  synthetic input is not a video.
* **S2** - Sol-Attn's entire claim is "same density, less loss." Unmeasurable today.
* **O5** - `vae_output_type="yuv420"` is the largest realized win to date (**-24% total**) and still
  is not the serving default *because* no pixel-comparing gate reads the yuv420 path.
* **O2's 256-token block** - explicitly "needs a quality gate."

Stand up a decoded-RGB PSNR/SSIM A/B (Sol reports exactly that pair for its MXFP8 mode: 13.82 dB /
0.540, which is also how we know to reject it) and/or VBench. This is the highest-leverage non-perf
item on the board and it is cheap relative to what it unblocks.

## Rejected

**MXFP8 DiT compute.** Sol quantizes attention and FFN linears in blocks 2-46 for **-11.1%** (1.655
-> 1.472 s) at **13.82 dB decoded-RGB PSNR and 0.540 SSIM** against its own BF16 output. That is a
severe fidelity cost for 11%. Independently, `OPTIMIZATIONS.md` S8 measures our denoise-precision
equivalent at ~0.5% with headroom only in the activation-dtype arm. Not worth the fidelity.

**Morton / z-order reordering per attention call.** Sol's header records this as a measured mistake:
only self-attention is order-sensitive, so the permutation belongs once at the block stack, and H3
needs none at all because the packed video tail is already a contiguous grid-ordered block. Our
`MiniMaxH3VSAGeometry.stream_order` already has a `zorder` option; this is evidence not to spend time
on it.

**Per-shape `tau` calibration.** See S2 point 3 - Sol tried it and it silently fell back to dense.

## Order

1. **S1** is done: **18.9 -> 18.5 s** at 15 s, -3.7% per forward, no fidelity cost. What is left of it
   is the typecast removal (0.08 s) and the unexplained 14 pp of gather time that does not scale with
   payload - settle that before spending effort on the AGMM dtype arm.
2. **S4** quality gate. Unblocks S2, O5 and O2's block-size question.
3. **S2** step 1 (host oracle, no device time) - can run in parallel with S4.
4. **S1** follow-ons: kill the typecast, then the AGMM dtype arm after O1's `program_config` arm.
5. **S2** steps 2-4.
6. **S3**, folded into O3's cleanup.
