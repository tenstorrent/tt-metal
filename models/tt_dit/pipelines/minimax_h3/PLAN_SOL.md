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

| | Sol-H3, 8x B300 | here, 4x8 BH Galaxy (job 886) |
|---|---|---|
| 15 s / 1344x768 / 24 fps + audio | **6.612 s** | **19.9 s** (denoise 13.4 / VAE 4.8 / audio 1.4) |

A 3x hardware-and-engine gap, not an 11x algorithm gap. Nothing in this plan closes 3x; the items
below are worth ~1.0-1.5 s in aggregate plus a quality result we currently cannot measure at all.

## The six Sol techniques, against this tree

| Sol technique | State here |
|---|---|
| Fused residual+RMSNorm+modulation, QKNorm+partial RoPE, SwiGLU | **Done.** `DitFusedDistributedRmsnorm` takes the modulation as `dynamic_weight`/`dynamic_bias`; the gated residual is folded into `to_out`'s matmul epilogue; `ff1` packs gate+up for a fused SwiGLU; `forward_fused_addcmul`. Sol's own `fusions.py` header rates this whole line at ~3% and "worth taking because it is lossless, not because it is large." |
| AdaLN precomputation | **Done.** `adaln_precompute.py` + `adaln_cache_minimax_h3.py`. Our residual cost is O7's six `Embeddings` gathers per block (1.06 s). Nothing to import. |
| Parallel VAE decoding (tile · batch · compile) | **Done.** 196 work units over waves of 32 (`vae_minimax_h3.py`). Sol's "batch" step is our wave batching; the leftover is O10 (repack the padded last wave). |
| Exact-KV sink + dense prefix query rows | **Done, under another name.** Sol's `sink_start`/`sink_tokens` plus dense recompute of prefix query rows is our `exempt_prefix` / `exempt_ids` / `dense_row_mask`. Same mechanism, same 64-token granularity. |
| Sparse setup fusion (1.206 -> 0.285 ms) | **Not done**, and the same diagnosis as O3's remainder. See S3 below. |
| INT8 QKV / FP8 output transport on the Ulysses all-to-all | **Not done.** See S1 below - the largest item here. |

Five of six already exist, several measured further than Sol's. The port that matters is narrow.

## S1 - Compress the collectives (landed, unmeasured)

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
* `test_vsa_attention_production_shape` parametrized on `gather_dtype` with its own PCC floor
  (`_GATHER_DTYPE_PCC`: bf16 0.985, bfp8_b 0.97 - the bf16 floor is selection noise across the
  bf16/fp32 boundary, so a narrowed gather should move the residual and not the selection).
* `VSA_GATHER_DTYPE` env knob on the block-perf, e2e-perf and pipeline paths for A/B without edits.

### What is predicted, and what has to be measured

Predicted: `AllGatherAsync` 8.40 ms/block = 1.68 s of the run -> ~0.74 s at half the payload, plus a
share of O1's four fused-collective matmuls (2.96 s) whose device time includes ~116 MB/device of
all-gather over 2 links. Against that, two `ttnn.typecast` calls per block each read the bf16 K or V
and write the bfp8 copy - DRAM-bound and *not* free; at ~10 heads/device x 14400 x 128 that is
~37 MB read + ~20 MB written per tensor per block.

**The typecast cost is the risk in this item.** If it eats the gather saving the whole lever is dead,
and that is decided by measurement, not by argument. Run in order:

1. `pytest tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa.py -k bfp8` - single device. Proves
   the reader honours the bfp8 tile size. **If this fails the item stops here** and becomes kernel work.
2. `test_vsa_attention_production_shape -k "15s_768p and zero and bfloat8_b"` - Galaxy. Confirms
   selection is unchanged and fixes the real PCC (the 0.97 floor is a guess until this runs).
3. `VSA_GATHER_DTYPE=bfloat8_b scripts/run_safe_pytest.sh --profile .../test_vsa_performance_minimax_h3.py -k 15s_768p`,
   Tracy, signpost window, 32-device merge. Read `AllGatherAsync` and the new `Typecast` rows
   directly. This is the number that decides the lever - the block profile resolves ~2.7 ms deltas
   that a single e2e run cannot.
4. Only then `VSA_GATHER_DTYPE=bfloat8_b test_vsa_e2e_perf_minimax_h3.py`, against job 886's 19.9 s.
   Run-to-run variance at fixed shape and seed is **+-8%** (~1.6 s), so a single e2e run cannot
   establish anything under ~1.6 s. Do not report an e2e delta from one pair of runs.

### Follow-ons once S1 is proven

* **Kill the typecast.** The QKV projection writes K/V in bf16 because QK-norm, RoPE and the coarse
  pooling all consume them. If the fine stage is the only consumer of the *gathered* copy, the
  narrowing can move into the gather's own output dtype (persistent buffer allocated bfp8_b, input
  bf16) if `all_gather_async` will accept a dtype change - check; if not, this is the op request that
  makes S1 strictly free.
* **The four AGMMs (O1).** `all_gather_minimal_matmul_async` fuses the TP gather into the matmul, so
  its input dtype is the lever on the same 2.96 s O1 already flags as `SLOW` for a different reason
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

**Problem 1: variable list length.** Threshold routing yields a data-dependent count per row instead
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

### What it is worth

**Not latency.** O2 already establishes the fine stage runs at 23-25% of HiFi2 peak with per-TRISC
busy at PACK 93% / MATH 89% / UNPACK 87%, and localizes the floor in per-visit SFPU bookkeeping that
cannot amortise when a row lists ~1 block in 9. Sol-Attn does not change that arithmetic; if
anything, the residual branch adds work.

What it buys is **fidelity at equal density** - keeping the dropped probability mass instead of
discarding it. Which is worth having precisely because it unblocks the *other* lever: O2's remaining
move is a 256-token VSA block (4x amortisation of every per-visit cost) and it is blocked on a
quality gate. Same gate, both items.

Order of work:

1. Vendor `triton_ref/{preprocess,fwd}.py` next to `models/transformers/minimax_h3/vsa_reference/`
   (Apache-2.0, same as the FastVideo VSA reference already vendored there) and add a host-only
   oracle test in the `test_vsa_oracle_minimax_h3.py` shape. **No device time.** This settles the
   threshold statistics and `tau=1.0` on H3's real shapes before any kernel work.
2. Add a threshold-routing mode to `MiniMaxH3VSACoarseStage` (fixed width + count). Gate against the
   oracle at sparsity-equivalent density.
3. Export row LSE from `vsa_sdpa` and replace the gated blend with the LSE merge.
4. A/B fidelity against top-k at matched measured density, via S4's gate.

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

1. **S1** measurement chain, steps 1-4. Code has landed; the numbers have not.
2. **S4** quality gate. Unblocks S2, O5 and O2's block-size question.
3. **S2** step 1 (host oracle, no device time) - can run in parallel with S4.
4. **S1** follow-ons: kill the typecast, then the AGMM dtype arm after O1's `program_config` arm.
5. **S2** steps 2-4.
6. **S3**, folded into O3's cleanup.
