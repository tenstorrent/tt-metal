# HunyuanVideo 1.5 optimization inventory

Last updated: 2026-08-06. Primary target: Blackhole Galaxy, 8x4 / 32 chips,
121 output frames, 50 denoising steps.

## READ FIRST: every VAE timing in this document is PROVISIONAL

**No VAE latency conclusion in this report is currently trustworthy.** The
recorded numbers are a mix of cold (first-decode, includes compile and weight
upload) and warm measurements, and they were never separated. Three
observations show the mix directly:

- 480p H/W-sharded decode was recorded at **92.97s** and later measured at
  **44.23s** for the same work. The 2.48x "H/W is slower" verdict rests on the
  92.97s figure.
- One chunked-attention run measured **140.40s cold versus 64.37s warm**, a
  2.18x gap attributable entirely to first-decode cost.
- 720p tile decode measured **13.07s** against 480p tile decode at **37.55s**.
  A higher resolution cannot decode faster than a lower one on the same path;
  the 480p figure is cold and the 720p figure is warm.

Every VAE timing below is therefore labelled `PROVISIONAL (cold?)` at its point
of use. The old numbers are **retained, not deleted**, so the clean table can be
diffed against them. The device-owning agent is producing an all-warm VAE table;
until it lands, do not use any VAE latency figure here to choose a default path,
and do not quote the 2.48x H/W regression or the tile-versus-H/W comparison.
Correctness and memory results in this document are unaffected -- PCC and
allocation figures do not depend on cache state.

## Current architecture and baseline

- HunyuanVideo 1.5 is a single-expert, 54-block dual-stream MMDiT. It is not
  Wan2.2's two-expert MoE and has no distilled checkpoint in this repository.
- Conditioning is Qwen2.5-VL text plus byT5 text, and (for I2V) SigLIP image
  tokens plus a VAE-encoded first-frame latent. T2V and I2V use separate
  checkpoints; both support 480p and 720p.
- The production 32-chip layout is sequence parallel 8 x tensor parallel 4.
  The DiT occupies the complete mesh. The DiT and tile-sharded VAE are on
  device; the scheduler, Qwen, byT5, and final media post-processing remain on
  the host by default.
- Validated 121-frame / 50-step baseline from `README.md`:
  - 480p T2V: 1:51 denoise, 5:09 end to end.
  - 720p T2V: 5:00 denoise, 9:12 end to end.
  - 480p I2V: 1:58 denoise, 5:19 end to end.
  - 720p I2V: 5:07 denoise, 9:27 end to end.
- 720p T2V must call `scheduler.set_shift(9.0)` (I2V uses 7.0).
  Updating only the scheduler config is not effective.

## Blackhole hardware validation (2026-08-06)

- Preflight found no active job or device holder after the authorized stale
  `tt-smi` termination. All 32 chips reported healthy DRAM and zero uncorrectable
  GDDR errors; no reset was performed.
- Focused gates: fused QKV versus legacy/reference passed all four PCC >= 0.99
  assertions; device CFG/Euler passed with CFG both off and on; VAE two-chip
  single-readback passed at PCC 0.999998593; SigLIP passed at PCC 0.994659.
  Qwen skipped because the required cached `text_encoder` weights were absent.
- Cached 480p I2V 13f/4-step, host text/SigLIP/VAE, seed 0:

  | configuration | denoise | wall | physical DiT runs | frame PCC vs legacy |
  |---|---:|---:|---:|---:|
  | legacy eager, default mixed lengths | 21s | 239.29s | 8 | 1.000000 |
  | QKV eager, default mixed lengths | 11s | 234.71s | 8 | 0.950332 |
  | legacy eager, equal lengths | 5s | 226.23s | 4 | 1.000000 |
  | QKV eager, equal lengths | 5s | 229.91s | 4 | 0.996543 |
  | resident eager, equal lengths | 8s | 238.54s | 4 | 0.998612 |
  | QKV + resident eager, equal lengths | 5s | 236.03s | 4 | 0.998012 |

  Each run made 8 diffusers transformer calls. Equal lengths were used for the
  resident comparison to avoid the legacy padding-quality tradeoff. Denoise is
  the tqdm elapsed time and includes first-use compilation; wall includes model
  loading, host text/SigLIP, host VAE, and media saving.
- Trace investigation found two correctness bugs. Mesh capture cannot compile
  uncached programs, and capture does not execute its output. Intermediate
  compile-output reuse variants produced frame PCC 0.422903 and 0.004703.
  The retained conservative implementation is compile warmup, capture, then
  explicit first trace execution; trace stays disabled pending a fresh real
  generation gate.
- A separate generation job acquired all 32 devices during the final trace
  retest. Validation stopped without collision. VAE timing A/B and the
  121f/50-step run were therefore not performed.
- Best safe flags from completed evidence:
  `HY_TRACE=0 HY_DEVICE_RESIDENT_DENOISE=0 HY_DIT_QKV_SPLIT=0`, retaining host
  text/SigLIP and the established tile-sharded TT VAE for production-length runs.
- Focused command:
  `HF_HUB_OFFLINE=1 pytest -svv models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_transformer_block_qkv_split_pcc.py models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_device_resident_denoise.py::test_resident_device_cfg_euler_matches_diffusers models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_vae_decoder.py::test_vae_sharded_rounds_stay_on_device_until_one_readback models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_siglip.py::test_siglip_on_device_pcc`.
- Generation command used for each matrix row:
  `HF_HUB_OFFLINE=1 HY_I2V=1 HY_IMAGE=/home/tt-admin/sdawle/hunyuanvideo1.5/hy480p_i2v_frame090.png HY_MESH=8,4 HY_DIT_SP=1 HY_DIT_BF16=1 HY_TT_QWEN=0 HY_TT_SIGLIP=0 HY_TT_VAE=0 HY_FRAMES=13 HY_STEPS=4 HY_OUT=<row-output> HY_DIT_QKV_SPLIT=<0|1> HY_DEVICE_RESIDENT_DENOISE=<0|1> HY_TRACE=<0|1> pytest -svv models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2`.

### Mixed-length fused-ring masking (implemented, not hardware-qualified)

- `ring_joint_scaled_dot_product_attention` now has an optional
  `joint_valid_lengths` vector, one valid joint-token prefix per batch row.
  The streaming ring kernel skips fully invalid joint chunks, narrows the
  boundary chunk, and applies a row-specific `-inf` partial-tile mask before
  softmax. The existing shared `logical_n` mask still removes SP spatial tail
  padding.
- Hunyuan's opt-in `HY_CFG_PADDING_POLICY=masked` packs each row as
  `[valid image, valid byT5, valid Qwen, padding]`. Joint attention is
  permutation-equivariant over encoder tokens, so this preserves valid-token
  order and reference latent math while making the mask a prefix. Invalid
  encoder query states are zeroed before/after each block.
- Qwen refinement is intentionally sliced to each row's exact valid length
  before padding/packing. This preserves its masked pooling and self-attention;
  applying only the later DiT key mask would still allow pre-DiT leakage.
- Execution contract: mixed positive/unconditional CFG goes from two full DiT
  forwards per denoise step to one batched forward (8 to 4 physical DiT runs
  for the measured 13f/4-step scenario). The small two-block Qwen token
  refiner remains per-row to preserve exact semantics.
- Host-only adapter/fail-closed tests: 3 passed. Python syntax and whitespace
  checks pass. The full `ttnn` target also builds successfully. Added
  fused-ring eager/trace golden coverage with batch lengths `[96, 37]`, SP
  padding (`N=250`), and PCC threshold 0.999.
- The focused hardware golden was attempted after the concurrent VAE test
  released the devices, but both eager and trace cases failed while opening
  the 2x4 mesh: fabric-router synchronization timed out on device 1 before any
  SDPA kernel executed. Per the no-reset constraint, no recovery/reset was
  attempted.
- **Superseded (2026-08-06).** That 2x4 open failure is a real, reproducible
  fabric defect and not contention: it recurs on a completely quiet machine
  (device 1 master `chan=3` at `0xa1b1c1d1`, `chan=4` stuck at `STARTED`
  `0xa0b0c0d0`) while an 8x4 open on the same machine succeeds. Tests needing a
  smaller mesh must open 8x4 and take a submesh.
- **The masked policy itself passes on hardware.**
  `HY_CFG_PADDING_POLICY=masked` with `HY_DEVICE_RESIDENT_DENOISE=1` completed a
  real i2v 480p 13-frame / 8-step generation in 225.04s wall at frame PCC
  1.000000 against the reference path, on a harness verified deterministic
  (seed pinned at `tests/e2e/test_stage2b_gen.py:280`; eager-versus-eager is
  bit-identical). Keep it opt-in pending a 121-frame 50-step quality run, and
  keep it paired with `HY_TRACE=0` -- traced replay is currently incorrect.

## Blackhole hardware validation, later session (2026-08-06)

A second validation pass on an otherwise quiet machine. Everything in this
section is measured unless explicitly marked otherwise.

### Trace replay is per-step incorrect (blocker)

- Trace **capture** is exact and trace **replay** is not. A 1-step traced
  generation is bit-identical to eager (PCC 1.000000). The same configuration
  at 8 steps gives aggregate PCC **0.237300**, with per-frame PCC degrading
  monotonically: 0.9747, 0.8648, then negative (-0.0657 down to -0.3122).
- Frame 0 survives only because this is i2v: the conditioning image anchors it.
  It is not evidence that the first replayed step is correct.
- This is the **same signature** as the already-rejected heterogeneous trace
  (aggregate PCC 0.235647, recorded at `README.md:158` and under "Mixed-shape
  trace is rejected in production" below). One defect, two symptoms: whatever
  the traced program is capturing is not being re-driven correctly per step.
- Ruled out by source inspection, so the cause is elsewhere: time-embedding and
  patch-embedding placement (both inside the captured region), the `traced=True`
  copy at `tt/pipeline.py:1135`, and the Euler update (eager and trace share the
  same code path, and eager is exact).
- Consequence: `HY_TRACE=1` **must not** be used with
  `HY_CFG_PADDING_POLICY=masked`, or with anything else, until this is fixed.
  `HY_TRACE=0` remains the only correct setting for a real generation.

### The generation harness is deterministic, so cross-path PCC is meaningful

- The seed is pinned inside the shared `_run()` helper --
  `torch.Generator().manual_seed(0)` at `tests/e2e/test_stage2b_gen.py:280` --
  so every configuration compared through that helper starts from the same
  noise.
- Verified rather than assumed: an eager-versus-eager repeat is bit-identical
  (frame PCC 1.000000, max absolute difference 0.000000).
- This retroactively validates the whole family of cross-path PNG comparisons in
  this report, including the masked-CFG 1.000000 result below. A PCC of 1.0
  between two paths means they agree, not that the harness re-used a file.

### Masked CFG works end to end on hardware

- `HY_CFG_PADDING_POLICY=masked` with `HY_DEVICE_RESIDENT_DENOISE=1` completed a
  real i2v 480p generation at 13 frames / 8 steps in **225.04s** wall, matching
  the reference path at frame PCC 1.000000 (see determinism above).
- This was previously blocked, not failing: the earlier attempt died in the
  fabric-router timeout described below, before any SDPA kernel ran. The mask
  itself was never the problem.
- Still not a production default: this is one short generation, not a 121-frame
  50-step quality run, and it must stay paired with `HY_TRACE=0`.

### Trace economics, and why trace is never actually reused

- Warm fits over step count `n`: eager ~ `1.72 + 1.016 * n` seconds, trace ~
  `7.27 + 0.416 * n`. Break-even is **~9.3 steps**, well inside the 50-step
  production default.
- That break-even is never reached in practice because the trace is thrown away
  after every generation: `tests/e2e/test_stage2b_gen.py:288-289` releases it in
  a `finally` block. Each run therefore pays the full ~7.27s capture.
- Reuse is structurally possible today. The pipeline guards capture on
  `_trace_id is None`, so a retained trace is simply not recaptured. Removing
  the unconditional release would drop break-even to **~3 steps**.
- This is worth doing only after the replay-correctness defect above is fixed.
  Reusing an incorrect trace across generations makes it worse, not cheaper.

### Chunked mid-block VAE attention: validated on hardware

- All 7 device cases in `tests/pcc/test_vae_attention_chunking.py` pass:
  PCC **0.9999985** at query chunks 1, 7 and 32, **0.9999988** at 512, and
  **0.9999984** in the sharded case.
- Peak DRAM for the measured block fell from **9,207,808 to 1,228,800 bytes**
  (7.5x).
- The pre-existing VAE suite passes unchanged with chunking forced on
  (38 tests), so chunking is not a behavioural fork.
- The 720p allocation gate that previously demanded **23.2 GiB** now needs
  **0.74 GiB**, which is what makes 720p H/W decode possible at all.
- Matched 480p A/B: cross-path video PCC **0.9999899** while reclaiming
  **4,753 MiB** of device memory, with host peak RSS dropping **32.3 -> 9.4
  GiB**.
- 720p chunked decode completes, producing output `[1, 3, 121, 720, 1280]`.
  This is the OOM from the previous session, resolved.
- Timing from these runs is deliberately not quoted here; see the cold/warm
  caveat at the top.

### Flash SDPA accepts num_heads=1 with head_dim=1024

- This was the one open question a static reading of the kernel could not
  settle, and the answer is that it **works**.
- `test_device_flash_sdpa_matches_the_matmul_blocks` passes at PCC **0.9999913**
  at `C = 1024` with the derived `k_chunk = 64`, and at **0.9999936** at
  `C = 64` with `k_chunk = 1760`.
- The predicted failure mode (a circular-buffer or kernel-geometry error) did
  not occur, and the L1 budget model that derived `k_chunk = 64` from the head
  dim was correct.

### byT5: the gate passes once one config field stops being enforced

- All 5 cases of `tests/pcc/test_byt5_encoder_pcc.py` initially failed closed on
  the adapter's checkpoint validation, on `tie_word_embeddings=True (expected
  False)` alone.
- The checkpoint's own `text_encoder_2/config.json` stores `false`;
  `T5Config.from_pretrained` returns `True`. HuggingFace does not round-trip the
  field. This is reproducible on host and is now asserted by
  `test_the_parsed_real_config_is_accepted_even_though_hf_rewrites_a_field`.
- The field governs only LM-head weight tying, and `T5EncoderModel` has no LM
  head, so it cannot affect an encoder activation. It is therefore no longer
  enforced; see "Reusing tt_dit T5/UMT5 for byT5" below for the contract
  decision and what still is enforced.
- With the checkpoint's true value in effect, all 5 cases pass: TP(1,1) PCC
  **0.999935**, TP(1,2) **0.999938**, full sequence without zero-padding
  **0.999931**, batched-row consistency ~**1.000**, and the adapter's own
  first-call self-check passes. All outputs are finite and non-zero.
- Two consequences worth folding into the design, not just recording:
  - **Padding neutralization is not load bearing.** The full-sequence case
    passes with `zero_padding=False`, so `HY_BYT5_ZERO_PAD=1` is a defensive
    measure rather than a correctness requirement.
  - **The `(mask - 1) * inf -> 0 * inf = NaN` hazard does not manifest** on
    device. It was a real risk in the additive-mask expression; hardware shows
    the shared T5 stack does not hit it at these shapes.

### Fabric: opening a 2x4 mesh directly genuinely fails

- On a completely quiet machine with no other job, opening a 2x4 mesh directly
  fails during fabric-router synchronization on **device 1**: the master reports
  `chan=3` at `0xa1b1c1d1` while `chan=4` is stuck at `STARTED` (`0xa0b0c0d0`).
- Opening the full **8x4** mesh on the same machine succeeds.
- **Contention was not the cause.** The earlier session attributed this failure
  to a concurrent job; that explanation is now disproven.
- The working pattern, and what device tests requesting a sub-8x4 mesh should
  use, is to open 8x4 and take a submesh from it.

### The distributed VAE attention device suite was blocked by its own parameters

- The 7 device cases in `tests/pcc/test_vae_attention_distributed.py` failed
  before reaching hardware with `ValueError: rank-local H edge fill requires at
  least one logical row on the final rank`.
- This was a test bug, not a code bug. The cases asked for a 5-row latent on a
  4-rank H axis, which is 2-row shards with a 3-row tail, leaving rank 3 holding
  nothing but padding and nothing to replicate an edge from. That partition is
  rejected by design, and the rejection is itself covered by
  `test_a_rank_made_entirely_of_padding_is_rejected`.
- Fixed by pairing each device case with a geometry its mesh can partition, and
  by adding `test_every_hardware_case_partitions_legally`, a host guard over the
  device parameter lists so the same edit cannot reach hardware again. The error
  message now names the offending H, the per-rank row count, the rank count, and
  the smallest legal H for that mesh.
- The suite now covers even, H-only-uneven, W-only-uneven and both-uneven
  partitions, plus the real 30x53 and 45x80 production grids on 8x4.

## Implemented and retained optimizations

### Sequence parallel DiT with ring joint attention

- Source pattern: Wan2.2 `DiTParallelConfig`, ring attention, and 4x8 Galaxy
  preset in `models/tt_dit/pipelines/wan/pipeline_wan.py`.
- Hunyuan implementation:
  `tt/pipeline.py` and `_stubs/hunyuan_video15_transformer_block.py`.
- Status: production path, enabled with `HY_MESH=8,4 HY_DIT_SP=1`.
- Benchmark delta: increasing SP from 4 to 8 reduced 480p denoise from
  3.24 s/step to 2.15 s/step. Persistent reduce-scatter plus the tuned SDPA
  compute config reduced it further to 2.02 s/step.
- Correctness: SP=8 versus SP=4 generated-frame PCC 0.9971; the component and
  end-to-end gates remain above their thresholds.

### Tensor-parallel weights and persistent collective buffers

- Source pattern: Wan row/column parallel linears and
  `CCLManager` persistent all-gather/reduce-scatter buffers.
- Hunyuan implementation:
  `_stubs/hunyuan_video15_transformer_block.py`.
- Status: production path. TP=4 follows the physical Galaxy dimension.
- Benchmark delta: included in the 2.02 s/step 480p result above.
- Correctness: real-weight DiT PCC threshold 0.99 and mesh component tests.

### Heads-major attention layout (device-profile driven)

- Motivation: a Tracy device profile (480p I2V, 13 frames, 1 step, SP8xTP4)
  ranked ops by `DEVICE KERNEL DURATION`. LAYOUT/MOVE was the largest category
  at 34.5% of device kernel time, with `ReshapeViewDeviceOperation` alone at
  26.9% -- the most expensive single op in the model, ahead of matmul (20.1%),
  elementwise (15.7%), SDPA (11.9%) and CCL (11.3%).
- Root cause: with TP=4 the LOCAL head count is 4. In `TILE_LAYOUT` that axis
  sits second-to-last in `(B, S, H, D)` and pads to a full 32-row tile, so every
  tensor routed through that intermediate moves 8x its necessary bytes. The
  per-shape breakdown confirmed it: the two `heads_split` reshapes cost
  310.9 ms and the two reverse reshapes 174.0 ms, against 121.3 ms for all
  eight permutes combined.
- Implementation: `_stubs/hunyuan_video15_transformer_block.py`.
  `HY_DIT_FUSED_HEADS=1` merges the joint-SDPA output with `nlp_concat_heads`.
  `HY_DIT_FUSED_QKV_HEADS=1` builds Q/K/V heads-major via
  `nlp_create_qkv_heads` (`transpose_k_heads=False`) directly from the fused
  `[q|k|v]` projection -- the layout each Hunyuan TP shard already stores --
  removing three slices, three reshapes and six permutes. RoPE moves to
  `_apply_rope_hm`/`_rope_bcast_hm`, whose collapse then reads `(S, D)` (both
  tile-aligned) rather than the padded head axis.
- Benchmark: matched 480p I2V 121f/50-step runs on a freshly reset Galaxy,
  production flags (`HY_TT_VAE=1 HY_VAE_TILE=1`, host Qwen/SigLIP, `HY_TRACE=0`).
  Three arms, so the CFG-policy cost is separated from the layout change.

  | configuration | CFG policy | DiT runs | denoise | s/step | e2e |
  |---|---|---:|---:|---:|---:|
  | current default | `separate` | 100 | 2:05 | 2.52 | 353.1s |
  | baseline config | `masked` | 50 | 1:58 | 2.38 | 345.5s |
  | both flags | `masked` | 50 | 1:20 | 1.60 | 289.5s |

  The `masked` legacy arm reproduces this README's published 1:58 denoise
  exactly, which anchors the comparison: **-32.2% denoise like-for-like**, and
  -16.2% e2e within the matched runs. Against what the pipeline does by default
  today (`separate`, no fusion) it is 2:05 -> 1:20, **-36% denoise**.
  All three arms are bit-identical to each other.
- 720p I2V 121f/25 steps, same flags and `masked` policy: 6.40 -> 4.79 s/step
  (-25.2% denoise), e2e 432.5s -> 376.2s (-13.0%), frame PCC 1.00000000 with
  max absolute pixel difference 0.0. The gain is smaller than 480p's -32.2%
  exactly as expected: attention is the one quadratic term, so at 111,600
  tokens it takes a larger share of the step and the layout saving is diluted.
- Side result: `HY_CFG_PADDING_POLICY=masked` was previously validated only at
  13f/8 steps. These runs validate it at 121f/50 steps, bit-identical to
  `separate`, and 7s faster in denoise (50 physical DiT runs instead of 100).
- Correctness: both are pure data-movement rearrangements, so generated output
  is **bit-identical**, not merely within tolerance -- frame PCC 1.00000000
  with max absolute pixel difference 0.0 across all 121 frames. The
  single-device block PCC gate passes with each flag independently on and off,
  and the SP/TP mesh gate passes with both enabled.
- Caveats: the legacy arm measured 2:17/6:02 against this README's published
  1:58/5:19, because the tree carried substantial uncommitted work when these
  runs were taken. The A/B is internally matched so the delta holds, but the
  absolute numbers need a re-baseline on a clean tree. **720p is unverified** --
  both arms failed at device init on a board that needed a reset. Expect a
  smaller relative gain there because attention, the one quadratic term,
  dominates more at 111,600 tokens.
- **Both flags now default ON.** The 720p A/B and the clean re-baseline are
  done, output is bit-identical at both resolutions, and the single-device and
  sharded-mesh gates pass with the new defaults in force (no env override).
  Set either flag to `0` to restore the legacy path for bisection.

### Wan SDPA chunk preset at Hunyuan shapes (rejected -- does not run)

- `HY_DIT_SDPA_PRESET=wan_bh_sp8tp4` (q=288, k=512) was carried as an
  unbenchmarked A/B candidate. Measured at 480p I2V 121f/25 steps on top of the
  fused-heads flags with `HY_CFG_PADDING_POLICY=masked`:

  | preset | s/step | result |
  |---|---:|---|
  | `hunyuan` (q=128, k=512) | 1.76 | passed |
  | `wan_bh_sp8tp4` (q=288, k=512) | -- | **aborts** |

- The failure is not a regression but a hard stop: `TT_THROW: Statically
  allocated circular buffers in program N clash with L1 buffers on core range
  [0-0 - 11-8]`, repeated across many programs. The larger query chunk needs
  more L1 for its circular buffers than is available alongside the resident L1
  buffers at Hunyuan's 49,408-token latent length.
- This confirms the original caution for a concrete reason: Wan's tuned query
  shape does not transfer to Hunyuan's token lengths and CCL core reservation.
  The `hunyuan` preset stays the only working setting; the candidate can be
  closed rather than left pending.

### Two more flags measured: one rejected, one neutral

- **`HY_VAE_ATTN_SDPA=1` on the TILED path: 4.4x slower. Reject.** VAE decode
  13.3s -> 57.9s at 480p/121f. This is the exact inverse of the sharded-path
  result, where the same flag was *required* to avoid materialising a 4.86 GB
  score matrix at 49,290 tokens. With 128px tiles each tile's sequence is short
  enough that materialising the matrix is cheap and the streaming kernel's
  overhead dominates. **Same flag, opposite verdict, decided by sequence length.**
- **`HY_DIT_RS_DOMAIN_BIAS=1`: bit-identical, but no measurable speedup.** It
  adds the row-parallel bias to the reduce-scatter output instead of after the
  all-gather, touching a tensor `tp` times narrower. Generated output is
  bit-identical (frame PCC 1.00000000, max abs pixel difference 0.0), confirming
  the parallel session's correctness argument. Timing across two batches:

  | | s/step round 1 | s/step round 2 |
  |---|---:|---:|
  | base | 1.69 | 1.66 |
  | `HY_DIT_RS_DOMAIN_BIAS=1` | 1.61 | 1.67 |

  -4.7% then +0.6%; the two ranges overlap. The bias add is a small elementwise
  op on the residual, so there is little to win. Safe to enable, not a win.

- **Methodology correction.** An earlier entry said back-to-back runs inside one
  batch are "reproducible to the hundredth of a second". That held for the
  heads-major A/B (2.46/2.46 vs 1.67/1.67) -- but that was a **32%** effect. The
  rows above are both within-batch and disagree by 4%. The honest rule is a
  **resolution floor of roughly 5% regardless of batching**: effects below it
  need many repeats or a device-level profile, not a wall-clock A/B.

### Prompt cache: MEASURED at last (-10.5s on a repeat prompt)

- `HY_PROMPT_CACHE=1` already existed but had never been benchmarked; this
  report previously said "no warm served Qwen or prompt-cache timing has been
  measured, so no warm speedup is claimed". Measured now, 480p i2v 121f with all
  other optimisations on:

  | | `text_encode_s` | wall |
  |---|---:|---:|
  | off | 10.152s | 140.2s |
  | cold (populates) | 10.178s | 138.5s |
  | **warm** | **0.013s** | **129.7s** |

- **-10.5s wall on a repeat prompt**, text encode 781x faster, generated output
  bit-identical (frame PCC 1.00000000, max abs pixel difference 0.0). Cache is
  16 MB. The cold arm costs essentially nothing to populate (10.178 vs 10.152s),
  so there is no penalty for enabling it speculatively.
- **This is a served-path win, not a one-shot one.** It helps only when the same
  prompt is generated again; a cold one-shot run pays full text-encode either
  way. Do not fold it into the headline one-shot numbers.

### VAE prepared-conv-weight cache (IMPLEMENTED: -11.8s, bit-identical)

- After the DiT work, `tt_vae_weight_upload_s` was the largest remaining phase
  at ~12.6s. `TTVAEDecodeAdapter` construction runs
  `ttnn.experimental.prepare_conv3d_weights` for every causal conv, reformatting
  each weight for the conv3d kernel on the host -- the same class of host-side
  preparation the DiT cache removes. This report flagged the gap long ago ("no
  serialized prepared-weight cache or explicit load/deallocate lifecycle").
- `HY_VAE_WEIGHT_CACHE=1` caches the prepared tensors:

  | | `tt_vae_weight_upload_s` | cache |
  |---|---:|---:|
  | off | 12.617s | -- |
  | cold (populates) | 19.061s | 1.7 GB |
  | **warm** | **0.809s** | 1.7 GB |

- **12.6s -> 0.81s, 15.6x.** Generated output bit-identical in both the cold and
  warm arms (frame PCC 1.00000000, max abs pixel difference 0.0 across 121
  frames), confirming the conv3d-prepared layout survives `DumpTensorMode.LOCAL`
  round-tripping exactly -- which was the open risk, since that layout is
  kernel-specific and differs from ordinary DiT weights.
- 1.7 GB per configuration, a tenth of the DiT cache's 16 GB. The cold run pays
  ~6.4s to populate. Convs are constructed in deterministic order so a
  sequential index is a sufficient key; mesh shape, core grid, dtype and the H/W
  sharding mode go in the directory tag.

### Unused mid-granularity part stubs (IMPLEMENTED: -28s, bit-identical)

- A phase breakdown of a 480p i2v 121f run put `tt_dit_weight_upload_s` at
  **63.7s -- the largest single phase**, larger than denoise, VAE decode and
  writeout combined, and that was already *with* the prepared-weight cache.
- Cause: `build_pipeline` unconditionally builds four 54-element stub lists --
  `s_adazero`, `s_adazero_ctx`, `s_ff`, `s_ff_ctx`. They are referenced only by
  `_transformer_block_from_parts`, which runs only at `granularity="mid"`.
  Generation calls `_forward_encoded(ctx, "composite")` and goes through the
  fused `s_blocks`, which extracts its own AdaLN and feed-forward weights.
  **216 stubs therefore upload a second copy of the model's largest weights
  (FFN, 2048x8192 and 8192x2048 per block) that the production path never
  reads.**
- `HY_DIT_SKIP_PARTS_STUBS=1` skips them:

  | | wall | `tt_dit_weight_upload_s` |
  |---|---:|---:|
  | baseline | 130.7s | 48.3s |
  | skip parts stubs | **102.7s** | **17.3s** |

- **-28s wall, upload -64%**, generated output bit-identical (frame PCC
  1.00000000, max abs pixel difference 0.0) -- necessarily so, since the skipped
  weights were never read.
- Skipped lists are replaced by a `_SkippedStubs` guard that raises on any
  access, so `granularity="mid"` fails loudly instead of silently taking a
  different path. Default off.

### DiT prepared-weight cache (IMPLEMENTED: -49.6s, bit-identical)

- Motivation: after the heads-major change, denoise is no longer the e2e
  bottleneck. A 480p i2v 121f/50-step run spends ~72% of its wall clock outside
  denoise, and a stage breakdown puts ~99s of silent work between "Enabling
  program cache" and the first denoise tick. Qwen and byT5 avoid re-preparing
  weights every process via tt_dit's `cache.load_model`; the DiT does not.
- Single-device measurement at the real per-block weight shapes confirms the
  cost is host-side preparation, not device transfer:

  | weight | MB | `from_torch` | cached `load` | speedup |
  |---|---:|---:|---:|---:|
  | qkv (col-parallel) | 25.2 | 72.5 ms | 7.9 ms | 9.2x |
  | ffn up | 33.6 | 96.3 ms | 10.5 ms | 9.2x |
  | ffn down | 33.6 | 96.4 ms | 10.5 ms | 9.2x |
  | adaLN proj | 50.3 | 144.6 ms | 15.6 ms | 9.3x |
  | **per block** | **235** | **653 ms** | **73 ms** | |

- **Implemented and measured: 231.5s -> 181.9s wall (-49.6s), output
  bit-identical (frame PCC 1.00000000, max abs pixel difference 0.0 across 121
  frames).** Opt-in via `HY_DIT_WEIGHT_CACHE=1`. Because the saving is setup
  cost it is constant in step count: a 50-step production run goes from ~290s to
  ~240s (-17% e2e). The cache is 16 GB / 1350 files per configuration.
- The key is `ttnn.DumpTensorMode.LOCAL`, the mode tt_dit's `Parameter.save`
  uses. It persists each device's own shard and restores the same placement, so
  the round trip is bit-exact and the cache is 1.00x the logical size. An
  earlier investigation reported this approach blocked on three counts -- all
  three were an artefact of calling `dump_tensor` WITHOUT that mode:
  1. `dump_tensor(from_device(t))` -- no mode -- serialises only ONE device
     shard and `load_tensor` restores it **replicated** across all 32 devices,
     silently wrong on 31 of them. With `mode=DumpTensorMode.LOCAL` the round
     trip is bit-exact. **This is the trap to avoid, not a blocker.**
  2. Hand-rolling a per-shard cache is 8x disk-bloated (32 shards of a 33.6 MB
     weight wrote 268.5 MB, because weights replicate across the 8 SP ranks).
     `LOCAL` mode dedups this: measured 1.00x, i.e. 16 GB for the DiT.
  3. `ttnn.aggregate_as_tensor` is absent in this build -- not needed, since
     `load_tensor(path, device=mesh)` restores placement directly.
- Scale correction: `from_torch` mesh-sharded costs ~84 ms warm, not the 535 ms
  a first (cold) call suggests. Weight preparation is therefore roughly 32s of
  the ~99s window, not all of it; the remainder is mesh setup and program-cache
  population.
- Implementation is contained rather than the Module refactor first scoped:
  `f32()` is the single choke point every block weight flows through, and
  weights are built in deterministic order, so a per-block prefix plus a
  sequential index is a sufficient key. Everything that changes dtype, shape or
  shard placement (mesh shape, tp/sp, axes, dtype, `HY_DIT_RS_DOMAIN_BIAS`) goes
  in the cache directory tag, so an entry can only be reused by an identical
  configuration.
- Prefetching remains a non-substitute: weights already land in device DRAM and
  the cost is host-side preparation, so overlap could hide at most the ~32s of
  concurrent setup -- less than the cache delivers, for more complexity.

### Trace replay defect: stale-timestep hypothesis ruled out

- `HY_TRACE=1` captures exactly (1-step traced generation is bit-identical to
  eager) but replay degrades across steps (aggregate PCC 0.237300). The natural
  explanation is a resident input buffer not refreshed between replays.
- That is not the cause: `denoise_write_inputs` copies **both** the latent and
  the timestep into their resident buffers on CQ1 every step
  (`tt/pipeline.py`, `copy_host_to_device_tensor` for `resident_hidden` and
  `r["timestep"]`). Ruled out, so the defect lies elsewhere.

### Media writeout (IMPLEMENTED: -40.4s, output unchanged)

- Splitting the ~90s post-denoise window at 121 frames gives roughly **50s of
  media writeout against 40s of VAE decode**, so writeout -- not the VAE -- is
  the larger post-denoise target, and it needs no kernel work.
- `tt/media_writeout.py` already implemented threaded PNG writing and a GIF
  gate, but nothing imported it: `tests/e2e/test_stage2b_gen.py` still called
  Pillow's `pil[0].save(..., save_all=True)` inline. Wiring it in, with defaults
  that reproduce the previous behaviour byte for byte:

  | mode | PNG | GIF | wall |
  |---|---:|---:|---:|
  | default | 19.0s | 13.2s | 177.6s |
  | `HY_FAST_WRITEOUT=1 HY_SAVE_GIF=0` | 0.41s | skipped | 137.2s |

- **-40.4s wall.** Threading the PNG writes alone is 46x (19.0s -> 0.41s) and
  the bytes are identical (frame PCC 1.00000000, max abs pixel difference 0.0);
  the mp4 is byte-for-byte the same size in both modes. The GIF is the single
  most expensive artifact at 13.2s and ~25 MB, against 1.4 MB for the mp4 that
  serves the same purpose.
- Sequencing note: this should be done **before** VAE spatial sharding. It is a
  larger slice of the post-denoise window, took one wiring change rather than a
  1137-line refactor, and does not touch files the VAE work would conflict with.

### HY_VAE_TILE_PX is at its ceiling (swept; larger tiles OOM)

- `tile_overlap_factor` is **0.25** (`diffusers/models/autoencoders/
  autoencoder_kl_hunyuanvideo15.py:700`). With the production
  `HY_VAE_TILE_PX=128` and spatial compression 16, the latent tile is 8 with
  stride 6, giving 45 tiles at 480p (2 rounds over 32 chips, 1.81x the image
  area decoded) and 112 at 720p (4 rounds, 1.99x). So **45-50% of VAE decode
  work is redundant overlap or edge padding.**
- Larger tiles do NOT reduce that redundancy -- it is roughly scale-invariant
  at this overlap factor and slightly worsens with tile size as edge padding
  grows (1.81x at 128px vs 2.42x at 256px for 480p). What they would reduce is
  the round count: 192px+ is a single round at 480p, 256px a single round at
  720p.
- Swept at 480p/121f with everything else held constant:

  | `HY_VAE_TILE_PX` | result |
  |---|---|
  | **128** | **passes, 149.6s** |
  | 192 | OOM, 1.14 GB DRAM allocation |
  | 256 | OOM, 1.02 GB |
  | 384 | OOM, 1.13 GB |

- All three failures are the same `TT_FATAL: Out of Memory ... DRAM buffer
  across 8 banks`. The default is already at the ceiling: the VAE time-shares
  all 32 chips with a resident DiT, so the per-tile upsample intermediate is
  what bounds tile size. **The knob is closed; do not re-sweep it.**
- This strengthens the case for spatial sharding beyond the redundancy
  argument: fracturing the latent across the mesh means each chip holds a
  fraction of the intermediate rather than a whole tile's upsample buffer,
  which relieves the exact constraint causing these OOMs *and* removes the
  45-50% redundant compute.

### VAE spatial sharding at 121f: UNBLOCKED, and a REGRESSION (~72s slower)

- Freeing the DiT before decode (`HY_FREE_DIT_BEFORE_VAE=1`, below) plus
  non-materialising attention (`HY_VAE_ATTN_SDPA=1`) makes `HY_VAE_HW_SHARD=1`
  run at 121 frames for the first time. Both arms pass. But it is **slower**,
  not faster:

  | configuration (480p i2v 121f/4-step) | wall |
  |---|---:|
  | tiled + free DiT | **150.4s** |
  | hw-shard + free + `HY_VAE_ATTN_SDPA=1` | 232.2s |
  | hw-shard + free + SDPA + `HY_VAE_ATTN_DIST=1` | 222.8s |

- **~72s slower than the tiled path.** The earlier estimate in this report
  (~18-20s *faster*, from removing 45-50% redundant overlap decode) had the sign
  wrong. The redundancy is real, but the sharded path pays a per-convolution
  halo exchange across 32 chips to avoid it, and there are many causal convs in
  the decoder. On this Galaxy CCL is fully exposed -- every model program runs on
  CQ0 in order, so a collective never overlaps compute (see the CQ0 note in the
  MMRS entry) -- so that exchange is pure added latency.
- The tiled path decodes 1.81x the image area yet wins, because tiles are
  **independent**: it performs no cross-device communication during decode at
  all. Redundant local compute is cheaper here than distributed communication.
- Output at 121f vs tiled: frame PCC 0.99864060, max abs pixel difference 119,
  mean 1.33 -- the same blending-vs-exact difference described for the 13f
  comparison, not a defect.
- **Conclusion: keep the tiled VAE.** Spatial sharding is fully implemented,
  now demonstrably runnable at production length, and measurably worse. Revisit
  only if the fabric gains collective/compute overlap, which would change the
  trade that decides this.

### How it was unblocked (retained: the DiT free is independently useful)

- **Correction to an earlier entry in this report.** Spatial sharding is not
  "scaffolded but unwired" and is not a refactor waiting to be written. It is
  fully implemented behind **`HY_VAE_HW_SHARD=1`**, which self-derives an 8x4
  H/W `VaeHWParallelConfig` and its own `CCLManager` from the mesh. Device-side
  neighbour-pad (halo) exchange already exists in `CausalConv3d`
  (`ccl_manager.neighbor_pad_persistent_buffer`,
  `get_np_ping_pong_semaphore`, `canonicalize_replicated_shard_edges`), and all
  nine decoder classes carry `parallel_config`/`ccl_manager` plumbing --
  36 CCL references in `tt/vae_decoder.py`. The earlier assessment came from
  reading `tt/vae_spatial.py`'s docstring instead of the decoder itself.
- **It works.** 480p i2v 13 frames with `HY_VAE_HW_SHARD=1` passes.
- **It cannot run at 121 frames**, for a memory reason rather than a functional
  one: `Out of Memory ... bank size 4,272,341,376 B (allocated 4,242,684,416 B,
  free 29,656,960 B)`. DRAM is **99.3% consumed by the resident DiT**, and the
  allocation that fails is only 101 MB.
- Output versus the tiled path at 13f: **frame PCC 0.99897280**, max absolute
  pixel difference 95, mean 1.17. Not identical, and that is expected: the tiled
  path blends overlapping regions on host (averaging two independently decoded
  copies of every overlap pixel) while the sharded path computes each pixel once
  through halo exchange. The sharded result is plausibly the more accurate of
  the two; deciding that needs a host-VAE reference, which has not been run.
- **The unlock is freeing the DiT before VAE decode**, not writing the sharded
  path. VAE decode is the last stage of a one-shot generation, so the DiT is not
  needed again, and the prepared-weight cache makes a reload cheap (~4s) if a
  served path ever wants one. No deallocation path exists today: the only
  `release_*` call in `tt/pipeline.py` is `release_trace`, and even the 720p
  transformer swap does not free the 480p weights.
- Expected payoff once unblocked: the tiled path decodes 1.81x the image area at
  480p and 1.99x at 720p (overlap 0.25, 45 and 112 tiles, 2 and 4 rounds), so
  45-50% of VAE decode is redundant overlap or edge padding. Against ~40s of
  decode that is ~18-20s. Unmeasured at 121f -- the sharded path has never run
  at that length.

### VAE spatial sharding: original assessment (superseded above)

- `tt/vae_spatial.py` provides a complete, property-tested toolkit
  (`SpatialShardPlan`, `host_shard_with_halo`, `stitch_host_shards`,
  `block_causal_chunk_plan`, edge-fill and canonicalisation helpers), but
  `tt/vae_decoder.py` references it once and the module docstring states the
  production decoder is not on this contract yet.
- VAE decode plus media save is ~70s, about 24% of a 290s run, so this is the
  second-largest e2e target after weight preparation. Wiring it means moving
  every causal convolution and the mid-block attention onto the halo-exchanged
  fractured contract.

### SDPA chunk tuning (rejected: inside measurement noise, and changes output)

- After the heads-major layout change, a re-profile shows LAYOUT/MOVE collapsed
  from 757.1 ms (34.5%) to 111.6 ms (4.0%) and `ReshapeViewDeviceOperation`
  gone from the top ten. Scaling the remaining categories to 121f (attention
  quadratic, everything else linear or constant) puts **SDPA at roughly 65% of
  production device time**, so the exposed `HY_DIT_SDPA_Q_CHUNK` /
  `HY_DIT_SDPA_K_CHUNK` knobs were swept at 480p/121f/25 steps.
- k=512 is confirmed best: k=256 is slower and k=1024 fails outright. For q,
  a first pass suggested q=192 beat the shipped q=128 by 3-9%.
- The timing advantage is probably real *within a batch*: in one back-to-back
  batch q=160/192/224 measured 1.76 / 1.60 / 1.76 s/step. But a q=128 reference
  measured 1.67 in one batch and 1.76 in another, so **comparisons across
  batches drift by about 5% while back-to-back repeats inside one batch are
  reproducible to the hundredth of a second** (a 2x2 repeat of the heads-major
  A/B returned 2.46/2.46 and 1.67/1.67). Always A/B inside a single batch.
- Worse, q=192 changes the generated sample: frame PCC **0.835** against q=128
  with max absolute pixel difference 255. The frames are not garbage (per-frame
  mean/std match to within a fraction of a percent) -- the video has diverged
  into a different but equally plausible sample, because chunk size alters
  softmax accumulation order and a 25-step sampler amplifies bf16 rounding.
  That fails the >=0.99 bar the rest of this pipeline holds to, and it is not
  comparable to the bit-identical layout change.
- Conclusion: keep the measured default `(q=128, k=512)`. Larger q is also
  bounded above by L1 -- q=288 aborts with a circular-buffer clash.
- Methodological note for future A/Bs on this pipeline: back-to-back runs in
  one batch are highly reproducible, but the same configuration measured in a
  different batch drifts by roughly 5%. Compare arms inside one batch, never
  against a number from an earlier session. The heads-major result reproduced
  exactly on a 2x2 repeat (-32.1%).

### Elementwise fusion (measured, low yield -- deprioritised)

- Motivation: the 13f device profile puts ELEMENTWISE at 15.7% of device kernel
  time (Ternary 9.7% + BinaryNg 4.8%) at 100% core utilisation. Core count is
  therefore not the lever; fusion is, because it cuts bytes moved. This is the
  pattern behind PR #51400's expert-tail megakernel (10 ops -> 1).
- Measured at the real per-rank 480p/121f shapes (M=6176, C=2048, bf16):

  | chain | time |
  |---|---:|
  | `multiply(h, g)` | 180.6 us |
  | `add(h, a)` | 173.6 us |
  | `addcmul(h, a, g)` (residual update) | 245.9 us |
  | `layer_norm(h)` | 143.6 us |
  | `layer_norm -> addcmul` (real norm2 chain) | 383.5 us |
  | `layer_norm(h, weight, bias)` (modulation folded) | **207.4 us** |

- The chain is additive (383.5 vs 389.5 us for the parts), so each op does pay a
  full DRAM round-trip and fusion is genuinely available. Folding the AdaLN
  modulation into LayerNorm's own `weight`/`bias` is **1.85x** on that chain.
- **But it is not legal in the configuration we want.** AdaLN's `scale`/`shift`
  are `(Bp, 1, C)`, i.e. per batch row, while `layer_norm`'s `weight`/`bias` are
  per channel. Folding requires `Bp == 1`, which means the `separate` CFG policy
  -- and `separate` costs 7s of denoise versus `masked` (100 physical DiT runs
  instead of 50). The two optimisations are mutually exclusive and `masked`
  is worth more.
- Scale check: the whole hidden-stream elementwise chain is roughly 136 ms of a
  ~1.6 s 121f step (~8.5%), and the realistic per-fusion savings are 1-3% each,
  each needing a custom JiT `generic_op` kernel. Poor return for the effort.
- Important caveat this exposes about the profile itself: **the 13f capture
  overstates elementwise share for production 121f runs.** Elementwise scales
  linearly with tokens while attention scales quadratically, so at 49,408 tokens
  attention takes a much larger slice and every linear-scaling category shrinks
  proportionally. Use 13f ratios to rank *within* the linear ops, not to size
  them against attention.

### Sharded LayerNorm across the full core grid (rejected on measurement)

- Hypothesis, from tt-metal PR #51400's width-sharded RMSNorm win on gpt-oss:
  the device profile shows `LayerNormDeviceOperation` on a median of **25 of 120
  cores (20.8% utilisation)** for 143.5 ms (6.5% of device kernel time). The
  block gates its sharded path behind `Mt > 8` (256 tokens) while the latent
  stream is 6176 tokens per SP8 rank at 121f, so it always takes the
  interleaved fallback. That looked like idle parallelism.
- Measurement (`M x 2048` bf16, one config per fresh tensor, BLOCK_SHARDED
  since WIDTH_SHARDED gives every core all rows and cannot hold a long
  sequence):

  | M | tensor | interleaved | sharded kernel | + round-trip |
  |---:|---:|---:|---:|---:|
  | 256 | 1.0 MB | 29.1 us | 76.7 us | 134.7 us |
  | 800 (13f) | 3.3 MB | 45.5 us | 82.7 us | 135.6 us |
  | 1600 | 6.6 MB | 51.6 us | 242.6 us | 409.0 us |
  | 3200 | 13.1 MB | 78.4 us | 341.1 us | 539.4 us |
  | 6176 (480p prod) | 25.3 MB | 141.9 us | cannot allocate in L1 | -- |
  | 13952 (720p prod) | 57.1 MB | 286.5 us | cannot allocate in L1 | -- |

- Sharding loses by 2-5x at every shape where it can be built, and at both
  production shapes the activation cannot be L1-resident at all. **The existing
  `Mt > 8` gate is correct and should not be relaxed.**
- Why the hypothesis was wrong, and the general lesson: **core utilisation is a
  misleading metric for bandwidth-bound ops.** The interleaved LayerNorm moves
  50.6 MB in 141.9 us (357 GB/s) at 6176 and 114.2 MB in 286.5 us (399 GB/s) at
  13952 -- most of Blackhole's DRAM bandwidth. 25 cores is already enough cores
  to saturate memory; more cores only add sharding overhead plus two
  `to_memory_config` round-trips, which cost more than the kernel in every row
  above. PR #51400's win applies to single-token decode, where the tensor is
  tiny, L1-resident and latency-bound. Same op, opposite regime.
- Corollary: fusion remains a valid lever for bandwidth-bound elementwise work
  because it reduces bytes moved, unlike sharding which only adds cores.

### Registering Hunyuan matmul blockings (rejected on measurement)

- Hypothesis: no matmul in the block passes a `program_config`; only the ring
  SDPA has a tuned one. `get_matmul_config` is keyed on exact `(M, K, N)` per
  core grid, and Hunyuan's M values (6176 at 480p, 13952 at 720p per SP8 rank)
  match no key on the BH Galaxy 11x10 grid, so the helper would return its
  generic `(8, 8, 8)` fallback.
- Measurement: 40 constraint-derived blockings swept per shape on one chip
  against ttnn's default heuristic. The default won or tied everywhere:
  qkv_latent 0.90x, attn_out 1.03x, ffn_up 0.98x, ffn_down 1.00x.
- The four per-block GEMMs total 1456 us, i.e. ~157 ms/step at 121f
  (~8% of a 2020 ms step) at 100-114 TFLOP/s. Matmul is efficient and is not
  the bottleneck; registering Hunyuan shapes is not a win.
- Kernel constraints learned, for anyone repeating this: `per_core_M`/`per_core_N`
  are output tiles per core with `ceil(Mt/per_core_M) <= grid.y` and
  `ceil(Nt/per_core_N) <= grid.x`; `in0_block_w` must divide `Kt`; and
  `out_subblock_h * out_subblock_w <= 4` (not 8) because `fp32_dest_acc_en`
  halves the dest register budget.

### Hunyuan latent-stream matmul/reduce-scatter overlap

- Implementation: `HY_DIT_MMRS_OVERLAP=1` uses tt_dit's
  `minimal_matmul_strided_reduce_scatter_async` and MMRS configuration helper
  for the long latent stream's attention-output and FFN-down row-parallel
  projections. Each fused program streams completed matmul blocks directly
  into TP reduce-scatter, hiding two reductions per dual-stream block (108
  reductions across 54 blocks). The persistent TP all-gather remains separate
  to restore Hunyuan's replicated activation contract.
- Hunyuan schedule: latent fused MMRS, complete the independent context
  row-projection on the retained legacy path, then persistent-all-gather the
  latent result and add bias. Both FF-up branches are prepared before this
  row-projection schedule. Residual updates remain latent-then-context after
  both branches complete.
- Safety: opt-in only, restricted to Blackhole TP4 and bf16. The context stream
  deliberately does not use fused MMRS because Blackhole has a documented
  fused-op race at some short `M` shapes. The flag defaults off and the old
  matmul + persistent reduce-scatter + persistent all-gather path is unchanged.
- Evidence available without hardware: 15 configuration/dependency-order tests
  pass, including strict topology/dtype rejection and exact optimized/legacy
  event order. A TP4 block PCC parameter was added for the fused path.
- Timing and quality: not measured. A real-weight VAE job owned the hardware
  during implementation, so no device test, 121f microbenchmark, end-to-end
  delta, or generated-frame PCC is claimed. Do not enable by default until the
  TP4 block PCC passes, representative SP8 local shapes (6,176 and 13,952
  latent tokens) beat legacy per-block latency, and a matched 50-step generated
  frame comparison passes.

### Fused QKV projection and split

- Source pattern: tt_dit's `ColParallelLinear(chunks=3)` path used by Wan2.2
  calls `ttnn.experimental.minimal_matmul_split`, producing local Q/K/V tensors
  directly from one column-parallel projection.
- Hunyuan implementation:
  `_stubs/hunyuan_video15_transformer_block.py`. Both latent and conditioning
  QKV projections use the shared primitive; `HY_DIT_QKV_SPLIT=0` retains the
  previous fused matmul plus three slices per stream.
- Compatibility: exact. Hunyuan already stores each TP4 weight shard as
  `[q_local | k_local | v_local]`, each local chunk is 512 columns
  (4 heads x 128), and is tile aligned. No residual or collective layout changes.
- Expected operation-count effect: removes six tensor slices per block, or 324
  slice operations across 54 blocks. This is an operation-count statement, not
  a latency claim.
- Evidence: static configuration tests pass. A focused
  `test_transformer_block_qkv_split_pcc.py` compares the optimized path with
  both the legacy kernel path and the torch reference at PCC >= 0.99, and the
  hardware test passed all four assertions. The short real-weight A/B was not
  conclusive: default mixed-length wall time improved 1.9% but frame PCC was
  0.950332; equal-length wall time regressed 1.6% with PCC 0.996543. The split
  therefore defaults off pending 50-step generated-quality validation.

### Configurable ring-SDPA chunks

- Hunyuan's retained, previously measured setting remains `q=128, k=512`.
- Wan2.2's Blackhole SP8xTP4 setting (`q=288, k=512`) uses the same ring joint
  SDPA primitive and the same local head shape (4 heads x 128), so it is exposed
  as `HY_DIT_SDPA_PRESET=wan_bh_sp8tp4` for a controlled A/B.
- It is not the default: at 121 output frames Hunyuan's latent lengths are
  49,290 tokens at 480p and 111,600 at 720p. After 256-token SP padding these
  become 6,176 and 13,952 tokens per SP8 rank, unlike Wan's tuned query shapes;
  Hunyuan also reserves a CCL core row rather than Wan's CCL core column.
- Explicit Q/K overrides are accepted only when positive multiples of 32.
  Invalid presets and incompatible topology/parallel-shape requests fail during
  pipeline construction rather than silently selecting an unsafe config.

### BF16 DiT weights and tuned fused operations

- Source pattern: mature tt_dit pipelines use reduced-precision weights,
  fused linear epilogues, sharded normalization, and fused elementwise ops.
- Hunyuan implementation: `HY_DIT_BF16=1`, fused linear bias, combined
  projections, `addcmul` modulation/residuals, sharded LayerNorm where it fits,
  and fused joint SDPA.
- Status: production path.
- Benchmark delta: the capped two-layer profile improved from 6.43 ms to
  5.10 ms. These dispatch wins are not a substitute for 121-frame measurements.
- Correctness: per-module PCC tests, real-weight PCC, and the generated-video
  validations recorded in `README.md`.

### Step-invariant conditioning and RoPE residency

- Source pattern: Wan caches prompt conditioning, RoPE, and persistent latent
  buffers before its denoising loop.
- Hunyuan implementation: `denoise_trace_setup()` stores projected text/image
  conditioning and RoPE once. Resident latent/timestep buffers are updated in
  place for trace replay.
- Status: implemented for trace mode. RoPE shape constants are also cached in
  the SP eager path. This task changed the one-shot generation default to eager
  (`HY_TRACE=0`); trace remains available with `HY_TRACE=1`.
- New matched 480p I2V/121f evidence supersedes the earlier 13f conclusion:
  eager steady state was ~2.33 s/step and trace was ~2.21 s/step (~5% faster).
  Eager denoise/e2e were 1:58/5:19; trace was 1:59/5:24. Trace step 1 was
  10.78 s while steps 46–49 were 2.21 s, isolating ~8.5 s fixed startup
  overhead. With ~0.12 s saved per replay, the measured crossover was ~71
  steps, so the old trace path lost at the 50-step default despite its genuine
  steady-state benefit.
- Hardware result: Blackhole mesh capture rejects uncached programs, so the
  compile warmup cannot be removed. Capture itself also does not execute its
  output. The retained correct sequence is compile warmup, capture, explicit
  blocking execution. Intermediate attempts to use the compile output as step 1
  failed real generated-frame checks despite passing a small synthetic replay.
- Correctness: host execution-order regressions pass. A final mesh retest was
  stopped when a separate generation job acquired the Galaxy. Keep eager as the
  one-shot default until the conservative path passes real 13f and 121f gates.
- **Retest result (2026-08-06): replay is incorrect and trace is a blocker.**
  Capture is exact -- a 1-step traced generation is bit-identical to eager
  (PCC 1.000000) -- but 8 steps give aggregate PCC 0.237300 with per-frame PCC
  falling 0.9747, 0.8648, then negative (-0.0657 to -0.3122). Frame 0 survives
  only because i2v anchors it to the conditioning image. This is the same
  signature as the rejected heterogeneous trace (0.235647), so the two are one
  defect in per-step replay rather than two. Time-embed/patch-embed placement,
  the `traced=True` copy at `tt/pipeline.py:1135`, and the Euler update (shared
  with the exact eager path) are ruled out at source level.
- **Trace economics, measured warm:** eager ~ `1.72 + 1.016 * n` seconds and
  trace ~ `7.27 + 0.416 * n` over `n` steps, so break-even is ~9.3 steps -- well
  inside the 50-step default, and much better than the earlier ~71-step estimate
  taken from a cold trace. The break-even is never realized because
  `tests/e2e/test_stage2b_gen.py:288-289` releases the trace in a `finally`, so
  every run repays the ~7.27s capture. The pipeline already guards capture on
  `_trace_id is None`, so retaining it across runs needs no new mechanism and
  would drop break-even to ~3 steps. Do this only after replay is correct.
- Lifecycle finding: LTX retains one tracer per fixed shape and explicit
  persistent constants/latents across `generate()` calls; Wan exposes
  `release_traces()`, while Mochi releases on model unload. Hunyuan currently
  retains a trace inside one adapter and now exposes `release_trace()` to free
  trace-region memory and reset resident state explicitly. Reusing it across
  prompts would still be incorrect because only latent/timestep have in-place
  update hooks; raw Qwen/byT5/SigLIP conditioning may change shape and value.
  A served trace cache must key on all fixed tensor properties, update every
  conditioning buffer in place on a hit, and recapture on a mismatch. Once
  validated, pre-capture becomes service initialization rather than request
  latency; at the measured replay delta it would save ~6 s over 50 steps.

### Classifier-free-guidance batching

- Source pattern: Wan performs conditional and unconditional work in one
  on-device `combined_step`; Wan distill eliminates CFG only because its
  checkpoint has CFG baked in.
- Hunyuan implementation: `TTTransformer` batches diffusers' separate
  condition calls into one on-device batch.
- Status: production path.
- Benchmark delta: avoids separate DiT dispatch groups; included in current
  end-to-end numbers.
- Correctness: preserves the checkpoint's guider semantics. CFG cannot be
  removed for the base Hunyuan checkpoints without a compatible distilled
  model.

### Tile-sharded VAE decode

- Source pattern: Wan spatially shards its VAE across mesh height/width and
  uses full-time decoding on BH Galaxy.
- Hunyuan implementation: `tt/vae_decoder.py` spatially tiles the latent and
  batch-shards tiles so each chip decodes one tile per round.
- Pre-change overhead audit:
  - `TTVAEDecodeAdapter.__init__` constructs the complete TT decoder once.
    Every causal conv prepares and uploads its weight and bias, and every norm
    uploads gamma. This is setup cost per adapter construction, not per tile;
    unlike the reusable Wan adapter, it has no serialized prepared-weight cache
    or explicit load/deallocate lifecycle.
  - `_tiled_decode` enumerates and zero-pads edge tiles on host, concatenates
    them into a host batch, and sends one latent tile to each device per round.
    Decode is full-T for each spatial tile; there is no temporal host loop.
  - The old `_decode_batch_sharded` called `ttnn.to_torch` with a mesh composer
    inside every round. Each call gathered the round and synchronized/read it
    back before the next round could be appended on host.
  - `AttnBlock.__call__` rebuilt the block-causal `seq x seq` torch mask and
    uploaded it for every tile round even though padded tile shapes are uniform.
  - After readback, `_tiled_decode` cropped every edge tile on host, then
    `_blend_v` and `_blend_h` assigned one output row/column at a time in Python
    before host `torch.cat` stitching. Diffusers performs its final video
    postprocess after this already-host-resident tensor is returned.
- Status: production path with
  `HY_TT_VAE=1 HY_VAE_TILE=1 HY_VAE_TILE_PX=128`.
- Benchmark delta **[PROVISIONAL (cold?) -- see the caveat at the top]**: at
  480p/121f, replacing replicated tile decode reduced VAE time from about 5
  minutes to about 2 minutes and end-to-end time from 9:47 to 5:59 before the
  later DiT improvements.
- Correctness: bottom-up VAE block PCC tests plus coherent 121-frame generated
  outputs.
- Postprocessing follow-up: tile-round decoder outputs now concatenate on device
  and perform one final D2H rather than one synchronized D2H per round. The
  attention block caches its shape-specific causal mask instead of rebuilding
  and uploading it each round. Host blending remains, but its per-pixel Python
  loops are replaced with vectorized torch operations preserving the exact
  overlap weights and mutation order.
- Fallbacks: `HY_VAE_LEGACY_TILE_READBACK=1` restores per-round D2H and
  `HY_VAE_LEGACY_TILE_BLEND=1` restores scalar blending.
- Evidence: 40 static property cases match the legacy stitched output
  bit-for-bit across odd edge tiles and overlap extents of zero, one, normal,
  and larger than an edge tile. One boundary-weight case and four device/round
  ordering cases also pass (45 static tests total). The focused 2-chip test
  verified device residency and passed tile-order restoration at PCC
  0.999998593. The representative VAE timing A/B was blocked by a later
  concurrent generation job, so no latency delta is claimed.
- Halo-sharding follow-up:
  - `tt/vae_spatial.py` defines equal-storage H/W partitions, replicate padding
    for uneven bottom/right edges, per-layer halo oracles, logical crop/scale
    metadata, and TTNN tile blend/crop/stitch.
  - 480p uses a 30x53 latent grid and 720p uses 45x80 at 16x compression. Both
    are covered on an 8x4 partition. T remains local on every rank; two
    asymmetric temporal upsample stages preserve 31 latent frames -> 121
    output frames.
  - A 3x3 convolution requires a one-cell H/W neighbor exchange before every
    layer. A one-time input halo is insufficient through the residual/upsample
    graph. Hunyuan boundary padding is replicate, so the divisible-storage pad
    also replicates the last logical row/column rather than using Wan's
    zero-masked edge convention.
  - Mid-block attention cannot use a finite halo: each query attends all
    spatial tokens in its current and causal-prior frames. It must all-gather
    H/W, run block-causal attention, then mesh-partition H/W again.
  - The opt-in `HY_VAE_DEVICE_STITCH=1` path performs blend, edge crop, and
    concatenation in TTNN when the decoded tile batch is replicated (currently
    one-device only), followed by one D2H. Multi-device tile-index sharding
    safely retains the existing one-readback host stitch.
  - Evidence added in `tests/pcc/test_vae_spatial_sharding.py`: 17 new host
    cases pass for uneven partition/crop, radius-one convolution halo
    equivalence, and 121-frame scale metadata. The focused one-chip TTNN
    stitch gate passes at PCC 0.9999977569. No timing was measured.
  - The decoder-wide graph conversion is implemented behind
    `HY_VAE_HW_SHARD=1`. `VaeHWParallelConfig` and `CCLManager` reach every
    convolution/residual/upsample/attention block; 3x3 convolutions use
    `neighbor_pad_persistent_buffer`; logical H/W doubles after DCAE upsample;
    attention gathers/crops H/W and repartitions after block-causal attention;
    and `fast_device_to_host` composes the final shards with one D2H.
  - Uneven replicate padding needs stronger handling than Wan's zero-masked
    convention. A four-chip mini-decoder initially reached only PCC 0.89785:
    storage-only edge cells had diverged after residual/upsample operations and
    polluted the next halo. `canonicalize_replicated_shard_edges` now restores
    the complete logical result after each spatial convolution and upsample
    without CCL: cached masks use the same 2D mesh fracture as the activation,
    and select rank-local replicated H/W tails only on final uneven ranks.
    H-before-W replacement makes the bottom-right corner exact.
  - Hardware evidence on an idle Galaxy: 1x2, 2x2, and full 8x4 random-weight
    end-to-end mini-decoder equivalence pass at PCC 1.0. Four-chip layer gates for
    radius-one halo convolution, uneven temporal/spatial upsample, and
    attention gather/repartition each pass at PCC 1.0. Device-native edge fill
    passes on 8x4 for the exact 30x53 and 45x80 logical grids with zero gathers
    (PCC >= 0.999). Host edge/corner properties cover every 8x4 rank at stages
    0..4; the full host VAE regression passes 68 cases.
  - Communication measurement on the random decoder: graph all-gathers changed
    from 11 to 1 on 1x2 and 22 to 2 on 2x2/8x4. The remaining count is exactly
    one per active H/W attention axis; convolutions and upsample edge repair
    issue none. Final transfer remains one D2H. Rank masks add one cached H2D
    per uneven axis/scale: 10 uploads / about 2.44 MB BF16 for 480p and 5 /
    about 2.62 MB for 720p over stages 0..4; all layers at a scale reuse them.
    Cached focused-test call duration changed 0.31s -> 0.30s (1x2) and 0.25s ->
    0.21s (2x2); these tiny random graphs are correctness/communication
    evidence, not production timing.
  - Real-weight offline gate: the cached 480p I2V VAE (shared by 720p) passed on
    8x4 for the smallest supported `(T,H,W)=(2,8,4)` latent. H/W-sharded and
    replicated output each matched the host at PCC 0.999980645. First decode was
    35.41s H/W versus 25.81s replicated **[PROVISIONAL (cold?)]**; the H/W graph
    issued two gathers. A one-frame latent is below the current
    temporal-upsample input domain and reaches a zero-length branch before PCC.
  - Matched 121-frame 480p VAE-only measurement **[PROVISIONAL (cold?) -- the
    92.97s figure was later re-measured at 44.23s; see the caveat at the top,
    and do not quote the 2.48x verdict]**: H/W `(31,30,53)` took 92.97s
    plus 77.6ms final D2H, versus 37.55s including readback/stitch for the
    tile/single-readback fallback. H/W was 2.48x slower. Cross-path video PCC was
    0.998960435 and final-frame PCC was 0.999074800. The fallback's 15 padded
    16x16 tiles represent 2.42x the logical latent area but run in one 32-device
    round. The H/W graph retained exactly two global gathers and one final D2H.
    The correctness half of this result (the two PCC figures and the gather/D2H
    counts) does not depend on cache state and stands.
  - Synchronized post-decode DRAM allocation was 8.61 GiB H/W versus 1.80 GiB
    tile. These are observed checkpoints, not allocator peak values. Host peak
    RSS was 31.58 GiB H/W and 24.47 GiB tile. A full CPU 480p reference was
    stopped after exceeding 23 minutes; small real-weight host PCC and matched
    full-video device-path PCC are reported instead.
  - The 121-frame 720p H/W gate did not fit. Mid-block attention requested a
    24,916,262,912-byte matmul buffer; each bank needed 3,114,532,864 bytes while
    only 579,611,264 bytes were free (largest contiguous block 568,861,056).
    No reset/retry, 720p timing, or end-to-end generation followed this VAE-only
    failure. **Resolved (2026-08-06):** with chunked mid-block attention the
    same gate needs 0.74 GiB rather than 23.2 GiB, and 720p chunked decode now
    completes at output `[1, 3, 121, 720, 1280]`.
  - This path decodes one full latent partition rather than repeatedly decoding
    overlapping tiles and performs one final D2H. The validated tile,
    single-readback, one-device device-stitch, and legacy paths remain the
    production defaults. `HY_VAE_HW_SHARD` remains opt-in: real-weight
    correctness passes at small scale, but 480p regresses and 720p global
    attention OOMs. Chunked or distributed mid-block attention is required
    before another performance qualification.
  - Spatial-sharding verdict **[the latency half is PROVISIONAL and probably
    wrong]**: the H/W path is correct, with cross-path video PCC 0.998960 and
    final-frame PCC 0.999075. It was recorded as 2.48x slower at 480p (92.97s
    versus 37.55s), but 480p H/W was later measured at 44.23s and the 37.55s
    tile figure is itself suspect (720p tile measured 13.07s, which cannot be
    right at a higher resolution). Keep `HY_VAE_HW_SHARD` off by default -- that
    default is unchanged -- but treat the speed comparison as unmeasured until
    the all-warm table lands.

### Mid-block attention memory, chunking, and distribution

- Memory derivation. The mid-block is a single-head attention over the whole
  spatiotemporal extent at `C = 1024`, with a block-causal mask letting a query
  in latent frame `f` read every token of frames `0..f`. There is exactly one
  such block. The monolithic form allocates three `S_pad x S_pad` bf16 tensors
  (cached mask, raw scores, scaled/softmaxed scores) where `S = T*H*W`. At 720p
  `S = 111,600` and `S_pad = 111,616`, so each is 23.21 GiB, which is exactly
  the observed 24,916,262,912-byte request. At 480p the same term is 4.53 GiB
  and accounts for 4.53 of the 8.61 GiB observed still resident after decode,
  because `_mask_cache` intentionally keeps the mask alive across tile rounds.
- Chunked attention (`HY_VAE_ATTN_CHUNK=<tokens>`, default 0 = off). Query rows
  share a key prefix exactly when they share a frame, so a block confined to one
  frame slices K/V to its causal prefix and needs no mask at all. Requests are
  clamped to `H*W` because a block may never cross a frame; 0 means disabled,
  not frame-granular.

  | resolution | monolithic (each of three) | chunk 1024 peak | reduction |
  |---|---:|---:|---:|
  | 720p (31,45,80) | 23.21 GiB | 218 MiB | 109x |
  | 480p (31,30,53) | 4.53 GiB | 96 MiB | 48x |

- Tile sizing verdict: attention is *not* what caps tile size. At the production
  128px tile the decoder tail is 484 MiB against 22.5 MiB of attention, and
  attention only overtakes the tail past 512px. This supersedes any claim that
  chunked attention would unlock larger tiles.
- Distributed attention (`HY_VAE_ATTN_DIST=1`, default off, requires
  `HY_VAE_HW_SHARD=1`). `RMSNorm` reduces only over channels and `to_q` is a
  kernel-1 `CausalConv3d`, whose `t_front` and `pad_hw` are both zero, so it
  performs no replicate padding and issues no neighbor exchange. Both are
  pointwise in H/W, so Q never needs the all-gather; only K and V do. Each rank
  normalizes and projects its own shard, all-gathers K and V, computes exactly
  the query rows it stores, and returns them fractured, which removes the
  post-attention `mesh_partition`.
- Work division measured from the plans: on 8x4 the per-rank score-element count
  drops 28.4x at 480p (1,253,937,600 to 44,163,840) and 30.0x at 720p
  (6,428,160,000 to 214,272,000). The shortfall from 32x is equal-storage
  padding overhead, 1.127x and 1.067x respectively. With no chunk request at
  all, one frame of rank-local queries is a 6.0 MiB block at 480p and 27.2 MiB
  at 720p.
- Replicate padding needs no explicit output repair under distribution. Given
  the shared keys and values, every remaining stage is a per-position map, so a
  padded row holding a copy of the last logical row produces a copy of that
  row's output. The block canonicalizes its input first so it does not depend on
  upstream hygiene, and padded K/V rows are still cropped before flattening
  because duplicated keys would reweight the softmax. Partitions leaving a rank
  with no logical row remain rejected, unchanged; both production grids are
  legal on 8x4 (30 rows to 4-row shards with a 2-row tail, 45 rows to 6-row
  shards with a 3-row tail).
- Evidence without hardware: 258 host cases pass across the chunking and
  distribution suites, including 24 rank-decomposition equivalence cases
  agreeing to 1e-12 in float64 over even, H-only, W-only, and both-uneven
  partitions plus the real 30x53 grid on a simulated 8x4 mesh, and a
  frozen-K/V case that isolates the query path to prove a remote rank's rows
  cannot reach this rank's query.
- **Chunking is now hardware-validated (2026-08-06).** All 7 device cases in
  `test_vae_attention_chunking.py` pass: PCC 0.9999985 at query chunks 1, 7 and
  32, 0.9999988 at 512, and 0.9999984 sharded. Block peak DRAM fell from
  9,207,808 to 1,228,800 bytes. The pre-existing VAE suite passes unchanged with
  chunking forced on (38 tests). The 720p allocation gate fell from 23.2 GiB to
  0.74 GiB and 720p chunked decode now completes at `[1, 3, 121, 720, 1280]`.
  A matched 480p A/B gave cross-path video PCC 0.9999899 while reclaiming
  4,753 MiB of device memory, with host peak RSS dropping 32.3 to 9.4 GiB. No
  latency is quoted; the runs mix cold and warm decodes (see the caveat at the
  top).
- **Distribution is not yet hardware-validated, but is no longer blocked.** The
  device cases failed on their own parameterization -- they requested 5 latent
  rows on a 4-rank H axis, leaving the final rank holding only padding, which
  the edge-fill contract rejects by design. The cases now pair each latent
  geometry with a mesh that can partition it, cover even / H-only-uneven /
  W-only-uneven / both-uneven partitions and the real 30x53 and 45x80 grids on
  8x4, and are guarded on host by `test_every_hardware_case_partitions_legally`
  so an illegal request cannot reach hardware again. The rejection message now
  names the offending H, the per-rank row count, the rank count and the smallest
  legal H for that mesh.

### Flash SDPA for the mid-block (implemented, hardware-validated)

- After chunking, each block is a plain non-causal mask-free attention over a
  key prefix, structurally identical to Wan's VAE usage of
  `ttnn.transformer.scaled_dot_product_attention(is_causal=False, ...)`. This
  would take peak from `O(q_chunk * S)` to `O(q_chunk * k_chunk)`.
- Static legality verdict: the geometry is legal. `sdpa_device_operation.cpp`
  imposes no head-dim or num-heads limit. It requires TILE layout, bf16/bfp8/bfp4,
  interleaved storage, `nqh >= nkv and nqh % nkv == 0` (trivial at
  `num_heads = 1`), tile-aligned chunk sizes, and no padding on the batch,
  num_heads, or head_dim axes. `head_dim = 1024` is exactly 32 tiles so it is
  unpadded; only the sequence axis is padded and that is the one axis SDPA
  explicitly allows to be padded. The default scale is `head_dim ** -0.5`, which
  equals the block's own scale. A ragged `kv_stop` is handled by the program
  factory's generated padding mask (`use_padded_mask` when `padded_Sk != Sk`).
- Reshaping 1024 channels into multiple pseudo-heads would **not** be valid.
  Splitting into `n` heads computes `n` independent softmaxes over partial dot
  products and concatenates them, which is a different function from one softmax
  over the full 1024-channel dot product. It is not a legal workaround, and it
  is not needed since the single-head geometry is already legal.
- The real constraint is L1 circular-buffer capacity, which is a runtime
  allocation failure rather than a validate assert. The factory sizes
  `k_tiles` and `v_tiles` at `Sk_chunk_t * DHt * 2` each, so at `DHt = 32` the
  key chunk dominates. Wan's `q=32, k=256` preset costs 888 KiB at its own
  `head_dim = 384` (base_dim 96 x dim_mult 4) but 2328 KiB at Hunyuan's 1024,
  against a conservative 1024 KiB budget of Blackhole's 1.5 MiB per-core L1.
  The implementation therefore derives the chunk from the head dim: the largest
  legal key chunk at `head_dim = 1024, q_chunk = 32` is 64, costing 780 KiB.
- Status: implemented behind `HY_VAE_ATTN_SDPA=1`, default off, with the CB
  budget model unit-tested against the C++ arithmetic.
- **Hardware answer (2026-08-06): the geometry works.**
  `test_device_flash_sdpa_matches_the_matmul_blocks` passes at PCC 0.9999913 at
  `head_dim = 1024` with the derived `k_chunk = 64`, and at 0.9999936 at
  `head_dim = 64` with `k_chunk = 1760`. The anticipated circular-buffer or
  kernel-geometry failure did not occur, and the L1 budget model that derived
  `k_chunk = 64` from the head dim was right. This was the only open question a
  static reading of the kernel could not settle. It stays opt-in until it is
  measured inside a real decode.

### Dynamic 720p transformer loading

- Source pattern: Wan dynamically loads mutually exclusive experts/components
  when DRAM is constrained.
- Hunyuan implementation: the 720p path frees the 480p transformer before
  loading only the cached 720p transformer shards; VAE and encoders are reused.
- Status: production path. It reduces host peak memory and avoids loading a
  second complete pipeline, but is not a denoise latency optimization.
- Correctness: all four task/resolution combinations have generated coherent
  121-frame output.

### Optional on-device Qwen2.5-VL

- Source pattern: tt_dit Qwen2.5-VL encoder used by Qwen-Image.
- Hunyuan implementation: `tt/qwen_encoder.py`, on a dedicated submesh when
  available or the shared full mesh with `HY_TT_QWEN_SHARED=1`.
- Status: experimental and off by default for the 8x4 one-shot path.
- Benchmark delta: shared-mesh Qwen made a measured one-shot run roughly one
  minute slower (5:59 host versus 6:58 shared TT) because weight upload and
  first compile do not amortize.
- Correctness: valid-token embedding PCC improved to about 0.9998 with an fp32
  attention core. The padding issue is now handled at the DiT boundary rather
  than by pretending zero padding is mask-equivalent: mixed-length CFG and
  multi-prompt rows run independently at their exact valid Qwen/byT5 lengths by
  default (`HY_CFG_PADDING_POLICY=separate`). This is required because the fused
  joint-attention kernel has no key mask and its learned projections make even
  zero padding non-neutral. `legacy` restores longest-row batching explicitly;
  `error` rejects it. Mixed-length eager and resident execution now use one
  exact-length condition/batch-row slot each. Trace mode owns one reusable trace
  per slot; all distinct programs are compiled before any trace becomes active,
  and all trace regions are released explicitly.
- Lifecycle/setup: generation pre-encodes text before TT DiT construction.
  Shared-mesh Qwen can therefore load, encode, and deallocate on the same 8x4
  mesh before DiT weights load, with no overlapping submesh. Qwen now uses
  `cache.load_model`; `TT_DIT_CACHE_DIR` enables serialized prepared weights.
  `HY_PROMPT_CACHE=1` persists the complete positive/negative Qwen+byT5 tensor
  tuple and skips both encoders on a warm repeated prompt.
- Timing split: the 5:59 versus 6:58 result is a cold one-shot comparison.
  No warm served Qwen or prompt-cache timing has been measured, so no warm
  speedup is claimed and host Qwen remains the default.

### Optional on-device SigLIP for I2V

- Source pattern: reuse the existing SigLIP so400m transformer blocks from
  `models/experimental/pi0/tt/ttnn_siglip.py`.
- Hunyuan implementation: `tt/siglip_encoder.py`; host patch embedding and
  post-LayerNorm surround a 27-layer on-device transformer.
- Status: experimental. The focused one-device adapter works, but a 32-chip
  SP=8 end-to-end attempt hung while constructing an overlapping 1x1 submesh:
  the resident DiT already owns all chips. The generation integration now only
  uses an explicitly reserved, non-overlapping 1x1 chip and otherwise falls
  back to host. It does not share a mesh context with DiT, VAE, or Qwen. The
  flag is not a production default on the primary layout.
- Benchmark delta: not yet established.
- Correctness: focused encoder PCC is about 0.9947; the adapter performs a
  first-call host comparison and rejects PCC below its threshold.

### Opt-in device-resident FlowMatch Euler and latent

- Hunyuan implementation: reusable `TTTransformerAdapter` and
  `DeviceResidentFlowMatchScheduler` in `tt/pipeline.py`, enabled by
  `HY_DEVICE_RESIDENT_DENOISE=1` in the generation path.
- The initial latent is sequence-sharded once. Conditional/unconditional model
  output stays on device, CFG is applied in TTNN, and the non-stochastic Euler
  update follows installed diffusers 0.38.0 schedule/indexing and dtype
  semantics. The updated latent is reused by the next eager or trace step; only
  the final latent is gathered for VAE handoff.
- Safety: opt-in and SP-only. Dynamic CFG start/stop, guidance rescale,
  stochastic sampling, and per-token timesteps are rejected rather than
  approximated. The default generation path is unchanged.
- Correctness: 25 host contract/property/guard cases pass, including unequal
  positive/negative lengths, condition-major multi-row splitting, resident
  eager/trace execution counts, shifts 5 and 9, two latent shapes,
  native/original CFG, and CFG on/off. Four focused Blackhole cases pass for
  actual TTNN CFG/Euler with batched and separate prediction tensors. The
  scheduler oracle matches host diffusers exactly; the device kernel tolerance
  is `rtol=atol=2e-2`.
- Cached real-weight equal-length 13f/4-step resident-only passed at frame PCC
  0.998612 versus eager, but wall regressed from 226.23s to 238.54s. Real mixed
  CFG (`HY_CFG_PADDING_POLICY=separate`) now passes resident-only at 250.30s and
  fused-QKV + resident at 239.10s, versus matched eager baselines of 239.29s and
  234.71s. The mixed path makes eight physical DiT executions (two exact-length
  conditions per step) while retaining latent/CFG/Euler on device. Outputs and
  logs are under `validation_20260805_mixedcfg_fix/`.
- Mixed-shape trace is rejected in production. After an authorized Galaxy
  reset, the focused 4-device equal-shape trace test passed first/replay PCC
  0.999979/0.999994. Two real mixed-CFG 13f/4-step designs then completed:
  per-shape traces took 260.89s and a single grouped heterogeneous trace took
  243.32s, each with eight physical DiT executions. Both emitted TT's warning
  that device-buffer allocation while a trace is active is unsafe, and both
  produced the same incorrect output: aggregate 13-frame PCC was 0.235647
  versus fused-QKV resident eager, 0.234625 versus fused-QKV eager, and
  0.229407 versus legacy eager (minimum per-frame PCC was negative). Artifacts
  are under `validation_20260805_mixedcfg_fix/qkv_resident_trace_fixed/` and
  `qkv_resident_grouped_trace/`. The adapter now fails closed with instructions
  to use `HY_TRACE=0`; it never switches to legacy padding. No 121-frame run was
  performed because correctness failed.

## Investigated but not applied

### Patterns from other mature tt_dit pipelines

- Stable Diffusion 3.5 Large uses independent tracers for CLIP, T5, each DiT
  submesh, and VAE, with `SectionStart`/`SectionEnd` callbacks around total,
  encoder, every denoise step, and VAE. Hunyuan already has a dedicated
  trace/2CQ implementation, but should adopt the common event API before the
  next full benchmark so setup, encode, denoise, VAE, and save time are reported
  consistently.
- Qwen-Image dynamically loads/offloads its Qwen encoder and VAE according to
  mesh capacity. That is appropriate when components cannot coexist. For
  Hunyuan's one-shot SP=8 path, CPU Qwen is faster and the resident DiT plus
  tile-sharded VAE fit, so dynamic device loading would add startup transfers.
- LTX/LTX-distilled keeps per-shape trace objects, preallocates persistent
  stage constants and latent buffers, and registers co-resident exclusions for
  DiT, VAE, upsampler, and audio stages. Hunyuan already applies the reusable
  fixed-shape resident-buffer pieces; its single DiT has no mutually exclusive
  expert/stage weights to unload.
- Mochi and SD3.5 expose independent `traced`, `encoder_traced`, and
  `vae_traced` controls. Hunyuan's `HY_TRACE` controls only DiT denoising and
  remains opt-in because that stage is compute/CCL bound. Encoder/VAE tracing
  should only be added after repeat-request measurements show compile
  amortization.

### Wan distillation and CFG elimination

- Wan's 4-step lightx2v graph is architecturally unchanged and gains roughly
  `10x fewer steps * 2x no CFG`; measured total speedups are 6.84x at 480p and
  7.34x at 720p.
- This is checkpoint semantics, not a runtime optimization. No compatible
  HunyuanVideo 1.5 distilled weights or scheduler contract is present, so
  reducing to four steps or disabling CFG would regress quality.

### Reusing tt_dit T5/UMT5 for byT5

- The Hunyuan byT5 checkpoint is a standard 12-layer gated-GELU T5 encoder
  (`d_model=1472`, `d_ff=3584`, 6 heads, 256 tokens), so its math and state-dict
  naming map strictly through `models/tt_dit/encoders/t5`.
- The shared blocker is fixed: `T5Config.attention_inner_dim` is
  `num_heads * d_kv`; q/k/v are column-parallel `d_model -> inner_dim`, local
  heads retain `d_kv`, and the established gathered column-parallel output
  projection is now `inner_dim -> d_model`. Residual/RMSNorm and feed-forward
  widths remain unchanged. Existing equal-width T5/UMT5 therefore preserve
  their execution and prepared-cache layout with `inner_dim == d_model`, not a
  fork.
- Placement conclusion: byT5 stays on host by construction for a committed
  32-chip run, and this is a structural result rather than a pending
  measurement. Tensor parallelism has to divide both `num_heads` (6) and
  `d_model` (1472 = 2^6 * 23), so only 1- and 2-device meshes are legal; neither
  Galaxy axis (8 or 4) can express either factor, so byT5 can never share the
  8x4 DiT mesh, and a committed run leaves no disjoint mesh to give it. Host
  byT5 costs a fraction of a second (12 layers x at most 256 tokens), so nothing
  is lost. The TP1/TP2 device path is only useful to a deployment that reserves
  chips outside the DiT mesh.
- The primary 8x4 DiT consumes the whole mesh. The existing T5 port tensor
  parallelizes attention heads, but tensor parallelism has to divide both
  `num_heads` (6) and `d_model` (1472 = 2^6 * 23), so the only legal factors are
  1 and 2 and neither primary mesh axis (8 or 4) can express them. The adapter
  consequently accepts only a dedicated, genuinely disjoint 1-device (TP1) or
  1x2/2x1 (TP2) mesh, supplied by the caller via `HY_BYT5_SUBMESH`. It never
  creates a submesh itself. It strictly validates the real Hunyuan config, loads
  the standard T5 state dict through `cache.load_model`, pre-encodes, and
  deallocates before DiT loading.
- The byte-level embedding table (1510 x 1472) is replicated, never sharded:
  1510 rows are not tile aligned and `ttnn.embedding` reads the table in
  ROW_MAJOR layout. The relative-position bias lives only in layer 0 (the other
  eleven layers reuse it) and its 32 x 6 table is fractured over heads with the
  rest of attention.
- Host-side contract: the diffusers pipeline always tokenizes to a fixed
  256-token window, but `plan_byt5_inputs` pads any shorter window up to a tile
  and masks the synthesized tail, which is exact because T5's relative bias for
  `(i, j)` depends only on `j - i`. `finalize_byt5_output` crops the padding back
  off and, under `HY_BYT5_ZERO_PAD=1` (default), zeroes masked positions so any
  padding surviving `_trim_to_valid` is neutral rather than arbitrary — the same
  mitigation the Qwen adapter applies for the same maskless fused joint-attention
  kernel.
- Evidence collected without hardware: 33 host-only tests pass
  (`tests/pcc/test_byt5_encoder_host.py`), including a strict key/shape match
  against the real `text_encoder_2/model.safetensors`, TP1/TP2 shard-shape
  verification through the production `load_torch_state_dict` path, exact
  agreement of the port's relative-position bucketing with HuggingFace, and an
  op-for-op torch mirror of the TT dataflow (independent q/k/v width, TP fracture
  and gather) matching `T5EncoderModel` to 2e-4 at both TP1 and TP2.
- **Hardware PCC gate: all 5 cases pass (2026-08-06).** TP(1,1) 0.999935,
  TP(1,2) 0.999938, full sequence without zero-padding 0.999931, batched-row
  consistency ~1.000, and the adapter's own fail-closed first-call self-check
  passes. All outputs are finite and non-zero.
- **Contract decision: `tie_word_embeddings` is no longer enforced.** The gate
  initially failed closed on that field and nothing else. The checkpoint's own
  `text_encoder_2/config.json` stores `false` while `T5Config.from_pretrained`
  returns `True` -- HuggingFace does not round-trip it, so the strict check was
  rejecting the exact checkpoint this port targets. The field ties an LM head to
  the input embedding table and `T5EncoderModel` has no LM head, so no value of
  it can change an encoder activation; it is the one field in the contract that
  provably cannot affect numerics. It was removed from `_EXPECTED_CONFIG` and is
  instead reported through `ByT5Support.reason` when the parsed value disagrees
  with the checkpoint, so the quirk stays visible rather than silently dropped.
  Everything else stays fail-closed, and the risk this check might have covered
  -- a checkpoint that genuinely carries an LM head -- is already caught by the
  strict unexpected-key check in `load_torch_state_dict`. Host coverage:
  `test_tie_word_embeddings_is_reported_but_never_rejected` and
  `test_the_parsed_real_config_is_accepted_even_though_hf_rewrites_a_field`,
  which reproduces the `false` -> `True` rewrite on the real snapshot without a
  device.
- **Padding neutralization is not load bearing.** The full-sequence case passes
  with `zero_padding=False`, so `HY_BYT5_ZERO_PAD=1` is defensive rather than
  required. Keep the default on, but it is no longer a correctness dependency.
- **The `(mask - 1) * inf -> 0 * inf = NaN` hazard does not manifest** on
  device. It was a legitimate concern in the shared T5 stack's additive-mask
  expression; hardware shows it does not occur at these shapes, which is why the
  host mirror's large-finite-negative stand-in is a modelling convenience rather
  than a divergence.
- Host byT5 stays the default (`HY_TT_BYT5` off) for a committed 32-chip run --
  that is a placement conclusion, not a correctness one, and it is unchanged.
  Generated-video quality and cold/warm timing on a reserved TP1/TP2 mesh are
  still unmeasured.

### CCL topology and link changes

- Wan uses Ring with two links on BH Galaxy. Hunyuan's joint ring SDPA already
  supplies the sequence-parallel scaling; its remaining TP collectives use the
  physically supported linear path.
- Prior experiments found Ring/extra links incompatible with this Galaxy
  `FABRIC_1D` topology. No default change is justified without a topology-aware
  A/B result.

### Wan expert dynamic loading, weight all-gather overlap, and FSDP

- Dynamic loading is valuable for Wan's two mutually exclusive 14B experts.
  Hunyuan has one 54-block expert, so unloading/reloading it within one video
  has no analogue.
- Wan's FSDP weight all-gather and overlap machinery targets weights fractured
  across another mesh dimension. Hunyuan's 8x4 layout already uses both axes
  for SP and TP, and its weights fit resident. Porting FSDP would add collectives
  without freeing a stage that needs co-residency.
- The reusable direction is Wan's overlapped matmul/reduce-scatter kernels in
  Hunyuan's block, but that requires per-op profiling and PCC validation rather
  than a pipeline-level switch.

### Wan collective/linear patterns rejected for the SP8 x TP4 production path

- **All-gather-matmul overlap:** Wan keeps the residual width fractured on TP
  and feeds that layout into `ColParallelLinear`, allowing AG+matmul overlap.
  Hunyuan's dual residual streams are replicated on TP after every row
  projection; AdaLN, residual gates, and both stream-specific LayerNorms consume
  that replicated width. Using AGMM without converting the complete block to a
  distributed-width residual contract would gather already-replicated data and
  be incorrect.
- **Matmul-reduce-scatter overlap:** the mature fused MMRS path returns a
  width-fractured residual and is selected by Wan only for its supported Ring
  topology. Hunyuan's validated Galaxy path uses physical `FABRIC_1D` Linear
  TP collectives and immediately all-gathers after reduce-scatter to restore
  the replicated dual-stream residual. The fused path therefore needs both a
  topology qualification and the larger distributed-residual refactor; it is
  not enabled here.
- **FSDP linears/weight overlap:** both Galaxy mesh axes are already assigned to
  SP8 and TP4, all Hunyuan block weights remain resident, and there are no
  mutually exclusive experts. FSDP would add weight collectives without
  unlocking memory or overlap needed by this checkpoint.
- **Wan self/cross-attention decomposition:** Wan runs spatial self-attention
  followed by text cross-attention. Hunyuan performs one dual-stream joint
  attention in which latent and conditioning queries both attend the combined
  key/value sequence and both outputs update persistent residual streams.
  Reusing Wan's self-attention dummy-joint trick or cross-SDPA path would change
  model semantics.
- **Experimental ring SDPA:** Wan enables `exp_ring_joint_*` only at SP32xTP4.
  Hunyuan targets SP8xTP4, so that qualification does not transfer.
- **Wan shape-specific matmul blockings:** Wan's registered shapes are based on
  width 5120 and FFN 13824. Hunyuan uses width 2048, FFN 8192, local TP widths
  512/2048, and per-SP sequence lengths 6,176/13,952. Reusing those blockings
  would be shape-mismatched; Hunyuan-specific values require a sweep and device
  profiler evidence before registration.

### Quantization beyond BF16

- No Hunyuan checkpoint-specific BFP8/INT8 quality evidence exists. Video
  diffusion is sensitive to accumulated block error, as the Qwen conditioning
  investigation demonstrates. Keep BF16 weights and validated accumulation
  settings until block and generated-quality A/B results support lower
  precision.

## Prioritized remaining work

1. **Produce the all-warm VAE timing table** and replace every figure marked
   `PROVISIONAL (cold?)` in this document. Until that exists, no VAE path
   selection can be justified on latency. Separate cold (first decode, includes
   compile and upload) from warm explicitly, at 480p and 720p, for the tile,
   H/W-sharded, and chunked paths.
2. **Fix traced per-step replay.** Capture is exact and replay is not (1 step
   PCC 1.000000, 8 steps 0.237300). This blocks `HY_TRACE=1` entirely, including
   with `HY_CFG_PADDING_POLICY=masked`. Placement of time/patch embed, the
   `traced=True` copy at `tt/pipeline.py:1135`, and the shared Euler update are
   already excluded. Once correct, retain the trace across generations instead
   of releasing it at `tests/e2e/test_stage2b_gen.py:288-289`, which moves
   break-even from ~9.3 steps to ~3.
3. On idle hardware, run the now-unblocked distributed mid-block attention
   device cases (`test_vae_attention_distributed.py`, 18 device cases including
   the 30x53 and 45x80 production grids on 8x4), and the batch-2 ring key-mask
   eager golden, Hunyuan block PCC/leakage and SP-padding tests. The masked CFG
   generation itself already passes (225.04s at 13f/8 steps, frame PCC
   1.000000); what remains is the 121-frame 50-step quality run. Skip the trace
   variants until item 2 is resolved. Any test wanting a mesh smaller than 8x4
   must open 8x4 and take a submesh -- a direct 2x4 open fails in fabric-router
   sync on device 1 even on a quiet machine.
4. On an idle SP8xTP4 Galaxy, run the existing block PCC tests with
   `HY_DIT_QKV_SPLIT=1` and `0`, then profile 121-frame QKV and ring-SDPA device
   time. Keep the split path only if PCC is at least 0.99; compare the retained
   SDPA chunks against `HY_DIT_SDPA_PRESET=wan_bh_sp8tp4`.
5. Complete SigLIP I2V A/B timing and generated-frame quality comparison; keep
   it opt-in unless one-shot end-to-end latency improves.
6. Finish qualifying the mid-block attention paths. Chunking and flash SDPA are
   correctness-validated on hardware and the 720p allocation gate now fits;
   what remains is `HY_VAE_ATTN_DIST` device correctness (item 3), then the
   matched all-warm 121-frame 480p VAE-only A/B against tile/single-readback
   (item 1), then 720p end to end. Keep tile/single-readback as the default
   unless both resolutions fit and improve latency on warm numbers.
7. On idle hardware, run the focused real-weight Qwen test and matched
   host-Qwen/TT-Qwen generations with mixed positive/negative lengths. Keep the
   device path opt-in unless trimmed conditioning PCC remains >=0.999 and visual
   quality matches. Measure cold prepared-weight creation, warm weight-cache
   load, first encode/compile, repeated encode, and prompt-cache hit separately.
8. byT5's real-weight PCC gate now passes on a reserved TP1/TP2 mesh. What is
   left is generated-video quality with `HY_TT_BYT5=1` on a glyph prompt, and
   cold/warm encode timing; the random-weight Hunyuan TP2 and UMT5 regression
   cases should be run on the same healthy 1x2 mesh. Placement is unchanged --
   host byT5 remains correct for a committed 32-chip run.
