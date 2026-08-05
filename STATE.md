# MiniMax-H3 on Blackhole Galaxy — state

Compacted 2026-08-05. This file was a 4972-line append-only campaign journal of 111 dated amendments;
it now carries only **pending work** and **pitfalls**. Nothing is lost — the full journal is the version
of this file at commit `c0a1a7029b3`:

```bash
git show c0a1a7029b3:STATE.md            # the complete 111-amendment journal
git log --follow --oneline -- STATE.md   # every amendment as its own commit message
```

Amendment numbers below (e.g. "am. 76") refer to that journal. When appending new amendments, continue
from **112** and keep the protocol in `shared/journal-protocol.md`: append-only, newest at the bottom,
entry written *before* advancing, retractions first-class.

References in priority order: diffusers PR #14355 pinned at `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`
(gives `AutoencoderKLMiniMaxH3` and `AutoencoderKLMiniMaxH3Audio` as importable reference classes —
compare against these, not a hand-written port), then sglang PR #33275, then the raw checkpoint under
`FL2VA/video_vae/source/`. Tests follow `tests/models/ltx/` and `tests/models/wan2_2/`.

---

## Working point

Everything measured in this campaign is at one production working point. Numbers taken at any other
shape are evidence about that shape alone (am. 76).

| | |
|---|---|
| Task modes | `t2va`, `fl2va` (first / last / first+last), `ref2va` |
| Canvas, frames | 1344x768, 124 frames @ 24 fps → 37 latent frames, 207 audio latents |
| Steps | 50 → 49 forwards |
| Mesh | 4x8 Blackhole Galaxy, TP=4 axis 0 / SP=8 axis 1, ring, 2 links |
| Packed layout | `[text \| cond \| audio \| target video]`, `rows_per_frame` = 1008 |
| Residues (mod 32) | cond 1008≡16, target 37296≡16, audio 414≡30, cond+target 38304≡0 |
| `padded_len` | t2va 37888, fl2va ~39936; `seq_local` 4736 / 4992 |
| Latency | fully warm 63–74 s; VAE decode 3.8 s; denoise is **92 %** of e2e |
| Status | t2va and all three fl2va modes green e2e; conditioning proven to drive generation |

Artifacts at `~/h3_t2va_artifacts/t2va.mp4`. Required environment for every run:

```bash
export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers
export MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache
```

---

## Pending work

### Audio decode precision — levers exhausted; one decision left (am. 111, 112, 113)
`MINIMAX_H3_AUDIO_ACCURATE=1` reaches **0.45 %** relative RMSE against the diffusers reference, from a
10.46 % default — 23x less error, PCC 99.5451 % → **99.9990 %**, PSNR 40.29 → **67.53 dB** — for ~3x the
stage time (4.03 s → 13.24 s, i.e. ~+9 s on a 63–74 s e2e). Three independent levers, each fixing a
different one of the three sources, strongly complementary because the chain error is set by whichever
source is worst:

| split | mac | tap | rel_rmse | PCC | PSNR | warm |
|---|---|---|---|---|---|---|
| off | 0 | 0 | 0.1046 | 99.5451 % | 40.29 dB | 4.03 s |
| full | 0 | 0 | 0.0538 | 99.8950 % | 46.07 dB | 5.36 s |
| off | 1 | 0 | 0.0920 | 99.6111 % | 41.41 dB | 8.72 s |
| full | 1 | 0 | 0.0320 | 99.9522 % | 50.58 dB | 9.50 s |
| full | 0 | 1 | 0.0371 | 99.9526 % | 49.31 dB | 9.97 s |
| **full** | **1** | **1** | **0.0045** | **99.9990 %** | **67.53 dB** | **13.24 s** |

**Open decision: whether to enable it by default.** Nothing technical blocks it — the traced path matches
eager exactly (PSNR inf) — but it changes published latency by ~12 % of e2e and obsoletes
`AUDIO_RELATIVE_RMSE = 0.12`, which deliberately still describes the default path. Accurate mode also
needs a larger **trace region** (375463936 B measured vs the default path's 300 MB; the test now asks for
450 MB) and exceeds `test_audio_trace_minimax_h3.py`'s 300 s pytest timeout, so it wants `--timeout=1200`.

**Exhausted, with measurements** (do not re-try these): a 3-way operand split is *bit-identical* to a
2-way one on both conv3d and matmul, so 2-way already recovers the whole operand mantissa and 3.13e-04 is
a hard matmul floor; `C_in_block` widening gains 1.48x on an isolated conv but nothing end to end (256 is
no better, 512 fails outright) because the chain is dominated by the 126 narrow-channel AMP convs where it
cannot widen; `ttnn.snake_beta` is already fp32-grade at 7.2e-08, as are the upsampler (7.4e-08) and `sin`
(4.2e-08), so none is worth replacing; and HiFi4 + `fp32_dest_acc_en` was already optimal across all 8
`(math_fidelity, fp32_dest_acc_en)` combinations. What remains after all three levers is the matmul
multiply floor itself.

Untouched: the encoder's convs (only the decode path was measured).

### Perf 1 — VAE decode readback (task #12)
Device-side stitch via all-gather is **a wash** and is left unwired behind
`MINIMAX_H3_VAE_DEVICE_STITCH` (default off): readback of the stitched canvas runs at 570 MB/s against
1.83 GB/s for the per-tile path (am. 106). Next lead is *why* — the canvas sits at the end of a
slice/concat chain, so try `ttnn.clone` / `to_memory_config` to re-materialize it, or
`ConcatMeshToTensor` over a replicated canvas.

### Perf 2 — audio decode layout round-trips (task #13)
`Conv2d` is 4.0 % of the stage against ~70 % layout ops (am. 103). The op mix, not the maths, is the
target.

### Perf 3 — vision encoder TP/SP (task #14)
The Qwen3-VL vision tower is built **replicated, no TP** (`loader_minimax_h3.py`). ~595 M params.

### Denoise — `measured_sdpa_chunk_sizes` is dead at every shipping shape
`attention_minimax_h3.py` keys on `seq_local ∈ {4768, 9216, 13632}`; production is **4736** (t2va) and
**4992** (fl2va), so the tuned `(320, 384)` — measured at **−13 % on `to_qkv`** — has never executed in
the pipeline. Both modes fall back to `(256, 512)`, so past comparisons are apples-to-apples. Denoise is
**92 % of e2e**, so this is the highest-leverage untouched perf item.

### T-parallel audio decode is wrong at every factor
Shards ≥ 1 emit saturated garbage, uniform *within* each shard, boundary/interior ratio 1.23 — so not a
halo-width bug (am. 107). `_t_neighbor_pad` is **exonerated** by an isolated 6-case gate that passes
(`test_neighbor_pad_t_minimax_h3.py`, am. 108). Remaining candidates: the gather-compute-partition round
trip in strided convs, the resample stages, `_t_padding` alignment, final assembly. Sharded audio decode
is off by default, so this is a perf opportunity, not a correctness regression.

### `test_fused_conditioner_real_weights` is `xfail(strict=True)`
Reads **98.6224 %** against a 0.99 bar. Not a bug — am. 95 showed it is *better* than its input error
predicts (the tower's own ~8 % relative RMSE enters 95 % of rows). `strict=True` means it **fails if it
ever passes**, forcing whoever improves tower precision back here to re-derive the bar from the
production row. Do not loosen the bar without a measured floor behind it (am. 76).

### Parked, deliberately
- Stale am. 85 VAE decode numbers, superseded by am. 103/104 but not deleted.
- `_prepare_vae` reads 10.4 GB of safetensors eagerly even on a full cache hit — outside every timed
  row, so it appears in no published number, but it is real wall time every process.
- `MINIMAX_H3_PIXEL_MEAN/STD` is defined twice (`pipeline_minimax_h3.py` decode, `conditioning.py`
  keyframe normalize). A drift between them is a *silent asymmetric* bug: encode with one, decode with
  the other. The VLM branch's 0.5/0.5 is a third, legitimately different, constant.

---

## Pitfalls

### Machine and process — these cost hours
- **`tt-smi -r` is FORBIDDEN.** It dropped all 32 chips off PCIe on CPLD < 1.16. Use `tt-smi -glx_reset`.
- **Every device run is timeout-gated, and every kill is followed by a reset**: `fuser -v /dev/tenstorrent/*`,
  kill the process *group*, `tt-smi -glx_reset`, `tt-smi -ls`. Skipping the reset after a crash makes the
  *next* run fail somewhere unrelated (`bank_manager.cpp:462`) and you will blame your code.
- **`TT_DIT_CACHE_DIR` unset degrades silently**: 713 s instead of ~64 s, one log line, no error.
- **Never pipe a device run to `tail`** — buffering leaves the log empty until exit, so a hang shows nothing.
- **Never `git add -A`.** `models/tt_dit/internal-prodia/`, `models/tt_dit/prodia`, `recover-logs/`,
  `sweep_results_minimax_h3_encoder/` are unrelated untracked work, and three pre-existing stashes must
  not be disturbed.
- **VBench has its own interpreter** at `/data/kevinmi/vbench_env/bin/python` and must **not** be
  installed into `python_env` — it pins numpy<2 and transformers 4.33, which breaks `ttnn`.

### A measurement is only about the configuration it ran in
- **Op config can be set by importing some other module.** `get_conv3d_config` reads `_FP32_BLOCKINGS`,
  and the H3 audio shapes are added by `register_h3_audio_blockings()`, which fires at **import of
  `decoder_minimax_h3_audio`**. A harness importing only `layers/audio_ops` silently gets
  `C_in_block = 32` instead of production's 128 — a different op — so every precision number from it is
  wrong (am. 111 retracted am. 110 for exactly this). Build the production object, or import the
  production module, and read the config off the thing that will actually run.
- **Precision measured on one op class does not transfer to another.** Elementwise fp32 is exact to
  3e-08 (`sin`), `add`/`multiply` are exact, but an fp32 *reduction* is ~1e-03 — five orders apart. Am.
  109 used the first to falsify a claim about the second and had to be retracted.
- **A gate's shape is part of the gate** (am. 76). Only test production shapes.
- **Prefer direct observation to introspection.** In one session a bare `_FP32_BLOCKINGS.get` (before
  registration) and a module walk (malformed — found 1 conv of 23) both produced confident wrong answers.
  What resolved it printed the lookup *and* the result in the same row, so no step was inferred.
- **A borrowed pattern is a hypothesis** (am. 86). `fast_device_to_host` looked 39 % faster because it
  was not moving the data — it returned `[24..31, 0, 0, …]`. `float_to_uint8` was a 3.6x regression.
  Check the contract, not the call site.

### Metrics that lie
- **Confounded anchors.** The fl2va gate's keyframe is frame 0 of the t2va generation, so a pipeline that
  *ignored* the keyframe and re-ran t2va scored ~0.997 on it too. Use a discriminator the null hypothesis
  fails: a mirrored keyframe, or the fractal image (0.9964 following vs 0.4108 not).
- **A seam ratio near 1.0 is what a smooth scene gives, not what a correct one gives** (am. 87). A
  keyframe-anchored video carries more high-gradient structure; do not read ratios under 2.0 as defects.
- **`imaging_quality` is no-reference IQA** — a visually perfect night scene scored 0.4884 against a 0.64
  bar. The gated prompt and the VBench/CLIP thresholds are a matched pair; changing content invalidates
  the calibration.
- **Nothing gates mesh reassembly order.** Every numerics test is single-device or per-shard, so a tensor
  rebuilt in the wrong order passes all of them. The e2e CLIP gate is what caught it (37.37 → 19.58). If
  you touch a readback or add a collective, check the order explicitly.
- **A blunt gate can pass on wrong metadata.** Feeding rows metadata belonging to other rows read
  0.999888 — only 8.6e-5 below the real number.
- **t2va's no-regression bar is not "PASSED"**: it is std **46.05**, frame delta **9.88**, audio peak
  **0.076**, CLIP **37.37**, to every digit (am. 78).

### ttnn / API traps
- **`l1_small_size` is mandatory** — 65536 for the audio decoder. A bare `ttnn.open_device(device_id=0)`
  omits it. This bit three separate times. The failure is `bank_manager.cpp:462` /
  `L1_SMALL ... bank size is 0 B`, which never names the parameter, and **"bank size is 0 B" means
  *unallocated*, not too small**.
- **Two-axis all-gather transposes dim 0**: gathered position `c*rows + r` holds shard `r*cols + c`
  (am. 105, pinned by `test_two_axis_all_gather_permutes_dim0_by_transpose`).
- **`ShardTensorToMesh(dim=1)` splits 32 ways** across the whole mesh, not 8 on one axis. Production uses
  `ttnn.mesh_partition(dim, cluster_axis)`.
- `ttnn.all_gather` takes **no** `mesh_device` kwarg.
- `ttnn.Shape` does **not** support slicing — materialize once with `list(t.shape)`.
- **bf16 has 7 stored mantissa bits** → 2^-7 spacing, so 2^-8 is *half* an ulp. And bit-exactness against
  a bf16-rounded golden is the wrong bar: `ttnn.add` and torch do not round identically.
- `ttnn.slice` at a non-tile row boundary is legal and lossless (routes through ROW_MAJOR) but
  **untilizes the whole input** — 103 MB/device on the packed sequence.

### Reference-comparison traps
- **Building a reference on `meta` + `to_empty` leaves non-persistent buffers uninitialized** (e.g.
  `inv_freq`) while strict `load_state_dict` reports success → NaN. Construct normally and assert
  finiteness over params **and** buffers.
- **HF's `hidden_states[i+1]` is captured *before* the deepstack add**; our `activation_layers` capture is
  *after* it, so taps below 3 are apples-to-oranges. Production reads `hidden_states[50]` = output of
  layer **49** over a 50-layer stack.
- **Two distinct taggings that differ.** H3's `token_tags` marks the **whole vision block** (including
  `vision_start`/`_end`) as video; Qwen3-VL's `mm_token_type_ids` marks **only the `<|image_pad|>` run**.
  Getting the H3 tag wrong mis-modulates AdaLN with no PCC signal anywhere.
- **`dec_in_proj` is a Conv1d expecting `(B, C, T)`** — do not transpose into it.
- **Argument orders differ from the reference's** in `keyframe_condition_noise` and
  `draw_request_latents`. Do not copy the reference call site.
- **`stretch` keys on position in the keyframe *list***, so a lone `last_image` is the geometry anchor and
  *is* stretched. That reads like a bug until the `last_only` e2e case passes.
- **fl2va at seed 0 will not reproduce t2va at seed 0**, even with a blank keyframe: condition noise is
  the first draw and shifts both downstream streams. Expected and correct.
- The Qwen3-VL tower has `head_dim` 72 padded to 96 with an explicit `scale=72**-0.5`; the conditioner's
  `head_dim` is **128**, not 5120/64 = 80.

### Test-harness traps
- **A test that reads its input from the directory it writes to skips silently.** Use separate env vars
  for read vs write artifacts (`MINIMAX_H3_T2VA_ARTIFACT_DIR` vs `MINIMAX_H3_ARTIFACT_DIR`), and give
  `_write_artifacts` a `stem` so one mode cannot overwrite what another reads.
- **Never insert a helper between `@pytest.mark.parametrize` and its `def`** — the decorator binds to the
  helper and the test runs undecorated with `l1_small_size = 0`.
- **`warmup` must be shaped like the measured call** and assert it: same prompt length, keyframe included,
  and `pipeline.last_padded_len` equal between warm and measured. t2va got away with a ~1-token warmup
  only by luck (both round to 37888).
- **Pre-populate the embedding cache before a timed run** — `warmup` runs `use_prompt_cache=False`, so a
  fresh cache key means the first measured call pays a full device conditioner encode inside the timed row.
- **Do not compare single runs**: ±8 % run to run at identical shape and seed, 56.6–71.3 s (am. 82).
- `decode_tile_grid` returns `((y_starts, lengths, overlaps), (x_starts, …))` — nested, not two flat lists.
- `_probe_streams` returns stream **dicts**, not durations; `_decoded_frames` takes `count=`.
- **Look at the frames.** Seams and flicker are the two defects every whole-tensor metric hides, and both
  are parallelism bugs.

---

## Amendment 112 (2026-08-05) — operand splitting implemented: audio decode e2e RMSE 10.46 % → 5.38 %

Implements what am. 111 measured. `MINIMAX_H3_AUDIO_CONV_SPLIT` ∈ `off` (default) / `weight` / `full`,
resolved **at layer construction** (so the env var cannot desync an allocated residual from what
`forward` expects) and forced to `off` for any non-fp32 dtype, since splitting only addresses the fp32
datapath.

`conv3d_maybe_split` sums the terms; `prepare_conv3d_weight_state(split=True)` prepares `w_hi` and the
exact residual `w_lo` from one already-padded weight, so the bias is never padded twice, and the bias is
applied to exactly one term because it is not a factor of the product being split. Covers
`Conv1dViaConv3d` (hence `_AlignedOutConv1d` and `ConvTranspose1dViaConv3d`, which delegate to it) and
`Conv2dViaConv3d`.

The activation split needs **no layout change**: on ROW_MAJOR fp32 a `bfloat16` typecast round trip
reproduces torch's `.bfloat16()` bit-for-bit and `hi + lo` reconstructs the input exactly, both verified.
That mattered — am. 103 profiled this stage at ~70 % layout ops, so a tilize round trip would likely have
cost more than the precision was worth.

### End to end, `test_decode`'s inputs (latent from the reference encoder, 5 s, stereo)

| mode | convs | rel_rmse | PCC | PSNR | mel dist | warm median (min–max) |
|---|---|---|---|---|---|---|
| `off` | 1 | 0.1046 | 99.5451 % | 40.29 dB | 0.0783 | 4.028 s (3.811–4.362) |
| `weight` | 2 | 0.0793 | 99.7397 % | 42.70 dB | 0.0600 | 5.003 s (3.149–5.286) |
| **`full`** | 3 | **0.0538** | **99.8950 %** | **46.07 dB** | **0.0478** | 5.100 s (4.617–5.604) |

`full` gives **1.94x** on RMSE — matching the 1.9x measured on `conv_pre` alone, so the per-conv gain
carries through the whole ~130-conv chain without attenuation. PCC error falls 4.3x (4.55e-03 → 1.05e-03),
PSNR gains 5.8 dB, mel distance drops 39 %.

**On the cost, honestly.** Medians say +24 % (`weight`) and +27 % (`full`), but the ranges overlap —
`weight`'s fastest rep (3.149 s) is below `off`'s fastest (3.811 s) — so at n=5 the two split modes are
**not separable in time**, and only the off-vs-split gap is. That `full` is no dearer than `weight`
despite an extra conv per layer is consistent with the layout-dominated profile: the convs are not what
this stage spends its time on. So the earlier guess that 3x convs would be "a single-digit stage cost"
was wrong in magnitude (~+25 %, not ~+8 %) while right that it is cheap relative to the accuracy won:
~+1 s on a 63–74 s e2e.

### Gates

New `test_conv_operand_split_improves_precision` at `conv_pre`'s production shape reproduces
1.857e-03 / 1.246e-03 / **9.979e-04** and asserts the *ordering* rather than absolute values, since the
absolutes are hardware- and blocking-dependent while the ordering is the claim. It also asserts the
allocation contract (`weight_lo` present iff splitting) and carries one loose absolute ceiling on the
baseline — 2.1e-03, between production's 1.86e-03 and the 2.40e-03 an unregistered `C_in_block = 32`
fallback gives — so am. 111's blocking-registration trap cannot silently return. Full audio suite
**8 passed** with the default, and `test_audio_decode_traced` passes with `full`, so the flag is safe on
the traced path production uses.

`AUDIO_RELATIVE_RMSE = 0.12` is unchanged and its comment now says explicitly that it describes the
**default** path, so it must be re-derived if the default ever flips. Its justification is also repaired:
it now cites the *measured* 1.86e-03 per fp32 conv rather than the inferred 0.17 %-per-activation figure
that am. 109/110 argued over.

**A measurement note.** A single warm rep put these at 1.29 / 1.56 / 2.02 s — about 3x too fast, because
the first call after compile is not steady state. Five reps and a median is the minimum here; am. 82's
±8 % run-to-run was measured on a warm loop, not on the first iteration of one.

---

## Amendment 113 (2026-08-05) — the audio decode's dominant error was never the convs: 10.46 % → 0.45 %

Ran the levers to exhaustion. `MINIMAX_H3_AUDIO_ACCURATE=1` now reaches **0.45 %** relative RMSE against
the diffusers reference (PCC **99.9990 %**, PSNR **67.53 dB**, mel distance 0.0034) from a 10.46 % default
— **23x** less error, for ~3x the stage time.

### The bisection that redirected everything

After am. 112 I assumed the convs were the whole story. Walking the chain per stage said otherwise: the
`ups` stages mostly *shrink* the relative error (0.55–1.4x) while every AMP resblock grows it, and with
operand splitting on the growth factors got **worse** (7.91x vs 4.67x at stage 0). That is the signature
of a fixed absolute injection the split does not touch.

Feeding one AMP block the reference's own exact activation separated injection from amplification:

| | split off | split full | split full + MAC |
|---|---|---|---|
| one AMP block injects | 5.868e-03 | 4.284e-03 | **1.088e-03** |
| one `Activation1d` injects | 1.544e-03 | **1.544e-03** (identical) | **1.052e-07** |

The activation's injection was *bit-identical* across split modes — entirely outside the conv path. Six
activations per block in quadrature is 3.78e-03 against the block's 4.28e-03: the anti-aliased activation
**was** the remaining error, not the convs.

### Which part, and the surprise

| part of `Activation1d` | rel_rmse |
|---|---|
| upsample | 7.443e-08 |
| **downsample** | **1.544e-03** |
| `ttnn.snake_beta` (fused) | 7.242e-08 |
| snake_beta as a composite | 6.091e-08 |
| `sin` alone | 4.228e-08 |

`downsample` alone was 100 % of it. I had suspected `ttnn.snake_beta` — a fused SFPU op called with **no**
`compute_kernel_config`, so its precision cannot be configured, only replaced — and it is fp32-grade.
Measuring beat the suspicion.

Both resample filters go through the same `depthwise_tap_filter`, so the difference had to be *which path*
they took. It was: `depthwise_tap_filter` tries a HEIGHT_SHARDED `ttnn.conv1d` and falls back to
shift-multiply-add when the DRAM slicer cannot configure it. Forcing MAC took the downsampler from
1.5437e-03 to **5.3334e-08** — ~29000x — and the reason is structural rather than incidental: **MAC is a
sum of elementwise multiplies and adds, and those are exact in fp32 here**, while `conv1d` goes through the
~11-bit FPU multiply. The upsampler only looked clean because at stage 0 it already fell back to MAC.

### The third lever, and the two that are not

With the filters exact, what was left was conv3d — and its residual after splitting is *not* operand
mantissa (a 3-way split is bit-identical to a 2-way one) but partial-sum rounding across `C_in_block`,
which matmul does not have. A stride-1 conv needs no im2col matrix to become a matmul:
`y[t] = sum_j W_j @ x[t + dilation*j]`. Measured 1.78–3.47x per conv across every real shape, including
k=7, k=11, causal padding and the transposed upsamplers.

Ruled out, with numbers: **3-way splitting** is bit-identical to 2-way on conv3d *and* matmul at every
shape, so 3.13e-04 is a hard matmul floor; **`C_in_block` widening** gains 1.48x isolated but nothing end
to end — 256 measures 3.31 % against 128's 3.20 % and 512 fails outright — because the chain is dominated
by the 126 narrow-channel AMP convs where the block cannot widen, so only 2 convs of 128 would gain.
`MAX_C_IN_BLOCK` therefore stays 128, now with a measurement behind it instead of being a stub value.

### Two bugs the tests found, both from bypassing the conv3d route

- `ConvTranspose1dViaConv3d` builds a **causal** inner conv and then forces `external_pad_front = 0`
  because it supplies its own symmetric padding. A tap path deriving padding from `padding_mode` would
  prepend `eff_k - 1` zeros to all 7 upsamplers. Fixed by deriving from `external_pad_front +
  internal_padding[0]`, the same fields conv3d uses.
- `_AlignedOutConv1d` rounds a non-32-multiple `C_out` up (ups[5] emits 16) and the **bias** is allocated
  at the rounded width; conv3d pads it inside `prepare_conv3d_weight_state`, which the tap route bypasses.
  Caught only at `decoder.ups.5.conv.bias` in a full-model load, because every shape in my unit cases was
  a 32-multiple. The gate now includes a narrow-`C_out` causal case.

### Gates

`test_depthwise_mac_is_more_accurate_than_conv1d` (1.855e-03 vs 8.082e-08 at the real stage-1 downsampler
shape) and `test_tap_matmul_beats_conv3d` (4 shapes incl. the narrow-`C_out` causal case), both gating
inequalities rather than hardware-dependent absolutes, plus an output-shape agreement assertion.
Suite **13 passed** on the default path; traced accurate mode matches eager exactly (PSNR inf).

**The method note.** Am. 112's cost estimate leaned on a profile of a *different* conv class and was wrong
in magnitude; this time the redirect came from measuring injection with an exact input rather than reasoning
from where the error appeared. A growth factor that gets *worse* when you improve the input is the tell that
the stage injects rather than amplifies — and that is what said "stop optimising the convs".
