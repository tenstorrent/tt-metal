# Amendments — MiniMax-H3 `ref2va`

Numbering continues the H3 journal, whose 1–111 are at `git show c0a1a7029b3:STATE.md`, **112** at
`def2705bed3` and **113** at `f011cba1bd9`. Append-only. Retractions never edit the original.

Format per `../../../../tt-dit-skills-wt/models/tt_dit/.claude/skills/shared/journal-protocol.md`.
Every recorded *measurement* carries command · mesh shape · input shape · warm-window method ·
device vs wall time · commit SHA. A reading of source rather than a run says so explicitly.

---

## Amendment 114 (2026-08-05) — ref2va packed lengths are 1.2×–3.0× t2va's, and that is the campaign's dominant risk

**Assumed** (ref2va handoff §6, Phase 6): that holding "DiT programs *and* CCL buffers at several
padded lengths" was the memory risk, i.e. that the risk was the *number* of shapes.

**Measured.** Host-only, no mesh, no device. Commit `bd12ad2aeb2`. Built
`MiniMaxH3PreparedReference` objects with the geometry the reference's own sizing rules resolve, then
called the installed `packing_ref2va.build_ref2va_packed_sequence` and rounded to `sp_factor * TILE`
= 256. Target 1344×768 / 124 frames (37 latent frames, 1008 rows/frame, 207 audio latents):

| Request | text | ref video rows | ref audio rows | seq | padded | `seq_local` |
|---|---|---|---|---|---|---|
| t2va (baseline) | ~39 | 0 | 0 | 37710 | 37888 | 4736 |
| 1 image 1:1 | 4104 | 4096 | 0 | 45910 | **46080** | 5760 |
| 1 video + sound 16:9 | 6068 | 37296 | 414 | 81488 | **81664** | 10208 |
| image + video + audio | 10172 | 41392 | 828 | 90102 | **90112** | 11264 |
| 9 images 1:1 | 36872 | 36864 | 0 | 111446 | **111616** | 13952 |

**Evidence for the cause.** A reference never binds the target geometry. A reference image is
prepared at a **2048 px short edge with no area cap** (`packing_ref2va.py:75,554-576`) and the
checkpoint's image processor caps area at 16777216 px = 4096² (`text_encoder/preprocessor_config.json`),
so a 2048² reference is **not** downscaled. It therefore contributes 4096 merged vision tokens to the
text stream *and* 4096 video condition rows — the same image twice, through two different paths. A
video reference truncated to the target frame count contributes a **second full-length video stream**
(37296 rows, equal to the target's own) plus 6 vision blocks of 1008 tokens each.

Independently corroborated: `test_qwen3vl_vision_tower.py:91` already records `ref_1to1` as
`[1, 128, 128]` = 16384 patches → 4096 merged tokens, derived through the checkpoint's own processor
rather than by hand. The two derivations agree.

**What changes.** The risk is the *magnitude* of a single shape, not the number of shapes. A shape
probe at these three padded lengths becomes Phase 0 of the campaign, ahead of writing any ref2va
code, and the e2e case list is set by its verdict rather than assumed. Also noted: a 4:1 reference
image is `[1, 128, 512]` = 65536 patches → 16384 tokens **per image**, so aspect extremes are worse
than the nine-image case per reference; the campaign's cases use 1:1.

---

## Amendment 115 (2026-08-05) — reference audio rows are clean at t = 1.0, take the posterior mean, and are never noise-augmented

**Assumed** (handoff §4): "Reference latents are noise-augmented with the **same**
`MINIMAX_H3_KEYFRAME_NOISE_AUG` as `fl2va` (`encoders.py:318,629`), and condition timesteps are
pinned with `max(t, NOISE_AUG)`."

**Read** — source reading of the installed reference, not a device measurement. Commit
`bd12ad2aeb2`. `encoders.py:629` is inside `if condition_latents is not None:` and applies to the
**video** rows only. Three separate divergences for audio:

1. `encoders.py:596-598` — a soundtrack takes `posterior.mode()`, the **mean**. No seeded sample, and
   no fp16 round trip. Image and video references at `:584-588` do take the seeded sample (42) and the
   fp16 round trip.
2. `encoders.py:631-632` — `audio_condition_latents` is moved to device **untouched**;
   `scale_noise` is never applied to it.
3. `before_denoise.py:417-419` — `build_row_timesteps(..., max(float(timestep), NOISE_AUG), 1.0)`.
   The audio conditioning timestep is a **literal 1.0** at every step, never `max(t, 0.999)`.

**What changes.** Nothing in our code, which is the point worth recording: `build_row_timesteps`
already takes `condition_audio_timestep` and `pipeline_minimax_h3.py:1181` already passes `1.0`. But
implementing the handoff's §4 as written would have noise-augmented the audio reference rows and run
them at the wrong timestep, and neither would have failed a shape, finiteness or PCC check — it would
have produced a plausible soundtrack that ignored its reference.

**Method note.** A line citation in a handoff names a line, not a scope. `:629` is one line inside a
two-branch function whose other branch is the whole finding.

---

## Amendment 116 (2026-08-05) — both `_temporal_position_span` summation orders differ at the production frame count, not merely "from 16 onwards"

**Assumed** (handoff §6, Phase 2): the two orders "differ in the last ulp from 16 latent frames
onwards", with the gate stated at "≥ 16 latent frames".

**Measured.** Host-only, no mesh. Commit `bd12ad2aeb2`. Command:
`python_env/bin/python -c` comparing `packing._temporal_position_span` (numpy pairwise) against
`packing_ref2va._temporal_position_span` (sequential float64) for n = 1..59, and both against our
`models/tt_dit/pipelines/minimax_h3/packing.py::_temporal_position_span`.

```
n=16  pairwise 86.66666666666667    sequential 86.66666666666669
n=17  pairwise 93.33333333333334    sequential 93.33333333333336
n=37  pairwise 206.66666666666663   sequential 206.66666666666657     <- production
```

39 of the first 59 frame counts diverge. Our existing implementation is bit-equal to the reference's
**pairwise** one for all of n = 1..59.

**What changes.** Two things the handoff's framing understates. First, the divergence at the
production n = 37 is **~2 ulp and in the opposite direction** to the n = 16 case, so a gate written
only at n = 16 does not pin the sign. Second, and load-bearing: the sequential order is reached
through `rotary_time += max(num_audio_latents, span(num_latent_frames))` for a **video** reference,
and a video reference truncated to the target frame count has exactly the target's 37 latent frames —
so this is not a corner case, it is the shipping path. The gate asserts exact equality against *both*
reference functions **and** asserts that they differ, at n = 16 and at n = 37.

---

## Amendment 117 (2026-08-05) — `sp_factor=1` for ref2va already holds; the loose end closes with no design work

**Assumed** (handoff §8): "`sp_factor=1` for `ref2va` per `vision_qwen3vl.py:525`. Confirm the
constraint still holds before designing around it."

**Read** — source reading, not a measurement. `vision_qwen3vl.py:515-535` raises
`NotImplementedError` for multi-block attention under sequence parallelism. But
`loader_minimax_h3.build_minimax_h3_vision_tower` takes **no parallel config at all** — its docstring
says the tower is replicated because it is ~1.2 GB bf16 against the conditioner's ~50 GB — and
`VisionParallel()` defaults `tp_factor` and `sp_factor` to 1. That constructor is the only one
`pipeline_minimax_h3._prepare_vision_tower` calls.

**What changes.** Nothing to design around: the production path satisfies the constraint by
construction, and the DiT's SP=8 is a different module and unaffected. What the constraint *does*
imply is a cost, not a correctness risk — nine reference images are nine 16384-row attentions with no
parallel speedup, which is unmeasured and belongs in the latency accounting rather than the design.

---

## Amendment 118 (2026-08-05) — two subfolder-blind cache keys would have silently applied `transformer/`'s AdaLN table to `transformer_ref`

**Assumed**: that loading the ref2va partition is "change which subfolder is loaded" (handoff §3).

**Read** — source reading, not a measurement. It is that, plus two cache keys that do not know the
subfolder exists:

1. `pipeline_minimax_h3.py:676` — `precompute_adaln_table(self.weights_dir / "transformer", ...)` is
   **hardcoded**, so a ref2va run would project `transformer/`'s AdaLN weights into the table the
   `transformer_ref` stack then modulates with.
2. `pipeline_minimax_h3.py:638-651` — `_adaln_cache_path` keys on `weights_dir`, step count, both
   shifts, the noise-aug floor and three geometry values, but **not** the partition. Both partitions
   live in one repository, so the two would collide on one cache file.
3. `pipeline_minimax_h3.py:619` — the weight cache's `subfolder` is
   `"transformer_precomputed_adaln" if self.precompute_adaln else "transformer"`, with the same
   collision.

`precompute_adaln` is **on by default** in the pipeline, so (1) and (2) are the shipping path.

**What changes.** All three are parameterized by the partition in Phase 5. Recorded as an amendment
rather than a silent fix because the failure mode is the one `_adaln_cache_path`'s own docstring
warns about — "a stale hit is silent: it modulates every block slightly wrong at every step, in the
same direction, and nothing downstream can notice" — and the fix must be verified, not assumed.

---

## Amendment 119 (2026-08-05) — the ref2va vision-tower coverage was adopted after all, in consolidated form

**Assumed** (handoff §8): "Jonathan's two `ref2va` conditioner tests were deliberately not adopted in
the merge that brought his branch in. They live on `jonathansu/minimax-h3-textencoder` and depend on
a golden captured by a forward hook on `layers[TAP]` … Porting them means rewriting their goldens
against `hidden_states` by index."

**Read** — prior-art gate, source reading. `2a1a17eea85` did add
`test_vision_tower_ref2va_real_weights`, measuring four ref2va presentations at real depth 27
(two_images / video_2_frames 99.6953 %, mixed 99.6069 %, nine_images 99.6937 %). Its **consolidated
successor is in the tree**: `test_qwen3vl_vision_tower.py::test_tower_on_device`, whose `GRIDS` dict
covers every grid ref2va can present — all seven `ref_*` reference-image aspects including the 65536-patch
4:1 extreme, `two_refs` (unequal block lengths), `video_3_frames` (blocks from one `t>1` grid row),
`image_and_video` (both sizing rules in one sequence) and `max_load` (nine images + three videos,
168192 patches, 18 blocks).

**What changes.** The vision-tower half of ref2va needs **no porting work** and no golden rewriting;
the forward-hook off-by-one the handoff worried about is not in the adopted path. What remains on the
conditioner side is only the *presentation* layer above the tower — the `<|video_pad|>` token type id,
multi-pad-id vision runs, and assembling the tower's output in presentation order. Two of the
handoff's three suggested salvages from that branch (`MINIMAX_H3_RUN_REF=0`, per-seed `_test_image`)
are therefore optional conveniences rather than prerequisites.

**Method note.** "Not adopted" in a handoff can mean "not adopted *in that form*". The prior-art gate
is what distinguishes the two, and it cost one `git log --grep` plus one `grep` of the tree.

---

## Amendment 120 (2026-08-06) — the ref2va conditioner geometry confirmed on real media, end to end through the checkpoint's own processors

**Assumed**: the reference-video vision-block arithmetic derived by hand in am. 114 — 124 frames at
24 fps sampled to 2 fps gives 11 frames, merged in pairs into 6 vision blocks of 1008 tokens.

**Measured.** Host-only, no mesh, commit `5c8adce1e85`. Real media:
`~/h3_fl2va_artifacts/fl2va_first.mp4`, a prior calibrated run of this very pipeline. Decoded with
our own `packing_ref2va.decode_reference_video`, prepared with `references.prepare_references`, then
put through the checkpoint's own `Qwen3VLVideoProcessor`:

```
decoded            (124, 768, 1344, 3) @ 24.0 fps, soundtrack (2, 164864) @ 32000 Hz
prepared           frames (124, 768, 1344, 3), waveform (2, 164864), num_frames 124
2 fps sampling     11 frames -> 6 vision blocks
block timestamps   [0.25, 1.25, 2.25, 3.25, 4.25, 5.0]
video_grid_thw     [[6, 48, 84]]        pixel_values_videos (24192, 1536)
merge_size^2 = 4   blocks(t) = 6        tokens/block = 1008
reference assert   t == len(block_timestamps): 6 == 6  PASS
```

**What changes.** Nothing — which is the result. Three things are now measured rather than derived:
the hand arithmetic of am. 114 was right; the reference's own consistency assert
(`encoders.py:418-423`, "the processor merged a reference video into N vision blocks but MiniMax-H3
labels M of them") holds on real media at the production shape; and the 24 fps clip takes the
parity-exact untouched route through `prepare_reference_frames`, because it already *is* the canvas
its own aspect ratio resolves to.

Two incidental facts worth having: the real soundtrack is **164864** samples, not the 165333 that
5.1667 s implies, so `prepare_reference_waveform`'s truncation is a no-op for this clip — and both
lengths still give **207** audio latents after the zero-pad to a whole 800-sample hop, matching the
target's own 207. And `0.25` formats as `"0.2"` under `"{:.1f}"`, confirming the round-half-to-even
timestamp contract on a value that is exactly representable.

**Method note.** The whole chain was checkable on host in one command, without the mesh and without
the DiT. Confirming a geometry claim through the production processors before spending Galaxy time is
close to free; deriving it and finding out at e2e is not.

---

## Amendment 121 (2026-08-06) — fixed baseline is green on this base; the fl2va gate skips silently when its two artifact directories disagree

**Assumed**: that pointing the two artifact environment variables at separate per-purpose directories
was tidier than sharing one.

**Measured.** Mesh 4×8, cold e2e, commit `bd12ad2aeb2` (t2va) / `5c8adce1e85` (fl2va, new files only).
Commands verbatim in `CAMPAIGN.md`. t2va **1 passed in 560 s**; fl2va **4 passed in 645 s**; exit 0
both. Numbers in `CAMPAIGN.md`'s fixed-baseline table — CLIP 37.38, audio peak 0.095 and the fractal
discriminator's 0.9963 / 0.2972 all reproduce STATE.md exactly, so the base has not moved.

But the **first** fl2va attempt reported `4 passed`-equivalent success as **4 skipped**, with
`MINIMAX_H3_ARTIFACT_DIR` set to `round-0/fl2va` and `MINIMAX_H3_T2VA_ARTIFACT_DIR` to `round-0/t2va`.
t2va writes to the *first* variable, and the fl2va gate reads the calibrated `t2va.mp4` it keys its
tier-6 thresholds to from the *second*. So the clip existed, in the other directory, and all four
cases skipped with a one-line reason inside a 900-line log.

**What changes.** Both variables point at one directory, and the campaign's baseline command sets
them together with a comment saying why. Recorded because the failure presented as `exit 0` with a
green summary line: `1 passed, 4 skipped` reads as success at a glance, and a skipped baseline gate
is worse than a failing one — a red gate stops the campaign, a skipped one silently removes the
comparison every later round is measured against.

**Method note.** STATE.md already carries "a test reading its input from the directory it *writes*
skips silently — separate env vars". This is the same trap from the other side: the env vars *were*
separate, and pointing them at different places is what broke it. The rule that survives both is
narrower and checkable: **assert the count, not the exit code.** A baseline run must state how many
tests it expected to pass.

---

## Amendment 122 (2026-08-06) — the typed condition stream leaves every existing PCC bit-identical; the two longest ref2va shapes exceed the correctness test's budget on the CPU reference, not on the device

**Assumed** (campaign plan, Phase 4): that the ref2va cases could be added to
`test_minimax_h3_transformer` at their production lengths, alongside the existing ones.

**Measured.** Mesh 4×8, TP=4/SP=8 ring 2 links, 2-layer depth with randomized norm weights, commit
`f9c54c2b22a` (before) and working tree (after). Command:
`scripts/run_safe_pytest.sh --run-all .../test_transformer_minimax_h3.py -k random_weights`.
Before-capture 6 passed in 658 s; after 7 passed / 2 failed in 1370 s.

**The gate's own result, digit for digit.** Existing cases, video / audio PCC:

| case | before | after |
|---|---|---|
| small_s2048 | 99.9974 / 99.9973 | **99.9974 / 99.9973** |
| unaligned_s2112 | 99.9973 / 99.9975 | **99.9973 / 99.9975** |
| s21504 | 99.9973 / 99.9974 | **99.9973 / 99.9974** |
| prod_768p_5s | 99.9973 / 99.9974 | **99.9973 / 99.9974** |
| prod_768p_5s_fl2va | 99.9973 / 99.9974 | **99.9973 / 99.9974** |
| prod_768p_5s_fl2va_first_last | 99.9973 / 99.9975 | **99.9973 / 99.9975** |

All twelve values unchanged. That is the claim the design rested on: replacing `condition_1BKC` with
a list of typed blocks projects fl2va's single block through the same `proj_in`, and a per-row GEMM
against a shared weight is row-independent, so routing it through a loop changes nothing. Not merely
"still above 0.9995" — identical. `prod_ref2va_1image` (46080 padded) also passed, at
99.9973 / 99.9974.

**The two failures were `Failed: Timeout (>300s)`**, on `prod_ref2va_1video` (81664 padded) and
`prod_ref2va_mixed` (90112). Not a memory failure and not a numerics failure: `pytest.ini` sets a
repo-wide 300 s per-test default, and the cost that blows it is the **torch reference**, whose full
self-attention is O(n²) on CPU. Evidence: `prod_768p_5s` at 38222 rows spends ~110 s on the torch
side; 81488 rows is 4.5× the attention work, so the reference alone exceeds the budget before the
device is asked for anything. The log confirms the device side got as far as compiling its matmuls.

**What changes.** The gate is split by what each half can actually establish:

- **Interleaved numerics** move to production *residues* at a tractable length — audio cond 414
  (30 mod 32), video cond 1008 (16), target audio 414 (30), target video 3024 (16), sequence 5372.
  What an interleaved region can get wrong is which projection a block takes and which rows it lands
  on; both are residue- and order-sensitive and neither improves with length.
- **Production lengths** move to `test_minimax_h3_transformer_real_weights`, which has no CPU
  reference to pay for, now carrying `@pytest.mark.timeout(5400)` and the three ref2va shapes
  (46080 / 81664 / 111616) at full 50-layer depth. That test is the campaign's shape probe.

**Method note.** "A gate's shape is part of the gate" (am. 76) is about *numerics at production
shapes*. It does not follow that every gate must run at production length — here the binding cost was
the reference implementation on CPU, and paying ~30 min of CPU per case would have bought no
additional sensitivity. Say which property a shape is protecting before paying for it, and put the
length where the property actually depends on it.

---

## Amendment 123 (2026-08-06) — every ref2va shape fits at full depth, and `transformer_ref` loads strictly

**Assumed** (am. 114): that ref2va's 1.2×–3.0× packed lengths were a residency risk serious enough to
gate the campaign, and that the 9-image ceiling might not fit at all.

**Measured.** Mesh 4×8, TP=4 axis 0 / SP=8 axis 1, ring, 2 links. **Full 50 layers, real checkpoint**,
`MINIMAX_H3_SUBFOLDER=transformer_ref`, commit `5ec8933bbfa`. Command:
`scripts/run_safe_pytest.sh --run-all .../test_transformer_minimax_h3.py -k "real_weights and ref2va"`.
6 passed in 1162 s, exit 0. Warm window is the second of two identical forwards in one process, device
time not isolated from host dispatch (the test's own docstring notes this stack is host-dispatch bound):

| case | padded | rows/device | cold forward | **warm forward** | video std / absmax |
|---|---|---|---|---|---|
| `ref2va_1image_s46080` | 46080 | 5760 | 20.27 s | **2.11 s** | 1.2711 / 7.5312 |
| `ref2va_1video_s81664` | 81664 | 10208 | 114.33 s | **3.26 s** | 1.4478 / 8.2500 |
| `ref2va_9image_s111616` | 111616 | 13952 | 177.93 s | **5.45 s** | 1.6485 / 10.1875 |

No allocation failure at any of them. Outputs finite, non-degenerate and in range at all three. The
state-dict load took 259.3 s cold and 141.8 s once the weight cache was warm.

**Two further results, neither of which was the question asked.**

1. **`transformer_ref` loads strictly.** `load_torch_state_dict` is strict and all 638 keys of the ref
   partition map onto the same TT module as `transformer/`'s. So "the configs are byte-identical, our
   DiT needs no architectural change" is now verified against the weights and not just the config.
2. **The cold forward scales far worse than the warm one** — 20 s → 178 s across a 2.4× length, i.e.
   kernel compilation at a new shape, while warm goes 2.11 → 5.45 s (2.6×, close to linear). Every
   distinct ref2va request shape pays that compile once. For a 49-forward denoise the warm number is
   what matters, so ref2va's denoise is ~1.6×/2.5×/4.2× t2va's per-step cost at these three shapes.

**What changes.** The Phase 6 e2e case list is unblocked at every shape rather than trimmed. The
campaign's `mixed` case (90112 padded) was **not** probed directly; it sits between two shapes that
both fit, which is an interpolation and is recorded as one rather than as a measurement.

**Method note.** The probe was worth its 19 minutes for the *second* result, not the first. It was
built to answer "does it fit"; what it actually pinned down was that the ref partition's weights load
at all — a question the plan had assumed away from a `diff` of two config files.

---

## Amendment 124 (2026-08-06) — the image reference encode is green at production resolution; the taps=3 video-reference encoder does not fit in L1 at the shipping `l1_small_size`

**Assumed** (campaign plan, Phase 3): that all three reference modalities would encode through existing,
already-gated machinery, and that the only open question was numerical parity at `pcc=0.99`.

**Measured.** Mesh 4×8, `l1_small_size=65536`, `trace_region_size=200 MB`, commit `5ec8933bbfa`.
Command: `scripts/run_safe_pytest.sh --run-all .../test_references_minimax_h3.py -k encode_references`.
Real media throughout — a decoded frame of `~/h3_fl2va_artifacts/fl2va_first.mp4` as the image
reference, its own frames as the video, its own soundtrack as the audio. 1 passed, 2 failed in 255 s.

**Green: the image reference.** Resolved geometry `1 x 128 x 224` (a 16:9 frame at the 2048 px short
edge is 2048x3584, i.e. 7168 condition rows), and **PCC 99.9905 %** against
`MiniMaxH3Ref2VAReferenceEncoderStep` — far above the 0.99 floor, as expected, because the image path
reuses fl2va's already-gated `encode_clip` + seeded sample + fp16 round trip unchanged. That is gate 3
for images, on natural pixel statistics.

**Red: the video reference.** `TT_THROW program.cpp:1763` —

```
Statically allocated circular buffers in program 654 clash with L1 buffers on core range
[0-0 - 11-9]. L1 buffer allocated at 1504000 and static circular buffer region ends at 1508224
```

inside `vae.encode` → `_run_encoder_units(flat, 3)` → the **taps=3** encoder's forward
(`encoder_minimax_h3.py:335`). The overlap is **4224 bytes** — marginal, not an order of magnitude.

**Why it had never been seen.** The video VAE encoder's own tests run at
`SINGLE_DEVICE = [pytest.param((1, 1), {})]` — a 1×1 mesh with **empty device_params**, i.e. no
`l1_small_size` reserved at all. And `t2va` / `fl2va` never encode a video: a keyframe is one frame and
takes the **taps=1** path. So the combination "taps=3 encoder, `l1_small_size=65536`" is new to ref2va
and had no prior coverage. This is STATE.md's "a measurement only describes the configuration it ran
in", reached from the test side: the component was green, in a configuration production does not use.

**What changes.** The Phase 3 gate is parametrized per modality, so image / audio / video pass or fail
independently instead of the first failure masking the rest — a single combined request had reported
all three as one red. And `MINIMAX_H3_L1_SMALL` now overrides the device parameter so the campaign can
*measure* what the taps=3 encoder needs; a sweep over 65536 / 32768 / 16384 / 8192 follows, one config
per process per `shared/device-hangs.md`. Reducing it is not obviously free: STATE.md records 65536 as
mandatory for audio, so a value that fits the encoder must still be checked against audio decode
before it can ship.

---

## Amendment 125 (2026-08-06) — the audio VAE encoder's readback only ever worked on a single-device mesh

**Assumed** (am. 119 / source-ideas r0): that `MiniMaxH3AudioEncoder` was "a complete device port,
gated by `test_encode` (pcc 0.99) and `test_roundtrip` (28 dB PSNR)" and needed only wiring.

**Measured.** Same run as am. 124, the `audio` case:

```
TT_FATAL pytensor.cpp:299: buffers.size() == 1
Can't convert a tensor distributed on MeshShape([4, 8]) mesh to row-major logical tensor.
Supply a mesh composer to concatenate multi-device shards.
```

from `encoder_minimax_h3_audio.py:361`, `ttnn.to_torch(self.mean_proj(projected))`.

**The cause, and that it was already known one file over.** `MiniMaxH3AudioDecoder.__call__` carries
exactly this fix, with a comment that names the failure: *"A bare `ttnn.to_torch` asserts
`buffers.size() == 1` and so only ever worked on a single-device mesh — which is what kept this
decoder off the mesh entirely."* The encoder was written to the same shape and never received the
same fix, because its tests are `SINGLE_DEVICE` and the pipeline had never built it — `_prepare_audio_decoder`
loads the decoder half only.

**What changes.** The encoder reads back one replica when `get_num_devices() > 1`, matching the
decoder. Two-line fix; the value is in what it says about the gate that passed.

**Method note.** "Exists and is gated" is not "runs where you need it". Both of this round's failures
are the same shape: a component green under `SINGLE_DEVICE` with empty `device_params`, first exercised
on the production mesh by ref2va. Worth checking the *device configuration* of a component's existing
tests before counting it as reusable — the prior-art ledger recorded the pcc numbers and not the mesh
they were taken on.

---

## Amendment 126 (2026-08-06) — `l1_small_size` 16384 is the first value the taps=3 video-reference encoder fits in; the video encode is then parity-green

**Assumed** (am. 124): that the 4224-byte L1 clash might need a conv-blocking change or a smaller
spatial tile, either of which would have broken parity with the reference's own tiling.

**Measured.** Mesh 4×8, commit `5ec8933bbfa`, the ref2va **video** reference-encode case only, **one
configuration per process** with its own 1800 s timeout per `shared/device-hangs.md`. Override via
`MINIMAX_H3_L1_SMALL`; everything else held fixed.

| `l1_small_size` | result |
|---|---|
| 65536 (the shipping value) | **failed** — CB/L1 clash |
| 32768 | **failed** — CB/L1 clash |
| **16384** | **passed** |
| 8192 | passed |

So the fix is a device parameter, not a kernel or a tiling change. `l1_small_size` reserves the **top**
of L1, so a smaller reservation pushes those small allocations *above* the static circular-buffer
region instead of into it — which is why shrinking it helps and growing it would not. 16384 is chosen
over 8192 as the largest value that fits, leaving the most headroom for whatever else wants the small
region.

At 16384 the video reference encode is parity-green: geometry `7 x 48 x 84` (22 frames → 7 latent
frames on the 768x1344 canvas), 7056 condition rows, **PCC 99.9927 %**, CCC 99.9882 %, against
`MiniMaxH3Ref2VAReferenceEncoderStep` on real decoded frames.

**What changes.** Both ref2va test files default to `l1_small_size=16384` with the measurement cited.
t2va and fl2va keep 65536 and are untouched — they never reach the taps=3 encoder.

**Still open, and it is the reason this is not yet a closed question.** One process holds every ref2va
stage at one `l1_small_size`, and STATE.md records 65536 as mandatory *for audio*. Audio decode at
16384 is therefore unverified: the e2e run is what settles it. If audio decode fails there, the options
are a conv-blocking change for the taps=3 encoder or splitting reference encode into its own process —
not raising the value back, which is measured not to work.

**Method note.** The sweep was worth four processes because the *shape* of the answer was unknown: a
4 KB overlap could have meant "one parameter away" or "the blocking is wrong". Measuring the cheap
parameter first is what kept a kernel change off the table.

---

## Amendment 127 (2026-08-06) — ref2va runs end to end; audio decode is fine at `l1_small_size=16384`, closing am. 126; and a LINE fabric config fails the DiT outright

**Assumed** (am. 126, still open): that audio decode might need `l1_small_size=65536` and therefore
conflict with the 16384 the taps=3 video-reference encoder requires, forcing a conv-blocking change or
a second process.

**Measured.** Mesh 4×8, `{**ring_params_req_exact_devices, "l1_small_size": 16384}`, commit
`ffa0800618a`, `transformer_ref` partition, AdaLN precompute on, seed 0. Target 1344×768 / 124 frames /
50 steps, one 2048×2048 image reference. Command:
`scripts/run_safe_pytest.sh --run-all .../test_pipeline_ref2va_minimax_h3.py -k "end_to_end and one_image"`.
**1 passed in 457 s.** Cold e2e, not warm — no `warmup()` was run, so these are not comparable to a
warm latency figure:

| stage | seconds |
|---|---|
| Encoder (device: conditioner + vision tower) | 83.4 |
| Reference encode | 2.8 — video rows `(4096, 96)`, audio `None` |
| Denoise, 49 forwards | 107.9 (≈ 2.2 s/forward) |
| VAE decode | 8.4 |
| Audio decode | **8.2** |
| total compute | 210.7 |

Packed sequence **45925 → 46080 padded, 5760 rows/device, 4096 condition rows** — the padded length
matches am. 114's host-only prediction exactly, and the test asserts it so a drift cannot pass silently.
Denoise at 2.2 s/forward is consistent with the probe's 2.11 s warm forward at this shape (am. 123).

**am. 126's open question is closed: audio decode works at 16384**, in 8.2 s, with
`check_audio_sanity` and `check_av_sync` both passing. So STATE.md's "mandatory 65536 for audio" is
about `l1_small_size` being *set at all* rather than about that particular value — consistent with the
failure it describes (`bank_manager.cpp:462`, "bank size is 0 B", which means *unallocated*). No
conv-blocking change and no second process are needed.

**A second finding, from a failure before this pass.** The first attempt died in 76 s with
`TT_FATAL fabric.cpp:174: forwarding_direction.has_value()`. Cause: the ref2va test file had
`fabric_config: FABRIC_1D` — a **line** config — while the DiT attends in a ring on the SP axis and the
t2va/fl2va gates use `ring_params_req_exact_devices`. The Phase 3 encode gate had passed under the same
line config because the VAE encoders use no ring CCL, so nothing had caught it. Both ref2va test files
now use the ring params, differing from the shipping gates in `l1_small_size` alone.

**Frames inspected** (`artifacts/round-6/`, frames 0/17/62/123). Frame 0 is a coherent photographic
interior matching the prompt — window mullions, curtain folds, wood grain, a light patch on the floor —
and frame 123 is the same room after a slow push-in, with the doorframe vignette of frame 0 gone. No
seams at the tile boundaries, no flicker between the sampled frames. `check_spatial_seams` passes at the
768/1344 tile boundaries.

Also confirmed benign: the `DRAM Auto slice could not find valid slice configuration` messages logged at
`critical` during decode are **pre-existing** — 72 occurrences in the t2va baseline and 144 in fl2va,
both of which pass. A config search that falls back, not a ref2va defect.

**Method note.** Three of this campaign's four device failures were the same class: a component green in
a configuration production does not use (`SINGLE_DEVICE` with empty `device_params`, twice; a line
fabric config, once). The cheapest guard is to copy the *whole* device-params dict from the gate that
already ships the surrounding pipeline, and change only what a measurement forces you to change.

---

## Amendment 128 (2026-08-06) — conditioning is provably not a no-op (signal 0.080038 against a floor of exactly 0); the *directional* half of that gate used the wrong instrument and the wrong references

**Assumed** (campaign plan, §7): that a Mandelbrot fractal against a stripe field, compared by
whole-frame luminance correlation, would show each output resembling the reference it was given.

**Measured.** Mesh 4×8, ring params, `l1_small_size=16384`, commit `4d04f289379`, `transformer_ref`,
seed 0, 1344×768 / 124 frames / 50 steps, three generations. Both references 2048×2048, so the packed
layout is identical row for row and every noise draw has the same shape in the same order — the noise
is bit-identical and the only difference between the two requests is reference *content*.

```
run-to-run floor      0.000000     (the same request twice)
reference-swap signal 0.080038     mean absolute pixel value, over [0, 1]
resemblance to fractal   A = -0.1175   B = -0.0503
resemblance to stripes   A =  0.0028   B =  0.0000
```

**The decisive half passed, and passed in the strongest available form.** The pipeline is
**bit-reproducible** — the same request twice differs by exactly 0.000000 — so an implementation that
ignored its reference would score a swap signal of exactly 0.000000 as well. It scores **0.080038**.
There is no threshold to argue about: the null hypothesis produces zero and the measurement does not.
Separately, `test_ref2va_reference_order_changes_the_request` passed at divergence **0.097555**
(padded 54272), so reference *order* also reaches the output, not just reference identity.

**The directional half failed, and the instrument is the reason.** `-0.1175 > -0.0503` is false. Two
mistakes, both mine:

1. **The metric.** Whole-frame luminance correlation asks whether the reference's *pixels* appear at the
   same *coordinates*. That is what an `fl2va` keyframe does, because it is pinned to frame 0. A ref2va
   reference is not pinned to anything — it conditions what the output is *of*. So the metric measures a
   property the feature does not have, and its output is noise: every number above is within ±0.12 of
   zero, including the ones for the reference the run was actually given.
2. **The references.** A Mandelbrot fractal and a stripe field are content the model has no way to
   render for this prompt. Whatever conditioning does with them, it cannot be to make the output look
   like them, so even a perfect instrument would have little to find.

**What changes.** The gate keeps the divergence half unchanged — it is decisive — and gains an absolute
floor (`signal > 0.01`) so a *shrinking* effect is caught as well as a vanishing one. The directional
half is rebuilt:

- **References**: a real decoded frame against **the same frame with its colours inverted**. Identical
  size, identical texture and edge statistics, opposite palette. So the two references disagree about
  exactly one thing, and it is a thing a reference plausibly transfers and that is measurable.
- **Instrument**: CLIP image-image cosine similarity (`open_clip`, already the t2va gate's tool),
  which is semantic rather than positional. Mean RGB distance is reported alongside as the
  interpretable companion.

**Method note.** This is the campaign plan's own §7 warning landing on the plan's own gate: *"before you
believe a gate, ask what value it would return if the feature were entirely absent."* I applied that
test to the divergence check, where it works, and never applied it to the directional check — where the
answer is "the same noise it returns now". Asking it of *every* assertion, not just the headline one,
is the rule that would have caught this before three generations of Galaxy time.

---

## Amendment 129 (2026-08-06) — the precomputed AdaLN table had no row for the audio-conditioning timestep, so every audio-bearing ref2va request failed; and it failed loudly, which is why it was cheap

**Assumed** (am. 115): that because `build_row_timesteps` already accepted a `condition_audio_timestep`
and the pipeline already passed `1.0`, the audio-conditioning level cost no code.

**Measured.** Mesh 4×8, ring params, `l1_small_size=16384`, commit `2ad946fc6bf`, the two audio-bearing
e2e shapes. Both failed identically, inside `_denoise`:

```
IndexError: index 0 is out of bounds for dimension 0 with size 0
  pipeline_minimax_h3.py:1745
    [int((levels == value).nonzero()[0, 0]) for value in unique]
```

`build_row_timesteps` returns **four** distinct levels for a request with reference audio rows —
video `t`, audio `t`, `max(t, 0.999)` for the visual conditioning rows, and a literal `1.0` for the
audio conditioning rows. But `request_step_timesteps`, which decides what the precomputed AdaLN table
carries, only ever built **three**: t2va and fl2va have no audio conditioning rows, so the fourth level
had never existed. The value lookup then found nothing to match.

Everything up to that point was correct, and the log says so: the video reference encoded to
`(37296, 96)` video rows plus `(414, 32)` audio rows in 90.1 s, and the packed sequence came out at
**81542 → 81664 padded**, matching am. 114's host-only prediction exactly. Only the table lookup was
short a row.

**What changes.** `request_step_timesteps` takes an optional `audio_condition_timestep` and the pipeline
passes it **for `ref2va` only**, so t2va's and fl2va's tables stay byte-unchanged and need no rebuild.
`self.task` joins the table's cache key, because a four-level table has more rows per step than a
three-level one and the two must not share a file. The timestep itself is now a named constant rather
than a literal `1.0` in two places.

Also corrected: the `mixed` case's padded length is **89856**, not the 90112 am. 114 predicted. That
prediction used a guessed presentation length and the real one tokenizes shorter; am. 123 had already
recorded `mixed` as an interpolation rather than a measurement. The e2e test now asserts the measured
value per case, so a shape drift cannot pass.

**Method note, and the one piece of good luck in this campaign.** This failure was *loud*. The table is
addressed by matching a row's timestep **by value**, so a missing level raises `IndexError` instead of
quietly selecting a neighbouring row — which is exactly the silent-wrong-modulation failure am. 118 was
about. Had the lookup been positional, or had it clamped, this would have shipped as "ref2va audio
conditioning is subtly wrong" with no gate able to see it. Value-matched lookups over positional ones
are worth their cost.

---

## Amendment 130 (2026-08-06) — the horizontal seam ratio of 2.29x on `video_with_sound` is scene content, not a seam; the ref2va horizontal bar is set to 3.0 on that evidence

**Assumed** (campaign plan, gate 8): that `check_spatial_seams`' 2.0 bar, calibrated on t2va's prompt
and content, would transfer to reference-driven content.

**Measured.** Mesh 4×8, ring params, `l1_small_size=16384`, commit `db76ad3807e`, the
`video_with_sound` e2e case (81664 padded). Seam ratios across the three ref2va shapes:

| case | vertical (x = 448, 896) | horizontal (y = 384) |
|---|---|---|
| `one_image` | pass | pass |
| `video_with_sound` | 1.315 | **2.29 — over the 2.0 bar** |
| `mixed` | 1.203 | pass |

**It is content.** Four independent pieces of evidence, from the frame the check flagged:

1. **The magnified boundary strip shows no discontinuity.** `boundary_strip_y384.png` (rows 354–414 at
   2×) is armchair backs, cushions, window sills and a wainscot band — real horizontal structure that
   happens to sit at mid-height, which is where the tile boundary is.
2. **The elevation is ~9 rows wide, not 1–2.** Per-row mean absolute vertical gradient: 1.13 at y=378
   rising to 5.46 at y=385 and back to 1.86 by y=390. A decoder seam is a discontinuity between two
   adjacent rows; this is the profile of an object edge.
3. **The frame's largest vertical gradient is elsewhere and larger.** 16.06 at **y = 306**, which is not
   a tile boundary, against 5.46 at the boundary. If the boundary were producing artefacts they would
   not be a third the size of ordinary scene structure.
4. **The decode path is identical across every case.** Same decoder, same 48×84 latent grid, same
   tiling. `one_image`, `mixed`, t2va and fl2va all pass with it. A decoder seam would appear in all of
   them, not in one scene.

**What changes.** The two axes get separate bars: vertical stays at **2.0** (measured 1.20–1.32, so it
still has real headroom to catch something) and horizontal is set to **3.0** for ref2va, with the
measurement above as the reason. Recorded as an amendment rather than edited in place, because the plan
is explicit that t2va's bars must not be inherited for reference-driven content and that a new bar needs
a dated entry with its reasoning.

**What this does *not* claim.** A content-sensitive ratio cannot distinguish a seam from an edge at
*any* threshold — 3.0 is a coarse net, not a proof of absence. What actually established there is no
seam is looking at the boundary strip, and that is why `_write` now runs before the checks: the first
time this fired, the check aborted the test before any frame was saved, leaving nothing to look at.

**Method note.** This is STATE.md am. 87 arriving from the opposite direction. That one recorded "a seam
ratio near 1.0 is what a smooth scene gives, not what a correct one gives" — a false *pass*. This is a
false *fail* from the same property. The instrument is one-dimensional and the defect is not, so the
frames are the gate and the ratio is the trigger for looking at them.

---

## Amendment 131 (2026-08-06) — ref2va quality measured on all three shapes; three of t2va's six bars would fail on it, and none of the three failures is a defect

**Assumed** (plan §7, and this is the one place the plan was right in advance): that t2va's VBench and
CLIP bars would not transfer to reference-driven content.

**Measured.** Mesh 4×8, ring params, `l1_small_size=16384`, commit `e10e6dda34e`, `transformer_ref`,
seed 0, 1344×768 / 124 frames / 50 steps. CLIP over 8 evenly spaced frames via `open_clip`; VBench in
its own interpreter over the muxed mp4, same runner and same file format the t2va gate uses. 3 passed
in 1383 s.

| dimension | one_image | video+sound | mixed | **min** | t2va | t2va bar |
|---|---|---|---|---|---|---|
| CLIP prompt alignment | 29.05 | 29.97 | 29.38 | **29.05** | 37.38 | 33.0 |
| subject_consistency | 0.9631 | 0.9344 | 0.9587 | **0.9344** | 0.9804 | 0.95 |
| background_consistency | 0.9569 | 0.9249 | 0.9397 | **0.9249** | 0.9812 | 0.95 |
| motion_smoothness | 0.9957 | 0.9952 | 0.9959 | **0.9952** | 0.9914 | 0.97 |
| dynamic_degree | 1.0000 | 1.0000 | 1.0000 | **1.0** | 1.0000 | 1.0 |
| imaging_quality | 0.4826 | 0.6575 | 0.5826 | **0.4826** | 0.6925 | 0.64 |

**Three of t2va's six bars would fail, and each failure is explainable and not a defect.**

1. **CLIP 29.05–29.97 against 33.0.** Prompt-dependent, not quality-dependent. t2va's bar was
   calibrated on a long dialogue-rich prompt naming four characters and a diner; ref2va's is one short
   clause. The spread *across* ref2va's three cases is 0.92 points, against an 8-point gap to t2va —
   so what the number tracks here is the prompt, not the pipeline.
2. **subject/background consistency 0.9249–0.9631 against 0.95.** These penalise change over time, and
   `dynamic_degree` is **1.0 in every case** — confirmed real motion. The lowest pair belongs to
   `video_with_sound`, the case conditioned on a *moving* clip. Lower consistency with confirmed motion
   is the expected trade, and the alternative reading — a frozen video — is exactly what
   `dynamic_degree` rules out.
3. **imaging_quality 0.4826–0.6575, a 0.17 spread on one pipeline and one prompt.** No-reference IQA,
   and content-sensitive: am. 87 already recorded 0.4884 on a visually perfect night scene against the
   same 0.64 bar. The frames behind the 0.4826 were inspected — a photographic interior with correct
   window mullions, curtain folds and wood grain — and are excellent.

**Where ref2va is better than t2va: motion_smoothness**, 0.9952–0.9959 against 0.9914. Worth stating,
because a table of numbers that are all lower invites reading the whole thing as a regression.

**Bars set**, below the minimum observed, with t2va's own headroom convention (its 33.0 sits ~4 points
under a measured 37.05): CLIP **25.0**, subject_consistency **0.90**, background_consistency **0.89**,
motion_smoothness **0.97** (t2va's, unchanged), dynamic_degree **1.0**, imaging_quality **0.44**.

**Method note.** Recording before setting is what made this legible. Had the bars been inherited, the
run would have failed on three dimensions at once and the obvious reading — "ref2va quality is worse" —
would have been wrong on all three counts. The `None`-bar pass costs one extra e2e run and buys the
distinction between a threshold that does not transfer and a defect that does.

---

## Amendment 132 (2026-08-06) — warm ref2va latency: 73.6 s at the smallest shape, 2.9× better than the cold number the campaign had been reporting; denoise is 85–90 % of it

**Assumed** (am. 127 and every ref2va figure before it): that the cold e2e totals — 210.7 / 270.5 /
372.0 s — described the pipeline. They described the pipeline *plus kernel compilation plus a 50 GB
conditioner weight read*, and were labelled cold, but they were the only numbers on record.

**Measured.** Mesh 4×8, TP=4 axis 0 / SP=8 axis 1, ring, 2 links, `l1_small_size=16384`, commit
`e10e6dda34e`, `transformer_ref`, seed 0, 1344×768 / 124 frames, 49 forwards. Warm window: **one full
warmup generation at the same shape with the same references**, plus a priming `encode_prompt` so the
embedding cache is populated; prepares and export excluded. Both `padded_len` values asserted equal
between warmup and the measured call, and equal to the gated value per case. Wall time, not isolated
device time. 3 passed in 1703 s.

| stage | one_image (46080) | video+sound (81664) | mixed (89856) |
|---|---|---|---|
| Encoder (cache) | 0.1 s | 0.8 s | 0.8 s |
| Reference encode | 0.8 s | **21.6 s** | **22.0 s** |
| **Denoise** (49 forwards) | **66.4 s (90.3 %)** | **164.4 s (85.0 %)** | **187.1 s (86.6 %)** |
| VAE decode | 4.5 s | 4.6 s | 4.5 s |
| Audio decode | 1.8 s | 1.9 s | 1.8 s |
| **Total (compute)** | **73.6 s** | **193.3 s** | **216.1 s** |
| per forward | 1355 ms | 3356 ms | 3818 ms |
| realtime factor | 14.2× | 37.4× | 41.8× |

Against the cold numbers: **2.9× / 1.4× / 1.7×** faster. The one_image gap is the largest because its
cold run paid an 83.4 s conditioner encode that the embedding cache removes entirely.

**Three things this establishes.**

1. **Denoise is 85–90 %**, matching STATE.md's 92 % for t2va. Nothing else is worth optimizing until it
   is, and the two decode stages are already 6 s combined.
2. **Per-forward scales superlinearly but sublinearly in the square**: 1.95× the rows per device
   (5760 → 11232) costs 2.82× the time, against 3.8× if it were pure attention and 1.95× if pure
   matmul. So ref2va at these lengths is **not** in the host-dispatch-bound regime the 2-layer test's
   docstring describes at seq_len 2048–21504 — device work dominates, which changes which levers are
   worth pulling.
3. **Reference encode is 11 % for a video reference** (21.6 s), encoding a 124-frame clip through the
   taps=3 temporal chunking. Second-largest single stage after denoise, and previously invisible.

**Named, with evidence, not yet attempted.** The warm log carries 12 distinct
`No known best blocking for (M, K, N) = ...; using default` warnings at ref2va's shapes — including
`(5760, 5376, 5376)` and `(5760, 7168, 1344)`, which are DiT block matmuls running 50× per forward. The
matmul blocking table has no entry for `seq_local` 5760 / 10208 / 11232, exactly as STATE.md's pending
item 2 records for `measured_sdpa_chunk_sizes` (keyed on 4768 / 9216 / 13632, so dead at 4736 / 4992
*and* at all three ref2va values). Both are configuration sweeps, both need their own measured baseline
and revalidation, and neither is attempted here.

**Method note.** The cold numbers were labelled cold and were still the wrong thing to reason from —
"denoise 232 s" for `video_with_sound` invited the conclusion that the loop was slow, when 68 s of that
was compilation. `warmup()` not accepting `references` was recorded as a non-blocking loose end for most
of this campaign; it was in fact blocking every perf statement.

---

## Amendment 133 (2026-08-06) — hoisting the provably-redundant per-step conditioning upload out of the denoise loop buys nothing measurable, and the reason is instructive

**Assumed**: that re-uploading the conditioning blocks every step was costing real time. The arithmetic
looked compelling — for a full-length video reference the block is 37296 × 96 bf16 = **7.16 MB**, and
the loop was sending it **49 times**, i.e. 351 MB of provably identical bytes, plus 49 host bf16
conversions of 3.58 M elements each. The blocks are invariant by construction and the invariant is
already gated (the loop writes only rows from `num_cond` on and raises if a conditioning row moved).

**Measured.** Same method as am. 132, same shapes, same warm-window definition, commit with the hoist
applied. 3 passed in 1371 s.

| case | baseline total | after hoist | Δ | baseline ms/fwd | after | Δ |
|---|---|---|---|---|---|---|
| one_image (46080) | 73.6 s | 73.0 s | −0.8 % | 1355.2 | 1341.6 | −1.0 % |
| video+sound (81664) | 193.3 s | 193.1 s | −0.1 % | 3356.1 | 3342.3 | −0.4 % |
| mixed (89856) | 216.1 s | 215.4 s | −0.3 % | 3818.5 | 3811.3 | −0.2 % |

**All three are inside the ±8 % single-run noise floor am. 82 established, so none of them is a win.**
And the prediction that mattered is falsified outright: the effect did **not** scale with the block
size. `video_with_sound`'s block is 9.1× `one_image`'s, and its improvement was *smaller*.

**Why.** The loop is asynchronous. `ttnn` enqueues work and the host upload for step *i* proceeds while
the device is still executing step *i−1* — 3.3 s of it at the larger shapes. Host work that is hidden
behind device work costs nothing, and removing it therefore saves nothing. The 351 MB was real and
redundant; it was also free.

**What changes.** The hoist is **kept** — it is bit-identical, strictly less work, and simpler to read —
but it is recorded in `attempts.md` and **not** in `optimizations.md`, which requires a change to be
correct *and measurably better*. Claiming it as a speedup would have been claiming a number inside the
noise band as a result.

**Method note.** Counting bytes is not profiling. "Provably redundant" and "provably costly" are
different claims, and on an asynchronous device the first does not imply the second. The cheap test that
would have settled it before the code change is the one am. 132 already contains: denoise scales
2.82× for 1.95× the rows per device, which is device-work scaling, not host-work scaling — a pipeline
bound on host dispatch would have scaled far more weakly, as the 2-layer test's own docstring records
for seq_len 2048–21504.
