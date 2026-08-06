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
