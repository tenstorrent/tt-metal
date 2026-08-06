# Campaign: MiniMax-H3 `ref2va` on a 4×8 Blackhole Galaxy

## Goal

MiniMax-H3's `ref2va` task — omni-reference conditioning on an ordered list of images, videos
(with their soundtracks) and audio clips — generating a video and its synchronized soundtrack end to
end on the mesh, with a quality gate that a no-op conditioning implementation would fail.

## Acceptance criteria

| # | Criterion | Proof command | Status |
|---|---|---|---|
| 1 | Host packing is bit-exact against the reference for image / audio / video-with-sound / video-without / nine mixed | `pytest models/tt_dit/tests/models/minimax_h3/test_packing_ref2va_minimax_h3.py` | open |
| 2 | Both `_temporal_position_span` orders exist and provably differ at n ≥ 16 and at production n = 37 | same file, `-k span` | open |
| 3 | Reference encode matches `MiniMaxH3Ref2VAReferenceEncoderStep` at pcc ≥ 0.99 on **real media**, and resolves the same latent geometry | `pytest .../test_references_minimax_h3.py` | open |
| 4 | The typed condition stream leaves t2va and fl2va transformer PCCs **identical**, and a modality-interleaved reference region runs at production residues | `pytest .../test_transformer_minimax_h3.py` | open |
| 5 | t2va and all three fl2va modes still pass end to end, unchanged against the Phase 1 baseline | `pytest .../test_pipeline_minimax_h3.py .../test_pipeline_fl2va_minimax_h3.py` | open |
| 6 | `ref2va` generates a video + soundtrack end to end for every case the shape probe admits | `pytest .../test_pipeline_ref2va_minimax_h3.py` | open |
| 7 | Conditioning is provably not a no-op: same prompt and seed, two references of identical geometry but different content, produce materially different output, and the output resembles the reference it was given more than the one it was not | same file, `-k discriminate` | open |
| 8 | Frames inspected by eye for seams and flicker | `artifacts/round-N/frames/` | open |

## Working point (fixed)

- Mesh **4×8 Blackhole Galaxy**, TP=4 on axis 0, SP=8 on axis 1, ring, 2 links.
- Target generation: **1344×768, 124 frames @ 24 fps** → 37 latent frames (48×84 latent, 1008
  rows/frame, 37296 video rows), 207 audio latents (414 audio rows), 50 steps → 49 forwards.
- Transformer partition **`transformer_ref/`** for `ref2va` (`transformer/` for t2va/fl2va);
  `config.json` byte-identical, so the model geometry is shared.
- AdaLN precompute **on** (`MINIMAX_H3_PRECOMPUTE_ADALN=1`, the pipeline default).
- dtype: DiT bf16 with fp32 `time_embedder`; both VAEs fp32; conditioner bf16.
- Seed 0 unless a test states otherwise.
- Reference geometry (measured, `test_qwen3vl_vision_tower.py` agrees): a reference **image** is
  2048 px short edge with **no area cap** → 128 patches on the short edge; a reference **video** uses
  the 768 px canvas of its own aspect ratio.

### ref2va packed lengths (measured host-only against the reference)

| Request | text | ref video rows | ref audio rows | seq | padded | `seq_local` |
|---|---|---|---|---|---|---|
| t2va (baseline) | ~39 | 0 | 0 | 37710 | 37888 | 4736 |
| 1 image 1:1 | 4104 | 4096 | 0 | 45910 | 46080 | 5760 |
| 1 video + sound 16:9 | 6068 | 37296 | 414 | 81488 | 81664 | 10208 |
| image + video + audio | 10172 | 41392 | 828 | 90102 | 90112 | 11264 |
| 9 images 1:1 | 36872 | 36864 | 0 | 111446 | 111616 | 13952 |

**Changing any of the above after Phase 1 invalidates the campaign.** The e2e case list is set by
the Phase 0 shape probe, which is a *measurement*, not a change of working point.

## Path boundaries

### Upper bound — the most the loop may do unattended

Port host-side layout and media preparation; add the typed condition stream to the transformer and
move its two call sites; wire the existing device audio VAE encoder into the pipeline; select the
transformer partition at construction and fix the subfolder-blind cache keys; add `references=` to
the pipeline call; add ref2va cases to existing device tests; write the three new test files; record
amendments; commit each round.

### Lower bound — the least that counts as a round

One gated change plus revalidation — unless the round's evidence proves no change is warranted, in
which case the evidence itself is the round.

### Can use

The installed reference at
`python_env/lib/python3.10/site-packages/diffusers/modular_pipelines/minimax_h3/` as the contract ·
existing tt_dit components (`conditioning.encode_keyframes`, `MiniMaxH3AudioEncoder`, `vae.encode` /
`encode_clip`, `mrope_position_ids`, `vision_cu_seqlens`, `scheduler.scale_noise`, `common_av.py`) ·
real media from `~/h3_fl2va_artifacts/` · `scripts/run_safe_pytest.sh` · focused host-only tests.

### Cannot use

- Changing the fixed working point, mesh or target after seeing results
- Weakening or skipping a quality gate to make something pass
- A gate whose value would also pass if the feature were entirely absent
- Inheriting t2va's VBench/CLIP thresholds for reference-driven content (am. 80/87)
- Noise-augmenting the audio reference rows, or sampling their posterior — they are clean, `mode()`
- A single shared `_temporal_position_span`
- A second ordering of the packed rows alongside the layout order
- `tt-smi -r`; any device command outside `../shared/device-hangs.md`
- `git add -A` (four untracked trees and three stashes must not be disturbed)
- Installing anything into `python_env` (notably VBench, which pins numpy<2)
- Piping a device run to `tail`
- `--no-verify`, force-push, or rewriting a landed round's history

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Fixed baseline recorded | t2va + fl2va green on `bd12ad2aeb2`, output frozen, transformer PCCs captured |
| 2 | Shape probe verdict | The three ref2va padded lengths are known to fit or not, with numbers |
| 3 | Host semantics bit-exact | Criteria 1–2 |
| 4 | Device conditioning correct | Criteria 3–4 |
| 5 | No regression | Criterion 5 |
| 6 | `ref2va` e2e with a falsifiable gate | Criteria 6–8 — a stop gate fires |

## Known constraints

- `transformer_ref/` is 62 GB locally; no precomputed-AdaLN partition exists for it, so its table is
  built on host once per (checkpoint, schedule) and cached.
- The vision tower is **replicated** (no TP, no SP), which is what satisfies
  `vision_qwen3vl.py:525`'s `sp_factor=1` requirement for multi-block attention. It also means nine
  reference images are nine full-size attentions with no parallel speedup — unmeasured.
- `TT_DIT_CACHE_DIR` unset degrades silently (713 s vs ~64 s), so it is exported in every command.
- Warm latency on this base is unmeasured; STATE.md's 63–74 s is flagged stale.
