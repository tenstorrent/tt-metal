# MiniMax-H3 on Blackhole Galaxy — state

**Status: t2va and all three fl2va modes green end to end.** Fully warm 63–74 s; denoise is 92 % of it.

History lives in git, not here. The full 113-amendment journal is `git show c0a1a7029b3:STATE.md`, and
every amendment after it is its own commit message (`git log --follow -- STATE.md`). "am. N" below points
there. New amendments continue from **114** per `shared/journal-protocol.md`.

## Working point

Numbers taken at any other shape are evidence about that shape alone (am. 76).

1344x768, 124 frames @ 24 fps → 37 latent frames / 207 audio latents, 50 steps → 49 forwards. Mesh 4x8,
TP=4 axis 0 / SP=8 axis 1, ring, 2 links. Layout `[text | cond | audio | target]`, `rows_per_frame` 1008.
`padded_len` 37888 (t2va) / ~39936 (fl2va); `seq_local` 4736 / 4992. Artifacts `~/h3_t2va_artifacts/`.

```bash
export MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers
export MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache
```

## Pending

1. **Decide whether `MINIMAX_H3_AUDIO_ACCURATE=1` becomes the default.** It takes audio decode from
   10.46 % to **0.45 %** rel RMSE (PCC 99.9990 %, PSNR 67.53 dB) for ~3x the stage, ~+12 % of e2e.
   Nothing technical blocks it. Flipping it obsoletes `AUDIO_RELATIVE_RMSE = 0.12`, which deliberately
   describes the *default* path, and needs `trace_region_size` 450 MB + `--timeout=1200` on the trace
   test. Levers are **exhausted** — see am. 113 and `MiniMaxH3.md` before reopening this.
2. **`measured_sdpa_chunk_sizes` is dead at every shipping shape.** Keys on `seq_local ∈ {4768, 9216,
   13632}`; production is 4736 / 4992, so the tuned `(320, 384)` (−13 % on `to_qkv`) has never run.
   Denoise is 92 % of e2e, so this is the highest-leverage untouched perf item.
3. **Vision encoder has no TP** — built replicated, ~595 M params (`loader_minimax_h3.py`).
4. **Audio decode is layout-bound**: `Conv2d` 4.0 % of the stage against ~70 % layout ops (am. 103).
5. **VAE decode readback**: device-side stitch is a wash and left unwired behind
   `MINIMAX_H3_VAE_DEVICE_STITCH`; the canvas reads at 570 MB/s vs 1.83 GB/s per-tile (am. 106). Next
   lead is `ttnn.clone` / `to_memory_config` to re-materialize it off the slice/concat chain.
6. **T-parallel audio decode is wrong at every factor** — off by default. Halo is *exonerated* (am. 108);
   remaining suspects are the gather-compute-partition round trip in strided convs, resample stages,
   `_t_padding`, final assembly.
7. **`test_fused_conditioner_real_weights` is `xfail(strict=True)`** at 98.6224 % vs a 0.99 bar. Not a bug
   (am. 95: better than its input error predicts). Strict, so it fails if it ever passes — re-derive the
   bar from the production row then. Do not loosen it without a measured floor (am. 76).

## Pitfalls

**Machine.** `tt-smi -r` is **forbidden** (dropped all 32 chips off PCIe on CPLD < 1.16) — use
`tt-smi -glx_reset`, and reset after *every* kill or the next run fails somewhere unrelated
(`bank_manager.cpp:462`) and you blame your code. Timeout-gate every device run. Never pipe one to `tail`
— buffering hides hangs. Never `git add -A`: `internal-prodia/`, `prodia`, `recover-logs/`,
`sweep_results_minimax_h3_encoder/` are unrelated, and three stashes must not be disturbed. VBench has its
own interpreter (`/data/kevinmi/vbench_env/bin/python`) and must not enter `python_env` — it pins numpy<2
and transformers 4.33. `TT_DIT_CACHE_DIR` unset degrades **silently**: 713 s instead of ~64 s.

**A measurement only describes the configuration it ran in.**
- Op config can be set by importing *another module*: the H3 conv blockings come from
  `register_h3_audio_blockings()`, which fires at import of `decoder_minimax_h3_audio`. A harness importing
  only `layers/audio_ops` silently gets `C_in_block = 32` instead of 128 — a different op. This invalidated
  a whole amendment (111 retracting 110). Build the production object; read config off what will run.
- Precision does not transfer across op *kinds*: elementwise fp32 is exact (3e-08), an fp32 **reduction** is
  ~1e-03 — five orders apart. Am. 109 used the first to falsify a claim about the second, and was retracted.
- Nor across op *classes*: a cost estimate from a `Conv2d` profile was wrong in magnitude for `Conv1d`.
- A gate's shape is part of the gate (am. 76). Only test production shapes.
- Prefer direct observation to introspection — a bare table lookup and a module walk both gave confident
  wrong answers in one session; printing lookup *and* result in the same row resolved it.
- A borrowed pattern is a hypothesis (am. 86): `fast_device_to_host` looked 39 % faster because it was not
  moving the data.

**Metrics that lie.** The fl2va anchor keyframe is frame 0 of the t2va run, so a pipeline *ignoring* the
keyframe scored ~0.997 too — use a discriminator the null hypothesis fails (mirrored/fractal keyframe:
0.9964 vs 0.4108). A seam ratio near 1.0 is what a smooth scene gives, not a correct one (am. 87).
`imaging_quality` is no-reference IQA — a perfect night scene scored 0.4884 against a 0.64 bar; prompt and
thresholds are a matched pair. Nothing gates **mesh reassembly order** — every numerics test is
single-device or per-shard; the e2e CLIP gate is what caught it (37.37 → 19.58). Feeding rows *other rows'*
metadata still read 0.999888. t2va's no-regression bar is std **46.05** / frame delta **9.88** / audio peak
**0.076** / CLIP **37.37**, to every digit.

**ttnn.** `l1_small_size` is mandatory (65536 for audio) and a bare `ttnn.open_device(device_id=0)` omits
it — this bit three times; the error is `bank_manager.cpp:462` / "bank size is 0 B", which never names the
parameter and means *unallocated*, not too small. Two-axis all-gather **transposes dim 0**: gathered
`c*rows + r` holds shard `r*cols + c` (am. 105). `ShardTensorToMesh(dim=1)` splits 32 ways across the whole
mesh — production uses `ttnn.mesh_partition(dim, cluster_axis)`. `ttnn.all_gather` takes no `mesh_device`
kwarg. `ttnn.Shape` does not slice. bf16 has 7 stored mantissa bits, so 2^-8 is *half* an ulp — and
bit-exactness against a bf16-rounded golden is the wrong bar, since `ttnn.add` and torch round differently.
`ttnn.slice` at a non-tile row boundary is lossless but untilizes the **whole** input (103 MB/device on the
packed sequence).

**Reference comparison.** Building a reference on `meta` + `to_empty` leaves non-persistent buffers (e.g.
`inv_freq`) uninitialized while strict `load_state_dict` reports success → NaN; construct normally and
assert finiteness over params *and* buffers. HF's `hidden_states[i+1]` is captured **before** the deepstack
add, ours after — taps below 3 are apples-to-oranges; production reads `hidden_states[50]` = layer **49** of
50. H3's `token_tags` marks the **whole vision block** as video while Qwen3-VL's `mm_token_type_ids` marks
**only** `<|image_pad|>` — getting it wrong mis-modulates AdaLN with no PCC signal. `dec_in_proj` is a
Conv1d expecting `(B, C, T)` — do not transpose in. Argument orders differ from the reference's in
`keyframe_condition_noise` / `draw_request_latents`. `stretch` keys on position in the keyframe *list*, so a
lone `last_image` is the geometry anchor and *is* stretched. fl2va at seed 0 will not reproduce t2va at seed
0 — condition noise is the first draw. Tower `head_dim` is 72 padded to 96 with explicit `scale`; the
conditioner's is **128**, not 80.

**Test harness.** A test reading its input from the directory it *writes* skips silently — separate env
vars, and give `_write_artifacts` a `stem`. Never insert a helper between `@pytest.mark.parametrize` and its
`def` (the decorator binds to the helper and the test runs with `l1_small_size = 0`). `warmup` must match
the measured call and assert it (`last_padded_len` equal); pre-populate the embedding cache or the first
timed call pays a device conditioner encode. Do not compare single runs: ±8 % at identical shape and seed
(am. 82) — and the *first* call after compile is not steady state (it read 3x fast). `decode_tile_grid`
returns `((y_starts, lengths, overlaps), (x_starts, …))`, nested. `_probe_streams` returns stream **dicts**;
`_decoded_frames` takes `count=`. **Look at the frames** — seams and flicker are what whole-tensor metrics
hide, and both are parallelism bugs.
