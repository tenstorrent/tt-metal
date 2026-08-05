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

### Audio decode precision — ~2.7x available, measured, not implemented (am. 111)
The fp32 floor is a **multiplier-mantissa** limit, flat in K (1.162e-03 at K=32 through 1.169e-03 at
K=14336): fp32 storage + fp32 accumulate with an fp16-grade multiply, ~11 significand bits at HiFi4.
Splitting operands into `bf16 hi + fp32 residual` beats it — 3.7x on matmul (3 terms; `lo*lo` exactly
negligible), 1.9x on the real `conv_pre` at production blocking, **2.7x** stacked with `C_in_block` 512
(1.857e-03 → 6.987e-04). A device `bfloat16` typecast round trip reproduces torch `.bfloat16()`
bit-exactly, so no host help is needed. Expected chain effect: 10.5 % → ~4 %, at 3x the conv count.
The cost estimate leans on am. 103's `Conv2d` profile (4.0 % of stage vs ~70 % layout) and must be
measured, not assumed. `C_in_block` 1024/2048 are rejected by the op's blocking rules.

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
