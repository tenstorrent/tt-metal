# Voxtral-TTS on TTNN — status and resumption notes

**Read this first when picking the work back up.** It is written to be self-contained: state,
measurements, the traps that cost time, and what to do next. Architecture detail and the
reference-side findings live in `reference/PROVENANCE.md`; this file is the *work* state.

Branch: `lserbedzija/voxtral-tts-ttnn` (pushed). All work is under
`models/experimental/voxtral_tts/`. Nothing else in the repo is touched.

---

## 1. Where things stand

| Piece | State |
|---|---|
| CPU reference, 3 blocks + tokenizer + end-to-end pipeline | **done**, 30/30 vs upstream |
| Block 3 — codec decoder on TTNN | **CLOSED**, 242x real-time, see §4 |
| Block 1 — 3.4B AR backbone on TTNN | **done — OURS** (`tt/ttnn_voxtral_gpt.py`), the default |
| Block 2 — flow-matching transformer on TTNN | **done** — velocity PCC 0.9999989 |
| **End-to-end on device** (text ids + voice → 24 kHz wav) | **works**, 0 long-form WER errors of 298 words, long-form RTF 0.57-0.64 |
| Codec **encoder** | **impossible** — weights absent from the public release |

**Block 1 now runs on our own implementation, not `tt_transformers`.** The wrapper is DELETED, not
parked behind a flag -- `git show d3dcb0fb7c6^:models/experimental/voxtral_tts/tt/ttnn_voxtral_backbone.py`
if you ever need it back for bisection (it also needs `scripts/export_backbone_hf.py` from the same
commit, plus `HF_MODEL`). Measured against the fp32 CPU reference on real prompts, and end to end
on the 15-case fixture:

| | ours | `tt_transformers` |
|---|---|---|
| prefill, last position | 0.999881 | 0.999564 |
| decode step | 0.99991 | 0.981 |
| decode ms/frame | 34.9 | 48 |
| natural-text WER | 0.88% | 1.17% |

**Performance: 83.7 → ~48 ms/frame, long-form RTF 0.57-0.64** (15 cases, all terminating on `[END_AUDIO]`; see §6.21 for why long-form is the number to quote; §6.31 for the Block 2 sweep that took it there). It touched 47.5 ms / RTF 0.60-0.65 with w2 in BFP8, and 2.5 ms of that was handed back deliberately: w2 cost 77% of the precision stack's accuracy for 15% of its speed (§6.16). Accuracy: Block 1 mean/p90 worst-sample **0.92% / 1.28%**, min PCC 0.999040 (the 8x4 norm grid, §6.18); long-form WER **0 wrong words of 298**.
Per frame: Block 1 ~23 ms, Block 2 **~20.8 ms** (§6.31), host 0.2 ms. w2 is in BFP8 as of §6.13 — the hang that
blocked it is fixed, by not calling `ttnn.conv1d` in the codec.
goes" map at the top with the ceiling for each line item. Two sweeps got here — §6.6 (GQA row fold,
width-sharded RMSNorm in Block 2, qkv fusion) and §6.8 (BFP8 on wqkv/wo, semantic head on device) —
and §6.7 is the one to read first: it explains why the WER headline cannot gate any of this.

**Shipped configuration**, pinned by `tests/test_tt_defaults.py`:
Block 1 mixed precision (BFP8 on FF1_FF3 only) + decode-native heads; Block 2 BFP8 weights at
HiFi4 + fp32 accumulation; no device traces (measured, removed -- see §6); no program-cache
clearing. There are no runtime toggles: every rejected alternative was deleted.

### THE ONE THING THAT WILL WASTE YOUR TIME: fixture case 4

Case 4 is "Hello." — one word of text on 74 prompt tokens, almost all voice conditioning. **The
MODEL is chaotic on it**, and that is not a port defect. Free-running frame counts, 9 correct:

    fp32 CPU reference (no device, no precision loss)   81 /  8 / 57
    tt_transformers + bf16 flow                          9 /  8 / 72 / 88 / 55 / 118
    ours, four different weight configurations          8 to 183

Every implementation, including pure torch, lands anywhere from 8 to 183 frames on the noise draw
alone. The likely mechanism is that one word cannot compete with 74 tokens of voice conditioning
at the last prefill position; the `ssinghal/voxtral_tts` branch reports the same effect
independently ("147 voice frames dilute text signal"). At 1 reference word a rambling take scores
tens of thousands of percent and swamps 340 good words, so `score_quality_set.py` now reports it
in a separate **model-unstable** bucket. It cost real debugging time TWICE — do not read it as a
regression, and do not tune precision against it.

**Block 3 is closed, not merely paused.** Six independent optimization attempts were measured and
all rejected (§4). The single remaining known win is worth ~0.1% end-to-end. Do not reopen it
without a new idea; read §4 and §5 first, because the obvious ideas are already disproven.

---

## 2. How to actually run things (do this before anything else)

There are **two** virtualenvs and neither is obvious.

**Main env — has ttnn, torch, torchaudio, transformers:**
```bash
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD      # from the repo root
```
This is the venv created for the XTTS-v2 work; the repo's own `python_env` is a docs env and does
**not** have torch/ttnn. `python_env` *does* have `black` if you need it (but see trap #8).

**Run the tests** (the repo `conftest.py` needs torchvision, which is absent, hence `--noconftest`):
```bash
python -m pytest models/experimental/voxtral_tts/tests/test_common_ref.py \
  models/experimental/voxtral_tts/tests/test_tokenizer_ref.py \
  models/experimental/voxtral_tts/tests/test_backbone_pcc.py \
  models/experimental/voxtral_tts/tests/test_flow_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py \
  --noconftest -p no:cacheprovider -q
# expect: 118 passed
```
Passing the directory instead of the file list collects nothing — pass the files.

**End-to-end on device, and the quality numbers in §3.1:**
```bash
python models/experimental/voxtral_tts/scripts/generate_quality_set.py
# No HF export and no HF_MODEL: Block 1 is ours and loads the Mistral-native checkpoint directly.
# Those were only ever needed by the deleted tt_transformers wrapper (see §1).
python models/experimental/voxtral_tts/scripts/score_quality_set.py \
  models/experimental/voxtral_tts/generated/results.json
```
All 15 fixture prompts take ~20 min on one N150. `--cases 0,1` for a quick check. The WAVs land in
**`generated/`** (gitignored — audio is large and CC BY-NC derived, so it never leaves the box) and
are the only way to actually *hear* the model. No metric substitutes for that.

**Opening a device for anything with convs** needs `l1_small_size`, or you get
`Out of Memory: ... bank size is 0 B`:
```python
dev = ttnn.open_device(device_id=0, l1_small_size=65536)   # as tests/test_codec_ttnn_pcc.py:31
```
Attention-only harnesses do not need it, which makes this an easy trap in a fresh script.

**Checkpoint** (8.0 GB, gitignored, CC BY-NC 4.0):
`reference/weights/` — `consolidated.safetensors`, `params.json`, `tekken.json`,
`voice_embedding/*.pt` (20 presets). Re-fetch per `reference/PROVENANCE.md`. Without it, 96
reference tests still pass; the 22 device/weight tests skip.

**Second env — upstream comparison only** (needs einops + mistral-inference, no GPU). Recipe and
rationale in `scripts/upstream_compare/README.md`. It lived in a session scratchpad and is **gone**;
recreate from the README. Also run `scripts/upstream_compare/fetch_upstream.py` to re-download the
two pinned vLLM-Omni files (gitignored on purpose).

**End-to-end reference speech** (~50 s of CPU for the reference backbone):
```bash
python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
    --text "..." --voice neutral_male --threads $(nproc)
```

---

## 3. What is validated, and how

**Reference vs upstream: 30/30.** Harness in `scripts/upstream_compare/`.
- Block 1 vs `mistral_inference` (Mistral's own): RoPE table + application **bit-exact**,
  RMSNorm/SwiGLU/repeat_kv bit-exact, full 26-layer stack PCC 0.99999988.
- Block 1 input side + Blocks 2/3 vs vLLM-Omni's `nn.Module`s: 37-code frame **bit-identical
  integers**, waveform PCC 0.99999982, all 8 decoder stages >= 0.99999.
- Tokenizer vs `mistral_common`: **exact token ids**, 15 prompts, 8 languages. Ground truth
  vendored at `tests/prompt_fixture.json`.

**End-to-end reference:** 0.0% WER (Whisper) on 4 runs — 24 words x 2 voices, a **125-word
paragraph in a single pass** (469 frames, 37.5 s), and French via `fr_female`.

**Block 3 on device:** PCC 0.9994–0.9998 vs the reference across T=1…1536, plus a real-speech
fixture (`tests/real_frames_fixture.pt`, 6.4 KB of genuine Block 1+2 output) that asserts **both**
PCC > 0.9999 **and** worst sample < 2% of peak. Default config measures PCC 0.999984 / worst 1.16%.
That worst-sample bound is the gate that matters — see trap #9.

**Blocks 1 and 2 on device, per block:**

| | vs fp32 reference |
|---|---|
| Block 1 prefill (last position, real prompts) | PCC 0.999881 |
| Block 1 decode | PCC 0.99991 |
| Block 2 velocity | PCC 0.9999989; semantic codes exact, 73/74 frame codes exact |

### 3.1 End-to-end on device — the numbers that actually matter

Per-block PCC cannot tell you whether the audio is good, and neither can a teacher-forced code
comparison. These runs are **free-running**: the model is fed its own codes, which is what serving
does. Harness: `scripts/generate_quality_set.py` then `scripts/score_quality_set.py`.

- **Intelligibility — 1.17% WER over 341 natural-text words, and every deviation is a scoring
  artefact.** 12 of 12 natural-text cases: English (4, incl. two paragraphs), French, German,
  Italian, Portuguese, Hindi all exactly **0.0%**. The two non-zero ones are not TTS errors —
  Spanish 50% is the model correctly reading "42" aloud as "cuarenta y dos" against a reference
  containing the digits (3 phantom errors on a 6-word clip), and Arabic 33% is one grapheme
  (`حالك`→`حاله`) on a 3-word clip. This matches the reference's own 0.0% baseline, so
  **Block 1's bf16 weight floor costs nothing measurable in intelligibility**.
- **Long-form is the strong result.** The 676-char (125-word) paragraph: **0.0% WER at 459 frames
  / 36.7 s** (`neutral_male`) and **0.0% at 489 frames / 39.1 s** (`casual_female`). 459 steps of
  autoregressive decode with no drift, no repetition, no collapse. The reference generates the same
  paragraph in 469 frames / 37.5 s — within 2% of the device's duration, independently.
- **Termination: 15/15 prompts stop on `[END_AUDIO]` naturally.** Nothing runs away.
- **Artifacts: clean.** Across all 15 — 0.000% clipped samples everywhere, |DC offset| < 3e-4,
  0 clicks on 13/15 (2 on one German clip), peak 0.10–0.71 so nothing is near full scale.
- **Voice identity carries through all three blocks.** Fixture cases 0 and 2 are the same
  `neutral_male` preset reading *different* text, which is a positive control: their long-term
  spectra match at **0.984** cosine, the highest pair by a clear margin (next is 0.882), and median
  F0 separates cleanly — males 110.6 / 106.7 Hz vs females 205.1 / 192.0 Hz, with the two male
  readings agreeing within 4 Hz on unrelated text.

**Listening pass: done, informally — verdict "sounds good".** Re-done on the POST-BLOCK-2-SWEEP build
(2026-08-06, `82d04f977a1`, Block 2 at ~21 ms/frame after §6.30/§6.31) via
`generated/SAMPLER_current_build.wav` — one 136 s file, 13 clips, both 37 s and 39 s long-form English
cases first, then prosody, then eight languages. Verdict from the author: **"sounds good"**. The earlier
pass was on the pre-sweep build (2026-07-30, verdict "sounds ok"). That clears the bar of "no audible defect the metrics
missed", which is what it was for. Read it as exactly that and no further: **it is not evidence of
listener-facing naturalness.** WER measures intelligibility, and one developer saying "ok" is not a
substitute for a MOS-style eval with real raters and a side-by-side against the fp32 reference
(`generated/ref_current.wav` and the `case11_*_REF.wav` pair are there for that). If someone needs
to claim naturalness — for a customer, a demo, or a comparison against another vendor — that eval
has not been done.

### 3.2 bf16 in Block 2 — what was checked before flipping the default

Block 2's output is 36 integers rounded onto 21 FSQ levels, so PCC is the wrong gate: what matters
is how often a value crosses a rounding boundary, and whether that compounds over hundreds of
autoregressive steps. All 15 prompts were regenerated at bf16 and rescored.

| | fp32 | bf16 |
|---|---|---|
| acoustic codes differing from the reference | 0.81% | 1.16% (all off-by-one) |
| semantic codes wrong (12 seeds, batch 2) | 0/24 | 0/24 |
| **natural-text WER, 341 words** | **1.17%** | **1.17%** |
| natural `[END_AUDIO]` termination | 15/15 | 15/15 |
| 125-word paragraph, free-running | 459 / 489 frames | **458 / 490 frames** |
| voice identity (same-voice vs next-nearest) | 0.984 / 0.882 | 0.985 / 0.884 |

The WER total is unchanged and its *composition* moved: German and Hindi each gained one word-error,
Spanish lost three. All single-token, in both directions, on 3–7 word clips where one word is
14–33% WER. The Spanish case shows it is divergence and not degradation — bf16 read "42" as digits
where fp32 spelled out "cuarenta y dos", and bf16 only scores better because the reference text
contains digits.

Two things were checked and one was a false alarm. Across 12 matched natural-speech pairs the output
is **~2% quieter** (median, ~0.2 dB, inaudible; negative in 11/12) and **F0 is unchanged** (median
−0.25%, negative in only 6/12). An apparent downward F0 bias seen on the first 3 pairs did **not**
survive the larger sample — the two large positives came from 0.7–2.9 s clips where a median-F0
estimator is unreliable. Spectral similarity between the fp32 and bf16 renderings is 0.9875–0.9992
on 11 of 12, i.e. tighter than the 0.984 that two utterances of the *same voice* score against each
other.

**Not covered:** nobody has A/B listened to fp32 vs bf16. Matched pairs are in `generated/` as
`caseN_<voice>.wav` against `caseN_<voice>_bf16flow.wav`. Level-match first — the 2% median is
irrelevant but cases 0, 1 and 5 differ by 11–16%, enough to bias a casual comparison.

Three fixture texts are deliberately adversarial — emoji, `!@#$%^&*()`, literal `\t`/`\n`. The
model tries to **vocalise** them (it renders `1234567890` correctly as "1 2 3 4 5 6 7 8 9 0", then
speaks the symbol run), so there is no well-defined reference transcript and they are reported
separately rather than folded into the headline WER.

**These are a model limitation, not a port defect — checked, not assumed.** Running the fp32 CPU
reference on the same two prompts: on the emoji text it collapses into a repetition loop
("emoji test kalinsan keep on faith i can say i can say i can say…", 6257% WER) where the device
at least keeps "and caps and mixed" (71%); on the symbol text both produce comparable nonsense
(reference 333%, device 450%). Duration agrees to within 2 frames on the symbol text (reference
242, device 240) and diverges on the emoji text (121 vs 64) — expected, since free-running
generation is chaotic and one differing code sends the two down different trajectories. **Do not
read the device being "better" on case 11 as a quality claim**; both outputs are unusable, and the
lesson is that the model is brittle on non-speech input. Sanitising or spelling out such text
belongs upstream of the model.

---

## 4. Block 3: performance, and everything that was tried

Warm, N150, defaults:

| T | audio | warm | RTF | vs real-time |
|---|---|---|---|---|
| 64 | 5.1 s | 43.9 ms | 0.0086 | 117x |
| 469 | 37.5 s | 155.5 ms | 0.0042 | 241x |
| 1500 | 120 s | 539.4 ms | 0.0045 | 222x |

Upstream report RTF **0.103 for their whole pipeline** on an H200, so Block 3 is ~4–8% of the total
budget. **It is not where the end-to-end answer gets decided** — Block 1 is (87% of the parameters,
12.5 sequential steps per second of audio). Do not over-invest here.

### Applied, with what each was worth
1. **bf16 attention** — best accuracy of four dtype configs *and* faster. Re-verified on the
   worst-sample metric, not just PCC (see below).
2. **Chunked windowed attention**, slab 512 — O(S²) → O(S·slab). At S=12000: warm 892→497 ms,
   cold 10580→1178 ms, mask 2304 MB→4.2 MB. **Exact**, not an approximation: chunked and unchunked
   give byte-identical worst-case error at all four stage lengths.
3. **Uniform slab-sized chunks** — one cached bias per window. Bias cache 23 tensors/53 MB →
   6/18.1 MB, and it stops growing with utterance length.
4. **Conv length bucketing**, 128 — on a stream of 12 distinct lengths, 120.9 s → 1.66 s (73x).
5. **Hoisted conv weight preparation** — 2.6x at T=128 (112.8→43.7 ms); host share 88%→24%.
6. **Content-deduplicated prepared weights** — 730 MB → 98 MB, bit-identical.

### Measured and REJECTED — do not retry without new information
| idea | result |
|---|---|
| **sdpa** for the attention interior | 1.44–2.27x FASTER, 3.3x worse worst-case error, failed 11 tests. See §4.1 |
| **Smaller slab** toward the compute optimum 2·window | slab=32 computes **9x fewer** scores and runs **9x slower** (334 vs 36 ms/pass) |
| **Device trace capture** | **1.00x** at every slab size, including 3570 ops. The async command queue already hides host dispatch behind device execution |
| **Batching the chunks** into one matmul | batched attention is 2.4x faster, but the gather costs more than it saves (11.30 vs 9.67 ms). Chunks overlap by `window`, and ttnn has no strided view, so the copy is unavoidable |
| **Unchunked attention** with a full [S,S] mask | identical accuracy, 3x slower at S=4096, mask grows to 268 MB. Only S<=1024 prefers it — a `chunk_min` question |
| **Changing the dtype default** | the current default already wins on *both* PCC and worst-sample |

The recurring lesson: **on this hardware, doing less arithmetic does not mean finishing sooner.**
Three of these reduce FLOPs and all three are slower. Cost is dominated by per-kernel overhead and
memory traffic.

### 4.1 The sdpa investigation (the deepest hole; read before touching attention)

sdpa **is** faster and its API **does** support ALiBi (via `attn_mask` with `is_causal=False`). The
old note claiming "no ALiBi support" was wrong. It is rejected on **accuracy**: against an exact
fp64 answer, worst-case error 1.95e-02 vs hand-rolled 5.85e-03.

Ruled out as the cause, each by measurement:
- our ALiBi mask — an **all-zero** mask still gives 4.7x (1.135e-03 vs 2.395e-04)
- `MASK_NEG` magnitude — identical to 4 digits from -1e4 to -1e30
- ALiBi precision — per-head error is uniform, *anti*-correlated with slope
- online-softmax rescaling — 1 k-block vs 4 k-blocks: identical
- `exp_approx_mode`, `math_fidelity` HiFi3/HiFi4, `fp32_dest_acc_en`
- legacy v1 vs streaming v2 kernel (`fp32_dest_acc_en` selects between them!) — 2.066e-02 vs 2.163e-02
- the fp32→bf16 input conversion — costs 3.70e-03, identically for both paths
- **patching `sdpa_program_factory.cpp` so `im_df`/`stats_df` are Float32, and rebuilding** —
  marker-verified live (`im_df=Float32`), error unchanged. So closed issue #13364 is a red herring.

Established: a **mask-independent precision floor intrinsic to the fused kernel**, 3.7–10.7x worse
worst-case. Telling detail: hand-rolled error *falls* with S (4.57e-04→1.27e-04 for S=128→2048)
because softmax averages more values, while sdpa stays flat (1.7e-03→1.4e-03). sdpa has a
fixed-magnitude error term that does not benefit from averaging. Numerical attribution matches
"bf16 accumulation of P@V" (6.8e-04) rather than "bf16 scores+probs" (1.1e-04, which is what
hand-rolled's 2.4e-04 is consistent with).

**Where it stopped:** the one-k-block result rules out repeated cross-block accumulation, so it is a
single fixed-precision step inside the fused pipeline. Pinning it further needs instrumenting
`kernels/compute/sdpa.cpp` to dump intermediates. Not done.

**This matters for Block 1** — see §7.

---

## 5. Traps that cost real time — read these

1. **A failed trace capture wedges the device.** If an exception escapes between
   `begin_trace_capture` and `end_trace_capture`, `close_device` hangs and **every later run on the
   card blocks**. Cost ~20 min and required killing by PID. Always `end_trace_capture` in a
   `finally`. (No trace code remains in Block 3, but Blocks 1/2 will want tracing.)
2. **Prepared conv weights are NOT length-independent.** Same shape at every length, *different
   values* — cross-length reuse computes PCC **0.19**. Shape equality and absence of a crash are not
   correctness. The cause: ttnn lowers conv to an implicit GEMM whose weight operand is
   `[C_in·K, C_out]` (no L in it, hence the constant shape), but the sharding/blocking plan is
   chosen from the GEMM's row count `N·L_out`, and the weight is pre-arranged to match that plan.
3. **`prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype**, not the weight dtype. Getting it
   backwards silently produced PCC 0.008.
4. **A stale benchmark outlived what it measured.** bf16 weights looked ~20% faster; that gap was
   conv weight-prep cost. Re-measure defaults after any change to the hot path.
5. **"Time is in the TTNN wrapper" != "time is dispatch."** A profile showed 98% inside
   `ttnn/decorators.py:__call__` and I concluded tracing was the fix. It was work inside *one op*.
   Tracing later measured 1.00x. Ask *which ops*, not just how much.
6. **Synthetic gates let the audible path rot.** Four numerical changes landed without re-running
   real speech. Now pinned by a 6.4 KB fixture. If you optimize Blocks 1/2, add the equivalent
   fixture test *first*.
7. **Comments outlive code.** Every review pass has found one: the conv-upload comment, an
   "fp32 attention" docstring after bf16 became default, a wrong sdpa rationale, and the same
   optimization quoted as both 2.4x and 2.6x. Re-read the docstring of anything you change.
8. **`models/experimental` is NOT covered by the repo's black config** (pyproject `include` lists
   `models/demos`). Black flags 13/16 files; the sibling xtts_v2 code fails it too. **Do not
   reformat.** Do keep lines <= 120.
9. **PCC HIDES OUTLIERS — the single most expensive lesson.** It is a correlation: it can sit at
   0.9998 while individual samples are badly wrong, and for audio the outliers are what you hear.
   sdpa passed at PCC 0.9998 per slab and still failed 11 tests, one stage measuring PCC 0.916.
   Always pair PCC with a worst-sample bound.
10. **Verifying a C++ rebuild is harder than it looks.** Three separate traps in one experiment:
    - `nohup cmd &` inside a backgrounded tool call gets reaped when the wrapper exits; the harness
      reports the *wrapper's* exit code, not the build's.
    - `ldd` resolved `_ttnncpp.so` from `build_Release/lib/`, a **stale separate copy**, not the
      freshly linked `build_Release/ttnn/` one. `LD_LIBRARY_PATH` fixes it (it is RUNPATH, so
      LD_LIBRARY_PATH wins).
    - **A relinked `.so` with a fresh timestamp is NOT evidence your source change is in it.** The
      first build reported exit 0, relinked both `.so`s, and had never recompiled the edited file
      (it only rebuilt `tt_metal/fabric`). Only a runtime `log_warning` marker proved it. Note
      `log_debug` is compiled out in Release, so it cannot be used for this.
11. **`fp32_dest_acc_en` selects the sdpa kernel variant**, it is not just a precision knob:
    `can_use_streaming_compute(fp32_dest_acc_en) { return !fp32_dest_acc_en; }`. Passing `True`
    forces the legacy v1 kernel. Holding it `True` in every test meant streaming v2 was never
    exercised for a long time.

12. **RANDOM ACTIVATIONS ARE A PESSIMISTIC PROXY FOR WEIGHT PRECISION — this one cost the most
    time on Block 1.** Feeding `randn * 0.02` embeddings through 26 layers reported PCC 0.892 for
    BFP8 MLP weights vs 0.969 for bf16, which looked like a crisis and drove a long precision hunt.
    On **real tokenized prompts** the same two configs score 0.99938 and 0.99970 — the entire
    spread is 3e-4, and every variant clears 0.999. Random embeddings are off-manifold, so
    activations do not sit in the range the weights were trained for, and block-float (one shared
    exponent per block) is punished hardest by an inflated dynamic range. Never quote a
    low-precision-vs-fp32 PCC from synthetic activations; use `tests/prompt_fixture.json`.

13. **A too-small frame budget is indistinguishable from a model that will not terminate.** Frames
    are 12.5 Hz and speech runs ~18 chars/s, so the 676-char fixture paragraphs need ~460 frames.
    Run them at `max_frames=200` and they stop mid-sentence and report "no natural stop" — which
    reads as a generation bug and sent me looking for one in the KV-cache padding. It also
    correlated with prefill length purely because the long texts are the long prompts.
    `generate_quality_set.py` derives the budget from text length for this reason.

14. **Three ASR-harness traps, all of which fake a bad TTS result.** Whisper's encoder is a fixed
    30 s window, so a plain `generate` call on a 37 s clip silently transcribes the first 30 s and
    charges the rest as deletions (~20% phantom WER); an `[^a-z0-9' ]` normaliser erases Hindi and
    Arabic and scores them 100% on perfect audio; and a voice-name prefix is not a language —
    fixture case 4 is English "Hello." spoken by `ar_male`, and forcing Arabic decoding made
    Whisper hallucinate a filler and report 100% WER on one word. All three are handled and
    documented in `scripts/score_quality_set.py`.

---

## 6. Open items

### Accuracy bookkeeping — read §6.15 before quoting any accuracy number
- **The decode gate's prompt-to-prompt spread (0.45 pp mean, 0.96 pp p90 over 8 prompts) is larger
  than every change that was gated with it**, w2's 0.10 pp included. §6.8 reported a 2-prompt pair to
  0.01 pp and did not record which two. The gate is sound for **paired, same-session, same-prompt**
  A/B — it is deterministic and repeats bit-identically — and unsound for absolute levels or
  cross-session comparison. Gate on all 8+ prompts, recorded by index, both arms in one session.
- **§6.8's absolute levels are unreachable today and the cause is unidentified.** It records mean
  0.84% / p90 1.06%; the best of 28 prompt pairs is 0.99% / 1.38%. Gate code, Block 1 code, the
  reference, the ttnn build and prompt selection have all been ruled out by direct check — see §6.15
  so nobody repeats them. Every "indistinguishable from shipped on mean/p90" call in §6.8 rests on
  those levels.

### Block 3 — closed, with two deferred decisions (not defects)
- **`chunk_min` should probably be 1024, not 512.** Measured crossover is S ~ 2000: at S=1024
  unchunked is 1.5x faster (1.40 vs 2.10 ms). Worth 1.06x on attention, ~1.7% of the block,
  **~0.1% end-to-end**, and costs a 16.8 MB bias. Recorded in the code; not worth a commit alone.
- **`bucket=128` is wrong for streaming.** A 1-second chunk costs the same 43.7 ms as a 10-second
  one, because everything below 128 frames pads to 128. Streaming wants 16 or 32 (floor: 12.8 ms at
  bucket 16). A **geometric** ladder (64,128,192,256,384,512,768,1024,1536) would beat the uniform
  grid — worst-case relative padding drops from 98% to ~50% — but tuning it needs a real
  utterance-length distribution.
- **Streaming needs boundary state throughout**, not just a bucket change. Three places fabricate
  or discard context that a continuing stream would supply: the transposed convs' trimmed tail, the
  forward convs' `_pad_causal` (which would replicate the chunk's own first frame instead of the
  previous chunk's real tail), and attention's first `window` rows. Ordinary work, but list it.
- **batch=1 only.** Upstream serve concurrency is 32. Not needed yet.
- **bf16 weights + bf16 attention has almost no margin**: 1.93% worst-sample against a 2.00% gate,
  for a ~1% speed gain. The knob exists for experiments; do not ship it.
- Minor: the suite tests to T=469; T=1500 is measured by hand but not pinned.

### Block 1 — DONE, ours (`tt/ttnn_voxtral_gpt.py`)

3.4B params, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RMSNorm, RoPE θ=1e6, tied
embeddings, **`n_heads*head_dim` (4096) != `dim` (3072)** so wq/wo are not square. Loads the
Mistral-native checkpoint directly — no HF export, no `HF_MODEL`, no RoPE-permute hazard.

Beats the `tt_transformers` wrapper it replaced on every metric: prefill last-position PCC
0.999881 vs 0.999564, decode 0.99991 vs 0.981, 34.9 ms/frame vs 48, natural-text WER 0.88% vs
1.17%. Read the module's "where the time goes" map before optimizing — it lists the ceiling for
each line item, not just the cost.

**What is left in Block 1, honestly:** not much. 23.6 of its 34.9 ms is weight streaming at
194 GB/s, which is the measured ceiling for a plain interleaved matmul here (hand-tuned matmul
program configs are SLOWER — 169 vs 193 GB/s on wq). 6 ms is RMSNorm whose fp32 accumulation is
load-bearing. So only fewer weight BYTES help, and that is capped by the hang below.

**THE HANG, and the precision boundary it imposes.** All-BFP8 weights are ~5 ms/frame faster and
trigger a silent, card-wedging hang in multi-utterance runs (recovery needs a `tt-smi` board
reset). It needs five things at once — all-BFP8 Block 1, Block 2 in the loop, Block 3 on device,
two distinct codec buckets, and a generation between two same-bucket decodes. Remove any one and
it completes. **BFP8 on FF1_FF3 is safe; adding FF2 brings it back.** `tt_transformers` never saw
it because it uses the same mixed precision. Ruled out by measurement, so do not retry: memory
(flat, 8 GB free at the hang), program-cache COUNT (576 entries fine; we died at 310–341 while
tt_transformers lived at 329), Block 3 length/content, a Block 1 leak (1400 steps clean), and
every distinctive Block 1 op. The underlying ttnn failure — a hang rather than an error — is
unreported upstream and still unexplained.

**Device trace** of the decode step is implemented and correct (PCC 0.99982 traced vs 0.99985)
and OFF: at equal window it is 0.7 ms SLOWER, because decode here is device-bound, not
host-dispatch bound. Tracing Block 1 AND Block 2 together also works now and is bit-identical
(0 differing codes of 2849) after fixing the capture ORDER — all warm-ups before any capture —
but costs ~6 ms/frame, so both stay off. The `ign/voxtral_p150_qb2` branch reports decode as
dispatch-bound; that is Blackhole, and it does not transfer to this N150 — see §6.5.

### 6.5 — `ign/voxtral_p150_qb2`, measured

Their code will NOT run on our tree (`TT_THROW: Only L1 buffers can have an associated circular
buffer!`; they sit ~1100 commits back). So their tt-metal was built from their own commit into
`/localdev/lserbedzija/ign_build`, with a separate python env at `/localdev/lserbedzija/ign_venv`
that layers our site-packages WITHOUT our ttnn editable-install finder — that finder is a meta-path
hook and wins over PYTHONPATH, which is why the obvious `PYTHONPATH=` approach silently kept using
our ttnn. Their build runs fine on this Wormhole N150 (8x8 grid).

    pytest models/experimental/voxtraltts/tests/perf/test_e2e_performant.py -q -s -k F128
    HF_MODEL=<our weights dir>   # their loader takes the same Mistral-native layout

**Result: 598.1 ms/frame, 1.67 frames/s over 128 decode steps** (build 24 s, warm-up 121 s), against
our 77.6. No spills, no host fallbacks, trace+1CQ, weights all-BFP8.

READ THAT WITH THE CAVEAT IT DESERVES. Their branch targets a Blackhole P150 — bigger grid, more
DRAM, and a config tuned for it. This measures THEIR CODE ON OUR CARD, which answers "could we just
adopt theirs on N150" (no) and NOT "is their P150 work slow" (unmeasured; we have no P150).

Two things it does settle. Their own comment calls the acoustic FM core "the 78%/step bottleneck",
which independently reproduces our Block 2 finding on a different implementation. And their Block 1
runs ALL-BFP8 — the exact config that hangs for us — without hanging, which is more evidence the
hang is a tt-metal-version property rather than something about our weights.

**Still open:** prefill beyond ~1024 tokens would need chunked prefill. batch=1 only.

### Block 2 — DONE, and the accuracy is not the problem
390M, 3 layers, **3-token sequence**, bidirectional (no RoPE, no mask), 7 Euler steps per frame with
CFG batched to 2B. Implemented in `tt/ttnn_voxtral_flow.py` at fp32/HiFi4: velocity PCC 0.9999989,
semantic codes exact. Semantic argmax and FSQ quantise stay on host (see that module's docstring).

**THE BIG ONE — a batched matmul re-reads the whole weight per batch element.** CFG makes the trunk
a batch-2 forward, and that was doubling every weight read for nothing. Measured on
`[*,3,3072] @ [3072,9216]`:

| activation | ms | effective |
|---|---|---|
| `[1,3,K]` | 0.294 | 192 GB/s |
| `[2,3,K]` (CFG batch 2) | 0.581 | 97 GB/s |
| `[4,3,K]` (CFG batch 4) | 1.154 | 49 GB/s |
| **`[1,6,K]` (batch 2 FOLDED into rows)** | **0.296** | **191 GB/s** |
| `[1,32,K]` (full tile row) | 0.294 | 193 GB/s |

Cost scales linearly with batch; folding the batch into the ROW dimension is **free**, because 6 rows
still fit one 32-row tile. A linear applies per row independently, so `_trunk` now runs the whole
trunk as `[1, B*3, 3072]` and only splits the batch out for attention. **Worth 2.23x on device time
(136.9 -> 61.3 ms) with BIT-IDENTICAL output** — verified against the previous build's WAVs on three
prompts including a 458-frame clip. This is what the "13% of DRAM peak" figure actually was: a single
matmul reaches 66% of peak, and CFG batching was throwing half of it away.

**Corollary — ~26 of 32 tile rows are still free, and that is throughput for nothing.** `[1,32,K]`
costs the same as `[1,6,K]` (0.294 vs 0.293 ms), and the fold uses only 6. Nothing in a single
utterance can use the rest: the Euler steps are sequential and frames are autoregressive. But
CONCURRENT REQUESTS fit exactly — 5 simultaneous utterances fold to 30 rows at essentially the same
per-frame cost, i.e. ~5x throughput free. Latency per stream would not improve; only throughput.

Running the two CFG branches as separate passes instead of folding was measured and is strictly
worse: 2 passes cost 0.586 ms against 0.293 folded, exactly 2x. Passing the SAME weight tensor to
both costs the same as two different weights (0.586 vs 0.587), which shows there is no cross-op
weight reuse -- every `ttnn.linear` streams its weights from DRAM independently. There is also no
parallelism to win on one device: a single matmul already spans the core grid and both branches want
the same bytes from the same DRAM. On two devices it would be a different question.

It also makes bfp8 weights unnecessary: bf16 **with** the fold (61.3 ms) beats bfp8 **without** it
(98.3 ms). `WEIGHT_DTYPE` stays `None`; bfp8 remains measured and available but costs accuracy for
less than the free fix delivers.

Also applied: **k and v share one weight and one matmul**, split by `nlp_create_qkv_heads` which
builds the head layout in the same op — 3 linears + 6 reshape/permutes become 2 linears + 1 op.
Worth 1.6%, numerically identical. The old comment claiming the fused op "needs a batch layout this
does not have" was simply wrong.

**DRAM-sharded weights — available, but the ceiling is ~1.5x and it is real plumbing.**
`create_dram_sharded_mem_config` + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, as
`tt_transformers` uses for decode. Probed at Block 2's real shape and it needs THREE things to line
up, each a hard TT_FATAL, found one at a time:
1. the ACTIVATION must be L1 WIDTH_SHARDED (`input_tensor_a.is_sharded()`, matmul_device_operation
   .cpp:1226ff), not just the weights;
2. the shard grid must be **TENSIX** cores, whose grid is 8x8 — using the 12-wide DRAM bank grid
   fails with "No core coordinate found at location: (8, 0, TENSIX, LOGICAL)". So the activation
   shards over at most 8 cores here (3072/8 = 384, tile-aligned), not 12;
3. the OUTPUT memory config must be sharded as well.
4. **`M` must be exactly ONE tile** — `TT_FATAL(M == 1, "currently only support in0 tensor height of
   tile height")`, matmul_device_operation.cpp:1245. Our folded trunk qualifies (6 rows in one tile);
   anything taller does not, so `M=64` is categorically unsupported.

With all four satisfied at `M=32`, it still fails to build: **circular buffers grow to 1,678,528 B
against L1's 1,499,136 B**, ~175 KB over. With only 8 compute cores and N=9216, `per_core_N` is 36
tiles and the streaming buffers plus the output shard do not fit. Shrinking `per_core_N` means more
cores — `K/cores` must stay tile-aligned so cores must divide 96, and 48 would give `per_core_N=6` —
but the weight shard lives on the **DRAM** grid (12 banks, y=1) while the activation lives on a
**Tensix** grid, so the two grids cannot both be 48, and that is exactly the `per_core_N` vs DRAM
shard width mismatch model_config.py:716 warns gives **silently bad PCC**.

So this needs a deliberate design pass reconciling three grids (DRAM banks, compute cores,
`per_core_N`) under an L1 budget, with the integer-code gate wired in from the first run. Not a
drop-in.

**The ceiling is modest.** After the batch fold, a plain interleaved matmul at this shape already
runs at **192 GB/s, ~66% of peak**, so perfect sharding is worth at most ~1.5x on the matmuls, and
less end to end. Block 1 is the larger target now (~48 ms of an ~80 ms frame). Do this as deliberate
work with a numerical gate, not as a quick win.

**Measured and REJECTED — sdpa in the attention interior.** `ttnn.transformer.scaled_dot_product_
attention` fuses 7 ops into 1 there (2x `repeat_interleave`, transpose, matmul, scale, softmax,
matmul) and supports GQA natively, so k/v go in unexpanded and the 4x expansion disappears. It
works and it is faster, but not for free:

| | hand-rolled | sdpa |
|---|---|---|
| frame, batch 2 | 145.5 ms | 122.3 ms (**1.19x**) |
| velocity PCC (worst of 12 seeds) | 0.9999428 | 0.9990333 |
| acoustic codes differing from reference | 1.16% | **4.28%** |
| max deviation | 1 FSQ level | **2 FSQ levels** |
| frames exactly right | 6/12 | 3/12 |

**3.7x more differing codes, and that factor is not a coincidence** — §4.1 measured sdpa's
mask-independent worst-case penalty in Block 3 at 3.7–10.7x. The same characteristic reproduced here
at the bottom of that range, which says the penalty travels with the op rather than with the shapes.
Rejected on sequencing, not on principle: tracing attacks the same dispatch bottleneck for **zero**
accuracy cost, and bf16 has already spent some of the code-error margin that protects the 0.0% WER
(0.81% -> 1.16%). Revisit **after** tracing, and gate it on end-to-end WER and long-form frame
counts rather than on code-diff counts. Note that sdpa's 19% will shrink once tracing removes the
host-dispatch component the two share. `is_causal` **defaults to True** and must be passed `False`
here; causal masking would cut position 0 off from the time and LLM tokens, which does not raise --
it silently produces a velocity conditioned on nothing.

**Tracing: IMPLEMENTED, verified, and worth 0.1% today — but keep it, it is a prerequisite.**
`_solve()` is a pure device graph and `_trace()` captures it (one trace per batch/n_steps/cfg_alpha).
Correct: 0/74 codes differ from the reference, repeatable, input refresh verified. The measurement
that matters, at batch 2:

| | host enqueue | device wait |
|---|---|---|
| traced | **0.0 ms** | 139.1 ms |
| eager | **121.7 ms** | 139.3 ms |

That 121.7 ms "host enqueue" figure is **backpressure, not host cost** — enqueueing faster than the
device drains blocks the enqueue call. The real host overhead is the constant gap between the traced
device floor and the eager frame time, and it is ~6.5 ms at every configuration measured (bf16 W
139.1/145.5, bfp8 W 100.5/107.0, bfp4 W 90.1/96.6). **So tracing is worth ~6.5 ms — about 6% at the
current device floor.** An earlier version of this section claimed dispatch would become a binding
wall at 121.7 ms once device work dropped; that was wrong, built on the backpressure misreading.

Also wrong in that earlier version: predicting the Block 3 tracing null result would not transfer,
on the grounds that a 3-token sequence means tiny ops. A tile is 32x32, so every matmul does 32 rows
of work for 3 useful tokens against 3072x9216 weights. Upstream's 47%-from-CUDA-graphs does not
apply — their bottleneck was launch overhead.

**Tracing is GONE from both blocks -- deleted, not disabled.** Both captures were built, proven
bit-identical, and measured on this N150: Block 1's decode trace showed no win, and Block 2's was
~6 ms/frame SLOWER. Under the no-toggles rule a dormant second path is not worth its weight, so the
capture machinery went with them (`git show 5beef54ad3c` for Block 1's, `d0653e476a6` for the
removal). The earlier "~6% from tracing" figure came from a device-bound A/B that the decode-native
work has since invalidated. If you rebuild it: read trap #1, and remember that Block 1 and Block 2
must finish ALL warm-ups before EITHER captures, or the second capture hangs.
Upstream's 47%-from-CUDA-graphs figure does not apply to us: their bottleneck was launch overhead,
ours is arithmetic. Read trap #1 before touching capture code.

### 6.6 — the optimization sweep: what landed, what failed, what is left

Swept the repo, the TTNN API and the shared branches against our two hot blocks. The framing that
made the difference: **Block 2 was not weight-read bound, whatever its docstring said.** Its
velocity net is 349M params at BFP8, so 7 Euler steps stream ~2.6 GB, and at the 194 GB/s Block 1
demonstrably reaches that is a 13.4 ms floor against a measured 35 ms — 38% of ceiling. The gap is
OP COUNT (~88 ops per step, 18 of which carry weights). Every win below removes ops, not bytes.

**Landed — 77.6 → ~66 ms/frame, RTF ~0.97 → 0.81–0.90, WER 0.88% unchanged, 15/15 termination:**

| change | where | gain | accuracy |
|---|---|---|---|
| **GQA row fold** — reshape q `[B,32,3,d]`→`[B,8,12,d]` instead of `repeat_interleave` on k/v | Block 2 `_block` | **1.40x on the block** | same velocity PCC, same 3/74 codes |
| **width-sharded RMSNorm** (8x1 grid, fp32 acc intact) | Block 2, 49 calls/frame | 1.46x on the norm+linear pair | velocity PCC 0.9999845 → 0.9999852 |

The row fold is the CFG lesson the module already preached, never applied to GQA: query head *j*
reads kv head *j//4*, so those 4 heads' 3 tokens stack as 12 ROWS against one kv head. The two
attention matmuls go batch-32 → batch-8, and rows inside a 32-row tile are free. Proven equivalent
on host in fp32 (agrees to 6e-07, pure reduction order) — not an approximation.

**Reverted — width-sharded RMSNorm in BLOCK 1.** Same 1.46x, ~5 ms/frame, and every cheap signal
said yes: 0.9999973 per op, decode PCC unmoved at 0.99991. But over 24 REAL teacher-forced frames
the **worst sample went 1.06% → 1.95%** while PCC stayed flat. Teacher-forced matters: both builds
see identical inputs at every step, so that comparison is deterministic. This is the second
instance of the trap `_norm` documents — a ~1e-6 per-op difference amplified through 26 layers —
and per-op PCC hid it both times. Block 2 keeps the sharded norm because 3 layers give nothing to
amplify. (This entry originally also cited WER 0.88% → 2.06%; see below for why that half of the
evidence was worthless.)

**This list has since been worked through — see §6.8 for the results.** Kept only as the record of
what was outstanding at the end of the first sweep:
1. ~~wqkv + wo → BFP8~~ **DONE**, 3.32 ms, no hang (§6.8).
2. ~~sdpa for Block 2's attention~~ **REJECTED** on a deterministic gate, 7/288 → 21/288 (§6.8).
3. ~~semantic argmax on device~~ **DONE**, 1.49 ms, fp32 (§6.8).
4. ~~Re-measure the device trace~~ **DONE**, still no (below).
5. **Fewer Euler steps** (7→5 removes 28% of the solve) — STILL OPEN. A MODEL change, needs a
   listening pass, not a metric.
6. **Concurrent requests** — STILL OPEN. Throughput, not latency.

**w2 in BFP8: RETRIED AND IT STILL HANGS.** Worth doing once wqkv and wo turned out fine, because
that made "all-BFP8 hangs" look over-attributed in general. It is not: w2 is genuinely the trigger.
It is worth **2.5 ms/step** (31.4 → 28.9) at PCC 0.99977–0.99985, so the hang is the only obstacle.

Two things the retry changed. First, it now hangs **earlier and harder**: the documented repro died
on the third utterance inside Block 3, whereas this died during the FIRST case, right after the
first compute op, with no pipeline output at all — today's op mix reaches the trigger sooner, so
the five-condition sequence is no longer minimal. Second, recovery: `tt-smi` is **not on PATH** on
this machine. The Wormhole build is at `/home/software/syseng/wh/tt-smi` and this vintage takes
`-wr 0`, not `-r`. `open_device` hangs indefinitely until you run it.

**Device tracing, re-measured after the op-count wins — still not worth it, but the reason
changed.** The old verdict was that tracing COST time (Block 1 -0.7 ms, Block 2 -6 ms/frame). That
penalty is gone; what is left is a gain too small to bank:

| | untraced | traced | delta |
|---|---|---|---|
| Block 1, 26-layer decode step | 34.75 ms | 34.55 ms | +0.20 ms (1.006x) |
| Block 2, whole 7-step solve | 26.07 ms | 25.91 ms | +0.16 ms (1.006x) |

Both captures are CORRECT: Block 1's PCC and worst-sample are unchanged (0.999893 / 1.06%) and
Block 2's integer codes are bit-identical to the untraced path. Both were timed with a readback
every frame, because without one the host runs ahead and pipelines the next frame's dispatch under
the current frame's device work — which hides exactly the cost a trace removes.

**What this settles: host dispatch is not a bottleneck for either block.** Tracing removes host
command submission almost entirely, and removing it changes nothing. So Block 2's remaining ~2x gap
over its 13.4 ms weight-read floor is DEVICE-side per-kernel cost — kernel launch on the cores,
circular-buffer setup, intermediates round-tripping DRAM — not commands queuing up on the host.
That is worth knowing before optimizing here: work that only reduces command COUNT will not help;
work that makes the kernels themselves bigger and fewer will.

0.5% does not justify what tracing costs to keep: `trace_region_size` on every caller, buffers
addressed by pointer so a fresh upload is silently ignored, a capture ORDER constraint between the
two blocks (all warm-ups before either capture, or the second hangs), and a reserved scratch cache
row so the warm-up and capture writes do not corrupt the prompt. Not re-added. The probes are in
the session scratchpad if this needs re-running after a big change.

**Matmul fusion: one win, and it CLOSES the matmul side of Block 2.** Following the trace result
above (device-side per-kernel cost, so make kernels bigger rather than merely fewer), the two pairs
of Block 2 matmuls that read the same input were merged: `wq`+`wkv`, and `w1`+`w3`. Only the first
paid. Per-matmul efficiency against the 194 GB/s ceiling explains exactly why:

| matmul | MB/call | us/call | GB/s | % of ceiling |
|---|---|---|---|---|
| B1 wqkv (bf16, already fused) | 37.7 | 185 | 204 | 105% |
| B1 wo (bf16) | 25.2 | 129 | 195 | 100% |
| B2 wq | 13.4 | 74 | 182 | 94% |
| **B2 wkv** | **6.7** | **74** | **91** | **47%** |
| B2 wqkv, fused | 20.1 | 101 | 198 | 102% |
| B2 w1 | 30.1 | 157 | 192 | 99% |

`wkv` was the ONLY launch-bound weight matmul anywhere in the model — at 2048 wide it cost the same
74 us as the 4096-wide `wq`, so there is a fixed ~40-50 us per launch and width past ~4096 is
nearly free. Folding q in with it took the pair from an effective 137 GB/s to 198, worth **0.96
ms/frame**. It is mathematically the same arithmetic — a linear computes each output column
independently, and 4096 is tile-aligned so the BFP8 blocks are unchanged — but NOT bit-identical:
a `[3072,6144]` matmul picks a different internal block decomposition than a 4096 and a 2048
separately, which reorders the sum over the 3072 inner dim. Same class of difference as the row
fold. 11 of 15 fixture cases reproduce their frame count exactly and 4 shift, which is the usual
chaotic-trajectory response to a last-bit change, so it is gated on WER like everything else.
`w1`+`w3` are 9216 wide each and already at 99% — merging them measured **0.998x**, and
0.951x once the output split is charged, so they stay separate. Cleared by the gates that resolve
anything: velocity PCC 0.99998522 (identical to pre-fusion), same 3/74 codes, long-form WER 0.00%,
15/15 termination. Its single-seed headline WER read 1.76%, which means nothing — see §6.7.

**Every weight matmul in both blocks is now at the DRAM ceiling.** There is no fusion left to do,
in either block: Block 1's are all 25-38 MB per call and were never launch-bound. Block 2's
remaining ~13 ms above its weight-read floor is entirely in the NON-matmul ops, and the next real
attack on it is sdpa (item 2 above), which replaces the inherently batch-8 attention interior — the
one part the row fold could not flatten, because each batch element needs a different k.

### 6.7 — the WER gate is noisier than the changes it was gating

**Same code. Only the generation seed changed.**

| seed | natural-text WER | long-form (298 w) | short (42 w) |
|---|---|---|---|
| 0 | 1.76% | **0.00%** | 14.29% |
| 1 | 2.06% | **0.00%** | 16.67% |
| 2 | 0.88% | **0.00%** | 7.14% |

0.88–2.06% is the ENTIRE range seen across every implementation variant in §6.6, endpoints
included. So **every single-seed WER comparison made during that sweep was uninformative**, and two
conclusions drawn from it have been corrected above: the row fold "held WER at 0.88%" (luck), and
the Block 1 sharded norm "cost 0.88% → 2.06%" (also luck — that revert stands on its worst-sample
measurement, which is deterministic, not on this).

The cause is word counts, not audio. Per case, same code, seeds 0/1/2:

| case | lang | words | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| 7 | spanish | 6 | 50.0% | 0.0% | 0.0% |
| 9 | portuguese | 6 | 0.0% | 83.3% | 0.0% |
| 13 | arabic | 3 | 33.3% | 0.0% | 33.3% |
| 6 | german | 7 | 28.6% | 28.6% | 28.6% |
| 0,1,2,3 | english | 298 total | 0.0% | 0.0% | 0.0% |

On a 6-word clip one Whisper disagreement is 17% and three is 50%, so 42 words of short prompts
swing the 340-word aggregate by more than a point while 298 words of English sit at exactly zero.
Case 6's 28.6% is the one short-prompt error that is REAL — it reproduces on every seed.

**The gate is now the long-form number**, which `score_quality_set.py` prints separately: 0.00%
over 298 words, in every run ever measured. Plus 15/15 natural termination and the voice-identity
check, both of which have also held everywhere.

**For judging a numerical change, prefer the deterministic gates entirely** — teacher-forced PCC
and worst-sample against the fp32 reference in `tests/tt_gates.py`. They feed both builds identical
inputs, so no chaotic trajectory is involved. That is what actually caught the Block 1 norm, and
what correctly cleared the qkv fusion. Use end-to-end WER to catch gross breakage, not to resolve
a last-bit difference — and if you must, run at least three seeds and compare ranges.

### 6.8 — second sweep, gated deterministically

Run after §6.7 established that end-to-end WER cannot resolve changes this small. Every accuracy
number below is teacher-forced against the fp32 reference — identical inputs to both builds, no
generation loop, no chaotic trajectory.

**Rejected — sdpa for Block 2's attention interior.** 1.147x (28.73 → 25.05 ms), and the accuracy
cost is real: integer codes differing from the fp32 reference go **7/288 → 21/288** over 8
independent (h, x_0) draws. That is the same ~3x §7 measured before BFP8, so the old rejection
holds and is now confirmed on a gate that can actually see it. Note an earlier single-draw check
in this session read 2/74 against the row fold's 3/74 and looked *better* — one draw is worthless.

**Rejected — two tiny-op reductions in Block 2, both worth nothing.**
- CFG combine as multiply-by-constant + sum instead of 2 slices + 2 multiplies + add: **1.001x**.
- `inplace=True` on the sharded norm program config: **0.997x**.
Removing small ops does not help. That is consistent with the trace result (§6.6): the cost is
device-side per KERNEL, and these ops are already at the per-kernel floor, so deleting a few of
them changes nothing. Only work that makes kernels BIGGER pays — which is why the qkv fusion did
and these did not. The same reasoning predicts the `concat([x, x])` idea is also worthless; it was
not isolated separately.

**Shipped — semantic head on device, 1.49 ms/frame.** Was a host CPU matmul at 2.74 ms; on device
in fp32 it is 1.25 ms. **fp32 is mandatory**: the output is an INDEX, and over 64 hidden states
bf16 weights pick a different index on 4 of them while fp32 matches the host answer 64/64. bf16
would save a further 0.2 ms in exchange for a wrong primary code on ~6% of frames.

**Shipped — wqkv and wo in BFP8 (Block 1), 3.32 ms/frame.** The untested precision point from
§6.6. It does NOT trigger the hang: the documented minimal repro (short gen + 128-bucket decode,
long gen + 512-bucket, long gen + 512-bucket as a pure cache hit) runs clean, as does the full
15-case set. w2 stays bf16 — that one is still the pinned trigger and was not retried.

**A statistics correction that matters for every future decision here.** Worst-sample was being
read as a MAX over frames, and a max is an unstable order statistic. Over 44 teacher-forced frames
across two real prompts:

| Block 1 config | ms/step | min PCC | mean ws | p90 ws | max ws |
|---|---|---|---|---|---|
| shipped (bf16 + FF BFP8) | 34.67 | 0.999884 | 0.86% | 1.11% | 1.34% |
| sharded norm only | 29.78 | 0.999867 | 0.92% | 1.16% | **4.28%** |
| + wqkv BFP8 | 32.51 | 0.999872 | 0.86% | 1.14% | 2.10% |
| + wqkv and wo BFP8 | 31.35 | 0.999852 | 0.86% | 1.10% | **1.28%** |
| + sharded norm too | 26.51 | 0.999851 | 0.84% | 1.06% | 1.40% |

The max column is not monotone in how much error is introduced — adding wo on top of wqkv took it
from 2.10% down to 1.28%. **Use mean and p90; treat max as an anecdote.** On mean/p90 the BFP8
rows are indistinguishable from shipped, which is why they ship.

This also weakens, without overturning, the §6.6 sharded-norm revert: that call cited max going
1.06% → 1.95%, and max is the unreliable column. On the stable statistics the sharded norm is
still worse alone (mean 0.86% → 0.92%), just by less than was claimed. It stays out, but the
honest reason is "slightly worse on a stable metric for 5 ms", not the dramatic doubling reported.

**Checked and ruled out** (so nobody re-runs them):
- `rms_norm(residual_input_tensor=)` returns only the normed value and discards the sum; a pre-norm
  block needs that sum for the next residual. Post-norm models (BERT) can use it, we cannot.
- **Feeding sharded activations straight to a matmul is SLOWER** — 8.94 vs 5.32 ms per 26
  norm+linear pairs, because a DRAM-interleaved weight makes the matmul gather the shards itself.
  This is why `ign`'s keep-everything-in-L1 approach does not transfer to us as-is.
- Fused SiLU in Block 2's multiply (`input_tensor_a_activations`): no measurable gain. Block 1
  already fuses it via `activation="silu"`.
- `ttnn.deallocate` on every intermediate (ign's style): no gain here.
- ~~Norm grids 8x2 / 8x4 / 8x8: all slower than 8x1.~~ **WRONG, see §6.18** -- measured on the
  isolated norm, which ranks grids backwards. 8x4 (32 cores) is the fastest end to end and now
  ships. 8x8 genuinely cannot build: 3072/32 = 96 tiles and 96/64 is not an integer.
- Row fold in Block 1 prefill (needs a 4x-tall mask, and prefill is ~1.3% of wall) and in the codec
  (it is MHA 8/8, no grouping to fold).


### 6.9 — the sharded norm, shipped on the second look

**Block 1's two RMSNorms are now width-sharded over 8 cores: 6.1 ms → 1.1 ms, worth 4.4 ms/frame
(31.4 → 26.6 ms/step, mean frame 59.9 → 55.5, RTF 0.74-0.82 → 0.68-0.78).**

THIS REVERSES §6.6, and the reason it was wrong is the useful part. That revert measured the
sharded norm while `wqkv` and `wo` were still bf16, where it read mean worst-sample 0.86% → 0.92%.
On the CURRENT weights it reads 0.86% → 0.84% mean and 1.10% → 1.06% p90 — no cost, reproduced in
two independent runs. **A precision change here is not separable from the others; re-measure against
the config you actually ship, never against a recorded number.**

Core count barely matters, and it is worth knowing why:

| norm | us/call | ms/step | mean ws | p90 ws |
|---|---|---|---|---|
| interleaved | 115.5 | 31.42 | 0.86% | 1.10% |
| 2-core | 42.4 | 27.50 | 0.83% | 1.08% |
| 4-core | 40.5 | 26.81 | 0.81% | 1.10% |
| 8-core ← shipped | 44.1 | 26.54 | 0.84% | 1.06% |

Flat, because the norm COMPUTE is ~16 us at any count and the two `to_memory_config` calls are the
other ~28. The reshard is the tax, not the reduction. (Feeding the sharded result straight to the
next matmul to dodge the second reshard is SLOWER — 8.94 vs 5.32 ms per 26 norm+linear pairs.)

**THE COST, stated exactly, because it is not zero.** Long-form errors over three seeds:

| config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| interleaved | 0 wrong | 0 wrong | 0 wrong |
| 8-core sharded | 1 wrong | 1 wrong | 0 wrong |

Both errors are the SAME word in the SAME sentence: the model contracts "I am" to "I'm" in "…now
that I have it, I am not going to be silent." That is the only difference across 894 long-form words
(298 × 3 seeds); termination is 15/15 and voice identity passes everywhere. Shipped on the judgement
that a contraction is what a human reader does, not a defect — an explicit call, so if a future
regression traces here, this is the trade to question first.


### 6.10 — nlp_create_qkv_heads: the op has a floor, but its OUTPUT CONFIG was free money

Chased the biggest non-matmul line in Block 2 (2.7 ms/frame, 129 us x21 — more than the wqkv matmul
that feeds it, and 18x Block 1's decode variant at 7.0 us).

**The op's cost is a FIXED ~97 us, not data movement.** The same call on S=32 — 10.7x the real data
— also measures 97 us. So there is nothing to win by feeding it less or laying the input out
differently, and indeed every restructuring is worse: hand-rolled slice+reshape+permute is 158 us
against the shipped chain's 140, and riding the CFG pair on the sequence dim is 259 us. Both were
verified bit-identical first, so those are speed results, not broken reimplementations. The sibling
ttnn ops (`create_qkv_heads`, `transformer.split_query_key_value_and_split_heads`) reject GQA shapes.

**What paid was the output memory config, and the op-level number hides why.** Isolated, an L1
output saves ~7 us on the op. In the real block it is worth **2.5 ms/frame**, because q/k/v then
stay in L1 for the four ops that consume them:

| output | transpose_k | ms/frame | codes ≠ fp32 ref (8 draws) |
|---|---|---|---|
| DRAM | False ← was | 26.75 | 7/288 |
| DRAM | True | 26.72 | 7/288 |
| L1 | False | 24.28 | 9/288 |
| L1 | True ← shipped | **24.17** (1.106x) | 9/288 |

L1 carries all the speed AND all the cost — 2 extra differing codes in 288. `transpose_k_heads=True`
is free (it emits k already transposed, deleting our transpose op) but worth 1.001x alone; it is on
for tidiness. NOT bit-exact end to end: the three tensors are identical either way, but an
L1-resident operand makes the downstream matmul choose a different program config. Velocity PCC
0.99998522 → 0.99998164.

Gated on the full run: **long-form 0 wrong words in 298**, 15/15 natural termination, voice PASS.
For scale, the accepted BFP8-weights trade was 1.23x for one extra code in 222; the rejected sdpa
trade was 1.147x for fourteen (7→21). This is 1.106x for two.

**Generalisable lesson: for the small tensors in this block, WHERE a tensor lives matters as much as
how big the kernel is.** Every previous win here came from making kernels bigger; this one came from
keeping an operand in L1 across its consumers.

**That generalised, and was worth another 1.15 ms.** Applying the same idea to the rest of
`_block`'s intermediates, cumulatively, at IDENTICAL accuracy throughout (9/288 differing codes in
every row):

| | ms/frame | vs q/k/v-only |
|---|---|---|
| q/k/v L1 only | 24.18 | — |
| + attention interior (scores, scaled, av) | 23.85 | 1.014x |
| + MLP intermediates (g, w3_out, u) | 23.22 | 1.041x |
| + residual stream | **23.04** | **1.049x** ← shipped |

The one candidate that does NOT pay is the `_norm` output: **0.999x alone**, so it stays DRAM. So
this is not "L1 everywhere is better" — it is specifically values with a consumer close behind, and
the norm's output is immediately eaten by a big DRAM-weight matmul that dominates it. The LIMIT is
still §6.6: a width-SHARDED activation into a DRAM-weight matmul is slower. Interleaved-L1 is the
useful middle.

**IT TRANSFERS TO BLOCK 1 TOO, for another 0.9 ms at zero accuracy cost** — min PCC 0.999850, mean
worst-sample 0.85%, p90 1.09% and even max 1.40% all byte-identical before and after, over 44
teacher-forced frames:

| Block 1 decode | ms/step | vs shipped |
|---|---|---|
| shipped | 26.43 | — |
| + wo output and residual L1 | 26.19 | 1.009x |
| + MLP intermediates (g, u) L1 | **25.53** | **1.035x** ← shipped |

And a THIRD negative that sharpens the rule: **sdpa_decode's output stays forced to DRAM.** That
`to_memory_config(o, DRAM)` looks like exactly the round trip this whole finding is about, and
routing it to L1 measures 0.999x — nothing. So the pattern is not "L1 is faster", nor even "avoid
DRAM round trips". It is specifically **values with a consumer close behind**; the norm output and
sdpa output both feed a single large op that dominates whatever the transfer cost was.

`_mlp` is shared with prefill, so the memory config is passed in at the call site rather than baked
in — prefill's `g` is [1,S,9216] with S up to 384, i.e. 6.8 MB, so it passes DRAM. That mirrors how
`h` is already handled, and for the same reason: it is what the two paths do differently.

**BLOCK 3 IS NOT THE NEXT TARGET, and the "~9% of wall" figure quoted in §6.6 and §6.8 was wrong.**
That number came from `wall − prefill − generate` on the 15-case run, which lumps FIRST-CALL KERNEL
COMPILATION for each new codec bucket in with the codec's actual compute. Measured directly:

| | T=461 | T=512 |
|---|---|---|
| cold (a length not compiled yet) | **3399 ms** | 97 ms |
| warm | 97 ms | 97 ms |
| `_graph` at an UNBUCKETED length | **13245 ms** | 95 ms |

So the decoder's steady-state compute is **97 ms against a 25.8 s generation — 0.4%**, and the
seconds-scale figure is compile cost, paid once per distinct bucket (12 of them for the ~1500-frame
ceiling, so a long-lived server pays it 12 times ever). Same class of mistake as reading case 0's
RTF 1.89 as a slow case. **Optimizing codec compute is worth ~0.4% of wall; there is nothing there.**

Two ideas were measured against it before that was understood, and both LOSE, so they are closed:
- **Per-stage SLAB sized to the window.** The single global 512 computes a 512x512 score matrix where
  only a (window+1)-wide band survives the mask — 97% waste at the last stage, 99.4% at the first.
  Shrinking it is much worse, because the per-chunk machinery dominates the wasted arithmetic: at
  L=4096, slab 64/128/256/512/1024 measures 40.4/19.5/12.1/**9.7**/11.5 ms. 512 is genuinely optimal
  for three of four stages, as its comment claimed. The one win is tf1 (L=1024) at slab 1024, where
  it becomes a single chunk — 2.7 ms of 94, and not bit-exact (2.4e-04).
- **The score chain in L1**, the trick that paid in §6.10. It LOSES here, 0.82–0.94x. The score
  tensor is 4.2 MB at slab 512, far past the ~100 KB scale where L1 helped in the other two blocks.
  **That is the size boundary on the §6.10 finding**: L1 for intermediates works at
  tens-to-hundreds of KB, not megabytes.

Also a useful negative on `ign/voxtral_opt`'s lead: the transformer layers are **92%** of the codec
(86.4 of 94.35 ms at T=512) and all the convs together are 8.5%, so their conv-split optimization
would not have transferred either.

If anything in Block 3 ever deserves attention it is the COMPILE cost, not the compute — and
`BUCKET = 128` already exists to cap it.


### 6.11 — the norm's second reshard cannot be dodged, both ways measured

`_norm_dec` width-shards, norms, then converts back to DRAM. That second `to_memory_config` is 0.33
ms/frame and looks removable. It is not, and the two candidates fail for different reasons.

**Default matmul, sharded activation: 8.94 vs 5.32 ms per 26 norm+linear pairs.** The cause is the
AXIS, not L1 vs DRAM. Width-sharding splits the matmul's CONTRACTION dimension, so each core forms
only a partial sum and the cross-core reduce is full-output-sized ([32,6144] x 8 cores). Interleaved,
ttnn splits by OUTPUT COLUMNS: each core owns its columns, reads the whole 6 KB activation, nothing to
reduce. **The same axis that makes the norm fast makes the matmul slow** — the norm REDUCES over
width (cross-core step is 8 scalars), the matmul CONTRACTS over it.

**The matmul that wants a sharded activation: builds, still loses.**
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` requires exactly this layout. The DRAM-sharded-weights note above recorded
it overflowing L1 at Block 2's shape (N=9216, per_core_N=36 tiles, CBs 1,678,528 B against
1,499,136). Block 1's wqkv is N=6144, per_core_N=**24** tiles, and it does build — the capacity
projection was right. It is still slower:

| | us | vs shipped |
|---|---|---|
| shipped: unshard + default matmul | **100.9** | 1.000x |
| DRAM-sharded, sharded activation in | 125.4 | 0.805x |
| + unshard the output | 128.5 | 0.785x |

**0.72 ms/frame worse**, and not bit-exact (4.9e-04). The reason is simply that the default path
already runs this matmul at ~198 GB/s, i.e. at the DRAM ceiling, so there is no bandwidth for the
DRAM-sharded machinery to win back — it only adds its own cost. That config earns its keep when the
weight read is contended or spread across devices, not when a plain 1D config already saturates.

**And sharding only the WEIGHT is not available.** Width-sharding the weight splits the OUTPUT
columns — the axis needing no reduction — so it looks like the free version of this idea. ttnn
couples them: a width-sharded `in1` is accepted by ONLY the DRAM-sharded config, which also demands
a width-sharded `in0`; every other config asserts `in1` is INTERLEAVED
(`matmul_device_operation.cpp:1188ff`, with the pairing spelled out in the comment at :1199). So
"sharded weight, interleaved activation" cannot be expressed, and the paired form is the 0.805x above.

An **L1-resident weight** fails the same assertion. It was the interesting one on paper — a resident
weight is never re-read from DRAM — but the ceiling kills it regardless: Block 1 streams ~3.9 GB per
frame against **96 MB of total L1 (64 x 1.5 MB), 2.4% of the model**, and that L1 is also wanted by
the activations and circular buffers. **The model not fitting in L1 is exactly why 194 GB/s is the
wall**, and no sharding of a 3.9 GB working set changes that.

**Generalisable:** before reaching for a fancier matmul config, check what the current one achieves.
Every hand-tuned config tried in this port has lost, and in each case the plain one was already at
the ceiling (see also the 169-vs-193 GB/s program-config result in ttnn_voxtral_gpt).


### 6.12 — the w2 hang, root-caused

Found with `TT_METAL_WATCHER=10`, which converts the hang into a clean abort plus a device-side dump.
**Investigating this no longer costs a board reset**, which is what made it tractable at all.

| | |
|---|---|
| **exact op** | `ttnn_voxtral_codec.py:589` — the codec's OUTPUT projection, `_conv1d(x, "out", 1024, 240, kernel=7, stride=1, "reflect")` at L=4102 |
| **exact kernel** | `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp`, stuck on both BRISC and NCRISC |
| **exact fault** | NCRISC on noc1 attempts a **unicast write of 13,897,728 bytes** from local `L1[0x15f0000]` to virtual core **18-52** `[addr=0x008ae800]` — a core that does not exist |
| **trigger** | the SECOND execution of that exact shape, i.e. a pure program-cache hit |

`13,897,728 = 3393 × 4096`, and 4096 B is exactly one input row (1024 fp32 channels). So it is trying
to push **3393 rows in a single NOC transaction, to nowhere.**

**It is not a deadlock** — it is a corrupted descriptor producing an out-of-range write. That is why
memory looked flat with 8 GB free, and why program-cache counting and leak hunting never found it.

**The trigger is the cache hit, confirmed by the conv trace.** Case 2 runs `out` at L=4102 and
completes; case 3's byte-identical call faults. That is exactly the "pure cache HIT" of the old
five-condition repro, now with a kernel name attached.

**And it is not a Block 1 bug.** w2's dtype matters only because it is the biggest lever we have on
DRAM allocation ADDRESSES — BFP8 frees ~690 MB across 26 layers, shifting every later allocation
including the codec's conv buffers. This is a latent address-dependent bug in ttnn's conv halo path
that w2 happens to expose; anything moving allocations by enough would do the same. That explains
why the five conditions looked so arbitrary: they are just a recipe for reaching a bad address
twice.

**Two of my own earlier claims were wrong and are corrected here.** §6.x said the hang "now fires
earlier and harder — during the FIRST case, with no pipeline output at all". It does not: Python's
stdout is block-buffered and the abort discards it, so the absence of output meant nothing. It hangs
in the THIRD utterance's codec decode, exactly where the original diagnosis put it. A single-case
run completes cleanly, consistent with needing ≥2 buckets.

**DODGED — see §6.13.** The second idea works: the output projection skips `conv1d` entirely.


### 6.13 — the hang, FIXED, and w2 shipped in BFP8

§6.12 named the faulting op. It is **our own call**, so it was ours to remove: the codec's output
projection no longer uses `ttnn.conv1d`. A k=7 stride-1 conv over a pre-padded tensor is a
sliding-window matmul —

    out[t] = sum_j  xpad[t+j] @ W[j]

— so 7 slices + 7 matmuls + 6 adds compute it exactly and touch no `halo_gather` kernel.

**It is slower here and overwhelmingly worth it**, because this runs once per utterance while w2
saves 2.5 ms per frame:

| output projection | ms (L=4096) | vs conv |
|---|---|---|
| `ttnn.conv1d` (was, and broken) | 4.29 | — |
| 7 matmuls, shifting the INPUT | 9.16 | +4.87 |
| 7 matmuls, shifting the OUTPUT ← shipped | **6.27** | **+1.98** |

**Shift the OUTPUT, not the input.** Both orders compute the same sum, but the shift must be a
slice, and slicing the 1024-wide input costs 0.624 ms a time against 0.145 for the 240-wide output.
Breaking down the 9.16 ms version: input slices 4.37 ms, the seven matmuls only 1.93, adds 1.16 —
**the slices cost more than the matmuls**. Multiplying the full padded input first and slicing the
narrow result instead saves 2.89 ms.

`+2.0 ms once` against `−2.5 ms × 460 frames = −1150 ms`.

**Not parallelisable**, before anyone asks: the seven passes are independent, but every ttnn op
already uses the whole 64-core grid, so running them concurrently would just give each pass 9 cores.
Nothing is idle. Holding `xp` in L1 to avoid re-reading it does not fit either — 16.8 MB overflows
the allocator. At 42 GB/s (22% of ceiling) there is still headroom here, but not via either route.

**Result, 3 seeds × 15 cases = 45 utterances:**

| | before | after |
|---|---|---|
| ms/frame | mean 50.4 | **mean 47.5** (45.9–52.1) |
| RTF | 0.62–0.71 | **0.58–0.81** |
| hang | w2 BFP8 impossible | **45/45 utterances clean** |
| termination | 15/15 | 15/15 in all three seeds |
| voice identity | PASS | PASS |
| codec waveform PCC | 0.999915 | **0.999915**, unchanged |

**THE COST, stated plainly.** Long-form errors across seeds 0/1/2 are **1 / 1 / 0 words of 298**,
against a shipped baseline that was 0 at seed 0. The errors land in DIFFERENT cases each seed (case 3
`"listened"→"listen"`, then case 2), which is the signature of noise rather than a systematic defect —
and 1/1/0 is the identical spread to the sharded-norm change accepted in §6.9. Shipped on that basis.
If a regression is ever traced to codec output quality, this and §6.9 are the two trades to question.

**The generalisable lesson:** the hang was called "unexplained" for months and treated as a hardware
mystery. It took one environment variable (`TT_METAL_WATCHER=10`, which turns the hang into a clean
abort with the kernel named) and then one observation — that the faulting op was ours, not ttnn's, so
we could simply stop calling it. Before writing something off as someone else's bug, check whether
you are the one invoking it.

### 6.14 — the codec projection: the question was "fuse the matmuls", the answer was the pad

The replacement projection from §6.13 cost 6.26 ms against `ttnn.conv1d`'s 4.29 — a regression we
accepted because it runs once per utterance and unlocks 2.5 ms *per frame*. The obvious next move is
to fuse its seven tap matmuls, since each one re-reads the same 16 MiB `xp` from DRAM: 118 MiB of
traffic to produce 3.9 MiB of output. **Fusing works and is nearly worthless. The padding was the
expense.** Six rounds of probes (`probe_fuse_taps*.py`), and the first three rounds each refuted the
previous round's premise.

**Round 1, the fusion itself.** Concat g taps side by side into one `[1024, g*pitch]` weight, one
matmul, then g column-block slices. DRAM traffic is *not* monotone in g, because the wider output
stops fitting L1 and starts paying a write plus a read-back:

| g | out | xp reads | best |
|---|---|---|---|
| 1 | 3.9 MiB → L1 | 7 | 6.25 ms |
| 2 | 8.0 MiB | 4 | 6.04 ms |
| 3 | 12.0 MiB → DRAM | 3 | 5.89 ms |
| 7 | 29.4 MiB → DRAM | 1 | worse |

1.06×. But the same run reported `_pad_causal` alone at **3.06 of the 6.26 ms**, which reframed
everything.

**Round 2 killed its own hypothesis.** The pad builds a 16 MiB tensor with a 7-input concat, so the
concat looked like the cost. Measuring four padding strategies showed otherwise — the variant that
builds *no* 16 MiB tensor at all, only a 12-row one, still cost 2.554 ms against the current 3.062.
A 48 KiB tensor cannot cost 2.5 ms. The concat was worth ~0.4 ms. What all four shared was **six
single-row `ttnn.slice` calls**.

**Round 3 found the actual currency: ops, not bytes.**

```
one single-row slice of the 16 MiB x     0.381 ms      six of them            2.282 ms
one SIX-row slice of x, rows 1:7         0.358 ms      ragged 7-way concat    1.815 ms
six single-row slices of a 24 KiB tensor 1.022 ms      aligned 2-way concat   0.281 ms
```

Six rows cost the same as one. A 4 KiB slice costs 170 µs. Cost is per *op*, and against a
TILE_LAYOUT tensor an unaligned single-row slice appears to pay something proportional to the whole
tensor.

**Rounds 4–6, the fix.** The prefix is rows x6,x5,x4,x3,x2,x1 — contiguous but **reversed**, and
there is no `ttnn.flip`. Five ways to reverse six rows:

| method | ops | ms | exact? |
|---|---|---|---|
| six 1-row slices + concat (what shipped) | 8 | 1.665 | yes |
| permutation matmul, HiFi3 | 3 | 0.195 | **no**, 2.4e-04 |
| permutation matmul, HiFi4 + fp32_dest_acc | 3 | 0.194 | **no**, 2.4e-04 |
| `ttnn.slice` with step −1 | 1 | 0.202 | **returned 0 rows, no error** |
| `ttnn.gather` into a 32-row prefix | 2 | **0.071** | **yes** |

`ttnn.gather` wins outright, and folding it straight into a tile-aligned 32-row prefix makes the
reversal, the alignment and the prefix cost 3 ops together. Output slices then start at 26 instead
of 0; prefix rows 0..25 are deliberate copies of `x[0]`, finite, and sliced away.

**Shipped: 6.26 → 3.45 ms, bit-identical, and finally below `ttnn.conv1d`'s 4.29.** Verified three
ways: probe max-abs-diff exactly 0.0 at L=4096; the codec gate's PCC lines diff *identically* against
the pre-change build; 26/26 tests, full real-prompt set clean under `TT_METAL_WATCHER=10`, RTF
0.60–0.65.

**Fusion was measured and NOT taken.** On top of the gather pad, g=3 gives 2.94 ms — 0.51 ms more —
but costs bit-exactness (1.1e-04: a `[1024,768]` matmul decomposes differently from three
`[1024,240]` ones, same phenomenon as Block 2's qkv fusion). Not worth trading exactness for 0.51 ms
on an op that is 0.015% of wall.

**Three ttnn behaviours worth knowing, all found by checking numerics that "obviously" could not be
wrong:**
- A 240-wide (half-tile) column block sliced out of an **L1** tensor comes back **silently wrong** —
  rel err 5e-01, no exception. 256-aligned blocks are fine.
- `ttnn.slice` with a negative step returns an **empty tensor**, no error.
- An **fp32 matmul multiplies at bf16 precision** here: a 0/1 permutation matrix loses 2.4e-04, and
  HiFi4 with `fp32_dest_acc_en` changes nothing. Worth remembering before trusting fp32 anywhere in
  this port for precision rather than range.

**Also fixed:** `test_prepared_weights_are_deduplicated` asserted 20 prepared `(conv, length)` pairs
and had been failing at 16 since §6.13 — `out` left `ttnn.conv1d`, so 5 convs × 4 buckets became 4 × 4.
Verified stale *before* this change, not caused by it.

**The generalisable lesson, and it is the same shape as §6.13's:** I set out to optimise the thing
the arithmetic said was expensive (118 MiB of redundant reads) and it was worth 6%. The thing
actually costing 49% was six lines of index bookkeeping nobody had timed. Decompose before
optimising — and when a decomposition contains an item that *cannot* cost what it measures, that
item is the finding.

### 6.15 — w2 in BFP8, finally gated deterministically (and §6.8's levels do not reproduce)

§6.13 shipped w2 in BFP8 on end-to-end WER alone — 1/1/0 wrong words of 298 across seeds 0/1/2 —
which §6.7 had already shown is **below the resolution of that gate**. §6.8's deterministic table
stops at `wqkv + wo BFP8 + sharded norm`, before w2. So the largest single precision drop in Block 1
shipped without the measurement every other precision change in this port was held to. Closed here.

Same session, same fixture, 44 teacher-forced real frames on 2 real prompts, only `WEIGHT_DTYPE`
differing:

| Block 1 decode | min PCC | mean WS | p90 WS | max WS | prefill WS |
|---|---|---|---|---|---|
| w2 **bf16** | 0.999756 | **0.94%** | 1.56% | 2.36% | 0.51 / 0.52 |
| w2 **BFP8** ← shipped | 0.999737 | **1.04%** | 1.55% | 1.97% | 0.46 / 0.50 |

**The cost is real and it is +0.10 pp on mean worst-sample.** Not scatter: the gate was re-run and
reproduced **bit-identically**, so it is deterministic and 0.10 is signal. But it appears on the mean
and nowhere else — p90 is flat, max improves, prefill improves. **Kept**, for 2.5 ms/frame (~5% of
frame time). Decision recorded so it can be revisited with a number rather than a WER reading that
could not see it.

**THE TWO CHANGES ARE NOW DECOUPLED, which they were not before.** w2 in BFP8 was only possible
because the codec stopped calling `ttnn.conv1d`; the codec fix now stands on its own. So w2 can be
reverted **alone** — one line, `ttnn_voxtral_gpt.py` `WEIGHT_DTYPE` — for ~2.5 ms/frame (RTF
0.60–0.65 → ~0.63–0.68), and it does **not** reintroduce the hang, which needed `conv1d`.

**⚠ THE BIGGER FINDING: THIS GATE WAS BEING READ TO 0.01 pp AND ITS PROMPT SPREAD IS 0.45 pp.**

Chasing why §6.8's levels would not reproduce turned up something that matters more than the w2
result above. Per prompt, 22 teacher-forced frames each, shipped build:

| case | voice | P | mean WS | p90 WS |
|---|---|---|---|---|
| 5 | fr_female | 118 | 1.20% | 1.65% |
| 7 | es_female | 158 | 1.34% | 2.34% |
| 1 | cheerful_female | 163 | **0.97%** | **1.38%** |
| 6 | de_male | 179 | **1.42%** | 2.28% |
| 8 | it_male | 184 | 1.35% | 2.16% |
| 0 | neutral_male | 200 | 1.02% | 1.55% |
| 2 | neutral_male | 312 | 1.07% | 1.38% |
| 3 | casual_female | 357 | 1.10% | 1.50% |

**Prompt choice moves mean worst-sample by 0.45 pp and p90 by 0.96 pp** — 4× and ~10× the +0.10 pp
that w2's precision drop costs, and larger than every §6.8 candidate that was accepted or rejected on
this metric. It is **not** a prompt-length effect: the shortest prompt (case 5, P=118) reads worse
than case 1 at P=163. §6.8 reported a 2-prompt pair to 0.01 pp and **did not record which two**.

This is §6.7's error repeated on the gate that was built to *replace* §6.7's gate: reading a number
to a precision far finer than its own spread. WER over 298 words could not resolve these changes;
neither can a 44-frame two-prompt worst-sample, **unless the prompts are held fixed**.

**What the gate does and does not support:**
- ✅ **Paired, same-session, same-prompts A/B.** Deterministic — a repeat run reproduced
  bit-identically. The w2 table above is valid on this basis.
- ❌ **Absolute levels.** ❌ **Cross-session comparison.** ❌ **Generalising an effect measured on
  one prompt pair** — w2's +0.10 pp is cases 0,2 only and may differ elsewhere.

**§6.8's levels are unreachable today, cause unidentified.** It records mean 0.84% / p90 1.06%; the
best of 28 prompt pairs across 8 cases is **0.99% / 1.38%**, and no single prompt reaches it either.
Ruled out by direct check, so nobody repeats them: the gate code (identical since §6.8 bar print
statements), `ttnn_voxtral_gpt.py` (identical bar docstring prose), the Block 1 reference
(untouched), the ttnn build (artifacts dated Jul 22/30, *predating* §6.8), the prefill rows (folding
them in gives 0.92%), and prompt selection (above). §6.9 reproduced 0.84%/1.06% at the time in a
second run, so it was not a one-off transcription slip. Unexplained.

**Practical rule going forward:** gate on **all 8+ prompts, recorded by index, both arms in one
session**. Anything else in this port has been over-read.

**Where quality stands overall, all measured on the current build:**

| | value |
|---|---|
| Block 1 decode | min PCC 0.999737, mean WS 1.04%, p90 1.55% |
| Block 2 velocity | PCC 0.99998164, semantic code exact, 3 of 74 codes differ |
| Block 3 codec | waveform PCC 0.999915 — unchanged by the hang fix, bit-identical after §6.14 |
| end-to-end | long-form WER 0.34% = **1 word of 298**, 15/15 on `[END_AUDIO]` |
| voice identity | PASS (same-voice pair most similar, F0 109–197 Hz) |

**One measurement trap in this gate, so nobody quotes it.** `--gate decode` prints ms/step, and it
read 69 ms for BFP8 against 59 for bf16 — i.e. BFP8 "slower", which is backwards. It runs a 3.4B
fp32 reference step on the CPU between device steps, starving host dispatch. **The accuracy columns
are what this gate is for; its timings are host-contended.** Real numbers come from the pipeline:
0.05 s/frame, long-form RTF 0.61–0.65 (§6.21).

### 6.16 — the precision stack re-gated on all 15 prompts: w2 is 77% of the cost for 15% of the win

§6.15 fixed the gate (all 15 prompts by default, pooled + per-case spread, case list printed). This
is the measurement it was fixed for, and it **overturns §6.8's central claim** that each BFP8 step was
free on mean/p90. Five arms, 15 prompts x 22 teacher-forced frames = 330 frames each, all in one
session, only the weight dtypes differing. `ms/step` for the three cumulative rows is the recorded
ladder; the two leave-one-out rows are derived from it and marked ~.

| Block 1 weights | ms/step | mean WS | p90 WS | max WS | min PCC |
|---|---|---|---|---|---|
| all bf16 | 45.8 | **0.86%** | **1.18%** | 3.01% | 0.999357 |
| BFP8 except w2 (FF + attn) | 31.4 | 0.93% | 1.35% | 3.05% | 0.998969 |
| BFP8 except attn (FF + w2) | ~32.2 | 1.13% | 1.69% | 4.07% | 0.998810 |
| BFP8 except FF (attn + w2) | ~40.0 | 1.13% | 1.69% | 3.73% | 0.998859 |
| **all BFP8 ← shipped** | **28.9** | **1.17%** | **1.75%** | 4.34% | 0.998045 |

**The total accumulated cost is +0.31 pp mean / +0.57 pp p90, and min PCC 0.999357 → 0.998045.**
§6.8 measured the increments as 0.86 → 0.86 → 0.84 — i.e. free, three times over — on two unrecorded
prompts where the prompt spread (0.44 pp) dwarfed each increment (~0.1 pp). The increments were never
free; the noise hid them, and **the accumulated total was never checked at all**. That is the concrete
demonstration of §6.15's warning, and it is why the gate now defaults to 15 prompts.

**Priced per millisecond, reverting one thing at a time from shipped:**

| revert | mean WS recovered | p90 recovered | ms/step given back | **pp per ms** |
|---|---|---|---|---|
| **w2** | **−0.24 pp** | **−0.40 pp** | 2.5 | **0.096** |
| wqkv + wo | −0.04 pp | −0.06 pp | 3.3 | 0.012 |
| FF1 + FF3 | −0.04 pp | −0.06 pp | 11.1 | 0.004 |

The three effects sum to 0.32 pp against a measured total of 0.31, so they are **additive** — and
**w2 is the whole story: 77% of the accuracy cost for 15% of the speed.** FF1+FF3 are the opposite,
66% of the speed for 13% of the cost, and 24x better value than w2. Note also that reverting attn and
reverting FF buy *identical* quality (1.13% / 1.69%) — so if 0.04 pp is ever wanted back, revert
**attn**, which costs 3.3 ms, never FF at 11.1 ms.

**APPLIED — w2 is back in bf16.** `ttnn_voxtral_gpt.py` `WEIGHT_DTYPE`. FF and attn stay BFP8; they
are cheap in accuracy and carry the bulk of the speedup. Measured after the flip, not predicted:

| | w2 BFP8 | **w2 bf16 ← ships** |
|---|---|---|
| Block 1 mean / p90 WS | 1.17% / 1.75% | **0.93% / 1.35%** |
| Block 1 min PCC | 0.998045 | **0.998969** |
| RTF, 15 cases | 0.60–0.65 | **0.64–0.69** |
| long-form WER | 1 word of 298 | **0 words of 298** |
| `[END_AUDIO]` termination | 15/15 | **15/15** |
| voice identity | PASS | **PASS** |

RTF landed at 0.64–0.69 against a predicted 0.63–0.68. **Do not read the WER 1 → 0 as the win** — at
298 words that is one word, and §6.7 showed this gate cannot resolve changes this size; it is
consistent with the deterministic improvement, not evidence of it. The gate table is the evidence.

Frame counts DO shift (case 6: 177 → 211 frames): changing w2's precision changes the emitted codes,
which changes trajectory length. That is the chaotic-trajectory property documented in §6.7, not a
defect — every case still terminates on `[END_AUDIO]` rather than running to the cap.

This reverses the recommendation given when w2 shipped and again in §6.15. Both earlier calls rested
on a 2-prompt estimate that put w2 at +0.10 pp; on 15 prompts it is **+0.24 pp, 2.4x larger**. The
2-prompt number was not wrong for cases 0,2 — it was not generalisable, exactly as §6.15 warned, and
I then generalised it anyway.

**What has NOT changed:** end-to-end quality is still good on the shipped all-BFP8 build — long-form
WER 1 word of 298, 15/15 `[END_AUDIO]`, voice identity PASS, Block 2 3-of-74 codes, codec
bit-identical. min PCC 0.998 is still far above tt_transformers' 0.981. This is a
margin-and-headroom argument, not a "the model is broken" argument.

### 6.17 — Block 2: BFP4 rejected, and three op-level wins that improve accuracy too

Block 2 is ~23 ms of a ~51 ms frame and the module docstring put it ~1.6x above a 13.4 ms weight-read
floor, so that was the target. First finding: **the matmuls are already AT the floor.** The five
weight matmuls measured 13.28 ms of the block's 19.24 (168–205 GB/s each by prefix timing), leaving
~6 ms of non-matmul work inside `_block` plus 3.6 ms outside it.

**BFP4 WEIGHTS — REJECTED, and it fails on both axes at once.** Never previously tried; `bfloat4_b`
exists and halves the bytes again (0.5625 vs 1.0625 B/param). 8 draws, per-weight granularity:

| weights in BFP4 | GB/frame | ms/frame | vel PCC | acoustic codes wrong |
|---|---|---|---|---|
| none — ships | 2.60 | 25.70 | 0.9999816 | **19 / 576** |
| w1, w3 (49%) | 2.00 | 22.99 | 0.9998741 | 93 / 576 |
| w2 (24%) | 2.30 | 25.68 | 0.9999413 | 49 / 576 |
| wqkv, wo (27%) | 2.27 | 25.28 | 0.9996631 | 117 / 576 |
| all five (100%) | 1.37 | 22.56 | 0.9994959 | **159 / 576** |

**8.4x the differing codes for 1.139x.** sdpa was rejected twice at 3x the codes for 1.147x — BFP4
is the same speed for nearly triple the damage. w2 in BFP4 is the clearest reject of all: 0.016 ms
(nothing) for 2.6x the errors.

**But the more useful half of that result is the timing.** Cutting bytes 47% (2.60 → 1.37 GB) returned
only 12% of the time (25.70 → 22.56 ms), where the bandwidth model predicts ~6.3 ms. **So weight bytes
are NOT the lever in Block 2 that they are in Block 1** — halving them returns about a quarter of what
the arithmetic promises, and the "1.6x above the floor" framing overstates how much of Block 2 that
floor governs. Do not plan future Block 2 work around byte counts.

**SHIPPED — three op-level changes, +1.19 ms/frame in isolation, and accuracy slightly BETTER:**

| | ms/frame | vel PCC | acoustic wrong |
|---|---|---|---|
| shipped before | 25.86 | 0.9999816 | 19 / 576 |
| A — silu fused into the w1 matmul | 25.70 | 0.9999816 | 19 / 576 |
| B — `SCALE` folded into wqkv's q rows | 25.58 | 0.9999851 | 16 / 576 |
| C — `_trunk` slices 36-wide, not 3072 | 24.77 | 0.9999816 | 19 / 576 |
| **A+B+C** | **24.67** | **0.9999851** | **16 / 576** |

- **A** — `activation="silu"` on the w1 linear. Bit-identical, and Block 1 has always done it; Block 2
  just never picked it up. Not the idea §6.8 rejected — that was fusing silu into the *multiply* via
  `input_tensor_a_activations`, which really is worthless.
- **B** — `1/sqrt(head_dim)` lives in the weight instead of a `multiply(s, SCALE)` per block call.
  21 launches gone, and **more accurate**: the scores round once instead of twice.
- **C** — the biggest, 1.09 of the 1.19. `_trunk` used to reshape to `[B,3,3072]`, slice position 0 at
  3072 wide, then project to 36. Now it projects the 6-row sequence first and reshapes/slices at 36
  wide: 85x less data moved, and the linear drops from batch-2 (which re-reads the whole weight per
  batch element) to batch-1 over 6 rows in one tile. **Same lesson as the codec in §6.13/6.14 — shift
  the narrow side** — which is now the third time it has paid here.

Gates: velocity PCC 0.99998164 → **0.99998510**, codes differing 3/74 → **2/74**, semantic exact,
long-form WER **0 of 298**, 15/15 `[END_AUDIO]`, voice identity PASS.

**⚠ THE ISOLATED WIN DOES NOT FULLY APPEAR END TO END.** Isolated Block 2 is −1.19 ms/frame
(deterministic, 30 reps). The pipeline over 8 comparable cases is **−0.36 ms/frame**, and the longest
case (463 → 485 frames) is −0.37. Per-case scatter is ±1 ms. I cannot account for the factor of ~3.
Quote **−0.4 ms/frame end to end**, ~0.7%, and treat the isolated number as evidence the change is
real rather than as the user-visible gain. All three are strictly fewer ops with no accuracy cost, so
they ship regardless of which number you prefer.

**A measurement trap this run walked into.** Raw RTF showed wild outliers (case 0 at 5.01, case 2 at
1.12) that survived a warm re-run and looked like a regression. They are not: changing Block 2's
numerics changes the emitted codes, which changes trajectory LENGTH (case 2: 443 → 465 frames, case
10: 211 → 271), and a new length is a new codec bucket that compiles ~5 conv programs at 1–5 s each.
Case 4 separately collapsed 73 → 8 frames — it is the chaotic one-word prompt. **Comparing RTF across
builds requires excluding cases whose frame count moved, or the compile lands in the number.** Same
family as the trap in §6.7 and the "case 0 RTF 1.89" one.

### 6.18 — the sharded norm's grid: §6.9's "8x2/8x4 are slower" is WRONG, and 32 cores ships

Prompted by "have we tried other `block_w` numbers, maybe smaller is faster?" -- which turned out to
rest on a false premise and still find something.

**`block_w` is not a free knob.** It is `DIM // cores // TILE`, so at 8 cores each core owns
3072/8 = 384 columns = 12 tiles and the kernel must walk 12. It only moves when the core count does,
and the two are the same decision.

**`subblock_w` IS free, was hard-coded at 4 in both blocks since the sharded norm shipped, and had
never been swept. It is inert.** Within a core count, 1/2/3/4 land within 0.02 ms/step of each
other, and at 8x1 in Block 2 subblock_w 4 and 1 measure *byte-identical* (24.628 both). `>= 6` does
not build at all -- a hard register-budget limit. So the knob nobody had tested turns out not to
matter, which is worth knowing precisely so nobody tests it again.

**What DOES matter is the core count**, contrary to §6.9. **THE COUNT MUST DIVIDE THE TILE COUNT.**
A 32x3072 tensor is 1 x 96 tiles, a tile is the indivisible unit, and `block_w` IS the quotient --
tiles per core. So only divisors of 96 are legal, which rules out 40, 56 and **64** (96/64 = 1.5).
That is why 8x8 does not build rather than merely running slowly.

Block 1, `_layer_step` x26, interleaved round-robin:

| grid | cores | 96/cores | legal? | isolated norm | **ms/step** |
|---|---|---|---|---|---|
| 2x1 | 2 | 48 | yes | 43.5 µs | 25.53 |
| 4x1 | 4 | 24 | yes | 43.9 µs | 24.84 |
| 8x1 ← was shipped | 8 | 12 | yes | 45.6 µs | 24.57 |
| 8x2 | 16 | 6 | yes | 48.2 µs | 24.45 |
| 8x3 | 24 | 4 | yes | — | 24.42 |
| **8x4 ← ships** | **32** | **3** | **yes** | **54.6 µs** | **24.41** |
| 8x5 | 40 | 2.4 | **no** | — | — |
| 8x6 | 48 | 2 | yes | — | 24.44 |
| 8x7 / 8x8 | 56 / 64 | 1.71 / 1.5 | **no** | — | — |

**IT IS NOT MONOTONE -- IT HAS A MINIMUM AT 32, AND I CLAIMED OTHERWISE.** The first sweep tested
2/4/8/16/32 by doubling, missed the non-power-of-two counts, saw a monotone curve and concluded "32 is
the largest that divides the work evenly." Both halves were wrong: **48 also divides evenly**, and it
is **slower** (24.44 vs 24.41, against a 0.007-0.011 ms spread, so ~4x the noise). 32 is the measured
optimum, not the largest legal grid. Plausibly the cross-core reduce and shard scatter start to
dominate once each core holds only 2 tiles, but that mechanism is a guess -- only the ordering is
measured.

**THE ISOLATED NORM RANKS THESE BACKWARDS.** 8x4 is the SLOWEST in isolation (54.6 µs vs 43.5) and
the FASTEST end to end. §6.9 concluded "the core count barely matters" and "8x2/8x4/8x8 are all
slower" from isolated numbers plus a partial 2-vs-8 end-to-end check — and the isolated measurement
is anti-correlated with the thing we care about. **A norm cannot be benchmarked alone**: it is ~16 µs
of reduction inside ~28 µs of resharding, and the reshard's cost depends on what consumes it next.

**Confirmed before believing it**: 7 interleaved rounds, round-robin so drift hits all arms equally.
Spread 0.007-0.013 ms per config and the ordering identical in every round — 8x1 24.571, 8x2 24.451,
8x4 24.407.

**Accuracy — paired, same session, all 15 prompts.** The 26-layer output differs by 1.0 absolute
between grids (a different cross-core reduce tree, amplified through 26 layers), which is the same
order as the whole w2 precision argument in §6.16, so it needed the real gate rather than a shrug:

| | 8x1 | **8x4** |
|---|---|---|
| mean worst-sample | 0.93% | **0.92%** |
| p90 worst-sample | 1.35% | **1.28%** |
| min PCC | 0.998969 | **0.999040** |
| per-case p90 spread | 0.55 pp | **0.44 pp** |

Better or equal on every column. Block 2 likewise: 24.628 → 24.465 ms/frame with acoustic codes
16/576 → 15/576 and velocity PCC 0.9999851 → 0.9999848. Gates after shipping both: flow velocity PCC
0.99998480, 2/74 codes, semantic exact, long-form WER **0 of 298**, 15/15 `[END_AUDIO]`, voice
identity PASS, 122/122 tests.

**Size, stated plainly.** Isolated: +0.163 ms/step in Block 1 and +0.163 ms/frame in Block 2, so
+0.33 ms/frame. End to end over 9 comparable cases: **−0.24 ms/frame**, ~0.5%. Same shortfall as
§6.17 and still unexplained. Quote the pipeline number. It ships because it is free — faster on a
stable measurement and no worse on any accuracy column — not because 0.5% matters.

**The transferable bit:** a config value that was copied once and never questioned (`subblock_w=4`)
turned out inert, while the one that *had* been measured and closed (the core count) was closed on
the wrong measurement. Re-check closed doors when the thing you closed them with was a proxy.

### 6.19 — `_QKV_SHARD`'s core count re-swept: inert above 6, and a different divisor set

Asked whether the qkv shard's core count had been tried. It had -- 8/12/16/24/48, recorded flat at
31.36-31.46 ms -- but on a build with w2 in BFP8 and the norm on 8x1, so it needed redoing against
what ships. It also only swept DOWN TO 8, and §6.18 had just shown a curve with an interior minimum.

5 interleaved rounds, mean ms/step, shipped build:

| cores | grid | cols/core | heads/core | ms/step | output diff |
|---|---|---|---|---|---|
| 1 | 1x1 | 6144 | 48 | 24.683 | 0.0 |
| 2 | 2x1 | 3072 | 24 | 24.520 | 0.0 |
| 4 | 4x1 | 1536 | 12 | 24.436 | 0.0 |
| 6 | 6x1 | 1024 | 8 | 24.416 | 0.0 |
| **8 ← ships** | **8x1** | **768** | **6** | **24.410** | **0.0** |
| 12 / 16 / 24 | 6x2 / 8x2 / 8x3 | 512 / 384 / 256 | 4 / 3 / 2 | 24.412 / 24.419 / 24.409 | 0.0 |
| 32 | 8x4 | 192 | **1.5** | **illegal** | — |
| 48 | 8x6 | 128 | 1 | 24.424 | 0.0 |

**Inert from 6 cores up** -- the whole 6-to-48 range sits inside the 0.020 ms within-config spread,
and 24's nominal 0.002 ms lead is 10x below the noise. **Below 6 it does cost**: one core is
+0.273 ms. So the old "it does not matter" needs the qualifier *at or above ~6*. Nothing to ship.

**The output is BIT-IDENTICAL at every count**, because this shard is pure data placement with no
reduction to reorder — the opposite of the norm, where changing the reduce tree moved the 26-layer
result by 1.0 absolute (§6.18).

**AND THE GRID NEVER REACHES THE CONSUMERS** — asked directly, since sharding differently ought to
affect `nlp_create_qkv_heads_decode`, `paged_update_cache` and `sdpa_decode`. It does not:
that op imposes its own output layout, and at input grids of 1 / 6 / 8 / 24 / 48 cores, `qh`, `kh`
and `vh` all come out **1 core, shard (32, 128)**, identically. So the two cache writes and
`sdpa_decode` never see this config; only the shard fill and the split op can. That is a structural
reason for the flatness, and a better one than "the aggregate did not move."

**A failed attempt to confirm that, worth recording as a method warning.** Timing the consumer chain
*in isolation* inverts the answer: it makes 1 core look 0.157 ms CHEAPER where the 26-layer step has
it 0.273 ms DEARER. Cause: the isolated probe feeds a pre-computed qkv tensor from DRAM instead of the
wqkv matmul's fresh output. The prefix split also produced a **negative** marginal cost for
`paged_update_cache V`, which is proof the method had broken down at these sizes rather than a small
inaccuracy. **Third instance today of the same failure** — after the norm grid (§6.18) and
`[gpt-05]`'s stale contrast — of an isolated measurement disagreeing with, and losing to, the
end-to-end one.

**Two different divisor sets, and it explains why 32 is the norm's optimum and illegal here.** The
qkv shard's unit is the HEAD: 6144 = 48 heads x 128 and the consumers want whole heads per core, so
the count must divide **48**. The norm's unit is the 32-wide TILE, so its count must divide **96**.
32 divides 96 but not 48. Read the unit before assuming a grid transfers between ops.

**Fixed a footgun while in there.** The config read `(TILE, _QKV_WIDTH // 8)` next to
`core_grid=CoreGrid(y=1, x=8)` -- one number written twice, where changing the grid and forgetting
the divisor yields a *silently wrong shard* rather than an error. Both now derive from `_QKV_GRID_X`.

**And a correction to `NOTES.md [gpt-05]`**, which argued the norm's count "DOES matter: 16.2 us on 8
cores against 21.2 on 32" as a contrast to this one. Those are isolated-norm timings, the metric
§6.18 showed is anti-correlated with end-to-end. The real contrast is much narrower: both grids are
nearly free, the norm's end-to-end spread is 0.16 ms and this one's is 0.02.

### 6.20 — sharding control DOES unlock something: 0.454 ms/frame, bit-exact

Asked whether hand-rolling `nlp_create_qkv_heads_decode`, `paged_update_cache` and `sdpa_decode`
would be faster, since we could then control the sharding. **Hand-rolling: no, and it was already
tried. The underlying intuition: yes, and it was worth the largest single win of the session.**

**Why not hand-rolling.** A hand-rolled decode interior was built and measured, and the fused
decode-native path beat it by **6.6 ms/frame** ([gpt-05]) — more than double the entire 2.34 ms those
three ops cost today. Block 2's hand-rolled head split measured 158 µs against the fused op's 122
([flow-10]). Cost here is per-LAUNCH, and every hand-rolled decomposition trades one op for three to
eight, so it is the wrong direction by construction.

**But the sharding really was blocking a fused op.** `ttnn.experimental.paged_fused_update_cache`
exists, writes both caches in one call, and we had never used it. It refuses K and V on the same
core — and `nlp_create_qkv_heads_decode` puts q, k and v all on core (0,0) by default. The chain to
unblock it, each link forced by the next:

1. `nlp_create_qkv_heads_decode(..., overlap_qk_coregrid=False)` puts q on (0,0), k on (1,0).
2. That flag asserts `head_dim % shard_width == 0` — a whole head per core. `_QKV_WIDTH/48 = 128` is
   **exactly one head**, and the only feasible width: 64 columns would need 96 cores. So the qkv
   shard moves 8 → 48 cores, which the §6.19 sweep had already shown costs nothing (+0.014 ms, inside
   the 0.020 ms spread).
3. With k on (1,0), rope breaks — **silently, returning 3.376e+38, uninitialised L1** — because
   `_ROPE_SHARD` pins cos/sin to (0,0) and rope reads its table from the tensor's own core. Fixed with
   a second cos/sin pinned to (1,0). Built once per FRAME, so ~4 extra launches.
4. `paged_fused_update_cache` then validates, and 52 cache-write launches per frame become 26.

| | ms/step | vs shipped | output |
|---|---|---|---|
| shipped: overlap=True, 2x update | 24.406 | — | — |
| overlap=False + k-rope, 2x update | 24.388 | +0.017 | **bit-exact** |
| **overlap=False + k-rope, 1x fused update** | **23.952** | **+0.454** | **bit-exact** |

Spread 0.004–0.014 ms, so 0.454 is ~30x the noise. Pipeline: **−0.44 ms/frame**, long-form RTF
**0.61–0.65**, WER 0 of 298, 15/15 `[END_AUDIO]`, voice identity PASS, 122/122 tests. Per case, on the
14 whose frame count is unchanged, **14 improved and 0 got worse** — deltas −0.00 to −0.04 RTF.

**Bit-exactness, and how far the evidence actually goes.** `--gate decode` on 15 prompts prints
byte-identical stats, but those are ROUNDED, so that is equality to 2-3 significant figures, not proof.
The real evidence is three direct comparisons: the fused cache write is identical at **18 positions**
(0, tile boundaries 31/32/33, 63/64, 127/128, 200, 311/312/313, 511/512, 767, 1023); Block 1 decode is
identical at **every one of 64 sequential steps** with a growing cache; and case 10's real frame loop
is identical **frame-for-frame over 221 frames**, hidden state and codes both.

**AND THE ISOLATED AND PIPELINE NUMBERS AGREE FOR ONCE** — 0.454 against 0.44, where §6.17 and §6.18
each showed a factor of ~3 shortfall. The difference is what the change *does*: removing launches is
layout-neutral, so nothing downstream re-optimises around it. Grid and blocking changes alter how
every consumer receives its data, which is exactly why they measure differently in isolation.

**⚠ SUPERSEDED BY §6.22.** The chain above (48-core shard, `overlap_qk_coregrid=False`, a second
rope table) works and is what shipped first, but it moves **K**. Moving **V** instead reaches the same
place with none of the coupling, measures the same end to end, and is what ships now. The fused write
itself, and everything above about why it needs non-overlapping cores, still stands.

### 6.21 — how to quote RTF, and a harness confound that cost an hour

Asked why RTF read 0.61–0.71 when it had been 0.64–0.69. **Both numbers were badly derived by me, in
different ways, and neither was comparable to the other:**

- **0.64–0.69** was eyeballed off a `tail -20` that showed only 10 of the 15 cases. The real range for
  that build, excluding case 0, was 0.62–0.69.
- **0.61–0.71** came from a filter `rtf < 1.2`, which dropped case 0's first-call compile but kept
  case 4 — the chaotic one-word prompt at 43 frames — as the top end.

So the "widening" was two inconsistent filters, not the model. Like for like, **14 of 14 comparable
cases improved and none got worse.**

**RTF is dominated by utterance LENGTH, not by compute speed**, because it is wall/audio and the fixed
costs (prefill, codec, first-call compiles) amortise over however much audio there is:

| bucket | frames | RTF |
|---|---|---|
| long-form | >= 100 | **0.61–0.65** |
| short | 20–99 | 0.62–0.71 |
| case 0 | any | 1.61 — carries the first-call compile |

**THE CONVENTION, from here on: quote long-form RTF (>= 100 frames).** It is the only bucket where
fixed costs are amortised enough to reflect per-frame speed. Quote short and case-0 numbers separately
or not at all, and never mix filters between two builds.

**And a harness confound worth more than the reporting fix.** Case 10's frame count moved 207 → 220
between two builds, reproducibly, which looked like a real numerical divergence and took an hour to
chase. It is not: **run case 10 ALONE and both builds give 220.** Its length depends on what ran
before it in the same process. `generate()` reseeds per call, so it is not RNG — the pipeline object
and its KV cache are reused across cases. **Determinism is not independence:** two runs of one build
match exactly (verified, 0 of 15 cases differ), which is what made the artifact look trustworthy.
**So a changed frame count in a multi-case run is not evidence of a changed model.** Confirm it by
running that case alone before believing it.

### 6.22 — move V, not K: same speed, none of the coupling (from the xtts branch)

Asked to look at how `lserbedzija/xtts-gpt-ttnn` does the same fusing, since it is "a bit different".
It is, and it is better. `xtts_v2/tt/ttnn_xtts_gpt.py` keeps `nlp_create_qkv_heads_decode` at its
default and **moves V to core (1,0)** with one `to_memory_config`, where §6.20 moved **K** via
`overlap_qk_coregrid=False`.

That branch is GPT-2 with learned positional embeddings, so it never had the RoPE problem that made
moving K awkward here. **Moving V sidesteps it entirely** — V never passes through RoPE.

| | moves | launches added | isolated ms/step | pipeline mean RTF, 12 like-for-like cases |
|---|---|---|---|---|
| baseline, 2 writes | — | — | 24.405 | 0.6539 |
| §6.20, `overlap_qk_coregrid=False` | K | 4 / frame | 23.953 (+0.452) | 0.6485 |
| **§6.22, `to_memory_config(v)`** | **V** | **26 / frame** | 24.000 (+0.405) | **0.6483** |

Both bit-exact. Moving K is 0.047 ms/step faster in isolation — **and indistinguishable end to end**
(0.6485 vs 0.6483, where per-case RTF resolution is 0.01). The 22 extra launches cost ~2.1 µs each,
much less than the 5-20 µs a launch usually runs here, because moving V is an 8 KB hop between
adjacent cores.

**So V ships, for robustness at no measurable cost.** Moving K carried two hazards, both now gone:
- `_QKV_SHARD` had to be **48 cores** (a whole head per core) or `overlap_qk_coregrid=False` would not
  build — a load-bearing constraint on a value §6.19 had just established as inert, i.e. exactly the
  kind of coupling that reads as free to change and is not.
- K then went through RoPE on a core whose cos/sin table lived elsewhere, which **does not raise** —
  it returns 3.4e38 from uninitialised L1. A second table fixed it, but the failure mode stayed.

Moving V needs neither: the shard width is free again and there is one rope table.

**A reporting slip worth recording, since §6.21 had just warned about it.** My first read of this
comparison said move-V was *worse* — mean RTF 0.7227 → 0.7302 over 13 cases. That mean included case
0, whose 1.6 RTF is a first-call compile, and it swamped everything. Excluding case 0 and any case
whose frame count moved gives 0.6485 vs 0.6483. **I wrote the convention an hour earlier and then
broke it in the next measurement.**

Gates: `--gate decode` byte-identical on all six columns, long-form WER 0 of 298, 15/15
`[END_AUDIO]`, voice identity PASS, 122/122 tests, long-form RTF 0.61–0.66.

### 6.23 — `rotary_embedding_llama_fused_qk`: REJECTED, wrong rotation convention for our weights

The two `rotary_embedding_hf` calls are 52 launches a frame, so a fused q+k pair was the obvious next
target after §6.20/§6.22. It exists, it is faster, and **it cannot be used without undoing a
deliberate weight decision.**

**IT IMPLEMENTS THE INTERLEAVED CONVENTION; WE USE HALF-SPLIT.** `get_rot_transformation_mat`'s own
docstring gives it away — the trans_mat "pairs ADJACENT dimensions ... [0,1] -> +1 at (0,1), -1 at
(1,0)", i.e. interleaved-pair rotation on `(r0,i0,r1,i1,...)`. Confirmed numerically rather than read
off a comment, by feeding one vector written both ways and comparing against explicit torch
references for each convention:

| fused op output vs | max diff |
|---|---|
| torch **interleaved-pair** reference | **9.6e-03** (bf16-level — correct) |
| torch **half-split** reference | **3.67** (wrong) |

Our checkpoint is Mistral-native and therefore interleaved, but `interleaved_to_halfsplit` permutes
wq/wk ONCE at load so the easy `rotate_half` form applies (module docstring / NOTES.md [gpt-01]).
Using the fused op means **reverting that permute** — and rope getting this wrong does not raise, it
produces fluent nonsense.

**What it would buy, measured at our shapes:**

| | µs/layer | ms/frame |
|---|---|---|
| 2 x `rotary_embedding_hf` (ships) | 33.5 | 0.871 |
| 1 x `rotary_embedding_llama_fused_qk` | 24.4 | 0.635 |
| | | **+0.236 (1.373x on this pair)** |

**What it would cost.** Un-permuting wq/wk, the riskiest class of change in this port. q and k back on
**disjoint cores** — the exact coupling §6.22 just removed. cos/sin rebuilt per frame as
`[1, 2*batch, 32, head_dim]`, height-sharded across both cores (it must be HEIGHT_SHARDED — it asserts).
A trans_mat sharded over >= q_cores + k_cores with (32,32) shards. And it is **not bit-exact**: the
rotation runs through a tile matmul.

**0.236 ms/frame is 0.46% of a frame. Not worth any of that**, and notably worse value than the fused
cache write, which bought 0.405 ms bit-exactly for one inline reshard and no convention change.

**Recorded because the convention table is the useful part**, and it is not written down anywhere
obvious: `rotary_embedding_hf` wants **half-split**, `rotary_embedding_llama` and
`rotary_embedding_llama_fused_qk` want **interleaved** via a trans_mat. `tt_transformers/tt/attention.py`
keeps `_hf_rope_decode` and `_mllama_rope_fused_qk_decode` as separate paths for exactly this reason —
the fused variant sits only on the llama side. If our weights were ever left un-permuted for another
reason, this becomes free money; until then it is not.

### 6.24 — fusing w1 and w3 is 4x SLOWER: matmul bandwidth collapses past ~9216 output columns

Asked whether `h@w1` and `h@w3` should be one matmul, the way qkv is. It should not, and the reason
turned out to be much bigger than the one on record.

**The prize was small before measuring anything.** w1 and w3 are each 3072x9216 BFP8 = 28.7 MB and run
at **190 GB/s — 98% of this device's 194 GB/s ceiling**. Together they read 57.4 MB, which at the
ceiling is 310.1 µs against the 316.5 they take. **The entire theoretical gain from removing one launch
is 6.4 µs/layer = 0.17 ms/frame.** Contrast the qkv fusion, which paid 1.449x because `wkv` alone was
2048 wide and cost the same as the 4096-wide `wq` — there was launch overhead to reclaim. Fusing pays
when one of the pair is too NARROW to earn its launch, and 9216 is not narrow.

**Measured, one layer's MLP, same 57.4 MB of weight either way:**

| | ops | µs | vs A | result |
|---|---|---|---|---|
| A — `linear(w1, silu)` + `linear(w3)` + multiply ← ships | 3 | **313.9** | 1.000x | — |
| B — fused + 2 slices + silu + multiply | 5 | 1253.3 | **0.250x** | same values |
| C — fused + `ttnn.swiglu` | 2 | 1253.8 | **0.250x** | same values |

**Both fused forms are 4x slower, and C has one op FEWER than what ships** — so this is not about op
count at all. The fused matmul itself is the problem:

| | MB read | µs | GB/s | % of ceiling |
|---|---|---|---|---|
| two separate 3072x9216 | 57.4 | 313.9 | **192** | **99%** |
| one fused 3072x18432 | 57.4 | 1253.5 | **48** | **25%** |

**Identical bytes, 4x the time. Matmul efficiency collapses somewhere between 9216 and 18432 output
columns**, and that is the transferable result: 3072 (w2), 6144 (wqkv) and 9216 (w1/w3) all hold the
ceiling; 18432 does not. **Any future fusion that pushes N past ~9216 should be assumed to collapse
until measured.** This also retro-explains why the qkv fusion was safe — it lands at 6144.

**Note this contradicts the recorded Block 2 number** (0.998x, and 0.951x charging the output split),
which is far milder than 0.250x at the same weight shapes. Block 2's measurement was on the whole
block rather than the MLP alone, so the fused matmul's cost was diluted; either way the direction is
the same and the isolated figure is the honest one for this decision.

**Two incidental findings about `ttnn.swiglu`**, since it looked like the elegant answer (one matmul
plus one fused SwiGLU = 2 ops, no split, no separate silu):
- it needs a **4-D** tensor. On `[1,1,18432]` it throws `ShapeBase[] index out of range. 3 not in [...]`
  from `shape_base.cpp:16` — it indexes `shape[3]` unconditionally.
- it applies SiLU to the **second** half and multiplies by the first, so our MLP needs the fused weight
  ordered `[w3 | w1]`, not `[w1 | w3]`. Both orders were measured; both collapse.
- and it operates on the PHYSICAL tile rows, so a `[1,1,1,18432]` input yields `[1,1,32,9216]` — the
  31 padding rows come through and would need slicing off.

Also worth stating: the fused weight is a **duplicate** of w1 and w3 concatenated — 57 MB/layer,
**1.46 GB over 26 layers** against a 3.9 GB working set. Even at parity it would be a poor trade.

### 6.25 — wo gets a program config: +0.196 ms/frame bit-exact, and the knob is `in0_block_w`

§6.24's ceiling argument produced a target. Each decode linear's time is bytes/ceiling plus overhead,
and the overhead is all a restructuring can attack — **only wo had any**:

| weight | bytes | floor µs | actual | overhead |
|---|---|---|---|---|
| wqkv | 20,054,016 | 103.4 | 103.2 | −0.2 → 0% |
| **wo** | 13,369,344 | 68.9 | 85.9 | **+17.0 → 20%** |
| w1 / w3 | 30,081,024 | 155.1 | 158.1 / 158.4 | +3.0 / +3.3 → 2% |
| w2 | 56,623,104 | 291.9 | 284.1 | −7.8 → −3% |

**THE KNOB IS `in0_block_w`, NOT THE CORE COUNT — my hypothesis was wrong.** I went in expecting
N=3072's 96 tiles splitting 1.5-per-core over 64 cores to be the problem. It is not: 8x8 / 8x6 / 8x4
measure 68.2 / 68.1 / 68.5 µs, i.e. the grid barely matters. What matters is how many of K's 128 tiles
load per inner iteration:

| in0_block_w | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| µs | 152.0 | 83.2 | **68.1** | 73.7 | 80.5 | 89.4 |
| GB/s | 88 | 161 | **196** | 182 | 166 | 150 |

The default runs at 162 GB/s; `in0_block_w=4` runs at **196 — the ceiling**. Both directions lose.

**TWO OPTIONS, and the bit-exact one ships:**

| config | µs | vs default | ms/frame | accuracy |
|---|---|---|---|---|
| default | 82.3 | 1.000x | — | baseline |
| **8x4, in0_block_w=2 ← SHIPS** | 74.7 | 1.102x | **+0.196** | **byte-identical on all six gate columns** |
| 8x6, in0_block_w=4 | 68.1 | 1.209x | +0.370 | mean +0.02, p90 +0.01 pp; min PCC 0.999040 → 0.999219 (better); max 3.05% → 4.76% |

The faster one costs 0.054 pp of mean per ms — the same ballpark as w2's 0.096 that §6.16 rejected —
and takes the worst sample from 3.05% to 4.76%. Max is an unstable order statistic and not decisive on
its own, but 0.174 ms/frame is 0.34% of a frame and not worth the argument. **`in0_block_w=4` is left
documented and available** if that trade is ever wanted; nothing else needs to change to take it.

Verified: 122/122 tests, decode gate byte-identical, long-form WER 0 of 298, 15/15 `[END_AUDIO]`,
voice identity PASS. Pipeline **−0.0024 RTF = −0.19 ms/frame** against the isolated +0.196 — they agree,
as in §6.22, because this changes no layout that a consumer has to re-optimise around.

**DECODE ONLY.** `per_core_M=1` assumes M is one tile, true of a single decode position and false of
prefill's S rows, so `_layer` deliberately has no program config.

**⚠ THE NEAR-MISS, and it is the most useful thing here.** The first sweep flagged every
`in0_block_w=4` variant as **WRONG** — they differed from the DEFAULT by 5.3e-03 relative, and those
were exactly the fast ones. Checking against a torch **fp64** reference instead showed every config
sitting at the *same* 6.640e-04 from truth, with PCC 0.9999742 vs the default's 0.9999745. **The
default is not ground truth.** Comparing a variant against it rather than against the real answer
almost discarded a genuine 1.2x as a correctness failure.

This also retires the older "hand-tuned matmul program configs measured SLOWER (169 vs 193 GB/s)"
finding as scope-limited rather than wrong: that was measured on **wq**, already at 94% of its floor.
Sweeping an op with no headroom finds none — which is why §6.24's floor table is the right way to pick
what to sweep.

### 6.26 — device tracing re-measured a third time: same answer, +0.35 ms/frame, still not taken

Re-run against the shipped config, because §6.6's numbers predate everything from §6.17-§6.25 and this
port's own rule is to re-measure rather than quote. Both probes are unchanged from the scratchpad; only
the build under them moved.

| | untraced (ships) | traced | delta | correct? |
|---|---|---|---|---|
| Block 1, 26-layer decode step | 24.86 ms | 24.69 ms | **+0.17 ms (1.007x)** | min PCC 0.999883 and worst-sample 1.28% IDENTICAL |
| Block 2, whole 7-step solve | 21.11 ms | 20.93 ms | **+0.18 ms (1.009x)** | integer codes bit-identical to untraced |
| **total** | | | **+0.35 ms/frame (~0.7%)** | |

**The verdict is unchanged and the number barely moved** — §6.6 measured +0.36 ms/frame on a build
10 ms/step slower. The absolute gain held while the base shrank, so as a *fraction* it grew slightly
(0.58% → 0.68% on Block 1). That makes sense: tracing recovers host command submission, which scales
with op COUNT, and today's work removed ~24 launches of ~476 — about 5%.

**A correction to my own reasoning for running this.** I suggested re-measuring "now that the launch
count has dropped", which is backwards: fewer launches means LESS host dispatch for a trace to recover,
so the expected direction was down. The legitimate reason was only that the recorded numbers described a
config that no longer exists. They turned out to still hold.

**Why it stays out at a gain comparable to things that shipped today.** 0.35 ms/frame sits right
alongside the fused cache write (0.405), the Block 2 op wins (0.44 isolated) and wo's program config
(0.196). On magnitude alone it is shippable. The difference is the FAILURE PROFILE, and it is not a
matter of taste:

| shipped today | how it fails if wrong |
|---|---|
| fused cache write | `paged_fused_update_cache` asserts on overlapping cores — **raises** |
| wo program config | wrong `per_core_M`/shape — **raises** |
| move V to core (1,0) | wrong placement — **raises** |

| tracing | how it fails if wrong |
|---|---|
| buffers held by POINTER | a fresh upload is silently ignored and the previous frame replays — **silent** |
| warm-up/capture write the KV cache | corrupts the prompt unless aimed at a scratch row; measured decode PCC 0.9998 → 0.86 with **no error raised** |
| capture ORDER across the two blocks | all warm-ups must precede either capture, or the second **hangs** |
| an exception inside a capture | `close_device` hangs and **every later run on the card blocks** — trap #1, cost ~20 min and a kill by PID once |
| `trace_region_size` | required on every caller of `open_device` |

Three of those are silent and one wedges the board. Everything shipped today fails loudly. **0.7% is
not worth converting four loud failure modes into three silent ones**, and that is the whole argument —
not the size of the gain.

**If it is ever wanted anyway**, tracing ONE block avoids the cross-block capture-order constraint for
about half the gain, and the probes (`probe_trace_b1.py`, `probe_trace_b2.py`) are current and ready.

### 6.27 — the overhead map, and what it found: sdpa_decode was 90% setup

§6.24's floor method applied to EVERY decode op, not just the matmuls: measure isolated, compute a DRAM
byte floor, rank by absolute overhead. That map answered "is there overhead left" precisely.

**The matmuls are finished.** wqkv, w3 and w2 now measure FASTER than a 194 GB/s floor (−1.7, −7.5,
−11.2 µs), which also says that ceiling is slightly conservative — w2 implies 202 GB/s. w1 is +3.0 and
wo +6.4 after §6.25. Nothing there.

**Everything left is small ops**, and the isolated overheads are large in relative terms:

| op | floor µs | actual | overhead | ×26 |
|---|---|---|---|---|
| **sdpa_decode** (pos=312) | 6.6 | 68.1 | **+61.6** | **1.60 ms** |
| ffn norm ×3 | 0.1 | 53.3 | +53.2 | 1.38 ms |
| multiply(g, w3) | 0 | 29.7 | +29.7 | 0.77 ms |
| add residual ×2 | 0 | 28.9 / 26.7 | +55.6 | 1.45 ms |
| the other 9 small ops | ~0.2 | ~185 | ~+185 | ~4.8 ms |

**But the reachable total is far smaller than that sums to.** Isolated ops sum to 29.930 ms against a
shipped step of 23.803 — 6 ms is already overlapped away — and the byte floor is 20.309 ms. So **at most
3.494 ms/frame, 15% of the step, is reachable**, and an op's isolated overhead is an upper bound on
that op rather than a promise.

**SHIPPED — `SDPAProgramConfig` on sdpa_decode, +0.673 ms/frame isolated, bit-exact.** The shipped call
passed no program config. Diagnosing the 10× gap first mattered: a **31× larger cache costs 21% more
time** (pos 32→1000, 0.12→3.91 MB, 68.4→82.5 µs), so ~62 of the 68.6 µs is FIXED setup, not the read.

| | µs | vs default |
|---|---|---|
| default | 68.6 | 1.000x |
| **k_chunk=512, grid=8x2 ← ships** | **42.2** | **1.625x** |
| k_chunk=256, grid=8x2 | 38.7 | 1.772x |
| k_chunk=32, any grid | 48–65 | **1.0e+00 rel — genuinely broken** |

**⚠ ONLY A POSITION SWEEP MAKES THIS SAFE.** Verified at 11 positions from 64 to 1000, spanning the
chunk boundary at 511/512/513, against both the default and a torch fp64 reference:

| config | bit-exact at | worse than default vs fp32 | ms/frame |
|---|---|---|---|
| **k=512 8x2** | **11/11** | 0 | 0.673 |
| k=512 8x1 | 6/11 | 0 | 0.706 |
| k=256 8x2 | 3/11 | **3** | 0.825 |

**`k=256` looks fine at pos=312 and is worse than the default at 480, 511 and 700.** The decode gate
pins ONE position per case, so it would have passed and degraded long utterances silently. Any
position-dependent config needs a position sweep, not a gate run.

Verified: 122/122 tests, decode gate byte-identical on all six columns, long-form WER 0 of 298, 15/15
`[END_AUDIO]`, voice identity PASS. Pipeline **−0.33 ms/frame** against +0.673 isolated — about half
shows up, since sdpa partly overlaps neighbouring work.

**REJECTED — folding the residual adds into their matmul as a bias.** In decode M=1, so the residual is
a `[1,3072]` row vector, exactly a bias. It works, and it is worth almost nothing:

| form | µs | vs 2-op | numerics |
|---|---|---|---|
| w2: `add(x, linear(u, w2))` ships | 285.4 | — | — |
| w2: `linear(u, w2, bias=x)` | 285.2 | 1.001x | **bit-exact, no gain** |
| wo: `add(x, linear(...))` ships | 79.4 | — | — |
| wo: `linear(a, wo, bias=x, prog_cfg)` | 75.4 | 1.054x | 3.0e-03 rel, not bit-exact |
| wo: `linear(a, wo, bias=x)` no prog_cfg | 86.5 | 0.918x | bit-exact but SLOWER |

**w2's add is already entirely free** — hidden behind a 285 µs matmul. Only wo's is visible, worth 4.0 µs
(0.104 ms/frame), and only in the form that loses bit-exactness. Not taken. This is also the clearest
demonstration of why the isolated map overstates: the same op costs 28.9 µs alone and 0 µs in place.

### 6.28 — the DRAM-sharded matmul, re-opened with tuned blocking: §6.9's rejection HOLDS

The last lead from §6.27's overhead map. The norm emits a width-sharded activation and the shipped path
immediately unshards it (11.7 µs/layer, 0.30 ms/frame) so an ordinary matmul can take it.
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` is the one config that wants the sharded form
directly. §6.9 rejected it at 125.4 µs against 100.9 — **with untuned blocking**, and §6.25 had since
shown blocking is worth 1.2x on `wo`. So the rejection looked like it might be an artifact.

**It is not. Measured from `hs` to the tensor `nlp_create_qkv_heads_decode` consumes, so each path
carries everything it needs:**

| weight placement | in0_block_w | µs | vs shipped | numerics |
|---|---|---|---|---|
| **Path A — unshard + normal matmul (ships)** | — | **108.6** | **1.000x** | — |
| 8c DRAM-sharded | 1 | 303.7 | 0.357x | 3.0e-03 rel |
| 8c DRAM-sharded | **6** | **180.7** | **0.601x** | 3.0e-03 rel |
| 8c DRAM-sharded | 12 | 181.7 | 0.598x | 3.0e-03 rel |
| 8c L1-sharded | — | — | **Out of Memory** | — |
| 32c DRAM-sharded | — | — | TT_FATAL on the weight reshard | — |
| 32c L1-sharded | 3 | 138.4 | 0.784x | **1.4e+14 — silently WRONG** |

**Tuning `in0_block_w` was worth 1.68x** (303.7 → 180.7), so the premise was right — §6.9 did leave it
untuned. **The conclusion was still wrong: even fully tuned it is 1.66x SLOWER.** §6.9's axis argument
survives intact — width-sharding in0 splits the matmul's CONTRACTION dimension, so each core forms only a
partial sum and a cross-core reduce over the full 6144-wide output follows. Better blocking makes that
reduce cheaper; it does not remove it.

**Two incidental findings, both silent failure modes:**
- an **L1-resident** width-sharded weight is Out of Memory at 8 cores — 13.4 MB over 8 shards is 1.7 MB
  per core. This confirms §6.9's note that an L1 weight "would be capped anyway", with the actual error.
- the **32-core L1** variant BUILDS, RUNS, and returns **1.4e+14** — garbage, no exception. Another entry
  for the list of ttnn configurations that are silently wrong rather than loud.

**So the norm's reshard pair is now closed three ways** — sharded straight to a normal matmul (8.94 vs
5.32 ms, §6.9), weight-only sharding (not expressible, §6.9), and this config tuned (1.66x slower). The
11.7 µs unshard stays.

**Process note, since it took four probe iterations and two were my own fault.** Attempt 1 passed no
output `memory_config` → *"Output memory config must be sharded"*, and I wrote **"§6.9's rejection
holds"** on the strength of it. It had not been tested at all. Attempt 2 left the weight interleaved →
*"Input B memory layout must be WIDTH_SHARDED"*. Only attempt 3 measured anything. **A config that fails
to BUILD has told you nothing about whether it is fast** — and I recorded the opposite once already
today (§6.27's `to_memory_config` reading). Read the assertion before writing the conclusion.

### 6.29 — Block 2's overhead map: 40% reachable, and NONE of it is in the matmuls

§6.27's method applied to Block 2. Every op isolated at its real shape and memory config, floor =
bytes / 194 GB/s, ranked by overhead x how many times it runs. `_block` runs **21x** a frame (3 layers
x 7 Euler steps); the `_solve`/`_trunk` ops run 7x; two run once.

Block 2 is shaped nothing like Block 1. **All five matmuls are AT the roofline** -- wqkv measures 1.8 us
BELOW its floor, w3 7.7 us below (i.e. above 194 GB/s, consistent with the true ceiling being ~202),
w1 +2.9, wo +12.6, w2 +23.8. So there is no matmul work left, and yet **8.99 of 22.55 ms/frame is not
weight reads** -- 40%, against Block 1's 12%. Every recoverable microsecond is in a small op.

| op | x/frame | floor us | actual us | ovhd us | ms/frame | recoverable |
|---|---|---|---|---|---|---|
| `nlp_create_qkv_heads` (transpose_k) | 21 | 0.0 | 122.1 | 122.1 | 2.564 | **2.564 ms** |
| `semantic_code` | 1 | 0.0 | 1241.3 | 1241.3 | 1.241 | **1.241 ms** |
| CFG combine + Euler (2 slice 3 mul 2 add) | 7 | 0.0 | 148.2 | 148.2 | 1.038 | 1.038 ms |
| ffn norm x3 (shard+norm+DRAM) | 21 | 2.0 | 47.5 | 45.5 | 0.998 | 0.955 ms |
| reshape+permute+reshape unfold heads | 21 | 0.0 | 43.7 | 43.7 | 0.918 | 0.918 ms |
| `_trunk` 3 reshape + concat + reshape | 7 | 0.0 | 129.8 | 129.8 | 0.909 | 0.909 ms |
| matmul q(row-fold) @ kT | 21 | 0.0 | 30.5 | 30.5 | 0.641 | 0.641 ms |
| add residual (ffn) | 21 | 0.0 | 26.9 | 26.9 | 0.565 | 0.565 ms |
| add residual (attn) | 21 | 0.0 | 26.7 | 26.7 | 0.560 | 0.560 ms |
| multiply(g, w3) | 21 | 0.0 | 26.6 | 26.6 | 0.558 | 0.558 ms |
| **linear w2** 9216x3072 BFP8 | 21 | 155.1 | 178.9 | 23.8 | 3.757 | 0.500 ms |
| attn norm 2/3 rms_norm sharded | 21 | 0.0 | 20.0 | 20.0 | 0.419 | 0.419 ms |
| softmax numeric_stable | 21 | 0.0 | 19.2 | 19.2 | 0.403 | 0.403 ms |
| `_trunk` final norm x3 | 7 | 2.0 | 53.0 | 51.0 | 0.371 | 0.357 ms |
| attn norm 1/3 -> width-sharded | 21 | 1.0 | 17.5 | 16.5 | 0.367 | 0.346 ms |
| matmul a @ v | 21 | 0.0 | 15.6 | 15.6 | 0.328 | 0.328 ms |
| **linear wo** 4096x3072 BFP8 | 21 | 68.9 | 81.5 | 12.6 | 1.712 | 0.265 ms |
| `_trunk` linear acoustic_codebook_output | 7 | 1.1 | 33.0 | 31.9 | 0.231 | 0.224 ms |
| attn norm 3/3 -> DRAM | 21 | 1.0 | 10.5 | 9.5 | 0.220 | 0.199 ms |
| `_solve` concat([x,x]) + typecast | 7 | 0.0 | 25.1 | 25.1 | 0.176 | 0.176 ms |
| `_trunk` reshape + slice (36 wide) | 7 | 0.0 | 22.1 | 22.1 | 0.155 | 0.155 ms |
| `_solve` linear input_projection | 7 | 1.1 | 17.5 | 16.4 | 0.122 | 0.115 ms |
| `_solve` typecast v -> fp32 | 7 | 0.0 | 9.7 | 9.7 | 0.068 | 0.068 ms |
| **linear w1** 3072x9216 BFP8 +silu | 21 | 155.1 | 158.0 | 2.9 | 3.317 | 0.061 ms |
| `_solve` linear llm_projection | 1 | 51.7 | 65.2 | 13.5 | 0.065 | 0.014 ms |
| **linear wqkv** 3072x6144 BFP8 | 21 | 103.4 | 101.6 | **-1.8** | 2.134 | -0.037 ms |
| **linear w3** 3072x9216 BFP8 | 21 | 155.1 | 147.4 | **-7.7** | 3.095 | -0.161 ms |
| TOTAL (isolated, sums high) | | | | | 26.931 | 13.379 ms |

Shipped Block 2 frame end to end: **22.545 ms**. Weight-read floor **13.553 ms**. The isolated sum
(26.93) exceeds the real frame by 4.4 ms, which is the overlap the live graph gets and the reason the
per-op column is an upper bound -- same caveat as §6.27.

**The two biggest single items:**
- `nlp_create_qkv_heads` at 2.56 ms/frame confirms NOTES [flow-10]'s recorded ~97 us floor. §6.30 splits
  it and shows the kernel really is that slow. This is the upstream issue still unwritten.
- `semantic_code` at **1241 us for one call** is the largest untouched item in the model. It is a
  host->device upload, an fp32 matmul against the semantic head, a mask add, an argmax and a
  device->host download. Not yet swept.

### 6.30 — Block 2 sweep: five candidates, ONE ships, and one of the three refutals is of my own theory

**A hypothesis I had before measuring, which the split killed.** Block 2 folds the CFG batch into 6 rows
so each weight is read once (NOTES [flow-11]), but `nlp_create_qkv_heads` needs 4D [B,1,S,QW], so the
fold must be undone: [1,6,6144] -> [2,1,3,6144]. In TILE_LAYOUT the first pads to [1,32,6144] -- one tile
row -- and the second to [2,1,32,6144] -- two. Different physical sizes, so that "reshape" moves data. I
expected the repack to be most of the 122 us.

    SPLIT: the reshape alone 25.9 us; the op on a pre-reshaped tensor 96.6 us  (together 122.5)

So the repack is **a fifth** of it, not most, and the kernel genuinely costs ~97 us. The row fold is not
quietly paying its weight-read win back in padding.

| sweep | shipped | best alternative | verdict |
|---|---|---|---|
| 1. rms_norm, 49 calls/frame | 47.5 us (3 ops) | one-op interleaved **115.1** | **rejected** -- 2.4x slower AND 4.8e-03 off |
| 2. qkv head split | 122.0 us | manual 3 slice+3 reshape+3 permute, L1 | **isolated INCONCLUSIVE (§6.33); ships on the whole-block A/B** |
| 3. unfold heads | 43.2 us | permute-straight-from-av 24.9 | **rejected** -- 1.77x but **1.4e+00 WRONG** |
| 4. CFG combine + Euler | 160.0 us | weighted reduce **112.6** | 1.363x isolated, **then rejected on the block** |
| 5. `_trunk` sequence build | 124.5 us | hoist the reshapes **105.8** | **SHIPS** -- bit-exact |

Notes on the rejections, because each says something:
- **the one-op norm is much worse.** Three sharded ops (17.5 shard + 20.0 norm + 10.5 unshard) beat one
  interleaved `rms_norm` by 2.4x. Sharding is not overhead here, it is the fast path.
- **the hand-rolled split, CORRECTED, is 1.086x FASTER** -- and my first reading of it was invalid.
  I gave the shipped op `memory_config=_L1` and let my slices and permutes take the default, so they
  landed in DRAM. NOTES [flow-10] measured exactly that difference and found the L1 output worth
  **2.5 ms/frame**, not on the op (~7 us) but on the four downstream consumers that then keep q/k/v in
  L1. So the manual route was being credited for skipping work the shipped one does. Like-for-like:

      shipped: nlp_create_qkv_heads(memory_config=L1)   122.0 us   1.001x   L1/L1/L1  bit-exact
      manual, DEFAULT mc  (what the first pass timed)   122.4 us   0.998x   DR/DR/DR  bit-exact
      manual, memory_config=L1  (like-for-like)         112.4 us   1.086x   L1/L1/L1  bit-exact

  0.202 ms/frame isolated, bit-exact, in the right memory. NOTE THIS CONTRADICTS [flow-10]'s recorded
  158 us for a hand-rolled split, so one of the two measurements is of a different construction -- that
  is unreconciled. And it trades 1 op for 9, i.e. more host dispatch, so on this session's record it is
  a candidate and not a win until the whole block says so.
- **the fastest unfold is wrong.** Permuting straight from `av` skips the [B,32,3,HD] regroup, is 1.77x
  faster, and returns garbage (1.4e+00 relative). §6.25's lesson, again, caught by the numerics column.

**SWEEP 5 CORRECTED, and this is the substantive error of the pass.** The first reading measured
`concat([p0, p12])` against a `p12 = concat(p1s[i], p2)` built once OUTSIDE the timed loop and reported
1.461x, +0.279 ms/frame. That is not shippable: p1s[i] varies per step, so in the real graph that concat
does not vanish, it MOVES -- 7 new concats a frame to save 7 cheaper ones, and the op count goes UP.
Re-measured against only spellings that could really ship:

    p0  = linear(x_t, input_projection)   changes every STEP   -- and is ALREADY [B,1,3072], no-op reshape
    p1s = the time schedule               constant for the model's life -- hoist into _schedule
    p2  = linear(h, llm_projection)       changes once per FRAME       -- hoist to once per frame

    shipped: 3 reshape + concat(3) + reshape        115.9 us   1.010x
    hoist p1,p2 reshapes                            105.8 us   1.107x   +0.079 ms/frame  bit-exact
    + drop p0's reshape too                         105.7 us   1.108x   (p0's was already free)
    concat(2) vs hoisted p12 (NOT shippable)         81.2 us   1.442x

So the real win is **+0.079 ms/frame, not +0.279** -- the 1.461x was the unshippable hoist.

**AND THEN THE BLOCK REFUSED TO PAY FOR THE CFG FOLD.** A/B on the whole Block 2 frame, same session:

| build | ms/frame | codes differing from the fp32 reference (of 37) |
|---|---|---|
| baseline | 22.583 | 1 |
| **sequence hoist only** | **22.476** | **1** |
| both wins | 22.483 | **2** |

The sequence hoist takes the **entire** 0.107 ms and leaves the numerics alone. The CFG fold -- 1.543x in
isolation, 7 ops down to 5 -- contributes **nothing** on the whole block and flips a discrete code, since
its 1.6e-04 reassociation is enough to cross an FSQ quantization boundary. Reverted, helper deleted.

That is the **fourth** time this session an isolated win vanished on the whole unit (§6.18, §6.19, §6.27's
`to_memory_config` reading, now this). The rule has earned its keep: **for anything touching memory
config, layout, or op count on small tensors, the smallest honest unit of measurement is the whole
block.** Isolation is for choosing what to try, never for deciding what to ship.

**GATED AND SHIPPED.** The sequence hoist is bit-exact end to end, verified as a paired A/B in one
session with the tree stashed and restored between halves:

| gate | before | after |
|---|---|---|
| flow: velocity PCC | 0.99998480, maxabs 2.833e-02 | identical |
| flow: semantic code | exact match `[262, 3346]` | identical |
| flow: full frame | 2 of 74 codes differ | identical |
| codes: semantic / acoustic | 1, 97/288 (33.7%) | identical |
| codes: the whole per-frame code table | | **byte-identical, every frame** |
| Block 2 alone | 22.583 ms/frame | **22.476** |

Byte-identical codes is a stronger statement than any PCC: if the integer codes do not move, the
waveform does not move, so WER cannot move. `--gate decode` was also run (pooled mean 0.92%, p90 1.28%,
max 3.05%, min PCC 0.999040, 15 prompts x 22 frames) and matches §6.13's recorded level -- expected,
since this change does not touch Block 1, and worth having as the check that it does not.

### 6.31 — the two wins the map actually produced: qkv head split and semantic_code

§6.30's sweep shipped one small thing. Following the map's two biggest lines properly produced the two
real wins, and both are BYTE-IDENTICAL.

**Block 2, cumulative, all three changes gated as byte-identical codes:**

| change | Block 2 ms/frame | commit |
|---|---|---|
| start of the sweep | 22.583 | |
| `_trunk` reshape hoist (§6.30) | 22.476 | `41194fe16f5` |
| hand-rolled qkv head split | **21.142** | `55d4148eb4b` |
| `semantic_code` mask+argmax to host | **20.801** | this one |
| **total** | **-1.782 ms/frame, 7.9%** | |

**1. The qkv head split, and the measurement lesson in it.** Both of my earlier readings compared the
manual route against the shipped op WITHOUT matching output memory configs -- the shipped op gets
`memory_config=_L1`, my slices and permutes took the default and landed in DRAM. NOTES [flow-10]
measured that exact difference at 2.5 ms/frame downstream. Corrected:

    shipped: nlp_create_qkv_heads(memory_config=L1)   122.0 us   1.001x   L1/L1/L1  bit-exact
    manual, DEFAULT mc (what I first timed)           122.4 us   0.998x   DR/DR/DR  bit-exact
    manual, memory_config=L1                          112.4 us   1.086x   L1/L1/L1  bit-exact

And then the whole block paid **six times** the isolated figure -- interleaved A/B, 200 frames each,
three rounds, the method swapped on the class so nothing else differs:

    round      shipped     manual     ratio
        0      22.369     21.131    1.0586x
        1      22.419     21.180    1.0585x
        2      22.336     21.116    1.0578x
     mean      22.375     21.142    1.0583x     = 1.233 ms/frame

**This is the FIRST time this session the whole block paid MORE than isolation**, after four cases of it
paying less (§6.18, §6.19, §6.27, §6.30). So the lesson is NOT "isolated numbers are optimistic". It is
that op granularity cannot see overlap or memory residency AT ALL, in either direction. Likely
mechanism: the fused op needs a 4D [B,1,3,QW] input and that reshape repacks tile padding
([1,32,QW] -> [2,1,32,QW], 25.9 us), while slicing the folded [1,B*3,QW] tensor never repacks.
Measured, not established.

UNRECONCILED: [flow-10] records a hand-rolled split at 158 us where this measures 112.4. Some difference
of construction that I did not find. Logged rather than written off -- the change ships on the
whole-block A/B and byte-identical codes, neither of which depends on settling it.

**2. `semantic_code`: the argmax was 40% of it and belongs on the host.** The split, corroborated two
ways -- the pieces sum to 1162.9 us of the whole call's 1229.5, and the composite `linear+mask+argmax`
measured 1086.8 against the pieces' 1088.4, agreeing to 0.1%:

    linear fp32 against a (3072, 8320) head   562.0 us   45.7%   102 MB at 182 GB/s -- AT the roofline
    argmax over 8320 values                   490.1 us   39.9%   33 KB at 0.07 GB/s -- ALL overhead
    from_torch H->D                            53.0 us    4.3%
    add semantic_mask                          36.3 us    3.0%
    to_torch D->H                              21.4 us    1.7%

The matmul is at the roofline and is not the problem. The argmax is 490 us to reduce 33 KB. And it costs
nothing to move, because this call ALREADY ended in a device->host copy -- doing the reduce on the host
does not add a round trip, it only makes that copy 8320 fp32 values instead of 1, i.e. 33 KB of PCIe.
Scored on REAL Block 1 hidden states from the fixture prompts:

    A shipped: linear fp32 + device mask + argmax   1213.5 us   1.000x   0 of 8 ids changed
    B host argmax, device mask                       893.5 us   1.386x   0 of 8
    C host argmax AND host mask   <- SHIPPED         860.3 us   1.439x   0 of 8
    D bf16 head, device argmax                       946.9 us   1.308x   0 of 8
    E bf16 head + host argmax + host mask            595.7 us   2.079x   0 of 8

C is exactly the same fp32 arithmetic -- same values, same order, the add and the reduce just happen on
a different processor -- so it carries no numerical risk. Predicted 0.353 ms/frame, delivered 0.341.

**E IS FASTER STILL AND IS DELIBERATELY NOT SHIPPED.** 2.079x, another 0.265 ms/frame, and 0 of 8 real
prompts moved. But this is an ARGMAX over a vocabulary: what decides a flip is the top-2 GAP, not a norm,
and 8 single frames is a thin sample for a discrete decision. The semantic token is the highest-stakes
integer in the model -- it feeds Block 1's next input embedding, so ONE flip redirects the entire
remaining generation, unlike an acoustic code which affects one frame. Needs a broad real-prompt gate.

**AND MY FIRST VERSION OF THIS PROBE WAS WORTHLESS.** Round 1 scored the bf16 candidate on 64 RANDOM
gaussian draws and reported 0 changed ids. Trap #12 already records random embeddings reading PCC 0.892
where real prompts gave 0.9994, and an argmax is precisely where that bites. Round 2 pulls real hidden
states out of Block 1. **ALWAYS GATE ON REAL PROMPTS** -- written in this file, and I still did it wrong
once today.

**Gate for both changes** (each run separately, against the immediately preceding commit):

| | qkv split | semantic host-argmax |
|---|---|---|
| flow: velocity PCC | 0.99998480, unchanged | 0.99998480, unchanged |
| flow: semantic code | exact `[262, 3346]` | exact `[262, 3346]` |
| flow: full frame | 2 of 74, unchanged | 2 of 74, unchanged |
| codes: semantic / acoustic | 1, 97/288, unchanged | 1, 97/288, unchanged |
| codes: per-frame code table | **byte-identical** | **byte-identical** |
| test_flow_pcc | 13 passed | 13 passed |

### 6.32 — the end-to-end gate for §6.31, and a stale-file error of mine that nearly went in the record

**Long-form RTF 0.61-0.65 -> 0.57-0.64, ~50.9 -> ~48.1 ms/frame.** Full 15-prompt quality set,
`resultsb2sweep.json`:

| case | frames | RTF | ms/frame |
|---|---|---|---|
| 2 | 461 | 0.644 | 51.5 |
| 3 | 487 | 0.573 | 45.8 |
| 10 | 220 | 0.589 | 47.1 |

Case 0's 1.855 is the compile case and is excluded, per §6.21's convention: quote long-form (>=100
frames) only.

**Quality, unchanged and at the bar:**

    long-form WER          0.00% over 298 words = 0 WRONG      (reference scores 0.0%)
    short                  0.00% over  42 words
    [END_AUDIO] natural    15/15 cases
    voice identity         PASS -- same-voice pair most similar, F0 spread 103-201 Hz

**AND THE STRONGEST NUMERICAL GATE THIS PROJECT HAS RUN.** A paired A/B against the pre-sweep commit
`6dc2dc2d460`, generating the three natural-text cases free-running on each build:

| case | pre frames | post frames | pre RTF | post RTF |
|---|---|---|---|---|
| 0 | 66 | 66 | 1.577 | 1.652 |
| 2 | 461 | **461** | 0.637 | 0.620 |
| 3 | 487 | **487** | 0.592 | 0.572 |

**Frame counts identical on every case.** That is much stronger than the 8-12 frame `--gate codes`
comparison: free-running generation is autoregressive, so ANY divergence compounds and moves the
termination frame. Reproducing 461 and 487 exactly means the whole generation is bit-identical, which is
what "byte-identical codes" was claiming on 8 frames. Use this as the gate for anything claiming
exactness in future -- it is cheap (two cases, ~90 s) and it cannot be fooled by a short window.

**THE ERROR, because it is a new failure mode for the list.** I first scored the run with

    score_quality_set.py generated/results_b2sweep.json || score_quality_set.py generated/results.json

`generate_quality_set.py` writes `results{tag}.json` with **no underscore**, so the first path did not
exist and the `||` silently scored `results.json` -- an unrelated file from 08-04. I reported "1 wrong
word of 274" from it. The tell was already printed in my own output: the scorer said case 0 = 71 frames,
case 2 = 449, case 3 = 495, while the generation log said 66, 461, 487. Correctly scored, the run gives
**0 wrong of 298**, matching the recorded baseline.

Two lessons, both cheap:
- **never write a `||` fallback into a gate.** A gate that silently substitutes a different input is
  worse than one that fails, because it returns a plausible number.
- **the results file carries the frame counts; check them against the generation log.** That single
  cross-check catches a stale file instantly, and it was sitting in the output I had already printed.

`generated/` holds 39 `results*.json` files from this project's sweeps. Any of them will score cleanly
and none of them will tell you it is the wrong one.

### 6.33 — CORRECTION to §6.31: the hand-rolled split is NOT faster in isolation, and the fused op is not flat

Two claims in §6.31 are wrong. The shipped change is still right, for a different reason.

**WRONG CLAIM 1: "manual 112.4 us vs fused 122.0 us, 1.086x isolated".** It does not reproduce. Settled
with 500 reps x 6 rounds, alternating A/B/B/A so neither op is always second, on both a real matmul
output and a synthetic tensor:

| source | | mean us | min | max | spread |
|---|---|---|---|---|---|
| real matmul output | fused | **122.3** | 122.2 | 122.4 | **0.2** |
| | manual | 125.7 | 122.0 | 127.8 | 5.8 |
| synthetic from_torch | fused | **122.1** | 122.0 | 122.2 | **0.2** |
| | manual | 133.4 | 118.3 | 146.0 | 27.6 |

**In isolation the fused op is at least as fast, and it is far more stable** -- 0.2 us spread against
5.8-27.6. The 112.4 was a low sample from a wide distribution, and a 10 us "win" was read off it. Nine
ops have variable dispatch cost; one op is deterministic. That variance is itself worth knowing.

**WRONG CLAIM 2: "the fused op is fixed cost".** NOTES [flow-10] measured it flat at ~97 us from S=3 to
S=32 and I generalised that to "fixed cost". It is flat in **S**, not in **B**:

| B | rows | qkv KB | fused us | manual us | manual/fused | winner |
|---|---|---|---|---|---|---|
| 2 | **6** | 72 | 122.9 | 127.1 | 1.034x | fused |
| 4 | 12 | 144 | 166.0 | 130.9 | 0.788x | manual |
| 8 | 24 | 288 | 200.0 | 165.8 | 0.829x | manual |
| 16 | 48 | 576 | 284.1 | 192.5 | 0.677x | manual |
| 32 | 96 | 1152 | 442.1 | 365.6 | 0.827x | manual |
| 64 | 192 | 2304 | 758.2 | 714.5 | 0.942x | manual |
| 128 | 384 | 4608 | **TT_FATAL** | -- | -- | fused fails |

6x the batch is 6x the data and 6.2x the time. Growing S and growing B are different axes and I conflated
them.

**A prediction that WAS right, directionally.** The manual route makes THREE passes over memory (slice
writes q/k/v, reshape rewrites, permute rewrites) where the fused op makes ONE, so its advantage should
erode as bytes start to dominate launch cost. It does: 0.788x at B=4 drifting to 0.942x at B=64. No
crossover appears before the fused op TT_FATALs at B=128, so the erosion never completes -- but the
mechanism is visible in the trend. Block 2 is structurally 3 tokens, so B is the only growth axis, and
B=64 is 32x anything planned.

**WHAT ACTUALLY STANDS, and it is a better result than the one I recorded.** The whole-block win is
unaffected and triple-measured:

    interleaved A/B, 3 rounds x 200 frames   22.375 -> 21.142 ms   1.0586 / 1.0585 / 1.0578
                                             spread 0.0008 on a 0.058 effect
    incremental, separate script             22.476 -> 21.142 ms
    end-to-end RTF, cases 2 and 3            0.637 -> 0.620, 0.592 -> 0.572 (all three changes)

**So the op is NOT faster, yet the block is 1.233 ms/frame faster.** The entire win is a system effect
around the op, not the op. That makes the mechanism MORE open than §6.31 claimed, not less. Candidates,
none verified: the fused path's 4-D reshape builds a 768 KB intermediate (384 KB -> 768 KB, since
[1,6,6144] pads to one 32-row slab and [2,1,3,6144] pads to two) that must be allocated, written and
re-read, which my column slices never create; the two paths' L1 layouts differ for the four downstream
consumers; the fused kernel may not overlap with its neighbours the way nine small ops do.

**The methodological point, which is the fourth version of the same lesson today.** §6.31 said the
whole block paid "six times" the isolated figure. The truth is worse: **the isolated figure had the
wrong SIGN.** At 6 rows the effect being measured (~10 us) is smaller than the run-to-run spread of the
thing measuring it (5.8-27.6 us), so the isolated comparison could not have resolved it in either
direction. Always report the spread next to the mean; a single number with no spread is not a
measurement. The decision was right only because it was taken on the whole block.

### 6.34 — project THEN duplicate, instead of duplicate THEN project: rejected

A natural question, so it is recorded to stop it being re-asked. `_solve` builds p0 as

    x2 = concat([x, x], dim=0)                  [B,1,36] -> [2B,1,36]
    p0 = linear(typecast(x2), input_projection) [2B,1,36] -> [2B,1,3072]

**and the premise is correct: `x` is byte-identical in both CFG halves** -- guidance differs in the
CONDITIONING (p2 is llm_hidden over zeros), never in the state -- so that matmul runs twice on the same
input. Projecting once and duplicating the result is mathematically identical for any B.

**It is still slower.** Isolated, 400 reps x 4 alternating rounds:

| variant | mean us | spread | vs shipped | numerics |
|---|---|---|---|---|
| **shipped: concat -> typecast -> linear** | **43.9** | 3.4 | 1.000x | -- |
| linear -> concat (project first) | 56.0 | 4.3 | **0.785x** | bit-exact |
| linear -> `ttnn.repeat` | 78.8 | 2.0 | 0.557x | bit-exact |
| typecast -> linear -> concat | 60.7 | 3.6 | 0.723x | bit-exact |

Whole Block 2 frame, interleaved A/B, 200 frames x 4 rounds: shipped 21.232 ms (spread 0.084),
project-first 21.279 (spread 0.057). Gap -0.047 ms, under the spread, so INCONCLUSIVE on the block and
clearly worse isolated. Codes identical either way -- the rewrite is correct, just unprofitable.

**WHY, and it generalises.** The redundant matmul is nearly free: input_projection measures 17.5 us
against a **1.1 us byte floor**, so the duplicated WORK is about 1 us and the rest is launch cost, which
both forms pay equally (3 ops either way). Moving the duplication downstream makes it operate on a 48x
wider tensor:

    duplicate the INPUT    [1,1,36]   tile-padded 32x64    =  16 KB
    duplicate the OUTPUT   [1,1,3072] tile-padded 32x3072  = 384 KB

That cost **+12 us**, an order of magnitude more than the ~1 us of redundant matmul it removed.

**The rule: redundant work is only worth removing if it is expensive relative to what removing it
costs.** At these tensor sizes almost nothing is expensive except launching ops, so a restructuring that
keeps the op count the same and moves data to a WIDER point in the graph will usually lose. Same shape of
reasoning as §6.24's fusion rule -- fusion pays only when an operand is too narrow to earn its launch.

Incidental: **`ttnn.repeat` is 1.8x worse than `ttnn.concat`** for the same duplication.

### 6.35 — what the CFG duplication costs (1.8%), and the batching headroom that falls out of it

The question: `x` is duplicated so the velocity net runs twice per Euler step, once with the llm
conditioning and once with zeros. That looks like paying 2x for guidance. What does it actually cost?

**First, it is semantically required.** The two passes need two independent attention contexts. Merging
them into one 4-token sequence `[p0, p1, p2_cond, p2_uncond]` is NOT equivalent -- attention is
bidirectional over the 3 tokens, so `p0` would then see both conditionings at once. And the duplication
is only redundant at the INPUT projection: after the first attention layer the two copies have attended
to different p2 and genuinely diverge. That is why §6.34's project-first rewrite was valid but could only
ever save that one projection.

**Second, it costs 1.8%.** One `_block` call, 3 layers, 300 reps x 4 alternating rounds:

| | rows | tile rows used | us/call | spread |
|---|---|---|---|---|
| no guidance (hypothetical) | 3 | 1 of 32 | 828.2 | 0.3 |
| **with guidance (ships)** | **6** | **1 of 32** | **843.6** | 0.5 |

**+15.3 us per call, +1.8%, i.e. 0.322 ms/frame** over 21 calls, out of Block 2's 20.8 ms. The naive
"CFG doubles the work" intuition simply does not apply: 3 rows and 6 rows both pad to ONE 32-row tile,
and the row fold (NOTES [flow-13]) reads each weight once regardless. **So do not chase CFG elimination
-- the entire prize is 0.322 ms and it would change the audio.**

**Third, the same fact makes a real redundancy free.** `_cfg_input` puts llm_hidden over ZEROS and
`llm_projection` has no bias, so **p2's unconditional half is exactly zero** (verified: max |value| =
0.000000). We do run a 3072x3072 matmul to produce a known-zero result. Exploiting it saves nothing:

    weight [3072,3072] BFP8 = 9.6 MB, floor 51.7 us
    [2,3072] @ W (ships)       65.1 us   spread 0.1
    [1,3072] @ W (cond only)   65.1 us   spread 0.1     delta +0.0

M=1 and M=2 both pad to one tile row, so the weight is read once either way.

**AND HERE IS THE PART WORTH ACTING ON.** If 6 rows is as cheap as 3, how far does that go? A tile is 32
rows and the shipped path uses 6:

| utterances | B | rows | tiles | us/`_block` | **us per utterance** | throughput |
|---|---|---|---|---|---|---|
| 1 | 2 | 6 | 1 | 844.9 | 844.9 | 1.0x |
| 2 | 4 | 12 | 1 | 884.8 | **442.4** | **1.9x** |
| 4 | 8 | 24 | 1 | 995.1 | **248.8** | **3.4x** |
| 8 | 16 | 48 | 2 | TT_FATAL | -- | -- |

**Four utterances cost 1.18x the time of one -- 3.4x the throughput.** This is the "31 unused tile rows"
headroom, measured rather than asserted, and it is real.

**The 32-row ceiling is OUR OWN CONSTANT, not the hardware.** `_NORM_SHARD`'s shard shape is hardcoded
`(32, 96)`, and at 48 rows it raises `!shard_grid_fit_error.has_value()`. `wqkv` at 48 rows is fine and
returns `(1, 48, 6144)`. So going past one tile needs a row-count-aware norm shard (cache one config per
batch size, as `_schedule` already does for the timestep tokens) -- not a new kernel.

Note this is THROUGHPUT, not latency: RTF per utterance does not improve, but a server doing 4 streams
would do them in 1.18x the time of one. Whether that is useful depends on the deployment, which is not
our call -- but it is now a measured option rather than a guess.

**A probe bug of mine, recorded.** The first version of the p2 comparison left `gen._up(...)` INSIDE the
benched lambda, so it timed a host->device upload every iteration and reported the SMALLER matmul as
2.8x slower (185.2 vs 65.1 us). Hoist every operand out of the timed callable; if a "smaller" thing
measures slower, suspect the harness before the hardware.

### 6.36 — Block 2's overhead map REGENERATED, with source line numbers (supersedes §6.29)

§6.29 is stale: three of the things it measured have since changed -- `nlp_create_qkv_heads` (its largest
line, 2.564 ms/frame) is now a hand-rolled 9-op split, `_trunk`'s three reshapes are hoisted, and
`semantic_code`'s mask and argmax moved to the host. **Rank off THIS table, not §6.29.** Line numbers are
into `tt/ttnn_voxtral_flow.py` at commit `f1d985a37fe`. 300 reps x 3 rounds; `_block` ops run 21x a
frame, `_solve`/`_trunk` ops 7x, two once.

| line | op | ×/frame | floor µs | actual | spread | ovhd | ms/frame | recoverable |
|---|---|---|---|---|---|---|---|---|
| `157` | ffn norm, all 3 ops | 21 | 2.0 | 56.7 | 8.7 | 54.7 | 1.191 | 1.148 |
| `267` | CFG combine: 2 multiply + 1 add | 7 | 0.0 | 122.1 | 14.6 | 122.1 | 0.855 | 0.855 |
| `142` | reshape+permute v (0,2,1,3) | 21 | 0.0 | 37.6 | 0.7 | 37.6 | 0.790 | 0.790 |
| `176` | reshape seq -> [1,6,3072] | 7 | 0.0 | 106.0 | 0.2 | 106.0 | 0.742 | 0.742 |
| `174` | concat([p0,p1,p2], dim=1) | 7 | 0.0 | 101.0 | 0.1 | 101.0 | 0.707 | 0.707 |
| `139` | reshape+permute k (0,2,3,1)  pre-transposed | 21 | 0.0 | 33.6 | 6.0 | 33.6 | 0.705 | 0.705 |
| `146` | matmul q(row-fold) @ kT -> scores | 21 | 0.0 | 33.2 | 8.2 | 33.2 | 0.696 | 0.696 |
| `162` | add residual (ffn) | 21 | 0.0 | 31.1 | 8.2 | 31.1 | 0.654 | 0.654 |
| `160` | multiply(g, w3) | 21 | 0.0 | 30.7 | 1.1 | 30.7 | 0.645 | 0.645 |
| `154` | add residual (attn) | 21 | 0.0 | 30.6 | 1.5 | 30.6 | 0.643 | 0.643 |
| `153` | permute+reshape -> folded rows | 21 | 0.0 | 28.4 | 1.3 | 28.4 | 0.597 | 0.597 |
| `162` | linear w2     9216x3072 BFP8 | 21 | 155.1 | 179.0 | 0.3 | 24.0 | 3.760 | 0.504 |
| `269` | Euler update: multiply + add | 7 | 0.0 | 70.8 | 6.6 | 70.8 | 0.496 | 0.496 |
| `137` | reshape q -> [2,3,32,128] | 21 | 0.0 | 22.6 | 0.3 | 22.6 | 0.474 | 0.474 |
| `125` | rms_norm sharded                 (attn norm 2/3) | 21 | 0.0 | 21.6 | 1.0 | 21.6 | 0.453 | 0.453 |
| `149` | softmax numeric_stable | 21 | 0.0 | 19.4 | 0.1 | 19.4 | 0.407 | 0.407 |
| `179` | _trunk final norm, all 3 ops | 7 | 2.0 | 55.7 | 4.3 | 53.7 | 0.390 | 0.376 |
| `125` | to_memory_config -> _NORM_SHARD  (attn norm 1/3) | 21 | 1.0 | 18.7 | 1.5 | 17.7 | 0.394 | 0.372 |
| `242` | semantic_code: upload + linear + HOST argmax | 1 | 527.0 | 895.8 | 10.6 | 368.8 | 0.896 | 0.369 |
| `150` | matmul a @ v | 21 | 0.0 | 17.3 | 2.3 | 17.3 | 0.363 | 0.363 |
| `152` | reshape av -> [2,32,3,128] | 21 | 0.0 | 16.6 | 1.3 | 16.6 | 0.348 | 0.348 |
| `136` | slice q  cols 0..4095    -> L1 | 21 | 0.0 | 14.3 | 0.3 | 14.3 | 0.301 | 0.301 |
| `137` | permute  q (0,2,1,3) | 21 | 0.0 | 14.2 | 2.9 | 14.2 | 0.298 | 0.298 |
| `136` | slice k  cols 4096..5119 -> L1 | 21 | 0.0 | 13.8 | 0.4 | 13.8 | 0.290 | 0.290 |
| `136` | slice v  cols 5120..6143 -> L1 | 21 | 0.0 | 13.7 | 0.0 | 13.7 | 0.287 | 0.287 |
| `154` | linear wo     4096x3072 BFP8 | 21 | 68.9 | 81.5 | 0.1 | 12.6 | 1.711 | 0.264 |
| `181` | linear acoustic_codebook_output  3072x36 | 7 | 1.1 | 32.9 | 0.3 | 31.8 | 0.230 | 0.223 |
| `128` | to_memory_config -> DRAM         (attn norm 3/3) | 21 | 1.0 | 11.6 | 0.1 | 10.5 | 0.243 | 0.221 |
| `262` | typecast x2 -> bf16 | 7 | 0.0 | 26.4 | 0.2 | 26.4 | 0.185 | 0.185 |
| `184` | slice -> [2,1,36] | 7 | 0.0 | 24.2 | 0.3 | 24.2 | 0.170 | 0.170 |
| `262` | linear input_projection   36x3072 | 7 | 1.1 | 17.5 | 0.7 | 16.4 | 0.123 | 0.115 |
| `261` | concat([x,x], dim=0) | 7 | 0.0 | 15.3 | 0.5 | 15.3 | 0.107 | 0.107 |
| `266` | slice v_unc | 7 | 0.0 | 14.0 | 2.1 | 14.0 | 0.098 | 0.098 |
| `265` | slice v_cond | 7 | 0.0 | 13.0 | 0.2 | 13.0 | 0.091 | 0.091 |
| `264` | typecast v -> fp32 | 7 | 0.0 | 11.3 | 0.9 | 11.3 | 0.079 | 0.079 |
| `183` | reshape out -> [2,3,36] | 7 | 0.0 | 10.8 | 0.1 | 10.8 | 0.076 | 0.076 |
| `158` | linear w1     3072x9216 BFP8 +silu | 21 | 155.1 | 157.8 | 0.3 | 2.7 | 3.314 | 0.058 |
| `255` | linear llm_projection  3072x3072  (once/frame) | 1 | 51.7 | 65.1 | 0.0 | 13.4 | 0.065 | 0.013 |
| `134` | linear wqkv   3072x6144 BFP8 | 21 | 103.4 | 101.8 | 0.1 | -1.6 | 2.137 | -0.033 |
| `160` | linear w3     3072x9216 BFP8 | 21 | 155.1 | 147.5 | 0.5 | -7.5 | 3.098 | -0.158 |

**Frame 21.243 ms. Weight-read floor 14.080 ms. Reachable ceiling 7.163 ms (34%.)** The floor now
includes `semantic_code`'s fp32 head (102 MB, 527 us), which §6.29 wrongly counted as zero -- that is why
the floor moved 13.553 -> 14.080.

**FOUR THINGS TO READ CAREFULLY, because the table looks more actionable than it is.**

1. **Isolated sums to 29.1 ms against a real 21.2 ms frame.** ~8 ms is overlapped in the live graph. The
   per-op ranking is usable; the total is an upper bound; "recoverable" means *if the op vanished and
   nothing else changed*, which §6.33 proved is not how this behaves.

2. **The hand-rolled head split sums HIGHER than what it replaced.** Lines 136/137/139/142 total ~150 us
   (41.8 in slices, ~108 in reshape/permutes) against the 122 us fused op -- and it is still 1.233
   ms/frame FASTER on the whole block. That contradiction is the single best argument against treating
   this table as a to-do list.

3. **Six of the top ten are already closed.** 157 (one-op interleaved norm: 2.4x SLOWER, §6.30), 267
   (weighted-reduce fold: 1.543x isolated, ZERO whole-block, flips an FSQ boundary, §6.30), 139/142/153
   (swept as part of the split and the unfold, §6.30), 125 (norm grid swept, [flow-07]).

4. **Read the SPREAD column before believing any row.** The residual adds show 8.2 us of spread on a 31
   us overhead; the CFG combine 14.6 on 122. Enough to rank, not enough to decide. §6.33 is the case
   where an effect smaller than its own measurement spread was reported with the wrong sign.

**The largest genuinely untouched item is the sequence build: lines 174 + 176, 1.449 ms/frame combined.**
The three reshapes were hoisted (§6.30/[flow-19]) but the `concat` of three [2,1,3072] tensors (101.0 us)
and the final reshape to [1,6,3072] (106.0 us) remain. After that: the two residual adds (1.297
combined), `multiply(g, w3)` (0.645), the scores matmul (0.696).

**And note what is NOT here.** wqkv measures 1.6 us and w3 7.5 us BELOW their 194 GB/s floors, which is
how we know the true ceiling is ~202. All five weight matmuls are at the roofline; there is no matmul
work left in Block 2.

### 6.37 — sdpa for Block 2's attention: 5.9x on the op, +0.816 ms/frame, and it COSTS A WORD. Rejected.

The first change this session that a gate caught at the WER level, so it is worth reading as a template
for how far to take a promising number.

**The idea.** Block 2's attention is bidirectional and unmasked over 3 tokens -- exactly what non-decode
`scaled_dot_product_attention` is for. The shipped path is four ops (§6.36 lines 146/149/150/152):
`matmul q@kT` 33.2 us + `softmax` 19.4 + `matmul a@v` 17.3 + `reshape` 16.6 = 86.5 us. sdpa also emits
`[B,32,3,HD]` directly, so the reshape disappears too. It handles GQA 32/8 natively, which makes the row
fold ([flow-11]) and the `REP` constant unnecessary.

**`scale=1.0` is mandatory.** SCALE is folded into wqkv's q rows, so leaving sdpa's scale at its default
applies 1/sqrt(d) a SECOND time: 3.8e-01 relative error against the shipped path, versus 2.2e-02 with
scale=1.0. Anyone retrying this must pass it.

**The speed is real, and confirmed on the whole block.** 6 rounds x 200 frames, interleaved both
directions:

| variant | ms/frame | spread | vs shipped | delta | codes vs shipped |
|---|---|---|---|---|---|
| A shipped | 21.238 | 0.179 | -- | -- | -- |
| C `add_`/`multiply_` in place + L1 | 21.237 | 0.082 | 1.0000x | **+0.001** | 0 of 37 |
| **D sdpa only** | **20.422** | 0.047 | **1.0399x** | **+0.816** | 0 of 37 |
| E sdpa + in place | 20.478 | 0.071 | 1.0371x | +0.760 | 0 of 37 |

**But it is measurably less accurate, gated against fp64 TRUTH rather than against the shipped path**
(§6.25's rule), with the attention recomputed on the host in float64 from the device's own q/k/v:

    shipped: 4 ops     max abs err 2.746e-03   rel 3.20e-03   PCC 0.9999964
    sdpa(scale=1.0)    max abs err 1.780e-02   rel 2.07e-02   PCC 0.9998503

6.48x the error. That is NOT a compare-against-the-default artifact; sdpa really is less accurate here.

**AND THE END-TO-END GATE FOUND THE COST.** Frame counts moved -- case 0 66 -> 64, case 2 461 -> 445
(1.3 s less audio), case 3 487 -> 488 -- so generation diverged, which the frame-count A/B (§6.32) is
precisely designed to detect. Scored like-for-like on the same three cases:

| | case 0 | case 2 | case 3 | long-form WER |
|---|---|---|---|---|
| **shipped** | 0.0% | 0.0% | 0.0% | **0 wrong of 274** |
| **sdpa** | 0.0% | **0.8%** | 0.0% | **1 wrong of 274** |

Generation is deterministic, so the dropped word is caused by sdpa. **REVERTED.** +0.816 ms/frame is
1.7% of a frame; the project's headline quality claim is zero WER errors, and §6.16 already set the
precedent by handing back 2.5 ms/frame rather than lose accuracy.

**ONE HONEST QUALIFICATION.** One sample cannot separate "sdpa is systematically worse" from "sdpa is
merely DIFFERENT and this draw was unlucky" -- a different-but-equally-valid generation can drop a word
by chance. Settling that needs several seeds across all 15 cases. If 0.8 ms/frame ever matters enough,
that is the experiment; the accuracy loss against truth (6.48x) means the prior should be "worse", not
"different".

**AND THE IN-PLACE CANDIDATES ARE DEAD, after three contradictory readings of my own.** `ttnn.add_` and
`ttnn.multiply_` are bit-exact and looked like a win:
  1. first measurement: BOTH SLOWER (37.1 vs 27.0) -- invalid, I wrapped them in `ttnn.clone` so the
     input survived repeated iterations, and the clone costs more than the allocation it saves. In the
     real `_block` the operand is dead immediately, so no clone is needed.
  2. without the clone: 1.143x and 1.196x isolated, and +0.104 ms on the block over 3 rounds.
  3. over 6 rounds: **+0.001 ms.** The +0.104 was inside the 0.179 ms spread. And variant E shows
     in-place is marginally NEGATIVE when combined with sdpa.
Three measurements, three different answers, and only the last had enough rounds to beat its own noise.
`ttnn.swiglu` is also not applicable -- it `TT_THROW`s on a concatenated pair, and using it would require
fusing w1/w3, which §6.24 measured at 4x slower.

**Residual-as-bias is NOT EXPRESSIBLE here**, unlike in Block 1: ttnn's `bias` is added per output
column, and the residual differs per row. Recorded so it is not attempted.

### 6.38 — the seed control (frame counts are noise), w2-BFP8 over three seeds, and a repo-wide reuse survey

Three things, and the first invalidates evidence I used twice.

**1. FRAME COUNTS ARE SEED NOISE.** §6.37 rejected sdpa partly because case 2's frame count moved
461 → 445, and the w2-BFP8 test showed case 3 moving 487 → 523. I called both "divergence". The control
— same shipped code, same prompts, **only the `x_0` noise draw changed**:

| build / seed | case 0 | case 2 | case 3 |
|---|---|---|---|
| shipped, seed 0 | 66 | 461 | 487 |
| shipped, seed 1 | 76 | 470 | 508 |
| shipped, seed 2 | 70 | 438 | 449 |
| **seed-only swing** | **10** | **32** | **59** |

Both movements I cited sit INSIDE the seed-only swing. The correct reading is asymmetric:

    frame counts IDENTICAL  ->  the change is bit-exact. Strong, and it has caught nothing false (§6.32).
    frame counts MOVED      ->  the change is not bit-exact. NOTHING MORE.

**WER, by contrast, is stable across seeds: 0 wrong of 274 at seeds 0, 1 and 2.** So the clean baseline
is reproducible rather than lucky, and a word appearing is real signal — but it must be judged over
several seeds, never one.

**2. w2 IN BFP8, RE-GATED ON THREE SEEDS.** §6.16 rejected it on Block 1 teacher-forced metrics alone —
no WER, no frame counts, no listening pass, none of which existed then. Re-measured:

    block 1 decode step   w2 bf16  23.151 ms (spread 0.004)
                          w2 BFP8  20.507 ms (spread 0.002)     -> 2.644 ms/step, 5.5% of a frame

    decode gate, 15 prompts    bf16  mean 0.92%  p90 1.28%  max 3.05%  min PCC 0.999040
                               BFP8  mean 1.16%  p90 1.68%  max 4.41%  min PCC 0.997815

    long-form WER      seed 0     seed 1     seed 2
      shipped          0 wrong    0 wrong    0 wrong      (0 of 822 words)
      w2 BFP8          1 wrong    0 wrong    0 wrong      (1 of 822 words)

**So the upstream degradation is real and reproducible; the output evidence is 1 word in 822 against 0 in
822, which no test would call significant.** §6.16's pricing stands (w2 = 77% of the accuracy cost for
15% of the speed), and the accumulated Block 1 error is unambiguous. But the claim "it costs a word" is
NOT established — it rests on one occurrence. **Left at bf16, and this is a genuine open decision, not a
closed one.** To settle it: all 15 cases x 3+ seeds on both arms. At 2.644 ms/step it is the largest
single win left anywhere in this model, so that experiment is worth running before dismissing it.

**3. REPO-WIDE REUSE SURVEY.** This model was built standalone; four things elsewhere are worth knowing.

- **`models/common/modules/rmsnorm/rmsnorm_1d.py::_create_sharded_norm_program_config`** — builds the
  config from the ROW COUNT (`block_h = tile_padded_batch_rows // tile_size`, `tile_padded_batch_rows =
  32*ceil(batch/32)`, plus a `subblock_w` search). **This is precisely what our hardcoded `_NORM_SHARD`
  `(32, 96)` / `block_h=1` cannot do, and it is the stated blocker on §6.35's measured 3.4x batching
  win.** `models/common/modules/mlp/mlp_1d.py` applies the same pattern to shard shapes and matmul
  configs. Start here if the batching lead is picked up.
- **`models/demos/gemma4/tt/spec_decode.py`** — verify step "runs ONE batched forward over
  `[anchor, d1, ..., dK]` … candidates in the batch dim", which is exactly the mechanism for our 31
  unused tile rows. Its docstring also reasons through worse-vs-different: a batched forward differs by
  ~1e-5, "flips only target near-ties (top-2 logit gap < ~1)", giving "an equally-valid greedy trajectory
  thereafter". **Do not borrow that conclusion** — their correctness is guaranteed by construction
  (committed tokens always come from the target verify); a precision change has no such guarantee.
- **`models/common/tests/modules/sampling/test_sampling_1d.py`** — classifies index disagreements as
  TIE-BREAK (acceptable) vs TRUE-MISMATCH (kernel bug); notes `ttnn.argmax` returns uint32 vs torch's
  int64. Bears on [flow-08a]: we moved the semantic argmax to the host, so ties would now be broken by
  `torch.argmax`. fp32 logits plus an additive mask make exact ties effectively impossible — that is
  *why* it is safe, and it was not checked when the change shipped.
- **`models/tt_dit/` and `models/demos/z_image_turbo`** — flow-matching / DiT pipelines, the nearest
  in-tree neighbours to Block 2 if its structure is revisited (e.g. 7→5 Euler steps). Not examined.

Checked and NOT reusable: `deepseek_v3_b1/fused_ops/lm_head_sampling/` is a bespoke multi-device kernel
on CCL/MoE infrastructure and our semantic matmul is already at 182 of ~194 GB/s; `ttnn.sampling` /
`ttnn.topk` exist but the host argmax is simpler and 1.439x faster; `speecht5_tts` and
`demos/audio/whisper` are different architectures.

**4. THE MEASUREMENT METHODOLOGY, CHECKED AGAINST THE REPO'S OWN — and it holds.**
`models/common/tests/modules/attention/profiling/` states outright that it uses trace capture "for
accurate device-side timing without host dispatch overhead". Every probe in this file times
`perf_counter` around `synchronize_device`, which INCLUDES host dispatch — so the whole session's numbers
were open to the objection that they measured dispatch, not device work. Tested directly on `_solve`
(a pure device graph by design, [flow-18], so it captures cleanly), 200 reps x 3 rounds:

    eager  (host dispatch included)   19.145 ms   spread 0.002
    traced (device work only)         19.230 ms   spread 0.002

**Host dispatch is 0% of what was measured** — the device is saturated and dispatch is entirely hidden.
Traced is even marginally SLOWER (+0.085 ms), independently reproducing §6.26's trace verdict on a
different unit. So "an op costs ~20 us to exist" is genuine DEVICE time, every probe in this file is
sound as written, and §6.33's unexplained whole-block gain is NOT a dispatch artifact. Note this is
trace-as-MEASUREMENT; §6.26's rejection of trace-as-SHIPPING is untouched.

The full tracy op profiler (`tools/tracy/process_ops_logs.py`, `python -m tracy -r -m ...`) would give
per-op device times directly, and is the right tool if the §6.33 mechanism is ever chased. It does not
run here as-is: `websockets` is absent from the venv, `generated/profiler/` is empty (never run on this
box) and `build_Release/profiler/` looks incomplete, so it likely needs a profiler-enabled rebuild.

**5. TWO CHOICES THE SHARED LIBRARY MAKES DIFFERENTLY — both already settled here.**
- `models/common/modules/mlp/mlp_1d.py` defaults to **HiFi2 with fp16 accumulation**; we use HiFi4 +
  `fp32_dest_acc_en=True` everywhere. Already tested and rejected: [flow-03] measured HiFi2/no-fp32acc
  at PCC 0.9998382 and 35/222 code errors against HiFi4's 0.9999845 and 4/222 — and **slower**, 48.72 vs
  42.57 ms. In Block 1, HiFi2/LoFi saves ~4 ms for 10-20x the integer-code errors. Not a missed knob.
- their `_matmul_config` derives `in0_block_w = _find_largest_divisor(k // (tile*grid_y))` as a heuristic.
  We swept it explicitly instead ([gpt-20]: 1/2/4/8/16/32 -> 152.0/83.2/68.1/73.7/80.5/89.4 us) and
  shipped the bit-exact 2 over the faster-but-inexact 4. A heuristic would not have surfaced that
  exactness cliff, so the hand-sweep stands.

**Code audit, same pass:** 0 broken `NOTES.md [id]` pointers, 0 orphan NOTES entries, 0 broken `§6.x`
references, no comment contradicting a current measurement, `test_tt_defaults` (7 guards on the shipped
constants) green, 122 tests green. Five module-level names in `reference/` are unreferenced
(`text_logits`, `TIED_EMBEDDINGS`, `FM_HIDDEN_DIM`, `CODEC_HIDDEN_DIM`, `DEC_WINDOWS`) and are
**deliberately kept** — they document checkpoint architecture, and deleting model facts from the fp32
ground truth to satisfy a linter is a bad trade.

### 6.39 — in-place ops swept EVERYWHERE in the decode path of both blocks: a dead end, four ways

§6.37 tested three sites in Block 2 and got +0.001 ms. That left two whole categories untested, which
this closes: **Block 1 entirely**, and **`inplace=True` on the norm program config** — a different
mechanism from op substitution, a kernel flag that lets the norm write into its own already-temporary
sharded input rather than allocating an output. Both blocks had it hardcoded `False` and neither had ever
tried `True`. It runs 52x per Block 1 step and 49x per Block 2 frame.

**Block 1 decode step, 100 steps x 6 rounds, interleaved both directions:**

| variant | ms/step | spread | vs shipped | max abs diff |
|---|---|---|---|---|
| A shipped | 23.152 | 0.009 | -- | -- |
| **B norm `inplace=True`** | **23.152** | 0.006 | **1.0000x** | **0.00e+00** |
| C ops in-place (`_mlp` multiply + add, `_layer_step` residual) | 23.215 | 0.005 | 0.9973x | 2.67e+03 |
| D both | 23.213 | 0.009 | 0.9973x | 4.13e+03 |

**Block 2 frame, 200 frames x 6 rounds:**

| variant | ms/frame | spread | delta | codes |
|---|---|---|---|---|
| A shipped | 21.050 | 0.090 | -- | 0 of 37 |
| B norm `inplace=True` | 21.013 | 0.067 | +0.037 | 0 of 37 |
| C `_solve` tail in-place (CFG combine + Euler) | 21.027 | 0.044 | +0.023 | 0 of 37 |
| D both | 20.975 | 0.038 | +0.074 | 0 of 37 |

**Verdicts.**

- **`inplace=True` on the norm is INERT.** Block 1 measures it to three decimal places identical with a
  0.006-0.009 ms spread, and the output is BIT-IDENTICAL. Block 2's +0.037 is inside its 0.090 spread.
  The flag does nothing on this path in either direction — a permanent answer, not a maybe.
- **Block 1's in-place ops are 0.063 ms SLOWER**, which is 7x the 0.009 spread, so that one IS real.
- **Block 2's remaining sites are within noise** — the best gap (0.074) is under the worst spread (0.090).

**So across four independent measurements — §6.37's `_block` sites (+0.001), Block 1's three sites
(-0.063), Block 2's `_solve` tail (within noise), and the norm flag (exactly inert) — in-place is a dead
end everywhere in the decode path.** It is the kind of change that looks obviously free and is not: the
op count is unchanged, and at these sizes the cost is the launch, not the output allocation.

**AND IT CARRIES AN ALIASING HAZARD WORTH RECORDING SEPARATELY.** Block 1's C/D rows show max abs diff
2.67e+03 on activations whose scale is ~0.02. That is **the benchmark harness, not the ops**: in-place
makes `_layer_step` mutate its argument, and the timing loop restarts from the same `x` tensor 100 times,
so the input drifts. It is not a correctness verdict on the ops — but it is a live demonstration that a
function which mutates its input is a landmine even where today's call sites happen to be safe. Two
reasons to decline, not one.

**Not swept, deliberately:** `_attend`'s `add(multiply(s, SCALE), mask)` and `_layer`'s residual are
PREFILL-only paths (~3% of wall time), and the codec's elementwise ops amortise to ~0.03 ms/frame. Both
are below the noise floor of anything measurable here.

### 6.40 — `gate_codes` was measuring accuracy on synthetic input, and its number is unusable

Raised by the P150 fork, reproduced here independently, and it is a defect in OUR gate too. I quoted
this gate's output as an accuracy figure repeatedly in §6.31/§6.37 without noting what it was fed.

`gate_codes` feeds `torch.randn(1,128,3072)*0.02`. Its own comment says it "checks the WIRING … without
needing the tokenizer or a voice preset" — an honest description that later got read as an accuracy
number. Same comparison, same commit, only the input changed:

| input | frames | acoustic | semantic | \|delta\| histogram | off-by-one |
|---|---|---|---|---|---|
| real fixture case 0 | 8 | 7/288 | 0 of 8 | `{1: 7}` | **100%** |
| real fixture case 2 | 8 | 8/288 | 0 of 8 | `{1: 8}` | **100%** |
| real fixture case 3 | 8 | 9/288 | 0 of 8 | `{1: 9}` | **100%** |
| **REAL pooled** | **24** | **24/864 (2.8%)** | **0 of 24** | | **100%** |
| synthetic `randn*0.02` | 8 | **97/288 (33.7%)** | 1 of 8 | `{1:75, 2:13, 3:4, 4:3, 6:1, 7:1}` | 77% |

**On real prompts every differing code is off by exactly one FSQ level of 21** — the smallest
representable disagreement, i.e. a bin-boundary flip. Synthetic input produces deltas up to **7**, which
is the part that is not a rounding story.

**AND IT CONFIRMS A DENOMINATOR COLLISION IN THIS FILE.** Case 0 measures exactly **7/288** and case 3
exactly **9/288** — the numbers scattered through [gpt-*]/[flow-*] and §6.8/§6.10. Those were always
REAL-prompt figures (8 frames x 36 codes on one fixture). `gate_codes`' 97/288 is 8 SYNTHETIC frames x 36.
**Identical denominator, incomparable measurement.** Anyone comparing them is comparing nothing.

**WHY, established by the fork's controls (which are the right ones and I would have asked for them):**

    reference fp32 vs its own fp64      0/288 on BOTH inputs   -- the reference flips nothing
    FSQ boundary margin, median         0.260 synthetic vs 0.253 real -- INDISTINGUISHABLE
    Block 2 device error, h held fixed  0.0334 synthetic vs 0.0389 real -- real is slightly WORSE
    Block 1 prefill PCC(h_dev, h_ref)   0.986540 synthetic vs 0.999894 real -- 22x, at step 0

So it is not FSQ geometry and not Block 2. **It is Block 1's prefill collapsing off-manifold** —
trap #12's PCC 0.892-vs-0.9994 arriving at the codes gate through Block 2. Mechanically: random
Gaussian embeddings sit far outside the distribution the weights were trained on, so BFP8's
shared-exponent blocks lose far more precision on the activations that result.

**THE STRONGEST RESULT is that the synthetic metric is NON-MONOTONIC IN PRECISION.** On the fork,
FF→bf16 — unambiguously better weights — made synthetic PCC *worse* (0.986540 → 0.976994). A metric
that degrades when you improve precision **cannot rank configurations**. That is stronger than
"pessimistic proxy": it is uninformative, and it must never be used to choose between arms.

**THE MOST ACTIONABLE RESULT, from the same fork sweep:** real-prompt prefill error is pinned at
**0.70% across the entire precision ladder** — shipped, +bf16 FF, and all-bf16 all read 0.70%, while
PCC moves only 0.999894 → 0.999934 for **+4.91 ms/step (~12% slower)**. That confirms §6.16's knee from
the other side and says the residual 0.70% **is not weight quantization** — there is no weight lever
left to pull, on either card.

**THE ASYMMETRY THAT DECIDES WHETHER THE GATE IS SALVAGEABLE**, and neither session stated it:

| use | synthetic input |
|---|---|
| **paired** A-vs-B ("did my change move anything?") | **VALID** — identical is identical on any input |
| **absolute** accuracy level ("we are at 33.7%") | **MEANINGLESS** |
| **ranking** configs by accuracy | **WORSE THAN MEANINGLESS** — non-monotonic |

My own §6.31/§6.37 usage was the paired one ("1, 97/288 → identical"), which is legitimate. What was
wrong was printing the absolute figure with no label, so a reader takes 33.7% as an accuracy level.

**FIXED IN `tt_gates.py`:** `--gate codes` now runs the three real fixtures FIRST under the heading
"THIS IS THE ACCURACY NUMBER", prints a `|delta|` histogram and off-by-one fraction after every run
with a note that a code is 1 of 21 FSQ levels, and labels the synthetic block "WIRING ONLY — do NOT
read as accuracy".

**And the P150 comparison, which is the question that started this:** N150 real **24/864 (2.8%)**,
P150 real **34/864 (3.9%)**, both **0 semantic**, both all-off-by-one. The fork's 85-88/288 is the same
synthetic artifact this card shows at 97/288, and their bisect showed the fork point (unmodified N150
code) already reading 85/288. **No P150 regression.**

### 6.41 — `activation="silu"` was NEVER fused, on either chip; the fix is P150-only

From `lserbedzija/voxtral-tts-ttnn_p150` (`1c91d3de4c0`), verified independently here. It corrects a
claim this project has carried since Block 2's bring-up.

**THE DIAGNOSIS IS RIGHT AND PORTABLE.** Same shape, same dtype, N150:

    linear(activation="silu")      157.9 us   PCC vs fp64 0.9999634
    linear() then ttnn.silu()      157.9 us   PCC vs fp64 0.9999634   <- IDENTICAL
    linear() plain, no silu        147.4 us

`activation="silu"` costs **exactly** what a separate op costs: **+10.3 us**. ttnn issues a second
kernel either way. Over 47 w1 calls a frame (26 in Block 1 + 21 in Block 2) that is **0.485 ms/frame**
of pure overhead we have been describing as fused.

**AND THE EVIDENCE WAS IN OUR OWN MAPS THE WHOLE TIME.** §6.27 and §6.36 both list w1 at 157.9 and w3 at
147.5 — **same shape, same dtype, same 155.1 us floor**, the only difference being the activation. I read
them as "+2.9 over floor" and "-7.5 under floor" and never compared the two rows to each other. That is
the second time this session a finding sat in a table I had already printed.

**[flow-12] IS CORRECTED.** It claimed "SiLU rides along on the w1 matmul instead of being its own op …
0.16 ms/frame". False, and false for the life of the port. [gpt-14]'s "one dispatch cheaper" is also
corrected — never measured.

**THE FIX DOES NOT TRANSFER.** Only a program config's `fused_activation` genuinely folds silu in, and
`per_core_N` must divide the 288 output tiles, which on 64 cores admits no good shape:

| grid | cores | per_core_N | best us | vs `activation="silu"` |
|---|---|---|---|---|
| **8x6** | 48 | 6 | **162.1** | **0.975x** |
| 8x4 | 32 | 9 | 176.8 | 0.894x |
| 8x3 | 24 | 12 | 184.7 | 0.856x |
| 8x2 | 16 | 18 | 207.6 | 0.761x |
| 4x8 | 32 | 9 | 261.0 | 0.605x |

Nothing wins. **And it loses on the whole block too** — 6 rounds, interleaved:

    BLOCK 1 step    A activation=silu 23.150 ms (spread 0.006)  B config 23.207  -0.057 ms
    BLOCK 2 frame   A activation=silu 20.825 ms (spread 0.065)  B config 20.903  -0.078 ms

Both gaps exceed their spreads, so A genuinely wins. **The 0.485 ms/frame is real and unreachable on
N150 by this route.** On P150 it IS reachable: a 12x6 grid of 130 cores gives `per_core_N=4` and folds
silu at 88.1 vs 98.8 us, worth 2.42 ms/frame there.

**THEIR OTHER FINDING IS A MECHANISM FOR §6.33, AND IT IS P150-SPECIFIC.** They measured the silu op at
**12.2 us isolated and ~54 us in-block** and explain it as *"a tight loop of identical ops pipelines, a
real block of differing ops does not"* — which is exactly why isolated timing mispredicts, and it
predicts isolated UNDERSTATES an op's in-block cost. **That amplification does not reproduce on N150**:
our isolated 4 us gap stayed a 0.057-0.078 ms whole-block gap, roughly consistent rather than 4.4x
amplified. Coherent with their own measurement that small ops cost 3.4x more on P150. So §6.33's
mechanism is *plausibly* pipelining, but on this chip the effect is far weaker than their numbers imply
and it still does not explain a whole-block gain from a slower op.

**Three more of their P150 results reverse our N150 conclusions**, all consistent with the same
"small ops cost 3.4x more there" regime — in-place elementwise pays (+0.929 ms/step, +0.790 ms/frame,
against our §6.39 null), sdpa for Block 2's interior pays (45.4 ms/frame, against our §6.37 rejection),
and dispatch is 3-4% there against the 0% §6.38 measured here. None of those is a contradiction; they
are a different cost regime, and that is the single most useful thing to carry into any P150 port.

### 6.42 — read the P150 fork at RTF 0.364 in full and tried everything: one correction, nothing ships

`lserbedzija/voxtral-tts-ttnn_p150` reached **RTF 0.3647 / 27.7 ms/frame** against our 0.57-0.64 / 48 ms.
Read all 35 commits including the reasoning, and tested every candidate that could apply here.

**WHAT TRANSFERRED: one correction to my own record.**

Their §6.65 measured 10.1 host crossings per frame costing 2.796 ms, and showed blocks timed in a TIGHT
LOOP are faster than the same blocks inside the real generation — because a tight loop never syncs.
**That is exactly the mistake §6.38 made.** Measured here:

| | ours (N150) | theirs (P150) |
|---|---|---|
| crossings/frame | **10.1** (3.0 D2H, 7.1 H2D) | 10.1 (3.0, 7.1) — exact match |
| sum of tight blocks | 44.194 ms | 35.515 |
| real generate loop | **45.505 ms** | 39.289 |
| **drain penalty** | **+1.311 ms/frame** | +3.77 ms |

So §6.38's "host dispatch is 0%" is **right about `_solve` in a tight loop and wrong as a description of
the frame**. The real loop pays 1.311 ms of serialization — about a third of their penalty, consistent
with N150's lower small-op cost. (The 15.8 ms that instrumenting attributes to `to_torch` is mostly the
device work being WAITED FOR at the sync, not extra overhead; the 1.311 ms gap is the honest number.)

**WHAT DID NOT TRANSFER, and why — all four have the same root cause.**

| their change | theirs | ours | why not |
|---|---|---|---|
| w1 program config to fuse silu (§6.41) | -2.42 ms/frame | **loses** at every legal grid | 288 tiles / 64 cores admits no good `per_core_N`; their 12x6 of 130 gives 4 |
| residual as matmul bias | **-1.918 ms/step** | **+0.086 ms/step**, real loop INCONCLUSIVE | their gain needed program configs making the matmul 2.3x faster to EXPOSE the add. Ours is at the roofline, so the add still hides in its shadow — which is what our old "w2's add is already free" note said |
| trace the frame loop | **-4.244 ms/frame** | §6.38 measured `_solve` traced at **19.230 vs 19.145 eager** | their Block 2 alone was 13.3% dispatch after their matmul speedups; our device is saturated |
| width-shard the decode norm | -5.399 ms/frame | **already ours** | their §6.39/6.40 had REMOVED it and this reinstated it; we never removed it |

**The single root cause: small ops cost ~3.4x more on their card (their own measurement), and their
matmuls were not at the roofline.** Every fix on that branch works by exposing or removing dispatch that
had been hiding behind slow device work. We have no such hiding place — all ten weight matmuls are at
194-202 GB/s and the tight-loop dispatch is 0%. **That is why a 1.6x-faster chip with 2x the cores needed
an entirely different optimization set, and why almost none of it is portable.**

**AND ONE OF OUR CHANGES WAS CHECKED AND SURVIVES.** Their §6.65 flagged their §6.50 as "probably stale
… pricing the OP in isolation while ignoring the D2H drain", which is exactly what [flow-08a] did when it
moved semantic_code's argmax to the host on an isolated 1.439x. Re-tested in the REAL loop, warmed and
interleaved, both paths returning token 3040:

    A host argmax (ships)   45.396 ms/frame  spread 0.087
    B device argmax         45.730 ms/frame  spread 0.173     host wins by 0.335 ms

The gap exceeds the spread, so **[flow-08a] is not stale on N150.** The D2H of 8320 logits is cheaper
than the 490 us device reduce even after the drain is priced in.

**THE METHODOLOGY POINT WORTH KEEPING, which is the most valuable thing on that branch.** They found
FIVE "expired premises" — rejections that were correct when written and silently stopped being correct
after a later change made something faster. Their words: *"a correct measurement whose PREMISE EXPIRED.
Not a bad harness."* Their rule: **any rejection whose margin was small against a then-slow baseline
deserves re-testing.** Our exposure is lower because our rejections were mostly made against a
roofline-saturated baseline that has not moved — but §6.41 (silu) is precisely one of these, caught by
them and not by us.

**Also worth recording from their branch, unverified here:** an eager op map ranks ops by LAUNCH cost,
not by what they cost in a traced path — their `concat` went 144.7 us eager to **2.6 us traced**, a
"ghost". §6.36 lists our `concat` (line 174) and `reshape` (line 176) at 101.0 and 106.0 us, 1.449
ms/frame, which I called the largest untouched item. If tracing ever lands here, that item may
substantially evaporate rather than reward eager work.

**STILL UNTESTED, and it is a judgement call rather than an oversight:** a full per-frame trace. Prize is
bounded by the 1.311 ms drain (2.9% of frame). Against it: three prior tracing measurements here were
negative, §6.38 measured `_solve` traced as slightly WORSE, and their own commit documents a K/V
corruption bug that reached a full gate run before being caught. Not attempted.

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

**All three blocks work, quality is good, and the port is no longer the bottleneck — the model is.**

Per frame, steady state on one N150: **Block 1 ~35 ms, Block 2 ~42.5 ms, host 0.2 ms, ~77 ms
total, RTF ~1.0.** Both hot modules open with a "where the time goes" map giving the ceiling for
each line item; read those before optimizing anything.

1. Re-read this file. Recreate the venvs (§2). Run the tests, then
   `generate_quality_set.py --cases 0,1` to confirm the device path still speaks.
2. **BLOCK 3 IS THE LEAST EXPLORED THING LEFT — ~9% of wall time and never swept.** Both hot
   blocks have had two optimization passes; the codec has had none since its own bring-up.
   `ign/voxtral_opt` has a `keep_sharded_splits` commit claiming ~18 ms → ~6 ms on their conv
   stack by removing ~104 layout conversions, which is exactly the class of thing that pays here.
3. **Both hot blocks are near their floors, for different reasons.** Block 1 (~31.4 ms): every
   weight matmul is at the 194 GB/s ceiling and the only bf16 weight left is w2, the pinned hang
   trigger — worth ~3.6 ms if it can ever be moved. Block 2 (~28 ms): sits ~2x above its 13.4 ms
   weight-read floor on device-side per-kernel cost, and everything op-level has been tried. The
   two ideas left there are structural — fewer Euler steps, or concurrent requests.
4. **Gate on the DETERMINISTIC metrics, not on WER.** This is the most expensive lesson in the
   file and it was learned twice. End-to-end WER cannot resolve a numerical change: the same code
   at seeds 0/1/2 spans 0.88–2.06% (§6.7). Worst-sample read as a MAX is also unreliable — it
   moved 1.28–4.28% non-monotonically across configs (§6.8). What works: teacher-forced MEAN and
   P90 worst-sample against the fp32 reference (`tests/tt_gates.py`), long-form WER for gross
   breakage, and integer-code counts over several draws for Block 2.
5. `tests/test_tt_defaults.py` pins the shipped configuration with the reason for each choice. If
   you change a default, change that test in the same commit.

**Deferred and still worth doing:**
- A **listening pass** — WER cannot see timbre, and nobody has A/B'd the current build.
- **Report the ttnn hang upstream** (§6, Block 1). A silent hang needing a board reset is their
  bug regardless of what we feed it, and we have a ~90 s repro.
- **Prefill beyond ~1024 tokens** needs chunked prefill.
- A **like-for-like comparison against `ign/voxtral_p150_qb2` on THEIR hardware** — see below for
  what we could and could not settle.
